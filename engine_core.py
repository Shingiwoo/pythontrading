import os, json, logging
from typing import Dict, Any, Tuple, Optional, Union, Any

import numpy as np
import pandas as pd
from ta.trend import EMAIndicator, SMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochRSIIndicator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice
from mtf_cache import update_cache, get_series as mtf_get_close
from regime import RegimeThresholds, get_regime, compute_adx as _regime_adx

# --- helpers (top-level util) ---
Numeric = Union[float, int, np.floating, np.integer]
ArrayLike = Union[pd.Series, np.ndarray, list, tuple]
ScalarOrArray = Union[Numeric, ArrayLike]

def as_scalar(x: Any) -> float:
    """Ambil angka scalar dari berbagai tipe.
    - pd.Series -> nilai terakhir
    - np.ndarray/list/tuple -> elemen terakhir
    - float/int -> cast float
    """
    if isinstance(x, pd.Series):
        if len(x) == 0:
            return 0.0
        return float(x.iloc[-1])
    if isinstance(x, (np.ndarray, list, tuple)):
        if len(x) == 0:
            return 0.0
        return float(x[-1])
    try:
        return float(x)
    except Exception:
        return 0.0

SAFE_EPS = float(os.getenv("SAFE_EPS", "1e-9"))


def to_scalar(x: Optional[Any], *, how: str = "last", default: float = 0.0) -> float:
    return as_scalar(x) if x is not None else default


def to_bool(x: Union[bool, ArrayLike]) -> bool:
    if isinstance(x, (pd.Series, np.ndarray)):
        if len(x) == 0:
            return False
        return bool(np.asarray(x).reshape(-1)[-1])
    return bool(x)


def safe_div(a: Any, b: Any, default: float = 0.0) -> float:
    aa, bb = as_scalar(a), as_scalar(b)
    if bb == 0.0 or np.isnan(bb):
        return default
    return aa / bb


def norm_freq(freq: str, default: str = "15min") -> str:
    f = (freq or default).strip().lower()
    if f.endswith("m") and not f.endswith("min"):
        if not f.endswith(("ms", "us")):
            f = f[:-1] + "min"
    return f


def rolling_twap(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=max(2, window // 2)).mean()


def floor_to_step(x: Any, step: Any) -> float:
    xf, sf = as_scalar(x), max(as_scalar(step), 1e-18)
    return float(np.floor(xf / sf) * sf)


def ceil_to_step(x: Any, step: Any) -> float:
    xf, sf = as_scalar(x), max(as_scalar(step), 1e-18)
    return float(np.ceil(xf / sf) * sf)


def clamp_scalar(x: ScalarOrArray, lo: ScalarOrArray, hi: ScalarOrArray) -> float:
    xv = to_scalar(x)
    lov = to_scalar(lo)
    hiv = to_scalar(hi)
    return float(min(max(xv, lov), hiv))


def as_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x) if x is not None else default
    except (TypeError, ValueError):
        return default

# -----------------------------------------------------
# Konversi tipe sederhana
# -----------------------------------------------------
def _to_float(v: Any, d: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(d)

def _to_int(v: Any, d: int) -> int:
    try:
        return int(v)
    except Exception:
        return int(d)

def _to_bool(v: Any, d: bool) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(int(v))
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1","true","y","yes","on"}: return True
        if s in {"0","false","n","no","off"}: return False
    return bool(d)

# -----------------------------------------------------
# Precision helpers
# -----------------------------------------------------
def round_to_step(x: float, step: float) -> float:
    if step <= 0:
        return float(x)
    return float(round(x / step) * step)


def round_to_tick(x: Any, tick: Any) -> float:
    xf, tf = as_scalar(x), max(as_scalar(tick), 1e-18)
    return float(np.round(xf / tf) * tf)

def enforce_precision(sym_cfg: Dict[str, Any], price: float, qty: float) -> Tuple[float, float]:
    p_step = _to_float(sym_cfg.get("tickSize", 0), 0)
    q_step = _to_float(sym_cfg.get("stepSize", 0), 0)
    price = round_to_step(price, p_step) if p_step > 0 else price
    qty = floor_to_step(qty, q_step) if q_step > 0 else qty
    return price, qty

def meets_min_notional(sym_cfg: Dict[str, Any], price: float, qty: float) -> bool:
    min_not = _to_float(sym_cfg.get("minNotional", 0), 0)
    return (price * qty) >= min_not if min_not > 0 else True


def _coerce_htf_cfg(coin_cfg: dict) -> dict:
    """
    Terima berbagai format HTF di coin_config:
    - dict: {"enabled": bool, "timeframe": "1h"/"4h", "ema_period": int, "sma_period": int, ...}
    - string: "1h" / "4h" → diasumsikan enabled=True dengan default period
    - legacy: gunakan "use_htf_filter" (0/1) dan "htf_tf" (jika ada)
    """
    try:
        v = coin_cfg.get("htf", None)
        d = {
            "enabled": bool(coin_cfg.get("use_htf_filter", False)),
            "timeframe": str(coin_cfg.get("htf_tf", "1h")).strip().lower(),
            "ema_period": 22,
            "sma_period": 22,
            "rule": "long_ema>=sma;short_ema<=sma",
            "fallback_pass": True,
        }
        if isinstance(v, str):
            d["enabled"] = True
            d["timeframe"] = v.strip().lower()
            return d
        if isinstance(v, dict):
            d.update(v)
            d["timeframe"] = str(d.get("timeframe", "1h")).strip().lower()
            d["ema_period"] = int(d.get("ema_period", 22))
            d["sma_period"] = int(d.get("sma_period", 22))
            if "fallback_pass" not in d:
                d["fallback_pass"] = True
            return d
        return d
    except Exception:
        return {
            "enabled": False,
            "timeframe": "1h",
            "ema_period": 22,
            "sma_period": 22,
            "rule": "long_ema>=sma;short_ema<=sma",
            "fallback_pass": True,
        }

def htf_timeframe(coin_cfg: dict) -> str | None:
    """Ambil timeframe HTF dari konfigurasi."""
    try:
        return _coerce_htf_cfg(coin_cfg).get("timeframe")
    except Exception:
        return None

# -----------------------------------------------------
# Config loader & merger
# -----------------------------------------------------
def load_coin_config(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def merge_config(symbol: str, base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge SYMBOL_DEFAULTS -> override oleh config per-simbol.
    Termasuk nested 'filters'.
    """
    merged: Dict[str, Any] = {}
    if isinstance(base_cfg, dict):
        # base defaults
        merged.update(base_cfg.get("SYMBOL_DEFAULTS", {}) or {})
        # per-symbol overrides
        sym_cfg = base_cfg.get(symbol, {}) if isinstance(symbol, str) else {}
        # gabung nested filters
        f = dict(merged.get("filters", {}) or {})
        f.update(sym_cfg.get("filters", {}) or {})
        merged.update({k: v for k, v in sym_cfg.items() if k != "filters"})
        if f:
            merged["filters"] = f
    merged["symbol"] = symbol
    return merged

# -----------------------------------------------------
# Indikator & sinyal sederhana
# -----------------------------------------------------
def compute_indicators(
    df: pd.DataFrame,
    heikin: bool = False,
    *,
    vwap_enabled: bool = False,
    stoch_enabled: bool = False,
) -> pd.DataFrame:
    d = df.copy()

    # (opsional) Heikin Ashi
    if heikin:
        ha = d.copy()
        ha['ha_close'] = (
            pd.to_numeric(ha['open'], errors='coerce')
            + pd.to_numeric(ha['high'], errors='coerce')
            + pd.to_numeric(ha['low'], errors='coerce')
            + pd.to_numeric(ha['close'], errors='coerce')
        ) / 4
        ha['ha_open'] = pd.to_numeric(ha['open'], errors='coerce').shift(1)
        ha.loc[ha.index[0], 'ha_open'] = (
            pd.to_numeric(ha.loc[ha.index[0], 'open'], errors='coerce')
            + pd.to_numeric(ha.loc[ha.index[0], 'close'], errors='coerce')
        ) / 2
        ha['ha_high'] = ha[['high', 'ha_open', 'ha_close']].max(axis=1)
        ha['ha_low'] = ha[['low', 'ha_open', 'ha_close']].min(axis=1)
        d[['open', 'high', 'low', 'close']] = ha[['ha_open', 'ha_high', 'ha_low', 'ha_close']]

    # EMA/MA, MACD, RSI
    d['ema_22'] = EMAIndicator(d['close'], 22).ema_indicator()
    d['ma_22'] = SMAIndicator(d['close'], 22).sma_indicator()
    macd = MACD(d['close'])
    d['macd'] = macd.macd()
    d['macd_signal'] = macd.macd_signal()
    d['rsi'] = RSIIndicator(d['close'], 25).rsi()

    # ATR (EMA Wilder-ish) & normalisasi
    prev_close = d['close'].shift(1)
    tr = pd.concat(
        [
            (d['high'] - d['low']).abs(),
            (d['high'] - prev_close).abs(),
            (d['low'] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    d['atr'] = tr.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().fillna(0.0)
    d['atr_pct'] = (d['atr'] / d['close']).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # Bollinger Band width (persentase)
    bb = BollingerBands(d['close'])
    width = bb.bollinger_hband() - bb.bollinger_lband()
    mid = bb.bollinger_mavg().replace(0, np.nan)
    d['bb_width_pct'] = (width / mid).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # ADX sederhana
    try:
        adx = ADXIndicator(high=d['high'], low=d['low'], close=d['close'])
        d['adx'] = adx.adx().fillna(0.0)
    except Exception:
        d['adx'] = 0.0

    # body/ATR + alias
    d['body'] = (d['close'] - d['open']).abs()
    d['body_to_atr'] = (d['body'] / d['atr']).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    d['body_atr'] = d['body_to_atr']  # alias untuk kompatibilitas

    # VWAP opsional
    if vwap_enabled and 'volume' in d.columns:
        vwap = VolumeWeightedAveragePrice(
            high=d['high'], low=d['low'], close=d['close'], volume=d['volume']
        )
        d['vwap'] = vwap.volume_weighted_average_price()

    # StochRSI opsional
    if stoch_enabled:
        stoch = StochRSIIndicator(d['close'])
        d['stoch_rsi'] = stoch.stochrsi()

    return d


USE_BACKTEST_ENTRY_LOGIC = bool(int(os.getenv("USE_BACKTEST_ENTRY_LOGIC", "1")))


def _rsi_pullback(rsi: float, side: str) -> bool:
    if side == 'LONG':
        return 10 <= rsi <= 45
    else:
        return 70 <= rsi <= 90


def _rsi_midrange(rsi: float, side: str) -> bool:
    if side == 'LONG':
        return 40 <= rsi <= 70
    else:
        return 30 <= rsi <= 60


def compute_base_signals(df: pd.DataFrame, coin_cfg: Dict[str, Any] | None = None) -> tuple[bool, bool]:
    """
    Sinyal dasar:
      - Arah tren: EMA22 vs MA22
      - Konfirmasi RSI (parametris via coin_cfg): rsi_long_max / rsi_short_min
      - (Opsional) Konfirmasi MACD (use_macd_confirm)
    """
    coin_cfg = coin_cfg or {}
    last = df.iloc[-1]
    ema = float(last.get('ema_22', 0))
    ma  = float(last.get('ma_22', 0))
    rsi_now = float(last.get('rsi', 50))
    trend_up = ema > ma
    trend_dn = ema < ma
    # Threshold dari config (default seperti backtest lama)
    rsi_long_max  = float(coin_cfg.get('rsi_long_max', 45.0))
    rsi_short_min = float(coin_cfg.get('rsi_short_min', 70.0))
    use_macd = bool(coin_cfg.get('use_macd_confirm', False))
    macd_now = float(last.get('macd', 0.0))
    macd_sig = float(last.get('macd_signal', 0.0))
    mode = str(coin_cfg.get('rsi_mode', 'PULLBACK')).upper()
    # Mode RSI
    if mode == 'MIDRANGE':
        long_ok = trend_up and _rsi_midrange(rsi_now, 'LONG')
        short_ok = trend_dn and _rsi_midrange(rsi_now, 'SHORT')
    else:
        long_ok = trend_up and (rsi_now <= rsi_long_max)
        short_ok = trend_dn and (rsi_now >= rsi_short_min)
    # Opsional MACD
    if use_macd:
        long_ok  = bool(long_ok  and (macd_now > macd_sig))
        short_ok = bool(short_ok and (macd_now < macd_sig))
    logging.getLogger(__name__).info(
        f"BASE ema22={ema:.6f} ma22={ma:.6f} rsi={rsi_now:.2f} mode={mode} macd_ok={use_macd} -> L={long_ok} S={short_ok}"
    )
    return bool(long_ok), bool(short_ok)


def compute_base_signals_backtest(df: pd.DataFrame, coin_cfg: Dict[str, Any] | None = None) -> tuple[bool, bool]:
    return compute_base_signals(df, coin_cfg)


def compute_base_signals_live(df: pd.DataFrame, coin_cfg: Dict[str, Any] | None = None) -> tuple[bool, bool]:
    return compute_base_signals(df, coin_cfg)


ML_WEIGHT = float(os.getenv("ML_WEIGHT", "1.5"))

# ----- Auto-relax filter parameters -----
FILTER_TARGET_PASS_RATIO = float(os.getenv("FILTER_TARGET_PASS_RATIO", "0"))
FILTER_HYSTERESIS_TIGHTEN = float(os.getenv("FILTER_HYSTERESIS_TIGHTEN", "1"))
FILTER_RELAX_STEP_ATR = float(os.getenv("FILTER_RELAX_STEP_ATR", "0"))
FILTER_RELAX_STEP_BODY = float(os.getenv("FILTER_RELAX_STEP_BODY", "0"))
FILTER_MAX_RELAX_ATR = float(os.getenv("FILTER_MAX_RELAX_ATR", "0"))
FILTER_MAX_RELAX_BODY = float(os.getenv("FILTER_MAX_RELAX_BODY", "0"))

# Auto-relax dinonaktifkan untuk tuning deterministik


def get_coin_ml_params(symbol: str, coin_config: dict) -> dict:
    d = coin_config.get(symbol, {}).get("ml", {})
    return {
        "enabled": bool(d.get("enabled", True)),
        "strict": bool(d.get("strict", False)),
        "up_prob": float(d.get("up_prob_long", 0.55)),
        "down_prob": float(d.get("down_prob_short", 0.45)),
        "score_threshold": float(d.get("score_threshold", 2.0)),
        "weight": float(d.get("weight", ML_WEIGHT)),
    }


def make_decision(
    df: pd.DataFrame,
    symbol: str,
    coin_cfg: dict,
    ml_up_prob: float | None,
    atr_ok: bool | None = None,
    body_ok: bool | None = None,
    meta: Dict[str, Any] | None = None,
    *,
    cooldown_active: bool = False,
    position: Any | None = None,
    ctx: Dict[str, Any] | None = None,
) -> Optional[str]:
    if USE_BACKTEST_ENTRY_LOGIC:
        long_base, short_base = compute_base_signals_backtest(df, coin_cfg)
    else:
        long_base, short_base = compute_base_signals_live(df, coin_cfg)
    base_long = long_base
    base_short = short_base
    last = df.iloc[-1]
    update_cache(symbol, df)
    if atr_ok is None or body_ok is None or meta is None:
        atr_ok, body_ok, meta = apply_filters(last, coin_cfg)
    dbg = (coin_cfg.get("debug") or {})
    bypass_twap = bool(dbg.get("bypass_twap", False))
    bypass_htf = bool(dbg.get("bypass_htf", False))
    bypass_ml = bool(dbg.get("bypass_ml", False))
    bypass_filters = bool(dbg.get("bypass_filters", False))
    force_base_entry = bool(dbg.get("force_base_entry", False))
    reasons: list[str] = []
    adx_penalty = 0.0
    adx_penalized = False
    g_filters_ok = bool(atr_ok and body_ok)
    long_base = bool(long_base)
    short_base = bool(short_base)
    
    # Inisialisasi htf_close dengan None
    htf_close = None    

    # HTF gating
    _htf = _coerce_htf_cfg(coin_cfg)
    htf_tf = _htf.get("timeframe")
    long_htf_ok = short_htf_ok = True
    if _htf.get("enabled") and htf_tf is not None:
        htf_close = mtf_get_close(symbol, htf_tf)
        if htf_close is not None and len(htf_close) >= max(_htf.get("ema_period",22), _htf.get("sma_period",22)):
            ema_htf = htf_close.ewm(span=_htf.get("ema_period",22), adjust=False).mean().iloc[-1]
            sma_htf = htf_close.rolling(_htf.get("sma_period",22)).mean().iloc[-1]
            rules = str(_htf.get("rule", "")).lower().split(';')
            if "long_ema>=sma" in rules:
                long_htf_ok = ema_htf >= sma_htf
            if "long_ema<=sma" in rules:
                long_htf_ok = ema_htf <= sma_htf
            if "short_ema>=sma" in rules:
                short_htf_ok = ema_htf >= sma_htf
            if "short_ema<=sma" in rules:
                short_htf_ok = ema_htf <= sma_htf
        elif not _htf.get("fallback_pass", True):
            long_htf_ok = short_htf_ok = False
    if bypass_htf:
        long_htf_ok = short_htf_ok = True

    score_long = 1.0 if long_base else 0.0
    score_short = 1.0 if short_base else 0.0
    ml_ready = (ml_up_prob is not None)
    reg_cfg = coin_cfg.get("regime", {})
    th = RegimeThresholds(
        adx_period=int(reg_cfg.get("adx_period", 14)),
        bb_period=int(reg_cfg.get("bb_period", 20)),
        trend_threshold=float(reg_cfg.get("trend_threshold", 20.0)),
        bb_width_chop_max=float(reg_cfg.get("bb_width_chop_max", 0.010)),
    )
    regime = get_regime(df, th) if reg_cfg.get("enabled", False) else "CHOP"

    filt_cfg = coin_cfg.get("filters", {}) if isinstance(coin_cfg.get("filters"), dict) else {}
    if bool(filt_cfg.get("adx_filter_enabled", False)):
        adx_now = float(_regime_adx(df, int(reg_cfg.get("adx_period", 14))).iloc[-1])
        min_adx = float(filt_cfg.get("min_adx", 16.0))
        if adx_now < min_adx:
            if str(filt_cfg.get("adx_mode", "soft")).lower() == "soft":
                adx_penalty = float(filt_cfg.get("adx_penalty", 0.25))
                score_long -= adx_penalty
                score_short -= adx_penalty
                adx_penalized = True
                reasons.append("adx_low_penalty")
            else:
                reasons.append("adx_low")
                g_filters_ok = False
    if bypass_filters:
        g_filters_ok = True
        atr_ok = body_ok = True
        if adx_penalized:
            score_long += adx_penalty
            score_short += adx_penalty
            adx_penalized = False

    if not g_filters_ok:
        long_base = False
        short_base = False
        score_long = 0.0
        score_short = 0.0

    tw_cfg = coin_cfg
    use_twap = bool(tw_cfg.get("use_twap_indicator", False))
    tw_bonus = 0.2
    twap15_w = int(tw_cfg.get("twap_15m_window", 20))
    twap5_w = int(tw_cfg.get("twap_5m_window", 36))
    twap1_w = int(tw_cfg.get("twap_1m_window", 120))

    if use_twap:
        atr = float(last.get("atr") or 0.0)
        if atr <= 0.0:
            try:
                hi = df["high"]
                lo = df["low"]
                cl = df["close"]
                tr = pd.concat([hi - lo, (hi - cl.shift()).abs(), (cl.shift() - lo).abs()], axis=1).max(axis=1)
                atr = float(tr.rolling(14).mean().iloc[-1])
            except Exception:
                atr = 0.0
        twap15_series = rolling_twap(df["close"], twap15_w)
        twap15 = twap15_series.iloc[-1]
        price_now = float(last["close"])
        cfg_adp = coin_cfg.get("twap_adaptive", {})
        if cfg_adp.get("enabled", False):
            if regime == "TREND":
                k = float(cfg_adp["trend"].get("twap_atr_k", 0.40))
                buffer_mult = float(cfg_adp["trend"].get("buffer_atr_mult", 0.08))
                confirm_bars = int(cfg_adp["trend"].get("confirm_bars", 1))
            else:
                k = float(cfg_adp["chop"].get("twap_atr_k", 0.70))
                buffer_mult = float(cfg_adp["chop"].get("buffer_atr_mult", 0.00))
                confirm_bars = int(cfg_adp["chop"].get("confirm_bars", 0))
        else:
            k = float(coin_cfg.get("twap_atr_k", 0.55))
            entry_cfg = coin_cfg.get("entry", {})
            buffer_mult = float(entry_cfg.get("buffer_atr_mult", 0.12))
            confirm_bars = int(entry_cfg.get("confirm_bars", 2))
        reasons.append(f"regime={regime}")
        reasons.append(f"k_used={k}")
        reasons.append(f"buffer_used={buffer_mult}")
        reasons.append(f"confirm_bars_used={confirm_bars}")
        entry_cfg = coin_cfg.get("entry", {})
        allow_ct_short = bool(entry_cfg.get("allow_countertrend_short", False))
        buf = buffer_mult * atr
        long_bonus_ok = short_bonus_ok = False
        if htf_close is not None and len(htf_close) >= 30:
            htwap = rolling_twap(htf_close, 22).iloc[-1]
            ema22_htf = htf_close.ewm(span=22, adjust=False).mean().iloc[-1]
            price_htf = htf_close.iloc[-1]
            long_bonus_ok = (price_htf >= htwap) and (ema22_htf >= htwap)
            short_bonus_ok = (price_htf <= htwap) and (ema22_htf <= htwap)
        # === PR6.4: Bind defaults to satisfy static analyzer & unify checks ===
        twap15_ok_long: bool = False
        twap15_ok_short: bool = False

        # guard untuk nilai yang dipakai di semua jalur
        mode: str = str(coin_cfg.get("twap15_trend_mode", "any_of")).lower()
        ema22_now: float = float(last.get("ema_22", 0.0)) if "ema_22" in last else float(df["close"].iloc[-1])
        buf: float = float(buffer_mult) * float(atr)

        # === PR6.4: TWAP logic (TREND any_of / CHOP deviasi) ===
        if regime == "TREND":
            cond_long  = [(price_now >= twap15 + buf), (ema22_now >= twap15)]
            cond_short = [(price_now <= twap15 - buf), (ema22_now <= twap15)]
            if mode == "all_of":
                twap15_ok_long  = all(cond_long)
                twap15_ok_short = all(cond_short)
            else:
                twap15_ok_long  = any(cond_long)
                twap15_ok_short = any(cond_short)
            reasons.append(f"twap15_trend_mode={mode}")
            reasons.append(f"ema22_vs_twap={'LONG' if ema22_now >= twap15 else 'SHORT'}")
        else:
            twap15_ok_long  = (price_now <= twap15 - k * atr)
            twap15_ok_short = (price_now >= twap15 + k * atr)

        def _confirmed(side: str, n: int) -> bool:
            if n <= 0:
                return True
            closes = df["close"].iloc[-n:]
            if side == "LONG":
                return bool((closes >= twap15).all())
            else:
                return bool((closes <= twap15).all())
        if not _confirmed("LONG", confirm_bars):
            twap15_ok_long = False
            reasons.append("trend_confirm_fail")
        if not _confirmed("SHORT", confirm_bars):
            twap15_ok_short = False
            reasons.append("trend_confirm_fail")
        # Jika gagal TWAP tapi HTF mendukung kuat, override ringan
        if not twap15_ok_long and regime == "TREND" and long_bonus_ok:
            twap15_ok_long = True
            reasons.append("twap15_override_htf_bonus")
        if not twap15_ok_short and regime == "TREND" and short_bonus_ok:
            twap15_ok_short = True
            reasons.append("twap15_override_htf_bonus")
        if regime != "TREND" and short_base and not allow_ct_short:
            short_base = False
            score_short = 0.0
            reasons.append("ct_short_disabled")
        twap5 = mtf_get_close(symbol, "5min")
        twap1 = mtf_get_close(symbol, "1min")
        ltf_ok_long = True
        ltf_ok_short = True
        # HTF bonus
        if long_base and long_bonus_ok:
            score_long += tw_bonus
            reasons.append("twap_htf_bonus")
        if short_base and short_bonus_ok:
            score_short += tw_bonus
            reasons.append("twap_htf_bonus")
    score_rule_only = max(score_long, score_short)
    reasons.append(f"score_rule_only={score_rule_only:.3f}")
    ml_cfg = coin_cfg.get("ml", {})
    ml_weight = float(ml_cfg.get("weight", 1.0))
    ml_thr = float(ml_cfg.get("score_threshold", 2.0))
    cond = ml_cfg.get("conditional_relax", {})
    if cond.get("enabled", False):
        gate = float(cond.get("rule_score_gate", 1.7))
        relaxed = float(cond.get("relaxed_threshold", 1.3))
        if score_rule_only >= gate:
            ml_thr = relaxed
    ml_thr_used = ml_thr
    reasons.append(f"ml_thr_used={ml_thr_used:.3f}")
    ml_prob_long = ml_up_prob if ml_up_prob is not None else 0.0
    ml_prob_short = 1 - ml_prob_long
    score_long = score_long + (ml_prob_long * ml_weight)
    score_short = score_short + (ml_prob_short * ml_weight)
    side = "LONG" if score_long > score_short else "SHORT"
    final_score = score_long if side == "LONG" else score_short
    if bypass_ml:
        reasons.append("bypass_ml_applied")
        ml_pass = True
    else:
        ml_pass = final_score >= ml_thr_used
        if not ml_pass:
            reasons.append("ml_gate_fail")
    decision = None
    if ml_pass:
        if side == "LONG" and long_base and score_long >= ml_thr_used:
            decision = "LONG"
        elif side == "SHORT" and short_base and score_short >= ml_thr_used:
            decision = "SHORT"
    g_ml_ok = bool(ml_pass)

    # === PR6.4: Collapse to single gate by side ===
    # Inisialisasi dengan nilai default
    twap15_ok_long = False
    twap15_ok_short = False
    twap15_ok: bool = twap15_ok_long if side == "LONG" else twap15_ok_short
    if not twap15_ok:
        reasons.append("twap15_ok=False")

    g_htf_ok = bool(long_htf_ok if side == "LONG" else short_htf_ok)
    g_twap_ok = bool(twap15_ok)
    g_cooldown_ok = not bool(cooldown_active)
    g_pos_free = not bool(position)

    if bypass_twap:
        g_twap_ok = True
        reasons.append("bypass_twap_applied")
    if bypass_htf:
        g_htf_ok = True
        reasons.append("bypass_htf_applied")
    if bypass_filters:
        g_filters_ok = True
        reasons.append("bypass_filters_applied")
    if bypass_ml:
        g_ml_ok = True
        reasons.append("bypass_ml_applied")

    allow_reasons: list[str] = []
    conds = [
        ("htf_ok", g_htf_ok),
        ("twap15_ok", g_twap_ok),
        ("filters_ok", g_filters_ok),
        ("ml_ok", g_ml_ok),
        ("cooldown_ok", g_cooldown_ok),
        ("position_free", g_pos_free),
    ]
    allow = True
    for name, ok in conds:
        if not ok:
            allow = False
            allow_reasons.append(name)

    if force_base_entry and ((side == "LONG" and base_long) or (side == "SHORT" and base_short)):
        reasons.append("force_base_entry_applied")
        decision = side
        allow = True
        allow_reasons = []

    reasons.append(f"allow_final={allow}")
    if not allow:
        reasons.append(f"blocked_by={'|'.join(allow_reasons) if allow_reasons else '-'}")
    reasons.append(f"twap15_ok={twap15_ok}")

    ema = float(last.get('ema_22', 0.0))
    ma = float(last.get('ma_22', 0.0))
    rsi = float(last.get('rsi', 0.0))
    bb_width = meta.get('bb_width_pct') if isinstance(meta, dict) else float('nan')

    logging.getLogger(__name__).info(
        f"[{symbol}] DECISION ema22={ema:.6f} ma22={ma:.6f} rsi={rsi:.2f} base(L={long_base},S={short_base}) "
        f"atr_ok={atr_ok} body_ok={body_ok} bb_width_pct={bb_width:.4f} "
        f"ml_up_prob={ml_up_prob} score_rule={score_rule_only:.2f} ml_thr={ml_thr_used:.2f} "
        f"score(L={score_long:.2f},S={score_short:.2f}) reasons={reasons} -> {decision}"
    )
    if ctx is not None:
        ctx["allow"] = allow
        ctx["allow_reasons"] = allow_reasons
        ctx["reasons"] = reasons
    return decision

def _norm_resample_freq(freq: str | None, default: str = "1h") -> str:
    """Normalisasi frekuensi untuk pandas.resample.
    - Pakai huruf kecil sesuai anjuran pandas >2.
    - Ubah akhiran 'm' ke 'min' agar tidak ambigu dengan bulan.
    """
    f = (str(freq or default)).strip().lower()
    if f.endswith("m") and not f.endswith("min"):
        if f.endswith("ms") or f.endswith("us"):
            return f
        f = f[:-1] + "min"
    return f


def htf_trend_ok(side: str, base_df: pd.DataFrame, htf: str = '1h') -> bool:
    try:
        d = base_df.copy()
        if 'timestamp' in d.columns:
            d = d.set_index(pd.to_datetime(d['timestamp'], utc=True))
        else:
            d.index = pd.to_datetime(d.index, utc=True)
        res = _norm_resample_freq(htf, "1h")
        htf_close = d['close'].resample(res).last().dropna()
        if len(htf_close) < 30:
            return True
        ema22 = htf_close.ewm(span=22, adjust=False).mean().iloc[-1]
        sma22 = htf_close.rolling(22).mean().iloc[-1]
        if np.isnan(ema22) or np.isnan(sma22):
            return True
        return (ema22 >= sma22) if side == 'LONG' else (ema22 <= sma22)
    except Exception:
        return True
def apply_filters(ind: pd.Series, coin_cfg: Dict[str, Any]) -> Tuple[bool, bool, Dict[str, Any]]:
    filters_cfg = (coin_cfg.get('filters') if isinstance(coin_cfg.get('filters'), dict) else {}) or {}
    min_atr = _to_float(filters_cfg.get('min_atr_threshold', coin_cfg.get('min_atr_pct', 0.0)), 0.0)
    max_atr = _to_float(coin_cfg.get('max_atr_pct', filters_cfg.get('max_atr_pct', 1.0)), 1.0)
    max_body = _to_float(filters_cfg.get('max_body_over_atr', coin_cfg.get('max_body_atr', 999.0)), 999.0)
    min_bb = _to_float(filters_cfg.get('min_bb_width', 0.0), 0.0)
    atr_filter_enabled = bool(filters_cfg.get('atr_filter_enabled', filters_cfg.get('atr', False)))
    body_filter_enabled = bool(filters_cfg.get('body_filter_enabled', filters_cfg.get('body', False)))
    adx_filter_enabled = bool(filters_cfg.get('adx_filter_enabled', False))
    min_adx = _to_float(filters_cfg.get('min_adx', 0.0), 0.0)

    atr_pct = float(ind.get('atr_pct', 0.0))
    bb_width_pct = float(ind.get('bb_width_pct', 0.0))
    body_val = ind.get('body_to_atr', ind.get('body_atr'))
    body_to_atr = float(body_val) if body_val is not None else float('nan')
    adx_val = float(ind.get('adx', 0.0))

    atr_ok = (atr_pct >= min_atr) and (atr_pct <= max_atr) and (bb_width_pct >= min_bb)
    body_ok = body_to_atr <= max_body
    adx_ok = adx_val >= min_adx if adx_filter_enabled else True

    if not atr_filter_enabled:
        atr_ok = True
    if not body_filter_enabled:
        body_ok = True

    if (not atr_ok or not body_ok) or (adx_filter_enabled and not adx_ok):
        logging.getLogger(__name__).info(
            f"[{coin_cfg.get('symbol', '?')}] FILTER INFO atr_ok={atr_ok} body_ok={body_ok} adx={adx_val:.2f} "
            f"bb_width_pct={bb_width_pct:.4f} atr_pct={atr_pct:.4f} body_to_atr={body_to_atr:.4f}"
        )

    meta = {
        'atr_pct': atr_pct,
        'bb_width_pct': bb_width_pct,
        'body_to_atr': body_to_atr,
        'adx': adx_val,
    }
    return atr_ok, body_ok, meta

def decide_base(ind: pd.Series, coin_cfg: Dict[str, Any]) -> Dict[str, bool]:
    return {'L': bool(ind.get('long_base', False)), 'S': bool(ind.get('short_base', False))}

def confirm_htf(htf_ind: pd.DataFrame, coin_cfg: Dict[str, Any]) -> bool:
    # placeholder konfirmasi HTF: true jika ema20>= ema22
    try:
        ema50 = htf_ind['close'].ewm(span=20, adjust=False).mean().iloc[-1]
        ema200 = htf_ind['close'].ewm(span=22, adjust=False).mean().iloc[-1]
        side = coin_cfg.get('side','LONG')
        return ema50 >= ema200 if side=='LONG' else ema50 <= ema200
    except Exception:
        return True

def apply_ml_gate(up_prob: Optional[float], ml_thr: float, eps: float = SAFE_EPS) -> bool:
    if up_prob is None:
        return True
    p = min(max(up_prob, eps), 1 - eps)
    up_odds = safe_div(p, 1 - p)
    return up_odds >= ml_thr

# -----------------------------------------------------
# Money management & PnL
# -----------------------------------------------------
def risk_size(balance: float, risk_pct: float, entry: float, stop: float,
              fee_bps: float, slip_bps: float, sym_cfg: Dict[str, Any]) -> float:
    lev = _to_int(sym_cfg.get('leverage', 1), 1)
    risk_val = balance * risk_pct
    diff = abs(entry - stop)
    if diff <= 0:
        return 0.0
    qty = safe_div((risk_val * lev), diff)
    step = _to_float(sym_cfg.get('stepSize', 0), 0)
    return floor_to_step(qty, step) if step>0 else to_scalar(qty)

def pnl_net(side: str, entry: float, exit: float, qty: float,
            fee_bps: float, slip_bps: float) -> Tuple[float, float]:
    fee = (entry + exit) * qty * fee_bps/10000.0
    slip = (entry + exit) * qty * slip_bps/10000.0
    gross = (exit - entry) * qty if side=='LONG' else (entry - exit) * qty
    pnl = gross - fee - slip
    roi = safe_div(pnl, (entry*qty)) if entry*qty>0 else 0.0
    return pnl, roi*100.0

def r_multiple(entry: float, sl: float, price: float) -> float:
    r = abs(entry - sl)
    return safe_div(abs(price - entry), r)

def r_multiple_signed(entry: float, sl: float, price: float, side: str) -> float:
    """R multiple bertanda; positif hanya jika bergerak sesuai arah profit."""
    R = abs(entry - sl)
    if R <= 0:
        return 0.0
    move = (price - entry) if side == 'LONG' else (entry - price)
    return safe_div(move, R)

def apply_breakeven_sl(side: str,
                       entry: float,
                       price: float,
                       sl: float | None,
                       tick_size: float = 0.0,
                       min_gap_pct: float = 0.0001,
                       be_trigger_r: float = 0.0,
                       be_trigger_pct: float = 0.0,
                       be_trigger_abs: float = 0.0) -> float | None:
    """
    Hitung SL BE baru bila trigger terpenuhi. Mengembalikan SL baru atau sl lama.
    - BE R-multiple: aktif hanya jika posisi sudah profit sesuai arah.
    - BE %: fallback jika R=0 atau be_trigger_r=0.
    - SL tidak ditempatkan menempel pada harga: beri gap min(max(tick, price*min_gap_pct)).
    """
    if entry is None or side not in ('LONG', 'SHORT'):
        return sl
    gap = max(tick_size or 0.0, abs(price) * (min_gap_pct or 0.0))

    # 0) Absolute profit trigger
    if be_trigger_abs and be_trigger_abs > 0:
        move = (price - entry) if side == 'LONG' else (entry - price)
        if move >= be_trigger_abs:
            target = max(sl or 0.0, entry) if side == 'LONG' else min(sl or 1e18, entry)
            return min(target, price - gap) if side == 'LONG' else max(target, price + gap)

    # 1) R-based (prioritas jika > 0)
    if be_trigger_r and be_trigger_r > 0:
        rnow = r_multiple_signed(entry, sl if sl is not None else entry, price, side)
        if side == 'LONG':
            if price > entry and rnow >= be_trigger_r:
                target = max(sl or 0.0, entry)
                return min(target, price - gap)
        else:  # SHORT
            if price < entry and rnow >= be_trigger_r:
                target = min(sl or 1e18, entry)
                return max(target, price + gap)

    # 2) % fallback
    if be_trigger_pct and be_trigger_pct > 0:
        if side == 'LONG':
            if safe_div((price - entry), entry) >= be_trigger_pct:
                target = max(sl or 0.0, entry)
                return min(target, price - gap)
        else:
            if safe_div((entry - price), entry) >= be_trigger_pct:
                target = min(sl or 1e18, entry)
                return max(target, price + gap)

    return sl

def roi_frac_now(side: str, entry: Optional[float], price: float, qty: float, leverage: int) -> float:
    if entry is None or not qty or leverage <= 0:
        return 0.0
    entry = float(entry)
    price = float(price)
    init_margin = safe_div((entry * abs(qty)), leverage)
    if init_margin <= 0:
        return 0.0
    if side == 'LONG':
        pnl = (price - entry) * qty
    else:
        pnl = (entry - price) * qty
    return safe_div(pnl, init_margin)

def base_supports_side(base_long: bool, base_short: bool, side: str) -> bool:
    return (side == 'LONG' and base_long and not base_short) or (side == 'SHORT' and base_short and not base_long)

# -----------------------------------------------------
# Journaling
# -----------------------------------------------------
def journal_row(ts: str, symbol: str, side: str, entry: float, qty: float,
                sl_init: Optional[float], tsl_init: Optional[float],
                exit_price: float, exit_reason: str,
                pnl_usdt: float, roi_pct: float, balance_after: float) -> Dict[str, Any]:
    return {
        'timestamp': ts,
        'symbol': symbol,
        'side': side,
        'entry': f"{entry:.6f}",
        'qty': f"{qty:.6f}",
        'sl_init': "" if sl_init is None else f"{sl_init:.6f}",
        'tsl_init': "" if tsl_init is None else f"{tsl_init:.6f}",
        'exit_price': f"{exit_price:.6f}",
        'exit_reason': exit_reason,
        'pnl_usdt': f"{pnl_usdt:.4f}",
        'roi_pct': f"{roi_pct:.2f}",
        'balance_after': f"{balance_after:.4f}",
    }

def init_stops(side: str, entry: float, ind: pd.Series, coin_cfg: Dict[str, Any]) -> Dict[str, Optional[float]]:
    atr = float(ind.get('atr', 0.0))
    sl = entry * (1 - _to_float(coin_cfg.get('sl_pct', 0.01), 0.01)) if side=='LONG' else entry * (1 + _to_float(coin_cfg.get('sl_pct', 0.01), 0.01))
    return {'sl': sl, 'tsl': None}

def step_trailing(side: str, bar: pd.Series, prev_state: Dict[str, Optional[float]], ind: pd.Series, coin_cfg: Dict[str, Any]) -> Optional[float]:
    trail_cfg = coin_cfg.get("trailing") if isinstance(coin_cfg.get("trailing"), dict) else None
    entry = prev_state.get('entry', 0.0)
    price = float(bar['close'])
    if entry is None or price is None or entry == 0:
        profit = 0.0
    else:
        profit = (price - entry) if side == 'LONG' else (entry - price)
    if trail_cfg and trail_cfg.get("enabled", True):
        atr = float(ind.get('atr', 0.0))
        trig = float(trail_cfg.get("trigger_atr_mult", 0.8)) * atr
        step_abs = float(trail_cfg.get("step_atr_mult", 0.35)) * atr
        if profit < trig:
            return prev_state.get('tsl')
        if side == 'LONG':
            new_tsl = price - step_abs
            return max(prev_state.get('tsl') or prev_state.get('sl') or 0.0, new_tsl)
        else:
            new_tsl = price + step_abs
            return min(prev_state.get('tsl') or prev_state.get('sl') or 1e18, new_tsl)
    else:
        trigger = _to_float(coin_cfg.get('trailing_trigger', 0.7), 0.7)
        step = _to_float(coin_cfg.get('trailing_step', 0.45), 0.45)
        profit_pct = safe_div(profit, entry) * 100 if entry else 0.0
        if profit_pct < trigger:
            return prev_state.get('tsl')
        if side == 'LONG':
            new_tsl = price*(1-step/100)
            return max(prev_state.get('tsl') or prev_state.get('sl') or 0.0, new_tsl)
        else:
            new_tsl = price*(1+step/100)
            return min(prev_state.get('tsl') or prev_state.get('sl') or 1e18, new_tsl)

def maybe_move_to_BE(side: str, entry: float, tsl: Optional[float], rule: Dict[str, Any]) -> Optional[float]:
    be = _to_float(rule.get('be_trigger_pct', 0.0), 0.0)
    price = rule.get('price', entry)
    if be <= 0:
        return tsl
    if side=='LONG' and safe_div((price-entry), entry) >= be:
        return max(tsl or 0.0, entry)
    if side=='SHORT' and safe_div((entry-price), entry) >= be:
        return min(tsl or 1e18, entry)
    return tsl

def check_time_stop(entry_time: float, now: float, roi: float, rule: Dict[str, Any]) -> bool:
    max_secs = _to_int(rule.get('time_stop_secs', 0), 0)
    return (max_secs>0) and ((now-entry_time) >= max_secs)

def cooldown_until(now: float, rule: Dict[str, Any]) -> float:
    secs = _to_int(rule.get('cooldown_secs', 0), 0)
    return now + max(0, secs)

def simulate_fill_on_candle(side: str, state: Dict[str, float], bar: pd.Series, sym_cfg: Dict[str, Any], fee_bps: float, slip_bps: float) -> Optional[Tuple[float, str]]:
    o, h, l, c = bar['open'], bar['high'], bar['low'], bar['close']
    sl = state.get('sl'); tsl = state.get('tsl')
    if side=='LONG':
        if tsl is not None and l <= tsl:
            return tsl, 'Hit Trailing SL'
        if sl is not None and l <= sl:
            return sl, 'Hit Hard SL'
    else:
        if tsl is not None and h >= tsl:
            return tsl, 'Hit Trailing SL'
        if sl is not None and h >= sl:
            return sl, 'Hit Hard SL'
    return None
