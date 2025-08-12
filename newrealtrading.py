#!/usr/bin/env python3
"""
newrealtrading.py — Engine trading (live/dry-run) sinkron dengan backtester

PATCH 2025-08-12 (v-tsv2-adx):
- Time Stop v2: hanya jika rugi (roi<MIN_ROI_TO_CLOSE_BY_TIME, default 0.0), tahan lebih lama jika BE aktif (max_hold_seconds_be)
- Early Stop wick-aware: pakai low/high bar (bukan close) untuk ukur adverse move pada N bar pertama → cegah Hard SL dini
- Trend-strength filter opsional: EMA200 + ADX ("ema200_adx")
- Tetap: trailing ATR adaptif, cooldown dinamis, BE+, exit pakai bar-time
"""
from __future__ import annotations
import os, json, time, argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List, Protocol, runtime_checkable, cast
import numpy as np
import pandas as pd

# =============================================================
# Helpers
# =============================================================
def _to_float(x: Any, d: float) -> float:
    try:
        if x is None: return float(d)
        return float(x)
    except Exception:
        return float(d)

def _to_int(x: Any, d: int) -> int:
    try:
        if x is None: return int(d)
        return int(float(x))
    except Exception:
        return int(d)

def _to_bool(x: Any, d: bool) -> bool:
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(int(x))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"1","true","y","yes","on"}: return True
        if s in {"0","false","n","no","off"}: return False
    return bool(d)

# =============================================================
# Defaults
# =============================================================
DEFAULTS: Dict[str, Any] = {
    "leverage": 10,
    "risk_per_trade": 0.05,
    "taker_fee": 0.0005,

    "min_atr_pct": 0.006,
    "max_atr_pct": 0.03,
    "max_body_atr": 1.0,
    "use_htf_filter": 0,

    "cooldown_seconds": 1200,

    "sl_mode": "ATR",
    "sl_pct": 0.008,
    "sl_atr_mult": 1.6,
    "sl_min_pct": 0.010,
    "sl_max_pct": 0.030,

    "use_breakeven": 1,
    "be_trigger_pct": 0.006,
    "be_offset_pct": 0.0008,

    "trailing_trigger": 1.0,   # %
    "trailing_step_min_pct": 0.45,
    "trailing_step_max_pct": 1.00,
    "trail_atr_k": 2.0,

    "cooldown_mul_hard_sl": 2.5,
    "cooldown_mul_trailing": 1.0,
    "cooldown_mul_early_stop": 1.2,

    # Early stop
    "early_stop_enabled": 1,
    "early_stop_bars": 3,
    "early_stop_adverse_atr": 0.6,  # kelipatan ATR terhadap harga entry

    # Time stop
    "max_hold_seconds": 3600,
    "min_roi_to_close_by_time": 0.0,
    "time_stop_only_if_loss": 1,
    "max_hold_seconds_be": 7200,

    # Trend filter
    "trend_filter_mode": "",     # "ema200" | "ema200_adx"
    "adx_period": 14,
    "adx_thresh": 18.0,
}

# =============================================================
# Indikator
# =============================================================

def _ema(s: pd.Series, window: int) -> pd.Series:
    return pd.Series(pd.to_numeric(s, errors='coerce')).ewm(span=window, adjust=False).mean()


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    c = pd.Series(pd.to_numeric(close, errors='coerce'))
    delta = c.diff()
    up = np.where(delta.to_numpy() > 0, delta.to_numpy(), 0.0)
    down = np.where(delta.to_numpy() < 0, -delta.to_numpy(), 0.0)
    roll_up = pd.Series(up).ewm(alpha=1/window, adjust=False).mean()
    roll_down = pd.Series(down).ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi).fillna(50.0)


def _macd(close: pd.Series, fast=12, slow=26, sign=9) -> Tuple[pd.Series, pd.Series]:
    c = pd.Series(pd.to_numeric(close, errors='coerce'))
    macd = _ema(c, fast) - _ema(c, slow)
    signal = macd.ewm(span=sign, adjust=False).mean()
    return macd, signal


def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    h = pd.to_numeric(df['high'], errors='coerce')
    l = pd.to_numeric(df['low'], errors='coerce')
    c = pd.to_numeric(df['close'], errors='coerce')
    prev_close = c.shift(1)
    tr = pd.concat([
        (h - l).abs(),
        (h - prev_close).abs(),
        (l - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h = pd.to_numeric(df['high'], errors='coerce')
    l = pd.to_numeric(df['low'], errors='coerce')
    c = pd.to_numeric(df['close'], errors='coerce')
    up_move = h.diff()
    down_move = -l.diff()
    plus_dm = np.where((pd.to_numeric(up_move, errors='coerce') > pd.to_numeric(down_move, errors='coerce')) & (pd.to_numeric(up_move, errors='coerce') > 0), up_move, 0.0)
    minus_dm = np.where((pd.to_numeric(down_move, errors='coerce') > pd.to_numeric(up_move, errors='coerce')) & (pd.to_numeric(down_move, errors='coerce') > 0), down_move, 0.0)

    tr1 = (h - l)
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(period).mean() / atr.replace(0.0, np.nan))
    minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean() / atr.replace(0.0, np.nan))
    dx = ( (plus_di - minus_di).abs() / (plus_di + minus_di).abs().replace(0.0, np.nan) ) * 100
    adx = dx.rolling(period).mean()
    return adx.fillna(0.0)


def calculate_indicators(df: pd.DataFrame, adx_period: int = 14) -> pd.DataFrame:
    d = df.copy()
    if 'timestamp' not in d.columns:
        if 'open_time' in d.columns:
            d['timestamp'] = pd.to_datetime(d['open_time'], unit='ms', errors='coerce')
        elif 'date' in d.columns:
            d['timestamp'] = pd.to_datetime(d['date'], errors='coerce')
    d['timestamp'] = pd.to_datetime(d['timestamp'], utc=True)
    d = d.sort_values('timestamp').reset_index(drop=True)

    d['ema_fast'] = _ema(d['close'], 20)
    d['ema_slow'] = _ema(d['close'], 50)
    d['ema_200'] = _ema(d['close'], 200)
    macd, macd_sig = _macd(d['close'])
    d['macd'] = macd
    d['macd_sig'] = macd_sig
    d['rsi'] = _rsi(d['close'], 14)
    d['atr'] = _atr(d, 14)
    d['adx'] = _adx(d, adx_period)
    d['body'] = (pd.to_numeric(d['close']) - pd.to_numeric(d['open'])).abs()
    d['body_atr'] = d['body'] / d['atr']
    return d

# =============================================================
# ML plugin (Protocol)
# =============================================================
@runtime_checkable
class MLSignalProto(Protocol):
    @property
    def use_ml(self) -> bool: ...
    def fit_if_needed(self, df: pd.DataFrame) -> None: ...
    def predict_up_prob(self, df: pd.DataFrame) -> Optional[float]: ...
    def score_and_decide(self, base_long: bool, base_short: bool, up_prob: Optional[float] = None) -> Tuple[bool, bool]: ...

try:
    from ml_signal_plugin import MLSignal as MLImplClass  # type: ignore
except Exception:
    MLImplClass = None  # type: ignore

class _MLStub:
    def __init__(self, coin_cfg: Dict[str, Any] | None):
        self.params = type("P", (), dict(use_ml=False, score_threshold=1.0, up_prob_thres=0.55, down_prob_thres=0.45))
    @property
    def use_ml(self) -> bool: return False
    def fit_if_needed(self, df: pd.DataFrame) -> None: return
    def predict_up_prob(self, df: pd.DataFrame) -> Optional[float]: return None
    def score_and_decide(self, base_long: bool, base_short: bool, up_prob: Optional[float] = None):
        th = float(getattr(self.params, "score_threshold", 1.0))
        sc_long = 1.0 if base_long else 0.0
        sc_short = 1.0 if base_short else 0.0
        return (sc_long >= th, sc_short >= th)

# =============================================================
# Position
# =============================================================
@dataclass
class Position:
    side: Optional[str] = None
    entry: Optional[float] = None
    qty: Optional[float] = None
    sl: Optional[float] = None
    trailing_sl: Optional[float] = None
    entry_time: Optional[pd.Timestamp] = None

# =============================================================
# Trader
# =============================================================
class CoinTrader:
    def __init__(self, symbol: str, coin_cfg: Dict[str, Any]):
        self.symbol = symbol
        self.config = coin_cfg or {}
        self.pos = Position()
        self.cooldown_until_ts: Optional[float] = None
        self.verbose = _to_bool(self.config.get('VERBOSE', os.getenv('VERBOSE','0')), False)

        self.cooldown_use_bar_time = _to_bool(self.config.get('COOLDOWN_USE_BAR_TIME', os.getenv('COOLDOWN_USE_BAR_TIME','1')), True)
        self.early_stop_enabled = _to_bool(self.config.get('early_stop_enabled', DEFAULTS['early_stop_enabled']), DEFAULTS['early_stop_enabled'])
        self.early_stop_bars = _to_int(self.config.get('early_stop_bars', DEFAULTS['early_stop_bars']), DEFAULTS['early_stop_bars'])
        self.early_stop_adverse_atr = _to_float(self.config.get('early_stop_adverse_atr', DEFAULTS['early_stop_adverse_atr']), DEFAULTS['early_stop_adverse_atr'])

        self.max_hold_seconds = _to_int(self.config.get('max_hold_seconds', os.getenv('MAX_HOLD_SECONDS', DEFAULTS['max_hold_seconds'])), DEFAULTS['max_hold_seconds'])
        self.max_hold_seconds_be = _to_int(self.config.get('max_hold_seconds_be', DEFAULTS['max_hold_seconds_be']), DEFAULTS['max_hold_seconds_be'])
        self.min_roi_time_close = _to_float(self.config.get('min_roi_to_close_by_time', os.getenv('MIN_ROI_TO_CLOSE_BY_TIME', DEFAULTS['min_roi_to_close_by_time'])), DEFAULTS['min_roi_to_close_by_time'])
        self.time_stop_only_if_loss = _to_bool(self.config.get('time_stop_only_if_loss', DEFAULTS['time_stop_only_if_loss']), DEFAULTS['time_stop_only_if_loss'])

        self.trend_filter_mode = str(self.config.get('trend_filter_mode', DEFAULTS['trend_filter_mode'])).lower().strip()
        self.adx_period = _to_int(self.config.get('adx_period', DEFAULTS['adx_period']), DEFAULTS['adx_period'])
        self.adx_thresh = _to_float(self.config.get('adx_thresh', DEFAULTS['adx_thresh']), DEFAULTS['adx_thresh'])

        if MLImplClass is not None:
            self.ml: MLSignalProto = cast(MLSignalProto, MLImplClass(self.config))
        else:
            self.ml: MLSignalProto = _MLStub(self.config)

    # ---------------------------- utils ----------------------------
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[{self.symbol}] {pd.Timestamp.utcnow().isoformat()} | {msg}")

    def _now_ts(self, last_ts: Optional[pd.Timestamp]) -> float:
        if self.cooldown_use_bar_time and isinstance(last_ts, pd.Timestamp) and pd.notna(last_ts):
            return float(last_ts.timestamp())
        return time.time()

    def _cooldown_active(self, now_ts: Optional[float] = None) -> bool:
        now = float(now_ts) if now_ts is not None else time.time()
        return bool(self.cooldown_until_ts and now < self.cooldown_until_ts)

    # ------------------------- sizing & SL -------------------------
    def _size_position(self, price: float, balance: float, atr: float) -> float:
        lev = _to_float(self.config.get('leverage', DEFAULTS['leverage']), DEFAULTS['leverage'])
        risk_pct = _to_float(self.config.get('risk_per_trade', DEFAULTS['risk_per_trade']), DEFAULTS['risk_per_trade'])
        risk_usdt = max(0.0001, balance * risk_pct)
        sl_atr_mult = _to_float(self.config.get('sl_atr_mult', DEFAULTS['sl_atr_mult']), DEFAULTS['sl_atr_mult'])
        sl_min_pct = _to_float(self.config.get('sl_min_pct', DEFAULTS['sl_min_pct']), DEFAULTS['sl_min_pct'])
        sl_dist_pct = max(sl_min_pct, (sl_atr_mult * (atr/price)))
        notional = (risk_usdt / max(sl_dist_pct, 1e-6)) * lev
        qty = max(0.0, notional / price)
        return round(qty, 3)

    def _hard_sl_price(self, price: float, atr: float, side: str) -> float:
        mode = str(self.config.get('sl_mode', DEFAULTS['sl_mode'])).upper()
        if mode == 'ATR':
            mult = _to_float(self.config.get('sl_atr_mult', DEFAULTS['sl_atr_mult']), DEFAULTS['sl_atr_mult'])
            min_pct = _to_float(self.config.get('sl_min_pct', DEFAULTS['sl_min_pct']), DEFAULTS['sl_min_pct'])
            max_pct = _to_float(self.config.get('sl_max_pct', DEFAULTS['sl_max_pct']), DEFAULTS['sl_max_pct'])
            sl_pct = min(max(min_pct, mult * (atr/price)), max_pct)
        else:
            sl_pct = _to_float(self.config.get('sl_pct', DEFAULTS['sl_pct']), DEFAULTS['sl_pct'])
        return price * (1.0 - sl_pct) if side == 'LONG' else price * (1.0 + sl_pct)

    # ------------------------ BE & Trailing ------------------------
    def _apply_breakeven(self, price: float) -> None:
        if not (self.pos and self.pos.side and self.pos.entry):
            return
        if not _to_bool(self.config.get('use_breakeven', DEFAULTS['use_breakeven']), DEFAULTS['use_breakeven']):
            return
        be_trg = _to_float(self.config.get('be_trigger_pct', DEFAULTS['be_trigger_pct']), DEFAULTS['be_trigger_pct'])
        be_off = _to_float(self.config.get('be_offset_pct', DEFAULTS['be_offset_pct']), DEFAULTS['be_offset_pct'])
        ent = float(self.pos.entry)
        if self.pos.side == 'LONG':
            if (price - ent)/ent >= be_trg:
                new_sl = ent * (1 + be_off)
                if not self.pos.sl or new_sl > self.pos.sl:
                    self.pos.sl = new_sl
        else:
            if (ent - price)/ent >= be_trg:
                new_sl = ent * (1 - be_off)
                if not self.pos.sl or new_sl < self.pos.sl:
                    self.pos.sl = new_sl

    def _update_trailing(self, price: float, atr: float) -> None:
        if not (self.pos and self.pos.side and self.pos.entry):
            return
        trg = _to_float(self.config.get('trailing_trigger', DEFAULTS['trailing_trigger']), DEFAULTS['trailing_trigger'])
        ent = float(self.pos.entry)
        moved_long = (price - ent)/ent * 100.0
        moved_short = (ent - price)/ent * 100.0
        if self.pos.side == 'LONG':
            if moved_long < trg: return
        else:
            if moved_short < trg: return
        atr_pct_now = (atr / max(price, 1e-9)) * 100.0
        step_min = _to_float(self.config.get('trailing_step_min_pct', DEFAULTS['trailing_step_min_pct']), DEFAULTS['trailing_step_min_pct'])
        step_max = _to_float(self.config.get('trailing_step_max_pct', DEFAULTS['trailing_step_max_pct']), DEFAULTS['trailing_step_max_pct'])
        k = _to_float(self.config.get('trail_atr_k', DEFAULTS['trail_atr_k']), DEFAULTS['trail_atr_k'])
        dyn_step = max(step_min, min(step_max, k * atr_pct_now))
        if self.pos.side == 'LONG':
            new_tsl = price * (1.0 - dyn_step/100.0)
            self.pos.trailing_sl = max(self.pos.trailing_sl or 0.0, new_tsl)
        else:
            new_tsl = price * (1.0 + dyn_step/100.0)
            self.pos.trailing_sl = min(self.pos.trailing_sl or 1e12, new_tsl)

    # -------------------------- Exit check -------------------------
    def _should_exit(self, price: float) -> Tuple[bool, Optional[str]]:
        if not self.pos.side:
            return False, None
        if self.pos.side == 'LONG':
            if self.pos.sl and price <= self.pos.sl: return True, 'Hit Hard SL'
            if self.pos.trailing_sl and price <= self.pos.trailing_sl: return True, 'Hit Trailing SL'
        else:
            if self.pos.sl and price >= self.pos.sl: return True, 'Hit Hard SL'
            if self.pos.trailing_sl and price >= self.pos.trailing_sl: return True, 'Hit Trailing SL'
        return False, None

    # -------------------- Early stop (wick-aware) ------------------
    def _early_stop_check(self, df: pd.DataFrame, now_ts: Optional[float]) -> bool:
        if not self.early_stop_enabled or not (self.pos and self.pos.side and self.pos.entry and self.pos.entry_time is not None):
            return False
        last = df.iloc[-1]
        atr = float(last.get('atr', 0.0) or 0.0)
        if atr <= 0: return False
        try:
            bars_since = int((df['timestamp'] > self.pos.entry_time).sum())
        except Exception:
            bars_since = 0
        if bars_since <= 0 or bars_since > int(self.early_stop_bars):
            return False
        ent = float(self.pos.entry)
        # wick-aware: LONG pakai low, SHORT pakai high
        low = float(last.get('low', last.get('close', ent)))
        high = float(last.get('high', last.get('close', ent)))
        adverse_thr = float(self.early_stop_adverse_atr) * (atr / max(ent, 1e-9))
        if self.pos.side == 'LONG':
            dd = (ent - low)/max(ent, 1e-9)
            if dd >= adverse_thr:
                self._exit_position(low, f"Early Stop wick (dd={dd*100:.2f}% >= {self.early_stop_adverse_atr}*ATR)", now_ts=now_ts)
                return True
        else:
            dd = (high - ent)/max(ent, 1e-9)
            if dd >= adverse_thr:
                self._exit_position(high, f"Early Stop wick (dd={dd*100:.2f}% >= {self.early_stop_adverse_atr}*ATR)", now_ts=now_ts)
                return True
        return False

    # ---------------------- Entry / Exit core ----------------------
    def _enter_position(self, side: str, price: float, atr: float, balance: float, now_ts: Optional[float] = None) -> None:
        if self._cooldown_active(now_ts):
            return
        qty = self._size_position(price, balance, atr)
        if qty <= 0: return
        sl = self._hard_sl_price(price, atr, side)
        et = pd.to_datetime(now_ts, unit='s', utc=True) if now_ts is not None else pd.Timestamp.utcnow()
        self.pos = Position(side=side, entry=price, qty=qty, sl=sl, trailing_sl=None, entry_time=et)
        self._log(f"ENTRY {side} price={price:.6f} qty={qty:.6f} sl={sl:.6f}")

    def _exit_position(self, price: float, reason: str, now_ts: Optional[float] = None) -> None:
        self._log(f"EXIT {reason} price={price:.6f}")
        cd_base = _to_int(self.config.get('cooldown_seconds', DEFAULTS['cooldown_seconds']), DEFAULTS['cooldown_seconds'])
        r = (reason or '').lower()
        mul = 1.0
        if 'hard sl' in r:
            mul = _to_float(self.config.get('cooldown_mul_hard_sl', DEFAULTS['cooldown_mul_hard_sl']), DEFAULTS['cooldown_mul_hard_sl'])
        elif 'trailing' in r:
            mul = _to_float(self.config.get('cooldown_mul_trailing', DEFAULTS['cooldown_mul_trailing']), DEFAULTS['cooldown_mul_trailing'])
        elif 'early stop' in r:
            mul = _to_float(self.config.get('cooldown_mul_early_stop', DEFAULTS['cooldown_mul_early_stop']), DEFAULTS['cooldown_mul_early_stop'])
        base_now = float(now_ts) if now_ts is not None else time.time()
        self.cooldown_until_ts = base_now + max(0, int(cd_base * mul))
        self.pos = Position()

    # ---------------------- Sinyal & Keputusan ---------------------
    def check_trading_signals(self, df: pd.DataFrame, balance: float) -> None:
        if df is None or len(df) < 50:
            return
        d = calculate_indicators(df, adx_period=self.adx_period)
        last = d.iloc[-1]
        price = float(last['close'])
        atr = float(last.get('atr', 0.0) or 0.0)
        last_ts = last['timestamp'] if 'timestamp' in last else None
        now_ts = self._now_ts(last_ts)

        # Filter volatilitas & body
        atr_pct = (atr / max(price, 1e-9))
        min_atr_pct = _to_float(self.config.get('min_atr_pct', DEFAULTS['min_atr_pct']), DEFAULTS['min_atr_pct'])
        max_atr_pct = _to_float(self.config.get('max_atr_pct', DEFAULTS['max_atr_pct']), DEFAULTS['max_atr_pct'])
        body_atr_ok = bool(last.get('body_atr', np.nan) <= _to_float(self.config.get('max_body_atr', DEFAULTS['max_body_atr']), DEFAULTS['max_body_atr']))
        atr_ok = (atr_pct >= min_atr_pct) and (atr_pct <= max_atr_pct)

        # Kelola posisi aktif dulu
        if self.pos.side:
            if self._early_stop_check(d, now_ts):
                return
            # Time Stop v2
            try:
                if self.pos.entry_time is not None:
                    hold_sec = max(0.0, float(now_ts) - float(self.pos.entry_time.timestamp()))
                    ent = float(self.pos.entry or price)
                    roi = (price - ent)/ent if self.pos.side == 'LONG' else (ent - price)/ent
                    be_active = False
                    if self.pos.sl is not None:
                        if self.pos.side == 'LONG' and self.pos.sl >= ent: be_active = True
                        if self.pos.side == 'SHORT' and self.pos.sl <= ent: be_active = True
                    max_hold = self.max_hold_seconds_be if be_active else int(self.max_hold_seconds)
                    roi_thr = float(self.min_roi_time_close)
                    if hold_sec >= max_hold:
                        cond = (roi < max(roi_thr, 0.0)) if self.time_stop_only_if_loss else (roi < roi_thr)
                        if cond:
                            self._exit_position(price, f"Time Stop (hold>={max_hold}s & roi<{roi_thr*100:.2f}%)", now_ts=now_ts)
                            return
            except Exception:
                pass

            # BE + trailing + exit
            self._apply_breakeven(price)
            self._update_trailing(price, atr)
            ex, rs = self._should_exit(price)
            if ex:
                self._exit_position(price, rs or 'Exit', now_ts=now_ts)
                return

        # Filter gagal → log & kesempatan exit via SL/TSL kalau ada posisi
        if not (atr_ok and body_atr_ok):
            if self.pos.side:
                self._apply_breakeven(price)
                self._update_trailing(price, atr)
                ex, rs = self._should_exit(price)
                if ex:
                    self._exit_position(price, rs or 'Filter Fail', now_ts=now_ts)
            self._log(f"FILTER BLOCK atr_ok={atr_ok} body_ok={body_atr_ok} price={price:.6f} pos={self.pos.side}")
            return

        # Base sinyal
        base_long = (bool(last['macd'] > last['macd_sig']) and bool(last['rsi'] > 52) and bool(last['ema_fast'] > last['ema_slow']))
        base_short = (bool(last['macd'] < last['macd_sig']) and bool(last['rsi'] < 48) and bool(last['ema_fast'] < last['ema_slow']))

        # Trend filter EMA200 / EMA200+ADX
        if self.trend_filter_mode in ('ema200', 'ema200_adx'):
            if base_long and not (bool(last['ema_fast'] > last['ema_slow']) and bool(last['ema_slow'] > last['ema_200'])):
                base_long = False
            if base_short and not (bool(last['ema_fast'] < last['ema_slow']) and bool(last['ema_slow'] < last['ema_200'])):
                base_short = False
        if self.trend_filter_mode == 'ema200_adx':
            if float(last.get('adx', 0.0) or 0.0) < self.adx_thresh:
                base_long = False
                base_short = False

        # ML (opsional)
        up_prob = None
        try:
            self.ml.fit_if_needed(d)
            up_prob = self.ml.predict_up_prob(d)
        except Exception:
            up_prob = None
        long_sig, short_sig = self.ml.score_and_decide(base_long, base_short, up_prob)

        self._log(f"DECISION price={price:.6f} base(L={base_long},S={base_short}) adx={float(last.get('adx',0.0)):.2f} up_prob={'n/a' if up_prob is None else round(up_prob,3)} -> L={long_sig} S={short_sig} pos={self.pos.side}")

        if not self.pos.side and not self._cooldown_active(now_ts):
            if long_sig:
                self._enter_position('LONG', price, atr, balance, now_ts=now_ts)
            elif short_sig:
                self._enter_position('SHORT', price, atr, balance, now_ts=now_ts)

# =============================================================
# Manager
# =============================================================
class TradingManager:
    def __init__(self, coin_config_path: str, symbols: List[str]):
        self.symbols = symbols
        self.config_map = self._load_config(coin_config_path)
        self.traders: Dict[str, CoinTrader] = {sym: CoinTrader(sym, self.config_map.get(sym, {})) for sym in symbols}

    def _load_config(self, path: str) -> Dict[str, Dict[str, Any]]:
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception:
            return {}

    def run_once(self, data_map: Dict[str, pd.DataFrame], balances: Dict[str, float]) -> None:
        for sym, df in data_map.items():
            if sym not in self.traders:
                self.traders[sym] = CoinTrader(sym, self.config_map.get(sym, {}))
            bal = float(balances.get(sym, 0.0))
            self.traders[sym].check_trading_signals(df, bal)

# =============================================================
# CLI
# =============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--coin_config', default='coin_config.json')
    ap.add_argument('--symbol', required=False, default=None)
    ap.add_argument('--csv', required=False, default=None)
    ap.add_argument('--balance', type=float, default=20.0)
    ap.add_argument('--verbose', action='store_true')

    ap.add_argument('--dry-run-loop', action='store_true')
    ap.add_argument('--sleep', type=float, default=0.0)
    ap.add_argument('--limit', type=int, default=0)

    args = ap.parse_args()

    if args.verbose:
        os.environ['VERBOSE'] = '1'

    if args.csv and os.path.exists(args.csv):
        df = pd.read_csv(args.csv)
        if 'timestamp' not in df.columns:
            if 'open_time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', errors='coerce')
            elif 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df = df.sort_values('timestamp').reset_index(drop=True)
        sym = (args.symbol or '').upper() or 'SYMBOL'

        if args.dry_run_loop:
            min_train = int(float(os.getenv('ML_MIN_TRAIN_BARS', '400')))
            warmup = max(300, min_train + 10)
            start_i = min(warmup, len(df)-1)
            steps = 0
            mgr = TradingManager(args.coin_config, [sym])
            if args.verbose:
                mgr.traders[sym].verbose = True
            for i in range(start_i, len(df)):
                data_map = {sym: df.iloc[:i+1].copy()}
                mgr.run_once(data_map, {sym: args.balance})
                steps += 1
                if args.limit and steps >= args.limit:
                    break
                if args.sleep and args.sleep > 0:
                    time.sleep(args.sleep)
            last_side = mgr.traders[sym].pos.side if sym in mgr.traders else None
            print(f"Dry-run completed: {steps} steps (start={start_i}, total_bars={len(df)}). Last position: {last_side}")
        else:
            data_map = {sym: df}
            TradingManager(args.coin_config, [sym]).run_once(data_map, {sym: args.balance})
            print("Run once completed (dummy). Cek log/print sesuai hook eksekusi.")
    else:
        print("Tidak ada CSV yang diberikan. Gunakan --csv untuk simulasi dry-run.")

if __name__ == '__main__':
    main()
