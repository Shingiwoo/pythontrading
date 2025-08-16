#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
newrealtrading.py — Engine trading (live/dry-run) sinkron dengan backtester

PATCH 2025-08-13:
- FIX: Position sizing TANPA mengalikan leverage (leverage hanya mempengaruhi margin, bukan risk di SL).
- Time Stop v2: hanya tutup kalau rugi (ROI < MIN_ROI_TO_CLOSE_BY_TIME); hold lebih lama jika BE aktif.
- Trailing ATR dinamis + BE offset.
- Early Stop wick-aware (bisa dimatikan via config, default OFF pada rekomendasi).
- Trend filter opsional: "ema200" atau "ema200_adx".
- CLI: --dry-run-loop, --sleep, --limit, --verbose. Log SIZING untuk verifikasi risk.

Catatan:
- tools_dryrun_summary.py meng-hook _enter_position/_exit_position; signature dijaga:
  _enter_position(..., now_ts=None), _exit_position(..., now_ts=None)
"""

from __future__ import annotations

import os, json, time, argparse
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List, Protocol, runtime_checkable, cast

import numpy as np
import pandas as pd


# ============================ Utils ============================

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
        if s in {"1","true","y","yes","on"}:  return True
        if s in {"0","false","n","no","off"}: return False
    return bool(d)


# ============================ Defaults ============================

DEFAULTS: Dict[str, Any] = {
    "leverage": 10,                 # hanya untuk perhitungan margin (opsional)
    "risk_per_trade": 0.05,         # proporsi balance → risk USDT

    "taker_fee": 0.0005,

    "min_atr_pct": 0.006,           # filter volatilitas min (ATR/price)
    "max_atr_pct": 0.03,            # filter volatilitas max
    "max_body_atr": 1.0,            # filter candle body relatif ATR
    "use_htf_filter": 0,

    "cooldown_seconds": 1200,       # jeda setelah exit

    # Stop-loss
    "sl_mode": "ATR",
    "sl_pct": 0.008,                # dipakai kalau mode=FIXED
    "sl_atr_mult": 1.6,
    "sl_min_pct": 0.010,
    "sl_max_pct": 0.030,

    # Breakeven & trailing
    "use_breakeven": 1,
    "be_trigger_pct": 0.006,
    "be_offset_pct": 0.0008,

    "trailing_trigger": 1.0,        # % move dari entry agar trailing aktif
    "trailing_step_min_pct": 0.45,  # dasar % jarak trailing
    "trailing_step_max_pct": 1.00,  # maksimum % jarak trailing
    "trail_atr_k": 2.0,             # step ~ k * atr_pct_now

    # Cooldown multiplier berdasar alasan exit
    "cooldown_mul_hard_sl": 2.5,
    "cooldown_mul_trailing": 1.0,
    "cooldown_mul_early_stop": 1.2,

    # Early Stop (wick-aware)
    "early_stop_enabled": 0,        # DISABLE by default (nyalakan jika perlu)
    "early_stop_bars": 2,           # cek hanya n bar pertama
    "early_stop_adverse_atr": 0.90, # proporsional ATR (agak longgar)
    "early_stop_wick_factor": 1.30, # bila pakai wick → ambang dinaikkan
    "adx_strong_thresh": 22.0,

    # Time Stop v2
    "max_hold_seconds": 3600,
    "min_roi_to_close_by_time": 0.0,
    "time_stop_only_if_loss": 1,
    "max_hold_seconds_be": 7200,    # kalau BE aktif

    # Trend filter
    "trend_filter_mode": "ema200",  # "", "ema200", "ema200_adx"
    "adx_period": 14,
    "adx_thresh": 18.0,
}


# ============================ Indikator ============================

def _ema(s: pd.Series, window: int) -> pd.Series:
    return pd.Series(pd.to_numeric(s, errors="coerce")).ewm(span=window, adjust=False).mean()

def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    c = pd.Series(pd.to_numeric(close, errors="coerce"))
    delta = c.diff()
    up = np.where(delta.gt(0), delta, 0.0)
    down = np.where(delta.lt(0), -delta, 0.0)
    roll_up = pd.Series(up).ewm(alpha=1/window, adjust=False).mean()
    roll_down = pd.Series(down).ewm(alpha=1/window, adjust=False).mean()
    rs = roll_up / roll_down.replace(0.0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return pd.Series(rsi).fillna(50.0)

def _macd(close: pd.Series, fast=12, slow=26, sign=9) -> Tuple[pd.Series, pd.Series]:
    c = pd.Series(pd.to_numeric(close, errors="coerce"))
    macd = _ema(c, fast) - _ema(c, slow)
    signal = macd.ewm(span=sign, adjust=False).mean()
    return macd, signal

def _atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    h = pd.to_numeric(df["high"], errors="coerce")
    l = pd.to_numeric(df["low"], errors="coerce")
    c = pd.to_numeric(df["close"], errors="coerce")
    prev_close = c.shift(1)
    tr = pd.concat([
        (h - l).abs(),
        (h - prev_close).abs(),
        (l - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    h = pd.to_numeric(df["high"], errors="coerce")
    l = pd.to_numeric(df["low"], errors="coerce")
    c = pd.to_numeric(df["close"], errors="coerce")

    up_move = h.diff()
    down_move = -l.diff()

    plus_dm = np.where((up_move > down_move) & (up_move.gt(0)), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move.gt(0)), down_move, 0.0)

    tr1 = (h - l)
    tr2 = (h - c.shift(1)).abs()
    tr3 = (l - c.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di  = 100 * (pd.Series(plus_dm).rolling(period).mean() / atr.replace(0.0, np.nan))
    minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean() / atr.replace(0.0, np.nan))
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).abs().replace(0.0, np.nan)) * 100
    adx = dx.rolling(period).mean()
    return adx.fillna(0.0)

def calculate_indicators(df: pd.DataFrame, adx_period: int = 14) -> pd.DataFrame:
    d = df.copy()
    if "timestamp" not in d.columns:
        if "open_time" in d.columns:
            d["timestamp"] = pd.to_datetime(d["open_time"], unit="ms", errors="coerce")
        elif "date" in d.columns:
            d["timestamp"] = pd.to_datetime(d["date"], errors="coerce")
    d["timestamp"] = pd.to_datetime(d["timestamp"], utc=True)
    d = d.sort_values("timestamp").reset_index(drop=True)

    d["ema_fast"] = _ema(d["close"], 20)
    d["ema_slow"] = _ema(d["close"], 50)
    d["ema_200"]  = _ema(d["close"], 200)
    macd, macd_sig = _macd(d["close"])
    d["macd"] = macd
    d["macd_sig"] = macd_sig
    d["rsi"] = _rsi(d["close"], 14)
    d["atr"] = _atr(d, 14)
    d["adx"] = _adx(d, adx_period)

    d["body"] = (pd.to_numeric(d["close"]) - pd.to_numeric(d["open"])).abs()
    d["body_atr"] = d["body"] / d["atr"].replace(0.0, np.nan)
    d["body_atr"] = d["body_atr"].fillna(np.inf)

    return d


# ============================ ML Plugin ============================

@runtime_checkable
class MLSignalProto(Protocol):
    @property
    def use_ml(self) -> bool: ...
    def fit_if_needed(self, df: pd.DataFrame) -> None: ...
    def predict_up_prob(self, df: pd.DataFrame) -> Optional[float]: ...
    def score_and_decide(
        self, base_long: bool, base_short: bool, up_prob: Optional[float] = None
    ) -> Tuple[bool, bool]: ...

try:
    from ml_signal_plugin import MLSignal as MLImplClass  # type: ignore
except Exception:
    MLImplClass = None  # type: ignore

class _MLStub:
    def __init__(self, coin_cfg: Dict[str, Any] | None):
        self.params = type("P", (), dict(use_ml=False, score_threshold=1.0))
    @property
    def use_ml(self) -> bool: return False
    def fit_if_needed(self, df: pd.DataFrame) -> None: return
    def predict_up_prob(self, df: pd.DataFrame) -> Optional[float]: return None
    def score_and_decide(self, base_long: bool, base_short: bool, up_prob: Optional[float] = None):
        th = float(getattr(self.params, "score_threshold", 1.0))
        sc_l = 1.0 if base_long else 0.0
        sc_s = 1.0 if base_short else 0.0
        return (sc_l >= th, sc_s >= th)


# ============================ Position ============================

@dataclass
class Position:
    side: Optional[str] = None
    entry: Optional[float] = None
    qty: Optional[float] = None
    sl: Optional[float] = None
    trailing_sl: Optional[float] = None
    entry_time: Optional[pd.Timestamp] = None


# ============================ Trader ============================

class CoinTrader:
    def __init__(self, symbol: str, coin_cfg: Dict[str, Any]):
        self.symbol = symbol
        self.config = coin_cfg or {}
        self.pos = Position()
        self.cooldown_until_ts: Optional[float] = None

        self.verbose = _to_bool(self.config.get("VERBOSE", os.getenv("VERBOSE","0")), False)
        self.cooldown_use_bar_time = _to_bool(self.config.get("COOLDOWN_USE_BAR_TIME", os.getenv("COOLDOWN_USE_BAR_TIME","1")), True)

        # Early stop
        self.early_stop_enabled = _to_bool(self.config.get("early_stop_enabled", DEFAULTS["early_stop_enabled"]), DEFAULTS["early_stop_enabled"])
        self.early_stop_bars = _to_int(self.config.get("early_stop_bars", DEFAULTS["early_stop_bars"]), DEFAULTS["early_stop_bars"])
        self.early_stop_adverse_atr = _to_float(self.config.get("early_stop_adverse_atr", DEFAULTS["early_stop_adverse_atr"]), DEFAULTS["early_stop_adverse_atr"])
        self.early_stop_wick_factor = _to_float(self.config.get("early_stop_wick_factor", DEFAULTS["early_stop_wick_factor"]), DEFAULTS["early_stop_wick_factor"])
        self.adx_strong_thresh = _to_float(self.config.get("adx_strong_thresh", DEFAULTS["adx_strong_thresh"]), DEFAULTS["adx_strong_thresh"])

        # Time stop v2
        self.max_hold_seconds = _to_int(self.config.get("max_hold_seconds", os.getenv("MAX_HOLD_SECONDS", DEFAULTS["max_hold_seconds"])), DEFAULTS["max_hold_seconds"])
        self.max_hold_seconds_be = _to_int(self.config.get("max_hold_seconds_be", DEFAULTS["max_hold_seconds_be"]), DEFAULTS["max_hold_seconds_be"])
        self.min_roi_time_close = _to_float(self.config.get("min_roi_to_close_by_time", os.getenv("MIN_ROI_TO_CLOSE_BY_TIME", DEFAULTS["min_roi_to_close_by_time"])), DEFAULTS["min_roi_to_close_by_time"])
        self.time_stop_only_if_loss = _to_bool(self.config.get("time_stop_only_if_loss", DEFAULTS["time_stop_only_if_loss"]), DEFAULTS["time_stop_only_if_loss"])

        # Trend filter
        self.trend_filter_mode = str(self.config.get("trend_filter_mode", DEFAULTS["trend_filter_mode"])).lower().strip()
        self.adx_period = _to_int(self.config.get("adx_period", DEFAULTS["adx_period"]), DEFAULTS["adx_period"])
        self.adx_thresh = _to_float(self.config.get("adx_thresh", DEFAULTS["adx_thresh"]), DEFAULTS["adx_thresh"])

        # ML
        if MLImplClass is not None:
            self.ml: MLSignalProto = cast(MLSignalProto, MLImplClass(self.config))
        else:
            self.ml: MLSignalProto = _MLStub(self.config)

    # ---------------------------- logging ----------------------------
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

    # ---------------------- sizing & stop level ----------------------
    def _size_position(self, price: float, balance: float, atr: float) -> float:
        # Risk in USDT (tidak pakai leverage multiplier)
        risk_pct   = _to_float(self.config.get("risk_per_trade", DEFAULTS["risk_per_trade"]), DEFAULTS["risk_per_trade"])
        risk_usdt  = max(0.0001, balance * risk_pct)

        # Stop distance (dalam %) berbasis ATR lalu dijepit min/max
        sl_atr_mult = _to_float(self.config.get("sl_atr_mult", DEFAULTS["sl_atr_mult"]), DEFAULTS["sl_atr_mult"])
        sl_min_pct  = _to_float(self.config.get("sl_min_pct",  DEFAULTS["sl_min_pct"]),  DEFAULTS["sl_min_pct"])
        sl_max_pct  = _to_float(self.config.get("sl_max_pct",  DEFAULTS["sl_max_pct"]),  DEFAULTS["sl_max_pct"])
        sl_dist_pct = max(sl_min_pct, sl_atr_mult * (atr / max(price, 1e-9)))
        sl_dist_pct = min(sl_dist_pct, sl_max_pct)

        # Notional TANPA leverage: loss di SL ≈ risk_usdt
        # loss ≈ notional * sl_dist_pct  => notional = risk_usdt / sl_dist_pct
        notional = risk_usdt / max(sl_dist_pct, 1e-6)

        # Opsional: batasi oleh kapasitas margin (balance * leverage)
        lev = _to_float(self.config.get("leverage", DEFAULTS["leverage"]), DEFAULTS["leverage"])
        max_margin = balance * 0.95  # jangan gunakan 100% saldo
        max_notional_by_margin = max_margin * max(lev, 1.0)
        if notional > max_notional_by_margin:
            notional = max_notional_by_margin

        qty = max(0.0, notional / max(price, 1e-9))
        # pembulatan 3 desimal (sesuaikan precision exchange)
        return round(qty, 3)

    def _hard_sl_price(self, price: float, atr: float, side: str) -> float:
        mode = str(self.config.get("sl_mode", DEFAULTS["sl_mode"])).upper()
        if mode == "ATR":
            mult   = _to_float(self.config.get("sl_atr_mult", DEFAULTS["sl_atr_mult"]), DEFAULTS["sl_atr_mult"])
            minpct = _to_float(self.config.get("sl_min_pct",  DEFAULTS["sl_min_pct"]),  DEFAULTS["sl_min_pct"])
            maxpct = _to_float(self.config.get("sl_max_pct",  DEFAULTS["sl_max_pct"]),  DEFAULTS["sl_max_pct"])
            sl_pct = min(max(minpct, mult * (atr/max(price, 1e-9))), maxpct)
        else:
            sl_pct = _to_float(self.config.get("sl_pct", DEFAULTS["sl_pct"]), DEFAULTS["sl_pct"])
        return price * (1.0 - sl_pct) if side == "LONG" else price * (1.0 + sl_pct)

    # ------------------------ BE & Trailing -------------------------
    def _apply_breakeven(self, price: float) -> None:
        if not (self.pos and self.pos.side and self.pos.entry):
            return
        if not _to_bool(self.config.get("use_breakeven", DEFAULTS["use_breakeven"]), DEFAULTS["use_breakeven"]):
            return
        be_trg = _to_float(self.config.get("be_trigger_pct", DEFAULTS["be_trigger_pct"]), DEFAULTS["be_trigger_pct"])
        be_off = _to_float(self.config.get("be_offset_pct",  DEFAULTS["be_offset_pct"]),  DEFAULTS["be_offset_pct"])
        ent = float(self.pos.entry)
        if self.pos.side == "LONG":
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
        trg = _to_float(self.config.get("trailing_trigger", DEFAULTS["trailing_trigger"]), DEFAULTS["trailing_trigger"])
        ent = float(self.pos.entry)
        moved_long  = (price - ent)/ent * 100.0
        moved_short = (ent - price)/ent * 100.0
        if self.pos.side == "LONG":
            if moved_long < trg:  return
        else:
            if moved_short < trg: return

        atr_pct_now = (atr / max(price, 1e-9)) * 100.0
        step_min = _to_float(self.config.get("trailing_step_min_pct", DEFAULTS["trailing_step_min_pct"]), DEFAULTS["trailing_step_min_pct"])
        step_max = _to_float(self.config.get("trailing_step_max_pct", DEFAULTS["trailing_step_max_pct"]), DEFAULTS["trailing_step_max_pct"])
        k = _to_float(self.config.get("trail_atr_k", DEFAULTS["trail_atr_k"]), DEFAULTS["trail_atr_k"])
        dyn_step = max(step_min, min(step_max, k * atr_pct_now))

        if self.pos.side == "LONG":
            new_tsl = price * (1.0 - dyn_step/100.0)
            self.pos.trailing_sl = max(self.pos.trailing_sl or 0.0, new_tsl)
        else:
            new_tsl = price * (1.0 + dyn_step/100.0)
            self.pos.trailing_sl = min(self.pos.trailing_sl or 1e12, new_tsl)

    # --------------------------- Exit check -------------------------
    def _should_exit(self, price: float) -> Tuple[bool, Optional[str]]:
        if not self.pos.side:
            return False, None
        if self.pos.side == "LONG":
            if self.pos.sl and price <= self.pos.sl:         return True, "Hit Hard SL"
            if self.pos.trailing_sl and price <= self.pos.trailing_sl: return True, "Hit Trailing SL"
        else:
            if self.pos.sl and price >= self.pos.sl:         return True, "Hit Hard SL"
            if self.pos.trailing_sl and price >= self.pos.trailing_sl: return True, "Hit Trailing SL"
        return False, None

    # --------------------- Early Stop (wick-aware) ------------------
    def _early_stop_check(self, df: pd.DataFrame, now_ts: Optional[float]) -> bool:
        if not self.early_stop_enabled or not (self.pos and self.pos.side and self.pos.entry and self.pos.entry_time is not None):
            return False

        last = df.iloc[-1]
        atr = float(last.get("atr", 0.0) or 0.0)
        if atr <= 0: return False

        try:
            bars_since = int((df["timestamp"] > self.pos.entry_time).sum())
        except Exception:
            bars_since = 0

        max_es_bars = int(self.early_stop_bars)
        if bars_since <= 0 or bars_since > max_es_bars:
            return False

        ent   = float(self.pos.entry)
        low   = float(last.get("low",   last.get("close", ent)))
        high  = float(last.get("high",  last.get("close", ent)))
        close = float(last.get("close", ent))

        adverse_prop = float(self.early_stop_adverse_atr) * (atr / max(ent, 1e-9))
        adx_last = float(last.get("adx", 0.0) or 0.0)
        use_close_only = (adx_last >= self.adx_strong_thresh) or (bars_since >= 2)

        if self.pos.side == "LONG":
            if use_close_only:
                dd = (ent - close) / max(ent, 1e-9)
                if dd >= adverse_prop:
                    self._exit_position(close, f"Early Stop close (dd={dd*100:.2f}% >= {adverse_prop*100:.2f}%)", now_ts=now_ts)
                    return True
            else:
                dd_wick = (ent - low) / max(ent, 1e-9)
                if dd_wick >= adverse_prop * float(self.early_stop_wick_factor):
                    self._exit_position(low, f"Early Stop wick (dd={dd_wick*100:.2f}% >= {(adverse_prop*float(self.early_stop_wick_factor))*100:.2f}%)", now_ts=now_ts)
                    return True
        else:
            if use_close_only:
                dd = (close - ent) / max(ent, 1e-9)
                if dd >= adverse_prop:
                    self._exit_position(close, f"Early Stop close (dd={dd*100:.2f}% >= {adverse_prop*100:.2f}%)", now_ts=now_ts)
                    return True
            else:
                dd_wick = (high - ent) / max(ent, 1e-9)
                if dd_wick >= adverse_prop * float(self.early_stop_wick_factor):
                    self._exit_position(high, f"Early Stop wick (dd={dd_wick*100:.2f}% >= {(adverse_prop*float(self.early_stop_wick_factor))*100:.2f}%)", now_ts=now_ts)
                    return True

        return False

    # ---------------------- Entry / Exit core -----------------------
    def _enter_position(self, side: str, price: float, atr: float, balance: float, now_ts: Optional[float] = None) -> None:
        if self._cooldown_active(now_ts):
            return
        qty = self._size_position(price, balance, atr)
        if qty <= 0: return
        sl = self._hard_sl_price(price, atr, side)
        et = pd.to_datetime(now_ts, unit="s", utc=True) if now_ts is not None else pd.Timestamp.utcnow()
        self.pos = Position(side=side, entry=price, qty=qty, sl=sl, trailing_sl=None, entry_time=et)

        # Log sizing sanity
        sl_atr_mult = _to_float(self.config.get("sl_atr_mult", DEFAULTS["sl_atr_mult"]), DEFAULTS["sl_atr_mult"])
        sl_min_pct  = _to_float(self.config.get("sl_min_pct",  DEFAULTS["sl_min_pct"]),  DEFAULTS["sl_min_pct"])
        sl_max_pct  = _to_float(self.config.get("sl_max_pct",  DEFAULTS["sl_max_pct"]),  DEFAULTS["sl_max_pct"])
        sl_dist_pct = max(sl_min_pct, sl_atr_mult * (atr / max(price, 1e-9)))
        sl_dist_pct = min(sl_dist_pct, sl_max_pct)

        risk_pct  = _to_float(self.config.get("risk_per_trade", DEFAULTS["risk_per_trade"]), DEFAULTS["risk_per_trade"])
        risk_usdt = max(0.0001, balance * risk_pct)
        theo_loss_at_sl = (qty * price) * sl_dist_pct  # kira-kira loss di SL

        self._log(
            f"SIZING price={price:.6f} qty={qty:.3f} sl={sl:.6f} sl_dist%={sl_dist_pct*100:.2f} "
            f"risk_usdt={risk_usdt:.4f} theo_loss_at_sl={theo_loss_at_sl:.4f}"
        )
        self._log(f"ENTRY {side} price={price:.6f} qty={qty:.6f} sl={sl:.6f}")

    def _exit_position(self, price: float, reason: str, now_ts: Optional[float] = None) -> None:
        self._log(f"EXIT {reason} price={price:.6f}")
        cd_base = _to_int(self.config.get("cooldown_seconds", DEFAULTS["cooldown_seconds"]), DEFAULTS["cooldown_seconds"])
        r = (reason or "").lower()
        mul = 1.0
        if "hard sl" in r:
            mul = _to_float(self.config.get("cooldown_mul_hard_sl", DEFAULTS["cooldown_mul_hard_sl"]), DEFAULTS["cooldown_mul_hard_sl"])
        elif "trailing" in r:
            mul = _to_float(self.config.get("cooldown_mul_trailing", DEFAULTS["cooldown_mul_trailing"]), DEFAULTS["cooldown_mul_trailing"])
        elif "early stop" in r:
            mul = _to_float(self.config.get("cooldown_mul_early_stop", DEFAULTS["cooldown_mul_early_stop"]), DEFAULTS["cooldown_mul_early_stop"])
        base_now = float(now_ts) if now_ts is not None else time.time()
        self.cooldown_until_ts = base_now + max(0, int(cd_base * mul))
        self.pos = Position()

    # ---------------------- Sinyal & Keputusan ----------------------
    def check_trading_signals(self, df: pd.DataFrame, balance: float) -> None:
        if df is None or len(df) < 50:
            return

        d = calculate_indicators(df, adx_period=self.adx_period)
        last = d.iloc[-1]
        price = float(last["close"])
        atr   = float(last.get("atr", 0.0) or 0.0)
        last_ts = last["timestamp"] if "timestamp" in last else None
        now_ts = self._now_ts(last_ts)

        # Filter volatilitas & body
        atr_pct = (atr / max(price, 1e-9))
        min_atr_pct = _to_float(self.config.get("min_atr_pct", DEFAULTS["min_atr_pct"]), DEFAULTS["min_atr_pct"])
        max_atr_pct = _to_float(self.config.get("max_atr_pct", DEFAULTS["max_atr_pct"]), DEFAULTS["max_atr_pct"])
        body_atr_ok = bool(last.get("body_atr", np.nan) <= _to_float(self.config.get("max_body_atr", DEFAULTS["max_body_atr"]), DEFAULTS["max_body_atr"]))
        atr_ok = (atr_pct >= min_atr_pct) and (atr_pct <= max_atr_pct)

        # Kelola posisi aktif dulu
        if self.pos.side:
            # Early stop (jika ON)
            if self._early_stop_check(d, now_ts):
                return

            # Time Stop v2
            try:
                if self.pos.entry_time is not None:
                    hold_sec = max(0.0, float(now_ts) - float(self.pos.entry_time.timestamp()))
                    ent = float(self.pos.entry or price)
                    roi = (price - ent)/ent if self.pos.side == "LONG" else (ent - price)/ent
                    be_active = False
                    if self.pos.sl is not None:
                        if self.pos.side == "LONG"  and self.pos.sl >= ent: be_active = True
                        if self.pos.side == "SHORT" and self.pos.sl <= ent: be_active = True
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
                self._exit_position(price, rs or "Exit", now_ts=now_ts)
                return

        # Filter gagal → log & kesempatan SL/TSL
        if not (atr_ok and body_atr_ok):
            if self.pos.side:
                self._apply_breakeven(price)
                self._update_trailing(price, atr)
                ex, rs = self._should_exit(price)
                if ex:
                    self._exit_position(price, rs or "Filter Fail", now_ts=now_ts)
            self._log(f"FILTER BLOCK atr_ok={atr_ok} body_ok={body_atr_ok} price={price:.6f} pos={self.pos.side}")
            return

        # Base sinyal
        base_long  = (bool(last["macd"] > last["macd_sig"]) and bool(last["rsi"] > 52) and bool(last["ema_fast"] > last["ema_slow"]))
        base_short = (bool(last["macd"] < last["macd_sig"]) and bool(last["rsi"] < 48) and bool(last["ema_fast"] < last["ema_slow"]))

        # Trend filter
        if self.trend_filter_mode in ("ema200", "ema200_adx"):
            if base_long and not (bool(last["ema_fast"] > last["ema_slow"]) and bool(last["ema_slow"] > last["ema_200"])):
                base_long = False
            if base_short and not (bool(last["ema_fast"] < last["ema_slow"]) and bool(last["ema_slow"] < last["ema_200"])):
                base_short = False
        if self.trend_filter_mode == "ema200_adx":
            if float(last.get("adx", 0.0) or 0.0) < self.adx_thresh:
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

        self._log(
            f"DECISION price={price:.6f} base(L={base_long},S={base_short}) "
            f"adx={float(last.get('adx',0.0)):.2f} up_prob={'n/a' if up_prob is None else round(up_prob,3)} "
            f"-> L={long_sig} S={short_sig} pos={self.pos.side}"
        )

        # Entry
        if not self.pos.side and not self._cooldown_active(now_ts):
            if long_sig:
                self._enter_position("LONG",  price, atr, balance, now_ts=now_ts)
            elif short_sig:
                self._enter_position("SHORT", price, atr, balance, now_ts=now_ts)


# ============================ Manager ============================

class TradingManager:
    def __init__(self, coin_config_path: str, symbols: List[str]):
        self.symbols = symbols
        self.config_map = self._load_config(coin_config_path)
        self.traders: Dict[str, CoinTrader] = {sym: CoinTrader(sym, self.config_map.get(sym, {})) for sym in symbols}

    def _load_config(self, path: str) -> Dict[str, Dict[str, Any]]:
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def run_once(self, data_map: Dict[str, pd.DataFrame], balances: Dict[str, float]) -> None:
        for sym, df in data_map.items():
            if sym not in self.traders:
                self.traders[sym] = CoinTrader(sym, self.config_map.get(sym, {}))
            bal = float(balances.get(sym, 0.0))
            self.traders[sym].check_trading_signals(df, bal)


# ============================ CLI ============================

def _load_csv_sorted(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        if "open_time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", errors="coerce")
        elif "date" in df.columns:
            df["timestamp"] = pd.to_datetime(df["date"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin_config", default="coin_config.json")
    ap.add_argument("--symbol", required=False, default=None)
    ap.add_argument("--csv", required=False, default=None)
    ap.add_argument("--balance", type=float, default=20.0)
    ap.add_argument("--verbose", action="store_true")

    ap.add_argument("--dry-run-loop", action="store_true")
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)

    args = ap.parse_args()

    if args.verbose:
        os.environ["VERBOSE"] = "1"

    if args.csv and os.path.exists(args.csv):
        df = _load_csv_sorted(args.csv)
        sym = (args.symbol or "").upper() or "SYMBOL"

        if args.dry_run_loop:
            min_train = int(float(os.getenv("ML_MIN_TRAIN_BARS", "400")))
            warmup = max(300, min_train + 10)
            start_i = min(warmup, len(df)-1)
            steps = 0

            mgr = TradingManager(args.coin_config, [sym])
            if args.verbose and sym in mgr.traders:
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


if __name__ == "__main__":
    main()
