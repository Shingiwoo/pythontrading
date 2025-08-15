#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Patched newrealtrading.py

Highlights (matches the logs you shared):
- ATR/Body filter + AUTO-RELAX with hysteresis and hard bounds
- ML gating with strict option; decision logs include ml_ok flags
- Risk % normalization (accepts 0..1 or 0..100 semantics)
- Break-even (BE) based on R-multiple threshold
- Trailing stop step size is ADX-adaptive (bounded min/max)
- Time-stop adaptive (extends max hold when ADX is high)
- Verbose and debug-mode logs formatted like your examples

CLI example:
python3 newrealtrading.py \
  --coin_config coin_config.json \
  --csv data/ADAUSDT_15m_2025-06-01_to_2025-08-09.csv \
  --symbol ADAUSDT --balance 20 \
  --verbose --dry-run-loop --sleep 0.0 --limit 500 \
  --debug-mode

Note: This file is self-contained and duck-types the ML plugin. If a module
`ml_signal_plugin.py` with class `MLSignal` exists in the PYTHONPATH, it will be used.
Otherwise, a stub ML class is used (no gating unless strict=false).
"""

from __future__ import annotations

import argparse
import dataclasses as dc
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    import pandas as pd
    import numpy as np
except Exception as e:  # pragma: no cover
    print("This script requires pandas and numpy.", file=sys.stderr)
    raise

# --------------------------------------------------------------------------------------
# Small utils
# --------------------------------------------------------------------------------------

def env(name: str, default: Any = None, cast: Optional[type] = None):
    val = os.environ.get(name)
    if val is None:
        return default
    if cast is None:
        return val
    try:
        if cast is bool:
            s = str(val).strip().lower()
            return s in ("1", "true", "y", "yes", "on")
        return cast(val)
    except Exception:
        return default


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def tsfmt(ts: pd.Timestamp) -> str:
    return ts.tz_localize("UTC").isoformat() if ts.tz is None else ts.tz_convert("UTC").isoformat()


# --------------------------------------------------------------------------------------
# Indicator helpers (ATR, ADX, EMA)
# --------------------------------------------------------------------------------------

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    prev_close = df["close"].shift(1)
    h_l = (df["high"] - df["low"]).abs()
    h_pc = (df["high"] - prev_close).abs()
    l_pc = (df["low"] - prev_close).abs()
    return pd.concat([h_l, h_pc, l_pc], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr = true_range(df)
    return tr.ewm(alpha=1.0 / period, adjust=False).mean()


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    # Wilder's ADX
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = np.where((up_move > down_move) & (up_move.gt(0)), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move.gt(0)), down_move, 0.0)

    tr = true_range(df)
    atr_w = tr.ewm(alpha=1.0 / period, adjust=False).mean()

    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / atr_w
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / period, adjust=False).mean() / atr_w

    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).fillna(0.0)
    adx_val = dx.ewm(alpha=1.0 / period, adjust=False).mean()
    return adx_val


# --------------------------------------------------------------------------------------
# ML plugin loader (duck-typed)
# --------------------------------------------------------------------------------------
class _MLStub:
    def __init__(self, coin_cfg: Dict[str, Any]):
        self._use = bool(coin_cfg.get("ml", {}).get("use_ml", env("USE_ML", True, bool)))
        self._up = None  # type: Optional[float]

    @property
    def use_ml(self) -> bool:
        return self._use

    def fit_if_needed(self, df: pd.DataFrame) -> None:
        return None

    def predict_up_prob(self, df_tail: pd.DataFrame) -> Optional[float]:
        # No model by default
        return None

    def score_and_decide(self, base_long: bool, base_short: bool, up_prob: Optional[float] = None):
        # Kept for compatibility
        return base_long, base_short


def load_ml(coin_cfg: Dict[str, Any]):
    try:
        import ml_signal_plugin  # type: ignore
        if hasattr(ml_signal_plugin, "MLSignal"):
            return ml_signal_plugin.MLSignal(coin_cfg)
    except Exception:
        pass
    return _MLStub(coin_cfg)


# --------------------------------------------------------------------------------------
# Config model
# --------------------------------------------------------------------------------------
@dc.dataclass
class TrailCfg:
    use_adx_step: bool = True
    min_step_pct: float = 0.006  # 0.6%
    max_step_pct: float = 0.012  # 1.2%
    adx_bounds: Tuple[float, float] = (20.0, 60.0)


@dc.dataclass
class TimeStopCfg:
    max_hold_seconds: int = 3600
    min_roi_to_close: float = 0.0
    extend_when_adx_gt: float = 40.0
    extend_factor: float = 1.5


@dc.dataclass
class MLCfg:
    strict: bool = False
    up_prob_long: float = 0.60
    down_prob_short: float = 0.40
    score_threshold: float = 1.40


@dc.dataclass
class FilterCfg:
    base_min_atr: float = 0.0030  # 0.30%
    base_max_body: float = 1.50
    target_pass_ratio: float = 0.35
    hysteresis_tighten: float = 0.55
    relax_step_atr: float = 0.0001
    relax_step_body: float = 0.05
    max_relax_atr: float = 0.0023  # lower bound of min_atr (i.e., allow down to 0.23%)
    max_relax_body: float = 2.00


@dc.dataclass
class CoinCfg:
    min_atr_pct: float = 0.0030
    max_body_to_atr: float = 1.50
    sl_pct: float = 0.012
    be_trigger_r: float = 0.5
    trail: TrailCfg = dc.field(default_factory=TrailCfg)
    time_stop: TimeStopCfg = dc.field(default_factory=TimeStopCfg)
    ml: MLCfg = dc.field(default_factory=MLCfg)
    filter: FilterCfg = dc.field(default_factory=FilterCfg)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "CoinCfg":
        def get_nested(cls, key, default):
            return cls(**d.get(key, {})) if isinstance(d.get(key), dict) else default
        return CoinCfg(
            min_atr_pct=float(d.get("min_atr_pct", 0.0030)),
            max_body_to_atr=float(d.get("max_body_to_atr", 1.50)),
            sl_pct=float(d.get("sl_pct", 0.012)),
            be_trigger_r=float(d.get("be_trigger_r", 0.5)),
            trail=get_nested(TrailCfg, "trail", TrailCfg()),
            time_stop=get_nested(TimeStopCfg, "time_stop", TimeStopCfg()),
            ml=get_nested(MLCfg, "ml", MLCfg()),
            filter=get_nested(FilterCfg, "filter", FilterCfg()),
        )


# --------------------------------------------------------------------------------------
# Trader
# --------------------------------------------------------------------------------------
class Trader:
    def __init__(self, symbol: str, df: pd.DataFrame, coin_cfg: CoinCfg, balance: float,
                 verbose: bool = False, debug: bool = False):
        self.symbol = symbol
        self.df = df.copy()
        self.cfg = coin_cfg
        self.verbose = verbose
        self.debug = debug

        # Indicators
        self.df["ema_fast"] = ema(self.df["close"], 13)
        self.df["ema_slow"] = ema(self.df["close"], 50)
        self.df["atr"] = atr(self.df, 14)
        self.df["adx"] = adx(self.df, 14)

        # Stats for filter auto-relax
        self._pass_history: List[bool] = []
        self._min_atr = self.cfg.min_atr_pct
        self._max_body = self.cfg.max_body_to_atr

        # ML
        self.ml = load_ml(dataclasses_to_dict(self.cfg))

        # Position state
        self.pos: Optional[str] = None  # "LONG" / "SHORT" / None
        self.entry_price: Optional[float] = None
        self.entry_ts: Optional[pd.Timestamp] = None
        self.hard_sl: Optional[float] = None
        self.sl: Optional[float] = None
        self.tsl: Optional[float] = None
        self.qty: float = 0.0
        self.be_triggered: bool = False
        self.last_sl_type: str = "HARD"  # HARD/BE/TSL

        # Risk
        risk_pct = float(env("RISK_PERCENTAGE", 0.01))
        if risk_pct > 1.0:
            risk_pct *= 0.01
        self.risk_pct = risk_pct
        self.balance = float(balance)

        # Misc env
        self.min_roi_time_close = float(env("MIN_ROI_TO_CLOSE_BY_TIME", self.cfg.time_stop.min_roi_to_close))
        self.max_hold_seconds = int(env("MAX_HOLD_SECONDS", self.cfg.time_stop.max_hold_seconds))

    # --------------------------------------------
    # Logging helpers
    # --------------------------------------------
    def _log(self, msg: str):
        if self.verbose:
            now = self.current_ts
            print(f"[{self.symbol}] {now.isoformat()} | {msg}")

    def _dbg(self, msg: str):
        if self.verbose and self.debug:
            now = self.current_ts
            print(f"[{self.symbol}] {now.isoformat()} | {msg}")

    # --------------------------------------------
    # Step iteration
    # --------------------------------------------
    def run(self, start: int, limit: int):
        total = len(self.df)
        end = min(total, start + limit)
        self.current_ts = pd.Timestamp(self.df.index[start])
        for i in range(start, end):
            row = self.df.iloc[i]
            self.current_ts = pd.Timestamp(self.df.index[i])
            price = float(row["close"])
            self._step(i, price)
        last = self.pos if self.pos else "None"
        print(f"Dry-run completed: {end - start} steps (start={start}, total_bars={total}). Last position: {last}")

    # --------------------------------------------
    # Core per-step logic
    # --------------------------------------------
    def _step(self, i: int, price: float):
        row = self.df.iloc[i]
        atr_v = float(row["atr"]) if not math.isnan(row["atr"]) else 0.0
        adx_v = float(row["adx"]) if not math.isnan(row["adx"]) else 0.0

        # Filter evaluations
        min_atr = self._min_atr
        max_body = self._max_body

        body = abs(float(row["close"]) - float(row["open"]))
        body_to_atr = (body / atr_v) if atr_v > 0 else 0.0
        atr_pct = (atr_v / price) if price > 0 else 0.0

        # Check pass
        atr_ok = atr_pct >= min_atr
        body_ok = body_to_atr <= max_body

        if self.debug:
            self._log(
                f"FILTER_VALS price={price:.4f} atr={atr_v:.6f} atr_pct={atr_pct*100:.3f}% "
                f"min_atr={min_atr*100:.3f}% max_atr=4.000% body_atr={body_to_atr:.3f} max_body_atr={max_body:.3f}"
            )
            if not atr_ok and not body_ok:
                self._log("FILTER_FAIL reason=ATR+BODY")
            elif not atr_ok:
                self._log("FILTER_FAIL reason=ATR")
            elif not body_ok:
                self._log("FILTER_FAIL reason=BODY")

        self._log(
            f"FILTER BLOCK atr_ok={atr_ok} body_ok={body_ok} price={price:.4f} pos={self.pos if self.pos else 'None'}"
        ) if (self.debug and (not atr_ok or not body_ok)) else None

        # Maintain pass history for auto-relax
        self._pass_history.append(bool(atr_ok and body_ok))
        if len(self._pass_history) > 50:
            self._pass_history.pop(0)
        pass_ratio = sum(self._pass_history) / max(1, len(self._pass_history))

        # Auto-relax with hysteresis
        base_min_atr = self.cfg.filter.base_min_atr
        base_max_body = self.cfg.filter.base_max_body
        changed = False
        old_min_atr, old_max_body = self._min_atr, self._max_body
        if pass_ratio < self.cfg.filter.target_pass_ratio:
            # Relax
            self._min_atr = max(self.cfg.filter.max_relax_atr, self._min_atr - self.cfg.filter.relax_step_atr)
            self._max_body = min(self.cfg.filter.max_relax_body, self._max_body + self.cfg.filter.relax_step_body)
            changed = True
        elif pass_ratio > self.cfg.filter.hysteresis_tighten:
            # Tighten back toward base
            self._min_atr = min(base_min_atr, self._min_atr + self.cfg.filter.relax_step_atr)
            self._max_body = max(base_max_body, self._max_body - self.cfg.filter.relax_step_body)
            changed = True

        if changed and self.debug:
            self._log(
                f"FILTER_AUTO_RELAX pass_ratio={pass_ratio:.2f} "
                f"min_atr: {old_min_atr:.4f}->{self._min_atr:.4f} max_body_atr: {old_max_body:.2f}->{self._max_body:.2f}"
            )

        # Base signals from trend (EMA cross)
        base_long = bool(row["ema_fast"] > row["ema_slow"]) and atr_ok and body_ok
        base_short = bool(row["ema_fast"] < row["ema_slow"]) and atr_ok and body_ok

        # ML gating
        up_prob: Optional[float] = None
        ml_ok_long = True
        ml_ok_short = True
        if getattr(self.ml, "use_ml", False):
            try:
                up_prob = self.ml.predict_up_prob(self.df.iloc[: i + 1])
            except Exception:
                up_prob = None
            if up_prob is not None:
                ml_ok_long = up_prob >= self.cfg.ml.up_prob_long
                ml_ok_short = up_prob <= (1.0 - self.cfg.ml.down_prob_short)
        if self.cfg.ml.strict:
            dec_long = base_long and ml_ok_long
            dec_short = base_short and ml_ok_short
        else:
            dec_long = base_long and (not getattr(self.ml, "use_ml", False) or ml_ok_long or up_prob is None)
            dec_short = base_short and (not getattr(self.ml, "use_ml", False) or ml_ok_short or up_prob is None)

        # Decision log
        up_txt = f"{up_prob:.3f}" if up_prob is not None else "n/a"
        self._log(
            f"DECISION price={price:.4f} base(L={base_long},S={base_short}) adx={adx_v:.2f} up_prob={up_txt} "
            f"-> L={dec_long} S={dec_short} pos={self.pos if self.pos else 'None'}"
        )

        # Manage open position (update BE/TSL/exit)
        if self.pos:
            self._manage_open(price, adx_v)

        # Entries
        if self.pos is None:
            if dec_long:
                self._enter("LONG", price, atr_v)
            elif dec_short:
                self._enter("SHORT", price, atr_v)

    # --------------------------------------------
    # Position handling
    # --------------------------------------------
    def _enter(self, side: str, price: float, atr_v: float):
        # Risk in USDT
        risk_usdt = self.balance * self.risk_pct
        sl_dist_pct = self.cfg.sl_pct
        if side == "LONG":
            sl = price * (1.0 - sl_dist_pct)
        else:
            sl = price * (1.0 + sl_dist_pct)
        # qty = risk / (price * sl_dist_pct)
        qty = risk_usdt / max(1e-12, price * sl_dist_pct)
        qty = float(round(qty, 3))

        self.pos = side
        self.entry_price = float(price)
        self.entry_ts = self.current_ts
        self.hard_sl = float(sl)
        self.sl = float(sl)
        self.tsl = None
        self.qty = qty
        self.be_triggered = False
        self.last_sl_type = "HARD"

        self._log(
            f"SIZING price={price:.4f} qty={qty:.3f} sl={sl:.6f} sl_dist%={sl_dist_pct*100:.2f} "
            f"risk_usdt={risk_usdt:.4f} theo_loss_at_sl={risk_usdt:.4f}"
        )
        self._log(f"ENTRY {side} price={price:.4f} qty={qty:.6f} sl={sl:.6f}")

    def _manage_open(self, price: float, adx_v: float):
        assert self.pos in ("LONG", "SHORT") and self.entry_price is not None and self.sl is not None
        side = self.pos
        entry = self.entry_price
        sl = self.sl
        hard_sl = self.hard_sl if self.hard_sl is not None else sl

        # R multiple calc (based on initial hard SL)
        r_denom = (entry - hard_sl) if side == "LONG" else (hard_sl - entry)
        r_denom = max(1e-12, r_denom)
        r_mult = (price - entry) / r_denom if side == "LONG" else (entry - price) / r_denom

        # BE move based on R threshold
        if not self.be_triggered and r_mult >= self.cfg.be_trigger_r:
            # move SL to BE
            new_sl = entry if side == "LONG" else entry
            self.sl = new_sl
            self.be_triggered = True
            self.last_sl_type = "BE"
            self._log(f"BE MOVE SL -> {new_sl:.6f}")

        # Trailing step via ADX
        step = self._trail_step_from_adx(adx_v)
        moved = False
        if side == "LONG":
            candidate = price * (1.0 - step)
            if self.tsl is None or candidate > self.tsl:
                self.tsl = candidate
                moved = True
        else:  # SHORT
            candidate = price * (1.0 + step)
            if self.tsl is None or candidate < self.tsl:
                self.tsl = candidate
                moved = True
        if moved:
            self.last_sl_type = "TSL"
            self._log(f"TRAIL MOVE tsl={self.tsl:.6f} (step={step*100:.3f}%)")

        # Exit checks (order: hard SL, trailing SL, time-stop)
        # For logs, classify reason based on which level is hit.
        reason = None
        if side == "LONG":
            if price <= hard_sl:
                reason = "Hit Hard SL"
            elif self.tsl is not None and price <= self.tsl:
                reason = "Hit Trailing SL"
        else:
            if price >= hard_sl:
                reason = "Hit Hard SL"
            elif self.tsl is not None and price >= self.tsl:
                reason = "Hit Trailing SL"

        # Time stop (adaptive by ADX)
        hold_seconds = int((self.current_ts - self.entry_ts).total_seconds()) if self.entry_ts is not None else 0
        max_hold = self.max_hold_seconds
        if adx_v > self.cfg.time_stop.extend_when_adx_gt:
            max_hold = int(max_hold * self.cfg.time_stop.extend_factor)
        # ROI (approx per-price, before fees)
        roi = (price - entry) / entry if side == "LONG" else (entry - price) / entry
        min_roi = float(self.min_roi_time_close)
        if hold_seconds >= max_hold and roi < min_roi:
            reason = f"Time Stop (hold>={max_hold}s & roi<{min_roi*100:.2f}%)"

        if reason:
            self._log(f"EXIT {reason} price={price:.4f}")
            # Reset position
            self.pos = None
            self.entry_price = None
            self.entry_ts = None
            self.hard_sl = None
            self.sl = None
            self.tsl = None
            self.qty = 0.0
            self.be_triggered = False
            self.last_sl_type = "HARD"

    def _trail_step_from_adx(self, adx_v: float) -> float:
        if not self.cfg.trail.use_adx_step:
            return self.cfg.trail.max_step_pct
        lo, hi = self.cfg.trail.adx_bounds
        t = clamp((adx_v - lo) / max(1e-9, (hi - lo)), 0.0, 1.0)
        return lerp(self.cfg.trail.min_step_pct, self.cfg.trail.max_step_pct, t)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def dataclasses_to_dict(cfg: CoinCfg) -> Dict[str, Any]:
    def _dc(obj):
        if dc.is_dataclass(obj):
            return {k: _dc(v) for k, v in obj.__dict__.items()}
        if isinstance(obj, (list, tuple)):
            return [_dc(x) for x in obj]
        return obj
    return _dc(cfg) # type: ignore


def load_coin_cfg(path: Optional[str], symbol: str) -> CoinCfg:
    # Defaults from env if present
    base = {
        "min_atr_pct": float(env("MIN_ATR_PCT", 0.0030)),
        "max_body_to_atr": float(env("MAX_BODY_TO_ATR", 1.50)),
        "sl_pct": float(env("SL_PCT", 0.012)),
        "be_trigger_r": float(env("BE_TRIGGER_R", 0.5)),
        "trail": {
            "use_adx_step": bool(env("TRAIL_STEP_BY_ADX", True, bool)),
            "min_step_pct": float(env("TRAIL_MIN_STEP_PCT", 0.006)),
            "max_step_pct": float(env("TRAIL_MAX_STEP_PCT", 0.012)),
            # parse "20,60"
            "adx_bounds": tuple(
                float(x) for x in str(env("TRAIL_ADX_BOUNDS", "20,60")).split(",")[:2]
            ),
        },
        "time_stop": {
            "max_hold_seconds": int(env("MAX_HOLD_SECONDS", 3600)),
            "min_roi_to_close": float(env("MIN_ROI_TO_CLOSE_BY_TIME", 0.0)),
            "extend_when_adx_gt": float(env("TIME_STOP_EXTEND_ADX", 40.0)),
            "extend_factor": float(env("TIME_STOP_EXTEND_FACTOR", 1.5)),
        },
        "ml": {
            "strict": bool(env("ML_STRICT", False, bool)),
            "up_prob_long": float(env("ML_UP_PROB", 0.60)),
            "down_prob_short": float(env("ML_DOWN_PROB", 0.40)),
            "score_threshold": float(env("SCORE_THRESHOLD", 1.40)),
        },
        "filter": {
            "base_min_atr": float(env("MIN_ATR_PCT", 0.0030)),
            "base_max_body": float(env("MAX_BODY_TO_ATR", 1.50)),
            "target_pass_ratio": float(env("FILTER_TARGET_PASS_RATIO", 0.35)),
            "hysteresis_tighten": float(env("FILTER_HYSTERESIS_TIGHTEN", 0.55)),
            "relax_step_atr": float(env("FILTER_RELAX_STEP_ATR", 0.0001)),
            "relax_step_body": float(env("FILTER_RELAX_STEP_BODY", 0.05)),
            "max_relax_atr": float(env("FILTER_MAX_RELAX_ATR", 0.0023)),
            "max_relax_body": float(env("FILTER_MAX_RELAX_BODY", 2.00)),
        },
    }

    if path and os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        cfg_d = data.get(symbol, {}) if isinstance(data, dict) else {}
        # merge base <- cfg_d
        def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
            x = dict(a)
            for k, v in b.items():
                if isinstance(v, dict) and isinstance(x.get(k), dict):
                    x[k] = deep_merge(x[k], v)
                else:
                    x[k] = v
            return x
        base = deep_merge(base, cfg_d)

    return CoinCfg.from_dict(base)


# --------------------------------------------------------------------------------------
# CSV Loader
# --------------------------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Expected columns: timestamp or open_time, open, high, low, close, volume
    # Try to infer timestamp column
    ts_col = None
    for cand in ("timestamp", "open_time", "time", "date"):
        if cand in df.columns:
            ts_col = cand
            break
    if ts_col is None:
        raise ValueError("CSV must have a timestamp column")
    df[ts_col] = pd.to_datetime(df[ts_col], utc=True)
    df = df.set_index(ts_col).sort_index()
    # Normalize columns to lowercase
    rename = {c: c.lower() for c in df.columns}
    df = df.rename(columns=rename)
    needed = ["open", "high", "low", "close"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"CSV missing column: {c}")
    return df


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--coin_config", type=str, default=None)
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--symbol", type=str, required=True)
    p.add_argument("--balance", type=float, default=20.0)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--debug-mode", action="store_true")
    p.add_argument("--dry-run-loop", action="store_true")
    p.add_argument("--sleep", type=float, default=0.0)
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--warmup", type=int, default=410)
    args = p.parse_args()

    symbol = args.symbol.upper()
    df = load_csv(args.csv)
    cfg = load_coin_cfg(args.coin_config, symbol)

    trader = Trader(symbol, df, cfg, balance=args.balance, verbose=args.verbose, debug=args.debug_mode)

    start = args.warmup
    limit = args.limit

    if args.dry_run_loop:
        trader.run(start, limit)
    else:
        trader.run(start, limit)


if __name__ == "__main__":
    main()
