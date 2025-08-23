#!/usr/bin/env python3
"""
Dry-run helper untuk newrealtrading.py
- Replay bar-by-bar dari CSV, 100% pakai logic real (ML + filter + trailing/BE)
- Hitung ringkasan: entries, exits, trades (complete), WinRate, ProfitFactor, avg PnL
- Export detail trades ke CSV (--out)

Contoh pakai:
python tools_dryrun_summary.py \
  --symbol ADAUSDT \
  --csv data/ADAUSDT_15m_2025-06-01_to_2025-08-09.csv \
  --coin_config coin_config.json \
  --steps 500 --balance 20 \
  --out ADA_dryrun_trades_500.csv

Tips percepat:
export USE_ML=1; export SCORE_THRESHOLD=2.0; export ML_RETRAIN_EVERY=5000
"""
# from __future__ import harus berada di awal
from __future__ import annotations

# --- force third-party deprecation ignores (so PYTHONWARNINGS=error won't fail) ---
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"^websockets(\.|$)")
warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"^binance(\.|$)")
# -----------------------------------------------------------------------------
import newrealtrading as nrt

import os, sys, time, argparse, json
import pandas as pd
import numpy as np
from engine_core import (
    compute_base_signals_backtest,
    apply_filters,
    get_coin_ml_params,
    rolling_twap,
)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*is deprecated and will be removed in a future version.*",
)

# Simpan method asli dan pasang wrapper yang tanda tangannya identik
_orig_enter = nrt.CoinTrader._enter_position
_orig_exit = nrt.CoinTrader._exit_position


def _enter_wrap(self, side: str, price: float, atr: float, available_balance: float, **kw) -> float:
    used_margin = _orig_enter(self, side, price, atr, available_balance, **kw)
    try:
        if used_margin and used_margin > 0:
            self._journal.append(
                {
                    "t": "entry",
                    "side": side,
                    "price": float(price),
                    "qty": float(self.pos.qty if self.pos and self.pos.qty else 0.0),
                    "ts": int((kw.get("now_ts") or time.time())),
                }
            )
    except Exception:
        pass
    return float(used_margin or 0.0)


def _exit_wrap(self, price: float, reason: str = "Exit", **kw) -> None:
    prev_cd = getattr(self, "cooldown_until_ts", None)
    _orig_exit(self, price, reason, **kw)
    try:
        if getattr(self, "cooldown_until_ts", None) and self.cooldown_until_ts != prev_cd:
            now_ts = kw.get("now_ts") or time.time()
            dur = int(self.cooldown_until_ts - now_ts)
            print(f"[{self.symbol}] COOLDOWN set {dur}s karena {reason}")
    except Exception:
        pass
    try:
        self._journal.append(
            {
                "t": "exit",
                "reason": str(reason),
                "price": float(price),
                "ts": int((kw.get("now_ts") or time.time())),
            }
        )
    except Exception:
        pass


nrt.CoinTrader._enter_position = _enter_wrap
nrt.CoinTrader._exit_position = _exit_wrap


def run_dry(
    symbol: str,
    csv_path: str,
    coin_config_path: str,
    steps_limit: int,
    balance: float,
    debug_reasons: bool = False,
) -> tuple[dict, pd.DataFrame, pd.DataFrame | None]:
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        if "open_time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", errors="coerce")
        elif "date" in df.columns:
            df["timestamp"] = pd.to_datetime(df["date"], errors="coerce")
    # Pastikan semua timestamp UTC-aware agar konsisten di seluruh stack
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # warmup supaya indikator & ML siap
    min_train = int(float(os.getenv("ML_MIN_TRAIN_BARS", "400")))
    warmup = max(300, min_train + 10)
    start_i = min(warmup, len(df) - 1)

    mgr = nrt.TradingManager(coin_config_path, [symbol])
    trader = mgr.traders[symbol]
    trader._log = lambda *args, **kwargs: None  # matikan log biar cepat
    trader._journal = []
    reasons_rows: list[dict] = [] if debug_reasons else []

    # Pre-fit ML sekali di warmup (opsional, untuk speed)
    try:
        warm_df = df.iloc[: start_i + 1].copy()
        ind_warm = nrt.calculate_indicators(warm_df)
        trader.ml.fit_if_needed(ind_warm)
    except Exception:
        pass


    # Replay loop
    t0 = time.time()
    steps = 0
    for i in range(start_i, min(len(df) - 1, start_i + steps_limit)):
        sub = df.iloc[: i + 1].copy()
        # ambil close time bar terakhir sebagai jam virtual
        last_ts = sub["timestamp"].iloc[-1]
        try:
            if isinstance(last_ts, pd.Timestamp):
                last_close_s = int(last_ts.tz_convert("UTC").timestamp())
            else:
                last_close_s = int(pd.to_datetime(last_ts, utc=True).timestamp())
        except Exception:
            # fallback aman
            last_close_s = int(pd.Timestamp.utcnow().tz_localize("UTC").timestamp())
        trader.check_trading_signals(sub, balance, now_ts=last_close_s)

        if debug_reasons:
            ind = nrt.calculate_indicators(sub, heikin=bool(trader.config.get('heikin', False)))
            last_ind = ind.iloc[-1]
            atr_ok, body_ok, meta = apply_filters(last_ind, trader.config)
            long_base0, short_base0 = compute_base_signals_backtest(ind, trader.config)
            up_prob = trader.ml.predict_up_prob(ind) if trader.ml.use_ml else None
            if up_prob is not None:
                try:
                    up_prob = float(up_prob)
                except Exception:
                    up_prob = float(up_prob)
            params = get_coin_ml_params(symbol, {symbol: trader.config})
            long_base = long_base0 and atr_ok and body_ok
            short_base = short_base0 and atr_ok and body_ok
            score_long = 1.0 if long_base else 0.0
            score_short = 1.0 if short_base else 0.0
            if params["enabled"] and up_prob is not None:
                if long_base and up_prob >= params["up_prob"]:
                    score_long += params["weight"]
                if short_base and up_prob <= params["down_prob"]:
                    score_short += params["weight"]
            tw_cfg = trader.config
            twap15_ok_long = twap15_ok_short = True
            htf_bonus_long = htf_bonus_short = False
            if bool(tw_cfg.get("use_twap_indicator", False)):
                twap15_w = int(tw_cfg.get("twap_15m_window", 20))
                k_atr = float(tw_cfg.get("twap_atr_k", 0.4))
                twap15 = rolling_twap(sub["close"], twap15_w).iloc[-1]
                dev = float(twap15 - sub["close"].iloc[-1])
                atrv = float(last_ind.get("atr", 0.0))
                twap15_ok_long = (dev >= k_atr * atrv)
                twap15_ok_short = (-dev >= k_atr * atrv)
                htf_tf = str(tw_cfg.get("htf", {}).get("timeframe", "1h")).lower()
                htf_close = sub["close"].resample(htf_tf).last().dropna()
                if len(htf_close) >= 30:
                    htwap = rolling_twap(htf_close, 22).iloc[-1]
                    ema22_htf = htf_close.ewm(span=22, adjust=False).mean().iloc[-1]
                    price_htf = htf_close.iloc[-1]
                    htf_bonus_long = (price_htf >= htwap) and (ema22_htf >= htwap)
                    htf_bonus_short = (price_htf <= htwap) and (ema22_htf <= htwap)
                if long_base and not twap15_ok_long:
                    long_base = False
                    score_long = 0.0
                if short_base and not twap15_ok_short:
                    short_base = False
                    score_short = 0.0
                if long_base and htf_bonus_long:
                    score_long += 0.2
                if short_base and htf_bonus_short:
                    score_short += 0.2
            thr = params["score_threshold"]
            if not params["strict"] and up_prob is None:
                thr = 1.0
            decision_dbg = None
            if score_long >= thr and score_long > score_short:
                decision_dbg = "LONG"
            elif score_short >= thr and score_short > score_long:
                decision_dbg = "SHORT"
            ts_iso = pd.to_datetime(last_ts, utc=True).isoformat()
            for side, base_flag in (("LONG", long_base0), ("SHORT", short_base0)):
                if base_flag and decision_dbg != side:
                    reasons = []
                    if not atr_ok:
                        reasons.append("ATR")
                    if not body_ok:
                        reasons.append("BODY")
                    if params["enabled"]:
                        if up_prob is None:
                            if params["strict"]:
                                reasons.append("ML_WARM")
                        else:
                            if side == "LONG" and up_prob < params["up_prob"]:
                                reasons.append("ML")
                            if side == "SHORT" and up_prob > params["down_prob"]:
                                reasons.append("ML")
                    reasons_rows.append(
                        {
                            "timestamp": ts_iso,
                            "side": side,
                            "atr_ok": atr_ok,
                            "body_ok": body_ok,
                            "twap15_ok": twap15_ok_long if side == "LONG" else twap15_ok_short,
                            "ltf_ok": True,
                            "htf_twap_bonus": htf_bonus_long if side == "LONG" else htf_bonus_short,
                            "score_long_after_bonus": score_long,
                            "score_short_after_bonus": score_short,
                            "base_long": bool(long_base0),
                            "base_short": bool(short_base0),
                            "ml_prob": up_prob,
                            "score_long": score_long,
                            "score_short": score_short,
                            "reason_list": ";".join(reasons) if reasons else "-",
                        }
                    )
        steps += 1
    elapsed = time.time() - t0

    # Ringkasan
    events = getattr(trader, "_journal", [])
    trades: list[dict] = []
    current = None
    for ev in events:
        if ev.get("t") == "entry":
            current = ev
        elif ev.get("t") == "exit" and current:
            trades.append(
                {
                    "entry_ts": pd.to_datetime(current["ts"], unit="s", utc=True).isoformat(),
                    "side": current.get("side"),
                    "entry": current.get("price"),
                    "qty": current.get("qty", 0.0),
                    "exit_ts": pd.to_datetime(ev["ts"], unit="s", utc=True).isoformat(),
                    "exit": ev.get("price"),
                    "reason": ev.get("reason", "Exit"),
                    "pnl": (
                        (ev.get("price") - current.get("price")) * current.get("qty", 0.0)
                        if current.get("side") == "LONG"
                        else (current.get("price") - ev.get("price")) * current.get("qty", 0.0)
                    ),
                }
            )
            current = None

    trades_df = pd.DataFrame(trades)
    closed_df = trades_df
    if not closed_df.empty:
        wins = (closed_df["pnl"] > 0).sum()
        losses = (closed_df["pnl"] <= 0).sum()
        pf = (
            closed_df.loc[closed_df["pnl"] > 0, "pnl"].sum()
            / abs(closed_df.loc[closed_df["pnl"] <= 0, "pnl"].sum())
            if losses > 0
            else np.inf
        )
        wr = (closed_df["pnl"] > 0).mean() * 100
        avg_pnl = closed_df["pnl"].mean()
    else:
        wins = losses = 0
        pf = np.nan
        wr = 0.0
        avg_pnl = 0.0

    entries = sum(1 for ev in events if ev.get("t") == "entry")
    exits = sum(1 for ev in events if ev.get("t") == "exit")

    summary = {
        "symbol": symbol,
        "rows_total": len(df),
        "warmup_index": start_i,
        "steps_executed": steps,
        "entries": entries,
        "exits": exits,
        "last_position": trader.pos.side,
        "trades": len(closed_df),
        "win_rate_pct": round(float(wr), 2),
        "profit_factor": float(pf) if pf == np.inf else (round(float(pf), 2) if not np.isnan(pf) else None),
        "avg_pnl": round(float(avg_pnl), 6),
        "elapsed_sec": round(elapsed, 2),
    }
    reasons_df = pd.DataFrame(reasons_rows) if debug_reasons else None
    return summary, closed_df, reasons_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--coin_config", default="coin_config.json")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--balance", type=float, default=20.0)
    ap.add_argument("--out", default=None, help="Path CSV untuk menyimpan trades")
    ap.add_argument("--use-ml", type=int, choices=[0,1], default=None)
    ap.add_argument("--ml-thr", type=float, default=None, help="Threshold odds ML (1.0=OR, 2.0=AND)")
    ap.add_argument("--trailing-step", type=float, default=None)
    ap.add_argument("--trailing-trigger", type=float, default=None)
    ap.add_argument("--debug-reasons", action="store_true")
    args = ap.parse_args()

    # saran env untuk speed
    os.environ.setdefault("USE_ML", "1")
    os.environ.setdefault("SCORE_THRESHOLD", "2.0")
    os.environ.setdefault("ML_MIN_TRAIN_BARS", "400")
    os.environ.setdefault("ML_RETRAIN_EVERY", "5000")

    if args.use_ml is not None:
        os.environ["USE_ML"] = str(int(args.use_ml))
    if args.ml_thr is not None:
        os.environ["SCORE_THRESHOLD"] = str(float(args.ml_thr))

    cfg_path = args.coin_config
    if args.trailing_step is not None or args.trailing_trigger is not None:
        try:
            with open(args.coin_config, "r") as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}
        sym_cfg = cfg.get(args.symbol.upper(), {})
        if args.trailing_step is not None:
            sym_cfg["trailing_step"] = float(args.trailing_step)
        if args.trailing_trigger is not None:
            sym_cfg["trailing_trigger"] = float(args.trailing_trigger)
        cfg[args.symbol.upper()] = sym_cfg
        tmp_cfg_path = f"_tmp_{args.symbol.upper()}_cfg.json"
        with open(tmp_cfg_path, "w") as f:
            json.dump(cfg, f)
        cfg_path = tmp_cfg_path

    summary, trades_df, reasons_df = run_dry(
        args.symbol.upper(),
        args.csv,
        cfg_path,
        args.steps,
        args.balance,
        debug_reasons=args.debug_reasons,
    )

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    if args.out and not trades_df.empty:
        trades_df.to_csv(args.out, index=False)
        print(f"\nTrades saved to: {args.out}")
        if args.debug_reasons and reasons_df is not None and not reasons_df.empty:
            reason_out = args.out.replace(".csv", "_reasons.csv")
            reasons_df.to_csv(reason_out, index=False)
            print(f"Reasons saved to: {reason_out}")
    elif args.debug_reasons and reasons_df is not None and not reasons_df.empty:
        reason_out = f"{args.symbol.upper()}_reasons.csv"
        reasons_df.to_csv(reason_out, index=False)
        print(f"Reasons saved to: {reason_out}")

if __name__ == "__main__":
    main()
