#!/usr/bin/env python3
"""
Backtester resmi untuk skenario scalping.
- Replay CSV bar-per-bar memakai logika real di newrealtrading.
- Ringkasan akhir: WinRate, ProfitFactor, avg PnL.
- Simpan detail trade ke CSV bila diminta.
"""
from __future__ import annotations
import os, argparse, json, time
import pandas as pd
import numpy as np

import newrealtrading as nrt
from newrealtrading import TradingManager, calculate_indicators
from engine_core import pnl_net


def run_backtest(args) -> tuple[dict, pd.DataFrame]:
    df = pd.read_csv(args.csv)
    if "timestamp" not in df.columns:
        if "open_time" in df.columns:
            df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", errors="coerce")
        elif "date" in df.columns:
            df["timestamp"] = pd.to_datetime(df["date"], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # warmup indikator & ML
    min_train = int(float(os.getenv("ML_MIN_TRAIN_BARS", "400")))
    warmup = max(300, min_train + 10)
    start_i = min(warmup, len(df) - 1)

    # siapkan config sementara
    with open(args.coin_config, "r") as f:
        cfg = json.load(f)
    sym_cfg = cfg.get(args.symbol, {})
    # (opsi) merge preset
    preset_name = args.preset or sym_cfg.get("use_preset") or (cfg.get("PRESETS") and "SCALP-ADA-15M-LEGACY")
    if preset_name and isinstance(cfg.get("PRESETS"), dict) and preset_name in cfg.get("PRESETS", {}):
        base_p = dict(cfg["PRESETS"][preset_name])
        merged = {**base_p, **sym_cfg}
        for k in ("filters", "ml"):
            if k in base_p or k in sym_cfg:
                mk = dict(base_p.get(k, {}))
                mk.update(sym_cfg.get(k, {}))
                merged[k] = mk
        sym_cfg = merged
    else:
        # Fallback: jika args.preset diminta tapi tidak ada, kembali ke use_preset atau LEGACY
        fb = sym_cfg.get("use_preset") or "SCALP-ADA-15M-LEGACY"
        if args.preset and fb in (cfg.get("PRESETS") or {}):
            base_p = dict(cfg["PRESETS"][fb])
            merged = {**base_p, **sym_cfg}
            for k in ("filters", "ml"):
                if k in base_p or k in sym_cfg:
                    mk = dict(base_p.get(k, {})); mk.update(sym_cfg.get(k, {})); merged[k] = mk
            sym_cfg = merged
            print(f"[INFO] preset '{args.preset}' tidak ditemukan. Fallback ke '{fb}'.")
        elif args.preset:
            print(f"[WARN] preset '{args.preset}' tidak ditemukan dan tidak ada fallback yang cocok.")
    sym_cfg["heikin"] = bool(args.heikin)
    sym_cfg["rsi_mode"] = args.rsi_mode
    if args.no_macd_confirm:
        sym_cfg["use_macd_confirm"] = False
    if args.cooldown is not None:
        sym_cfg["cooldown_seconds"] = int(args.cooldown)
    sym_cfg["taker_fee"] = float(args.fee_bps) / 10000.0
    sym_cfg["SLIPPAGE_PCT"] = float(args.slip_bps) * 0.01
    if args.htf:
        sym_cfg["use_htf_filter"] = 1
        sym_cfg["htf"] = args.htf
    else:
        sym_cfg["use_htf_filter"] = 0

    # sinkronkan parameter ML
    ml_cfg = sym_cfg.setdefault("ml", {})
    if args.use_ml is not None:
        os.environ["USE_ML"] = str(int(args.use_ml))
        ml_cfg["enabled"] = bool(int(args.use_ml))
    if args.ml_thr is not None:
        os.environ["SCORE_THRESHOLD"] = str(float(args.ml_thr))
        ml_cfg["score_threshold"] = float(args.ml_thr)

    cfg[args.symbol] = sym_cfg
    tmp_cfg = f"_tmp_cfg_{args.symbol}.json"
    with open(tmp_cfg, "w") as f:
        json.dump(cfg, f, indent=2)
    
    # Log preset final yang dipakai agar mudah diverifikasi dari CLI
    try:
        print(f"[INFO] Preset dipakai: {sym_cfg.get('use_preset', '(inline merged)')} / rsi_mode={sym_cfg.get('rsi_mode')} macd_confirm={sym_cfg.get('use_macd_confirm', False)}")
    except Exception:
        pass

    mgr = TradingManager(
        tmp_cfg,
        [args.symbol],
        no_atr_filter=args.no_atr_filter,
        no_body_filter=args.no_body_filter,
        ml_override=args.ml_override,
    )
    trader = mgr.traders[args.symbol]
    trader._log = lambda *a, **k: None

    # pre-fit ML
    try:
        warm_df = df.iloc[: start_i + 1].copy()
        ind_warm = calculate_indicators(warm_df, heikin=bool(args.heikin))
        trader.ml.fit_if_needed(ind_warm)
    except Exception:
        pass

    trades: list[dict] = []
    trader._entry_count = 0
    trader._exit_count = 0
    _orig_enter = trader._enter_position
    _orig_exit = trader._exit_position

    def _enter_wrap(side: str, price: float, atr: float, available_balance: float, **kw) -> float:
        trader._entry_count += 1
        used = _orig_enter(side, price, atr, available_balance, **kw)
        trader._last_used_margin = used
        return used

    def _exit_wrap(price, reason, **kw):
        pos = trader.pos
        ts = kw.get("now_ts", None)
        exit_time = pd.Timestamp.utcnow()
        if ts is not None:
            try:
                exit_time = pd.to_datetime(int(ts), unit="s", utc=True)  # type: ignore[arg-type]
            except Exception:
                pass
        if pos.side and pos.entry and pos.qty:
            pnl, roi = pnl_net(pos.side, float(pos.entry), float(price), float(pos.qty), args.fee_bps, args.slip_bps)
            trades.append({
                "symbol": args.symbol,
                "side": pos.side,
                "entry_price": float(pos.entry),
                "exit_price": float(price),
                "qty": float(pos.qty),
                "pnl": float(pnl),
                "roi_pct": float(roi),
                "used_margin": float(getattr(trader, "_last_used_margin", 0.0)),
                "reason": reason,
                "entry_time": pos.entry_time,
                "exit_time": exit_time,
            })
        trader._exit_count += 1
        _orig_exit(price, reason, **kw)

    trader._enter_position = _enter_wrap
    trader._exit_position = _exit_wrap

    t0 = time.time()
    steps = 0
    for i in range(start_i, min(len(df), start_i + args.steps)):
        sub = df.iloc[: i + 1].copy()
        # pakai jam virtual dari bar terakhir (epoch detik UTC)
        try:
            now_ts = int(pd.to_datetime(sub["timestamp"].iloc[-1]).tz_convert("UTC").timestamp())
        except Exception:
            now_ts = int(pd.to_datetime(sub["timestamp"].iloc[-1], utc=True).timestamp())
        trader.check_trading_signals(sub, args.balance, now_ts=now_ts)
        steps += 1
    elapsed = time.time() - t0

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        wins = (trades_df["pnl"] > 0).sum()
        losses = (trades_df["pnl"] <= 0).sum()
        pf = (
            trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()
            / abs(trades_df.loc[trades_df["pnl"] <= 0, "pnl"].sum())
            if losses > 0
            else np.inf
        )
        wr = (trades_df["pnl"] > 0).mean() * 100
        avg_pnl = trades_df["pnl"].mean()
    else:
        wins = losses = 0
        pf = np.nan
        wr = 0.0
        avg_pnl = 0.0

    summary = {
        "symbol": args.symbol,
        "rows_total": len(df),
        "warmup_index": start_i,
        "steps_executed": steps,
        "entries": trader._entry_count,
        "exits": trader._exit_count,
        "last_position": trader.pos.side,
        "trades": len(trades),
        "win_rate_pct": round(float(wr), 2),
        "profit_factor": float(pf) if pf == np.inf else (round(float(pf), 2) if not np.isnan(pf) else None),
        "avg_pnl": round(float(avg_pnl), 6),
        "elapsed_sec": round(elapsed, 2),
    }
    return summary, trades_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=os.getenv("BT_SYMBOL"))
    ap.add_argument("--csv", default=os.getenv("BT_CSV"))
    ap.add_argument("--coin_config", default="coin_config.json")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--balance", type=float, default=20.0)
    ap.add_argument("--use-ml", type=int, choices=[0,1], default=None)
    ap.add_argument("--ml-thr", type=float, default=2.0)
    ap.add_argument("--htf", default=None)
    ap.add_argument("--heikin", action="store_true")
    ap.add_argument("--preset", default=os.getenv("BT_PRESET"), help="Nama preset di coin_config.json/`PRESETS`")
    ap.add_argument("--cooldown", type=int, default=None, help="Override cooldown_seconds")
    ap.add_argument("--no-macd-confirm", action="store_true", help="Matikan konfirmasi MACD di sinyal dasar")
    ap.add_argument("--fee-bps", type=float, default=10.0)
    ap.add_argument("--slip-bps", type=float, default=0.0)
    ap.add_argument("--no-atr-filter", action="store_true")
    ap.add_argument("--no-body-filter", action="store_true")
    ap.add_argument("--ml-override", action="store_true")
    ap.add_argument("--rsi-mode", choices=["PULLBACK","MIDRANGE"], default="PULLBACK")
    ap.add_argument("--out", default=None, help="simpan trade ke CSV")
    args = ap.parse_args()
    if not args.symbol or not args.csv:
        ap.error("--symbol dan --csv wajib diisi (atau set env BT_SYMBOL / BT_CSV).")

    os.environ.setdefault("USE_ML", "1")
    os.environ.setdefault("SCORE_THRESHOLD", "2.0")
    os.environ.setdefault("ML_MIN_TRAIN_BARS", "400")
    os.environ.setdefault("ML_RETRAIN_EVERY", "5000")
    os.environ.setdefault("ML_WEIGHT", "1.0")

    summary, trades_df = run_backtest(args)

    print("\n=== RINGKASAN ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    if args.out and not trades_df.empty:
        trades_df.to_csv(args.out, index=False)
        print(f"\nTrades tersimpan di: {args.out}")


if __name__ == "__main__":
    main()
