# papertrade.py — Live paper feed + CSV journal (Binance Futures, no orders)
# ==============================================================
# Kebutuhan:
#   pip install python-binance pandas numpy ta python-dotenv
# Catatan:
#   - Membaca sinyal dari newrealtrading.CoinTrader (patch A–D sudah di sana)
#   - Tidak mengirim order; hanya mencatat entry/exit & PnL netto (fees+slippage) ke CSV
# ==============================================================

import os, sys, time, json, math, csv, argparse
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone

import pandas as pd
import numpy as np

from dotenv import load_dotenv
load_dotenv(override=True)

# Binance
from binance.client import Client
from binance.enums import HistoricalKlinesType

# Import core engine dari file yang sudah kamu punya
try:
    from newrealtrading import (
        CoinTrader, floor_to_step, _to_float, _to_bool,
        load_coin_config, merge_config
    )
except ImportError as e:
    print(f"[ERROR] Failed to import from newrealtrading: {e}")
    sys.exit(1)

# -----------------------------
# Util waktu & interval
# -----------------------------
INTERVAL_MAP = {
    "1m":  Client.KLINE_INTERVAL_1MINUTE,
    "3m":  Client.KLINE_INTERVAL_3MINUTE,
    "5m":  Client.KLINE_INTERVAL_5MINUTE,
    "15m": Client.KLINE_INTERVAL_15MINUTE,
    "30m": Client.KLINE_INTERVAL_30MINUTE,
    "1h":  Client.KLINE_INTERVAL_1HOUR,
    "2h":  Client.KLINE_INTERVAL_2HOUR,
    "4h":  Client.KLINE_INTERVAL_4HOUR,
    "6h":  Client.KLINE_INTERVAL_6HOUR,
    "12h": Client.KLINE_INTERVAL_12HOUR,
    "1d":  Client.KLINE_INTERVAL_1DAY,
}

def interval_seconds(s: str) -> int:
    num = int(s[:-1]); unit = s[-1]
    if unit == 'm': return num * 60
    if unit == 'h': return num * 3600
    if unit == 'd': return num * 86400
    raise ValueError(f"Interval tidak didukung: {s}")

def next_close_sleep(interval_s: int, skew: float = 1.0) -> float:
    # tidur sampai boundary bar berikutnya + skew detik
    now = time.time()
    return (math.floor(now / interval_s) + 1) * interval_s + skew - now

# -----------------------------
# Binance client wrapper (retry)
# -----------------------------
class BinanceFutures:
    def __init__(self, requests_timeout: int = 20, max_retries: int = 5):
        self.client = Client(requests_params={"timeout": requests_timeout})
        self.max_retries = max_retries

    def _retry(self, fn, *args, **kwargs):
        delay = 0.75
        last_ex = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                last_ex = e
                print(f"[WARN] Binance call failed (attempt {attempt}/{self.max_retries}): {e}")
                time.sleep(delay)
                delay = min(delay * 1.7, 8.0)
        raise last_ex if last_ex else RuntimeError("Unknown error occurred during Binance call retries.")

    def exchange_info(self) -> dict:
        # Futures USDⓈ-M
        return self._retry(self.client.futures_exchange_info)

    def klines(self, symbol: str, interval: str, limit: int = 500) -> List[list]:
        # Ambil klines futures (closed bars)
        result = self._retry(
            self.client.futures_klines,
            symbol=symbol.upper(),
            interval=interval,
            limit=limit
        )
        if not isinstance(result, list) or not all(isinstance(item, list) for item in result):
            raise TypeError("Expected a List of lists, but got a different type.")
        return result

# -----------------------------
# Inject exchange filters -> coin_config.json
# -----------------------------
def ensure_filters_to_coin_config(bnc: BinanceFutures, coin_cfg_path: str, symbols: List[str]) -> dict:
    cfg_all = load_coin_config(coin_cfg_path) if os.path.exists(coin_cfg_path) else {}
    info = bnc.exchange_info()
    by_sym = {s['symbol']: s for s in info.get('symbols', [])}

    changed = False
    for sym in symbols:
        ex = by_sym.get(sym.upper())
        if not ex:
            print(f"[WARN] {sym}: tidak ditemukan di exchangeInfo (futures). Lewati.")
            continue

        lot = next((f for f in ex.get('filters', []) if f.get('filterType') == 'LOT_SIZE'), None)
        min_notional = next((f for f in ex.get('filters', []) if f.get('filterType') == 'MIN_NOTIONAL'), None)

        step_size = float(lot.get('stepSize')) if lot else 0.0
        min_qty = float(lot.get('minQty')) if lot else 0.0
        qty_prec = int(ex.get('quantityPrecision', 0))
        min_not = float(min_notional.get('notional')) if (min_notional and 'notional' in min_notional) else 0.0

        if sym not in cfg_all: cfg_all[sym] = {}
        tgt = cfg_all[sym]

        # Tulis hanya jika belum ada (atau kosong)
        if float(tgt.get('stepSize', 0) or 0) == 0.0:
            tgt['stepSize'] = step_size; changed = True
        if float(tgt.get('minQty', 0) or 0) == 0.0:
            tgt['minQty'] = min_qty; changed = True
        if int(tgt.get('quantityPrecision', 0) or 0) == 0:
            tgt['quantityPrecision'] = qty_prec; changed = True
        if float(tgt.get('minNotional', 0) or 0) == 0.0 and min_not > 0:
            tgt['minNotional'] = min_not; changed = True

        # default biaya/slippage bila kosong (boleh kamu set ENV)
        if 'taker_fee' not in tgt: tgt['taker_fee'] = _to_float(os.getenv('TAKER_FEE', '0.0005'), 0.0005); changed = True
        if 'SLIPPAGE_PCT' not in tgt: tgt['SLIPPAGE_PCT'] = _to_float(os.getenv('SLIPPAGE_PCT', '0.02'), 0.02); changed = True

    if changed:
        with open(coin_cfg_path, 'w') as f:
            json.dump(cfg_all, f, indent=2)
        print(f"[INFO] coin_config.json diupdate dengan filter exchange untuk: {', '.join(symbols)}")
    return cfg_all

# -----------------------------
# CSV Journal
# -----------------------------
class Journal:
    def __init__(self, logs_dir: str, cfg_by_sym: Dict[str, dict], start_balance: float):
        self.dir = logs_dir
        os.makedirs(self.dir, exist_ok=True)
        self.cfg = cfg_by_sym
        self.balance = {s: float(start_balance) for s in cfg_by_sym.keys()}
        self.open = {}  # symbol -> dict

    def _csv_path(self, symbol: str) -> str:
        return os.path.join(self.dir, f"{symbol}_paper_trades.csv")

    def _ensure_header(self, path: str):
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            with open(path, 'w', newline='') as f:
                w = csv.writer(f)
                w.writerow(["timestamp","symbol","side","entry","qty","sl_init","tsl_init",
                            "exit_price","exit_reason","pnl_usdt","roi_pct","balance_after"])

    def on_entry(self, symbol: str, side: str, entry: float, qty: float, sl_init: Optional[float], tsl_init: Optional[float]):
        self.open[symbol] = {
            "ts": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S%z"),
            "side": side, "entry": float(entry), "qty": float(qty),
            "sl_init": float(sl_init) if sl_init is not None else "",
            "tsl_init": float(tsl_init) if tsl_init is not None else ""
        }

    def on_exit(self, symbol: str, exit_price: float, reason: str):
        if symbol not in self.open:
            return
        o = self.open.pop(symbol)
        bal_before = self.balance.get(symbol, 0.0)
        taker_fee = _to_float(self.cfg[symbol].get("taker_fee", 0.0005), 0.0005)
        slip_pct = _to_float(self.cfg[symbol].get("SLIPPAGE_PCT", 0.02), 0.02) / 100.0

        side = o["side"]; qty = o["qty"]; entry = o["entry"]
        # PnL koin USDT-M futures (qty dalam coin):
        if side == "LONG":
            gross = (exit_price - entry) * qty
        else:
            gross = (entry - exit_price) * qty

        # Biaya & slippage model sederhana (kedua sisi):
        fees = taker_fee * (entry + exit_price) * qty
        slipp = slip_pct * (entry + exit_price) * qty
        pnl = gross - fees - slipp

        bal_after = bal_before + pnl
        roi_pct = (pnl / bal_before * 100.0) if bal_before > 0 else 0.0
        self.balance[symbol] = bal_after

        path = self._csv_path(symbol)
        self._ensure_header(path)
        with open(path, 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([
                o["ts"], symbol, side, f"{entry:.6f}", f"{qty:.6f}",
                o["sl_init"], o["tsl_init"], f"{exit_price:.6f}",
                reason, f"{pnl:.4f}", f"{roi_pct:.2f}", f"{bal_after:.4f}"
            ])

# -----------------------------
# Trader yg di-journal
# -----------------------------
class JournaledCoinTrader(CoinTrader):
    def __init__(self, symbol: str, config: dict, journal: Journal):
        super().__init__(symbol, config)
        self.journal = journal

    def _size_position(self, price: float, balance: float) -> float:
        qty = super()._size_position(price, balance)
        # Enforce minNotional bila ada
        try:
            min_not = _to_float(self.config.get("minNotional", 0.0), 0.0)
            step = _to_float(self.config.get("stepSize", 0.0), 0.0)
            if min_not > 0 and price > 0:
                min_qty_by_notional = min_not / price
                if qty * price < min_not:
                    qty = max(qty, min_qty_by_notional)
                    qty = floor_to_step(qty, step) if step > 0 else qty
        except Exception:
            pass
        return qty

    def _enter_position(self, side: str, price: float, atr: float, balance: float) -> None:
        before_qty = getattr(self.pos, "qty", 0.0) or 0.0
        super()._enter_position(side, price, atr, balance)
        # Jika benar-benar masuk (qty > 0 & ada side)
        if self.pos.side and self.pos.qty > 0 and self.pos.qty != before_qty:
            self.journal.on_entry(
                self.symbol, self.pos.side, self.pos.entry if self.pos.entry is not None else 0.0, self.pos.qty,
                self.pos.sl, self.pos.trailing_sl
            )

    def _exit_position(self, price: float, reason: str) -> None:
        # simpan dulu utk CSV
        if self.pos.side:
            self.journal.on_exit(self.symbol, price, reason)
        super()._exit_position(price, reason)

# -----------------------------
# Live runner
# -----------------------------
def build_df_from_klines(kl: List[list]) -> pd.DataFrame:
    cols = ["open_time","open","high","low","close","volume","close_time",
            "qav","num_trades","tbbav","tbqav","ignore"]
    d = pd.DataFrame(kl, columns=cols)
    d["open"] = d["open"].astype(float)
    d["high"] = d["high"].astype(float)
    d["low"] = d["low"].astype(float)
    d["close"] = d["close"].astype(float)
    d["volume"] = d["volume"].astype(float)
    # pakai close_time sebagai timestamp bar
    d["timestamp"] = pd.to_datetime(d["close_time"], unit="ms", utc=True)
    return d[["timestamp","open","high","low","close","volume"]]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-paper", action="store_true", help="Mode live paper (tanpa order)")
    ap.add_argument("--symbols", required=True, help="Comma separated, e.g. ADAUSDT,DOGEUSDT")
    ap.add_argument("--interval", default="15m", choices=list(INTERVAL_MAP.keys()))
    ap.add_argument("--balance", type=float, default=20.0, help="Start balance per symbol (USDT)")
    ap.add_argument("--coin_config", default="coin_config.json")
    ap.add_argument("--logs_dir", default="logs")
    ap.add_argument("--risk_pct", type=float, default=None, help="Override risk_per_trade (contoh: 0.01)")
    ap.add_argument("--limit_bars", type=int, default=600, help="Kline limit pull (<= 1500)")
    ap.add_argument("--timeout", type=int, default=20, help="HTTP timeout seconds")
    ap.add_argument("--retries", type=int, default=6, help="HTTP retries")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("VERBOSE", "1" if args.verbose else "0")

    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    interval_key = INTERVAL_MAP[args.interval]
    int_s = interval_seconds(args.interval)

    bnc = BinanceFutures(requests_timeout=args.timeout, max_retries=args.retries)
    cfg_all = ensure_filters_to_coin_config(bnc, args.coin_config, syms)

    # Build trader per symbol
    cfg_by_sym = {}
    traders: Dict[str, JournaledCoinTrader] = {}
    for s in syms:
        merged = merge_config(s, cfg_all)
        if args.risk_pct is not None:
            merged["risk_per_trade"] = float(args.risk_pct)
        cfg_by_sym[s] = merged

    journal = Journal(args.logs_dir, cfg_by_sym, args.balance)
    for s in syms:
        traders[s] = JournaledCoinTrader(s, cfg_by_sym[s], journal)

    print(f"[SYSTEM] Live paper feed start symbols={','.join(syms)} interval={args.interval}")

    # Loop live
    last_bar_ts: Dict[str, Optional[pd.Timestamp]] = {s: None for s in syms}
    while True:
        # Ambil data per simbol
        for s in syms:
            try:
                raw = bnc.klines(s, interval_key, limit=min(max(args.limit_bars, 200), 1500))
                df = build_df_from_klines(raw)
                # proses hanya jika ada bar baru
                last_ts = df["timestamp"].iloc[-1]
                if last_bar_ts[s] is not None and last_ts <= last_bar_ts[s]:
                    # belum ada bar close baru, lanjut simbol lain
                    continue
                last_bar_ts[s] = last_ts
                # kirim ke engine
                traders[s].check_trading_signals(df, journal.balance[s])
            except Exception as e:
                print(f"[ERROR] live loop {s}: {e}")

        # tidur sampai close bar berikutnya
        sleep_s = next_close_sleep(int_s, skew=1.0)
        if sleep_s > 0:
            time.sleep(sleep_s)

if __name__ == "__main__":
    if not _to_bool(os.getenv("BINANCE_FORCE_PUBLIC_ONLY", "1"), True):
        # (opsional) Masukkan API KEY kalau mau private call lain — untuk feed klines tidak wajib.
        Client(os.getenv("BINANCE_API_KEY",""), os.getenv("BINANCE_API_SECRET",""))
    try:
        main()
    except KeyboardInterrupt:
        print("\n[EXIT] Stopped by user")
