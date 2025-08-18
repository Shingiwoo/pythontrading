# newrealtrading.py — FULL PATCH (revised)
# =============================================================
# Update: 2025-08-16
# - Melengkapi ekspor simbol supaya `from newrealtrading import ...` dikenali Pylance/IDE
# - Tetap kompatibel dengan papertrade.py (live paper feed / tanpa order)
# - Struktur modular: CoinTrader, TradingManager, utils, indikator, loader config, ML plugin hook
# =============================================================
from __future__ import annotations
import os, json, time, math, threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd
import numpy as np

# --- TA
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.momentum import RSIIndicator

# --- ML plugin (pastikan file ml_signal_plugin.py ada 1 folder dengan file ini)
from ml_signal_plugin import MLSignal

# (opsional) dotenv agar ENV dari .env terbaca jika ada
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from engine_core import (
    _to_float, _to_int, _to_bool, floor_to_step,
    load_coin_config, merge_config, compute_indicators as calculate_indicators,
    htf_trend_ok
)

__all__ = [
    "CoinTrader",
    "TradingManager",
    "floor_to_step",
    "_to_float",
    "_to_bool",
    "load_coin_config",
    "merge_config",
    "calculate_indicators",
]

# ============================
# Config
# ============================
DEFAULTS = {
    "leverage": 15,
    "risk_per_trade": 0.08,
    "taker_fee": 0.0005,  # fraksi per sisi
    "min_atr_pct": 0.006,
    "max_atr_pct": 0.03,
    "max_body_atr": 0.95,
    "use_htf_filter": 1,
    "cooldown_seconds": 1500,
    # SL / BE / Trailing
    "sl_mode": "ATR",
    "sl_pct": 0.008,
    "sl_atr_mult": 1.6,
    "sl_min_pct": 0.010,
    "sl_max_pct": 0.030,
    "use_breakeven": 1,
    "be_trigger_pct": 0.0045,  # fraksi (0.45%)
    "trailing_trigger": 0.7,   # %
    "trailing_step": 0.45,     # %
    "max_hold_seconds": 3600,
    "min_roi_to_close_by_time": 0.005,
    # lot size & precision (opsional, jika tersedia dari exchange info)
    "stepSize": 0.0,
    "minQty": 0.0,
    "quantityPrecision": 0,
}

ENV_DEFAULTS = {
    "SLIPPAGE_PCT": 0.02,  # % per sisi
    "SCORE_THRESHOLD": 1.2,
}

# ============================
# Loader coin_config.json
# ============================

# ============================
# Trader per-coin
# ============================
@dataclass
class Position:
    side: Optional[str] = None   # 'LONG' | 'SHORT' | None
    entry: Optional[float] = None
    qty: float = 0.0
    sl: Optional[float] = None
    trailing_sl: Optional[float] = None
    entry_time: Optional[pd.Timestamp] = None


class CoinTrader:
    def __init__(self, symbol: str, config: Dict[str, Any]):
        self.symbol = symbol.upper()
        self.config = config
        self.ml = MLSignal(self.config)
        self.pos = Position()
        self.cooldown_until_ts: Optional[float] = None
        self.verbose = _to_bool(self.config.get('VERBOSE', os.getenv('VERBOSE','0')), False)

    def _log(self, msg: str) -> None:
        if getattr(self, 'verbose', False):
            print(f"[{self.symbol}] {pd.Timestamp.utcnow().isoformat()} | {msg}")

    # Hook: ganti dengan sumber data live kamu
    def fetch_recent_klines(self) -> pd.DataFrame:
        """Return DataFrame dengan kolom: timestamp, open, high, low, close, volume"""
        raise NotImplementedError("Implement fetch_recent_klines() sesuai exchange kamu.")

    def _safe_trailing_params(self) -> Tuple[float, float]:
        taker_fee = _to_float(self.config.get('taker_fee', DEFAULTS['taker_fee']), DEFAULTS['taker_fee'])
        slippage_pct = _to_float(self.config.get('SLIPPAGE_PCT', ENV_DEFAULTS['SLIPPAGE_PCT']), ENV_DEFAULTS['SLIPPAGE_PCT'])
        roundtrip_fee_pct = taker_fee * 2.0 * 100.0
        roundtrip_slip_pct = slippage_pct * 2.0
        safe_buffer_pct = roundtrip_fee_pct + roundtrip_slip_pct + 0.05
        trailing_trigger = _to_float(self.config.get('trailing_trigger', DEFAULTS['trailing_trigger']), DEFAULTS['trailing_trigger'])
        trailing_step = _to_float(self.config.get('trailing_step', DEFAULTS['trailing_step']), DEFAULTS['trailing_step'])
        safe_trigger = max(trailing_trigger, safe_buffer_pct + trailing_step)
        return safe_trigger, trailing_step

    def _size_position(self, price: float, balance: float) -> float:
        risk = _to_float(self.config.get('risk_per_trade', DEFAULTS['risk_per_trade']), DEFAULTS['risk_per_trade'])
        lev = _to_int(self.config.get('leverage', DEFAULTS['leverage']), DEFAULTS['leverage'])
        raw_qty = (balance * risk * lev) / max(price, 1e-12)
        step = _to_float(self.config.get('stepSize', 0.0), 0.0)
        q = floor_to_step(raw_qty, step) if step>0 else raw_qty
        prec = _to_int(self.config.get('quantityPrecision', 0), 0)
        try:
            q = float(f"{q:.{prec}f}")
        except Exception:
            q = float(q)
        minq = _to_float(self.config.get('minQty', 0.0), 0.0)
        return q if q >= minq else 0.0

    def _hard_sl_price(self, entry: float, atr: float, side: str) -> float:
        mode = str(self.config.get('sl_mode', DEFAULTS['sl_mode'])).upper()
        sl_min = _to_float(self.config.get('sl_min_pct', DEFAULTS['sl_min_pct']), DEFAULTS['sl_min_pct'])
        sl_max = _to_float(self.config.get('sl_max_pct', DEFAULTS['sl_max_pct']), DEFAULTS['sl_max_pct'])
        if mode == 'PCT':
            sl_pct = _to_float(self.config.get('sl_pct', DEFAULTS['sl_pct']), DEFAULTS['sl_pct'])
        else:
            sl_atr_mult = _to_float(self.config.get('sl_atr_mult', DEFAULTS['sl_atr_mult']), DEFAULTS['sl_atr_mult'])
            sl_pct = (sl_atr_mult * atr / max(entry, 1e-12)) if atr>0 else _to_float(self.config.get('sl_pct', DEFAULTS['sl_pct']), DEFAULTS['sl_pct'])
        sl_pct = max(sl_min, min(sl_pct, sl_max))
        return entry * (1 - sl_pct) if side=='LONG' else entry * (1 + sl_pct)

    def _apply_breakeven(self, price: float) -> None:
        if not _to_bool(self.config.get('use_breakeven', DEFAULTS['use_breakeven']), DEFAULTS['use_breakeven']):
            return
        be = _to_float(self.config.get('be_trigger_pct', DEFAULTS['be_trigger_pct']), DEFAULTS['be_trigger_pct'])
        if self.pos.side=='LONG' and self.pos.entry:
            if (price - self.pos.entry)/self.pos.entry >= be:
                self.pos.sl = max(self.pos.sl or 0.0, self.pos.entry)
        elif self.pos.side=='SHORT' and self.pos.entry:
            if (self.pos.entry - price)/self.pos.entry >= be:
                self.pos.sl = min(self.pos.sl or 1e18, self.pos.entry)

    def _update_trailing(self, price: float) -> None:
        safe_trigger, step = self._safe_trailing_params()
        if not (self.pos.entry and self.pos.side):
            return
        if self.pos.side=='LONG':
            profit_pct = (price - self.pos.entry)/self.pos.entry*100.0
            if profit_pct >= safe_trigger:
                new_ts = price * (1 - step/100.0)
                prev = self.pos.trailing_sl
                self.pos.trailing_sl = max(self.pos.trailing_sl or self.pos.sl or 0.0, new_ts)
                if self.pos.trailing_sl != prev:
                    self._log(f"TRAIL LONG -> {self.pos.trailing_sl:.6f} (prev={prev})")
        else:
            profit_pct = (self.pos.entry - price)/self.pos.entry*100.0
            if profit_pct >= safe_trigger:
                new_ts = price * (1 + step/100.0)
                prev = self.pos.trailing_sl
                self.pos.trailing_sl = min(self.pos.trailing_sl or self.pos.sl or 1e18, new_ts)
                if self.pos.trailing_sl != prev:
                    self._log(f"TRAIL SHORT -> {self.pos.trailing_sl:.6f} (prev={prev})")

    def _cooldown_active(self) -> bool:
        return bool(self.cooldown_until_ts and time.time() < self.cooldown_until_ts)

    def _enter_position(self, side: str, price: float, atr: float, balance: float) -> None:
        if self._cooldown_active():
            return
        qty = self._size_position(price, balance)
        if qty <= 0:
            return
        sl = self._hard_sl_price(price, atr, side)
        self.pos = Position(side=side, entry=price, qty=qty, sl=sl, trailing_sl=None, entry_time=pd.Timestamp.utcnow())
        self._log(f"ENTRY {side} price={price:.6f} qty={qty:.6f} sl={sl:.6f}")
        # TODO: place order ke exchange di sini
        # self.exchange.open_market_order(self.symbol, side, qty)

    def _should_exit(self, price: float) -> Tuple[bool, Optional[str]]:
        if not self.pos.side:
            return False, None
        # trailing/hard SL
        if self.pos.side=='LONG':
            if self.pos.trailing_sl is not None and price <= self.pos.trailing_sl:
                return True, 'Hit Trailing SL'
            if (self.pos.sl is not None) and price <= self.pos.sl:
                return True, 'Hit Hard SL'
        else:
            if self.pos.trailing_sl is not None and price >= self.pos.trailing_sl:
                return True, 'Hit Trailing SL'
            if (self.pos.sl is not None) and price >= self.pos.sl:
                return True, 'Hit Hard SL'
        return False, None

    def _exit_position(self, price: float, reason: str) -> None:
        # TODO: close position ke exchange
        # self.exchange.close_market_order(self.symbol, self.pos.side, self.pos.qty)
        self._log(f"EXIT {reason} price={price:.6f}")
        self.pos = Position()  # reset
        # cooldown
        cd = _to_int(self.config.get('cooldown_seconds', DEFAULTS['cooldown_seconds']), DEFAULTS['cooldown_seconds'])
        self.cooldown_until_ts = time.time() + max(cd, 0)
        try:
            self._log(f"COOLDOWN until {pd.Timestamp.utcfromtimestamp(self.cooldown_until_ts).isoformat()}")
        except Exception:
            pass

    def check_trading_signals(self, df_raw: pd.DataFrame, balance: float) -> None:
        if df_raw is None or df_raw.empty:
            return
        heikin = _to_bool(self.config.get('heikin', False), False)
        df = calculate_indicators(df_raw, heikin=heikin)
        last = df.iloc[-1]

        # Filter ATR & body/ATR
        atr_ok = (last['atr_pct'] >= _to_float(self.config.get('min_atr_pct', DEFAULTS['min_atr_pct']), DEFAULTS['min_atr_pct'])) \
                 and (last['atr_pct'] <= _to_float(self.config.get('max_atr_pct', DEFAULTS['max_atr_pct']), DEFAULTS['max_atr_pct']))

        # GUARD: pakai body_to_atr kalau ada, kalau tidak fallback ke body_atr
        body_val = last.get('body_to_atr', last.get('body_atr'))
        body_ok = (float(body_val) <= _to_float(self.config.get('max_body_atr', DEFAULTS['max_body_atr']), DEFAULTS['max_body_atr'])) if body_val is not None else False

        if not (atr_ok and body_ok):
            # kelola exit jika sudah di posisi
            if self.pos.side:
                self._apply_breakeven(last['close'])
                self._update_trailing(last['close'])
                ex, rs = self._should_exit(last['close'])
                if ex:
                    self._exit_position(last['close'], rs or 'Filter Fail')
            # log kenapa diblok (supaya --verbose selalu ada output)
            self._log(f"FILTER BLOCK atr_ok={atr_ok} body_ok={body_ok} price={float(last['close']):.6f} pos={self.pos.side}")
            return

        # HTF filter (opsional)
        if _to_bool(self.config.get('use_htf_filter', DEFAULTS['use_htf_filter']), DEFAULTS['use_htf_filter']):
            if last['ema']>last['ma'] and not htf_trend_ok('LONG', df):
                long_htf_ok = False
            else:
                long_htf_ok = True
            if last['ema']<last['ma'] and not htf_trend_ok('SHORT', df):
                short_htf_ok = False
            else:
                short_htf_ok = True
        else:
            long_htf_ok = short_htf_ok = True

        # Skor base
        long_base = bool(last.get('long_base', False)) and long_htf_ok
        short_base = bool(last.get('short_base', False)) and short_htf_ok

        # ML
        up_prob = None
        if self.ml.use_ml:
            self.ml.fit_if_needed(df)
            up_prob = self.ml.predict_up_prob(df)
        ml_cfg = self.config.get("ml", {}) if isinstance(self.config.get("ml"), dict) else {}
        if "score_threshold" in ml_cfg:
            self.ml.params.score_threshold = _to_float(ml_cfg.get("score_threshold"), ENV_DEFAULTS["SCORE_THRESHOLD"])
        else:
            self.ml.params.score_threshold = _to_float(self.config.get("SCORE_THRESHOLD", ENV_DEFAULTS["SCORE_THRESHOLD"]), ENV_DEFAULTS["SCORE_THRESHOLD"])
        long_sig, short_sig = self.ml.score_and_decide(long_base, short_base, up_prob)

        # Kelola posisi
        price = float(last['close']); atr = float(last['atr']) if not pd.isna(last['atr']) else 0.0
        self._log(f"DECISION price={price:.6f} base(L={long_base},S={short_base}) up_prob={(up_prob if up_prob is not None else 'n/a')} thr={self.ml.params.score_threshold} -> {('LONG' if long_sig else ('SHORT' if short_sig else '-'))} pos={self.pos.side}")
        # Update SL/TS saat pegang posisi
        if self.pos.side:
            self._apply_breakeven(price)
            self._update_trailing(price)
            ex, reason = self._should_exit(price)
            if ex:
                self._exit_position(price, reason or 'Exit')
                return

        # ---- Time-stop (selaras backtest) ----
        max_hold = _to_int(self.config.get("max_hold_seconds", DEFAULTS.get("max_hold_seconds", 3600)), 3600)
        min_roi  = _to_float(self.config.get("min_roi_to_close_by_time", DEFAULTS.get("min_roi_to_close_by_time", 0.005)), 0.005)

        if self.pos.side and self.pos.entry_time and max_hold > 0:
            elapsed = (pd.Timestamp.utcnow() - self.pos.entry_time).total_seconds()
            lev = _to_int(self.config.get("leverage", DEFAULTS["leverage"]), DEFAULTS["leverage"])
            if self.pos.entry is not None and self.pos.qty is not None:
                init_margin = (self.pos.entry * self.pos.qty) / max(lev, 1)
                roi_frac = 0.0
                if init_margin > 0:
                    if self.pos.side == "LONG":
                        roi_frac = ((price - self.pos.entry) * self.pos.qty) / init_margin
                    else:
                        roi_frac = ((self.pos.entry - price) * self.pos.qty) / init_margin

                if elapsed >= max_hold and roi_frac >= min_roi:
                    self._exit_position(price, f"Max hold reached (ROI {roi_frac*100:.2f}%)")
                    return
                elif elapsed >= max_hold:
                    self.pos.entry_time = pd.Timestamp.utcnow()
        # --------------------------------------

        # Entry baru
        if not self.pos.side and not self._cooldown_active():
            if long_sig:
                self._enter_position('LONG', price, atr, balance)
            elif short_sig:
                self._enter_position('SHORT', price, atr, balance)


# ============================
# Manager (contoh sederhana)
# ============================
class TradingManager:
    def __init__(self, coin_config_path: str, symbols: List[str]):
        self.coin_config_path = coin_config_path
        self.symbols = [s.upper() for s in symbols]
        self._cfg = load_coin_config(coin_config_path)
        self.traders: Dict[str, CoinTrader] = {s: CoinTrader(s, merge_config(s, self._cfg)) for s in self.symbols}
        self._stop = False
        # watcher config pada thread terpisah
        t = threading.Thread(target=self._watch_config, daemon=True)
        t.start()

    def _watch_config(self):
        last_ts = 0.0
        safe_keys = {
            "risk_per_trade","leverage","trailing_trigger","trailing_step","taker_fee",
            "min_atr_pct","max_atr_pct","max_body_atr","use_htf_filter","cooldown_seconds",
            "allow_sar","reverse_confirm_bars","min_hold_seconds","max_hold_seconds","min_roi_to_close_by_time",
            # ML & scoring
            "USE_ML","SCORE_THRESHOLD","ML_MIN_TRAIN_BARS","ML_LOOKAHEAD",
            "ML_RETRAIN_EVERY","ML_UP_PROB","ML_DOWN_PROB",
            # precision
            "stepSize","minQty","quantityPrecision"
        }
        while not self._stop:
            try:
                ts = os.path.getmtime(self.coin_config_path)
                if ts != last_ts:
                    last_ts = ts
                    cfg_all = load_coin_config(self.coin_config_path)
                    for sym, trader in self.traders.items():
                        new_cfg = merge_config(sym, cfg_all)
                        # hanya update key aman
                        for k, v in list(new_cfg.items()):
                            if k in safe_keys:
                                trader.config[k] = v
                        # sinkron threshold ke plugin
                        ml_cfg = trader.config.get("ml", {}) if isinstance(trader.config.get("ml"), dict) else {}
                        if "score_threshold" in ml_cfg:
                            trader.ml.params.score_threshold = _to_float(ml_cfg.get("score_threshold"), ENV_DEFAULTS["SCORE_THRESHOLD"])
                        else:
                            trader.ml.params.score_threshold = _to_float(trader.config.get("SCORE_THRESHOLD", ENV_DEFAULTS["SCORE_THRESHOLD"]), ENV_DEFAULTS["SCORE_THRESHOLD"])
                time.sleep(1.0)
            except Exception:
                time.sleep(2.0)

    # Hook: implement loop fetch + dispatch ke trader
    def run_once(self, data_map: Dict[str, pd.DataFrame], balance_by_sym: Dict[str, float]):
        for sym, trader in self.traders.items():
            df = data_map.get(sym)
            bal = balance_by_sym.get(sym, 20.0)
            if isinstance(df, pd.DataFrame) and len(df)>0:
                trader.check_trading_signals(df, bal)

    def stop(self):
        self._stop = True


# ============================
# Contoh penggunaan (dummy)
# ============================
if __name__ == "__main__":
    # contoh minimal: jalankan 1x pakai CSV lokal untuk verifikasi logika
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--coin_config", default="coin_config.json")
    ap.add_argument("--csv", required=False, help="Path CSV OHLCV (timestamp, open, high, low, close, volume)")
    ap.add_argument("--symbol", default="ADAUSDT")
    ap.add_argument("--balance", type=float, default=20.0)
    ap.add_argument("--verbose", action="store_true", help="Print log keputusan & aksi")
    ap.add_argument("--dry-run-loop", action="store_true", help="Replay CSV bar-by-bar (simulasi real-time)")
    ap.add_argument("--sleep", type=float, default=0.0, help="Delay per step (detik) saat dry-run")
    ap.add_argument("--limit", type=int, default=0, help="Batasi jumlah langkah dry-run (0 = semua)")
    args = ap.parse_args()
    if args.verbose:
        os.environ["VERBOSE"] = "1"

    cfg_all = load_coin_config(args.coin_config) if os.path.exists(args.coin_config) else {}
    cfg = merge_config(args.symbol.upper(), cfg_all)

    mgr = TradingManager(args.coin_config, [args.symbol])

    if args.csv and os.path.exists(args.csv):
        df = pd.read_csv(args.csv)
        if 'timestamp' not in df.columns:
            if 'open_time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', errors='coerce')
            elif 'date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        sym = args.symbol.upper()

        if args.dry_run_loop:
            # warmup agar indikator & ML cukup data
            try:
                min_train = int(float(os.getenv('ML_MIN_TRAIN_BARS', '400')))
            except Exception:
                min_train = 400
            warmup = max(300, min_train + 10)
            start_i = min(warmup, len(df)-1)
            steps = 0
            mgr = TradingManager(args.coin_config, [sym])  # <-- PAKAI MANAGER YANG SAMA (state posisi tersimpan)
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
        print("Tidak ada CSV. Mode dummy selesai.")
