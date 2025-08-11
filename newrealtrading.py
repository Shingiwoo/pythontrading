"""
newrealtrading.py — presisi entri v3.1 (Hard SL + Breakeven + Anti-SAR Whipsaw)

Tambahan utama v3.1:
- Hard Stop Loss (mode PCT atau ATR) dihitung saat ENTRY, disimpan ke state, dan dicek real-time.
- Breakeven opsional: geser SL ke harga masuk saat profit >= be_trigger_pct.
- Filter presisi entri v2 tetap aktif: ATR regime, hindari candle kepanjangan, konfirmasi tren 1H, cooldown.
- **Anti-SAR Whipsaw**: default **tidak** langsung menutup posisi hanya karena sinyal berlawanan muncul. Opsi Stop-And-Reverse (SAR) bisa diaktifkan dengan konfirmasi beberapa bar dan minimal waktu hold.
- Perbaikan ReduceOnly -2022 (abaikan jika posisi memang sudah 0).
- Fitur lama tetap: SYMBOLS via ENV, logging ramah read-only, WS loop recv(), leverage await, residual-close fix.
"""

import os
import json
import asyncio
import time
import math
import logging
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from binance import AsyncClient, BinanceSocketManager
from binance.enums import ORDER_TYPE_MARKET

load_dotenv()

# ===== Logging (ramah read-only) =====
LOG_DIR = os.getenv("LOG_DIR", "/app/logs")
LOG_FILE = os.getenv("LOG_FILE", "trading_bot.log")
handlers = []
try:
    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, LOG_FILE)
    handlers.append(logging.FileHandler(log_path))
except Exception:
    handlers.append(logging.FileHandler("/tmp/trading_bot.log"))
handlers.append(logging.StreamHandler())

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=handlers
)
logger = logging.getLogger('binance_future_scalping')

# ===== Konfigurasi umum =====
CONFIG_FILE = "coin_config.json"
TIMEFRAME = os.getenv("TIMEFRAME", "15m")

DEFAULT_SYMBOLS = ['XRPUSDT', 'DOGEUSDT', 'TURBOUSDT']
env_syms = os.getenv("SYMBOLS", "")
SYMBOLS = [s.strip().upper() for s in env_syms.split(",") if s.strip()] or DEFAULT_SYMBOLS

USE_MULTIPLEX = bool(int(os.getenv("USE_MULTIPLEX", "1")))
WS_MAX_RETRY = int(os.getenv("WS_MAX_RETRY", "0"))  # 0=tak terbatas

TAKER_FEE_DEFAULT = float(os.getenv('TAKER_FEE', '0.0005'))
MAKER_FEE_DEFAULT = float(os.getenv('MAKER_FEE', '0.0002'))
MIN_ROI_TO_CLOSE_BY_TIME_DEFAULT = float(os.getenv('MIN_ROI_TO_CLOSE_BY_TIME', '0.05'))
MAX_HOLD_SECONDS_DEFAULT = int(os.getenv('MAX_HOLD_SECONDS', '3600'))

STATE_FILE = os.getenv('STATE_FILE', '/data/active_position_state.json')
EXCHANGEINFO_CACHE = os.getenv('EXCHANGEINFO_CACHE', '/data/exchange_info_cache.json')
EXCHANGEINFO_TTL_SECONDS = int(os.getenv('EXCHANGEINFO_TTL_SECONDS', '21600'))  # 6 jam

# ===== Utilities =====

def _extract_filled_qty(order: Dict[str, Any]) -> float:
    """Ambil qty terisi dari berbagai format respons order Futures Binance."""
    try:
        for k in ("executedQty", "cumQty"):
            if k in order and order[k] is not None:
                return float(order[k])
        if "origQty" in order and order["origQty"] is not None:
            return float(order["origQty"])
        fills = order.get("fills") or []
        if fills:
            total = 0.0
            for f in fills:
                v = f.get("qty") or f.get("executionQty") or 0
                try:
                    total += float(v)
                except Exception:
                    pass
            if total > 0:
                return total
    except Exception:
        pass
    return 0.0

def floor_to_step(x: float, step: float) -> float:
    if not step or step <= 0:
        return float(x)
    return math.floor(float(x)/float(step))*float(step)

async def _send_telegram(text: str):
    import aiohttp
    token = os.getenv('TELEGRAM_BOT_TOKEN')
    chat_id = os.getenv('TELEGRAM_CHAT_ID')
    if not token or not chat_id:
        return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, data=payload) as resp:
                await resp.text()
    except Exception as e:
        logger.error(f"Telegram send failed: {e}")

async def _load_cached_exchange_info():
    try:
        if os.path.exists(EXCHANGEINFO_CACHE):
            with open(EXCHANGEINFO_CACHE, 'r') as f:
                data = json.load(f)
            if time.time() - data.get('ts', 0) <= EXCHANGEINFO_TTL_SECONDS:
                return data.get('info')
    except Exception as e:
        logger.warning(f"Read exchangeInfo cache failed: {e}")
    return None

async def _save_cached_exchange_info(info: Dict[str, Any]):
    try:
        os.makedirs(os.path.dirname(EXCHANGEINFO_CACHE), exist_ok=True)
        with open(EXCHANGEINFO_CACHE, 'w') as f:
            json.dump({'ts': time.time(), 'info': info}, f)
    except Exception as e:
        logger.warning(f"Write exchangeInfo cache failed: {e}")

async def _get_exchange_filters(client: AsyncClient, symbol: str) -> Dict[str, Any]:
    info = await _load_cached_exchange_info()
    if not info:
        info = await client.futures_exchange_info()
        await _save_cached_exchange_info(info)
    s = next((item for item in info['symbols'] if item['symbol'] == symbol), None)
    if not s:
        raise RuntimeError(f"Symbol {symbol} not found in exchangeInfo")
    quantity_precision = s.get('quantityPrecision', 0)
    price_precision = s.get('pricePrecision', 0)

    step = 0.0
    min_qty = 0.0
    min_notional = 0.0
    tick_size = 0.0

    for f in s['filters']:
        t = f.get('filterType')
        if t in ('MARKET_LOT_SIZE', 'LOT_SIZE'):
            step = float(f.get('stepSize', step or '0'))
            min_qty = float(f.get('minQty', min_qty or '0'))
        if t in ('NOTIONAL', 'MIN_NOTIONAL', 'MARKET_MIN_NOTIONAL'):
            min_notional = float(f.get('minNotional', f.get('notional', min_notional or '0')))
        if t == 'PRICE_FILTER':
            tick_size = float(f.get('tickSize', tick_size or '0'))

    return {
        'step': step,
        'min_qty': min_qty,
        'min_notional': min_notional,
        'quantity_precision': quantity_precision,
        'price_precision': price_precision,
        'tick_size': tick_size
    }

def _normalize_qty(qty: float, price: float, f: Dict[str, Any]) -> float:
    if qty is None or qty <= 0 or price is None or price <= 0:
        return 0.0
    step = float(f.get('step', 0.0))
    min_qty = float(f.get('min_qty', 0.0))
    qprec = int(f.get('quantity_precision', 0))
    min_notional = float(f.get('min_notional', 0.0))

    q = float(qty)
    if step and step > 0:
        q = math.floor(q / step) * step
    try:
        q = float(f"{q:.{qprec}f}")
    except Exception:
        pass
    if min_qty and q < min_qty:
        q = min_qty
    if min_notional and (q * price) < min_notional:
        needed = min_notional / price
        if step and step > 0:
            needed = math.ceil(needed / step) * step
        try:
            needed = float(f"{needed:.{qprec}f}")
        except Exception:
            pass
        if min_qty and needed < min_qty:
            needed = min_qty
        q = needed
    return q

# ===== Persistence =====

def _save_state(state: dict):
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, default=str)
    except Exception as e:
        logger.warning(f"save state failed: {e}")

def _load_state() -> dict:
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"load state failed: {e}")
    return {}

# ===== Trader =====
class CoinTrader:
    def __init__(self, symbol: str, api_key: str, api_secret: str, capital_allocated: float, config: dict):
        self.symbol = symbol
        self.api_key = api_key
        self.api_secret = api_secret
        self.capital = capital_allocated
        self.config = config

        self.data = pd.DataFrame(columns=['timestamp','open','high','low','close','volume'])

        # state posisi
        self.in_position = False
        self.position_type: Optional[str] = None
        self.entry_price: Optional[float] = None
        self.stop_loss: Optional[float] = None
        self.take_profit: Optional[float] = None
        self.trailing_stop: Optional[float] = None
        self.position_size: float = 0.0
        self.hold_start_ts: Optional[float] = None
        self.client: Optional[AsyncClient] = None

        # presisi entri
        self.cooldown_until: float = 0.0
        self._htf_cache = None
        self._htf_cache_ts = 0.0

        # anti whipsaw / reversal guard
        self.opp_sig_count: int = 0  # jumlah bar berturut-turut sinyal berlawanan

    async def require_client(self) -> AsyncClient:
        if self.client is None:
            self.client = await AsyncClient.create(self.api_key, self.api_secret)
        return self.client

    async def _get_mark_price(self) -> Optional[float]:
        try:
            client = await self.require_client()
            mp = await client.futures_mark_price(symbol=self.symbol)
            return float(mp['markPrice'])
        except Exception as e:
            logger.error(f"{self.symbol} get mark price failed: {e}")
            return None

    async def _get_position_amt(self) -> float:
        try:
            client = await self.require_client()
            info = await client.futures_position_information(symbol=self.symbol)
            for pos in info:
                if pos.get('symbol') == self.symbol:
                    return abs(float(pos.get('positionAmt', 0) or 0))
        except Exception as e:
            logger.error(f"{self.symbol} get position info failed: {e}")
        return 0.0

    async def initialize(self):
        if self.client is None:
            self.client = await AsyncClient.create(self.api_key, self.api_secret)

        # Recovery posisi aktif
        st = _load_state()
        if st.get('symbol') == self.symbol and st.get('in_position'):
            self.in_position = True
            self.position_type = st.get('position_type')
            self.entry_price = float(st.get('entry_price', 0))
            self.position_size = float(st.get('position_size', 0))
            self.stop_loss = st.get('stop_loss')
            self.take_profit = st.get('take_profit')
            self.trailing_stop = st.get('trailing_stop')
            self.hold_start_ts = st.get('hold_start_ts')
            logger.info(f"Recovered active position from state: {self.position_type} {self.position_size} @ {self.entry_price}")
            try:
                asyncio.create_task(_send_telegram(f"♻️ <b>{self.symbol}</b> posisi aktif direstore: {self.position_type} {self.position_size} @ {self.entry_price}"))
            except Exception:
                pass

        # set leverage (sesuai config)
        try:
            client = await self.require_client()
            await client.futures_change_leverage(symbol=self.symbol, leverage=int(self.config['leverage']))
        except Exception as e:
            logger.error(f"{self.symbol} change leverage failed on init: {e}")

        # load data awal
        try:
            client = await self.require_client()
            klines = await client.futures_klines(symbol=self.symbol, interval=TIMEFRAME, limit=1000)
            rows = []
            for k in klines:
                rows.append({'timestamp':k[0],'open':float(k[1]),'high':float(k[2]),'low':float(k[3]),'close':float(k[4]),'volume':float(k[5])})
            self.data = pd.DataFrame(rows)
            logger.info(f"{self.symbol} initialized with {len(self.data)} candles")
        except Exception as e:
            logger.warning(f"{self.symbol} init klines failed: {e}")

    async def apply_config_update(self, new_cfg: dict, changed: dict):
        self.config.update({k:new_cfg.get(k, self.config.get(k)) for k in new_cfg.keys()})
        if 'leverage' in changed:
            try:
                client = await self.require_client()
                await client.futures_change_leverage(symbol=self.symbol, leverage=int(self.config['leverage']))
                try:
                    asyncio.create_task(_send_telegram(f"⚙️ <b>{self.symbol}</b> leverage diubah -> {self.config['leverage']}"))
                except Exception:
                    pass
                logger.info(f"{self.symbol} leverage updated to {self.config['leverage']} via hot-reload")
            except Exception as e:
                logger.error(f"{self.symbol} change leverage failed: {e}")
        other = {k:v for k,v in changed.items() if k!='leverage'}
        if other:
            try:
                msg = " | ".join([f"{k}:{v[0]}→{v[1]}" for k,v in other.items()])
                asyncio.create_task(_send_telegram(f"♻️ <b>{self.symbol}</b> hot-reload: {msg}"))
            except Exception:
                pass
        return True

    def calculate_indicators(self) -> pd.DataFrame:
        df = self.data.copy()
        if df.empty:
            return df
        # indikator dasar
        df['ema'] = df['close'].ewm(span=22, adjust=False).mean()
        df['ma'] = df['close'].rolling(20, min_periods=20).mean()
        macd_line = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        rs = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean() / (loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean() + 1e-12)
        df['rsi'] = 100 - (100/(1+rs))
        # ATR & rasio body
        prev_close = df['close'].shift(1)
        tr = (df['high'] - df['low']).to_frame('a')
        tr['b'] = (df['high'] - prev_close).abs()
        tr['c'] = (df['low'] - prev_close).abs()
        df['tr'] = tr.max(axis=1)
        df['atr'] = df['tr'].ewm(alpha=1/14, adjust=False, min_periods=14).mean()
        df['body'] = (df['close'] - df['open']).abs()
        df['atr_pct'] = df['atr'] / df['close']
        df['body_to_atr'] = df['body'] / df['atr']
        # sinyal dasar
        df['long_signal'] = (df['ema']>df['ma']) & (df['macd']>df['macd_signal']) & (df['rsi'].between(40,70))
        df['short_signal'] = (df['ema']<df['ma']) & (df['macd']<df['macd_signal']) & (df['rsi'].between(30,60))
        return df

    async def _htf_trend_ok(self, side: str) -> bool:
        # cache 30 menit
        if time.time() - (self._htf_cache_ts or 0) > 1800 or self._htf_cache is None:
            client = await self.require_client()
            h1 = await client.futures_klines(symbol=self.symbol, interval='1h', limit=300)
            close = pd.Series([float(k[4]) for k in h1])
            ema50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
            ema200 = close.ewm(span=200, adjust=False).mean().iloc[-1]
            self._htf_cache = {'ema50': float(ema50), 'ema200': float(ema200)}
            self._htf_cache_ts = time.time()
        ema50 = self._htf_cache['ema50']; ema200 = self._htf_cache['ema200']
        if side == 'LONG':
            return ema50 >= ema200
        else:
            return ema50 <= ema200

    async def place_order(self, side: str, quantity: float, reduce_only: bool = False):
        client = await self.require_client()
        f = await _get_exchange_filters(client, self.symbol)

        price = await self._get_mark_price()
        if price is None:
            price = self.entry_price if self.entry_price else 0.0

        qty = _normalize_qty(quantity, price, f)
        if qty <= 0:
            logger.error(f"{self.symbol} qty invalid after normalize (qty={qty})")
            await _send_telegram(f"❌ <b>{self.symbol}</b> order gagal: qty invalid setelah normalize")
            return None, None, 0.0

        try:
            order = await client.futures_create_order(
                symbol=self.symbol, side=side, type=ORDER_TYPE_MARKET,
                quantity=qty, reduceOnly=reduce_only, newOrderRespType='RESULT'
            )
            filled = _extract_filled_qty(order)
            logger.info(f"Order executed: sent={qty}, filled={filled}, resp={order}")
            await _send_telegram(f"✅ <b>{self.symbol}</b> {side} sent={qty} filled={filled} @MARKET")
            return order, qty, filled
        except Exception as e:
            msg = str(e)
            if "-2022" in msg and reduce_only:
                logger.info(f"{self.symbol} reduceOnly rejected (no position) → ignore")
                return None, None, 0.0
            logger.error(f"Create order failed: {e}")
            await _send_telegram(f"❌ <b>{self.symbol}</b> order gagal: {e}")
            return None, None, 0.0

    async def _close_position_reduce_only(self, current_price: Optional[float] = None):
        pos_amt = await self._get_position_amt()
        if not self.in_position or pos_amt <= 0 or self.entry_price is None:
            return

        side = 'SELL' if self.position_type == 'LONG' else 'BUY'
        client = await self.require_client()
        f = await _get_exchange_filters(client, self.symbol)

        price = current_price
        if price is None:
            mp = await self._get_mark_price()
            price = mp if mp is not None else self.entry_price

        step = float(f.get('step', 0.0)) or 0.0
        min_qty = float(f.get('min_qty', 0.0)) or 0.0
        min_notional = float(f.get('min_notional', 0.0)) or 0.0

        def ceil_to_step(x, step):
            if not step or step <= 0:
                return float(x)
            return math.ceil(float(x)/float(step))*float(step)

        qty_req = ceil_to_step(pos_amt, step)
        if min_qty and qty_req < min_qty:
            qty_req = ceil_to_step(min_qty, step)
        if min_notional and price and (qty_req * price) < min_notional:
            need = min_notional / price
            qty_req = max(qty_req, ceil_to_step(need, step))

        order, sent, filled = await self.place_order(side, qty_req, reduce_only=True)
        used_qty = filled if filled and filled > 0 else 0.0

        remain = await self._get_position_amt()
        epsilon = max(step/2 if step else 0.0, 1e-8)

        if remain > epsilon:
            qty_req2 = qty_req + (step or 0.0)
            order2, sent2, filled2 = await self.place_order(side, qty_req2, reduce_only=True)
            if filled2 and filled2 > 0:
                used_qty += filled2
            remain = await self._get_position_amt()

        if current_price is None:
            mp2 = await self._get_mark_price()
            current_price = mp2 if mp2 is not None else self.entry_price
        taker_fee = float(self.config.get('taker_fee', TAKER_FEE_DEFAULT))

        if used_qty > 0:
            if self.position_type == 'LONG':
                raw_pnl = (current_price - self.entry_price) * used_qty
            else:
                raw_pnl = (self.entry_price - current_price) * used_qty
            fee = (self.entry_price + current_price) * taker_fee * used_qty
            pnl = raw_pnl - fee
            logger.info(f"{self.symbol} Close partial | used={used_qty} remain={remain} | PnL: ${pnl:.6f}")
        else:
            pnl = 0.0
            logger.warning(f"{self.symbol} Close attempt had 0 filled qty. remain={remain}")

        if remain <= epsilon:
            self.in_position = False
            self.position_type = None
            self.entry_price = None
            self.stop_loss = None
            self.take_profit = None
            self.trailing_stop = None
            self.position_size = 0.0
            self.hold_start_ts = None
            _save_state({'symbol': self.symbol, 'in_position': False})
            # cooldown setelah exit
            try:
                cd = int(self.config.get('cooldown_seconds', 1200))
                self.cooldown_until = time.time() + max(cd, 0)
            except Exception:
                self.cooldown_until = time.time() + 1200
            asyncio.create_task(_send_telegram(f"🏁 <b>{self.symbol}</b> CLOSE done | PnL ${pnl:.4f}"))
        else:
            self.position_size = remain
            _save_state({'symbol': self.symbol, 'in_position': True, 'position_type': self.position_type, 'entry_price': self.entry_price, 'position_size': self.position_size, 'stop_loss': self.stop_loss, 'take_profit': self.take_profit, 'trailing_stop': self.trailing_stop, 'hold_start_ts': self.hold_start_ts})
            asyncio.create_task(_send_telegram(f"⚠️ <b>{self.symbol}</b> residual remain: {remain} (akan dicoba close lagi)"))

    async def on_kline(self, kline: dict):
        try:
            self.data.loc[len(self.data)] = {
                'timestamp': kline['T'],
                'open': float(kline['o']),
                'high': float(kline['h']),
                'low': float(kline['l']),
                'close': float(kline['c']),
                'volume': float(kline['v'])
            }
        except Exception:
            pass
        await self.check_trading_signals()

    async def check_trading_signals(self):
        df = self.calculate_indicators()
        if df.empty:
            return
        last = df.iloc[-1]
        price = float(last['close'])
        lev = float(self.config.get('leverage', 10))
        risk = float(self.config.get('risk_per_trade', 0.05))

        long_sig = bool(last.get('long_signal', False))
        short_sig = bool(last.get('short_signal', False))
        cur_sig = 'LONG' if long_sig else ('SHORT' if short_sig else None)

        # ====== POSISI TERBUKA ======
        if self.in_position:
            mp = await self._get_mark_price()
            if mp is None:
                return

            # update counter sinyal berlawanan (untuk opsi SAR)
            if cur_sig and cur_sig != self.position_type:
                self.opp_sig_count += 1
            else:
                self.opp_sig_count = 0

            # ===== Hard Stop Loss check =====
            if self.stop_loss is not None:
                if self.position_type == 'LONG' and mp <= self.stop_loss:
                    await self._close_position_reduce_only(mp)
                    return
                if self.position_type == 'SHORT' and mp >= self.stop_loss:
                    await self._close_position_reduce_only(mp)
                    return

            # ===== Breakeven (geser SL ke entry saat profit >= be_trigger_pct) =====
            use_be = bool(int(self.config.get('use_breakeven', 1)))
            be_trigger = float(self.config.get('be_trigger_pct', 0.006))
            if use_be and self.entry_price:
                if self.position_type == 'LONG':
                    pnl_pct = (mp - self.entry_price)/self.entry_price
                    if pnl_pct >= be_trigger:
                        self.stop_loss = max(self.stop_loss or 0.0, self.entry_price)
                else:
                    pnl_pct = (self.entry_price - mp)/self.entry_price
                    if pnl_pct >= be_trigger:
                        self.stop_loss = min(self.stop_loss or 1e18, self.entry_price)

            # ===== Trailing stop (saat profit) =====
            trailing_trigger = float(self.config.get('trailing_trigger', 0.5))
            trailing_step = float(self.config.get('trailing_step', 0.3))
            if self.position_type == 'LONG':
                ep = self.entry_price or mp
                profit_pct = (mp - ep)/ep * 100.0
                if profit_pct >= trailing_trigger:
                    new_ts = mp * (1 - trailing_step/100)
                    self.trailing_stop = max(self.trailing_stop or self.stop_loss or 0, new_ts)
                if self.trailing_stop and mp <= self.trailing_stop:
                    await self._close_position_reduce_only(mp)
                    return
            else:
                ep = self.entry_price or mp
                profit_pct = (ep - mp)/ep * 100.0
                if profit_pct >= trailing_trigger:
                    new_ts = mp * (1 + trailing_step/100)
                    self.trailing_stop = min(self.trailing_stop or self.stop_loss or 1e18, new_ts)
                if self.trailing_stop and mp >= self.trailing_stop:
                    await self._close_position_reduce_only(mp)
                    return

            # ===== Optional Stop-And-Reverse (SAR) dengan konfirmasi =====
            allow_sar = bool(int(self.config.get('allow_sar', 0)))
            reverse_confirm_bars = int(self.config.get('reverse_confirm_bars', 2))
            min_hold_seconds = int(self.config.get('min_hold_seconds', 1200))
            held = time.time() - (self.hold_start_ts or time.time())
            if allow_sar and cur_sig and cur_sig != self.position_type:
                if held >= min_hold_seconds and self.opp_sig_count >= reverse_confirm_bars:
                    await self._close_position_reduce_only(mp)
                    self.in_position = False
                    self.opp_sig_count = 0
                    return

            # jika masih pegang posisi, stop di sini
            return
            mp = await self._get_mark_price()
            if mp is None:
                return

            # ===== Hard Stop Loss check =====
            if self.stop_loss is not None:
                if self.position_type == 'LONG' and mp <= self.stop_loss:
                    await self._close_position_reduce_only(mp)
                    return
                if self.position_type == 'SHORT' and mp >= self.stop_loss:
                    await self._close_position_reduce_only(mp)
                    return

            # ===== Breakeven (geser SL ke entry saat profit >= be_trigger_pct) =====
            use_be = bool(int(self.config.get('use_breakeven', 1)))
            be_trigger = float(self.config.get('be_trigger_pct', 0.006))
            if use_be and self.entry_price:
                if self.position_type == 'LONG':
                    pnl_pct = (mp - self.entry_price)/self.entry_price
                    if pnl_pct >= be_trigger:
                        self.stop_loss = max(self.stop_loss or 0.0, self.entry_price)
                else:
                    pnl_pct = (self.entry_price - mp)/self.entry_price
                    if pnl_pct >= be_trigger:
                        self.stop_loss = min(self.stop_loss or 1e18, self.entry_price)

            # ===== Trailing stop (saat profit) =====
            trailing_trigger = float(self.config.get('trailing_trigger', 0.5))
            trailing_step = float(self.config.get('trailing_step', 0.3))
            if self.position_type == 'LONG':
                ep = self.entry_price or mp
                profit_pct = (mp - ep)/ep * 100.0
                if profit_pct >= trailing_trigger:
                    new_ts = mp * (1 - trailing_step/100)
                    self.trailing_stop = max(self.trailing_stop or self.stop_loss or 0, new_ts)
                if self.trailing_stop and mp <= self.trailing_stop:
                    await self._close_position_reduce_only(mp)
            else:
                ep = self.entry_price or mp
                profit_pct = (ep - mp)/ep * 100.0
                if profit_pct >= trailing_trigger:
                    new_ts = mp * (1 + trailing_step/100)
                    self.trailing_stop = min(self.trailing_stop or self.stop_loss or 1e18, new_ts)
                if self.trailing_stop and mp >= self.trailing_stop:
                    await self._close_position_reduce_only(mp)
            return

        if not (long_sig or short_sig):
            return

        # --- Filter Presisi Entry ---
        atr_pct = float(last.get('atr_pct', 0) or 0)
        body_to_atr = float(last.get('body_to_atr', 0) or 0)
        min_atr_pct = float(self.config.get('min_atr_pct', 0.003))
        max_atr_pct = float(self.config.get('max_atr_pct', 0.03))
        max_body_atr = float(self.config.get('max_body_atr', 1.0))
        use_htf_filter = bool(int(self.config.get('use_htf_filter', 1)))

        if not (min_atr_pct <= atr_pct <= max_atr_pct):
            return
        if body_to_atr > max_body_atr:
            return
        if self.cooldown_until and time.time() < self.cooldown_until:
            return
        if use_htf_filter:
            want = 'LONG' if long_sig else 'SHORT'
            ok = await self._htf_trend_ok(want)
            if not ok:
                return

        # ===== Hitung sizing =====
        avail = await self._get_available_balance()
        max_cost = avail * risk
        if max_cost <= 0 or price <= 0:
            return
        raw_qty = (max_cost * lev) / price
        client = await self.require_client()
        f = await _get_exchange_filters(client, self.symbol)
        qty = _normalize_qty(raw_qty, price, f)
        if qty <= 0:
            logger.info(f"Skip open: qty invalid/raw {raw_qty} @ {price}")
            await _send_telegram(f"⏭️ <b>{self.symbol}</b> skip open (qty<min) raw={raw_qty:.4f} price={price}")
            return

        # ===== Hitung Hard SL saat ENTRY =====
        sl_mode = str(self.config.get('sl_mode', 'ATR')).upper()
        sl_min_pct = float(self.config.get('sl_min_pct', 0.006))
        sl_max_pct = float(self.config.get('sl_max_pct', 0.025))
        sl_pct = None
        if sl_mode == 'PCT':
            sl_pct = float(self.config.get('sl_pct', 0.015))
        else:
            atr = float(last.get('atr', 0) or 0)
            sl_atr_mult = float(self.config.get('sl_atr_mult', 1.2))
            if atr > 0 and price > 0:
                sl_pct = (sl_atr_mult * atr) / price
            else:
                sl_pct = float(self.config.get('sl_pct', 0.015))
        # clamp
        sl_pct = max(sl_min_pct, min(sl_pct, sl_max_pct))
        if long_sig:
            calc_sl = price * (1 - sl_pct)
        else:
            calc_sl = price * (1 + sl_pct)

        # ===== set state & buka posisi =====
        self.position_size = qty
        self.entry_price = price
        self.position_type = 'LONG' if long_sig else 'SHORT'
        self.stop_loss = calc_sl
        self.trailing_stop = None
        self.in_position = True
        self.hold_start_ts = time.time()
        _save_state({'symbol': self.symbol, 'in_position': True, 'position_type': self.position_type, 'entry_price': self.entry_price, 'position_size': self.position_size, 'stop_loss': self.stop_loss, 'take_profit': self.take_profit, 'trailing_stop': self.trailing_stop, 'hold_start_ts': self.hold_start_ts})
        asyncio.create_task(_send_telegram(f"🚀 <b>{self.symbol}</b> OPEN {self.position_type} {self.position_size} @ {self.entry_price} SL={self.stop_loss}"))

        side = 'BUY' if self.position_type == 'LONG' else 'SELL'
        await self.place_order(side, self.position_size, reduce_only=False)

    async def _get_available_balance(self) -> float:
        try:
            client = await self.require_client()
            balances = await client.futures_account_balance()
            for a in balances:
                if a.get('asset') == 'USDT':
                    return float(a.get('availableBalance', 0))
        except Exception as e:
            logger.error(f"{self.symbol} get balance failed: {e}")
        return 0.0

    async def start_websocket(self):
        client = await self.require_client()
        bsm = BinanceSocketManager(client)
        conn = bsm.kline_socket(symbol=self.symbol, interval=TIMEFRAME)
        retry = 0
        while True:
            try:
                async with conn as stream:
                    while True:
                        msg = await stream.recv()
                        k = msg.get('k', {}) if msg else {}
                        if not k or not k.get('x'):
                            continue
                        await self.on_kline({
                            'T': k.get('T'),
                            'o': k.get('o'),
                            'h': k.get('h'),
                            'l': k.get('l'),
                            'c': k.get('c'),
                            'v': k.get('v')
                        })
            except Exception as e:
                retry += 1
                if WS_MAX_RETRY and retry > WS_MAX_RETRY:
                    logger.error(f"{self.symbol} WS stopped after {retry} retries: {e}")
                    break
                await asyncio.sleep(min(2**retry, 30))

    async def stop(self):
        try:
            if self.client:
                await self.client.close_connection()
        except Exception:
            pass

# ===== Manager =====
class TradingManager:
    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.traders: Dict[str, CoinTrader] = {}
        self.configs = self.load_configs()
        self.shared_client: Optional[AsyncClient] = None
        self._cfg_mtime = None
        self._watch_task: Optional[asyncio.Task] = None

    def load_configs(self):
        try:
            with open(CONFIG_FILE, 'r') as f:
                configs = json.load(f)
            for s in SYMBOLS:
                if s not in configs:
                    raise ValueError(f"Configuration missing for {s}")
            logger.info("Configuration loaded successfully")
            return configs
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            raise

    async def _watch_config(self):
        interval = int(os.getenv("CONFIG_WATCH_INTERVAL", "5"))
        if interval <= 0:
            return
        safe_keys = {"risk_per_trade","leverage","trailing_trigger","trailing_step","max_hold_seconds","min_roi_to_close_by_time","taker_fee","maker_fee","min_atr_pct","max_atr_pct","max_body_atr","use_htf_filter","cooldown_seconds","sl_mode","sl_pct","sl_atr_mult","sl_min_pct","sl_max_pct","use_breakeven","be_trigger_pct","allow_sar","reverse_confirm_bars","min_hold_seconds"}
        while True:
            try:
                mtime = os.path.getmtime(CONFIG_FILE) if os.path.exists(CONFIG_FILE) else None
                if mtime and self._cfg_mtime != mtime:
                    new_cfgs = self.load_configs()
                    for sym, trader in self.traders.items():
                        old = trader.config
                        new = new_cfgs.get(sym, {})
                        if not new:
                            continue
                        changed = {k:(old.get(k), new.get(k)) for k in safe_keys if str(old.get(k)) != str(new.get(k))}
                        if changed:
                            await trader.apply_config_update(new, changed)
                    self.configs = new_cfgs
                    self._cfg_mtime = mtime
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"config watch error: {e}")
                await asyncio.sleep(interval)

    async def get_futures_balance(self) -> float:
        try:
            client = self.shared_client
            if client is None:
                client = await AsyncClient.create(self.api_key, self.api_secret)
                self.shared_client = client
            balances = await client.futures_account_balance()
            for a in balances:
                if a.get('asset') == 'USDT':
                    return float(a.get('availableBalance', 0))
        except Exception as e:
            logger.error(f"get_futures_balance failed: {e}")
        return 0.0

    async def initialize(self):
        if self.shared_client is None:
            self.shared_client = await AsyncClient.create(self.api_key, self.api_secret)
        total_balance = await self.get_futures_balance()
        risk_pct = float(os.getenv("RISK_PERCENTAGE", "1.0"))
        trading_balance = total_balance * risk_pct
        capital_per_coin = trading_balance / max(len(SYMBOLS), 1)
        logger.info(f"Total balance: {total_balance:.2f} USDT")
        logger.info(f"Allocated trading balance: {trading_balance:.2f} USDT")
        logger.info(f"Capital per coin: {capital_per_coin:.2f} USDT")

        for s in SYMBOLS:
            t = CoinTrader(s, self.api_key, self.api_secret, capital_per_coin, self.configs[s])
            t.client = self.shared_client
            self.traders[s] = t
            await t.initialize()

    async def _start_multiplex(self):
        client = self.shared_client
        if client is None:
            client = await AsyncClient.create(self.api_key, self.api_secret)
            self.shared_client = client
        bsm = BinanceSocketManager(client)
        streams = [f"{s.lower()}@kline_{TIMEFRAME}" for s in SYMBOLS]
        ms = bsm.multiplex_socket(streams)
        retry = 0
        logger.info(f"Starting with multiplex WebSocket (single connection).")
        while True:
            try:
                async with ms as stream:
                    logger.info(f"Multiplex WS started for {len(streams)} streams")
                    while True:
                        msg = await stream.recv()
                        data = msg.get('data', {}) if msg else {}
                        k = data.get('k', {})
                        if not k or not k.get('x'):
                            continue
                        sym = data.get('s')
                        if not isinstance(sym, str) or not sym:
                            continue
                        trader = self.traders.get(sym)
                        if trader:
                            await trader.on_kline({
                                'T': k.get('T'),
                                'o': k.get('o'),
                                'h': k.get('h'),
                                'l': k.get('l'),
                                'c': k.get('c'),
                                'v': k.get('v')
                            })
            except Exception as e:
                retry += 1
                if WS_MAX_RETRY and retry > WS_MAX_RETRY:
                    logger.error(f"Multiplex WS stopped after {retry} retries: {e}")
                    break
                await asyncio.sleep(min(2**retry, 30))

    async def start_trading(self):
        self._cfg_mtime = os.path.getmtime(CONFIG_FILE) if os.path.exists(CONFIG_FILE) else None
        self._watch_task = asyncio.create_task(self._watch_config())
        if USE_MULTIPLEX:
            await self._start_multiplex()
        else:
            logger.info("Starting with per-symbol WebSocket (one connection per symbol).")
            tasks = [t.start_websocket() for t in self.traders.values()]
            await asyncio.gather(*tasks)
        while True:
            await asyncio.sleep(3600)

    async def stop(self):
        if self._watch_task:
            self._watch_task.cancel()
            try:
                await self._watch_task
            except Exception:
                pass
        for t in self.traders.values():
            await t.stop()
        if self.shared_client:
            try:
                await self.shared_client.close_connection()
            except Exception:
                pass

# ===== Main =====
async def main():
    api_key = os.getenv('BINANCE_API_KEY')
    api_secret = os.getenv('BINANCE_API_SECRET')
    if not api_key or not api_secret:
        raise ValueError("API keys not found in environment variables")
    manager = TradingManager(api_key, api_secret)
    await manager.initialize()
    try:
        await manager.start_trading()
    finally:
        await manager.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down traders...")
    except Exception as e:
        logger.error(f"Critical error: {e}")
    finally:
        logger.info("Trading bot stopped")
