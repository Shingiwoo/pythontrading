# >>> PATCH (2025-08-10): 
import os
import json
import asyncio
import time
import math
import logging
from typing import Optional

import pandas as pd
import numpy as np
from dotenv import load_dotenv
from binance import AsyncClient, BinanceSocketManager
from binance.enums import ORDER_TYPE_MARKET

load_dotenv()

# Logger setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("trading_bot.log"), logging.StreamHandler()]
)
logger = logging.getLogger('binance_future_scalping')

# ===== Konfigurasi =====
CONFIG_FILE = "coin_config.json"
DEFAULT_SYMBOLS = ['XRPUSDT', 'DOGEUSDT', 'TURBOUSDT']
TIMEFRAME = '15m'

# DEFAULT fallback kalau ENV kosong
env_syms = os.getenv("SYMBOLS", "")
if env_syms.strip():
    SYMBOLS = [s.strip().upper() for s in env_syms.split(",") if s.strip()]
else:
    SYMBOLS = DEFAULT_SYMBOLS

# WS guards
USE_MULTIPLEX = bool(int(os.getenv("USE_MULTIPLEX", "1")))
WS_MAX_RETRY = int(os.getenv("WS_MAX_RETRY", "0"))

# Fees & ROI/Time Guards (defaults, bisa di-override via config per coin)
TAKER_FEE_DEFAULT = float(os.getenv('TAKER_FEE', '0.0005'))
MAKER_FEE_DEFAULT = float(os.getenv('MAKER_FEE', '0.0002'))
MIN_ROI_TO_CLOSE_BY_TIME_DEFAULT = float(os.getenv('MIN_ROI_TO_CLOSE_BY_TIME', '0.05'))
MAX_HOLD_SECONDS_DEFAULT = int(os.getenv('MAX_HOLD_SECONDS', '3600'))

# State & exchange info cache
STATE_FILE = os.getenv('STATE_FILE', 'active_position_state.json')
EXCHANGEINFO_CACHE = os.getenv('EXCHANGEINFO_CACHE', 'exchange_info_cache.json')
EXCHANGEINFO_TTL_SECONDS = int(os.getenv('EXCHANGEINFO_TTL_SECONDS', '21600'))  # 6 jam

# ===== Utilities =====

def _extract_filled_qty(order: dict) -> float:
    """Ambil qty terisi dari berbagai format respons order Futures Binance."""
    try:
        for k in ("executedQty", "cumQty"):
            if k in order and order[k] is not None:
                return float(order[k])
        if "origQty" in order and order["origQty"] is not None:  # fallback
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

def floor_to_step(x, step):
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
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=payload, timeout=10) as resp:
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

async def _save_cached_exchange_info(info):
    try:
        with open(EXCHANGEINFO_CACHE, 'w') as f:
            json.dump({'ts': time.time(), 'info': info}, f)
    except Exception as e:
        logger.warning(f"Write exchangeInfo cache failed: {e}")

async def _get_exchange_filters(client, symbol: str):
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

def _normalize_qty(qty: float, price: float, f) -> float:
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
        return 0.0
    if min_notional and (q * price) < min_notional:
        needed = min_notional / price
        if step and step > 0:
            needed = math.ceil(needed / step) * step
        try:
            needed = float(f"{needed:.{qprec}f}")
        except Exception:
            pass
        if needed < min_qty:
            needed = min_qty
        q = needed
    return q

# ===== Kelas Trader =====
class CoinTrader:
    def __init__(self, symbol: str, api_key: str, api_secret: str, capital_allocated: float, config: dict):
        self.symbol = symbol
        self.api_key = api_key
        self.api_secret = api_secret
        self.capital = capital_allocated  # tracking lokal
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

    async def require_client(self) -> AsyncClient:
        if self.client is None:
            self.client = await AsyncClient.create(self.api_key, self.api_secret)
        return self.client

    async def _get_mark_price(self) -> Optional[float]:
        try:
            mp = await self.require_client().futures_mark_price(symbol=self.symbol)
            return float(mp['markPrice'])
        except Exception as e:
            logger.error(f"{self.symbol} get mark price failed: {e}")
            return None
    
    async def _get_position_amt(self) -> float:
        try:
            info = await self.require_client().futures_position_information(symbol=self.symbol)
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
        await self.require_client().futures_change_leverage(symbol=self.symbol, leverage=int(self.config['leverage']))

        # load data awal buat indikator internal jika perlu
        try:
            klines = await self.require_client().futures_klines(symbol=self.symbol, interval=TIMEFRAME, limit=1000)
            rows = []
            for k in klines:
                rows.append({'timestamp':k[0],'open':float(k[1]),'high':float(k[2]),'low':float(k[3]),'close':float(k[4]),'volume':float(k[5])})
            self.data = pd.DataFrame(rows)
            logger.info(f"{self.symbol} initialized with {len(self.data)} candles")
        except Exception as e:
            logger.warning(f"{self.symbol} init klines failed: {e}")

    async def apply_config_update(self, new_cfg: dict, changed: dict):
        # Update config internal
        self.config.update({k:new_cfg.get(k, self.config.get(k)) for k in new_cfg.keys()})
        # Terapkan perubahan leverage segera
        if 'leverage' in changed:
            try:
                await self.require_client().futures_change_leverage(symbol=self.symbol, leverage=int(self.config['leverage']))
                try:
                    asyncio.create_task(_send_telegram(f"⚙️ <b>{self.symbol}</b> leverage diubah -> {self.config['leverage']}"))
                except Exception:
                    pass
                logger.info(f"{self.symbol} leverage updated to {self.config['leverage']} via hot-reload")
            except Exception as e:
                logger.error(f"{self.symbol} change leverage failed: {e}")
        # Notif perubahan lain
        other = {k:v for k,v in changed.items() if k!='leverage'}
        if other:
            try:
                msg = " | ".join([f"{k}:{v[0]}→{v[1]}" for k,v in other.items()])
                asyncio.create_task(_send_telegram(f"♻️ <b>{self.symbol}</b> hot-reload: {msg}"))
            except Exception:
                pass
        return True

    async def _get_exchange_filters(self):
        return await _get_exchange_filters(await self.require_client(), self.symbol)

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
            logger.error(f"Create order failed: {e}")
            await _send_telegram(f"❌ <b>{self.symbol}</b> order gagal: {e}")
            return None, None, 0.0

    async def _close_position_reduce_only(self, current_price: Optional[float] = None):
        # Ambil size posisi aktual di exchange
        pos_amt = await self._get_position_amt()
        if not self.in_position or pos_amt <= 0 or self.entry_price is None:
            return

        side = 'SELL' if self.position_type == 'LONG' else 'BUY'
        f = await _get_exchange_filters(await self.require_client(), self.symbol)

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
            import math
            return math.ceil(float(x)/float(step))*float(step)

        # Overshoot agar lolos minQty/minNotional (reduceOnly mencegah flip posisi)
        qty_req = ceil_to_step(pos_amt, step)
        if min_qty and qty_req < min_qty:
            qty_req = ceil_to_step(min_qty, step)
        if min_notional and price and (qty_req * price) < min_notional:
            need = min_notional / price
            qty_req = max(qty_req, ceil_to_step(need, step))

        order, sent, filled = await self.place_order(side, qty_req, reduce_only=True)
        used_qty = filled if filled and filled > 0 else 0.0

        # Cek sisa posisi
        remain = await self._get_position_amt()
        epsilon = max(step/2 if step else 0.0, 1e-8)

        if remain > epsilon:
            # Coba sekali lagi dengan overshoot sedikit lebih besar
            qty_req2 = qty_req + (step or 0.0)
            order2, sent2, filled2 = await self.place_order(side, qty_req2, reduce_only=True)
            if filled2 and filled2 > 0:
                used_qty += filled2
            remain = await self._get_position_amt()

        # Harga final untuk PnL
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

        # Finalize state jika benar2 nol; jika tidak, simpan sisa utk dicoba lagi nanti
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
            asyncio.create_task(_send_telegram(f"🏁 <b>{self.symbol}</b> CLOSE done | PnL ${pnl:.4f}"))
        else:
            self.position_size = remain
            _save_state({'symbol': self.symbol, 'in_position': True, 'position_type': self.position_type, 'entry_price': self.entry_price, 'position_size': self.position_size, 'stop_loss': self.stop_loss, 'take_profit': self.take_profit, 'trailing_stop': self.trailing_stop, 'hold_start_ts': self.hold_start_ts})
            asyncio.create_task(_send_telegram(f"⚠️ <b>{self.symbol}</b> residual remain: {remain} (akan dicoba close lagi)"))

    def calculate_indicators(self) -> pd.DataFrame:
        df = self.data.copy()
        if df.empty:
            return df
        # indikator minimal untuk trigger; bisa diganti sesuai app_swing.py
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

        df['long_signal'] = (df['ema']>df['ma']) & (df['macd']>df['macd_signal']) & (df['rsi'].between(40,70))
        df['short_signal'] = (df['ema']<df['ma']) & (df['macd']<df['macd_signal']) & (df['rsi'].between(30,60))
        return df

    async def on_kline(self, kline: dict):
        # update data realtime
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
        lev = float(self.config['leverage'])
        risk = float(self.config['risk_per_trade'])

        long_sig = bool(last.get('long_signal', False))
        short_sig = bool(last.get('short_signal', False))

        # Reversal bila sinyal berlawanan
        if self.in_position and ((self.position_type == 'LONG' and short_sig) or (self.position_type == 'SHORT' and long_sig)):
            await self._close_position_reduce_only(price)
            self.in_position = False

        if self.in_position:
            # update trailing berdasarkan config terkini
            mp = await self._get_mark_price()
            if mp is None:
                return
            trailing_trigger = float(self.config.get('trailing_trigger', 0.5))
            trailing_step = float(self.config.get('trailing_step', 0.3))
            if self.position_type == 'LONG':
                profit_pct = (mp - self.entry_price)/self.entry_price*100
                if profit_pct >= trailing_trigger:
                    new_ts = mp * (1 - trailing_step/100)
                    self.trailing_stop = max(self.trailing_stop or self.stop_loss or 0, new_ts)
                if self.trailing_stop and mp <= self.trailing_stop:
                    await self._close_position_reduce_only(mp)
            else:
                profit_pct = (self.entry_price - mp)/self.entry_price*100
                if profit_pct >= trailing_trigger:
                    new_ts = mp * (1 + trailing_step/100)
                    self.trailing_stop = min(self.trailing_stop or self.stop_loss or 1e18, new_ts)
                if self.trailing_stop and mp >= self.trailing_stop:
                    await self._close_position_reduce_only(mp)
            return

        if not (long_sig or short_sig):
            return

        # Hitung margin dari available balance (per coin)
        avail = await self._get_available_balance()
        max_cost = avail * risk
        if max_cost <= 0 or price <= 0:
            return
        raw_qty = (max_cost * lev) / price
        f = await _get_exchange_filters(await self.require_client(), self.symbol)
        qty = _normalize_qty(raw_qty, price, f)
        if qty <= 0:
            logger.info(f"Skip open: qty invalid/raw {raw_qty} @ {price}")
            await _send_telegram(f"⏭️ <b>{self.symbol}</b> skip open (qty<min) raw={raw_qty:.4f} price={price}")
            return

        # set state & buka posisi
        self.position_size = qty
        self.entry_price = price
        self.position_type = 'LONG' if long_sig else 'SHORT'
        self.in_position = True
        self.hold_start_ts = time.time()
        _save_state({'symbol': self.symbol, 'in_position': True, 'position_type': self.position_type, 'entry_price': self.entry_price, 'position_size': self.position_size, 'stop_loss': self.stop_loss, 'take_profit': self.take_profit, 'trailing_stop': self.trailing_stop, 'hold_start_ts': self.hold_start_ts})
        asyncio.create_task(_send_telegram(f"🚀 <b>{self.symbol}</b> OPEN {self.position_type} {self.position_size} @ {self.entry_price}"))

        # eksekusi market
        side = 'BUY' if self.position_type == 'LONG' else 'SELL'
        await self.place_order(side, self.position_size, reduce_only=False)

    async def _get_available_balance(self) -> float:
        try:
            # ambil balance futures USDT
            balances = await self.require_client().futures_account_balance()
            for a in balances:
                if a.get('asset') == 'USDT':
                    return float(a['availableBalance'])
        except Exception as e:
            logger.error(f"{self.symbol} get balance failed: {e}")
        return 0.0

    async def start_websocket(self):
        bsm = BinanceSocketManager(await self.require_client())
        if USE_MULTIPLEX:
            return  # ditangani di manager
        conn = bsm.kline_socket(symbol=self.symbol, interval=TIMEFRAME)
        retry = 0
        while True:
            try:
                async with conn as stream:
                    async for msg in stream:
                        k = msg.get('k', {})
                        if not k.get('x'):
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

# ===== Persistence =====
def _save_state(state: dict):
    try:
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

# ===== Manager =====
class TradingManager:
    async def _watch_config(self):
        """Hot-reload CONFIG_FILE setiap 5 detik. Hanya update parameter yang aman."""
        interval = int(os.getenv("CONFIG_WATCH_INTERVAL", "5"))
        safe_keys = {"risk_per_trade","leverage","trailing_trigger","trailing_step","max_hold_seconds","min_roi_to_close_by_time","taker_fee","maker_fee"}
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

    def __init__(self, api_key: str, api_secret: str):
        self.api_key = api_key
        self.api_secret = api_secret
        self.traders: dict[str, CoinTrader] = {}
        self.configs = self.load_configs()
        self.shared_client: Optional[AsyncClient] = None
        self._cfg_mtime = None
        self._watch_task = None

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

    async def get_futures_balance(self) -> float:
        try:
            client = self.shared_client
            if client is None:
                client = await AsyncClient.create(self.api_key, self.api_secret)
                self.shared_client = client
            balances = await client.futures_account_balance()
            for a in balances:
                if a.get('asset') == 'USDT':
                    return float(a['availableBalance'])
        except Exception as e:
            logger.error(f"get_futures_balance failed: {e}")
        return 0.0

    async def initialize(self):
        if self.shared_client is None:
            self.shared_client = await AsyncClient.create(self.api_key, self.api_secret)
        total_balance = await self.get_futures_balance()
        risk_pct = float(os.getenv("RISK_PERCENTAGE", "1.0"))  # alokasi dari total balance, 1.0 = 100%
        trading_balance = total_balance * risk_pct
        capital_per_coin = trading_balance / len(SYMBOLS)
        logger.info(f"Total balance: {total_balance:.2f} USDT")
        logger.info(f"Allocated trading balance: {trading_balance:.2f} USDT")
        logger.info(f"Capital per coin: {capital_per_coin:.2f} USDT")

        for s in SYMBOLS:
            t = CoinTrader(s, self.api_key, self.api_secret, capital_per_coin, self.configs[s])
            t.client = self.shared_client
            self.traders[s] = t
            await t.initialize()

    async def _start_multiplex(self):
        bsm = BinanceSocketManager(self.shared_client)
        streams = [f"{s.lower()}@kline_{TIMEFRAME}" for s in SYMBOLS]
        ms = bsm.multiplex_socket(streams)
        retry = 0
        while True:
            try:
                async with ms as stream:
                    async for msg in stream:
                        data = msg.get('data', {})
                        k = data.get('k', {})
                        if not k.get('x'):
                            continue
                        sym = data.get('s')
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
        # start watcher
        self._cfg_mtime = os.path.getmtime(CONFIG_FILE) if os.path.exists(CONFIG_FILE) else None
        self._watch_task = asyncio.create_task(self._watch_config())
        if USE_MULTIPLEX:
            logger.info("Starting with multiplex WebSocket (single connection).")
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