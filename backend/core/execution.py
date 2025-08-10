import asyncio
from datetime import datetime
from .binance_client import BinanceClient
from .sizing import calculate_qty
from .reporting import log_trade
from .config import Settings


class Executor:
    def __init__(self, client: BinanceClient):
        self.client = client
        self.settings = Settings()

    async def open_position(self, symbol: str, side: str, entry: float, sl: float, tp: float, step_size: float, reason: str):
        equity = await self.client.get_equity()
        qty = calculate_qty(entry, sl, equity, self.settings.RISK_PER_TRADE, step_size)
        order_side = "BUY" if side == "long" else "SELL"
        res = await self.client.place_market_order(symbol, order_side, qty)
        await self.client.place_sl_tp(symbol, order_side, qty, sl, tp)
        log_trade(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "symbol": symbol,
                "side": side,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "qty": qty,
                "fee": 0.0,
                "pnl": 0.0,
                "pnl_pct": 0.0,
                "reason": reason,
            }
        )
        return res
