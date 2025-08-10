import asyncio
from typing import AsyncGenerator
from binance import AsyncClient, BinanceSocketManager

from .config import Settings


class BinanceClient:
    def __init__(self):
        self.settings = Settings()
        self.client: AsyncClient | None = None
        self.bsm: BinanceSocketManager | None = None

    async def connect(self):
        self.client = await AsyncClient.create(
            api_key=self.settings.BINANCE_API_KEY,
            api_secret=self.settings.BINANCE_API_SECRET,
            testnet=self.settings.BINANCE_TESTNET,
        )
        self.bsm = BinanceSocketManager(self.client)

    async def close(self):
        if self.client:
            await self.client.close_connection()

    async def get_equity(self) -> float:
        assert self.client
        info = await self.client.futures_account_balance()
        for item in info:
            if item["asset"] == "USDT":
                return float(item["balance"])
        return 0.0

    async def set_leverage(self, symbol: str, leverage: int):
        assert self.client
        await self.client.futures_change_leverage(symbol=symbol, leverage=leverage)

    async def stream_kline(self, symbol: str, interval: str = "1m") -> AsyncGenerator[dict, None]:
        assert self.bsm
        ts = self.bsm.futures_kline_socket(symbol=symbol, interval=interval)
        async with ts as tscm:
            while True:
                msg = await tscm.recv()
                yield msg

    async def place_market_order(self, symbol: str, side: str, quantity: float):
        assert self.client
        return await self.client.futures_create_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=quantity,
        )

    async def place_sl_tp(self, symbol: str, side: str, quantity: float, sl: float, tp: float):
        assert self.client
        opp_side = "SELL" if side == "BUY" else "BUY"
        await self.client.futures_create_order(
            symbol=symbol,
            side=opp_side,
            type="STOP_MARKET",
            stopPrice=sl,
            closePosition=True,
            reduceOnly=True,
        )
        await self.client.futures_create_order(
            symbol=symbol,
            side=opp_side,
            type="TAKE_PROFIT_MARKET",
            stopPrice=tp,
            closePosition=True,
            reduceOnly=True,
        )
