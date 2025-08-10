from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional

from .core import config as config_module
from .core import state as state_module
from .core import reporting
from .core import strategy
from .core import sizing

app = FastAPI(title="Trading Bot API")

# CORS untuk frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bot_state = state_module.BotState()


class StartRequest(BaseModel):
    symbol: str


@app.post("/api/bot/start")
async def start_bot(req: StartRequest, background: BackgroundTasks):
    background.add_task(bot_state.start_symbol, req.symbol)
    return {"status": "started", "symbol": req.symbol}


@app.post("/api/bot/stop")
async def stop_bot(req: StartRequest):
    await bot_state.stop_symbol(req.symbol)
    return {"status": "stopped", "symbol": req.symbol}


@app.get("/api/status")
async def status():
    return bot_state.status()


@app.get("/api/config")
async def get_config():
    return config_module.load_config()


class ConfigUpdate(BaseModel):
    symbol: str
    params: Dict[str, Optional[str]]


@app.put("/api/config")
async def update_config(update: ConfigUpdate):
    cfg = config_module.load_config()
    for sym in cfg["symbols"]:
        if sym["symbol"] == update.symbol:
            sym.update(update.params)
    config_module.save_config(cfg)
    return {"status": "updated", "symbol": update.symbol}


@app.get("/api/trades")
async def get_trades(symbol: Optional[str] = None):
    return reporting.get_trades(symbol)


@app.get("/api/summary/daily")
async def daily_summary(date: Optional[str] = None):
    return reporting.daily_summary(date)
