from dataclasses import dataclass, asdict
from typing import Dict, Optional
import json
import os

STATE_PATH = os.path.join(os.path.dirname(__file__), "..", "state.json")


@dataclass
class PositionState:
    symbol: str
    direction: Optional[str] = None
    entry: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None
    qty: Optional[float] = None
    trailing_sl: Optional[float] = None
    hold: int = 0


class BotState:
    def __init__(self):
        self.positions: Dict[str, PositionState] = {}
        self.load()

    def load(self):
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH, "r") as f:
                data = json.load(f)
                for sym, pos in data.items():
                    self.positions[sym] = PositionState(**pos)

    def save(self):
        data = {sym: asdict(pos) for sym, pos in self.positions.items()}
        with open(STATE_PATH, "w") as f:
            json.dump(data, f)

    async def start_symbol(self, symbol: str):
        if symbol not in self.positions:
            self.positions[symbol] = PositionState(symbol=symbol)
        # di sini logika start trading per simbol
        self.save()

    async def stop_symbol(self, symbol: str):
        if symbol in self.positions:
            # logika stop trading
            pass
        self.save()

    def status(self):
        return {sym: asdict(pos) for sym, pos in self.positions.items()}
