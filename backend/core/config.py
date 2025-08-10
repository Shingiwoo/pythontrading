import os
import yaml
from pydantic import BaseSettings
from typing import Any, Dict

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")


class Settings(BaseSettings):
    BINANCE_API_KEY: str = ""
    BINANCE_API_SECRET: str = ""
    BINANCE_TESTNET: bool = True
    DEFAULT_LEVERAGE: int = 14
    RISK_PER_TRADE: float = 0.08

    class Config:
        env_file = ".env"


def load_config() -> Dict[str, Any]:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    return {"symbols": []}


def save_config(data: Dict[str, Any]):
    with open(CONFIG_PATH, "w") as f:
        yaml.safe_dump(data, f)
