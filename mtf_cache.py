import pandas as pd
from typing import Dict

_cache: Dict[str, Dict[str, pd.Series]] = {}
_last_ts: Dict[str, pd.Timestamp] = {}


def _norm(freq: str) -> str:
    f = (freq or "").strip().lower()
    if f.endswith("m") and not f.endswith("min"):
        if not f.endswith(("ms", "us")):
            f = f[:-1] + "min"
    return f


def update_cache(symbol: str, df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            d = df.set_index(pd.to_datetime(df['timestamp'], utc=True))
        else:
            df.index = pd.to_datetime(df.index, utc=True)
            d = df
    else:
        d = df
    last_ts = d.index[-1]
    if _last_ts.get(symbol) == last_ts:
        return
    freqs = ["1h", "4h", "5min", "1min"]
    store = _cache.setdefault(symbol, {})
    for f in freqs:
        res = _norm(f)
        store[res] = d['close'].resample(res).last().dropna()
    _last_ts[symbol] = last_ts


def get_series(symbol: str, freq: str):
    res = _norm(freq)
    return _cache.get(symbol, {}).get(res)
