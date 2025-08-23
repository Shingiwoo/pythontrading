from dataclasses import dataclass
import pandas as pd
import numpy as np

@dataclass
class RegimeThresholds:
    adx_period: int = 14
    bb_period: int = 20
    trend_threshold: float = 20.0
    bb_width_chop_max: float = 0.010


def _true_range(df: pd.DataFrame) -> pd.Series:
    high = df["high"].astype(float); low = df["low"].astype(float); close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr


def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"].astype(float); low = df["low"].astype(float)
    up = high.diff(); down = -low.diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = _true_range(df)
    # Wilder's smoothing (EMA alpha = 1/period)
    alpha = 1.0 / period
    tr_s = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=df.index).ewm(alpha=alpha, adjust=False).mean()
    plus_di = 100.0 * (plus_dm_s / tr_s).replace({0: np.nan})
    minus_di = 100.0 * (minus_dm_s / tr_s).replace({0: np.nan})
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace({0: np.nan})) * 100.0
    adx = dx.ewm(alpha=alpha, adjust=False).mean().fillna(0.0)
    return adx


def compute_bb_width(close: pd.Series, period: int = 20) -> pd.Series:
    ma = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = ma + 2.0 * std
    lower = ma - 2.0 * std
    with np.errstate(divide='ignore', invalid='ignore'):
        width = (upper - lower) / ma.replace(0, np.nan)
    return width.fillna(0.0)


def get_regime(df: pd.DataFrame, th: RegimeThresholds) -> str:
    adx = compute_adx(df, th.adx_period)
    bbw = compute_bb_width(df["close"], th.bb_period)
    adx_now = float(adx.iloc[-1]) if len(adx) else 0.0
    bbw_now = float(bbw.iloc[-1]) if len(bbw) else 0.0
    if adx_now >= th.trend_threshold and bbw_now >= th.bb_width_chop_max:
        return "TREND"
    return "CHOP"

