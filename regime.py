import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass
class RegimeThresholds:
    adx_period: int = 14
    bb_period: int = 20
    trend_threshold: float = 20.0
    bb_width_chop_max: float = 0.010

def compute_adx(df: pd.DataFrame, period: int) -> pd.Series:
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    close = df['close'].astype(float)
    
    up = high.diff()
    down = -low.diff()
    
    # Use numpy where for more explicit type handling
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    
    # Convert back to pandas Series if needed
    plus_dm = pd.Series(plus_dm, index=df.index)
    minus_dm = pd.Series(minus_dm, index=df.index)
    
    # Rest of the calculation remains the same
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (close.shift() - low).abs()
    ], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/period, adjust=False).mean() / atr)
    
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()
    
    return adx

def compute_bb_width(close: pd.Series, period: int=20) -> pd.Series:
    ma = close.rolling(period).mean()
    std = close.rolling(period).std(ddof=0)
    upper = ma + 2*std
    lower = ma - 2*std
    width = (upper - lower) / ma
    return width

def get_regime(df: pd.DataFrame, th: RegimeThresholds) -> str:
    adx = compute_adx(df, th.adx_period)
    bbw = compute_bb_width(df['close'], th.bb_period)
    adx_now = float(adx.iloc[-1]) if len(adx) else 0.0
    bbw_now = float(bbw.iloc[-1]) if len(bbw) else 0.0
    if adx_now >= th.trend_threshold and bbw_now >= th.bb_width_chop_max:
        return 'TREND'
    return 'CHOP'
