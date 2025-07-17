import pandas as pd
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands
from ta.momentum import RSIIndicator

def add_indicators(df):
    df['ema_22'] = EMAIndicator(df['close'], 22).ema_indicator()
    df['ma_22'] = SMAIndicator(df['close'], 22).sma_indicator()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['rsi'] = RSIIndicator(df['close'], 15).rsi()
    atr = AverageTrueRange(df['high'], df['low'], df['close'])
    df['atr'] = atr.average_true_range()
    bb = BollingerBands(df['close'])
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
    return df