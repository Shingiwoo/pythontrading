import pandas as pd
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands
from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np


class Strategy:
    def __init__(self, config: dict):
        self.config = config

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        df = df.copy()
        df["ema"] = EMAIndicator(df["close"], cfg["ema_period"]).ema_indicator()
        df["ma"] = SMAIndicator(df["close"], cfg["sma_period"]).sma_indicator()
        macd = MACD(df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["rsi"] = RSIIndicator(df["close"], cfg["rsi_period"]).rsi()
        bb = BollingerBands(df["close"], window=20, window_dev=int(cfg["bb_std"]))
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["close"]
        df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()

        df["lag_ret"] = df["close"].pct_change().shift(1)
        df["vol"] = df["close"].rolling(20).std().shift(1)
        ml_df = df[["rsi", "macd", "atr", "bb_width", "lag_ret", "vol"]].dropna()
        target = (df["close"].shift(-5) > df["close"]).astype(int).loc[ml_df.index]
        model = RandomForestClassifier(n_estimators=100)
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        preds = np.zeros(len(ml_df))
        for train_idx, _ in skf.split(ml_df, target):
            model.fit(ml_df.iloc[train_idx], target.iloc[train_idx])
            preds[train_idx] = model.predict(ml_df.iloc[train_idx])
        df["ml_signal"] = 0
        df.loc[ml_df.index, "ml_signal"] = preds

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        df = df.copy()
        df["long_signal"] = False
        df["short_signal"] = False
        for i in range(1, len(df)):
            score_long = 0
            if df["ema"].iloc[i] > df["ma"].iloc[i] and df["macd"].iloc[i] > df["macd_signal"].iloc[i]:
                score_long += 1
            if 40 < df["rsi"].iloc[i] < 70:
                score_long += 0.5
            if df["ml_signal"].iloc[i] == 1:
                score_long += 1
            if score_long >= cfg["score_threshold"]:
                df.loc[df.index[i], "long_signal"] = True

            score_short = 0
            if df["ema"].iloc[i] < df["ma"].iloc[i] and df["macd"].iloc[i] < df["macd_signal"].iloc[i]:
                score_short += 1
            if 30 < df["rsi"].iloc[i] < 60:
                score_short += 0.5
            if df["ml_signal"].iloc[i] == 0:
                score_short += 1
            if score_short >= cfg["score_threshold"]:
                df.loc[df.index[i], "short_signal"] = True
        return df
