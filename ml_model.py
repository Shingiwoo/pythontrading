from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np

def train_ml_signal(df):
    df['lag_return'] = df['close'].pct_change().shift(1)
    df['volatility'] = df['close'].rolling(20).std().shift(1)

    features = df[['rsi', 'macd', 'atr', 'bb_width', 'lag_return', 'volatility']].dropna()
    target = (df['close'].shift(-5) > df['close']).astype(int).loc[features.index]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    preds = np.zeros(len(features))

    for train_idx, _ in skf.split(features, target):
        model.fit(features.iloc[train_idx], target.iloc[train_idx])
        preds[train_idx] = model.predict(features.iloc[train_idx])

    df['ml_signal'] = 0
    df.loc[features.index, 'ml_signal'] = preds
    return df