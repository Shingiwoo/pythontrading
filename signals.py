from config import MIN_SCORE_THRESHOLD

def generate_signals(df):
    df['long_signal'] = False
    df['short_signal'] = False
    for i in range(1, len(df)):
        score = 0.0
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        if row['ema_22'] > row['ma_22'] and prev['ema_22'] <= prev['ma_22']:
            score += 1.0
        if row['macd'] > row['macd_signal']:
            score += 0.5
        if 40 < row['rsi'] < 70:
            score += 0.5
        if row['ml_signal'] == 1:
            score += 1.5
        if score >= MIN_SCORE_THRESHOLD:
            df.loc[i, 'long_signal'] = True

        score = 0.0
        if row['ema_22'] < row['ma_22'] and prev['ema_22'] >= prev['ma_22']:
            score += 1.0
        if row['macd'] < row['macd_signal']:
            score += 0.5
        if 30 < row['rsi'] < 60:
            score += 0.5
        if row['ml_signal'] == 0:
            score += 1.5
        if score >= MIN_SCORE_THRESHOLD:
            df.loc[i, 'short_signal'] = True
    return df