from config import RISK_PER_TRADE, LEVERAGE, MAX_HOLD_BARS, TRAILING_SL_ENABLED, TRAILING_TRIGGER, TRAILING_STEP

def backtest(df, initial_capital=20.0):
    capital = initial_capital
    trades = []
    in_position = False
    trailing_sl = None

    for i in range(len(df)):
        row = df.iloc[i]
        price = row['close']

        if not in_position:
            if row['long_signal']:
                entry = price
                direction = 'long'
                atr = row['atr']
                sl = entry - atr * 1.5
                tp = entry + atr * 3.0
                size = (capital * RISK_PER_TRADE * LEVERAGE) / entry
                in_position = True
                hold = 0
                trailing_sl = sl

            elif row['short_signal']:
                entry = price
                direction = 'short'
                atr = row['atr']
                sl = entry + atr * 1.5
                tp = entry - atr * 3.0
                size = (capital * RISK_PER_TRADE * LEVERAGE) / entry
                in_position = True
                hold = 0
                trailing_sl = sl
            continue

        if in_position and TRAILING_SL_ENABLED:
            if direction == 'long':
                profit_pct = ((price - entry) / entry) * 100
                if profit_pct >= TRAILING_TRIGGER:
                    new_sl = price - (TRAILING_STEP / 100) * price
                    trailing_sl = max(trailing_sl, new_sl)
            elif direction == 'short':
                profit_pct = ((entry - price) / entry) * 100
                if profit_pct >= TRAILING_TRIGGER:
                    new_sl = price + (TRAILING_STEP / 100) * price
                    trailing_sl = min(trailing_sl, new_sl)

        hold += 1
        exit_trade = False
        if direction == 'long':
            if price <= trailing_sl or price >= tp or hold > MAX_HOLD_BARS:
                pnl = (price - entry) * size
                exit_trade = True
        else:
            if price >= trailing_sl or price <= tp or hold > MAX_HOLD_BARS:
                pnl = (entry - price) * size
                exit_trade = True

        if exit_trade:
            capital += pnl
            trades.append({'type': direction, 'entry': entry, 'exit': price, 'pnl': pnl, 'exit_reason': 'TP/SL/Trailing'})
            in_position = False
            trailing_sl = None

    return trades, capital