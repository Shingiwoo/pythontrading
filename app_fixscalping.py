import streamlit as st
import pandas as pd
import os
from datetime import datetime
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.volatility import AverageTrueRange, BollingerBands
from ta.momentum import RSIIndicator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Scalping Strategy GUI", layout="wide")
st.title("📈 Scalping Strategy Backtester")

# === Sidebar Settings ===
st.sidebar.header("⚙️ Pengaturan Strategi")

sma_period = st.sidebar.slider("SMA Period", 7, 50, 22)
ema_period = st.sidebar.slider("EMA Period", 5, 50, 22)
rsi_period = st.sidebar.slider("RSI Period", 5, 30, 14)
bb_std = st.sidebar.slider("Bollinger Band StdDev", 1.0, 3.0, 1.4, step=0.1)
score_threshold = st.sidebar.slider("Minimal Skor Sinyal", 1.0, 5.0, 2.0, step=0.1)

use_trailing = st.sidebar.checkbox("Gunakan Trailing SL", True)
trailing_trigger = st.sidebar.slider("Trigger Trailing SL (%)", 0.1, 3.0, 0.5)
trailing_step = st.sidebar.slider("Step Trailing SL (%)", 0.1, 2.0, 0.3)

initial_capital = st.sidebar.number_input("Initial Capital (USD)", value=20.0)
risk_per_trade = st.sidebar.slider("Risk per Trade (%)", 1.0, 10.0, 5.0) / 100
leverage = st.sidebar.slider("Leverage", 1, 50, 10)

# === File Selection ===
data_dir = "./data"
csv_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
selected_file = st.selectbox("Pilih data file", csv_files)

if selected_file:
    df = pd.read_csv(os.path.join(data_dir, selected_file), parse_dates=['timestamp'])
    df.dropna(inplace=True)

    # === Apply Indicators ===
    df['ema'] = EMAIndicator(df['close'], ema_period).ema_indicator()
    df['ma'] = SMAIndicator(df['close'], sma_period).sma_indicator()
    macd = MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['rsi'] = RSIIndicator(df['close'], rsi_period).rsi()
    bb = BollingerBands(df['close'], window=20, window_dev=int(bb_std))
    df['bb_upper'] = bb.bollinger_hband()
    df['bb_lower'] = bb.bollinger_lband()
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
    df['atr'] = AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()

    # === ML Signal ===
    df['lag_ret'] = df['close'].pct_change().shift(1)
    df['vol'] = df['close'].rolling(20).std().shift(1)
    ml_df = df[['rsi', 'macd', 'atr', 'bb_width', 'lag_ret', 'vol']].dropna()
    target = (df['close'].shift(-5) > df['close']).astype(int).loc[ml_df.index]
    model = RandomForestClassifier(n_estimators=100)
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    preds = np.zeros(len(ml_df))
    for train_idx, _ in skf.split(ml_df, target):
        model.fit(ml_df.iloc[train_idx], target.iloc[train_idx])
        preds[train_idx] = model.predict(ml_df.iloc[train_idx])
    df['ml_signal'] = 0
    df.loc[ml_df.index, 'ml_signal'] = preds

    # === Generate Signals ===
    df['long_signal'] = False
    df['short_signal'] = False
    for i in range(1, len(df)):
        score_long = 0
        if df['ema'].iloc[i] > df['ma'].iloc[i] and df['macd'].iloc[i] > df['macd_signal'].iloc[i]:
            score_long += 1
        if 40 < df['rsi'].iloc[i] < 70:
            score_long += 0.5
        if df['ml_signal'].iloc[i] == 1:
            score_long += 1
        if score_long >= score_threshold:
            df.loc[i, 'long_signal'] = True

        score_short = 0
        if df['ema'].iloc[i] < df['ma'].iloc[i] and df['macd'].iloc[i] < df['macd_signal'].iloc[i]:
            score_short += 1
        if 30 < df['rsi'].iloc[i] < 60:
            score_short += 0.5
        if df['ml_signal'].iloc[i] == 0:
            score_short += 1
        if score_short >= score_threshold:
            df.loc[i, 'short_signal'] = True

    # === Backtest ===
    in_position = False
    entry = sl = tp = trailing_sl = direction = None
    size = hold = 0
    long_count = short_count = 0
    capital = initial_capital
    fee_rate = 0.0004
    trades = []

    for i in range(len(df)):
        row = df.iloc[i]
        price = row['close']

        if not in_position:
            if row['long_signal']:
                direction = 'long'
                entry = price
                sl = entry - row['atr'] * 1.5
                tp = entry + row['atr'] * 3.0
                size = (capital * risk_per_trade * leverage) / entry
                in_position = True
                trailing_sl = sl
                hold = 0
                long_count += 1
            elif row['short_signal']:
                direction = 'short'
                entry = price
                sl = entry + row['atr'] * 1.5
                tp = entry - row['atr'] * 3.0
                size = (capital * risk_per_trade * leverage) / entry
                in_position = True
                trailing_sl = sl
                hold = 0
                short_count += 1
            continue

        hold += 1
        if use_trailing and entry:
            profit_pct = (price - entry) / entry * 100 if direction == 'long' else (entry - price) / entry * 100
            if profit_pct >= trailing_trigger:
                if direction == 'long' and trailing_sl is not None:
                    trailing_sl = max(trailing_sl, price - (trailing_step / 100) * price)
                elif direction == 'short' and trailing_sl is not None:
                    trailing_sl = min(trailing_sl, price + (trailing_step / 100) * price)

        exit_cond = False
        if direction == 'long' and (price <= trailing_sl or price >= tp or hold > 100):
            exit_cond = True
        elif direction == 'short' and (price >= trailing_sl or price <= tp or hold > 100):
            exit_cond = True

        if exit_cond and entry and size > 0:
            if direction == 'long':
                raw_pnl = (price - entry) * size
                fee = (entry + price) * fee_rate * size
                pnl = raw_pnl - fee
            else:
                raw_pnl = (entry - price) * size
                fee = (entry + price) * fee_rate * size
                pnl = raw_pnl - fee

            capital += pnl
            trades.append({'entry': entry, 'exit': price, 'type': direction, 'pnl': pnl})
            in_position = False
            entry = sl = tp = trailing_sl = direction = None
            size = hold = 0

    # === Result Display ===
    st.success(f"✅ Backtest selesai: {len(trades)} trades")
    df_trades = pd.DataFrame(trades)
    wins = df_trades[df_trades['pnl'] > 0]
    losses = df_trades[df_trades['pnl'] <= 0]
    win_rate = len(wins) / len(df_trades) * 100 if len(df_trades) else 0
    profit_factor = wins['pnl'].sum() / abs(losses['pnl'].sum()) if not losses.empty else float('inf')

    col1, col2, col3 = st.columns(3)
    col1.metric("Final Capital", f"${capital:.2f}")
    col2.metric("Win Rate", f"{win_rate:.2f}%")
    col3.metric("Profit Factor", f"{profit_factor:.2f}")
    col4, col5 = st.columns(2)
    col4.metric("Jumlah Long Trades", long_count)
    col5.metric("Jumlah Short Trades", short_count)

    st.subheader("📈 Equity Curve")
    equity = [initial_capital]
    for t in trades:
        equity.append(equity[-1] + t['pnl'])
    fig, ax = plt.subplots()
    ax.plot(equity)
    ax.set_ylabel("Equity")
    ax.set_xlabel("Trade #")
    st.pyplot(fig)

    st.subheader("📊 Trade History")
    st.dataframe(df_trades)
    csv = df_trades.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Trades CSV", csv, "trades.csv", "text/csv")