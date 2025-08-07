import streamlit as st
import pandas as pd
import numpy as np
import os
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import matplotlib.pyplot as plt

st.set_page_config(page_title="Scalping Strategy Futures", layout="wide")
st.title("⚡ Scalping Strategy - Binance Futures Style")

st.sidebar.header("🔧 Pengaturan Strategi")

# === Konfigurasi Strategi ===
ema_fast = st.sidebar.slider("EMA Fast", 5, 50, 9)
ema_slow = st.sidebar.slider("EMA Slow", 10, 100, 21)
rsi_period = st.sidebar.slider("RSI Period", 5, 20, 14)
rsi_long_thres = st.sidebar.slider("RSI Min for Long", 40, 70, 50)
rsi_short_thres = st.sidebar.slider("RSI Max for Short", 30, 60, 45)

sl_tp_mode = st.sidebar.selectbox("SL/TP Mode", ["ATR", "Persentase"])
sl_mult = st.sidebar.slider("Stop Loss (ATR× / %)", 0.2, 5.0, 1.0)
tp_mult = st.sidebar.slider("Take Profit (ATR× / %)", 0.5, 10.0, 1.5)

use_trailing = st.sidebar.checkbox("Gunakan Trailing SL", value=False)
max_hold = st.sidebar.slider("Max Hold Candle", 5, 50, 20)

capital = st.sidebar.number_input("Initial Capital (USD)", value=1000.0)
risk_pct = st.sidebar.slider("Risk per Trade (%)", 0.5, 10.0, 1.0) / 100
leverage = st.sidebar.slider("Leverage", 1, 50, 10)

# === Data loader ===
data_dir = "./data"
files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
file = st.selectbox("📁 Pilih file data harga", files)

if file:
    df = pd.read_csv(os.path.join(data_dir, file), parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["ema_fast"] = EMAIndicator(df["close"], ema_fast).ema_indicator()
    df["ema_slow"] = EMAIndicator(df["close"], ema_slow).ema_indicator()
    df["rsi"] = RSIIndicator(df["close"], rsi_period).rsi()
    df["atr"] = AverageTrueRange(df["high"], df["low"], df["close"]).average_true_range()

    df["long_signal"] = (df["ema_fast"] > df["ema_slow"]) & (df["rsi"] > rsi_long_thres)
    df["short_signal"] = (df["ema_fast"] < df["ema_slow"]) & (df["rsi"] < rsi_short_thres)

    if st.button("🚀 Jalankan Backtest Scalping Strategy"):
        trades = []
        capital_now = capital
        in_position = False
        entry = None
        sl = None
        tp = None
        pnl = 0
        direction = None
        size = 0
        entry_index = 0

        for i in range(1, len(df)):
            row = df.iloc[i]
            price = row["close"]

            if not in_position:
                if row["long_signal"]:
                    direction = "long"
                    entry = price
                    atr = row["atr"]
                    sl = entry - (sl_mult * atr if sl_tp_mode == "ATR" else entry * sl_mult / 100)
                    tp = entry + (tp_mult * atr if sl_tp_mode == "ATR" else entry * tp_mult / 100)
                    size = (capital_now * risk_pct * leverage) / abs(entry - sl)
                    entry_index = i
                    in_position = True
                elif row["short_signal"]:
                    direction = "short"
                    entry = price
                    atr = row["atr"]
                    sl = entry + (sl_mult * atr if sl_tp_mode == "ATR" else entry * sl_mult / 100)
                    tp = entry - (tp_mult * atr if sl_tp_mode == "ATR" else entry * tp_mult / 100)
                    size = (capital_now * risk_pct * leverage) / abs(entry - sl)
                    entry_index = i
                    in_position = True
                continue

            # Exit Logic
            hold = i - entry_index if entry_index is not None else 0
            price = row["close"]
            exit_cond = False

            if direction == "long" and entry is not None:
                if price <= sl:
                    pnl = (price - entry) * size
                    exit_cond = "SL"
                elif price >= tp:
                    pnl = (price - entry) * size
                    exit_cond = "TP"
            else:
                if price >= sl:
                    pnl = (entry - price) * size
                    exit_cond = "SL"
                elif price <= tp:
                    pnl = (entry - price) * size
                    exit_cond = "TP"

            if hold >= max_hold and not exit_cond:
                pnl = (price - entry) * size if direction == "long" else (entry - price) * size
                exit_cond = "TIME"

            if exit_cond:
                trades.append({
                    "type": direction,
                    "entry": entry,
                    "exit": price,
                    "pnl": pnl,
                    "exit_reason": exit_cond
                })
                capital_now += pnl
                in_position = False

        df_trades = pd.DataFrame(trades)
        st.success(f"✅ Backtest selesai: {len(df_trades)} trade")

        equity = [capital]
        for p in df_trades["pnl"]:
            equity.append(equity[-1] + p)

        st.subheader("📊 Ringkasan Hasil")
        col1, col2, col3 = st.columns(3)
        col1.metric("Final Capital", f"${equity[-1]:.2f}")
        win_rate = len(df_trades[df_trades['pnl'] > 0]) / len(df_trades) * 100 if len(df_trades) > 0 else 0
        col2.metric("Win Rate", f"{win_rate:.2f}%")
        pf = df_trades[df_trades["pnl"] > 0]["pnl"].sum() / abs(df_trades[df_trades["pnl"] < 0]["pnl"].sum()) if not df_trades[df_trades["pnl"] < 0].empty else float("inf")
        col3.metric("Profit Factor", f"{pf:.2f}")

        st.subheader("📈 Equity Curve")
        fig, ax = plt.subplots()
        ax.plot(equity)
        ax.set_xlabel("Trade #")
        ax.set_ylabel("Equity")
        st.pyplot(fig)

        st.subheader("📋 Trade History")
        st.dataframe(df_trades)

        csv = df_trades.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download CSV", csv, "scalping_trades.csv", "text/csv")