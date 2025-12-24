import streamlit as st
import pandas as pd
import numpy as np
import ccxt

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Infinity Pro Scanner & Backtester",
    layout="wide"
)

# =====================================================
# BINANCE EXCHANGE (SAFE MODE)
# =====================================================
exchange = ccxt.binance({
    "enableRateLimit": True
})

# =====================================================
# INDICATORS (PURE PANDAS – NO pandas_ta)
# =====================================================
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def bollinger_bands(series, period=20, std=2.5):
    sma = series.rolling(period).mean()
    std_dev = series.rolling(period).std()
    upper = sma + std * std_dev
    lower = sma - std * std_dev
    return upper, lower

# =====================================================
# DATA FETCH
# =====================================================
def fetch_historical_data(symbol, timeframe="15m", limit=500):
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=[
            "timestamp", "open", "high", "low", "close", "volume"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        return None

# =====================================================
# STRATEGY LOGIC
# =====================================================
def apply_strategy(df):
    df["EMA_200"] = ema(df["close"], 200)
    df["RSI"] = rsi(df["close"], 14)
    df["BBU"], df["BBL"] = bollinger_bands(df["close"])

    df["BUY"] = (
        (df["close"] > df["EMA_200"]) &
        (df["RSI"] < 30) &
        (df["close"] <= df["BBL"])
    )

    df["SELL"] = (
        (df["close"] < df["EMA_200"]) &
        (df["RSI"] > 70) &
        (df["close"] >= df["BBU"])
    )

    return df

# =====================================================
# BACKTEST ENGINE
# =====================================================
def run_backtest(df, forward_bars=5):
    wins = 0
    trades = 0

    for i in range(len(df) - forward_bars):
        entry = df["close"].iloc[i]
        future = df["close"].iloc[i + forward_bars]

        if df["BUY"].iloc[i]:
            trades += 1
            if future > entry:
                wins += 1

        elif df["SELL"].iloc[i]:
            trades += 1
            if future < entry:
                wins += 1

    winrate = (wins / trades * 100) if trades > 0 else 0
    return winrate, trades

# =====================================================
# UI
# =====================================================
st.title("📊 Infinity Pro – Scanner & Backtester")

tab1, tab2 = st.tabs(["🔴 Live Scanner", "🧪 Strategy Backtester"])

# =====================================================
# LIVE SCANNER (DEMO)
# =====================================================
with tab1:
    st.subheader("Live Market Scanner (Demo Mode)")
    st.info("Live scanning logic can be integrated here (Crypto / OTC / AI Engine).")

    if st.button("Start Scan"):
        st.success("Scanner running... (Demo)")

# =====================================================
# BACKTESTER
# =====================================================
with tab2:
    st.subheader("Historical Strategy Accuracy Test")

    symbol = st.selectbox(
        "Select Market",
        [
            "BTC/USDT",
            "ETH/USDT",
            "BNB/USDT",
            "SOL/USDT",
            "XRP/USDT"
        ]
    )

    timeframe = st.selectbox("Timeframe", ["5m", "15m", "1h"])
    forward_bars = st.slider("Bars Ahead for Result", 1, 20, 5)

    if st.button("Run Backtest"):
        with st.spinner("Running backtest..."):
            df = fetch_historical_data(symbol, timeframe)

            if df is not None and len(df) > 250:
                df = apply_strategy(df)
                winrate, trades = run_backtest(df, forward_bars)

                col1, col2 = st.columns(2)
                col1.metric("📈 Win Rate", f"{winrate:.2f}%")
                col2.metric("📊 Total Trades", trades)

                if winrate >= 60:
                    st.success("✅ Strategy shows strong performance.")
                elif winrate >= 50:
                    st.warning("⚠️ Average performance – use risk management.")
                else:
                    st.error("❌ Weak performance – avoid trading.")

                st.subheader("Recent Signals")
                st.dataframe(
                    df[["timestamp", "close", "BUY", "SELL"]].tail(20),
                    use_container_width=True
                )
            else:
                st.error("Not enough data to run backtest.")
