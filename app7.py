import streamlit as st
import pandas as pd
import pandas_ta as ta
import ccxt
import numpy as np

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Infinity Pro: Scanner & Backtester",
    layout="wide"
)

# ===============================
# EXCHANGE (GLOBAL INSTANCE)
# ===============================
exchange = ccxt.binance({
    "enableRateLimit": True
})

# ===============================
# DATA FETCH
# ===============================
def fetch_historical_data(symbol, timeframe="15m", limit=500):
    """
    Fetch OHLCV data safely from Binance
    """
    try:
        bars = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(bars, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return None

# ===============================
# STRATEGY LOGIC
# ===============================
def get_signals_for_backtest(df):
    """
    Apply EMA + RSI + Bollinger Bands strategy
    """
    df['EMA_200'] = ta.ema(df['close'], length=200)
    df['RSI'] = ta.rsi(df['close'], length=14)

    bb = ta.bbands(df['close'], length=20, std=2.5)
    df = pd.concat([df, bb], axis=1)

    df['buy_sig'] = (
        (df['close'] > df['EMA_200']) &
        (df['RSI'] < 30) &
        (df['close'] <= df['BBL_20_2.5'])
    )

    df['sell_sig'] = (
        (df['close'] < df['EMA_200']) &
        (df['RSI'] > 70) &
        (df['close'] >= df['BBU_20_2.5'])
    )

    return df

# ===============================
# BACKTEST ENGINE
# ===============================
def run_backtest(df, target_bars=5):
    wins = 0
    total_trades = 0

    for i in range(len(df) - target_bars):
        entry_price = df['close'].iloc[i]
        future_price = df['close'].iloc[i + target_bars]

        if df['buy_sig'].iloc[i]:
            total_trades += 1
            if future_price > entry_price:
                wins += 1

        elif df['sell_sig'].iloc[i]:
            total_trades += 1
            if future_price < entry_price:
                wins += 1

    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    return win_rate, total_trades

# ===============================
# STREAMLIT UI
# ===============================
st.title("📊 Infinity Pro Scanner + Backtester")

tab1, tab2 = st.tabs(["🔴 Live Scanner", "🧪 Strategy Backtester"])

# ===============================
# TAB 1 – SCANNER (PLACEHOLDER)
# ===============================
with tab1:
    st.header("Live Market Scanner")
    st.info("Live scanner logic can be plugged here (OTC / Crypto / AI Engine)")
    if st.button("Start Scan"):
        st.success("Scanner started (demo mode).")

# ===============================
# TAB 2 – BACKTESTER
# ===============================
with tab2:
    st.header("Historical Strategy Accuracy Test")

    target_symbol = st.selectbox(
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
    bars_forward = st.slider("Bars to Check Result", 1, 20, 5)

    if st.button("Run Accuracy Test"):
        with st.spinner("Analyzing historical data..."):
            df = fetch_historical_data(target_symbol, timeframe)

            if df is not None and len(df) > 200:
                df = get_signals_for_backtest(df)
                win_rate, trades = run_backtest(df, bars_forward)

                col1, col2 = st.columns(2)
                col1.metric("📈 Strategy Win Rate", f"{win_rate:.2f}%")
                col2.metric("📊 Total Signals", trades)

                if win_rate >= 60:
                    st.success("✅ High accuracy — Strategy is viable for this market.")
                elif win_rate >= 50:
                    st.warning("⚠️ Medium accuracy — Use strict risk management.")
                else:
                    st.error("❌ Low accuracy — Strategy NOT recommended.")

                # Optional preview
                st.subheader("Last 20 Signals")
                st.dataframe(
                    df[['ts', 'close', 'buy_sig', 'sell_sig']].tail(20),
                    use_container_width=True
                )
            else:
                st.error("Not enough data to evaluate strategy.")
