import streamlit as st
import pandas as pd
import pandas_ta as ta
import ccxt
import numpy as np

# --- SETTINGS ---
st.set_page_config(page_title="Infinity Pro: Scanner & Backtester", layout="wide")

def fetch_historical_data(symbol, limit=500):
    """Fetches a larger dataset for backtesting."""
    exchange = ccxt.binance()
    bars = exchange.fetch_ohlcv(symbol, timeframe='15m', limit=limit)
    df = pd.DataFrame(bars, columns=['ts', 'open', 'high', 'low', 'close', 'vol'])
    return df

def get_signals_for_backtest(df):
    """Applies the strategy to the entire history to calculate win rates."""
    df['EMA_200'] = ta.ema(df['close'], length=200)
    df['RSI'] = ta.rsi(df['close'], length=14)
    bb = ta.bbands(df['close'], length=20, std=2.5)
    df = pd.concat([df, bb], axis=1)
    
    # Define Buy/Sell conditions
    df['buy_sig'] = (df['close'] > df['EMA_200']) & (df['RSI'] < 30) & (df['close'] <= df['BBL_20_2.5'])
    df['sell_sig'] = (df['close'] < df['EMA_200']) & (df['RSI'] > 70) & (df['close'] >= df['BBU_20_2.5'])
    return df

def run_backtest(df, target_bars=5):
    """Calculates accuracy: Did the price go up after a Buy or down after a Sell?"""
    wins = 0
    total_trades = 0
    
    for i in range(len(df) - target_bars):
        if df['buy_sig'].iloc[i]:
            total_trades += 1
            # Win if price is higher after 5 bars
            if df['close'].iloc[i + target_bars] > df['close'].iloc[i]:
                wins += 1
        elif df['sell_sig'].iloc[i]:
            total_trades += 1
            # Win if price is lower after 5 bars
            if df['close'].iloc[i + target_bars] < df['close'].iloc[i]:
                wins += 1
                
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    return win_rate, total_trades

# --- STREAMLIT UI ---
st.title("📊 Infinity Pro Scanner + Backtester")

tab1, tab2 = st.tabs(["Live Scanner", "Strategy Backtester"])

with tab1:
    st.header("Live 60+ Market Scanner")
    if st.button("Start Scan"):
        st.write("Scanning OTC markets... (As coded in previous step)")

with tab2:
    st.header("Historical Accuracy Test")
    target_symbol = st.selectbox("Select Market to Test", ["BTC/USDT", "ETH/USDT", "EUR/USD", "GOLD/USD"])
    
    if st.button("Run Accuracy Test"):
        with st.spinner('Analyzing historical patterns...'):
            data = fetch_historical_data(target_symbol)
            data_with_sigs = get_signals_for_backtest(data)
            win_rate, trades = run_backtest(data_with_sigs)
            
            # Psychology Metrics
            st.metric("Strategy Win Rate", f"{win_rate:.2f}%")
            st.metric("Total Signals Found", trades)
            
            if win_rate > 60:
                st.success("✅ High Accuracy Detected. Strategy is viable for this market.")
            else:
                st.warning("⚠️ Low Accuracy. Avoid this market or adjust EMA/RSI periods.")


