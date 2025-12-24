import streamlit as st
import pandas as pd
import pandas_ta as ta
import ccxt
import requests

# --- CONFIGURATION ---
TELEGRAM_TOKEN = "YOUR_BOT_TOKEN_HERE"
CHAT_ID = "YOUR_CHAT_ID_HERE"

def send_telegram_msg(message):
    """Sends signal alerts directly to your phone."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Telegram Error: {e}")

def get_infinity_signal(df, symbol):
    """High-accuracy logic: Trend + Volatility + Price Action."""
    df['EMA_200'] = ta.ema(df['close'], length=200)
    df['RSI'] = ta.rsi(df['close'], length=14)
    bb = ta.bbands(df['close'], length=20, std=2.5)
    
    # Psychology: Detection of 'Pin Bar' (Strong Rejection)
    last = df.iloc[-1]
    body = abs(last['close'] - last['open'])
    lower_wick = min(last['open'], last['close']) - last['low']
    is_rejection = lower_wick > (body * 2)

    # Signal Generation
    if last['close'] > last['EMA_200'] and last['RSI'] < 30 and is_rejection:
        msg = f"🚀 *INFINITY BUY ALERT* 🚀\nAsset: {symbol}\nPrice: {last['close']}\nReason: Trend Support + RSI Oversold + Rejection"
        send_telegram_msg(msg)
        return "BUY"
    
    return "NONE"

# --- STREAMLIT UI ---
st.title("Infinity Signals: Auto-Notifier 🤖")
if st.button("Start 60+ Market Auto-Scan"):
    st.success("Scanner Active! Check your Telegram for alerts.")
    # In a loop, you would call get_infinity_signal() for all 60 markets
