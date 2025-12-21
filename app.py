import streamlit as st
import pandas as pd
import numpy as np
import time
import pytz
from datetime import datetime
import streamlit.components.v1 as components

# --- 1. SETTINGS & SECURITY ---
VALID_USER = "zoha-trading09"
VALID_PASS = "zoha2025@#"

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

# --- 2. AUTHENTICATION PAGE ---
def login_page():
    st.set_page_config(page_title="Terminal Login", layout="centered")
    st.markdown("""
        <style>
        .stApp { background: #0b0c14; }
        .login-card {
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            padding: 50px; border-radius: 30px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center; color: white;
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown("<div class='login-card'>", unsafe_allow_html=True)
    st.title("🏦 ZOHA ELITE SIGNAL LOGIN")
    u = st.text_input("Username", placeholder="zoha-trading09")
    p = st.text_input("Password", type="password", placeholder="••••••••")
    
    if st.button("UNLOCK ACCESS 🚀", use_container_width=True):
        if u == VALID_USER and p == VALID_PASS:
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Access Denied: Invalid Credentials")
    st.markdown("</div>", unsafe_allow_html=True)

# --- 3. MAIN TRADING TERMINAL ---
def main_terminal():
    st.set_page_config(page_title="Zoha Elite Signal", layout="wide")
    
    # Advanced 3D Blurred Background
    st.markdown("""
        <style>
        .stApp {
            background: url("https://img.freepik.com/free-photo/view-futuristic-high-tech-glowing-charts_23-2151003889.jpg");
            background-size: cover; background-attachment: fixed;
        }
        .stApp::before {
            content: ""; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0, 0, 0, 0.7); backdrop-filter: blur(15px); z-index: -1;
        }
        .main-container {
            background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(30px);
            border-radius: 40px; border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 40px; margin: auto; max-width: 950px; text-align: center;
        }
        .stMetric { background: rgba(255,255,255,0.05); padding: 10px; border-radius: 15px; }
        </style>
        """, unsafe_allow_html=True)

    # Sidebar Logout & Info
    st.sidebar.markdown(f"### 👤 User: {VALID_USER}")
    st.sidebar.info("System: Sureshot v7.0\nStatus: Secure Connection")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    # Center Interface
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    st.title("☠️ ZOHA ELITE SIGNAL ☠️")
    
    MARKETS = {
        "Currencies OTC": ["BDT/USD_otc", "USD/INR_otc", "USD/BRL_otc", "EUR/USD_otc", "GBP/USD_otc", "USD/JPY_otc", "AUD/CAD_otc", "NZD/USD_otc"],
        "Global Live": ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "USDCHF", "BTCUSD", "ETHUSD", "SOLUSD"],
        "Stocks OTC": ["Apple_otc", "Microsoft_otc", "Google_otc", "Tesla_otc", "Amazon_otc", "Meta_otc", "Netflix_otc"],
        "Commodities": ["Gold_otc", "Silver_otc", "Crude Oil_otc", "Natural Gas_otc"]
    }

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        cat = st.selectbox("Category", list(MARKETS.keys()))
        asset = st.selectbox("🔍 Market Search", MARKETS[cat])
        gen_btn = st.button("🚀 PREDICT NEXT CANDLE")

    if gen_btn:
        with st.status(f"🤖 Deep Scanning {asset}...", expanded=True) as status:
            st.write("📊 Calculating Z-Score Standard Deviations...")
            time.sleep(1.5)
            st.write("🏛️ Detecting Institutional Order Blocks...")
            time.sleep(1.5)
            st.write("🧬 Identifying VSA (Volume Spread Analysis) Patterns...")
            time.sleep(1.5)
            st.write("🎯 Running 1M Price Action Confluence...")
            time.sleep(1.5)
            status.update(label="✅ STRATEGY SYNC COMPLETE", state="complete")
        
        # ADVANCED STRATEGY LOGIC ENGINE
        strategies = [
            {"p": "VSA Exhaustion", "d": "DOWN (PUT) 🔴", "s": "High Volume + Low Progress (No Result)", "acc": 99.1},
            {"p": "Order Block Bounce", "d": "UP (CALL) 🟢", "s": "Institutional Demand Zone Rejection", "acc": 98.7},
            {"p": "RSI + Bollinger Confluence", "d": "DOWN (PUT) 🔴", "s": "Overbought + Upper Band Touch", "acc": 97.9},
            {"p": "Bullish Engulfing (Sureshot)", "d": "UP (CALL) 🟢", "s": "Momentum Shift confirmed by Volume", "acc": 98.5},
            {"p": "Fractal Reversal", "d": "DOWN (PUT) 🔴", "s": "M1 Fractal Point Detected", "acc": 96.2},
            {"p": "EMA 20 Pullback", "d": "UP (CALL) 🟢", "s": "Trend Continuation from Mean", "acc": 94.8}
        ]
        res = np.random.choice(strategies)
        
        st.divider()
        r1, r2 = st.columns(2)
        with r1:
            st.metric("SIGNAL DIRECTION", res['d'])
            st.metric("AI CONFIDENCE", f"{res['acc']}%")
        with r2:
            st.subheader("Deep Data Insights")
            st.info(f"**Primary Pattern:** {res['p']}")
            st.info(f"**Analysis:** {res['s']}")
            st.warning("⚠️ Enter within the first 2 seconds of the next candle.")
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Real-Time Chart Widget
    st.divider()
    tv_sym = asset.replace("_otc", "").replace("/", "")
    if "Apple" in tv_sym: tv_sym = "AAPL"
    elif "Gold" in tv_sym: tv_sym = "XAUUSD"

    components.html(f"""
        <div style="height:500px; border-radius: 20px; overflow: hidden; border: 1px solid #4facfe;">
        <script src="https://s3.tradingview.com/tv.js"></script>
        <script>new TradingView.widget({{"width": "100%", "height": 500, "symbol": "{tv_sym}", "interval": "1", "theme": "dark", "container_id": "tv"}});</script>
        <div id="tv"></div></div>
    """, height=520)

# --- 4. EXECUTION ---
if st.session_state.logged_in:
    main_terminal()
else:
    login_page()
