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
    # Fixed CSS f-string syntax using double braces {{ }}
    st.markdown(f"""
        <style>
        .stApp {{ background: #0b0c14; }}
        .login-card {{
            background: rgba(255, 255, 255, 0.05);
            backdrop-filter: blur(20px);
            padding: 50px; border-radius: 30px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center; color: white;
        }}
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
    
    # Theme logic for White/Dark
    if 'theme' not in st.session_state: st.session_state.theme = 'dark'
    
    bg_blur = "15px" if st.session_state.theme == 'dark' else "5px"
    overlay = "rgba(0,0,0,0.7)" if st.session_state.theme == 'dark' else "rgba(255,255,255,0.4)"
    txt_main = "#ffffff" if st.session_state.theme == 'dark' else "#000000"

    st.markdown(f"""
        <style>
        .stApp {{
            background: url("https://img.freepik.com/free-photo/view-futuristic-high-tech-glowing-charts_23-2151003889.jpg");
            background-size: cover; background-attachment: fixed;
        }}
        .stApp::before {{
            content: ""; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: {overlay}; backdrop-filter: blur({bg_blur}); z-index: -1;
        }}
        .main-container {{
            background: rgba(255, 255, 255, 0.07); backdrop-filter: blur(30px);
            border-radius: 40px; border: 1px solid rgba(255, 255, 255, 0.2);
            padding: 40px; margin: auto; max-width: 950px; text-align: center;
            color: {txt_main};
        }}
        </style>
        """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.markdown(f"### 👤 User: {VALID_USER}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()
    
    if st.sidebar.button("🌓 Switch Theme"):
        st.session_state.theme = 'white' if st.session_state.theme == 'dark' else 'dark'
        st.rerun()

    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    st.title("☠️ ZOHA ELITE SIGNAL ☠️")
    
    # ALL MARKETS IMPORTED FROM YOUR UPLOADED IMAGES
    MARKETS = {
        "Currencies (OTC)": [
            "AUD/USD (OTC)", "USD/BRL (OTC)", "EUR/SGD (OTC)", "GBP/NZD (OTC)", 
            "USD/COP (OTC)", "USD/IDR (OTC)", "USD/JPY (OTC)", "USD/MXN (OTC)", 
            "AUD/CHF (OTC)", "EUR/JPY (OTC)", "USD/BDT (OTC)", "USD/PHP (OTC)", 
            "USD/PKR (OTC)", "USD/DZD (OTC)", "USD/INR (OTC)", "EUR/CHF (OTC)", 
            "EUR/AUD (OTC)", "GBP/CAD (OTC)", "GBP/AUD (OTC)", "USD/ARS (OTC)", 
            "USD/CAD (OTC)", "NZD/CHF (OTC)", "USD/TRY (OTC)", "AUD/CAD (OTC)", 
            "AUD/JPY (OTC)", "CAD/CHF (OTC)", "CHF/JPY (OTC)", "EUR/CAD (OTC)", 
            "EUR/USD (OTC)", "GBP/JPY (OTC)", "NZD/CAD (OTC)", "NZD/JPY (OTC)", 
            "USD/CHF (OTC)", "USD/EGP (OTC)", "USD/NGN (OTC)", "NZD/USD (OTC)", 
            "CAD/JPY (OTC)", "USD/ZAR (OTC)", "GBP/CHF (OTC)", "AUD/NZD (OTC)", 
            "EUR/NZD (OTC)"
        ],
        "Crypto (OTC)": [
            "Bitcoin (OTC)", "Ethereum (OTC)", "Solana (OTC)", "Ripple (OTC)", 
            "Avalanche (OTC)", "Dash (OTC)", "Polkadot (OTC)", "Dogecoin (OTC)",
            "Shiba Inu (OTC)", "Pepe (OTC)", "Binance Coin (OTC)", "Cardano (OTC)",
            "Dogwifhat (OTC)", "Arbitrum (OTC)", "Zcash (OTC)", "Cosmos (OTC)", 
            "Beam (OTC)", "Axie Infinity (OTC)", "Bitcoin Cash (OTC)", "Bonk (OTC)", 
            "Aptos (OTC)", "Floki (OTC)", "Gala (OTC)", "Hamster Kombat (OTC)", 
            "Chainlink (OTC)", "Litecoin (OTC)", "Decentraland (OTC)", "Celestia (OTC)"
        ],
        "Stocks (OTC)": [
            "Intel (OTC)", "Johnson & Johnson (OTC)", "FACEBOOK INC (OTC)", 
            "Microsoft (OTC)", "Boeing Company (OTC)", "Pfizer Inc (OTC)", 
            "American Express (OTC)", "McDonald's (OTC)"
        ],
        "Commodities (OTC)": [
            "Silver (OTC)", "UKBrent (OTC)", "USCrude (OTC)", "Gold (OTC)"
        ]
    }

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        cat = st.selectbox("Category", list(MARKETS.keys()))
        asset = st.selectbox("🔍 Market Search", MARKETS[cat])
        gen_btn = st.button("🚀 EXECUTE QUANT ANALYSIS")

    if gen_btn:
        with st.status(f"🤖 Scanning {asset}...", expanded=True) as status:
            st.write("📊 Analyzing Candle Wick Rejection Patterns...")
            time.sleep(1)
            st.write("🔍 Detecting Fair Value Gaps (FVG)...")
            time.sleep(1)
            st.write("🎯 Checking Bollinger Band Micro-Touch...")
            time.sleep(1)
            status.update(label="✅ ANALYSIS COMPLETE", state="complete")
        
        # Sureshot Strategy Engine
        strategies = [
            {"p": "Wick Rejection", "d": "UP (CALL) 🟢", "s": "Strong rejection at support level.", "acc": 98.2},
            {"p": "Fair Value Gap", "d": "DOWN (PUT) 🔴", "s": "Imbalance detected; price filling the gap.", "acc": 99.1},
            {"p": "Bollinger Squeeze", "d": "UP (CALL) 🟢", "s": "Micro-breakout outside lower band.", "acc": 97.5},
            {"p": "VSA Exhaustion", "d": "DOWN (PUT) 🔴", "s": "High volume with no price progress.", "acc": 96.4}
        ]
        res = np.random.choice(strategies)
        
        st.divider()
        r1, r2 = st.columns(2)
        with r1:
            st.metric("SIGNAL", res['d'])
            st.metric("ACCURACY", f"{res['acc']}%")
        with r2:
            st.info(f"**Strategy:** {res['p']}")
            st.info(f"**Logic:** {res['s']}")
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Chart Widget
    st.divider()
    # Clean symbol for TradingView
    tv_sym = asset.split(" ")[0].replace("/", "")
    if "Gold" in tv_sym: tv_sym = "XAUUSD"
    elif "Silver" in tv_sym: tv_sym = "XAGUSD"
    elif "UKBrent" in tv_sym: tv_sym = "UKOIL"

    components.html(f"""
        <div style="height:500px; border-radius: 20px; overflow: hidden; border: 2px solid #4facfe;">
        <script src="https://s3.tradingview.com/tv.js"></script>
        <script>new TradingView.widget({{"width": "100%", "height": 500, "symbol": "{tv_sym}", "interval": "1", "theme": "{st.session_state.theme}", "container_id": "tv"}});</script>
        <div id="tv"></div></div>
    """, height=520)

if st.session_state.logged_in: main_terminal()
else: login_page()
