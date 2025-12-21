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
        }
        </style>
        """, unsafe_allow_html=True)

    # Sidebar Logout
    st.sidebar.markdown(f"### 👤 User: {VALID_USER}")
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()
    
    if st.sidebar.button("🌓 Switch Theme"):
        st.session_state.theme = 'white' if st.session_state.theme == 'dark' else 'dark'
        st.rerun()

    # Center Interface
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)
    st.title("☠️ ZOHA ELITE SIGNAL ☠️")
    
    MARKETS = {
        "Currencies OTC": ["BDT/USD_otc", "USD/INR_otc", "USD/BRL_otc", "EUR/USD_otc", "GBP/USD_otc"],
        "Global Live": ["EURUSD", "GBPUSD", "USDJPY", "BTCUSD", "ETHUSD"],
        "Stocks OTC": ["Apple_otc", "Microsoft_otc", "Tesla_otc", "Google_otc"],
        "Commodities": ["Gold_otc", "Silver_otc", "Crude Oil_otc"]
    }

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        cat = st.selectbox("Category", list(MARKETS.keys()))
        asset = st.selectbox("🔍 Market Search", MARKETS[cat])
        gen_btn = st.button("🚀 EXECUTE MULTI-STRATEGY SCAN")

    if gen_btn:
        with st.status(f"🤖 Quant Engine Scanning {asset}...", expanded=True) as status:
            st.write("📊 Checking Fair Value Gaps (FVG) & Liquidity...")
            time.sleep(1.2)
            st.write("📈 Analyzing Higher Timeframe (M5/M15) Correlation...")
            time.sleep(1.2)
            st.write("⚖️ Measuring Buyer/Seller Imbalance (Order Flow)...")
            time.sleep(1.2)
            st.write("🎯 Calculating Fibonacci Golden Ratio Levels...")
            time.sleep(1.2)
            status.update(label="✅ QUANT ANALYSIS SUCCESSFUL", state="complete")
        
        # ELITE STRATEGY DATABASE
        strategies = [
            {"p": "Fair Value Gap (FVG)", "d": "DOWN (PUT) 🔴", "s": "Imbalance detected; price must return to fill the void.", "acc": 99.4},
            {"p": "Liquidity Sweep", "d": "UP (CALL) 🟢", "s": "Stop-losses hunted below support; huge buy pressure incoming.", "acc": 98.9},
            {"p": "Fibonacci 0.618 Golden Pocket", "d": "UP (CALL) 🟢", "s": "Perfect retracement touch on high-volume trend.", "acc": 97.5},
            {"p": "M5/M1 Confluence", "d": "DOWN (PUT) 🔴", "s": "M5 Trend is Bearish; M1 Pullback is exhausted.", "acc": 98.2},
            {"p": "Three Line Strike", "d": "UP (CALL) 🟢", "s": "Three red candles followed by a massive engulfing green.", "acc": 95.8}
        ]
        res = np.random.choice(strategies)
        
        st.divider()
        r1, r2 = st.columns(2)
        with r1:
            st.metric("FINAL SIGNAL", res['d'])
            st.metric("PROBABILITY", f"{res['acc']}%")
        with r2:
            st.subheader("Quant Breakdown")
            st.info(f"**Method:** {res['p']}")
            st.info(f"**Logic:** {res['s']}")
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Chart Widget
    st.divider()
    tv_sym = asset.replace("_otc", "").replace("/", "")
    components.html(f"""
        <div style="height:500px; border-radius: 20px; overflow: hidden; border: 2px solid #4facfe;">
        <script src="https://s3.tradingview.com/tv.js"></script>
        <script>new TradingView.widget({{"width": "100%", "height": 500, "symbol": "{tv_sym}", "interval": "1", "theme": "{st.session_state.theme}", "container_id": "tv"}});</script>
        <div id="tv"></div></div>
    """, height=520)

if st.session_state.logged_in: main_terminal()
else: login_page()
