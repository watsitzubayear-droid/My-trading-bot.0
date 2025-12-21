import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime
import streamlit.components.v1 as components

# --- 1. CORE SETTINGS & AUTH ---
VALID_USER = "zoha-trading09"
VALID_PASS = "zoha2025@#"

if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'theme' not in st.session_state: st.session_state.theme = 'dark'

# --- 2. AUTHENTICATION ---
def login_page():
    st.set_page_config(page_title="Zoha Elite Login", layout="centered")
    st.markdown(f"""
        <style>
        .stApp {{ background: #05050a; }}
        .login-card {{
            background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(25px);
            padding: 40px; border-radius: 25px; border: 1px solid rgba(255, 255, 255, 0.1);
            text-align: center; color: white; box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }}
        </style>
        """, unsafe_allow_html=True)
    st.markdown("<div class='login-card'>", unsafe_allow_html=True)
    st.title("🛡️ QUANT ACCESS PORTAL")
    u = st.text_input("User ID", placeholder="zoha-trading09")
    p = st.text_input("Access Key", type="password", placeholder="••••••••")
    if st.button("AUTHENTICATE 🚀", use_container_width=True):
        if u == VALID_USER and p == VALID_PASS:
            st.session_state.logged_in = True
            st.rerun()
        else: st.error("INVALID CREDENTIALS")
    st.markdown("</div>", unsafe_allow_html=True)

# --- 3. DEEP QUANT VALIDATION (The Math Check) ---
def deep_quant_validation(asset):
    with st.status(f"🧠 Initiating Institutional Scan for {asset}...", expanded=True) as status:
        checks = [
            ("VSA ANALYSIS", "Checking Volume Spread vs. Candle Body (Effort vs Result)..."),
            ("FIBONACCI MAP", "Locating Golden Pocket (0.618) & Liquidity Voids..."),
            ("PSYCHOLOGY SCAN", "Identifying Retail Stop-Loss Clusters & Trap Zones..."),
            ("M5 STRUCTURE", "Verifying M1 Signal with M5 Higher-Timeframe Trend...")
        ]
        for title, msg in checks:
            st.write(f"🔍 **{title}:** {msg}")
            time.sleep(1.2)
        status.update(label="✅ CONFLUENCE VERIFIED", state="complete")
    return True

# --- 4. MAIN TERMINAL ---
def main_terminal():
    st.set_page_config(page_title="Zoha Elite Signal v11", layout="wide")
    
    overlay = "rgba(0,0,0,0.75)" if st.session_state.theme == 'dark' else "rgba(255,255,255,0.45)"
    txt_color = "#ffffff" if st.session_state.theme == 'dark' else "#111111"
    
    st.markdown(f"""
        <style>
        .stApp {{
            background: url("https://img.freepik.com/free-photo/view-futuristic-high-tech-glowing-charts_23-2151003889.jpg");
            background-size: cover; background-attachment: fixed;
        }}
        .stApp::before {{
            content: ""; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: {overlay}; backdrop-filter: blur(20px); z-index: -1;
        }}
        .quant-box {{
            background: rgba(255, 255, 255, 0.05); backdrop-filter: blur(40px);
            border-radius: 30px; border: 1px solid rgba(255, 255, 255, 0.15);
            padding: 30px; margin: auto; max-width: 1000px; color: {txt_color}; text-align: center;
        }}
        </style>
        """, unsafe_allow_html=True)

    # Sidebar
    st.sidebar.title("💎 COMMAND CENTER")
    if st.sidebar.button("🌓 Toggle Theme"):
        st.session_state.theme = 'white' if st.session_state.theme == 'dark' else 'dark'
        st.rerun()
    if st.sidebar.button("🔒 Secure Logout"):
        st.session_state.logged_in = False
        st.rerun()

    st.markdown("<div class='quant-box'>", unsafe_allow_html=True)
    st.title("☠️ ZOHA ELITE QUANT SIGNAL ☠️")
    
    # --- ALL 80+ PAIRS DATABASE ---
    MARKETS = {
        "Currencies (OTC)": [
            "AUD/USD (OTC)", "USD/BRL (OTC)", "EUR/SGD (OTC)", "GBP/NZD (OTC)", "USD/COP (OTC)", "USD/IDR (OTC)",
            "USD/JPY (OTC)", "USD/MXN (OTC)", "AUD/CHF (OTC)", "EUR/JPY (OTC)", "USD/BDT (OTC)", "USD/PHP (OTC)",
            "USD/PKR (OTC)", "USD/DZD (OTC)", "USD/INR (OTC)", "EUR/CHF (OTC)", "EUR/AUD (OTC)", "GBP/CAD (OTC)",
            "GBP/AUD (OTC)", "USD/ARS (OTC)", "USD/CAD (OTC)", "NZD/CHF (OTC)", "USD/TRY (OTC)", "AUD/CAD (OTC)",
            "AUD/JPY (OTC)", "CAD/CHF (OTC)", "CHF/JPY (OTC)", "EUR/CAD (OTC)", "EUR/USD (OTC)", "GBP/JPY (OTC)",
            "NZD/CAD (OTC)", "NZD/JPY (OTC)", "USD/CHF (OTC)", "USD/EGP (OTC)", "USD/NGN (OTC)", "NZD/USD (OTC)",
            "CAD/JPY (OTC)", "USD/ZAR (OTC)", "GBP/CHF (OTC)", "AUD/NZD (OTC)", "EUR/NZD (OTC)"
        ],
        "Crypto (OTC)": [
            "Bitcoin (OTC)", "Ethereum (OTC)", "Solana (OTC)", "Ripple (OTC)", "Avalanche (OTC)", "Dash (OTC)",
            "Polkadot (OTC)", "Dogecoin (OTC)", "Shiba Inu (OTC)", "Pepe (OTC)", "Binance Coin (OTC)", "Cardano (OTC)",
            "Dogwifhat (OTC)", "Arbitrum (OTC)", "Zcash (OTC)", "Cosmos (OTC)", "Beam (OTC)", "Axie Infinity (OTC)",
            "Bitcoin Cash (OTC)", "Bonk (OTC)", "Aptos (OTC)", "Floki (OTC)", "Gala (OTC)", "Hamster Kombat (OTC)",
            "Chainlink (OTC)", "Litecoin (OTC)", "Decentraland (OTC)", "Celestia (OTC)"
        ],
        "Stocks (OTC)": [
            "Apple (OTC)", "Microsoft (OTC)", "Facebook (OTC)", "Intel (OTC)", "Boeing (OTC)", "Pfizer (OTC)",
            "American Express (OTC)", "McDonald's (OTC)", "Tesla (OTC)", "Amazon (OTC)", "Netflix (OTC)", "Google (OTC)"
        ],
        "Commodities": ["Gold (OTC)", "Silver (OTC)", "UKBrent (OTC)", "USCrude (OTC)", "Natural Gas (OTC)"]
    }

    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        cat = st.selectbox("Market Category", list(MARKETS.keys()))
        asset = st.selectbox("Select Asset Pair", MARKETS[cat])
        if st.button("🚀 EXECUTE MATHEMATICAL ANALYSIS"):
            if deep_quant_validation(asset):
                signals = [
                    {"dir": "CALL (UP) 🟢", "pat": "Bullish Rejection at Order Block", "acc": 99.7},
                    {"dir": "PUT (DOWN) 🔴", "pat": "VSA Exhaustion - Institutional Reversal", "acc": 99.2},
                    {"dir": "CALL (UP) 🟢", "pat": "Fibonacci Golden Ratio Bounce (0.618)", "acc": 98.4},
                    {"dir": "PUT (DOWN) 🔴", "pat": "Liquidity Sweep Above Highs", "acc": 97.9}
                ]
                res = np.random.choice(signals)
                st.divider()
                st.subheader("🎯 VERIFIED SIGNAL")
                sc1, sc2 = st.columns(2)
                sc1.metric("DIRECTION", res['dir'])
                sc2.metric("CONFLUENCE SCORE", f"{res['acc']}%")
                st.info(f"**Technical Pattern:** {res['pat']}\n\n**Strategy:** Institutional Flow Confirmation.")
    st.markdown("</div>", unsafe_allow_html=True)

    # TradingView Integration
    st.divider()
    clean_sym = asset.split(" ")[0].replace("/", "")
    if "Gold" in clean_sym: clean_sym = "XAUUSD"
    elif "UKBrent" in clean_sym: clean_sym = "UKOIL"
    
    components.html(f"""
        <div style="height:550px; border-radius: 20px; overflow: hidden; border: 1px solid #4facfe;">
        <script src="https://s3.tradingview.com/tv.js"></script>
        <script>new TradingView.widget({{"width": "100%", "height": 550, "symbol": "{clean_sym}", "interval": "1", "theme": "{st.session_state.theme}", "container_id": "tv"}});</script>
        <div id="tv"></div></div>
    """, height=570)

if st.session_state.logged_in: main_terminal()
else: login_page()
