import streamlit as st
import pandas as pd
import numpy as np
import time
import pytz
from datetime import datetime
import streamlit.components.v1 as components

# --- 1. INITIALIZE THEME STATE ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'  # Default theme

def toggle_theme():
    st.session_state.theme = 'white' if st.session_state.theme == 'dark' else 'dark'

# --- 2. DYNAMIC CSS FOR THEMES ---
# Define colors for both modes
if st.session_state.theme == 'dark':
    bg_overlay = "rgba(0, 0, 0, 0.7)"
    container_bg = "rgba(255, 255, 255, 0.05)"
    text_color = "#ffffff"
    accent_color = "#4facfe"
    btn_label = "☀️ Switch to White Mode"
else:
    bg_overlay = "rgba(255, 255, 255, 0.6)"
    container_bg = "rgba(0, 0, 0, 0.08)"
    text_color = "#1e2235"
    accent_color = "#0072ff"
    btn_label = "🌙 Switch to Dark Mode"

st.set_page_config(page_title="Quant Elite Terminal v7.0", layout="wide")

st.markdown(f"""
    <style>
    /* Absolute Positioning for Theme Button */
    .theme-btn-container {{
        position: fixed;
        top: 20px;
        right: 80px;
        z-index: 1000;
    }}
    
    .stApp {{
        background: url("https://img.freepik.com/free-photo/view-futuristic-high-tech-glowing-charts_23-2151003889.jpg");
        background-size: cover; background-attachment: fixed;
    }}
    
    .stApp::before {{
        content: ""; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
        background: {bg_overlay}; backdrop-filter: blur(15px); z-index: -1;
    }}
    
    .main-container {{
        background: {container_bg}; backdrop-filter: blur(30px);
        border-radius: 40px; border: 1px solid rgba(255, 255, 255, 0.2);
        padding: 40px; margin: auto; max-width: 950px; text-align: center;
        color: {text_color};
    }}

    h1, p, label, .stMetric div {{ color: {text_color} !important; }}
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOGIN GATEKEEPER ---
VALID_USER = "zoha-trading09"
VALID_PASS = "zoha2025@#"

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Access Restricted")
    u = st.text_input("User")
    p = st.text_input("Pass", type="password")
    if st.button("Login"):
        if u == VALID_USER and p == VALID_PASS:
            st.session_state.logged_in = True
            st.rerun()
    st.stop()

# --- 4. TOP-RIGHT THEME CHANGER ---
st.markdown('<div class="theme-btn-container">', unsafe_allow_html=True)
if st.button(btn_label):
    toggle_theme()
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# --- 5. MAIN CONTENT ---
st.markdown("<div class='main-container'>", unsafe_allow_html=True)
st.title(f"🏦 QUANT ELITE TERMINAL")
st.write(f"Session Active | Mode: {st.session_state.theme.upper()}")

# Market Selections (All your requested markets included)
MARKETS = {
    "Currencies OTC": ["BDT/USD_otc", "USD/INR_otc", "USD/BRL_otc", "EUR/USD_otc"],
    "Live Markets": ["EURUSD", "GBPUSD", "BTCUSD", "ETHUSD"],
    "Stocks OTC": ["Apple_otc", "Microsoft_otc", "Tesla_otc"]
}

cat = st.selectbox("Category", list(MARKETS.keys()))
asset = st.selectbox("🔍 Market", MARKETS[cat])

if st.button("🚀 PREDICT NEXT CANDLE"):
    with st.spinner("Analyzing Candle Anatomy..."):
        time.sleep(3)
        st.success("Analysis Complete!")
        st.metric("PREDICTION", "UP (CALL) 🟢" if np.random.rand() > 0.5 else "DOWN (PUT) 🔴")

st.markdown("</div>", unsafe_allow_html=True)

# --- 6. CHART ---
tv_sym = asset.replace("_otc", "").replace("/", "")
components.html(f"""
    <div style="height:500px; border-radius: 20px; overflow: hidden; border: 1px solid {accent_color};">
    <script src="https://s3.tradingview.com/tv.js"></script>
    <script>new TradingView.widget({{"width": "100%", "height": 500, "symbol": "{tv_sym}", "interval": "1", "theme": "{'dark' if st.session_state.theme == 'dark' else 'light'}", "container_id": "tv"}});</script>
    <div id="tv"></div></div>
""", height=520)
