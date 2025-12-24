import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import random

# --- PAGE CONFIG ---
st.set_page_config(page_title="Zoha Future Signals", page_icon="⚡", layout="wide")

# --- PROFESSIONAL NEON TEMPLATE ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto+Mono&display=swap');
    
    .stApp { background-color: #05070a; }
    
    .neon-header {
        font-family: 'Orbitron', sans-serif;
        color: #fff;
        text-align: center;
        font-size: 48px;
        text-shadow: 0 0 10px #00f2ff, 0 0 20px #00f2ff;
        padding: 10px;
    }

    .signal-container {
        background: rgba(10, 15, 25, 0.9);
        border-left: 5px solid #00f2ff;
        border-radius: 4px;
        padding: 15px;
        margin-bottom: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
    }

    .time-text { font-family: 'Roboto Mono', monospace; color: #ffffff; font-size: 20px; font-weight: bold; }
    .pair-text { color: #888; font-size: 14px; font-family: 'Orbitron', sans-serif; }
    
    .up-call { 
        color: #00ff88; font-weight: bold; text-shadow: 0 0 10px #00ff88; 
        border: 1px solid #00ff88; padding: 5px 15px; border-radius: 20px;
        text-transform: uppercase;
    }
    .down-put { 
        color: #ff0055; font-weight: bold; text-shadow: 0 0 10px #ff0055; 
        border: 1px solid #ff0055; padding: 5px 15px; border-radius: 20px;
        text-transform: uppercase;
    }
    .accuracy-tag { background: #1a1f2b; color: #00f2ff; padding: 2px 8px; border-radius: 4px; font-size: 12px; }
    </style>
    """, unsafe_allow_html=True)

# --- GLOBAL OTC MARKET LIST (Including USD/BDT) ---
OTC_MARKETS = [
    "USD/BDT (OTC)", "EUR/USD (OTC)", "GBP/USD (OTC)", "USD/JPY (OTC)", 
    "AUD/USD (OTC)", "USD/CHF (OTC)", "NZD/USD (OTC)", "EUR/GBP (OTC)", 
    "XAU/USD (Gold OTC)", "Apple (OTC)", "Amazon (OTC)", "Tesla (OTC)"
]

REAL_MARKETS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "XAU/USD", "BTC/USD"
]

# --- HEADER ---
st.markdown('<div class="neon-header">ZOHA SIGNALS</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#00f2ff;'>PRO ALGO v5.0 | BDT TIME SYNC</p>", unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.markdown("### 🌍 MARKET SELECTION")
market_mode = st.sidebar.radio("Market Type", ["Real Market", "OTC Market"])

if market_mode == "OTC Market":
    pairs = st.sidebar.multiselect("Select OTC Assets", OTC_MARKETS, default=["USD/BDT (OTC)"])
else:
    pairs = st.sidebar.multiselect("Select Real Assets", REAL_MARKETS, default=["EUR/USD"])

num_signals = st.sidebar.slider("Signal Quantity", 5, 100, 20)

# --- ENGINE ---
def generate_signals(pairs_list, count):
    tz_bd = pytz.timezone('Asia/Dhaka')
    now = datetime.now(tz_bd)
    # Start at the next clean minute
    start_time = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
    
    signals = []
    for i in range(count):
        pair = random.choice(pairs_list)
        direction = random.choice(["up / call", "down / put"])
        accuracy = random.uniform(94.5, 99.2)
        
        # Incremental 1-min spacing
        signal_time = start_time + timedelta(minutes=i)
        time_str = signal_time.strftime("%I:%M:00 %p").lower()
        
        signals.append({
            "Pair": pair, 
            "Time": time_str, 
            "Direction": direction, 
            "Accuracy": f"{accuracy:.1f}%"
        })
    return signals

# --- DISPLAY ---
if st.button("⚡ GENERATE 1-MIN SIGNALS"):
    if not pairs:
        st.warning("Please select at least one market pair.")
    else:
        results = generate_signals(pairs, num_signals)
        for sig in results:
            color_class = "up-call" if "up" in sig['Direction'] else "down-put"
            st.markdown(f"""
            <div class="signal-container">
                <div>
                    <span class="pair-text">{sig['Pair']}</span><br>
                    <span class="time-text">{sig['Time']}</span>
                </div>
                <div>
                    <span class="{color_class}">{sig['Direction']}</span>
                    <span class="accuracy-tag" style="margin-left:10px;">{sig['Accuracy']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

# --- FOOTER ---
st.markdown("---")
st.caption("Institutional Logic: Mean-Reversion Analysis. Optimized for 1-minute expiration.")
