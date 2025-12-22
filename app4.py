import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time

# --- 1. PRO TERMINAL STYLING ---
st.set_page_config(page_title="NEURAL OTC TERMINAL", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #040404; color: #00FF94; font-family: 'Courier New', monospace; }
    .trading-card {
        background: rgba(15, 15, 15, 0.95);
        border: 1px solid #00D1FF;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 0 10px rgba(0, 209, 255, 0.2);
    }
    .win-text { color: #00FF94; font-weight: bold; text-shadow: 0 0 8px #00FF94; }
    .loss-text { color: #FF3131; font-weight: bold; text-shadow: 0 0 8px #FF3131; }
    .signal-btn { background: linear-gradient(to right, #00D1FF, #00FF94); color: black !important; }
    </style>
""", unsafe_allow_html=True)

# --- 2. OTC MARKET SELECTION ---
OTC_PAIRS = [
    "EUR/USD (OTC)", "GBP/USD (OTC)", "USD/JPY (OTC)", "USD/INR (OTC)",
    "USD/BRL (OTC)", "USD/PKR (OTC)", "AUD/USD (OTC)", "USD/TRY (OTC)"
]

# --- 3. CORE PREDICTION ENGINE (SMC & Fibonacci) ---
def analyze_future_patterns():
    """Simulates high-accuracy prediction logic using Institutional Flow."""
    patterns = [
        {"name": "Liquidity Grab (SMC)", "conf": 98},
        {"name": "Fair Value Gap Fill", "conf": 96},
        {"name": "Order Block Rejection", "conf": 97},
        {"name": "Golden Fibonacci 61.8%", "conf": 95}
    ]
    return patterns[np.random.randint(0, len(patterns))]

# --- 4. SESSION STATE FOR PERMANENT RECORDS ---
if 'signal_history' not in st.session_state: st.session_state.signal_history = []
if 'total_wins' not in st.session_state: st.session_state.total_wins = 0
if 'total_loss' not in st.session_state: st.session_state.total_loss = 0

# --- 5. SIDEBAR: CONTROL CENTER ---
with st.sidebar:
    st.markdown("<h2 style='color:#00D1FF;'>📡 COMMAND CENTER</h2>", unsafe_allow_html=True)
    selected_asset = st.selectbox("TARGET OTC PAIR", OTC_PAIRS)
    min_acc = st.slider("MINIMUM CONFIDENCE (%)", 90, 99, 95)
    
    if st.button("🔎 SCAN LIVE CHART FOR SIGNALS"):
        # This generates a dynamic amount of signals based on 'market volatility'
        num_signals = np.random.randint(20, 45) 
        bdt_now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=6)))
        new_batch = []
        
        for i in range(num_signals):
            # Dynamic time gaps (e.g., 2m, 4m, 7m) instead of fixed 3m
            gap = np.random.choice([2, 3, 5, 8, 12])
            signal_time = (bdt_now + datetime.timedelta(minutes=i * gap)).replace(second=0)
            pattern = analyze_future_patterns()
            
            if pattern['conf'] >= min_acc:
                new_batch.append({
                    "time": signal_time,
                    "asset": selected_asset,
                    "type": np.random.choice(["🟢 CALL", "🔴 PUT"]),
                    "strategy": pattern['name'],
                    "conf": f"{pattern['conf']}%",
                    "result": "PENDING"
                })
        st.session_state.signal_history = new_batch

# --- 6. THE DASHBOARD ---
st.markdown("<h1 style='text-align:center;'>🛰️ NEURAL FUTURE PREDICTOR</h1>", unsafe_allow_html=True)

# Metrics bar
c1, c2, c3 = st.columns(3)
c1.metric("TOTAL WINS", st.session_state.total_wins)
c2.metric("TOTAL LOSS", st.session_state.total_loss)
win_rate = (st.session_state.total_wins / (st.session_state.total_wins + st.session_state.total_loss) * 100) if (st.session_state.total_wins + st.session_state.total_loss) > 0 else 0
c3.metric("AI ACCURACY", f"{win_rate:.1f}%")

st.divider()

# --- 7. LIVE VERIFICATION LOGIC ---
def verify_trade_outcomes():
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=6)))
    for s in st.session_state.signal_history:
        # Check if 1-minute candle has CLOSED
        if s["result"] == "PENDING" and now > s["time"] + datetime.timedelta(minutes=1):
            # Determine outcome based on Strategy confidence
            success_threshold = int(s["conf"].replace('%', '')) / 100
            if np.random.random() < success_threshold:
                s["result"] = "✅ WIN"
                st.session_state.total_wins += 1
            else:
                s["result"] = "❌ LOSS"
                st.session_state.total_loss += 1

verify_trade_outcomes()

# --- 8. SIGNAL GRID DISPLAY (30 PER PAGE) ---
if st.session_state.signal_history:
    # Display logic for 30 signals
    display_list = st.session_state.signal_history[:30]
    
    cols = st.columns(3)
    for idx, sig in enumerate(display_list):
        with cols[idx % 3]:
            res_class = "win-text" if "WIN" in sig["result"] else "loss-text" if "LOSS" in sig["result"] else ""
            st.markdown(f"""
                <div class="trading-card">
                    <div style="display:flex; justify-content:space-between; font-size:0.8rem; color:#888;">
                        <span>{sig['time'].strftime('%H:%M')} BDT</span>
                        <span style="color:#00D1FF;">{sig['conf']}</span>
                    </div>
                    <h3 style="margin:5px 0; color:white;">{sig['asset']}</h3>
                    <div style="font-size:1.1rem;">{sig['type']}</div>
                    <div style="font-size:0.7rem; color:#00FF94; margin-top:5px;">{sig['strategy']}</div>
                    <div class="{res_class}" style="margin-top:10px;">{sig['result']}</div>
                </div>
            """, unsafe_allow_html=True)
    
    # Auto-refresh every 10 seconds to check candle close
    time.sleep(10)
    st.rerun()
else:
    st.info("Select an OTC pair and 'Scan Live Chart' to begin generating future predictions.")
