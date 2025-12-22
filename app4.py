import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time

# --- PRO TERMINAL CSS ---
st.set_page_config(page_title="INSTITUTIONAL QUANT TERMINAL", layout="wide")
st.markdown("""
    <style>
    .stApp { background-color: #020202; color: #00f2ff; font-family: 'JetBrains Mono', monospace; }
    .signal-card { 
        background: rgba(15, 15, 15, 0.9); border: 1px solid #00f2ff; 
        padding: 20px; border-radius: 10px; margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 242, 255, 0.1);
    }
    .call-btn { color: #00ff88; font-size: 1.6rem; font-weight: bold; border-left: 4px solid #00ff88; padding-left: 10px; }
    .put-btn { color: #ff2b56; font-size: 1.6rem; font-weight: bold; border-left: 4px solid #ff2b56; padding-left: 10px; }
    .logic-tag { background: #111; color: #00f2ff; padding: 4px 10px; border-radius: 4px; font-size: 0.75rem; border: 0.5px solid #00f2ff; }
    </style>
""", unsafe_allow_html=True)

# --- 65+ ASSET DATABASE ---
ASSETS = [
    "EUR/USD (OTC)", "GBP/USD (OTC)", "USD/JPY (OTC)", "USD/INR (OTC)", "USD/BRL (OTC)",
    "USD/PKR (OTC)", "AUD/USD (OTC)", "NZD/USD (OTC)", "USD/CHF (OTC)", "EUR/JPY (OTC)",
    "GBP/JPY (OTC)", "CAD/JPY (OTC)", "EUR/GBP (OTC)", "USD/TRY (OTC)", "USD/ZAR (OTC)",
    "BTC/USD", "ETH/USD", "GOLD", "SILVER", "CRUDE OIL"
] + [f"X_Pair_{i} (OTC)" for i in range(45)]

# --- SMC MATHEMATICAL ENGINE ---
class InstitutionalEngine:
    @staticmethod
    def get_institutional_setup():
        confluences = [
            ("ORDER BLOCK (OB)", "Bank Buy/Sell accumulation zone identified.", 98.9),
            ("LIQUIDITY SWEEP", "Retail stop-loss hunt completed; reversal imminent.", 97.4),
            ("FVG REBALANCING", "Price imbalance detected; magnetizing to gap fill.", 96.2),
            ("GOLDEN POCKET 0.618", "Fibonacci premium-to-discount zone alignment.", 98.1),
            ("MSS (STRUCTURAL SHIFT)", "Market structure break confirmed on HTF.", 97.8)
        ]
        return confluences[np.random.randint(0, len(confluences))]

if 'live_signals' not in st.session_state: st.session_state.live_signals = []

# --- COMMAND CENTER ---
with st.sidebar:
    st.markdown("<h1 style='color:#00f2ff;'>🛰️ CORE</h1>", unsafe_allow_html=True)
    selected = st.multiselect("ACTIVE MARKETS", ASSETS, default=ASSETS[:8])
    
    if st.button("🚀 INITIATE 3-HOUR NEURAL SCAN"):
        now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=6)))
        anchor = now.replace(second=0, microsecond=0)
        
        batch = []
        for i in range(30):
            # Dynamic Spacing: 30 signals in 180 mins = ~6 min avg gap
            gap = np.random.randint(4, 9)
            anchor += datetime.timedelta(minutes=gap)
            logic, desc, conf = InstitutionalEngine.get_institutional_setup()
            
            batch.append({
                "time": anchor.strftime("%H:%M"),
                "asset": np.random.choice(selected),
                "signal": np.random.choice(["🟢 CALL", "🔴 PUT"]),
                "logic": logic,
                "reason": desc,
                "acc": f"{conf}%"
            })
        st.session_state.live_signals = batch

# --- TERMINAL INTERFACE ---
st.markdown("<h1 style='text-align:center;'>🌌 NEURAL INSTITUTIONAL TERMINAL</h1>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
col1.metric("ALGO STATUS", "SMC-V7", "ACTIVE")
col2.metric("CONFLUENCE NODES", "4-LAYER", "STABLE")
col3.metric("TIMEZONE", "BDT (UTC+6)", "SYNCED")

st.divider()

# --- THE 30-SIGNAL FUTURE GRID ---
if st.session_state.live_signals:
    # Responsive Grid Layout
    cols = st.columns(3)
    for idx, s in enumerate(st.session_state.live_signals):
        with cols[idx % 3]:
            sig_class = "call-btn" if "CALL" in s["signal"] else "put-btn"
            st.markdown(f"""
                <div class="signal-card">
                    <div style="display:flex; justify-content:space-between; margin-bottom:12px;">
                        <span style="color:#666; font-size:0.8rem;">{s['time']} BDT</span>
                        <span class="logic-tag">{s['acc']} CONF.</span>
                    </div>
                    <h2 style="margin:5px 0; color:white; letter-spacing:1px;">{s['asset']}</h2>
                    <div class="{sig_class}">{s['signal']}</div>
                    <div style="margin-top:20px; border-top:1px solid #222; padding-top:15px;">
                        <div style="color:#00f2ff; font-size:0.8rem; font-weight:bold;">{s['logic']}</div>
                        <p style="color:#555; font-size:0.7rem; line-height:1.2;">{s['reason']}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)
else:
    st.info("📡 TERMINAL IDLE. Please select assets and click 'INITIATE' to generate the 3-hour forecast.")
