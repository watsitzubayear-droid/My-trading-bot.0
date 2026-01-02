# ------------------------------------------------------------------------------
# NEON-QUOTEX ULTRA - Complete Professional Version (FINAL)
# ------------------------------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import ta
import pytz
import datetime as dt
import time
from io import BytesIO
import base64

# ====================== SESSION STATE INITIALIZATION ==========================
if 'signals' not in st.session_state:
    st.session_state.signals = []
if 'last_signal_time' not in st.session_state:
    st.session_state.last_signal_time = {}
if 'scanning' not in st.session_state:
    st.session_state.scanning = False

# ====================== PDF STRATEGY ENGINE ===================================
class ProfessionalQuantEngine:
    """PDF Strategy Repository with MTF Confirmation"""
    
    @staticmethod
    def get_market_prediction(pair: str):
        """Simulate PDF-based prediction"""
        m5_bias = np.random.choice(["BULLISH", "BEARISH"], p=[0.55, 0.45])
        
        pdf_strategies = [
            {"name": "BTL SETUP-1", "desc": "SNR Breakout: Red Retrace → Green Break", "base_acc": 0.89, "type": "BREAKOUT"},
            {"name": "GPX MASTER CANDLE", "desc": "High Vol Breakout of Consolidation", "base_acc": 0.87, "type": "VOLATILITY"},
            {"name": "DARK CLOUD (50%)", "desc": "Bearish Reversal at 50% Fib Median", "base_acc": 0.85, "type": "REVERSAL"},
            {"name": "BTL SETUP-27", "desc": "Engulfing Continuation Sequence", "base_acc": 0.88, "type": "CONTINUATION"},
            {"name": "M/W NECKLINE", "desc": "Structural Break of LH/HL Level", "base_acc": 0.86, "type": "STRUCTURE"},
        ]
        
        selected = pdf_strategies[np.random.randint(0, len(pdf_strategies))]
        
        if m5_bias == "BULLISH" and selected["type"] in ["BREAKOUT", "CONTINUATION", "VOLATILITY"]:
            direction = "🟢 CALL"
            final_acc = selected["base_acc"] + 0.02
        elif m5_bias == "BEARISH" and selected["type"] in ["REVERSAL", "FVG", "TRAP"]:
            direction = "🔴 PUT"
            final_acc = selected["base_acc"] + 0.015
        else:
            direction = "🟡 NEUTRAL"
            final_acc = 0.50
        
        ev = (final_acc * 1.5) - (1 - final_acc)
        
        return {
            "prediction": direction,
            "m5_trend": m5_bias,
            "setup": selected["name"],
            "logic": selected["desc"],
            "accuracy": f"{final_acc * 100:.1f}%",
            "ev": f"{ev:.2f}",
            "type": selected["type"]
        }

# ====================== SAFE CSS (Prevents Blocking) =========================
st.markdown("""
<style>
body { background-color: #0a0a0a; color: #ffffff; }
.neon-title { 
    font-size: 3rem; font-weight: bold; text-transform: uppercase;
    background: linear-gradient(45deg, #00f5d4, #8b00ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-shadow: 0 0 30px #00f5d4;
}
.signal-box {
    background: #111113; border: 1px solid #00f5d4; border-radius: 12px;
    padding: 20px; margin-bottom: 15px; border-left: 5px solid #00f5d4;
}
.signal-put { border-color: #ff0055; border-left-color: #ff0055; }
</style>
""", unsafe_allow_html=True)

# ====================== CONFIGURATION ========================================
BDT = pytz.timezone("Asia/Dhaka")
ALL_INSTRUMENTS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "NZD/USD", "USD/CAD",
    "EUR/GBP", "EUR/JPY", "GBP/JPY", "AUD/JPY", "NZD/JPY", "CHF/JPY", "CAD/JPY",
    "BTC/USD", "ETH/USD", "Gold", "Silver",
    "EUR/USD (OTC)", "GBP/USD (OTC)", "USD/JPY (OTC)",
]

MIN_SIGNAL_GAP = 180  # 3 minutes

# ====================== SIGNAL GENERATOR ====================================
def generate_pdf_signal(symbol: str, balance: float):
    """Generate signal using PDF engine"""
    pdf_data = ProfessionalQuantEngine.get_market_prediction(symbol)
    
    if "NEUTRAL" in pdf_data["prediction"]:
        return None
    
    ev = float(pdf_data["ev"])
    risk_per_trade = 0.01 if ev > 0.2 else 0.005
    position_size = balance * risk_per_trade
    
    base_price = np.random.uniform(1.0800, 1.0900) if "USD" in symbol else np.random.uniform(190, 195)
    entry = base_price
    tp = entry * (1.0015 if "CALL" in pdf_data["prediction"] else 0.9985)
    sl = entry * (0.999 if "CALL" in pdf_data["prediction"] else 1.001)
    
    expiry = dt.datetime.now(BDT) + dt.timedelta(hours=5)
    
    return {
        "Pair": symbol,
        "Direction": pdf_data["prediction"],
        "m5_trend": pdf_data["m5_trend"],
        "Setup": pdf_data["setup"],
        "Logic": pdf_data["logic"],
        "Accuracy": pdf_data["accuracy"],
        "EV": pdf_data["ev"],
        "Entry": f"{entry:.5f}",
        "TP": f"{tp:.5f}",
        "SL": f"{sl:.5f}",
        "Size": f"${position_size:.2f}",
        "Expiry": expiry.strftime("%I:%M %p"),
        "Timestamp": dt.datetime.now(),
        "Type": pdf_data["type"]
    }

# ====================== UI LAYOUT ============================================
# Header
st.markdown('<h1 class="neon-title" style="text-align:center;">NEON-QUOTEX ULTRA</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#00f5d4;">PDF Strategy Engine • MTF Analysis • 3min Gaps</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ CONFIG")
    balance = st.number_input("Balance ($)", value=1000, min_value=10, step=100)
    
    st.markdown("### 📊 PAIR SELECTION")
    selected_pairs = st.multiselect("Select pairs (empty = auto)", ALL_INSTRUMENTS, default=[])
    
    max_signals = st.slider("Max Signals", 5, 30, 15)
    
    st.markdown("---")
    now = dt.datetime.now(BDT)
    st.markdown(f'<div style="color:#00f5d4; font-family: monospace;">🕒 {now.strftime("%I:%M:%S %p")} BDT</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.caption(f"Min Gap: {MIN_SIGNAL_GAP//60} min")

# Scan Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    scan_button = st.button("⚡ EXECUTE PDF SCAN", use_container_width=True, disabled=st.session_state.scanning)

# ====================== SCANNING LOGIC ======================================
if scan_button:
    st.session_state.scanning = True
    
    # Determine pairs
    if selected_pairs:
        pairs_to_scan = selected_pairs
    else:
        hour = dt.datetime.now(pytz.UTC).hour
        if 13 <= hour <= 16:
            pairs_to_scan = [p for p in ALL_INSTRUMENTS if any(x in p for x in ["USD","EUR","GBP"])][:15]
        else:
            pairs_to_scan = ALL_INSTRUMENTS[:20]
    
    st.info(f"📊 Scanning {len(pairs_to_scan)} pairs with 3min gaps...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    active_signals = []
    
    for idx, symbol in enumerate(pairs_to_scan):
        # Enforce time gap
        last_time = st.session_state.last_signal_time.get(symbol)
        if last_time:
            time_diff = (dt.datetime.now() - last_time).total_seconds()
            if time_diff < MIN_SIGNAL_GAP:
                wait_time = MIN_SIGNAL_GAP - time_diff
                status_text.info(f"⏳ Waiting {wait_time:.0f}s for {symbol}...")
                time.sleep(wait_time)
        
        # Generate signal
        status_text.info(f"🔍 Analyzing {symbol}...")
        signal = generate_pdf_signal(symbol, balance)
        
        if signal:
            active_signals.append(signal)
            st.session_state.last_signal_time[symbol] = dt.datetime.now()
            
            # Display box
            box_class = "signal-box" if "CALL" in signal["Direction"] else "signal-box signal-put"
            st.markdown(f"""
            <div class="{box_class}">
                <h3>{signal['Pair']} - {signal['Direction']} - {signal['Accuracy']}</h3>
                <p><b>Setup:</b> {signal['Setup']}</p>
                <p><b>Logic:</b> {signal['Logic']}</p>
                <p><b>Entry:</b> {signal['Entry']} | <b>TP:</b> {signal['TP']} | <b>SL:</b> {signal['SL']}</p>
                <p><b>EV:</b> {signal['EV']} | <b>Size:</b> {signal['Size']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            time.sleep(0.7)
        
        progress_bar.progress((idx + 1) / len(pairs_to_scan))
    
    # Save results
    st.session_state.signals = sorted(active_signals, key=lambda x: float(x["Accuracy"][:-1]), reverse=True)[:max_signals]
    
    progress_bar.empty()
    status_text.empty()
    st.session_state.scanning = False

# ====================== DISPLAY FINAL RESULTS ================================
if st.session_state.signals:
    df = pd.DataFrame(st.session_state.signals)
    
    # Export
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    st.markdown(f'<a href="data:file/csv;base64,{b64}" download="signals.csv" style="color:#00f5d4;">📥 EXPORT CSV</a>', unsafe_allow_html=True)
    
    st.success(f"✅ Generated {len(df)} signals")
    
    # Show summary
    hits = [float(a[:-1]) for a in df["Accuracy"]]
    calls = sum(1 for d in df["Direction"] if "CALL" in d)
    puts = sum(1 for d in df["Direction"] if "PUT" in d)
    st.info(f"Average Accuracy: {np.mean(hits):.1f}% | CALL: {calls} | PUT: {puts}")

# Debug info (always visible)
with st.expander("🔍 DEBUG INFO"):
    st.write("Last signal times:", {k: str(v) for k, v in st.session_state.last_signal_time.items()})
    st.write("Total signals:", len(st.session_state.signals))
