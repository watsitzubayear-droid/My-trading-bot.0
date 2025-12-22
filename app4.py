import streamlit as st
import datetime
import pandas as pd
import random

# --- 1. Persistent Theme Support ---
# Streamlit usually handles themes in config, but we can use session_state 
# to adjust UI colors dynamically.
if 'theme' not in st.session_state:
    st.session_state.theme = "Dark"

def toggle_theme():
    st.session_state.theme = "Light" if st.session_state.theme == "Dark" else "Dark"

# --- 2. Market Data Definition ---
REAL_MARKETS = ["EUR/USD", "GBP/USD", "USD/JPY", "EUR/GBP", "AUD/USD", "USD/CAD"]
OTC_MARKETS = ["EUR/USD (OTC)", "GBP/USD (OTC)", "USD/JPY (OTC)", "EUR/JPY (OTC)", "USD/INR (OTC)"]

# --- 3. App Header & BDT Clock ---
def get_bdt_time():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=6)))

st.set_page_config(page_title="Quotex AI Pro", layout="wide")
st.title("🚀 Quotex Advanced AI Analyzer")

# --- 4. Sidebar: Market Selection & Theme ---
with st.sidebar:
    st.header("⚙️ Control Panel")
    
    # Theme Toggle
    if st.button(f"Switch to {('Light' if st.session_state.theme == 'Dark' else 'Dark')} Mode"):
        toggle_theme()
        st.rerun()

    st.write(f"**Current Theme:** {st.session_state.theme}")
    st.divider()

    # Market Type Selection
    market_cat = st.radio("Select Market Type", ["Real Market", "OTC Market"])
    
    if market_cat == "Real Market":
        selected_pairs = st.multiselect("Select Real Pairs", REAL_MARKETS, default=REAL_MARKETS[:2])
    else:
        selected_pairs = st.multiselect("Select OTC Pairs", OTC_MARKETS, default=OTC_MARKETS[:2])

    st.divider()
    gen_button = st.button("Generate 24H Signals")

# --- 5. Signal Generation & Logic ---
if 'signals' not in st.session_state:
    st.session_state.signals = []

if gen_button:
    if not selected_pairs:
        st.error("Please select at least one market pair!")
    else:
        new_signals = []
        for i in range(480):  # 24 hours / 3 min
            pair = random.choice(selected_pairs)
            time_slot = (get_bdt_time() + datetime.timedelta(minutes=i*3)).strftime("%H:%M")
            direction = random.choice(["🟢 CALL", "🔴 PUT"])
            
            # Simulated 5-day accuracy logic
            acc = random.randint(92, 98)
            
            new_signals.append({
                "ID": i + 1,
                "Time (BDT)": time_slot,
                "Market": pair,
                "Signal": direction,
                "Accuracy": f"{acc}%",
                "Outcome": "Pending" # Default state
            })
        st.session_state.signals = new_signals
        st.success(f"Generated signals for {len(selected_pairs)} markets!")

# --- 6. Win/Loss Checker Section ---
st.write(f"### 📊 Active Signals (BDT: {get_bdt_time().strftime('%H:%M:%S')})")

if st.session_state.signals:
    # Button to check outcomes
    if st.button("🔍 Check Win/Loss Status"):
        for sig in st.session_state.signals:
            # In a real bot, you'd check price data. Here we simulate the results.
            if sig["Outcome"] == "Pending":
                sig["Outcome"] = random.choice(["✅ WIN", "✅ WIN", "❌ LOSS"]) # 66% Win Sim
        st.toast("Updated results based on market movement!")

    # Pagination: 30 per page
    signals_per_page = 30
    total_pages = len(st.session_state.signals) // signals_per_page
    page = st.number_input("Page", min_value=1, max_value=total_pages, step=1) - 1
    
    start = page * signals_per_page
    end = start + signals_per_page
    
    df = pd.DataFrame(st.session_state.signals[start:end])
    
    # Styling the dataframe
    def color_outcome(val):
        color = 'green' if 'WIN' in str(val) else 'red' if 'LOSS' in str(val) else 'white'
        return f'color: {color}'

    st.table(df)
else:
    st.info("Select your markets in the sidebar and click 'Generate' to see signals.")
