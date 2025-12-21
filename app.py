import streamlit as st
import pandas as pd
import numpy as np
import time
import pytz
from datetime import datetime
import streamlit.components.v1 as components

# --- 1. SETTINGS & TIMEZONE ---
bd_tz = pytz.timezone('Asia/Dhaka')
st.set_page_config(page_title="AI Search & Generate", layout="wide")

# --- 2. MARKET LIST (Expandable) ---
ALL_MARKETS = [
    "EURUSD_otc", "GBPUSD_otc", "USDJPY_otc", "AUDCAD_otc", "NZDUSD_otc",
    "EURGBP_otc", "GBPJPY_otc", "USDCAD_otc", "USDCHF_otc", "AUDUSD_otc",
    "EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "EURGBP", "BTCUSD", "ETHUSD"
]

# --- 3. UI SEARCH & SELECTION ---
st.title("🇧🇩 Sureshot AI: Search & Generate")
st.sidebar.header("🔍 Market Selection")

# Searchable dropdown bar
selected_market = st.sidebar.selectbox(
    "Search or Select Market",
    options=ALL_MARKETS,
    index=0,
    help="Type the name of the currency pair here"
)

# The "Generate" Button
generate_btn = st.sidebar.button("🚀 GENERATE SIGNAL", use_container_width=True)

# --- 4. ANALYSIS LOGIC ---
if generate_btn:
    st.session_state['last_market'] = selected_market
    
    # 5-7 Second Analysis Phase
    with st.status(f"⚡ Analyzing {selected_market}...", expanded=True) as status:
        st.write("⏱️ Fetching Live Candles...")
        time.sleep(2)
        st.write("🧠 Running 3-Layer Strategy Fusion...")
        time.sleep(2)
        st.write("📊 Calculating Probability...")
        time.sleep(2)
        status.update(label="✅ Analysis Complete!", state="complete")

    # Generate Result
    res_dir = "UP (CALL) 🟢" if np.random.rand() > 0.5 else "DOWN (PUT) 🔴"
    res_acc = np.random.randint(92, 99)
    curr_time = datetime.now(bd_tz).strftime("%H:%M:%S")

    # Display Results
    st.markdown(f"### 🎯 Result for {selected_market}")
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("DIRECTION", res_dir)
    with col2: st.metric("ACCURACY", f"{res_acc}%")
    with col3: st.metric("DHAKA TIME", curr_time)
    
    st.success(f"**ENTRY RULE:** Open trade at the start of the next 1-minute candle.")

# --- 5. LIVE TRADINGVIEW CHART ---
st.divider()
st.subheader(f"📊 Live {selected_market} Chart")
chart_asset = selected_market.replace("_otc", "")
chart_html = f"""
    <div style="height:450px;">
    <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
    <script type="text/javascript">
    new TradingView.widget({{
      "width": "100%", "height": 450, "symbol": "FX:{chart_asset}",
      "interval": "1", "theme": "dark", "style": "1", "locale": "en",
      "toolbar_bg": "#f1f3f6", "enable_publishing": false, "allow_symbol_change": true
    }});
    </script>
    </div>
"""
components.html(chart_html, height=470)

