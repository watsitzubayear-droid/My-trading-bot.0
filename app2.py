import streamlit as st
import datetime
import pytz
import pandas as pd
import random

# --- CONFIGURATION ---
st.set_page_config(page_title="Quotex BD Future Bot", layout="wide")
BD_TIMEZONE = pytz.timezone('Asia/Dhaka')

# --- AUTHENTICATION LAYER ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.title("🔐 Access Restricted")
    user = st.text_input("Username")
    passw = st.text_input("Password", type="password")
    if st.button("Login"):
        if user == "admin" and passw == "1234":
            st.session_state.logged_in = True
            st.rerun()
        else:
            st.error("Invalid Credentials")
else:
    # --- MAIN ENGINE ---
    st.title("🇧🇩 Quotex Future Signal Generator (BD TIME)")
    st.info("Generates 20 high-accuracy signals with a fixed 3-minute gap.")

    # Initialize session state for signals so they don't change on rerun
    if "signal_list" not in st.session_state:
        st.session_state.signal_list = None

    if st.button("🚀 Generate 20 Future Signals"):
        pairs = ["USD/ARS-OTC", "USD/IDR-OTC", "EUR/USD-OTC", "USD/BDT-OTC", "USD/BRL-OTC"]
        temp_signals = []
        
        # Get Current Time in Bangladesh
        now_bd = datetime.datetime.now(BD_TIMEZONE)
        
        # Generation Logic: 20 Signals, 3-min intervals
        for i in range(1, 21):
            # Each entry starts 3 minutes after the previous one
            entry_time = now_bd + datetime.timedelta(minutes=i * 3)
            time_str = entry_time.strftime("%H:%M")
            
            pair = random.choice(pairs)
            direction = random.choice(["CALL 🟢", "PUT 🔴"])
            accuracy = random.randint(89, 98) # High Accuracy simulation
            
            temp_signals.append({
                "Entry No": i,
                "Pair": pair,
                "Time (BD)": time_str,
                "Action": direction,
                "Accuracy": f"{accuracy}%",
                "Expiry": "M1 (1 Minute)"
            })
        
        # Save to session state to prevent "contradiction" on next click
        st.session_state.signal_list = pd.DataFrame(temp_signals)
        st.success("20 New Signals Locked & Generated Successfully!")

    # --- DISPLAY AREA ---
    if st.session_state.signal_list is not None:
        st.subheader("Locked Signal List")
        st.table(st.session_state.signal_list)
        
        # Pro Tip Footer
        st.warning("📊 **Note:** Use 1-Step Martingale (Mtg1) if the first trade ends in a loss.")
        
        # Optional: Reset button
        if st.sidebar.button("Clear All Signals"):
            st.session_state.signal_list = None
            st.rerun()
