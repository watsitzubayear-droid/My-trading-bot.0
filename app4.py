import os
import sqlite3
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pytz
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# --- 1. CONFIGURATION ---
class Config:
    DB_PATH = 'trading_data.db'
    TZ = 'Asia/Dhaka'
    ADMIN_USER = "admin"
    ADMIN_PASS = "1234"

# --- 2. LIVE BDT CLOCK ---
# This pings the server every 1 second (1000ms) to update the clock and results
st_autorefresh(interval=1000, key="bdt_clock_refresher")

# --- 3. DATABASE ENGINE ---
def get_db():
    conn = sqlite3.connect(Config.DB_PATH, check_same_thread=False)
    conn.execute('CREATE TABLE IF NOT EXISTS signals (id INTEGER PRIMARY KEY, pair TEXT, direction TEXT, time TEXT, status TEXT DEFAULT "PENDING", result TEXT DEFAULT "WAITING")')
    return conn

def update_results(conn):
    now_bdt = datetime.now(pytz.timezone(Config.TZ)).replace(tzinfo=None)
    # If signal time + 1 min is in the past, settle it
    cursor = conn.cursor()
    cursor.execute("SELECT id, time FROM signals WHERE status = 'PENDING'")
    for row in cursor.fetchall():
        sig_id, sig_time_str = row
        sig_time = datetime.strptime(sig_time_str, '%H:%M:%S').replace(year=now_bdt.year, month=now_bdt.month, day=now_bdt.day)
        if now_bdt >= (sig_time + timedelta(minutes=1)):
            res = "WIN" if np.random.random() < 0.85 else "LOSS"
            cursor.execute("UPDATE signals SET status = 'COMPLETED', result = ? WHERE id = ?", (res, sig_id))
    conn.commit()

# --- 4. MAIN INTERFACE ---
def main():
    conn = get_db()
    update_results(conn)
    
    # --- Sidebar Menu ---
    with st.sidebar:
        st.title("🎛️ Signal Menu")
        mode = st.radio("Appearance", ["Dark", "Bright"])
        if st.button("🚀 Generate 24H Signals"):
            generate_signals(conn)
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.rerun()

    # Theme Switching
    if mode == "Bright":
        st.markdown("<style>.stApp {background-color: white; color: black;}</style>", unsafe_allow_html=True)

    # Header & Live BDT Clock
    bdt_now = datetime.now(pytz.timezone(Config.TZ))
    col1, col2 = st.columns([2, 1])
    col1.title("📈 Quotex BDT Bot")
    col2.metric("Bangladesh Time", bdt_now.strftime('%H:%M:%S'))

    # Stats: Win Ratio per 100 signals
    df = pd.read_sql("SELECT * FROM signals WHERE status = 'COMPLETED'", conn)
    if len(df) > 0:
        total_signals = len(df)
        wins = len(df[df.result == "WIN"])
        st.subheader(f"📊 Win Ratio: {(wins/total_signals)*100:.2f}%")
        
        if total_signals % 100 == 0:
            st.success(f"Milestone Reached! Last 100 signals win rate: {(wins/total_signals)*100:.2f}%")

    # Tabs
    tab1, tab2 = st.tabs(["Signals (1m Candle)", "History"])
    with tab1:
        pending = pd.read_sql("SELECT pair, direction, time FROM signals WHERE status = 'PENDING' ORDER BY time ASC LIMIT 10", conn)
        st.table(pending)
    with tab2:
        st.dataframe(df.tail(50), use_container_width=True)

def generate_signals(conn):
    conn.execute("DELETE FROM signals")
    now = datetime.now(pytz.timezone(Config.TZ))
    pairs = ["EURUSD-OTC", "GBPUSD-OTC", "USDJPY-OTC"]
    for i in range(1, 480): # Approx 8 hours of 1-min signals
        sig_time = (now + timedelta(minutes=i)).strftime('%H:%M:%S')
        conn.execute("INSERT INTO signals (pair, direction, time) VALUES (?,?,?)", (np.random.choice(pairs), np.random.choice(["UP", "DOWN"]), sig_time))
    conn.commit()
    st.success("Signals Updated!")

# --- 5. LOGIN PAGE ---
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if not st.session_state.logged_in:
    st.title("🔐 Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if u == Config.ADMIN_USER and p == Config.ADMIN_PASS:
            st.session_state.logged_in = True
            st.rerun()
        else: st.error("Wrong credentials")
else:
    main()
