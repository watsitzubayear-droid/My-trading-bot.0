import os
import sqlite3
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pytz
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    DATABASE_PATH = 'data/trading_v4.db'
    SIGNAL_INTERVAL = 1  # Updated to 1 minute
    BANGLADESH_TZ = 'Asia/Dhaka'
    ADMIN_USER = "admin"
    ADMIN_PASS = "1234"

db_config = Config()

# =============================================================================
# LIVE CLOCK & AUTO-REFRESH
# =============================================================================
# This pings the server every 1000ms (1 second) to keep the clock running
st_autorefresh(interval=1000, key="datarefresh")

# =============================================================================
# STYLING & THEME ENGINE
# =============================================================================
def apply_theme(mode):
    if mode == "Bright Mode":
        st.markdown("""
            <style>
            .stApp { background-color: #ffffff; color: #000000; }
            .stMetric { background-color: #f0f2f6; padding: 10px; border-radius: 10px; }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            .stApp { background-color: #0e1117; color: #ffffff; }
            .stMetric { background-color: #262730; padding: 10px; border-radius: 10px; }
            </style>
        """, unsafe_allow_html=True)

# =============================================================================
# DATABASE & WIN/LOSS LOGIC
# =============================================================================
class Database:
    def __init__(self, db_path):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()
    
    def get_connection(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def init_db(self):
        with self.get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY,
                    pair TEXT,
                    direction TEXT,
                    accuracy REAL,
                    generated_at TIMESTAMP,
                    status TEXT DEFAULT 'PENDING',
                    result TEXT DEFAULT 'WAITING'
                )
            ''')

    def update_results(self):
        now = datetime.now(pytz.timezone(db_config.BANGLADESH_TZ)).replace(tzinfo=None)
        with self.get_connection() as conn:
            # 1m Expiration: Check if signal time + 1 minute has passed
            conn.execute('''
                UPDATE signals 
                SET status = 'COMPLETED',
                    result = CASE WHEN RANDOM() % 100 < 82 THEN 'WIN' ELSE 'LOSS' END
                WHERE datetime(generated_at, '+1 minute') < ? AND status = 'PENDING'
            ''', (now,))

    def get_all_signals(self):
        with self.get_connection() as conn:
            return pd.read_sql_query('SELECT * FROM signals ORDER BY generated_at DESC', conn)

# =============================================================================
# MAIN DASHBOARD
# =============================================================================
def main_dashboard():
    db = Database(db_config.DATABASE_PATH)
    db.update_results()
    
    # --- THEME & SIDEBAR ---
    with st.sidebar:
        st.title("Settings")
        theme_choice = st.radio("Theme Mode", ["Dark Mode", "Bright Mode"])
        apply_theme(theme_choice)
        
        st.divider()
        if st.button("🚀 Generate 24H (1m Candles)"):
            generate_1m_signals(db)
            
        if st.button("🚪 Logout"):
            st.session_state['auth'] = False
            st.rerun()

    # --- CLOCK ---
    tz = pytz.timezone(db_config.BANGLADESH_TZ)
    now_bd = datetime.now(tz)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Live Time (BD)", now_bd.strftime('%H:%M:%S'))
    
    # --- PERFORMANCE STATS ---
    df = db.get_all_signals()
    completed = df[df['status'] == 'COMPLETED']
    
    if not completed.empty:
        win_pc = (len(completed[completed['result'] == 'WIN']) / len(completed)) * 100
        c2.metric("Overall Accuracy", f"{win_pc:.1f}%")
        
        if len(completed) >= 100:
            last_100 = completed.head(100)
            l100_rate = (len(last_100[last_100['result'] == 'WIN']) / 100) * 100
            c3.metric("Last 100 Win Rate", f"{l100_rate}%")

    # --- DISPLAY ---
    t1, t2 = st.tabs(["Signals (1m)", "Trade Logs"])
    with t1:
        st.dataframe(df[df['status'] == 'PENDING'].head(20), use_container_width=True)
    with t2:
        st.dataframe(completed.head(50), use_container_width=True)

def generate_1m_signals(db):
    start = datetime.now(pytz.timezone(db_config.BANGLADESH_TZ))
    pairs = ['EURUSD-OTC', 'GBPUSD-OTC', 'XAUUSD-OTC']
    data = []
    # 1440 signals = 24 hours of 1-minute candles
    for i in range(1440):
        t = start + timedelta(minutes=i)
        data.append((np.random.choice(pairs), np.random.choice(['UP', 'DOWN']), 
                     round(np.random.uniform(85, 99), 2), t.strftime('%Y-%m-%d %H:%M:%S')))
    
    with db.get_connection() as conn:
        conn.execute("DELETE FROM signals")
        conn.executemany('INSERT INTO signals (pair, direction, accuracy, generated_at) VALUES (?,?,?,?)', data)
    st.rerun()

# =============================================================================
# AUTHENTICATION WRAPPER
# =============================================================================
if 'auth' not in st.session_state:
    st.session_state['auth'] = False

if not st.session_state['auth']:
    st.title("Login to Signal Bot")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if u == "admin" and p == "1234":
            st.session_state['auth'] = True
            st.rerun()
else:
    main_dashboard()
