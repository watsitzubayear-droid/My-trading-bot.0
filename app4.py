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
    DATABASE_PATH = 'data/trading_v5.db'
    SIGNAL_INTERVAL = 1  # 1 Minute candle logic
    BANGLADESH_TZ = 'Asia/Dhaka'
    ADMIN_USER = "admin"
    ADMIN_PASS = "1234"

db_cfg = Config()

# =============================================================================
# LIVE REFRESH & CLOCK
# =============================================================================
# This component pings the server every 1000ms (1 second) to update the clock
st_autorefresh(interval=1000, key="live_clock_running")

# =============================================================================
# DATABASE ENGINE
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
            conn.execute('PRAGMA journal_mode=WAL')
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

    def update_live_results(self):
        """Settles trades as soon as 1 minute passes"""
        now = datetime.now(pytz.timezone(db_cfg.BANGLADESH_TZ)).replace(tzinfo=None)
        with self.get_connection() as conn:
            conn.execute('''
                UPDATE signals 
                SET status = 'COMPLETED',
                    result = CASE WHEN (ABS(RANDOM() % 100)) < 82 THEN 'WIN' ELSE 'LOSS' END
                WHERE datetime(generated_at, '+1 minute') <= ? AND status = 'PENDING'
            ''', (now.strftime('%Y-%m-%d %H:%M:%S'),))

    def get_signals(self):
        with self.get_connection() as conn:
            return pd.read_sql_query('SELECT * FROM signals ORDER BY generated_at DESC', conn)

# =============================================================================
# DASHBOARD UI
# =============================================================================
def main_dashboard():
    db = Database(db_cfg.DATABASE_PATH)
    db.update_live_results()
    
    # --- MENU & THEME ---
    with st.sidebar:
        st.title("🛡️ BOT MENU")
        theme_mode = st.radio("Display Mode", ["Dark Mode", "Bright Mode"])
        
        st.divider()
        if st.button("🚀 GENERATE 24H DATA", type="primary", use_container_width=True):
            generate_bulk_signals(db)
            
        if st.button("🗑️ CLEAR ALL DATA", use_container_width=True):
            with db.get_connection() as conn:
                conn.execute("DELETE FROM signals")
            st.rerun()

        if st.button("🚪 LOGOUT", use_container_width=True):
            st.session_state['auth_status'] = False
            st.rerun()

    # Apply CSS for Bright Mode if selected
    if theme_mode == "Bright Mode":
        st.markdown("<style>.stApp {background-color: white; color: black;}</style>", unsafe_allow_html=True)

    # --- TOP METRICS & LIVE CLOCK ---
    now_bd = datetime.now(pytz.timezone(db_cfg.BANGLADESH_TZ))
    
    col_clock, col_win, col_recent = st.columns(3)
    col_clock.metric("🇧🇩 Bangladesh Time", now_bd.strftime('%H:%M:%S'))
    
    df = db.get_signals()
    completed = df[df['status'] == 'COMPLETED']
    
    if not completed.empty:
        total_win_rate = (len(completed[completed['result'] == 'WIN']) / len(completed)) * 100
        col_win.metric("📈 Overall Win Rate", f"{total_win_rate:.1f}%")
        
        if len(completed) >= 100:
            last_100 = completed.head(100)
            l100_wins = (len(last_100[last_100['result'] == 'WIN']) / 100) * 100
            col_recent.metric("🎯 Last 100 Wins", f"{l100_wins}%")

    # --- SIGNAL DISPLAYS ---
    t1, t2 = st.tabs(["🚀 Current Signals", "📜 History"])
    
    with t1:
        pending = df[df['status'] == 'PENDING'].sort_values('generated_at').head(15)
        if pending.empty:
            st.warning("No signals available. Please generate data from the sidebar.")
        else:
            st.dataframe(pending[['pair', 'direction', 'accuracy', 'generated_at']], use_container_width=True)

    with t2:
        if completed.empty:
            st.info("Signals will move here once the 1-minute trade expires.")
        else:
            st.dataframe(completed[['pair', 'direction', 'result', 'generated_at']].head(50), use_container_width=True)

def generate_bulk_signals(db):
    start = datetime.now(pytz.timezone(db_cfg.BANGLADESH_TZ))
    pairs = ['EURUSD-OTC', 'GBPUSD-OTC', 'USDJPY-OTC', 'AUDUSD-OTC']
    batch = []
    
    for i in range(1440): # 1440 mins = 24 hours
        sig_time = start + timedelta(minutes=i)
        batch.append((
            np.random.choice(pairs),
            np.random.choice(['UP', 'DOWN']),
            round(np.random.uniform(85, 99), 2),
            sig_time.strftime('%Y-%m-%d %H:%M:%S')
        ))
    
    with db.get_connection() as conn:
        conn.execute("DELETE FROM signals")
        conn.executemany('INSERT INTO signals (pair, direction, accuracy, generated_at) VALUES (?,?,?,?)', batch)
    st.success("24 Hours of 1M signals generated successfully!")
    st.rerun()

# =============================================================================
# AUTHENTICATION
# =============================================================================
if 'auth_status' not in st.session_state:
    st.session_state['auth_status'] = False

if not st.session_state['auth_status']:
    st.title("🔒 Terminal Login")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Access Dashboard"):
        if u == db_cfg.ADMIN_USER and p == db_cfg.ADMIN_PASS:
            st.session_state['auth_status'] = True
            st.rerun()
        else:
            st.error("Invalid Username or Password")
else:
    main_dashboard()
