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
    SIGNAL_INTERVAL = 1  
    BANGLADESH_TZ = 'Asia/Dhaka'
    ADMIN_USER = "admin"
    ADMIN_PASS = "1234"

db_cfg = Config()

# =============================================================================
# LIVE REFRESH (Running Clock)
# =============================================================================
# Refreshes every 1 second to keep the clock moving
st_autorefresh(interval=1000, key="live_clock")

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
        """Automatically settles trades after 1 minute"""
        now = datetime.now(pytz.timezone(db_cfg.BANGLADESH_TZ)).replace(tzinfo=None)
        with self.get_connection() as conn:
            # If 1 min has passed since 'generated_at', mark as WIN/LOSS
            conn.execute('''
                UPDATE signals 
                SET status = 'COMPLETED',
                    result = CASE WHEN (ABS(RANDOM() % 100)) < 85 THEN 'WIN' ELSE 'LOSS' END
                WHERE datetime(generated_at, '+1 minute') <= ? AND status = 'PENDING'
            ''', (now.strftime('%Y-%m-%d %H:%M:%S'),))

    def get_signals(self):
        with self.get_connection() as conn:
            return pd.read_sql_query('SELECT * FROM signals ORDER BY generated_at DESC', conn)

# =============================================================================
# DASHBOARD
# =============================================================================
def main_dashboard():
    db = Database(db_cfg.DATABASE_PATH)
    db.update_live_results()
    
    # --- SIDEBAR MENU ---
    with st.sidebar:
        st.title("🎛️ BOT MENU")
        theme = st.selectbox("Appearance Mode", ["Dark Mode", "Bright Mode"])
        
        st.divider()
        if st.button("🚀 GENERATE 24H SIGNALS (1M)", type="primary", use_container_width=True):
            generate_24h_data(db)
            
        if st.button("🗑️ RESET DATABASE", use_container_width=True):
            with db.get_connection() as conn:
                conn.execute("DELETE FROM signals")
            st.rerun()

        if st.button("🚪 LOGOUT", use_container_width=True):
            st.session_state['logged_in'] = False
            st.rerun()

    # Apply Theme Styling
    if theme == "Bright Mode":
        st.markdown("<style>.stApp {background-color: white; color: black;}</style>", unsafe_allow_html=True)

    # --- TOP HEADER: LIVE CLOCK ---
    now_bd = datetime.now(pytz.timezone(db_cfg.BANGLADESH_TZ))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("🇧🇩 BANGLADESH TIME", now_bd.strftime('%H:%M:%S'))
    
    # --- CALCULATE WIN RATIO ---
    df = db.get_signals()
    completed = df[df['status'] == 'COMPLETED']
    
    if not completed.empty:
        total_wins = len(completed[completed['result'] == 'WIN'])
        win_ratio = (total_wins / len(completed)) * 100
        with col2:
            st.metric("📊 OVERALL WIN RATE", f"{win_ratio:.2f}%")
        
        # Win ratio for last 100
        if len(completed) >= 100:
            last_100 = completed.head(100)
            l100_wins = len(last_100[last_100['result'] == 'WIN'])
            with col3:
                st.metric("🎯 LAST 100 WIN RATE", f"{l100_wins}%")

    # --- TABS FOR SIGNALS ---
    tab1, tab2 = st.tabs(["🚀 FUTURE TRADES", "📜 SIGNAL HISTORY"])
    
    with tab1:
        # Show pending signals for the next 10 minutes
        pending = df[df['status'] == 'PENDING'].sort_values('generated_at', ascending=True)
        if not pending.empty:
            st.dataframe(pending[['pair', 'direction', 'accuracy', 'generated_at']].head(15), use_container_width=True)
        else:
            st.info("No signals. Use the sidebar to generate 24H data.")

    with tab2:
        if not completed.empty:
            st.dataframe(completed[['pair', 'direction', 'result', 'generated_at']].head(50), use_container_width=True)
        else:
            st.info("History will appear after trades expire (1 min).")

def generate_24h_data(db):
    start = datetime.now(pytz.timezone(db_cfg.BANGLADESH_TZ))
    pairs = ['EURUSD-OTC', 'GBPUSD-OTC', 'USDJPY-OTC', 'XAUUSD-OTC', 'BTCUSD-OTC']
    batch = []
    
    # Generate 1440 signals (24 hours * 60 minutes)
    for i in range(1440):
        sig_time = start + timedelta(minutes=i)
        batch.append((
            np.random.choice(pairs),
            np.random.choice(['UP', 'DOWN']),
            round(np.random.uniform(88, 98), 2),
            sig_time.strftime('%Y-%m-%d %H:%M:%S')
        ))
    
    with db.get_connection() as conn:
        conn.execute("DELETE FROM signals") # Clear old data
        conn.executemany('INSERT INTO signals (pair, direction, accuracy, generated_at) VALUES (?,?,?,?)', batch)
    st.success("24 Hours of 1M signals generated!")
    st.rerun()

# =============================================================================
# LOGIN PAGE
# =============================================================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.title("🛡️ QUOTEX SIGNAL BOT LOGIN")
    user = st.text_input("Username")
    passw = st.text_input("Password", type="password")
    if st.button("ENTER DASHBOARD"):
        if user == db_cfg.ADMIN_USER and passw == db_cfg.ADMIN_PASS:
            st.session_state['logged_in'] = True
            st.rerun()
        else:
            st.error("Access Denied")
else:
    main_dashboard()
