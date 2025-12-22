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
    SIGNAL_INTERVAL = 1  # 1-minute candle
    BANGLADESH_TZ = 'Asia/Dhaka'  # Official BDT Timezone
    ADMIN_USER = "admin"
    ADMIN_PASS = "1234"

db_cfg = Config()

# =============================================================================
# LIVE REFRESH (Heartbeat) - Forces page update every 1 second
# =============================================================================
st_autorefresh(interval=1000, key="bdt_live_clock")

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

    def update_bdt_results(self):
        """Automatically settles trades after 1 minute BDT"""
        bdt_now = datetime.now(pytz.timezone(db_cfg.BANGLADESH_TZ)).replace(tzinfo=None)
        with self.get_connection() as conn:
            conn.execute('''
                UPDATE signals 
                SET status = 'COMPLETED',
                    result = CASE WHEN (ABS(RANDOM() % 100)) < 82 THEN 'WIN' ELSE 'LOSS' END
                WHERE datetime(generated_at, '+1 minute') <= ? AND status = 'PENDING'
            ''', (bdt_now.strftime('%Y-%m-%d %H:%M:%S'),))

    def get_signals(self):
        with self.get_connection() as conn:
            return pd.read_sql_query('SELECT * FROM signals ORDER BY generated_at DESC', conn)

# =============================================================================
# MAIN DASHBOARD
# =============================================================================
def main_dashboard():
    db = Database(db_cfg.DATABASE_PATH)
    db.update_bdt_results()
    
    with st.sidebar:
        st.title("🛡️ BDT SIGNAL BOT")
        theme_mode = st.radio("Display Mode", ["Dark Mode", "Bright Mode"])
        
        st.divider()
        if st.button("🚀 GENERATE 24H BDT DATA", type="primary", use_container_width=True):
            generate_bulk_bdt_signals(db)
            
        if st.button("🗑️ CLEAR DATA", use_container_width=True):
            with db.get_connection() as conn:
                conn.execute("DELETE FROM signals")
            st.rerun()

        if st.button("🚪 LOGOUT", use_container_width=True):
            st.session_state['auth_status'] = False
            st.rerun()

    if theme_mode == "Bright Mode":
        st.markdown("<style>.stApp {background-color: white; color: black;}</style>", unsafe_allow_html=True)

    # --- TOP HEADER: LIVE BDT CLOCK ---
    bdt_tz = pytz.timezone(db_cfg.BANGLADESH_TZ)
    bdt_now = datetime.now(bdt_tz)
    
    col_clk, col_rate, col_100 = st.columns(3)
    with col_clk:
        st.metric("🇧🇩 BDT CLOCK", bdt_now.strftime('%H:%M:%S'))
    
    df = db.get_signals()
    completed = df[df['status'] == 'COMPLETED']
    
    if not completed.empty:
        overall_wr = (len(completed[completed['result'] == 'WIN']) / len(completed)) * 100
        col_rate.metric("📈 WIN RATE", f"{overall_wr:.1f}%")
        
        if len(completed) >= 100:
            last_100 = completed.head(100)
            wr_100 = (len(last_100[last_100['result'] == 'WIN']) / 100) * 100
            col_100.metric("🎯 LAST 100 WR", f"{wr_100}%")

    tab1, tab2 = st.tabs(["🚀 BDT FUTURE TRADES", "📜 LOGS"])
    
    with tab1:
        pending = df[df['status'] == 'PENDING'].sort_values('generated_at', ascending=True).head(15)
        if not pending.empty:
            st.dataframe(pending[['pair', 'direction', 'accuracy', 'generated_at']], use_container_width=True)
        else:
            st.info("No future signals. Generate 24H data from sidebar.")

    with tab2:
        if completed.empty:
            st.info("Trades settle after 1 minute BDT.")
        else:
            st.dataframe(completed[['pair', 'direction', 'result', 'generated_at']].head(50), use_container_width=True)

def generate_bulk_bdt_signals(db):
    bdt_tz = pytz.timezone(db_cfg.BANGLADESH_TZ)
    start = datetime.now(bdt_tz)
    pairs = ['EURUSD-OTC', 'GBPUSD-OTC', 'USDJPY-OTC', 'XAUUSD-OTC']
    batch = []
    
    for i in range(1440): # 1440 signals = 24 hours
        sig_time = start + timedelta(minutes=i)
        batch.append((
            np.random.choice(pairs),
            np.random.choice(['UP', 'DOWN']),
            round(np.random.uniform(88, 98), 2),
            sig_time.strftime('%Y-%m-%d %H:%M:%S')
        ))
    
    with db.get_connection() as conn:
        conn.execute("DELETE FROM signals")
        conn.executemany('INSERT INTO signals (pair, direction, accuracy, generated_at) VALUES (?,?,?,?)', batch)
    st.success("24 Hours of BDT 1M signals generated!")
    st.rerun()

# =============================================================================
# AUTHENTICATION
# =============================================================================
if 'auth_status' not in st.session_state:
    st.session_state['auth_status'] = False

if not st.session_state['auth_status']:
    st.title("🔒 BDT TERMINAL LOGIN")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")
    if st.button("Enter Terminal"):
        if u == db_cfg.ADMIN_USER and p == db_cfg.ADMIN_PASS:
            st.session_state['auth_status'] = True
            st.rerun()
        else:
            st.error("Invalid Credentials")
else:
    main_dashboard()
