import os
import sqlite3
import pandas as pd
import numpy as np
import pytz
import streamlit as st
from datetime import datetime, timedelta
from streamlit_autorefresh import st_autorefresh

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    DATABASE_PATH = 'data/trading_v5.db'
    SIGNAL_INTERVAL = 1  # 1-minute candles
    BANGLADESH_TZ = 'Asia/Dhaka' 
    ADMIN_USER = "admin"
    ADMIN_PASS = "1234"

db_cfg = Config()

# =============================================================================
# LIVE REFRESH - Keeps the BDT Clock running every 1 second
# =============================================================================
st_autorefresh(interval=1000, key="bdt_live_clock")

# =============================================================================
# DATABASE & LOGIC
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

    def update_results(self):
        """Settles trades after 1 minute has passed in BDT"""
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
# UI DASHBOARD
# =============================================================================
def main_dashboard():
    db = Database(db_cfg.DATABASE_PATH)
    db.update_results()
    
    with st.sidebar:
        st.title("⚙️ SETTINGS")
        theme = st.selectbox("Mode", ["Dark Mode", "Bright Mode"])
        
        if st.button("🚀 GENERATE 24H BDT DATA", type="primary", use_container_width=True):
            generate_bulk_data(db)
        
        if st.button("🚪 LOGOUT", use_container_width=True):
            st.session_state['auth'] = False
            st.rerun()

    if theme == "Bright Mode":
        st.markdown("<style>.stApp {background-color: white; color: black;}</style>", unsafe_allow_html=True)

    # --- TOP HEADER: BDT CLOCK ---
    bdt_now = datetime.now(pytz.timezone(db_cfg.BANGLADESH_TZ))
    c1, c2, c3 = st.columns(3)
    c1.metric("🇧🇩 BDT CLOCK", bdt_now.strftime('%H:%M:%S'))
    
    df = db.get_signals()
    completed = df[df['status'] == 'COMPLETED']
    
    if not completed.empty:
        win_rate = (len(completed[completed['result'] == 'WIN']) / len(completed)) * 100
        c2.metric("📈 OVERALL WIN %", f"{win_rate:.1f}%")
        
        if len(completed) >= 100:
            last_100 = completed.head(100)
            wr_100 = (len(last_100[last_100['result'] == 'WIN']))
            c3.metric("🎯 LAST 100 WIN %", f"{wr_100}%")

    # --- TABS ---
    t1, t2 = st.tabs(["🚀 FUTURE SIGNALS", "📜 TRADE LOGS"])
    with t1:
        st.dataframe(df[df['status'] == 'PENDING'].head(15), use_container_width=True)
    with t2:
        st.dataframe(completed.head(50), use_container_width=True)

def generate_bulk_data(db):
    bdt_tz = pytz.timezone(db_cfg.BANGLADESH_TZ)
    start = datetime.now(bdt_tz)
    pairs = ['EURUSD-OTC', 'GBPUSD-OTC', 'USDJPY-OTC']
    batch = []
    for i in range(1440): # 24 hours
        batch.append((np.random.choice(pairs), np.random.choice(['UP', 'DOWN']), 
                     round(np.random.uniform(88, 98), 2), (start + timedelta(minutes=i)).strftime('%Y-%m-%d %H:%M:%S')))
    with db.get_connection() as conn:
        conn.execute("DELETE FROM signals")
        conn.executemany('INSERT INTO signals (pair, direction, accuracy, generated_at) VALUES (?,?,?,?)', batch)
    st.rerun()

# =============================================================================
# LOGIN
# =============================================================================
if 'auth' not in st.session_state: st.session_state['auth'] = False

if not st.session_state['auth']:
    st.title("🔒 BDT TERMINAL LOGIN")
    u, p = st.text_input("User"), st.text_input("Pass", type="password")
    if st.button("Login"):
        if u == db_cfg.ADMIN_USER and p == db_cfg.ADMIN_PASS:
            st.session_state['auth'] = True
            st.rerun()
else:
    main_dashboard()
