import os
import sqlite3
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pytz
import streamlit as st

# =============================================================================
# CONFIGURATION & THEME
# =============================================================================
class Config:
    DATABASE_PATH = 'data/trading_v3.db'
    SIGNAL_INTERVAL = 3  
    SIGNALS_PER_BATCH = 480 
    BANGLADESH_TZ = 'Asia/Dhaka'
    ADMIN_USER = "admin"
    ADMIN_PASS = "1234"

db_config = Config()

# =============================================================================
# DATABASE WITH WIN/LOSS TRACKING
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

    def add_signals_bulk(self, signals):
        with self.get_connection() as conn:
            conn.executemany(
                'INSERT INTO signals (pair, direction, accuracy, generated_at) VALUES (?, ?, ?, ?)',
                signals
            )

    def update_results(self):
        """Simulates outcome for past signals"""
        now = datetime.now(pytz.timezone(db_config.BANGLADESH_TZ)).replace(tzinfo=None)
        with self.get_connection() as conn:
            # Mark signals older than current time as Win or Loss randomly (for simulation)
            conn.execute('''
                UPDATE signals 
                SET status = 'COMPLETED',
                    result = CASE WHEN RANDOM() % 100 < 85 THEN 'WIN' ELSE 'LOSS' END
                WHERE generated_at < ? AND status = 'PENDING'
            ''', (now,))

    def get_all_signals(self):
        with self.get_connection() as conn:
            return pd.read_sql_query('SELECT * FROM signals ORDER BY generated_at DESC', conn)

# =============================================================================
# AUTHENTICATION
# =============================================================================
def login_page():
    st.title("🔐 Secure Login")
    with st.form("login"):
        user = st.text_input("Username")
        pw = st.text_input("Password", type="password")
        if st.form_submit_button("Login"):
            if user == db_config.ADMIN_USER and pw == db_config.ADMIN_PASS:
                st.session_state['auth'] = True
                st.rerun()
            else:
                st.error("Invalid credentials")

# =============================================================================
# UI COMPONENTS
# =============================================================================
def main_dashboard():
    db = Database(db_config.DATABASE_PATH)
    db.update_results()
    
    # --- SIDEBAR MENU ---
    with st.sidebar:
        st.title("🎛️ Control Panel")
        theme = st.radio("Appearance", ["Dark Mode", "Bright Mode"])
        
        st.divider()
        if st.button("🚀 Generate 24H Future Signals"):
            generate_signals(db)
            
        if st.button("🗑️ Clear History"):
            with db.get_connection() as conn:
                conn.execute("DELETE FROM signals")
            st.rerun()
            
        if st.button("🚪 Logout"):
            st.session_state['auth'] = False
            st.rerun()

    # --- TOP METRICS & CLOCK ---
    tz = pytz.timezone(db_config.BANGLADESH_TZ)
    now_bd = datetime.now(tz)
    
    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        st.header(f"📈 Trading Terminal")
    with col_t2:
        st.metric("Bangladesh Time", now_bd.strftime('%H:%M:%S'))

    # --- WIN RATIO LOGIC ---
    df = db.get_all_signals()
    completed = df[df['status'] == 'COMPLETED']
    
    if len(completed) > 0:
        win_count = len(completed[completed['result'] == 'WIN'])
        win_pc = (win_count / len(completed)) * 100
        
        # Win Ratio per 100 signals
        st.subheader(f"📊 Accuracy Performance")
        st.progress(win_pc / 100, text=f"Overall Win Rate: {win_pc:.2f}%")
        
        if len(completed) >= 100:
            last_100 = completed.head(100)
            l100_win = (len(last_100[last_100['result'] == 'WIN']) / 100) * 100
            st.info(f"Last 100 Signals Win Rate: {l100_win}%")

    # --- SIGNAL TABS ---
    tab1, tab2 = st.tabs(["🚀 Future Signals", "📜 Signal History"])
    
    with tab1:
        upcoming = df[df['status'] == 'PENDING'].sort_values('generated_at').head(50)
        if upcoming.empty:
            st.write("No upcoming signals. Generate some from the sidebar.")
        else:
            st.table(upcoming[['pair', 'direction', 'accuracy', 'generated_at']])

    with tab2:
        if completed.empty:
            st.write("No history recorded yet.")
        else:
            st.dataframe(completed[['pair', 'direction', 'result', 'generated_at']], use_container_width=True)

def generate_signals(db):
    start_time = datetime.now(pytz.timezone(db_config.BANGLADESH_TZ))
    pairs = ['EURUSD-OTC', 'GBPUSD-OTC', 'AUDUSD-OTC', 'XAUUSD-OTC', 'USDCAD-OTC']
    new_signals = []
    
    for i in range(db_config.SIGNALS_PER_BATCH):
        sig_time = start_time + timedelta(minutes=i * db_config.SIGNAL_INTERVAL)
        new_signals.append((
            np.random.choice(pairs),
            np.random.choice(['UP', 'DOWN']),
            round(np.random.uniform(82, 98), 2),
            sig_time.strftime('%Y-%m-%d %H:%M:%S')
        ))
    db.add_signals_bulk(new_signals)
    st.success("24H Signals Ready!")
    time.sleep(1)
    st.rerun()

# =============================================================================
# APP ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    if 'auth' not in st.session_state:
        st.session_state['auth'] = False
        
    if not st.session_state['auth']:
        login_page()
    else:
        main_dashboard()
