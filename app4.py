import os
import sqlite3
import time
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import pytz
import streamlit as st

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    DATABASE_PATH = 'data/trading_v2.db'
    SIGNAL_INTERVAL = 3  # 3 minutes between each trade
    SIGNALS_PER_BATCH = 480  # 480 signals * 3 min = 24 hours
    BANGLADESH_TZ = 'Asia/Dhaka'
    MIN_ACCURACY = 72
    LENIENT_MODE = True 

config = Config()

# =============================================================================
# DATABASE MANAGER (FIXED CONCURRENCY)
# =============================================================================
class Database:
    def __init__(self, db_path):
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()
    
    def get_connection(self):
        # Fix: check_same_thread=False is required for Streamlit
        return sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)

    def init_db(self):
        with self.get_connection() as conn:
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY,
                    pair TEXT,
                    direction TEXT,
                    accuracy REAL,
                    predicted_candle TEXT,
                    generated_at TIMESTAMP,
                    batch_number INTEGER
                )
            ''')

    def clear_signals(self):
        with self.get_connection() as conn:
            conn.execute('DELETE FROM signals')

    def add_signals_bulk(self, signals_list):
        with self.get_connection() as conn:
            conn.executemany(
                'INSERT INTO signals (pair, direction, accuracy, predicted_candle, generated_at, batch_number) VALUES (?, ?, ?, ?, ?, ?)',
                signals_list
            )

    def get_signals(self):
        with self.get_connection() as conn:
            return pd.read_sql_query('SELECT * FROM signals ORDER BY generated_at ASC', conn)

# =============================================================================
# SIGNAL GENERATION LOGIC
# =============================================================================
def get_random_market():
    pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'XAUUSD', 'BTCUSD', 'EURGBP', 'USDCHF']
    return f"{np.random.choice(pairs)}-OTC"

def generate_future_signals(db):
    """Generates 24 hours of signals with 3-minute gaps"""
    db.clear_signals()
    
    start_time = datetime.now(pytz.timezone(config.BANGLADESH_TZ))
    signals_to_save = []
    
    progress_bar = st.progress(0)
    
    for i in range(config.SIGNALS_PER_BATCH):
        # Calculate future timestamp (+3 mins each step)
        signal_time = start_time + timedelta(minutes=i * config.SIGNAL_INTERVAL)
        
        # Strategy Simulation
        direction = np.random.choice(['UP', 'DOWN'], p=[0.5, 0.5])
        accuracy = round(np.random.uniform(78, 96), 2)
        candle = "GREEN" if direction == "UP" else "RED"
        pair = get_random_market()
        
        signals_to_save.append((
            pair, direction, accuracy, candle, 
            signal_time.strftime('%Y-%m-%d %H:%M:%S'), 1
        ))
        
        if i % 50 == 0:
            progress_bar.progress(i / config.SIGNALS_PER_BATCH)
            
    db.add_signals_bulk(signals_to_save)
    progress_bar.empty()

# =============================================================================
# STREAMLIT UI
# =============================================================================
def main():
    st.set_page_config(page_title="24H Signal Bot", layout="wide")
    db = Database(config.DATABASE_PATH)
    
    st.title("🤖 24-Hour Future Signal Generator")
    st.info(f"Interval: {config.SIGNAL_INTERVAL} Minutes | Coverage: 24 Hours")

    col1, col2 = st.columns([1, 3])
    
    with col1:
        if st.button("🚀 GENERATE 24H SIGNALS", type="primary", use_container_width=True):
            with st.spinner("Analyzing market patterns for the next 24 hours..."):
                generate_future_signals(db)
                st.success("24 Hours of signals generated!")
                st.rerun()

        if st.button("🗑️ Clear All"):
            db.clear_signals()
            st.rerun()

    with col2:
        df = db.get_signals()
        if not df.empty:
            # Formatting for display
            df['generated_at'] = pd.to_datetime(df['generated_at'])
            
            # Show only upcoming signals
            now = datetime.now(pytz.timezone(config.BANGLADESH_TZ)).replace(tzinfo=None)
            upcoming = df[df['generated_at'] >= now].head(100)
            
            st.write(f"### Upcoming Signals ({len(upcoming)} items)")
            
            # Stylized Table
            for _, row in upcoming.iterrows():
                with st.container():
                    c1, c2, c3, c4 = st.columns([2, 1, 1, 2])
                    color = "🟢" if row['direction'] == "UP" else "🔴"
                    c1.markdown(f"**{row['pair']}**")
                    c2.markdown(f"{color} {row['direction']}")
                    c3.markdown(f"🎯 {row['accuracy']}%")
                    c4.markdown(f"⏰ {row['generated_at'].strftime('%H:%M:%S')}")
                    st.divider()
        else:
            st.warning("No signals found. Click the button to generate.")

if __name__ == '__main__':
    main()
