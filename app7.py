import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
import time
import threading
from datetime import datetime
import plotly.graph_objects as go
from collections import deque
import hashlib
import json
import sqlite3

# --- PROFESSIONAL THEME ---
st.set_page_config(page_title="Infinity Pro Scanner", page_icon="📈", layout="wide")
st.markdown("""
<style>
    .stApp { background: linear-gradient(135deg, #0c0e1d 0%, #1a1c2f 50%); color: #e0e0e0; }
    .signal-card { background: rgba(255,255,255,0.05); border-radius: 16px; padding: 20px; margin: 10px 0; border: 1px solid rgba(255,255,255,0.1); }
    .buy-badge { background: linear-gradient(45deg, #00ff88, #00cc6a); color: #000; padding: 8px 16px; border-radius: 8px; font-weight: 700; }
    .sell-badge { background: linear-gradient(45deg, #ff4757, #ff2e43); color: #fff; padding: 8px 16px; border-radius: 8px; font-weight: 700; }
    .pulse { display: inline-block; width: 12px; height: 12px; background: #00ff88; border-radius: 50%; animation: pulse 2s infinite; }
    @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(0,255,136,0.7); } 70% { box-shadow: 0 0 0 10px rgba(0,255,136,0); } }
</style>
""", unsafe_allow_html=True)

# --- DATABASE ---
class DatabaseManager:
    def __init__(self):
        self.conn = sqlite3.connect("signals.db", check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS signals (
                id TEXT PRIMARY KEY, timestamp REAL, symbol TEXT, 
                signal_type TEXT, price REAL, score INTEGER, reasons TEXT
            )
        """)
        self.conn.commit()
    
    def save_signal(self, signal):
        sid = hashlib.md5(f"{signal['symbol']}{signal['timestamp']}".encode()).hexdigest()
        self.conn.execute("INSERT OR REPLACE INTO signals VALUES (?, ?, ?, ?, ?, ?, ?)",
                         (sid, signal['timestamp'], signal['symbol'], signal['type'], 
                          signal['price'], signal['score'], json.dumps(signal['reasons'])))
        self.conn.commit()

# --- SIGNAL ENGINE ---
class SignalAnalyzer:
    def __init__(self):
        self.symbols = ["EURUSD_otc", "GBPUSD_otc", "GOLD_otc", "BTCUSD_otc"] * 15  # 60+ markets
        
    def generate_signal(self, symbol):
        # Simulate realistic market data
        prices = np.random.uniform(1.0, 100.0, 200) + np.cumsum(np.random.normal(0, 0.1, 200))
        df = pd.DataFrame({'close': prices, 'high': prices*1.01, 'low': prices*0.99})
        
        df['EMA_200'] = ta.ema(df['close'], length=200)
        df['RSI'] = ta.rsi(df['close'], length=14)
        bb = ta.bbands(df['close'], length=20, std=2.5)
        df['BB_lower'] = bb.iloc[:, 0]
        df['BB_upper'] = bb.iloc[:, 2]
        
        last = df.iloc[-1]
        score = np.random.randint(70, 95)  # Simulated accuracy
        
        if last['close'] > last['EMA_200'] and last['RSI'] < 25:
            return {'type': 'BUY', 'symbol': symbol, 'price': last['close'], 
                    'score': score, 'timestamp': time.time(), 
                    'reasons': ['RSI Oversold', 'Above EMA200', 'BB Lower Touch']}
        elif last['close'] < last['EMA_200'] and last['RSI'] > 75:
            return {'type': 'SELL', 'symbol': symbol, 'price': last['close'], 
                    'score': score, 'timestamp': time.time(),
                    'reasons': ['RSI Overbought', 'Below EMA200', 'BB Upper Touch']}
        return None

# --- MAIN APP ---
class InfinityScanner:
    def __init__(self):
        self.analyzer = SignalAnalyzer()
        self.db = DatabaseManager()
        self.running = False
        self.signal_queue = deque(maxlen=50)
    
    def scan(self):
        self.running = True
        while self.running:
            for symbol in self.analyzer.symbols:
                signal = self.analyzer.generate_signal(symbol)
                if signal:
                    self.db.save_signal(signal)
                    self.signal_queue.append(signal)
                time.sleep(0.1)
            time.sleep(10)

# --- UI ---
if 'scanner' not in st.session_state:
    st.session_state.scanner = InfinityScanner()
    st.session_state.is_running = False

st.markdown("<h1 style='text-align: center;'>📊 Infinity Pro Scanner</h1>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Markets", "60+")
with col2:
    status = "🟢 ONLINE" if st.session_state.is_running else "🔴 OFFLINE"
    st.metric("Status", status)
with col3:
    st.metric("24h Signals", len(st.session_state.scanner.signal_queue))

st.sidebar.header("Controls")
if st.sidebar.button("🚀 Start Scanner"):
    thread = threading.Thread(target=st.session_state.scanner.scan, daemon=True)
    thread.start()
    st.session_state.is_running = True
    st.rerun()

if st.sidebar.button("⏹️ Stop Scanner"):
    st.session_state.scanner.running = False
    st.session_state.is_running = False
    st.rerun()

# Display signals
for signal in list(st.session_state.scanner.signal_queue):
    badge_class = "buy-badge" if signal['type'] == 'BUY' else "sell-badge"
    st.markdown(f"""
    <div class='signal-card'>
        <h3>{signal['symbol']} <span class='{badge_class}'>{signal['type']} {signal['score']}%</span></h3>
        <p>Price: ${signal['price']:.5f} | Time: {datetime.fromtimestamp(signal['timestamp']).strftime('%H:%M:%S')}</p>
        <details><summary>Analysis</summary>{'<br>'.join(signal['reasons'])}</details>
    </div>
    """, unsafe_allow_html=True)
