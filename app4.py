#!/usr/bin/env python3
"""
⚠️ CRITICAL DISCLAIMERS
=======================
1. MANUAL SIGNAL GENERATION - Click button to generate
2. NO LOGIN REQUIRED - Open access
3. PLAINTEXT STORAGE - Not for production
4. BATCH GENERATION - 50 signals per click
5. STREAMLIT CLOUD 100% COMPATIBLE
"""

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
    DATABASE_PATH = 'data/trading.db'
    SIGNAL_INTERVAL = 3  # minutes between signals
    ANALYSIS_LOOKBACK = 5  # days of historical data
    SIMULATION_MODE = True  # ⚠️ NEVER DISABLE FOR SAFETY
    BANGLADESH_TZ = 'Asia/Dhaka'
    MIN_ACCURACY = 60
    SIGNALS_PER_BATCH = 50
    MAX_TOTAL_SIGNALS = 480  # 24 hours worth (24*60/3)

config = Config()

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================
def init_session_state():
    """Initialize session state variables"""
    if 'batch_number' not in st.session_state:
        st.session_state['batch_number'] = 0
    if 'total_signals_generated' not in st.session_state:
        st.session_state['total_signals_generated'] = 0

# =============================================================================
# PURE PYTHON INDICATORS
# =============================================================================
def manual_rsi(close, period=14):
    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def manual_sma(close, period):
    return close.rolling(window=period).mean()

def manual_ema(close, period):
    return close.ewm(span=period, adjust=False).mean()

def manual_macd(close, fast=12, slow=26, signal=9):
    exp1 = manual_ema(close, fast)
    exp2 = manual_ema(close, slow)
    macd = exp1 - exp2
    signal_line = manual_ema(macd, signal)
    return macd, signal_line

def manual_bbands(close, period=20, std=2):
    sma = manual_sma(close, period)
    std_dev = close.rolling(window=period).std()
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    return upper, sma, lower

def manual_stoch(high, low, close, period=14):
    lowest_low = low.rolling(window=period).min()
    highest_high = high.rolling(window=period).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d = manual_sma(k, 3)
    return k, d

# =============================================================================
# SYNTHETIC DATA GENERATOR
# =============================================================================
def generate_synthetic_data(days=5, seed=42):
    """Generate realistic synthetic market data"""
    periods = days * 24 * 60
    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days), 
        periods=periods, 
        freq='1min', 
        tz='UTC'
    )
    np.random.seed(seed)
    returns = np.random.normal(0, 0.001, periods)
    trend = np.linspace(0, 0.005, periods)
    price = 1.0 + np.cumsum(returns + trend)
    high = price + np.abs(np.random.normal(0, 0.001, periods))
    low = price - np.abs(np.random.normal(0, 0.001, periods))
    
    df = pd.DataFrame({
        'Open': price,
        'High': high,
        'Low': low,
        'Close': price,
        'Volume': np.random.randint(100, 1000, periods)
    }, index=dates)
    return df

# =============================================================================
# DATABASE MANAGER
# =============================================================================
class Database:
    def __init__(self, db_path):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            # Enable foreign keys and WAL mode for better performance
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY,
                    pair TEXT,
                    direction TEXT,
                    accuracy REAL,
                    generated_at TIMESTAMP,
                    batch_number INTEGER,
                    executed BOOLEAN DEFAULT 0
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY,
                    pair TEXT,
                    direction TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    result TEXT,
                    executed_at TIMESTAMP,
                    closed_at TIMESTAMP
                )
            ''')
            conn.commit()
    
    def add_signal_with_time(self, pair, direction, accuracy, timestamp, batch_number):
        """Add signal with specific timestamp and batch number"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    'INSERT INTO signals (pair, direction, accuracy, generated_at, batch_number) VALUES (?, ?, ?, ?, ?)',
                    (pair, direction, accuracy, timestamp, batch_number)
                )
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            print(f"Database error: {e}")
            return None
    
    def get_recent_signals(self, limit=500):
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(
                'SELECT * FROM signals ORDER BY generated_at ASC LIMIT ?',
                conn, params=(limit,)
            )
    
    def get_signals_by_batch(self, batch_number):
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(
                'SELECT * FROM signals WHERE batch_number = ? ORDER BY generated_at ASC',
                conn, params=(batch_number,)
            )
    
    def get_max_batch_number(self):
        """Get the highest batch number in the database"""
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute('SELECT MAX(batch_number) FROM signals').fetchone()
            return result[0] if result[0] is not None else 0
    
    def get_total_signal_count(self):
        with sqlite3.connect(self.db_path) as conn:
            result = conn.execute('SELECT COUNT(*) FROM signals').fetchone()
            return result[0] if result[0] is not None else 0
    
    def add_trade(self, pair, direction, entry_price):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                'INSERT INTO trades (pair, direction, entry_price, executed_at, result) VALUES (?, ?, ?, ?, ?)',
                (pair, direction, entry_price, datetime.now(pytz.timezone(config.BANGLADESH_TZ)), 'PENDING')
            )
            conn.commit()
    
    def get_performance_stats(self):
        with sqlite3.connect(self.db_path) as conn:
            stats = conn.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses
                FROM trades WHERE result != 'PENDING'
            ''').fetchone()
            return {
                'total': stats[0] or 0,
                'wins': stats[1] or 0,
                'losses': stats[2] or 0,
                'accuracy': (stats[1] / stats[0] * 100) if stats[0] > 0 else 0
            }

db = Database(config.DATABASE_PATH)

# =============================================================================
# SIGNAL GENERATOR
# =============================================================================
class SignalGenerator:
    def __init__(self):
        self.pairs = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD',
            'EURJPY', 'GBPJPY', 'EURGBP', 'XAUUSD', 'BTCUSD',
            'ETHUSD', 'BNBUSD', 'XRPUSD', 'ADAUSD', 'SOLUSD',
            'DOGEUSD', 'DOTUSD', 'MATICUSD', 'SHIBUSDT', 'AVAXUSD',
            'NZDUSD', 'USDCHF', 'GBPCHF', 'EURCHF', 'AUDJPY',
            'GBPAUD', 'EURAUD', 'USDMXN', 'USDZAR', 'USDTRY'
        ]
    
    def calculate_indicators(self, df):
        df = df.copy()
        df['RSI'] = manual_rsi(df['Close'])
        df['MACD'], df['MACD_signal'] = manual_macd(df['Close'])
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = manual_bbands(df['Close'])
        df['MA_10'] = manual_sma(df['Close'], 10)
        df['MA_20'] = manual_sma(df['Close'], 20)
        df['MA_50'] = manual_sma(df['Close'], 50)
        df['Stoch_K'], df['Stoch_D'] = manual_stoch(df['High'], df['Low'], df['Close'])
        return df
    
    def generate_signal(self, pair):
        """Generate a single signal for a pair"""
        try:
            df = generate_synthetic_data(config.ANALYSIS_LOOKBACK)
            df = self.calculate_indicators(df)
            if len(df) < 50:
                return None
            
            latest = df.iloc[-1]
            previous = df.iloc[-2]
            signals = []
            
            # RSI Divergence
            if latest['RSI'] < 30 and previous['RSI'] >= 30:
                signals.append('BUY')
            elif latest['RSI'] > 70 and previous['RSI'] <= 70:
                signals.append('SELL')
            
            # MACD Crossover
            if latest['MACD'] > latest['MACD_signal'] and previous['MACD'] <= previous['MACD_signal']:
                signals.append('BUY')
            elif latest['MACD'] < latest['MACD_signal'] and previous['MACD'] >= previous['MACD_signal']:
                signals.append('SELL')
            
            # MA Cross
            if latest['MA_10'] > latest['MA_20'] and previous['MA_10'] <= previous['MA_20']:
                signals.append('BUY')
            elif latest['MA_10'] < latest['MA_20'] and previous['MA_10'] >= previous['MA_20']:
                signals.append('SELL')
            
            # Bollinger Bands
            if latest['Close'] <= latest['BB_lower'] and latest['RSI'] < 30:
                signals.append('BUY')
            elif latest['Close'] >= latest['BB_upper'] and latest['RSI'] > 70:
                signals.append('SELL')
            
            # Stochastic
            if latest['Stoch_K'] < 20 and latest['Stoch_D'] < 20:
                signals.append('BUY')
            elif latest['Stoch_K'] > 80 and latest['Stoch_D'] > 80:
                signals.append('SELL')
            
            buy_signals = signals.count('BUY')
            sell_signals = signals.count('SELL')
            
            if buy_signals >= 3:
                direction = 'UP'
            elif sell_signals >= 3:
                direction = 'DOWN'
            else:
                return None
            
            # Calculate accuracy
            accuracy = self.calculate_accuracy(df, direction)
            if accuracy >= config.MIN_ACCURACY:
                return {
                    'pair': pair,
                    'direction': direction,
                    'accuracy': round(accuracy, 2)
                }
            return None
        except Exception as e:
            st.error(f"Error generating signal for {pair}: {e}")
            return None
    
    def calculate_accuracy(self, df, predicted_direction):
        """Backtest accuracy on historical data"""
        try:
            test_period = min(100, len(df) - 1)
            correct_predictions = 0
            for i in range(len(df) - test_period, len(df) - 1):
                future_return = (df.iloc[i + 1]['Close'] - df.iloc[i]['Close']) / df.iloc[i]['Close']
                if predicted_direction == 'UP' and future_return > 0:
                    correct_predictions += 1
                elif predicted_direction == 'DOWN' and future_return < 0:
                    correct_predictions += 1
            return (correct_predictions / test_period) * 100
        except:
            return 50
    
    def generate_batch_signals(self, start_time):
        """Generate batch of signals with sequential timestamps"""
        signals = []
        for i, pair in enumerate(self.pairs):
            # Only generate up to batch size
            if i >= config.SIGNALS_PER_BATCH:
                break
            
            signal = self.generate_signal(pair)
            if signal:
                # Calculate future timestamp
                signal_time = start_time + timedelta(minutes=config.SIGNAL_INTERVAL * i)
                signals.append({
                    **signal,
                    'timestamp': signal_time
                })
        
        # Sort by accuracy and return top signals
        signals.sort(key=lambda x: x['accuracy'], reverse=True)
        return signals[:config.SIGNALS_PER_BATCH]

signal_gen = SignalGenerator()

# =============================================================================
# BATCH SIGNAL GENERATION
# =============================================================================
def generate_signal_batch():
    """
    Generate next batch of 50 signals with future timestamps
    Returns: Number of signals generated
    """
    # Check if we've reached the limit
    total_count = db.get_total_signal_count()
    if total_count >= config.MAX_TOTAL_SIGNALS:
        return -1  # Indicates limit reached
    
    # Calculate start time for this batch
    if total_count == 0:
        # First batch starts now
        start_time = datetime.now(pytz.timezone(config.BANGLADESH_TZ))
    else:
        # Subsequent batches start after the last signal
        last_signal_time = db.get_recent_signals(1)['generated_at'].iloc[0]
        start_time = pd.to_datetime(last_signal_time) + timedelta(minutes=config.SIGNAL_INTERVAL)
    
    # Increment batch number
    batch_number = db.get_max_batch_number() + 1
    
    # Generate signals
    batch_signals = signal_gen.generate_batch_signals(start_time)
    
    if not batch_signals:
        return 0
    
    # Store signals in database
    signals_stored = 0
    for signal in batch_signals:
        result = db.add_signal_with_time(
            signal['pair'], 
            signal['direction'], 
            signal['accuracy'], 
            signal['timestamp'],
            batch_number
        )
        if result:
            signals_stored += 1
    
    return signals_stored

# =============================================================================
# STREAMLIT UI
# =============================================================================
def dashboard_page():
    st.title("📊 Quotex Trading Bot Dashboard")
    st.warning("⚠️ SIMULATION MODE ONLY - NO REAL TRADES")
    
    # Bangladesh Time Clock
    bd_time = datetime.now(pytz.timezone(config.BANGLADESH_TZ)).strftime('%Y-%m-%d %H:%M:%S')
    st.sidebar.info(f"🕐 Bangladesh Time: {bd_time}")
    
    # Performance Stats
    stats = db.get_performance_stats()
    col1, col2, col3 = st.sidebar.columns(3)
    col1.metric("🟢 Wins", stats['wins'])
    col2.metric("🔴 Losses", stats['losses'])
    col3.metric("📊 Accuracy", f"{stats['accuracy']:.1f}%")
    
    # Get current counts
    total_count = db.get_total_signal_count()
    max_batch = db.get_max_batch_number()
    
    # Progress indicator
    progress = total_count / config.MAX_TOTAL_SIGNALS if config.MAX_TOTAL_SIGNALS > 0 else 0
    st.sidebar.progress(progress, text=f"Progress: {total_count}/{config.MAX_TOTAL_SIGNALS} signals")
    
    # Show current batch info
    if max_batch > 0:
        st.sidebar.info(f"📦 Last batch generated: #{max_batch}")
    
    # Generate button
    if total_count < config.MAX_TOTAL_SIGNALS:
        if st.button("🚀 Generate Next 50 Signals", type="primary", use_container_width=True):
            with st.spinner("Generating signals..."):
                count = generate_signal_batch()
                if count > 0:
                    st.success(f"✅ Generated {count} signals")
                    time.sleep(0.5)
                    st.rerun()
                elif count == -1:
                    st.warning("⚠️ Maximum 24-hour signal limit reached!")
                else:
                    st.warning("⚠️ No signals generated in this batch")
    else:
        st.sidebar.success("🎉 All 24-hour signals generated!")
        st.info("✅ Maximum signal limit reached. No more signals can be generated.")
    
    # Refresh button
    if st.button("🔄 Refresh Display", use_container_width=True):
        st.rerun()
    
    # Main content
    tab1, tab2 = st.tabs(["Live Signals", "Trade History"])
    
    with tab1:
        st.subheader("📈 Live Signals (Next 24 Hours)")
        
        if total_count == 0:
            st.info("👆 Click 'Generate Next 50 Signals' to start")
        else:
            # Get signals from database
            signals = db.get_recent_signals(100)
            
            if not signals.empty:
                # Show batch filter
                batch_numbers = sorted(signals['batch_number'].unique())
                selected_batch = st.selectbox(
                    "Filter by Batch:", 
                    ["All"] + [f"Batch #{int(b)}" for b in batch_numbers]
                )
                
                # Filter signals
                if selected_batch != "All":
                    batch_num = int(selected_batch.split("#")[1])
                    signals_to_show = db.get_signals_by_batch(batch_num)
                else:
                    signals_to_show = signals.head(50)
                
                # Display signals
                st.markdown(f"**Showing {len(signals_to_show)} signals**")
                
                for _, signal in signals_to_show.iterrows():
                    with st.container():
                        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                        pair_name = signal['pair']
                        direction = signal['direction']
                        accuracy = signal['accuracy']
                        time_str = pd.to_datetime(signal['generated_at']).strftime('%Y-%m-%d %H:%M')
                        
                        col1.markdown(f"**{pair_name}**")
                        col2.markdown(f"{'🟢 BUY' if direction == 'UP' else '🔴 SELL'}")
                        col3.markdown(f"**{accuracy}%**")
                        col4.markdown(f"*{time_str}*")
                        st.divider()
                
                # Statistics
                st.info(f"📊 Total signals in database: {total_count}")
            else:
                st.warning("⚠️ No signals found in database")
    
    with tab2:
        st.subheader("📜 Trade History")
        with sqlite3.connect(config.DATABASE_PATH) as conn:
            trades = pd.read_sql_query(
                'SELECT * FROM trades ORDER BY executed_at DESC LIMIT 100', conn
            )
        if not trades.empty:
            trades['symbol'] = trades['pair']
            trades['emoji'] = trades['result'].apply(
                lambda x: '🟢' if x == 'WIN' else '🔴' if x == 'LOSS' else '⏳'
            )
            st.dataframe(
                trades[['executed_at', 'symbol', 'direction', 'entry_price', 'result', 'emoji']],
                hide_index=True,
                use_container_width=True
            )
        else:
            st.info("No trade history yet")

def main():
    # Initialize session state
    init_session_state()
    
    # Set page config
    st.set_page_config(
        page_title="Quotex Trading Bot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Always show dashboard directly
    dashboard_page()

if __name__ == '__main__':
    main()
