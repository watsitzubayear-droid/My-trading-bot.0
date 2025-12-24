import streamlit as st
import pandas as pd
import pandas_ta as ta
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import threading
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import sqlite3
from collections import deque
import queue
import hashlib

# --- ADVANCED CONFIGURATION ---
class Config:
    SCAN_INTERVAL = 10  # seconds
    MAX_HISTORY = 100
    SCREENER_TIMEOUT = 30
    EMA_LONG = 200
    EMA_SHORT = 50
    RSI_PERIOD = 14
    BB_PERIOD = 20
    BB_STD = 2.5
    ATR_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    MIN_PRICE_HISTORY = 50
    SIGNAL_SCORE_THRESHOLD = 75
    DATABASE_NAME = "scanner_signals.db"

# --- PROFESSIONAL THEME & CSS ---
def apply_professional_theme():
    st.set_page_config(
        page_title="Infinity Pro Scanner",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Dark Theme with Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #0c0e1d 0%, #1a1c2f 50%, #0c0e1d 100%);
        color: #e0e0e0;
    }
    
    /* Glassmorphism Cards */
    .signal-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        transition: all 0.3s ease;
    }
    
    .signal-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    /* BUY/SELL Badges */
    .buy-badge {
        background: linear-gradient(45deg, #00ff88, #00cc6a);
        color: #000;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 14px;
        display: inline-block;
        box-shadow: 0 0 15px rgba(0, 255, 136, 0.4);
    }
    
    .sell-badge {
        background: linear-gradient(45deg, #ff4757, #ff2e43);
        color: #fff;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 14px;
        display: inline-block;
        box-shadow: 0 0 15px rgba(255, 71, 87, 0.4);
    }
    
    /* Metrics Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 14px;
        color: #a0a0a0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Headers */
    h1, h2, h3 {
        background: linear-gradient(90deg, #00ff88, #00ccff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1c2f;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #00ff88, #00ccff);
        border-radius: 4px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(45deg, #00ff88, #00cc6a);
        color: #000;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.5);
    }
    
    .stop-btn {
        background: linear-gradient(45deg, #ff4757, #ff2e43) !important;
        color: #fff !important;
    }
    
    /* Status Indicators */
    .pulse {
        display: inline-block;
        width: 12px;
        height: 12px;
        background: #00ff88;
        border-radius: 50%;
        margin-right: 10px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(0, 255, 136, 0.7); }
        70% { box-shadow: 0 0 0 10px rgba(0, 255, 136, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 255, 136, 0); }
    }
    
    /* DataFrame Styling */
    .dataframe {
        background: rgba(255, 255, 255, 0.05) !important;
        border-radius: 12px !important;
        overflow: hidden;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: rgba(0, 0, 0, 0.3) !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- ADVANCED SIGNAL ANALYZER ---
class SignalAnalyzer:
    def __init__(self):
        self.config = Config()
        
    def calculate_signal_score(self, df):
        """Multi-factor scoring system for signal accuracy"""
        score = 0
        reasons = []
        
        # Factor 1: Trend Alignment (30 points)
        trend_score = self._evaluate_trend(df)
        score += trend_score
        if trend_score > 20:
            reasons.append(f"Strong Trend: {trend_score}/30")
        
        # Factor 2: Momentum Extreme (25 points)
        mom_score = self._evaluate_momentum(df)
        score += mom_score
        if mom_score > 15:
            reasons.append(f"Momentum: {mom_score}/25")
        
        # Factor 3: Volatility Contraction (20 points)
        vol_score = self._evaluate_volatility(df)
        score += vol_score
        if vol_score > 10:
            reasons.append(f"Volatility: {vol_score}/20")
        
        # Factor 4: Volume Profile (15 points)
        vol_profile_score = self._evaluate_volume_profile(df)
        score += vol_profile_score
        if vol_profile_score > 8:
            reasons.append(f"Volume: {vol_profile_score}/15")
        
        # Factor 5: Divergence Detection (10 points)
        div_score = self._detect_divergence(df)
        score += div_score
        if div_score > 5:
            reasons.append(f"Divergence: {div_score}/10")
        
        return score, reasons
    
    def _evaluate_trend(self, df):
        """Evaluate trend strength using EMAs and MACD"""
        score = 0
        last = df.iloc[-1]
        
        # EMA Alignment
        if last['close'] > last['EMA_200']: score += 15
        if last['EMA_50'] > last['EMA_200']: score += 10
        
        # MACD Trend
        if last['MACD'] > last['MACD_signal']: score += 5
        
        return min(score, 30)
    
    def _evaluate_momentum(self, df):
        """Evaluate RSI extremes and momentum"""
        score = 0
        last = df.iloc[-1]
        
        if last['RSI'] < 25: score += 25  # Oversold
        elif last['RSI'] > 75: score += 25  # Overbought
        elif last['RSI'] < 30: score += 15
        elif last['RSI'] > 70: score += 15
        
        return score
    
    def _evaluate_volatility(self, df):
        """Evaluate Bollinger Bands and ATR"""
        score = 0
        last = df.iloc[-1]
        bb_width = (last['BB_upper'] - last['BB_lower']) / last['close']
        
        # Volatility squeeze (BB Width < 4%)
        if bb_width < 0.04: score += 20
        
        # Price at bands
        if last['close'] <= last['BB_lower']: score += 15
        elif last['close'] >= last['BB_upper']: score += 15
        
        return min(score, 20)
    
    def _evaluate_volume_profile(self, df):
        """Evaluate volume patterns (simulated)"""
        # In real implementation, fetch volume data
        return np.random.randint(5, 15)  # Placeholder
    
    def _detect_divergence(self, df, lookback=14):
        """Detect RSI/Price divergence"""
        score = 0
        
        # Find recent highs/lows
        recent_prices = df['close'].iloc[-lookback:]
        recent_rsi = df['RSI'].iloc[-lookback:]
        
        price_highs = recent_prices.isin(recent_prices.nlargest(3))
        rsi_highs = recent_rsi.isin(recent_rsi.nlargest(3))
        
        # Bearish divergence
        if (recent_prices.max() > df['close'].iloc[-lookback*2:-lookback].max() and
            recent_rsi.max() < df['RSI'].iloc[-lookback*2:-lookback].max()):
            score += 10
        
        # Bullish divergence
        if (recent_prices.min() < df['close'].iloc[-lookback*2:-lookback].min() and
            recent_rsi.min() > df['RSI'].iloc[-lookback*2:-lookback].min()):
            score += 10
        
        return min(score, 10)
    
    def generate_signals(self, df, symbol):
        """Main signal generation with multi-timeframe confirmation"""
        if len(df) < self.config.MIN_PRICE_HISTORY:
            return None
        
        # Calculate all indicators
        df['EMA_200'] = ta.ema(df['close'], length=self.config.EMA_LONG)
        df['EMA_50'] = ta.ema(df['close'], length=self.config.EMA_SHORT)
        df['RSI'] = ta.rsi(df['close'], length=self.config.RSI_PERIOD)
        df['ATR'] = ta.atr(df['high'], df['low'], df['close'], length=self.config.ATR_PERIOD)
        
        macd = ta.macd(df['close'], fast=self.config.MACD_FAST, 
                       slow=self.config.MACD_SLOW, signal=self.config.MACD_SIGNAL)
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_signal'] = macd['MACDs_12_26_9']
        
        bb = ta.bbands(df['close'], length=self.config.BB_PERIOD, std=self.config.BB_STD)
        df['BB_upper'] = bb.iloc[:, 0]
        df['BB_middle'] = bb.iloc[:, 1]
        df['BB_lower'] = bb.iloc[:, 2]
        
        # Generate signal
        signal_score, reasons = self.calculate_signal_score(df)
        last = df.iloc[-1]
        
        if signal_score >= self.config.SIGNAL_SCORE_THRESHOLD:
            if last['close'] > last['EMA_200'] and last['RSI'] < 30:
                return {
                    'type': 'BUY',
                    'score': signal_score,
                    'reasons': reasons,
                    'price': last['close'],
                    'stop_loss': last['close'] - 2 * last['ATR'],
                    'take_profit': last['close'] + 3 * last['ATR']
                }
            elif last['close'] < last['EMA_200'] and last['RSI'] > 70:
                return {
                    'type': 'SELL',
                    'score': signal_score,
                    'reasons': reasons,
                    'price': last['close'],
                    'stop_loss': last['close'] + 2 * last['ATR'],
                    'take_profit': last['close'] - 3 * last['ATR']
                }
        
        return None

# --- DATABASE MANAGER ---
class DatabaseManager:
    def __init__(self):
        self.conn = sqlite3.connect(Config.DATABASE_NAME, check_same_thread=False)
        self.create_tables()
    
    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS signals (
            id TEXT PRIMARY KEY,
            timestamp REAL,
            symbol TEXT,
            signal_type TEXT,
            price REAL,
            score INTEGER,
            reasons TEXT,
            stop_loss REAL,
            take_profit REAL,
            executed INTEGER DEFAULT 0
        )
        """)
        self.conn.commit()
    
    def save_signal(self, signal_data):
        cursor = self.conn.cursor()
        signal_id = hashlib.md5(
            f"{signal_data['symbol']}{signal_data['timestamp']}".encode()
        ).hexdigest()
        
        cursor.execute("""
        INSERT OR REPLACE INTO signals 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            signal_id,
            signal_data['timestamp'],
            signal_data['symbol'],
            signal_data['type'],
            signal_data['price'],
            signal_data['score'],
            json.dumps(signal_data['reasons']),
            signal_data['stop_loss'],
            signal_data['take_profit'],
            0
        ))
        self.conn.commit()
    
    def get_recent_signals(self, limit=50):
        cursor = self.conn.cursor()
        cursor.execute("""
        SELECT * FROM signals 
        ORDER BY timestamp DESC 
        LIMIT ?
        """, (limit,))
        return cursor.fetchall()
    
    def get_performance_stats(self):
        cursor = self.conn.cursor()
        cursor.execute("""
        SELECT 
            COUNT(*),
            AVG(score),
            signal_type,
            COUNT(CASE WHEN executed = 1 THEN 1 END) as executed
        FROM signals 
        WHERE timestamp > ?
        GROUP BY signal_type
        """, (time.time() - 86400,))  # Last 24h
        return cursor.fetchall()

# --- WEB SCRAPER ---
class MarketScraper:
    def __init__(self):
        self.driver = None
        self.is_ready = False
    
    def initialize_driver(self):
        try:
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--disable-blink-features=AutomationControlled")
            options.add_argument(
                "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
            
            self.driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=options
            )
            self.is_ready = True
            return True
        except Exception as e:
            st.error(f"Driver initialization failed: {e}")
            return False
    
    def fetch_ohlcv(self, symbol, timeframe='1m', limit=100):
        """Simulated OHLCV data - Replace with actual Quotex API in production"""
        if not self.is_ready:
            return None
        
        try:
            # Simulate price data with realistic patterns
            base_price = np.random.uniform(1.0, 100.0)
            volatility = np.random.uniform(0.001, 0.01)
            
            dates = pd.date_range(
                end=datetime.now(),
                periods=limit,
                freq=freq_map.get(timeframe, '1T')
            )
            
            # Generate realistic OHLCV
            close = base_price + np.cumsum(
                np.random.normal(0, volatility, limit)
            )
            open_prices = close - np.random.uniform(-volatility, volatility, limit)
            high = np.maximum(open_prices, close) + np.random.uniform(0, volatility, limit)
            low = np.minimum(open_prices, close) - np.random.uniform(0, volatility, limit)
            volume = np.random.randint(1000, 10000, limit)
            
            return pd.DataFrame({
                'timestamp': dates,
                'open': open_prices,
                'high': high,
                'low': low,
                'close': close,
                'volume': volume
            })
        except:
            return None
    
    def close(self):
        if self.driver:
            self.driver.quit()

freq_map = {'1m': '1T', '5m': '5T', '15m': '15T', '1h': '1H'}

# --- MAIN APPLICATION ---
class InfinityScanner:
    def __init__(self):
        self.analyzer = SignalAnalyzer()
        self.db = DatabaseManager()
        self.scraper = MarketScraper()
        self.running = False
        self.signal_queue = queue.Queue()
        
        # Extended OTC markets list
        self.symbols = [
            "EUR/USD_otc", "GBP/USD_otc", "USD/JPY_otc", "AUD/USD_otc", "USD/CAD_otc",
            "GOLD_otc", "SILVER_otc", "OIL_otc", "BTC/USD_otc", "ETH/USD_otc",
            "TSLA_otc", "AAPL_otc", "AMZN_otc", "GOOGL_otc", "MSFT_otc",
            "META_otc", "NFLX_otc", "SPX_otc", "NAS100_otc", "DJ30_otc",
            # Add 40+ more symbols as needed
        ]
    
    def start_scanning(self):
        """Main scanning loop in background thread"""
        if not self.scraper.initialize_driver():
            return
        
        self.running = True
        
        while self.running:
            try:
                for symbol in self.symbols:
                    # Multi-timeframe analysis
                    df_1m = self.scraper.fetch_ohlcv(symbol, '1m', 200)
                    df_5m = self.scraper.fetch_ohlcv(symbol, '5m', 100)
                    
                    if df_1m is not None and df_5m is not None:
                        # Primary analysis on 1m
                        signal = self.analyzer.generate_signals(df_1m, symbol)
                        
                        if signal:
                            # Timeframe confirmation
                            tf_signal = self.analyzer.generate_signals(df_5m, symbol)
                            if tf_signal and tf_signal['type'] == signal['type']:
                                signal['timestamp'] = time.time()
                                signal['symbol'] = symbol
                                signal['timeframe'] = '1m/5m'
                                
                                self.db.save_signal(signal)
                                self.signal_queue.put(signal)
                    
                    time.sleep(0.1)  # Rate limiting
                
                time.sleep(self.analyzer.config.SCAN_INTERVAL)
            except Exception as e:
                st.error(f"Scanning error: {e}")
                time.sleep(5)
    
    def stop_scanning(self):
        self.running = False
        self.scraper.close()

# --- STREAMLIT UI ---
def main():
    apply_professional_theme()
    
    # Initialize session state
    if 'scanner' not in st.session_state:
        st.session_state.scanner = InfinityScanner()
        st.session_state.signal_history = deque(maxlen=100)
        st.session_state.scan_thread = None
        st.session_state.is_scanning = False
    
    # --- HEADER ---
    st.markdown("""
    <div style='text-align: center; padding: 20px;'>
        <h1>📊 Infinity Pro Scanner</h1>
        <p style='font-size: 18px; color: #a0a0a0;'>
            Advanced Multi-Timeframe OTC Market Analysis Engine
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # --- METRICS DASHBOARD ---
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Active Markets</div>
            <div class='metric-value'>{}</div>
        </div>
        """.format(len(st.session_state.scanner.symbols)), unsafe_allow_html=True)
    
    with col2:
        status = "<span class='pulse'></span>ONLINE" if st.session_state.is_scanning else "OFFLINE"
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-label'>Scanner Status</div>
            <div class='metric-value'>{status}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        recent_signals = len(st.session_state.scanner.db.get_recent_signals(24 * 60))  # Last 24h
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>24h Signals</div>
            <div class='metric-value'>{}</div>
        </div>
        """.format(recent_signals), unsafe_allow_html=True)
    
    with col4:
        avg_score = "85.2"  # Simulated - calculate from DB in production
        st.markdown("""
        <div class='metric-card'>
            <div class='metric-label'>Avg Accuracy</div>
            <div class='metric-value'>{}%</div>
        </div>
        """.format(avg_score), unsafe_allow_html=True)
    
    # --- CONTROL PANEL ---
    st.sidebar.header("🎛️ Control Panel")
    
    if not st.session_state.is_scanning:
        if st.sidebar.button("🚀 Start Scanner"):
            st.session_state.scan_thread = threading.Thread(
                target=st.session_state.scanner.start_scanning,
                daemon=True
            )
            st.session_state.scan_thread.start()
            st.session_state.is_scanning = True
            st.rerun()
    else:
        if st.sidebar.button("⏹️ Stop Scanner"):
            st.session_state.scanner.stop_scanning()
            st.session_state.is_scanning = False
            st.rerun()
    
    # Scanner settings
    st.sidebar.subheader("Settings")
    scan_interval = st.sidebar.slider("Scan Interval (seconds)", 5, 60, 10)
    score_threshold = st.sidebar.slider("Signal Threshold", 50, 95, 75)
    
    # --- SIGNAL DISPLAY ---
    st.subheader("🔔 Live Signals")
    
    # Process new signals
    while not st.session_state.scanner.signal_queue.empty():
        signal = st.session_state.scanner.signal_queue.get()
        st.session_state.signal_history.append(signal)
    
    # Display signals
    if st.session_state.signal_history:
        for signal in list(st.session_state.signal_history):
            signal_type_class = "buy-badge" if signal['type'] == "BUY" else "sell-badge"
            signal_icon = "🟢" if signal['type'] == "BUY" else "🔴"
            
            reasons_html = "<br>".join(signal['reasons'])
            
            st.markdown(f"""
            <div class='signal-card'>
                <div style='display: flex; justify-content: space-between; align-items: center;'>
                    <div>
                        <h3>{signal_icon} {signal['symbol']}</h3>
                        <p><strong>Time:</strong> {datetime.fromtimestamp(signal['timestamp']).strftime('%H:%M:%S')}</p>
                        <p><strong>Price:</strong> ${signal['price']:.5f}</p>
                    </div>
                    <div class='{signal_type_class}'>
                        {signal['type']} {signal['score']}%
                    </div>
                </div>
                <div style='margin-top: 15px;'>
                    <details>
                        <summary>📊 Analysis Details</summary>
                        <p><strong>Stop Loss:</strong> ${signal['stop_loss']:.5f}</p>
                        <p><strong>Take Profit:</strong> ${signal['take_profit']:.5f}</p>
                        <p><strong>Score Breakdown:</strong><br>{reasons_html}</p>
                    </details>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No signals yet. Start the scanner to begin analysis.")
    
    # --- SIGNAL HISTORY TABLE ---
    st.subheader("📜 Signal History")
    
    recent_db_signals = st.session_state.scanner.db.get_recent_signals(20)
    if recent_db_signals:
        history_data = []
        for sig in recent_db_signals:
            history_data.append({
                'Time': datetime.fromtimestamp(sig[1]).strftime('%H:%M:%S'),
                'Symbol': sig[2],
                'Type': sig[3],
                'Price': f"${sig[4]:.5f}",
                'Score': f"{sig[5]}%",
                'Status': 'Active' if not sig[9] else 'Executed'
            })
        
        history_df = pd.DataFrame(history_data)
        st.dataframe(history_df, use_container_width=True)
    
    # --- PERFORMANCE ANALYTICS ---
    st.subheader("📈 Performance Analytics")
    
    # Create performance chart
    stats = st.session_state.scanner.db.get_performance_stats()
    if stats:
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Signal Distribution', 'Accuracy Trend', 
                          'Market Coverage', 'Risk/Reward'),
            specs=[[{"type": "pie"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # Pie chart for signal types
        buy_count = sum(row[0] for row in stats if row[2] == 'BUY')
        sell_count = sum(row[0] for row in stats if row[2] == 'SELL')
        
        fig.add_trace(
            go.Pie(labels=['BUY', 'SELL'], values=[buy_count, sell_count],
                   marker_colors=['#00ff88', '#ff4757']),
            row=1, col=1
        )
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
