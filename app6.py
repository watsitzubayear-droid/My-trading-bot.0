import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import pytz
import time
import logging
import json
import hashlib
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATIONS & LOGGING ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_terminal.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

VERSION = "2.0.0"
CONFIG_FILE = "trading_config.json"

# --- ADVANCED TECHNICAL ANALYSIS ENGINE ---
class TechnicalAnalysisEngine:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.data = {}
        self.score = 0
        self.analysis_results = {}
        
    def fetch_data(self, period: str = "1d", interval: str = "1m") -> pd.DataFrame:
        """Fetch real market data from Yahoo Finance"""
        try:
            logger.info(f"Fetching data for {self.symbol} - {interval}")
            ticker = yf.Ticker(self.symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                logger.warning(f"No data found for {self.symbol}")
                return pd.DataFrame()
                
            df.reset_index(inplace=True)
            return df
        except Exception as e:
            logger.error(f"Data fetch error for {self.symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate comprehensive technical indicators"""
        if df.empty:
            return df
            
        # Trend Indicators
        df['SMA_20'] = ta.sma(df['Close'], length=20)
        df['EMA_9'] = ta.ema(df['Close'], length=9)
        df['EMA_21'] = ta.ema(df['Close'], length=21)
        
        # Momentum
        df['RSI'] = ta.rsi(df['Close'], length=14)
        macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        df = pd.concat([df, macd], axis=1)
        
        # Volatility
        bbands = ta.bbands(df['Close'], length=20)
        df = pd.concat([df, bbands], axis=1)
        df['ATR'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        
        # Volume & Price Action
        df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
        df['OBV'] = ta.obv(df['Close'], df['Volume'])
        
        # Support/Resistance
        df['prev_high'] = df['High'].shift(1)
        df['prev_low'] = df['Low'].shift(1)
        
        return df
    
    def analyze_market_structure(self, df: pd.DataFrame) -> dict:
        """Analyze SMC/ICT concepts - Market Structure"""
        results = {
            'bullish_break': False,
            'bearish_break': False,
            'premium_zone': False,
            'discount_zone': False,
            'equilibrium': False
        }
        
        current_price = df['Close'].iloc[-1]
        bb_middle = df['BBM_20_2.0'].iloc[-1]
        bb_upper = df['BBU_20_2.0'].iloc[-1]
        bb_lower = df['BBL_20_2.0'].iloc[-1]
        
        # Premium/Discount zones (Bollinger Bands)
        results['premium_zone'] = current_price > bb_upper
        results['discount_zone'] = current_price < bb_lower
        results['equilibrium'] = bb_lower <= current_price <= bb_upper
        
        # Structure breaks
        results['bullish_break'] = current_price > df['prev_high'].iloc[-1]
        results['bearish_break'] = current_price < df['prev_low'].iloc[-1]
        
        return results
    
    def score_indicator(self, condition: bool, points: int, name: str):
        """Helper to score individual conditions"""
        if condition:
            self.score += points
            self.analysis_results[name] = "✓ PASSED"
        else:
            self.analysis_results[name] = "✗ FAILED"
    
    def calculate_signal_score(self, current_timeframe: str = "1m") -> dict:
        """Core scoring logic - 100 point system"""
        self.score = 0
        self.analysis_results = {}
        
        # Fetch multi-timeframe data
        df_1m = self.fetch_data("1d", "1m")
        df_5m = self.fetch_data("5d", "5m")
        df_15m = self.fetch_data("15d", "15m")
        df_1h = self.fetch_data("1mo", "1h")
        
        if df_1m.empty:
            return {"error": "No data available"}
        
        # Calculate indicators for all timeframes
        df_1m = self.calculate_indicators(df_1m)
        df_5m = self.calculate_indicators(df_5m)
        df_15m = self.calculate_indicators(df_15m)
        df_1h = self.calculate_indicators(df_1h)
        
        current_price = df_1m['Close'].iloc[-1]
        
        # --- SCORING SYSTEM (100 Points) ---
        
        # Trend Alignment (30 points)
        ema_9_1m = df_1m['EMA_9'].iloc[-1]
        ema_21_1m = df_1m['EMA_21'].iloc[-1]
        ema_9_5m = df_5m['EMA_9'].iloc[-1]
        ema_21_5m = df_5m['EMA_21'].iloc[-1]
        
        trend_bullish_1m = ema_9_1m > ema_21_1m
        trend_bullish_5m = ema_9_5m > ema_21_5m
        
        self.score_indicator(trend_bullish_1m and trend_bullish_5m, 15, "Multi-TF Trend Alignment")
        self.score_indicator(abs(current_price - ema_9_1m) / current_price < 0.002, 15, "Price Near EMA")
        
        # Momentum (25 points)
        rsi_1m = df_1m['RSI'].iloc[-1]
        rsi_15m = df_15m['RSI'].iloc[-1]
        
        # RSI conditions
        self.score_indicator(30 < rsi_1m < 70, 10, "RSI in Neutral Zone")
        self.score_indicator(rsi_15m > 50, 10, "15m RSI Bullish")
        self.score_indicator(df_1m['MACD_12_26_9'].iloc[-1] > df_1m['MACDs_12_26_9'].iloc[-1], 5, "MACD Crossover")
        
        # Volatility & S/R (20 points)
        market_structure = self.analyze_market_structure(df_1m)
        
        self.score_indicator(market_structure['equilibrium'], 10, "Price in Equilibrium")
        self.score_indicator(df_1m['ATR'].iloc[-1] > df_1m['ATR'].mean() * 0.8, 5, "Healthy Volatility")
        self.score_indicator(current_price > df_1m['VWAP'].iloc[-1], 5, "Above VWAP")
        
        # Volume Profile (15 points)
        volume_increase = df_1m['Volume'].iloc[-1] > df_1m['Volume'].rolling(20).mean().iloc[-1]
        self.score_indicator(volume_increase, 15, "Volume Confirmation")
        
        # Risk Filter (10 points)
        # Spread check (simulated)
        spread = (df_1m['High'].iloc[-1] - df_1m['Low'].iloc[-1]) / current_price * 100
        self.score_indicator(spread < 0.1, 10, "Low Spread Condition")
        
        # Determine direction
        if self.score >= 70:
            direction = "UP (CALL) 🟢" if trend_bullish_1m else "DOWN (PUT) 🔴"
        else:
            direction = "NEUTRAL ⚪"
            
        return {
            "score": min(self.score, 100),
            "direction": direction,
            "analysis": self.analysis_results,
            "price": current_price,
            "timestamp": get_bdt_time().strftime("%Y-%m-%d %H:%M:%S"),
            "timeframe": current_timeframe
        }

# --- GLOBAL FUNCTIONS ---
def get_bdt_time():
    return datetime.datetime.now(pytz.timezone('Asia/Dhaka'))

def load_config():
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "win_rate": 0.0,
            "total_signals": 0,
            "profitable_signals": 0,
            "risk_per_trade": 1.0
        }

def save_config(config):
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

# --- STREAMLIT UI ---
st.set_page_config(
    page_title="Zoha Neural-100 Pro Terminal v2.0",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stApp { background: #0a0e17; color: #e6edf3; }
    .signal-card { 
        background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
        border: 1px solid #30363d; 
        padding: 25px; border-radius: 15px; 
        border-top: 4px solid #00f2ff;
        margin-bottom: 20px;
        box-shadow: 0 4px 20px rgba(0, 242, 255, 0.1);
        transition: transform 0.2s;
    }
    .signal-card:hover { transform: translateY(-5px); }
    .high-score { border-top-color: #00f2ff; }
    .medium-score { border-top-color: #ffa500; }
    .low-score { border-top-color: #ff2e63; }
    .score-box { font-size: 38px; font-weight: bold; }
    .pair-name { font-size: 20px; font-weight: bold; color: #58a6ff; }
    .direction-text { font-size: 28px; font-weight: bold; margin: 15px 0; }
    .condition-list { font-size: 11px; color: #8b949e; line-height: 1.5; }
    .legend-badge {
        display: inline-block; padding: 4px 12px; margin: 2px;
        border-radius: 20px; font-size: 10px; font-weight: bold;
        border: 1px solid;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.title("🛠️ Control Panel")
    st.markdown(f"### `{get_bdt_time().strftime('%H:%M:%S')}` BDT")
    st.divider()
    
    # Market Selection
    market_type = st.selectbox(
        "Market Type", 
        ["Forex", "Crypto", "Stocks", "Commodities"],
        help="Select the market for analysis"
    )
    
    # Symbol Input
    st.subheader("Asset Configuration")
    symbols = st.text_area(
        "Trading Symbols (one per line)",
        "EURUSD=X\nGBPUSD=X\nBTC-USD\nETH-USD\nGC=F",
        help="Use Yahoo Finance symbols. Add '=X' for forex, '-USD' for crypto"
    )
    
    symbol_list = [s.strip() for s in symbols.split('\n') if s.strip()]
    
    # Analysis Settings
    st.subheader("Analysis Parameters")
    min_score_threshold = st.slider(
        "Minimum Signal Score", 60, 95, 75,
        help="Higher values = fewer but higher quality signals"
    )
    
    timeframe = st.select_slider(
        "Primary Timeframe",
        options=["1m", "5m", "15m", "30m", "1h", "4h"],
        value="5m"
    )
    
    # Risk Management
    st.subheader("Risk Management")
    risk_percent = st.slider("Risk per Trade (%)", 0.5, 5.0, 1.0, 0.1)
    
    # Action Buttons
    st.divider()
    analyze_btn = st.button("🚀 Analyze Markets", use_container_width=True)
    backtest_btn = st.button("📈 Run Backtest", use_container_width=True)
    clear_cache = st.button("🗑️ Clear Cache")
    
    if clear_cache:
        st.cache_data.clear()
        st.success("Cache cleared!")

# Main Area
st.title("🤖 ZOHA NEURAL-100 PRO TERMINAL v2.0")
st.markdown("""
    <p style='color: #8b949e; font-size: 14px;'>
    Advanced Multi-Timeframe Analysis | Real Technical Indicators | Risk Management
    </p>
""", unsafe_allow_html=True)

# Performance Metrics
config = load_config()
st.divider()
metrics = st.columns(4)
with metrics[0]:
    st.metric("Session Win Rate", f"{config['win_rate']:.1f}%")
with metrics[1]:
    st.metric("Total Signals", config['total_signals'])
with metrics[2]:
    st.metric("Profitable", config['profitable_signals'])
with metrics[3]:
    st.metric("Risk/Trade", f"{config['risk_per_trade']:.1f}%")

st.divider()

# Analysis Results Area
if analyze_btn:
    if not symbol_list:
        st.error("❌ No symbols configured. Add symbols in the sidebar.")
    else:
        results_container = st.container()
        cols = st.columns(3)
        count = 0
        
        for symbol in symbol_list:
            with st.spinner(f"Analyzing {symbol}..."):
                engine = TechnicalAnalysisEngine(symbol)
                result = engine.calculate_signal_score(timeframe)
                
                if "error" in result:
                    st.warning(f"⚠️ {symbol}: {result['error']}")
                    continue
                
                if result['score'] >= min_score_threshold:
                    with cols[count % 3]:
                        # Score-based styling
                        score_class = "high-score" if result['score'] >= 80 else \
                                     "medium-score" if result['score'] >= 70 else "low-score"
                        
                        direction_color = "#00ffa3" if "CALL" in result['direction'] else \
                                        "#ff2e63" if "PUT" in result['direction'] else "#ffa500"
                        
                        # Condition list
                        conditions_html = "<br>".join([
                            f"{'✅' if 'PASSED' in v else '❌'} {k}: {v}"
                            for k, v in result['analysis'].items()
                        ])
                        
                        st.markdown(f"""
                            <div class="signal-card {score_class}">
                                <div style="display:flex; justify-content:space-between; align-items:center;">
                                    <span class="pair-name">{symbol}</span>
                                    <span class="legend-badge" style="background: rgba(35, 134, 54, 0.2); color: #238636; border-color: #238636;">
                                        Score: {result['score']}
                                    </span>
                                </div>
                                <div class="direction-text" style="color:{direction_color};">
                                    {result['direction']}
                                </div>
                                <div style="display:flex; justify-content:space-between; align-items:end;">
                                    <div>
                                        <div style="font-size:11px; color:#8b949e;">NEURAL SCORE</div>
                                        <div class="score-box" style="color: {'#00f2ff' if result['score'] >= 80 else '#ffa500'};">
                                            {result['score]}
                                        </div>
                                    </div>
                                    <div style="text-align:right;">
                                        <div style="font-size:11px; color:#8b949e;">CURRENT PRICE</div>
                                        <div style="font-size:18px; font-weight:bold;">${result['price']:.5f}</div>
                                    </div>
                                </div>
                                <hr style="border-color:#30363d; margin: 15px 0;">
                                <div class="condition-list">
                                    {conditions_html}
                                </div>
                                <div style="margin-top: 15px; font-size: 10px; color: #666;">
                                    📊 {result['timestamp']} | TF: {result['timeframe']}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Risk calculation
                        atr = result.get('atr', 0.001)
                        position_size = (risk_percent / 100) / atr if atr > 0 else 0
                        
                        with st.expander("📊 Risk Calculation"):
                            st.write(f"**Stop Loss:** {result['price'] - atr:.5f}")
                            st.write(f"**Take Profit:** {result['price'] + atr*2:.5f}")
                            st.write(f"**Position Size:** {position_size:.4f} units")
                        
                        count += 1
        
        if count == 0:
            st.warning(f"⚠️ No signals met the minimum score threshold of {min_score_threshold}")

# Backtesting Section
if backtest_btn:
    st.subheader("📈 Backtesting Results")
    
    symbol = st.selectbox("Select Symbol for Backtest", symbol_list)
    days = st.slider("Backtest Period (Days)", 1, 30, 7)
    
    if st.button("Start Backtest"):
        with st.spinner("Running backtest..."):
            engine = TechnicalAnalysisEngine(symbol)
            df = engine.fetch_data(f"{days}d", "5m")
            
            if df.empty:
                st.error("No historical data for backtesting")
            else:
                # Simulate signals
                signals = []
                balance = 1000
                wins = 0
                total = 0
                
                for i in range(50, len(df), 5):
                    current_price = df['Close'].iloc[i]
                    # Simplified signal generation for backtest
                    engine.score = 0
                    mock_result = {
                        "score": np.random.randint(70, 101),
                        "direction": "UP" if df['Close'].iloc[i] > df['SMA_20'].iloc[i] else "DOWN",
                        "price": current_price
                    }
                    
                    # Simulate outcome
                    outcome = "WIN" if mock_result['direction'] == "UP" and df['Close'].iloc[min(i+5, len(df)-1)] > current_price else "LOSS"
                    if outcome == "WIN":
                        wins += 1
                    total += 1
                    balance += 10 if outcome == "WIN" else -10
                    signals.append(outcome)
                
                # Display results
                win_rate = (wins / total * 100) if total > 0 else 0
                st.metric("Backtest Win Rate", f"{win_rate:.2f}%")
                st.metric("Final Balance", f"${balance:.2f}")
                
                # Update config
                config['win_rate'] = win_rate
                config['total_signals'] += total
                config['profitable_signals'] += wins
                save_config(config)
                
                # Chart
                st.bar_chart(pd.Series(signals).value_counts())

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; font-size: 11px;'>
        <p>⚠️ Trading involves substantial risk. This tool is for educational purposes only.</p>
        <p>Always backtest strategies and use proper risk management. Never risk more than you can afford to lose.</p>
        <p>v2.0 | Multi-Timeframe Analysis Engine</p>
    </div>
""", unsafe_allow_html=True)
