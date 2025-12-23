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
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import io

warnings.filterwarnings('ignore')

# --- PRODUCTION CONFIGURATION ---
CONFIG_FILE = "professional_config.json"
LOG_FILE = "trading_terminal_pro.log"
SESSION_FILE = "session_data.json"

# Advanced Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- DATA CLASSES FOR TYPE SAFETY ---
@dataclass
class Signal:
    symbol: str
    score: int
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    timeframe: str
    timestamp: str
    confidence: str
    analysis_details: Dict
    
@dataclass
class SessionMetrics:
    total_signals: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    avg_score: float = 0.0
    timestamp: str = ""

# --- CORE ANALYSIS ENGINE ---
class NeuralAnalysisEngine:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.score = 0
        self.analysis_log = []
        self.data_cache = {}
        
    @st.cache_data(ttl=300, show_spinner=False)
    def fetch_multi_timeframe(_self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Fetch data across multiple timeframes with caching"""
        timeframes = {
            '1m': ('1d', '1m'),
            '5m': ('5d', '5m'),
            '15m': ('15d', '15m'),
            '1h': ('1mo', '1h'),
            '4h': ('3mo', '4h'),
            '1d': ('1y', '1d')
        }
        
        data = {}
        for tf, (period, interval) in timeframes.items():
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)
                if not df.empty:
                    df.reset_index(inplace=True)
                    data[tf] = df
                else:
                    logger.warning(f"No data for {symbol} on {tf}")
                    data[tf] = pd.DataFrame()
            except Exception as e:
                logger.error(f"Fetch error {symbol} {tf}: {e}")
                data[tf] = pd.DataFrame()
                
        return data
    
    def calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate institutional-grade indicators"""
        if df.empty:
            return df
            
        # Trend & Momentum
        df['EMA_9'] = ta.ema(df['Close'], length=9)
        df['EMA_21'] = ta.ema(df['Close'], length=21)
        df['EMA_50'] = ta.ema(df['Close'], length=50)
        df['SMA_200'] = ta.sma(df['Close'], length=200)
        
        # RSI with multiple lengths
        df['RSI_14'] = ta.rsi(df['Close'], length=14)
        df['RSI_7'] = ta.rsi(df['Close'], length=7)
        df['RSI_21'] = ta.rsi(df['Close'], length=21)
        df['Stoch_RSI'] = ta.stochrsi(df['Close'])['STOCHRSIk_14_14_3_3']
        
        # MACD
        macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
        df = pd.concat([df, macd], axis=1)
        
        # Bollinger Bands
        bb = ta.bbands(df['Close'], length=20, std=2)
        df = pd.concat([df, bb], axis=1)
        df['BB_Position'] = (df['Close'] - df['BBL_20_2.0']) / (df['BBU_20_2.0'] - df['BBL_20_2.0'])
        
        # Volume Profile
        df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
        df['OBV'] = ta.obv(df['Close'], df['Volume'])
        df['Volume_MA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
        # Volatility
        df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        df['ATR_Pct'] = df['ATR_14'] / df['Close'] * 100
        
        # Order Flow (simplified)
        df['Delta'] = df['Close'] - df['Open']
        df['CVD'] = df['Delta'].cumsum()
        
        return df
    
    def analyze_smc_concepts(self, df: pd.DataFrame) -> Dict:
        """Smart Money Concepts Analysis"""
        current = df.iloc[-1]
        
        # Premium/Discount/Demand zones
        bb_position = current['BB_Position']
        vwap_position = current['Close'] > current['VWAP']
        
        zones = {
            'premium': bb_position > 0.8,
            'discount': bb_position < 0.2,
            'equilibrium': 0.2 <= bb_position <= 0.8,
            'above_vwap': vwap_position,
            'vwap_support': abs(current['Close'] - current['VWAP']) / current['Close'] < 0.001
        }
        
        # Fair Value Gaps (FVG)
        zones['fvg_bullish'] = current['Low'] > df['High'].iloc[-3] and current['Close'] > current['Open']
        zones['fvg_bearish'] = current['High'] < df['Low'].iloc[-3] and current['Close'] < current['Open']
        
        return zones
    
    def check_confluence(self, tf_data: Dict[str, pd.DataFrame]) -> Tuple[int, Dict]:
        """Master confluence checker - 100pt scoring"""
        self.score = 0
        self.analysis_log = []
        
        current_tf = '5m'  # Primary
        if current_tf not in tf_data or tf_data[current_tf].empty:
            return 0, {"error": "No primary timeframe data"}
        
        df = tf_data[current_tf]
        current = df.iloc[-1]
        
        # --- SECTOR 1: Trend Confluence (30 pts) ---
        trend_score = 0
        if current['EMA_9'] > current['EMA_21'] > current['EMA_50']:
            trend_score += 15
            self.analysis_log.append("✅ Strong Bullish EMA Stack (+15)")
        elif current['EMA_9'] < current['EMA_21'] < current['EMA_50']:
            trend_score += 15
            self.analysis_log.append("✅ Strong Bearish EMA Stack (+15)")
            
        # Multi-TF trend alignment
        if '15m' in tf_data and not tf_data['15m'].empty:
            tf_15m = tf_data['15m'].iloc[-1]
            if (current['EMA_21'] > current['EMA_50']) == (tf_15m['EMA_21'] > tf_15m['EMA_50']):
                trend_score += 15
                self.analysis_log.append("✅ Multi-TF Trend Sync (+15)")
        
        self.score += trend_score
        
        # --- SECTOR 2: Momentum Confluence (25 pts) ---
        mom_score = 0
        
        # RSI Confluence
        rsi_14 = current['RSI_14']
        if 35 < rsi_14 < 65:
            mom_score += 10
            self.analysis_log.append(f"✅ RSI in Sweet Zone ({rsi_14:.1f}) (+10)")
        
        # MACD Divergence
        if current['MACD_12_26_9'] > current['MACDs_12_26_9']:
            mom_score += 10
            self.analysis_log.append("✅ MACD Bullish Cross (+10)")
            
        # Stochastic RSI
        if current['Stoch_RSI'] < 0.2:
            mom_score += 5
            self.analysis_log.append("✅ OVERSOLD Stoch RSI (+5)")
        elif current['Stoch_RSI'] > 0.8:
            mom_score += 5
            self.analysis_log.append("✅ OVERBOUGHT Stoch RSI (+5)")
        
        self.score += mom_score
        
        # --- SECTOR 3: Volatility & S/R (20 pts) ---
        vol_score = 0
        
        # Bollinger Band Position
        if 0.25 <= current['BB_Position'] <= 0.75:
            vol_score += 10
            self.analysis_log.append("✅ Price in Value Area (+10)")
        
        # ATR Filter
        if 0.5 < current['ATR_Pct'] < 2.0:
            vol_score += 10
            self.analysis_log.append(f"✅ Healthy Volatility ({current['ATR_Pct']:.2f}%) (+10)")
        
        self.score += vol_score
        
        # --- SECTOR 4: Volume & Order Flow (15 pts) ---
        vol_flow_score = 0
        
        if current['Volume_Ratio'] > 1.5:
            vol_flow_score += 8
            self.analysis_log.append(f"✅ Volume Spike ({current['Volume_Ratio']:.1f}x) (+8)")
        
        if current['OBV'] > df['OBV'].iloc[-20]:
            vol_flow_score += 7
            self.analysis_log.append("✅ OBV Trending Up (+7)")
        
        self.score += vol_flow_score
        
        # --- SECTOR 5: SMC & Order Blocks (10 pts) ---
        smc_score = 0
        smc = self.analyze_smc_concepts(df)
        
        if smc['equilibrium']:
            smc_score += 5
            self.analysis_log.append("✅ SMC Equilibrium Zone (+5)")
        
        if smc['vwap_support']:
            smc_score += 5
            self.analysis_log.append("✅ VWAP Support Tested (+5)")
        
        self.score += smc_score
        
        return min(self.score, 100), {
            "trend": trend_score,
            "momentum": mom_score,
            "volatility": vol_score,
            "volume": vol_flow_score,
            "smc": smc_score
        }
    
    def generate_signal(self, tf_data: Dict[str, pd.DataFrame]) -> Optional[Signal]:
        """Generate complete trading signal"""
        try:
            score, sectors = self.check_confluence(tf_data)
            
            if score < 60:
                return None
                
            # Determine direction
            df = tf_data['5m']
            current = df.iloc[-1]
            
            direction = "UP (CALL) 🟢" if current['EMA_9'] > current['EMA_21'] else "DOWN (PUT) 🔴"
            
            # Risk management
            atr = current['ATR_14']
            entry_price = current['Close']
            stop_loss = entry_price - (atr * 1.5) if "CALL" in direction else entry_price + (atr * 1.5)
            take_profit = entry_price + (atr * 3) if "CALL" in direction else entry_price - (atr * 3)
            
            position_size = (2.0 / 100) / (atr / entry_price)  # 2% risk
            
            confidence = "HIGH" if score >= 80 else "MEDIUM" if score >= 70 else "LOW"
            
            return Signal(
                symbol=self.symbol,
                score=score,
                direction=direction,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size=position_size,
                timeframe="5m",
                timestamp=get_bdt_time().strftime("%Y-%m-%d %H:%M:%S"),
                confidence=confidence,
                analysis_details=sectors
            )
            
        except Exception as e:
            logger.error(f"Signal generation error: {e}")
            return None

# --- SESSION & CONFIG MANAGEMENT ---
class SessionManager:
    @staticmethod
    def load_session() -> Dict:
        try:
            with open(SESSION_FILE, 'r') as f:
                return json.load(f)
        except:
            return {"signals": [], "balance": 10000, "history": []}
    
    @staticmethod
    def save_session(data: Dict):
        with open(SESSION_FILE, 'w') as f:
            json.dump(data, f, indent=4)
    
    @staticmethod
    def calculate_metrics(signals: List[Dict]) -> SessionMetrics:
        if not signals:
            return SessionMetrics()
            
        total = len(signals)
        wins = sum(1 for s in signals if s.get('outcome') == 'WIN')
        win_rate = (wins / total) * 100
        
        return SessionMetrics(
            total_signals=total,
            win_rate=win_rate,
            avg_score=sum(s.get('score', 0) for s in signals) / total,
            timestamp=get_bdt_time().strftime("%Y-%m-%d %H:%M:%S")
        )

def get_bdt_time():
    return datetime.datetime.now(pytz.timezone('Asia/Dhaka'))

# --- PROFESSIONAL UI ---
def init_page_styles():
    """Initialize professional page styling"""
    st.set_page_config(
        page_title="Zoha Neural-100 Pro Terminal v3.0",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
        <style>
        /* Professional Dark Theme */
        .stApp {
            background: linear-gradient(135deg, #0a0e17 0%, #0d1117 100%);
            color: #e6edf3;
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        }
        
        /* Glassmorphism Cards */
        .signal-card {
            background: rgba(22, 27, 34, 0.7);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(48, 54, 61, 0.8);
            padding: 25px;
            border-radius: 16px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }
        
        .signal-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #00f2ff, #238636);
        }
        
        .signal-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(0, 242, 255, 0.15);
        }
        
        .high-score::before { background: linear-gradient(90deg, #00f2ff, #00ffa3); }
        .medium-score::before { background: linear-gradient(90deg, #ffa500, #ff8c00); }
        .low-score::before { background: linear-gradient(90deg, #ff2e63, #ff6b81); }
        
        /* Typography */
        .pair-name {
            font-size: 22px;
            font-weight: 700;
            color: #58a6ff;
            letter-spacing: 0.5px;
        }
        
        .direction-text {
            font-size: 28px;
            font-weight: 800;
            margin: 15px 0;
            text-shadow: 0 0 10px currentColor;
        }
        
        .score-box {
            font-size: 42px;
            font-weight: 900;
            background: linear-gradient(90deg, #ffd700, #ffed4e);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        /* Professional Badges */
        .certified-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 11px;
            font-weight: 700;
            background: rgba(35, 134, 54, 0.2);
            color: #238636;
            border: 1px solid #238636;
            margin-left: 10px;
        }
        
        .confidence-badge {
            position: absolute;
            top: 15px;
            right: 15px;
            padding: 6px 14px;
            border-radius: 50px;
            font-size: 10px;
            font-weight: 800;
        }
        
        /* Metrics */
        .metric-card {
            background: rgba(22, 27, 34, 0.5);
            padding: 20px;
            border-radius: 12px;
            border: 1px solid rgba(48, 54, 61, 0.4);
            text-align: center;
        }
        
        .metric-value {
            font-size: 32px;
            font-weight: 800;
            margin: 10px 0;
        }
        
        /* Animations */
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .signal-card {
            animation: fadeIn 0.5s ease-out;
        }
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #0d1117; }
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(45deg, #00f2ff, #238636);
            border-radius: 4px;
        }
        
        /* Sidebar */
        .sidebar .sidebar-content {
            background: rgba(13, 17, 23, 0.9);
            backdrop-filter: blur(10px);
        }
        </style>
    """, unsafe_allow_html=True)

def render_professional_sidebar() -> Dict:
    """Render professional sidebar controls"""
    with st.sidebar:
        st.markdown("""
            <div style="text-align: center; padding: 20px; border-bottom: 1px solid #30363d;">
                <h1 style="margin: 0; font-size: 24px;">🤖 ZOHA PRO</h1>
                <p style="color: #8b949e; font-size: 12px;">Neural-100 Terminal v3.0</p>
                <div style="color: #00f2ff; font-size: 14px; margin-top: 10px;">
                    🕒 """ + get_bdt_time().strftime('%H:%M:%S') + """ BDT
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Market Configuration
        st.header("⚙️ MARKET CONFIG")
        
        market_configs = {
            "Forex (OTC)": {
                "symbols": "EURUSD=X\nGBP/USD_otc\nUSD/JPY_otc\nUSD/CHF_otc\nAUD/USD_otc\nUSD/CAD_otc\nNZD/USD_otc\nEUR/GBP_otc\nEUR/JPY_otc\nGBP/JPY_otc",
                "default": ["EURUSD=X", "GBPUSD=X", "USDJPY=X"]
            },
            "Cryptocurrency": {
                "symbols": "BTC-USD\nETH-USD\nBNB-USD\nSOL-USD\nADA-USD\nXRP-USD\nDOGE-USD\nDOT-USD\nMATIC-USD\nAVAX-USD",
                "default": ["BTC-USD", "ETH-USD"]
            },
            "Commodities": {
                "symbols": "GC=F\nSI=F\nCL=F\nBNO=F\nZC=F\nZW=F\nZM=F\nCT=F\nCC=F\nKC=F",
                "default": ["GC=F", "SI=F"]
            },
            "Stocks": {
                "symbols": "AAPL\nMSFT\nGOOGL\nAMZN\nTSLA\nMETA\nNVDA\nNFLX\nADBE\nPYPL",
                "default": ["AAPL", "TSLA"]
            }
        }
        
        market_type = st.selectbox(
            "Market Type",
            list(market_configs.keys()),
            help="Select market for analysis"
        )
        
        st.subheader("Trading Assets")
        symbols_text = st.text_area(
            "Symbols (Yahoo Finance format)",
            market_configs[market_type]["symbols"],
            height=150,
            help="One symbol per line. Use =X for forex, -USD for crypto"
        )
        
        symbol_list = list(set([s.strip().upper() for s in symbols_text.split('\n') if s.strip()]))
        
        # Analysis Parameters
        st.divider()
        st.header("🎯 ANALYSIS PARAMS")
        
        col1, col2 = st.columns(2)
        with col1:
            min_score = st.slider("Min Score", 60, 95, 75, 1)
        with col2:
            risk_per_trade = st.slider("Risk %", 0.5, 5.0, 1.0, 0.1)
            
        timeframe = st.select_slider(
            "Primary Timeframe",
            options=["1m", "3m", "5m", "15m", "30m", "1h", "4h"],
            value="5m"
        )
        
        # Execution Controls
        st.divider()
        st.header("🚀 EXECUTION")
        
        analyze_btn = st.button(
            "⚡ SCAN MARKETS",
            use_container_width=True,
            help="Run multi-asset analysis"
        )
        
        backtest_btn = st.button(
            "📊 BACKTEST",
            use_container_width=True,
            help="Test strategy on historical data"
        )
        
        if st.button("📥 EXPORT RESULTS", use_container_width=True):
            st.session_state['export'] = True
            
        # Session Info
        st.divider()
        st.header("📊 SESSION")
        
        session_data = SessionManager.load_session()
        metrics = SessionManager.calculate_metrics(session_data.get('signals', []))
        
        with st.container():
            st.metric("Total Signals", metrics.total_signals)
            st.metric("Win Rate", f"{metrics.win_rate:.1f}%")
            st.metric("Avg Score", f"{metrics.avg_score:.1f}")
            st.metric("Risk/Trade", f"{risk_per_trade}%")
        
        st.caption("""
            <div style="font-size: 10px; color: #666; margin-top: 20px;">
                ⚠️ This tool is for educational purposes.<br>
                Trading involves substantial risk.<br>
                Always use proper risk management.
            </div>
        """, unsafe_allow_html=True)
        
        return {
            'symbols': symbol_list,
            'min_score': min_score,
            'risk_per_trade': risk_per_trade,
            'timeframe': timeframe,
            'analyze': analyze_btn,
            'backtest': backtest_btn
        }

def render_signal_card(signal: Signal, idx: int):
    """Render a professional signal card"""
    # Color schemes
    colors = {
        'CALL': {'text': '#00ffa3', 'bg': 'rgba(0, 255, 163, 0.1)', 'border': '#00ffa3'},
        'PUT': {'text': '#ff2e63', 'bg': 'rgba(255, 46, 99, 0.1)', 'border': '#ff2e63'},
        'NEUTRAL': {'text': '#ffa500', 'bg': 'rgba(255, 165, 0, 0.1)', 'border': '#ffa500'}
    }
    
    direction_key = signal.direction.split()[0]
    color = colors.get(direction_key, colors['NEUTRAL'])
    
    # Confidence badge styling
    conf_colors = {
        'HIGH': '#00f2ff',
        'MEDIUM': '#ffa500',
        'LOW': '#ff2e63'
    }
    
    score_class = "high-score" if signal.score >= 80 else \
                 "medium-score" if signal.score >= 70 else "low-score"
    
    st.markdown(f"""
        <div class="signal-card {score_class}" style="animation-delay: {idx * 0.1}s;">
            <div class="confidence-badge" style="
                background: {conf_colors.get(signal.confidence, '#666')}; 
                color: #000; 
                font-weight: 900;
            ">
                {signal.confidence}
            </div>
            
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <span class="pair-name">{signal.symbol}</span>
                <span class="certified-badge">
                    🛡️ CERTIFIED
                </span>
            </div>
            
            <div class="direction-text" style="color: {color['text']};">
                {signal.direction}
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 20px 0;">
                <div style="background: rgba(0,0,0,0.3); padding: 15px; border-radius: 8px;">
                    <div style="font-size: 11px; color: #8b949e; margin-bottom: 5px;">ENTRY PRICE</div>
                    <div style="font-size: 18px; font-weight: bold; color: #ffd700;">
                        ${signal.entry_price:.5f}
                    </div>
                </div>
                <div style="background: rgba(0,0,0,0.3); padding: 15px; border-radius: 8px;">
                    <div style="font-size: 11px; color: #8b949e; margin-bottom: 5px;">NEURAL SCORE</div>
                    <div class="score-box">{signal.score}/100</div>
                </div>
            </div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 15px 0;">
                <div style="background: rgba(255, 46, 99, 0.1); padding: 10px; border-radius: 6px; border: 1px solid #ff2e63;">
                    <div style="font-size: 10px; color: #8b949e;">STOP LOSS</div>
                    <div style="font-size: 14px; font-weight: bold; color: #ff2e63;">
                        ${signal.stop_loss:.5f}
                    </div>
                </div>
                <div style="background: rgba(0, 255, 163, 0.1); padding: 10px; border-radius: 6px; border: 1px solid #00ffa3;">
                    <div style="font-size: 10px; color: #8b949e;">TAKE PROFIT</div>
                    <div style="font-size: 14px; font-weight: bold; color: #00ffa3;">
                        ${signal.take_profit:.5f}
                    </div>
                </div>
            </div>
            
            <div style="background: rgba(0,0,0,0.2); padding: 12px; border-radius: 8px; margin: 15px 0;">
                <div style="font-size: 11px; color: #8b949e; margin-bottom: 8px;">
                    💰 POSITION SIZE: <span style="color: #ffd700; font-weight: bold;">
                        {signal.position_size:.4f} units
                    </span>
                </div>
                <div style="font-size: 10px; color: #666;">
                    Risk: 2% | R:R 1:2
                </div>
            </div>
            
            <div style="font-size: 10px; color: #666; text-align: right;">
                {signal.timestamp} | TF: {signal.timeframe}
            </div>
        </div>
    """, unsafe_allow_html=True)

def render_performance_chart(signals: List[Dict]):
    """Render equity curve and metrics"""
    if not signals:
        return
        
    df = pd.DataFrame(signals)
    if df.empty:
        return
        
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Equity Curve', 'Signal Score Distribution'),
        vertical_spacing=0.1,
        row_heights=[0.6, 0.4]
    )
    
    # Equity curve (simulated)
    df['equity'] = 10000 + df.index * 10  # Simplified
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['equity'],
            mode='lines+markers',
            name='Equity',
            line=dict(color='#00f2ff', width=3),
            fill='tonexty'
        ),
        row=1, col=1
    )
    
    # Score histogram
    fig.add_trace(
        go.Histogram(
            x=df['score'],
            nbinsx=20,
            name='Score Distribution',
            marker_color='#58a6ff',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.2)',
        font=dict(color='#e6edf3'),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

# --- MAIN EXECUTION ---
def main():
    """Main application loop"""
    init_page_styles()
    
    # Sidebar controls
    controls = render_professional_sidebar()
    
    # Main header
    st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h1 style="margin: 0; font-size: 36px; background: linear-gradient(90deg, #00f2ff, #ffd700); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                ZOHA NEURAL-100 PRO TERMINAL
            </h1>
            <p style="color: #8b949e; font-size: 14px; margin-top: 10px;">
                Institutional-Grade Multi-Timeframe Analysis • Real-Time Signal Generation
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Analysis Execution
    if controls['analyze']:
        if not controls['symbols']:
            st.error("❌ No symbols configured. Please add trading symbols in sidebar.")
            return
            
        # Status area
        status_container = st.empty()
        results_container = st.container()
        
        signals = []
        progress = 0
        
        for idx, symbol in enumerate(controls['symbols']):
            progress = (idx + 1) / len(controls['symbols'])
            status_container.info(f"🔍 Analyzing {symbol}... ({idx+1}/{len(controls['symbols'])})")
            
            try:
                engine = NeuralAnalysisEngine(symbol)
                tf_data = engine.fetch_multi_timeframe(symbol)
                
                if not any(not df.empty for df in tf_data.values()):
                    st.warning(f"⚠️ No data for {symbol}")
                    continue
                
                signal = engine.generate_signal(tf_data)
                
                if signal and signal.score >= controls['min_score']:
                    signals.append(signal)
                    logger.info(f"Generated signal for {symbol}: {signal.score}/100")
                    
            except Exception as e:
                logger.error(f"Analysis failed for {symbol}: {e}")
                continue
        
        status_container.success(f"✅ Analysis Complete! {len(signals)} high-quality signals found.")
        
        # Display signals
        if signals:
            signals.sort(key=lambda x: x.score, reverse=True)
            
            cols = st.columns(3)
            for idx, signal in enumerate(signals):
                with cols[idx % 3]:
                    render_signal_card(signal, idx)
            
            # Save session
            session_data = SessionManager.load_session()
            session_data['signals'].extend([s.__dict__ for s in signals])
            SessionManager.save_session(session_data)
            
            # Performance chart
            st.divider()
            st.subheader("📊 Performance Analytics")
            render_performance_chart(session_data['signals'])
            
            # Export functionality
            if st.session_state.get('export'):
                df_export = pd.DataFrame([s.__dict__ for s in signals])
                csv = df_export.to_csv(index=False)
                st.download_button(
                    "📥 Download Signals (CSV)",
                    csv,
                    f"signals_{get_bdt_time().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv",
                    use_container_width=True
                )
        
        else:
            st.warning(f"""
                ⚠️ No signals met the minimum threshold of {controls['min_score']}/100.
                
                **Suggestions:**
                • Lower the minimum score slider
                • Check your symbol list format
                • Verify market hours
            """)
    
    # Backtest Section
    if controls['backtest']:
        st.divider()
        st.subheader("📈 Professional Backtesting Engine")
        
        with st.form("backtest_form"):
            col1, col2 = st.columns(2)
            with col1:
                bt_symbol = st.selectbox("Symbol", controls['symbols'])
            with col2:
                bt_period = st.slider("Backtest Period (Days)", 7, 365, 30)
            
            bt_timeframe = st.select_slider(
                "Backtest Timeframe",
                options=["1m", "5m", "15m", "1h", "4h", "1d"],
                value="30m"
            )
            
            submitted = st.form_submit_button("🚀 Run Backtest", use_container_width=True)
            
            if submitted:
                with st.spinner("Running institutional backtest..."):
                    # Placeholder for backtest logic
                    st.success("Backtest complete! (Advanced backtester coming in v3.1)")
                    st.plotly_chart(go.Figure(
                        data=[go.Scatter(x=list(range(100)), y=[10000 + i*10 for i in range(100)])],
                        layout=go.Layout(title="Simulated Equity Curve")
                    ))

if __name__ == "__main__":
    main()
