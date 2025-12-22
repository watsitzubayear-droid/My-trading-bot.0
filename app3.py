# app3.py - Streamlit Version
import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import pytz
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# ============ CONFIGURATION ============
MIN_CONFIDENCE_THRESHOLD = 70
BANGLADESH_TZ = pytz.timezone('Asia/Dhaka')

# ============ SESSION STATE INIT ============
if 'running' not in st.session_state:
    st.session_state.running = False
if 'signal_history' not in st.session_state:
    st.session_state.signal_history = []
if 'page' not in st.session_state:
    st.session_state.page = 'login'
if 'username' not in st.session_state:
    st.session_state.username = ''
if 'password' not in st.session_state:
    st.session_state.password = ''

# ============ ENUMS & CLASSES ============
class Signal(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0

class TechnicalAnalyzer:
    @staticmethod
    def calculate_sma(data, period):
        return data['close'].rolling(window=period).mean()
    
    @staticmethod
    def calculate_ema(data, period):
        return data['close'].ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(data, period=14):
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def calculate_macd(data):
        ema12 = data['close'].ewm(span=12, adjust=False).mean()
        ema26 = data['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal
    
    @staticmethod
    def calculate_bollinger_bands(data, period=20, std_dev=2):
        sma = data['close'].rolling(window=period).mean()
        std = data['close'].rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, sma, lower
    
    @staticmethod
    def calculate_stochastic(data, period=14):
        low_min = data['low'].rolling(window=period).min()
        high_max = data['high'].rolling(window=period).max()
        k_percent = 100 * (data['close'] - low_min) / (high_max - low_min)
        d_percent = k_percent.rolling(window=3).mean()
        return k_percent, d_percent
    
    @staticmethod
    def calculate_adx(data, period=14):
        plus_dm = data['high'].diff()
        minus_dm = data['low'].diff()
        plus_dm = plus_dm.where(plus_dm > 0, 0)
        minus_dm = minus_dm.where(minus_dm > 0, 0)
        
        tr1 = data['high'] - data['low']
        tr2 = abs(data['high'] - data['close'].shift())
        tr3 = abs(data['low'] - data['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        return adx, plus_di, minus_di
    
    @staticmethod
    def calculate_fibonacci_levels(data):
        recent_high = data['high'].rolling(window=50).max().iloc[-1]
        recent_low = data['low'].rolling(window=50).min().iloc[-1]
        diff = recent_high - recent_low
        return {
            '0%': recent_high,
            '23.6%': recent_high - 0.236 * diff,
            '38.2%': recent_high - 0.382 * diff,
            '50%': recent_high - 0.5 * diff,
            '61.8%': recent_high - 0.618 * diff,
            '78.6%': recent_high - 0.786 * diff,
            '100%': recent_low
        }
    
    @staticmethod
    def calculate_vwap(data):
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        return (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
    
    @staticmethod
    def detect_support_resistance(data, window=20):
        recent_data = data.tail(50)
        resistance = recent_data['high'].rolling(window=window).max().iloc[-1]
        support = recent_data['low'].rolling(window=window).min().iloc[-1]
        return support, resistance

class OTCDataProvider:
    def __init__(self):
        self.base_rate = 110.50
        self.volatility = 0.08
        
    def fetch_data(self, periods=200):
        data = []
        current_time = datetime.now(BANGLADESH_TZ)
        
        for i in range(periods):
            timestamp = current_time - timedelta(minutes=(periods-i)*3)
            change = np.random.normal(0, self.volatility/100)
            self.base_rate *= (1 + change)
            
            hour = timestamp.hour
            if 10 <= hour <= 16:
                self.base_rate += np.random.normal(0, 0.02)
            
            open_price = self.base_rate + np.random.uniform(-0.05, 0.05)
            high_price = open_price + np.random.uniform(0, 0.10)
            low_price = open_price - np.random.uniform(0, 0.10)
            close_price = low_price + np.random.uniform(0, high_price - low_price)
            
            data.append({
                'timestamp': timestamp,
                'open': round(open_price, 4),
                'high': round(high_price, 4),
                'low': round(low_price, 4),
                'close': round(close_price, 4),
                'volume': np.random.randint(5000, 80000)
            })
        
        return pd.DataFrame(data)

class StrategyEngine:
    def __init__(self):
        self.analyzer = TechnicalAnalyzer()
        self.weights = {
            'ma_crossover': 0.15,
            'rsi': 0.10,
            'macd': 0.15,
            'bollinger': 0.10,
            'stochastic': 0.10,
            'adx': 0.10,
            'fibonacci': 0.10,
            'vwap': 0.10,
            'support_resistance': 0.10
        }
    
    def calculate_strategy_scores(self, data):
        scores = {}
        
        data['SMA_20'] = self.analyzer.calculate_sma(data, 20)
        data['SMA_50'] = self.analyzer.calculate_sma(data, 50)
        scores['ma_crossover'] = 1 if data['SMA_20'].iloc[-1] > data['SMA_50'].iloc[-1] else -1
        
        data['RSI'] = self.analyzer.calculate_rsi(data)
        rsi_val = data['RSI'].iloc[-1]
        if rsi_val < 30: scores['rsi'] = 1
        elif rsi_val > 70: scores['rsi'] = -1
        else: scores['rsi'] = 0
        
        data['MACD'], data['MACD_SIGNAL'] = self.analyzer.calculate_macd(data)
        scores['macd'] = 1 if data['MACD'].iloc[-1] > data['MACD_SIGNAL'].iloc[-1] else -1
        
        data['BB_UPPER'], data['BB_MIDDLE'], data['BB_LOWER'] = self.analyzer.calculate_bollinger_bands(data)
        price = data['close'].iloc[-1]
        if price <= data['BB_LOWER'].iloc[-1]: scores['bollinger'] = 1
        elif price >= data['BB_UPPER'].iloc[-1]: scores['bollinger'] = -1
        else: scores['bollinger'] = 0
        
        data['STOCH_K'], data['STOCH_D'] = self.analyzer.calculate_stochastic(data)
        k_val = data['STOCH_K'].iloc[-1]
        if k_val < 20: scores['stochastic'] = 1
        elif k_val > 80: scores['stochastic'] = -1
        else: scores['stochastic'] = 0
        
        data['ADX'], data['PLUS_DI'], data['MINUS_DI'] = self.analyzer.calculate_adx(data)
        adx_val = data['ADX'].iloc[-1]
        plus_di = data['PLUS_DI'].iloc[-1]
        minus_di = data['MINUS_DI'].iloc[-1]
        if adx_val > 25: scores['adx'] = 1 if plus_di > minus_di else -1
        else: scores['adx'] = 0
        
        fib_levels = self.analyzer.calculate_fibonacci_levels(data)
        price = data['close'].iloc[-1]
        if abs(price - fib_levels['61.8%']) < 0.10: scores['fibonacci'] = 1
        elif abs(price - fib_levels['38.2%']) < 0.10: scores['fibonacci'] = -1
        else: scores['fibonacci'] = 0
        
        data['VWAP'] = self.analyzer.calculate_vwap(data)
        scores['vwap'] = 1 if data['close'].iloc[-1] > data['VWAP'].iloc[-1] else -1
        
        support, resistance = self.analyzer.detect_support_resistance(data)
        price = data['close'].iloc[-1]
        if abs(price - support) < 0.10: scores['support_resistance'] = 1
        elif abs(price - resistance) < 0.10: scores['support_resistance'] = -1
        else: scores['support_resistance'] = 0
        
        return scores, data
    
    def generate_signal(self, data):
        scores, data = self.calculate_strategy_scores(data)
        final_score = sum(scores[strategy] * weight for strategy, weight in self.weights.items())
        total_weight = sum(self.weights.values())
        normalized_score = final_score / total_weight
        confidence = abs(normalized_score) * 100
        
        if confidence >= MIN_CONFIDENCE_THRESHOLD:
            if normalized_score > 0:
                signal_emoji = "🟢"
                signal_type = Signal.BUY
                signal_text = f"{signal_emoji} BUY {confidence:.0f}%"
            else:
                signal_emoji = "🔴"
                signal_type = Signal.SELL
                signal_text = f"{signal_emoji} SELL {confidence:.0f}%"
        else:
            signal_emoji = "❌"
            signal_type = Signal.HOLD
            signal_text = f"{signal_emoji} HOLD ({confidence:.0f}% confidence)"
        
        return signal_type, signal_text, confidence, scores, data

# ============ STREAMLIT UI ============
def login_page():
    st.markdown("""
        <style>
        .login-container { max-width: 400px; margin: auto; padding: 50px; }
        .login-title { font-size: 32px; font-weight: bold; text-align: center; margin-bottom: 30px; }
        .stButton > button { width: 100%; }
        </style>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.markdown('<div class="login-title">🔐 Trading Bot Login</div>', unsafe_allow_html=True)
        
        username = st.text_input("Username", value="admin")
        password = st.text_input("Password", type="password", value="admin123")
        
        if st.button("Login"):
            if username == "admin" and password == "admin123":
                st.session_state.page = 'bot'
                st.session_state.username = username
                st.success("✅ Login successful!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("❌ Invalid credentials!")
        
        st.markdown('</div>', unsafe_allow_html=True)

def bot_page():
    st.markdown("""
        <style>
        .main-title { font-size: 36px; font-weight: bold; text-align: center; color: #1f77b4; }
        .signal-box { padding: 30px; border-radius: 10px; text-align: center; }
        .metric-card { background-color: #f0f2f6; padding: 15px; border-radius: 8px; }
        </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="main-title">🇧🇩 USD/BDT OTC Trading Bot</div>', unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center;'>🕐 Bangladesh Time: {datetime.now(BANGLADESH_TZ).strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("▶ START BOT", key="start", disabled=st.session_state.running):
            st.session_state.running = True
            st.rerun()
        
        if st.button("⏹ STOP BOT", key="stop", disabled=not st.session_state.running):
            st.session_state.running = False
            st.rerun()
    
    # Running indicator
    status_placeholder = st.empty()
    signal_placeholder = st.empty()
    details_placeholder = st.empty()
    history_placeholder = st.empty()
    
    data_provider = OTCDataProvider()
    strategy_engine = StrategyEngine()
    
    cycle_count = 0
    
    while st.session_state.running:
        cycle_count += 1
        
        with status_placeholder:
            st.info(f"Cycle #{cycle_count} | Fetching market data...")
        
        data = data_provider.fetch_data(200)
        
        with status_placeholder:
            st.info(f"Cycle #{cycle_count} | Analyzing with 9 strategies...")
        
        signal_type, signal_text, confidence, scores, data = strategy_engine.generate_signal(data)
        current_time = datetime.now(BANGLADESH_TZ)
        
        # Update signal display
        with signal_placeholder:
            color = "green" if signal_type == Signal.BUY else ("red" if signal_type == Signal.SELL else "orange")
            st.markdown(f"""
                <div class="signal-box" style="background-color: {color}; color: white;">
                    <h1>{signal_text}</h1>
                </div>
            """, unsafe_allow_html=True)
        
        # Update details
        with details_placeholder:
            fib_levels = strategy_engine.analyzer.calculate_fibonacci_levels(data)
            support, resistance = strategy_engine.analyzer.detect_support_resistance(data)
            
            breakdown = "\n".join([
                f"{strategy.replace('_', ' ').title()}: {'✓' if score > 0 else ('✗' if score < 0 else '○')} ({score:.2f})"
                for strategy, score in scores.items()
            ])
            
            st.markdown(f"""
                <div class="metric-card">
                    <h3>Analysis Details ({current_time.strftime('%H:%M:%S')})</h3>
                    <p><strong>Rate:</strong> {data['close'].iloc[-1]:.4f} BDT/USD</p>
                    <p><strong>Confidence:</strong> {confidence:.1f}%</p>
                    <pre>{breakdown}</pre>
                    <p><strong>Key Levels:</strong> Support {support:.4f} | Resistance {resistance:.4f}</p>
                </div>
            """, unsafe_allow_html=True)
        
        # Update history
        st.session_state.signal_history.append((current_time, signal_text, data['close'].iloc[-1]))
        if len(st.session_state.signal_history) > 10:
            st.session_state.signal_history.pop(0)
        
        with history_placeholder:
            hist_df = pd.DataFrame(st.session_state.signal_history, columns=['Time', 'Signal', 'Rate'])
            hist_df['Time'] = hist_df['Time'].dt.strftime('%H:%M:%S')
            st.markdown("<h3>Signal History (Last 10)</h3>", unsafe_allow_html=True)
            st.dataframe(hist_df, use_container_width=True)
        
        # Wait 3 minutes
        with status_placeholder:
            st.success(f"Cycle #{cycle_count} | Waiting for next cycle...")
        
        for i in range(180):
            if not st.session_state.running:
                break
            time.sleep(1)
    
    if not st.session_state.running:
        with status_placeholder:
            st.error("Bot Stopped")

# ============ MAIN APP ============
def main():
    if st.session_state.page == 'login':
        login_page()
    else:
        bot_page()

if __name__ == "__main__":
    main()
