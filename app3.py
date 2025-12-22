# quotex_trading_bot.py
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
DEEP_ANALYSIS_THRESHOLD = 65  # Lower threshold for deep analysis mode
BANGLADESH_TZ = pytz.timezone('Asia/Dhaka')

# Quotex-style OTC pairs (simulated)
QUOTEX_PAIRS = {
    'USD/BDT': {'volatility': 0.08, 'base_rate': 110.50},
    'EUR/USD': {'volatility': 0.05, 'base_rate': 1.0850},
    'GBP/USD': {'volatility': 0.06, 'base_rate': 1.2700},
    'USD/JPY': {'volatility': 0.04, 'base_rate': 148.50},
    'AUD/USD': {'volatility': 0.055, 'base_rate': 0.6580},
    'USD/CAD': {'volatility': 0.045, 'base_rate': 1.3550},
    'EUR/GBP': {'volatility': 0.04, 'base_rate': 0.8550},
    'USD/CHF': {'volatility': 0.035, 'base_rate': 0.8850}
}

# ============ SESSION STATE ============
if 'running' not in st.session_state:
    st.session_state.running = False
if 'signal_history' not in st.session_state:
    st.session_state.signal_history = []
if 'page' not in st.session_state:
    st.session_state.page = 'bot'
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'deep_analysis' not in st.session_state:
    st.session_state.deep_analysis = False

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

class MultiPairDataProvider:
    def __init__(self):
        self.pairs = QUOTEX_PAIRS
        
    def fetch_pair_data(self, pair, periods=200):
        """Fetch data for a specific pair"""
        config = self.pairs[pair]
        base_rate = config['base_rate']
        volatility = config['volatility']
        
        data = []
        current_time = datetime.now(BANGLADESH_TZ)
        
        for i in range(periods):
            timestamp = current_time - timedelta(minutes=(periods-i)*3)
            change = np.random.normal(0, volatility/100)
            base_rate *= (1 + change)
            
            # Market hours pattern
            hour = timestamp.hour
            if 10 <= hour <= 16:
                base_rate += np.random.normal(0, volatility/2)
            
            open_price = base_rate + np.random.uniform(-volatility/10, volatility/10)
            high_price = open_price + np.random.uniform(0, volatility/8)
            low_price = open_price - np.random.uniform(0, volatility/8)
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
        
        # Calculate all indicators
        data['SMA_20'] = self.analyzer.calculate_sma(data, 20)
        data['SMA_50'] = self.analyzer.calculate_sma(data, 50)
        data['RSI'] = self.analyzer.calculate_rsi(data)
        data['MACD'], data['MACD_SIGNAL'] = self.analyzer.calculate_macd(data)
        data['BB_UPPER'], data['BB_MIDDLE'], data['BB_LOWER'] = self.analyzer.calculate_bollinger_bands(data)
        data['STOCH_K'], data['STOCH_D'] = self.analyzer.calculate_stochastic(data)
        data['ADX'], data['PLUS_DI'], data['MINUS_DI'] = self.analyzer.calculate_adx(data)
        data['VWAP'] = self.analyzer.calculate_vwap(data)
        
        # Score each strategy
        scores['ma_crossover'] = 1 if data['SMA_20'].iloc[-1] > data['SMA_50'].iloc[-1] else -1
        
        rsi_val = data['RSI'].iloc[-1]
        scores['rsi'] = 1 if rsi_val < 30 else (-1 if rsi_val > 70 else 0)
        
        scores['macd'] = 1 if data['MACD'].iloc[-1] > data['MACD_SIGNAL'].iloc[-1] else -1
        
        price = data['close'].iloc[-1]
        if price <= data['BB_LOWER'].iloc[-1]: scores['bollinger'] = 1
        elif price >= data['BB_UPPER'].iloc[-1]: scores['bollinger'] = -1
        else: scores['bollinger'] = 0
        
        k_val = data['STOCH_K'].iloc[-1]
        scores['stochastic'] = 1 if k_val < 20 else (-1 if k_val > 80 else 0)
        
        adx_val = data['ADX'].iloc[-1]
        plus_di = data['PLUS_DI'].iloc[-1]
        minus_di = data['MINUS_DI'].iloc[-1]
        if adx_val > 25: scores['adx'] = 1 if plus_di > minus_di else -1
        else: scores['adx'] = 0
        
        fib_levels = self.analyzer.calculate_fibonacci_levels(data)
        if abs(price - fib_levels['61.8%']) < 0.10: scores['fibonacci'] = 1
        elif abs(price - fib_levels['38.2%']) < 0.10: scores['fibonacci'] = -1
        else: scores['fibonacci'] = 0
        
        scores['vwap'] = 1 if price > data['VWAP'].iloc[-1] else -1
        
        support, resistance = self.analyzer.detect_support_resistance(data)
        if abs(price - support) < 0.10: scores['support_resistance'] = 1
        elif abs(price - resistance) < 0.10: scores['support_resistance'] = -1
        else: scores['support_resistance'] = 0
        
        return scores, data
    
    def generate_signal(self, data, deep_mode=False):
        scores, data = self.calculate_strategy_scores(data)
        final_score = sum(scores[strategy] * weight for strategy, weight in self.weights.items())
        total_weight = sum(self.weights.values())
        normalized_score = final_score / total_weight
        confidence = abs(normalized_score) * 100
        
        # Use lower threshold in deep analysis mode
        threshold = DEEP_ANALYSIS_THRESHOLD if deep_mode else MIN_CONFIDENCE_THRESHOLD
        
        if confidence >= threshold:
            if normalized_score > 0:
                return Signal.BUY, f"🟢 BUY {confidence:.0f}%", confidence, scores, data
            else:
                return Signal.SELL, f"🔴 SELL {confidence:.0f}%", confidence, scores, data
        else:
            return Signal.HOLD, f"❌ HOLD ({confidence:.0f}% confidence)", confidence, scores, data

def analyze_all_pairs():
    """Analyze all Quotex pairs and return the best opportunity"""
    provider = MultiPairDataProvider()
    engine = StrategyEngine()
    
    results = []
    
    for pair_name, config in QUOTEX_PAIRS.items():
        # Fetch data for this pair
        data = provider.fetch_pair_data(pair_name, periods=200)
        
        # Generate signal
        signal_type, signal_text, confidence, scores, data = engine.generate_signal(data)
        
        # If HOLD, try deep analysis
        if signal_type == Signal.HOLD:
            data_deep = provider.fetch_pair_data(pair_name, periods=300)
            signal_type, signal_text, confidence, scores, data = engine.generate_signal(data_deep, deep_mode=True)
        
        results.append({
            'pair': pair_name,
            'signal': signal_type,
            'signal_text': signal_text,
            'confidence': confidence,
            'rate': data['close'].iloc[-1],
            'scores': scores
        })
    
    # Sort by confidence (descending)
    results.sort(key=lambda x: x['confidence'], reverse=True)
    
    return results

# ============ STREAMLIT UI ============
def create_sidebar():
    """Create sidebar with menu and theme toggle"""
    with st.sidebar:
        st.markdown("## ⚙️ Menu")
        
        # Theme toggle
        theme = st.radio(
            "Theme",
            ["🌙 Dark Mode", "☀️ Bright Mode"],
            index=0 if st.session_state.dark_mode else 1,
            key="theme_radio"
        )
        
        if theme == "🌙 Dark Mode":
            st.session_state.dark_mode = True
        else:
            st.session_state.dark_mode = False
        
        # Pair selector
        st.markdown("### 📊 Pairs to Analyze")
        selected_pairs = st.multiselect(
            "Select pairs",
            list(QUOTEX_PAIRS.keys()),
            default=list(QUOTEX_PAIRS.keys())[:4]
        )
        
        # Settings
        st.markdown("### ⚡ Settings")
        st.session_state.deep_analysis = st.checkbox("Enable Deep Analysis", value=True)
        
        return selected_pairs

def main_bot_interface():
    """Main trading bot interface"""
    # Apply theme
    if st.session_state.dark_mode:
        st.markdown("""
            <style>
            .stApp { background-color: #0e1117; color: white; }
            .signal-box { background-color: #262730; padding: 20px; border-radius: 10px; }
            </style>
        """, unsafe_allow_html=True)
    
    # Bangladesh Clock at top
    clock_placeholder = st.empty()
    
    # Main title
    st.markdown('<h1 style="text-align: center;">🇧🇩 Quotex Multi-Pair Trading Bot</h1>', unsafe_allow_html=True)
    
    # Control buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("▶ START BOT", disabled=st.session_state.running, key="start_btn"):
            st.session_state.running = True
            st.rerun()
        
        if st.button("⏹ STOP BOT", disabled=not st.session_state.running, key="stop_btn"):
            st.session_state.running = False
            st.rerun()
    
    # Status indicator
    status_placeholder = st.empty()
    
    # Results container
    results_placeholder = st.empty()
    
    # History
    history_placeholder = st.container()
    
    cycle_count = 0
    
    while st.session_state.running:
        cycle_count += 1
        
        # Update BDT clock
        with clock_placeholder:
            current_bdt = datetime.now(BANGLADESH_TZ).strftime('%Y-%m-%d %H:%M:%S')
            st.markdown(f"<h3 style='text-align: center;'>🕐 Bangladesh Time: {current_bdt}</h3>", unsafe_allow_html=True)
        
        with status_placeholder:
            st.info(f"🔄 Cycle #{cycle_count} | Analyzing all Quotex pairs...")
        
        # Analyze all pairs
        all_results = analyze_all_pairs()
        
        # Find best signal
        best_signal = None
        for result in all_results:
            if result['signal'] != Signal.HOLD:
                best_signal = result
                break
        
        # If no high-confidence signals, show top HOLD
        if best_signal is None:
            best_signal = all_results[0]
        
        # Display results
        with results_placeholder:
            col_left, col_center, col_right = st.columns([1, 2, 1])
            
            with col_center:
                # Main signal
                signal_color = "green" if best_signal['signal'] == Signal.BUY else ("red" if best_signal['signal'] == Signal.SELL else "orange")
                st.markdown(f"""
                    <div class="signal-box">
                        <h2>{best_signal['signal_text']}</h2>
                        <h3>🎯 Pair: {best_signal['pair']}</h3>
                        <p>Rate: {best_signal['rate']:.4f}</p>
                        <a href="https://quotex.com/en/trade/{best_signal['pair'].replace('/', '')}" target="_blank">
                            Open on Quotex →
                        </a>
                    </div>
                """, unsafe_allow_html=True)
            
            # All pairs table
            st.markdown("### 📊 All Pairs Analysis")
            df_display = pd.DataFrame([
                {
                    'Pair': r['pair'],
                    'Signal': r['signal_text'],
                    'Confidence': f"{r['confidence']:.1f}%",
                    'Rate': f"{r['rate']:.4f}",
                    'Action': 'Trade' if r['signal'] != Signal.HOLD else 'Wait'
                } for r in all_results
            ])
            st.dataframe(df_display, use_container_width=True)
        
        # Update history
        st.session_state.signal_history.append({
            'time': datetime.now(BANGLADESH_TZ).strftime('%H:%M:%S'),
            'pair': best_signal['pair'],
            'signal': best_signal['signal_text'],
            'rate': best_signal['rate']
        })
        
        if len(st.session_state.signal_history) > 10:
            st.session_state.signal_history.pop(0)
        
        with history_placeholder:
            st.markdown("### 📝 Signal History (Last 10)")
            hist_df = pd.DataFrame(st.session_state.signal_history)
            st.dataframe(hist_df, use_container_width=True)
        
        # Wait 3 minutes
        with status_placeholder:
            st.success(f"✅ Cycle #{cycle_count} complete. Waiting 3 minutes...")
        
        for i in range(180):
            if not st.session_state.running:
                break
            
            # Update clock every second
            with clock_placeholder:
                current_bdt = datetime.now(BANGLADESH_TZ).strftime('%Y-%m-%d %H:%M:%S')
                st.markdown(f"<h3 style='text-align: center;'>🕐 Bangladesh Time: {current_bdt}</h3>", unsafe_allow_html=True)
            
            time.sleep(1)
    
    if not st.session_state.running:
        with status_placeholder:
            st.error("⏹ Bot Stopped")

# ============ MAIN APP ============
def main():
    st.set_page_config(
        page_title="Quotex Trading Bot",
        page_icon="🇧🇩",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'signal_history' not in st.session_state:
        st.session_state.signal_history = []
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    
    # Create sidebar menu
    selected_pairs = create_sidebar()
    
    # Main interface
    main_bot_interface()

if __name__ == "__main__":
    main()
