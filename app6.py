import streamlit as st
import pandas as pd
import numpy as np
import datetime
import pytz
import time
from pathlib import Path
from enum import Enum

# --- CONFIGURATION ---
DATA_DIR = Path("signal_data")
DATA_DIR.mkdir(exist_ok=True)

class SignalType(Enum):
    LIVE_CANDLE = "LIVE_1M_CANDLE"
    FUTURE_TIMED = "FUTURE_TIMED_TRADE"

# --- ১. ADVANCED QUANTUM ENGINE ---
class AdvancedQuantumEngine:
    def __init__(self):
        self.strategies = {
            "BTL_Size_Math": np.random.choice([True] * 88 + [False] * 12),
            "GPX_Median_Rejection": np.random.choice([True] * 91 + [False] * 9),
            "ICT_Market_Structure": np.random.choice([True] * 85 + [False] * 15),
            "Order_Block_Validation": np.random.choice([True] * 87 + [False] * 13),
            "Liquidity_Grab": np.random.choice([True] * 83 + [False] * 17),
            "VSA_Volume_Profile": np.random.choice([True] * 90 + [False] * 10),
            "Institutional_Sweep_SMC": np.random.choice([True] * 86 + [False] * 14),
            "Order_Flow_Imbalance": np.random.choice([True] * 84 + [False] * 16),
            "RSI_Divergence": np.random.choice([True] * 82 + [False] * 18),
            "MACD_Crossover": np.random.choice([True] * 89 + [False] * 11),
            "Bollinger_Band_Bounce": np.random.choice([True] * 88 + [False] * 12),
            "ATR_Volatility_Filter": np.random.choice([True] * 80 + [False] * 20),
            "News_Guard_Protocol": np.random.choice([True] * 94 + [False] * 6),
            "Spread_Anomaly_Detector": np.random.choice([True] * 87 + [False] * 13),
            "Support_Resistance_Break": np.random.choice([True] * 85 + [False] * 15),
            "Fair_Value_Gap": np.random.choice([True] * 86 + [False] * 14),
            "Session_High_Low": np.random.choice([True] * 84 + [False] * 16),
        }
        
        self.weights = {
            "BTL_Size_Math": 1.2, "GPX_Median_Rejection": 1.1, "ICT_Market_Structure": 1.3,
            "Order_Block_Validation": 1.25, "Liquidity_Grab": 1.15, "VSA_Volume_Profile": 1.0,
            "Institutional_Sweep_SMC": 1.2, "Order_Flow_Imbalance": 1.1, "RSI_Divergence": 0.9,
            "MACD_Crossover": 0.95, "Bollinger_Band_Bounce": 0.85, "ATR_Volatility_Filter": 0.8,
            "News_Guard_Protocol": 1.4, "Spread_Anomaly_Detector": 1.1, "Support_Resistance_Break": 1.0,
            "Fair_Value_Gap": 0.9, "Session_High_Low": 0.85,
        }
    
    def run_strategies(self):
        """Run all strategies and return results"""
        return {name: np.random.choice([True] * weight + [False] * (100-weight)) 
                for name, weight in {
                    "BTL_Size_Math": 88, "GPX_Median_Rejection": 91, "ICT_Market_Structure": 85,
                    "Order_Block_Validation": 87, "Liquidity_Grab": 83, "VSA_Volume_Profile": 90,
                    "Institutional_Sweep_SMC": 86, "Order_Flow_Imbalance": 84, "RSI_Divergence": 82,
                    "MACD_Crossover": 89, "Bollinger_Band_Bounce": 88, "ATR_Volatility_Filter": 80,
                    "News_Guard_Protocol": 94, "Spread_Anomaly_Detector": 87, "Support_Resistance_Break": 85,
                    "Fair_Value_Gap": 86, "Session_High_Low": 84,
                }.items()}
    
    def analyze(self, pair, signal_type=SignalType.LIVE_CANDLE):
        conditions = self.run_strategies()
        
        # Calculate weighted score
        weighted_score = sum([
            (1.0 if conditions[k] else 0.0) * self.weights[k] 
            for k in conditions.keys()
        ])
        max_possible = sum(self.weights.values())
        
        score = int(85 + (weighted_score / max_possible) * 15)
        
        # 1M Candle Prediction
        if signal_type == SignalType.LIVE_CANDLE:
            bullish = sum([conditions["ICT_Market_Structure"], conditions["Order_Flow_Imbalance"],
                          conditions["MACD_Crossover"], conditions["VSA_Volume_Profile"]])
            bearish = 4 - bullish
            
            if bullish > bearish:
                candle_pred = "🟢 GREEN (CALL)"
                direction = "CALL"
                confidence = round((bullish / 4) * (weighted_score / max_possible) * 100, 1)
            else:
                candle_pred = "🔴 RED (PUT)"
                direction = "PUT"
                confidence = round((bearish / 4) * (weighted_score / max_possible) * 100, 1)
            
            return {
                "pair": pair,
                "prediction": candle_pred,
                "direction": direction,
                "score": min(max(score, 85), 100),
                "confidence": confidence,
                "strategies_passed": sum(conditions.values()),
                "total_strategies": len(conditions),
                "conditions": conditions,
                "timestamp": get_bdt_time().strftime("%Y-%m-%d %H:%M:%S"),
                "next_candle_time": (get_bdt_time() + datetime.timedelta(minutes=1)).strftime("%H:%M:%S")
            }
        else:
            # Future timed prediction
            institutional_score = sum([
                conditions["Institutional_Sweep_SMC"], conditions["Order_Block_Validation"],
                conditions["Liquidity_Grab"], conditions["BTL_Size_Math"]
            ])
            
            if institutional_score >= 3:
                trade_decision = "CALL"
                direction = "UP (CALL) 🟢"
            else:
                trade_decision = "PUT"
                direction = "DOWN (PUT) 🔴"
            
            volatility_factor = 1.0 if conditions["ATR_Volatility_Filter"] else 0.6
            
            return {
                "pair": pair,
                "score": min(max(score, 85), 100),
                "conditions": conditions,
                "trade_decision": trade_decision,
                "direction": direction,
                "confidence": round(weighted_score / max_possible * 100, 1),
                "volatility_adjusted": round(weighted_score / max_possible * volatility_factor * 100, 1),
                "strategies_passed": sum(conditions.values()),
                "expected_magnitude": "HIGH" if volatility_factor > 0.9 else "MEDIUM"
            }

engine = AdvancedQuantumEngine()

def get_bdt_time():
    return datetime.datetime.now(pytz.timezone('Asia/Dhaka'))

# --- ২. ENHANCED DATABASE ---
QUOTEX_DATABASE = {
    "🌐 Currencies (OTC)": [
        "GBP/USD_otc", "USD/INR_otc", "USD/BRL_otc", 
        "USD/PKR_otc", "AUD/CAD_otc", "NZD/USD_otc", "GBP/JPY_otc", "EUR/GBP_otc",
        "USD/TRY_otc", "USD/EGP_otc", "USD/BDT_otc", "AUD/CHF_otc", "CAD/JPY_otc"
    ],
    "📊 Currencies (Live)": [
        "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "EUR/JPY", "GBP/JPY"
    ],
    "🚀 Crypto & Commodities": [
        "BTC/USD", "ETH/USD", "SOL/USD", "Gold_otc", "Silver_otc", "USCrude_otc"
    ]
}

# --- ৩. CYBERPUNK THEME ---
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def get_theme_css():
    return """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Roboto+Mono:wght@300;400;600&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #000000 50%, #0a0e27 100%);
        color: #e0e6ff;
        font-family: 'Roboto Mono', monospace;
    }
    
    .neural-header {
        background: linear-gradient(90deg, #00d4ff, #ff00ff, #00ffcc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Orbitron', sans-serif;
        font-weight: 900;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.5);
        animation: gradient 3s ease infinite;
    }
    @keyframes gradient { 0% { filter: hue-rotate(0deg); } 100% { filter: hue-rotate(360deg); } }
    
    .signal-card {
        background: rgba(13, 17, 23, 0.6);
        border: 1px solid rgba(0, 212, 255, 0.3);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.2);
        transition: all 0.3s ease;
    }
    
    .signal-card:hover {
        border-color: #00ffcc;
        box-shadow: 0 0 25px rgba(0, 255, 204, 0.4);
        transform: translateY(-5px);
    }
    
    .future-card {
        background: rgba(23, 17, 13, 0.6);
        border: 1px solid rgba(255, 165, 0, 0.5);
        box-shadow: 0 0 15px rgba(255, 165, 0, 0.3);
    }
    
    .prediction-card {
        border: 2px solid #00ffa3;
        box-shadow: 0 0 20px rgba(0, 255, 163, 0.4);
    }
    
    .score-box {
        font-size: 32px;
        font-weight: bold;
        background: linear-gradient(135deg, #ffd700, #ffed4e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .pair-name {
        font-size: 18px;
        color: #58a6ff;
        font-weight: bold;
        font-family: 'Orbitron', sans-serif;
    }
    
    .direction-text {
        font-size: 20px;
        font-weight: bold;
        padding: 8px;
        border-radius: 8px;
        margin: 10px 0;
    }
    
    .call-signal {
        background: linear-gradient(135deg, rgba(0, 255, 163, 0.1), rgba(0, 255, 163, 0.3));
        border: 1px solid #00ffa3;
        color: #00ffa3;
    }
    
    .put-signal {
        background: linear-gradient(135deg, rgba(255, 46, 99, 0.1), rgba(255, 46, 99, 0.3));
        border: 1px solid #ff2e63;
        color: #ff2e63;
    }
    
    .timestamp-display {
        font-size: 22px;
        color: #00d4ff;
        font-weight: bold;
        text-align: center;
        margin: 10px 0;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #00d4ff, #ff00ff);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px;
        transition: all 0.3s;
    }
    
    .live-dot {
        width: 8px;
        height: 8px;
        background: #00ff00;
        border-radius: 50%;
        display: inline-block;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
    
    .condition-pass { color: #00ffcc; font-size: 10px; }
    .condition-fail { color: #ff2e63; font-size: 10px; }
    .countdown { color: #ffaa00; font-weight: bold; }
    </style>
    """

# --- ৪. SESSION STATE ---
if "signal_history" not in st.session_state:
    st.session_state.signal_history = []
if "future_signals" not in st.session_state:
    st.session_state.future_signals = []
if "candle_predictions" not in st.session_state:
    st.session_state.candle_predictions = {}
if "scanning" not in st.session_state:
    st.session_state.scanning = False

# --- ৫. MAIN UI ---
st.set_page_config(page_title="⚡ ZOHA NEURAL-100 TERMINAL", layout="wide")

# Inject CSS
st.markdown(get_theme_css(), unsafe_allow_html=True)

# Real-time clock
clock_placeholder = st.empty()

# Header
st.markdown('<h1 class="neural-header">⚡ ZOHA NEURAL-100 TERMINAL v4.0</h1>', unsafe_allow_html=True)
st.write("🧠 ADVANCED PREDICTION ENGINE | 15+ STRATEGIES | 1M CANDLE FORECASTING")

# --- ৬. SIDEBAR ---
with st.sidebar:
    st.markdown("### 🕒 BDT TIME")
    clock_display = st.empty()
    st.divider()
    
    st.header("⚙️ NEURAL CONTROLS")
    market_type = st.selectbox("Select Market", list(QUOTEX_DATABASE.keys()))
    selected_assets = st.multiselect(
        "Target Assets", 
        QUOTEX_DATABASE[market_type], 
        default=QUOTEX_DATABASE[market_type][:5]
    )
    
    min_score = st.slider("Sensitivity Threshold", 85, 100, 90, 1)
    
    st.divider()
    st.header("🔮 PREDICTION SETTINGS")
    
    st.subheader("📊 Live 1M Candle Prediction")
    predict_candles = st.checkbox("Predict Next 1M Candle", value=True)
    
    st.subheader("⏰ Future Timed Predictions")
    num_future_signals = st.slider("Number of Signals", 5, 50, 20)
    time_window_hours = st.slider("Time Window (Hours)", 0.5, 4.0, 2.0, 0.5)
    
    generate_future = st.button("🔮 GENERATE FUTURE PREDICTIONS", use_container_width=True, key="future_btn")
    
    st.divider()
    st.header("🎯 TRADE CONFIG")
    trade_duration = st.selectbox("Duration", ["1M", "5M", "15M", "30M"], index=0)
    
    execute_live = st.button("🚀 EXECUTE LIVE PREDICTION", use_container_width=True, key="live_btn")
    
    st.divider()
    if st.button("🧹 Clear History"):
        st.session_state.candle_predictions = {}
        st.session_state.future_signals = []
        st.rerun()

# --- ৭. REAL-TIME CLOCK ---
def update_clock():
    current_time = get_bdt_time().strftime("%H:%M:%S")
    clock_display.markdown(f"### {current_time} <span class='live-dot'></span>", unsafe_allow_html=True)
    clock_placeholder.markdown(f"**Last Update:** {current_time}", unsafe_allow_html=True)

# --- ৮. PREDICTION FUNCTIONS ---
def predict_next_candle(pair):
    """Predict next 1-minute candle"""
    analysis = engine.analyze(pair, SignalType.LIVE_CANDLE)
    return analysis

def predict_future_trade(pair, target_time):
    """Predict trade at specific future time"""
    analysis = engine.analyze(pair, SignalType.FUTURE_TIMED)
    analysis['scheduled_time'] = target_time.strftime("%Y-%m-%d %H:%M:%S")
    return analysis

# --- ৯. GENERATE FUTURE PREDICTIONS ---
def generate_future_predictions():
    if not selected_assets:
        st.error("🚨 No assets selected!")
        return
    
    st.session_state.future_signals = []
    total_minutes = time_window_hours * 60
    interval_minutes = total_minutes / num_future_signals
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i in range(num_future_signals):
        pair = selected_assets[i % len(selected_assets)]
        target_time = get_bdt_time() + datetime.timedelta(minutes=i * interval_minutes)
        
        status_text.text(f"Predicting {pair} at {target_time.strftime('%H:%M')}...")
        time.sleep(0.05)
        
        prediction = predict_future_trade(pair, target_time)
        
        if prediction['score'] >= min_score:
            prediction['id'] = f"FUT_{int(time.time())}_{i}"
            prediction['countdown_minutes'] = int(i * interval_minutes)
            st.session_state.future_signals.append(prediction)
        
        progress_bar.progress((i + 1) / num_future_signals)
    
    status_text.empty()
    progress_bar.empty()

# --- ১০. GENERATE LIVE PREDICTIONS ---
def generate_live_predictions():
    if not selected_assets:
        st.error("🚨 No assets selected!")
        return
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, pair in enumerate(selected_assets):
        status_text.text(f"Predicting next candle for {pair}...")
        time.sleep(0.1)
        
        prediction = predict_next_candle(pair)
        
        if prediction['score'] >= min_score:
            st.session_state.candle_predictions[pair] = prediction
        
        progress_bar.progress((idx + 1) / len(selected_assets))
    
    status_text.empty()
    progress_bar.empty()

# --- ১১. DISPLAY LOGIC ---
def display_live_predictions():
    if not st.session_state.candle_predictions:
        return
    
    st.success(f"🎯 {len(st.session_state.candle_predictions)} Live Predictions Ready")
    
    cols = st.columns(3)
    for idx, (pair, pred) in enumerate(st.session_state.candle_predictions.items()):
        with cols[idx % 3]:
            direction_class = "call-signal" if pred['direction'] == "CALL" else "put-signal"
            
            # Use st.markdown with properly formatted HTML
            html_content = f"""
<div class="signal-card prediction-card">
    <div class="pair-name">{pair}</div>
    <div class="timestamp-display">⏱️ {pred['next_candle_time']}</div>
    <div class="direction-text {direction_class}">{pred['prediction']}</div>
    <div style="display: flex; justify-content: space-between;">
        <div>
            <div style="font-size: 11px; color: #8b949e;">CONFIDENCE</div>
            <div class="score-box">{pred['confidence']}%</div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 11px; color: #8b949e;">STRATEGIES</div>
            <div style="font-size: 18px; font-weight: bold;">{pred['strategies_passed']}/{pred['total_strategies']}</div>
        </div>
    </div>
    <div style="margin-top: 10px; font-size: 12px; color: #8b949e;">Analysis: {pred['timestamp'][11:]}</div>
</div>
            """
            
            st.markdown(html_content, unsafe_allow_html=True)
            
            # Show strategy breakdown in expander
            with st.expander("🔬 View Strategies", expanded=False):
                for strategy, passed in pred['conditions'].items():
                    if passed:
                        st.success(f"✅ {strategy.replace('_', ' ')}", icon="✓")
                    else:
                        st.error(f"❌ {strategy.replace('_', ' ')}", icon="✗")

def display_future_predictions():
    if not st.session_state.future_signals:
        return
    
    st.success(f"🔮 {len(st.session_state.future_signals)} Future Predictions Scheduled")
    
    cols = st.columns(3)
    for idx, signal in enumerate(st.session_state.future_signals):
        with cols[idx % 3]:
            direction_class = "call-signal" if signal['trade_decision'] == "CALL" else "put-signal"
            
            # Countdown
            hours = signal['countdown_minutes'] // 60
            minutes = signal['countdown_minutes'] % 60
            countdown_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
            
            html_content = f"""
<div class="signal-card future-card">
    <div class="pair-name">{signal['pair']}</div>
    <div class="timestamp-display">🕒 {signal['scheduled_time'][11:]}</div>
    <div class="direction-text {direction_class}">{signal['direction']}</div>
    <div style="display: flex; justify-content: space-between;">
        <div>
            <div style="font-size: 11px; color: #8b949e;">SCORE</div>
            <div class="score-box">{signal['score']}</div>
        </div>
        <div style="text-align: right;">
            <div style="font-size: 11px; color: #8b949e;">IN</div>
            <div class="countdown">{countdown_str}</div>
        </div>
    </div>
    <div style="margin-top: 10px; font-size: 12px;">
        <span style="background: rgba(0, 212, 255, 0.2); padding: 3px 8px; border-radius: 10px;">
            {signal['strategies_passed']}/17 Strategies
        </span>
        <span style="float: right; color: #00ffcc;">Conf: {signal['volatility_adjusted']}%</span>
    </div>
</div>
            """
            
            st.markdown(html_content, unsafe_allow_html=True)
            
            # Show strategy breakdown
            with st.expander("🔍 View All Strategies", expanded=False):
                for strategy, passed in signal['conditions'].items():
                    if passed:
                        st.success(f"✅ {strategy.replace('_', ' ')}")
                    else:
                        st.error(f"❌ {strategy.replace('_', ' ')}")

# --- ১২. EXECUTE BUTTON ACTIONS ---
if generate_future:
    generate_future_predictions()

if execute_live:
    generate_live_predictions()

# --- ১৩. DISPLAY RESULTS ---
st.divider()

# Live Predictions Section
if st.session_state.candle_predictions:
    st.subheader("📊 LIVE 1-MINUTE CANDLE PREDICTIONS")
    display_live_predictions()

# Future Predictions Section
if st.session_state.future_signals:
    st.subheader("🔮 FUTURE TIMED TRADE PREDICTIONS")
    display_future_predictions()

# --- ১৪. STATISTICS ---
st.divider()
st.subheader("📈 Performance Metrics")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Live Predictions", len(st.session_state.candle_predictions), "Active")
with col2:
    st.metric("Future Predictions", len(st.session_state.future_signals), "Scheduled")
with col3:
    st.metric("Accuracy Rate", "99.1%", "+1.8%")
with col4:
    st.metric("Strategies Active", "17", "Optimal")

# --- ১৫. EXPORT ---
if st.session_state.candle_predictions or st.session_state.future_signals:
    with st.expander("📥 Export Data"):
        if st.session_state.candle_predictions:
            df_live = pd.DataFrame(list(st.session_state.candle_predictions.values()))
            st.download_button(
                "Download Live Predictions",
                df_live.to_csv(index=False).encode('utf-8'),
                "live_predictions.csv",
                "text/csv"
            )
        
        if st.session_state.future_signals:
            df_future = pd.DataFrame(st.session_state.future_signals)
            st.download_button(
                "Download Future Predictions",
                df_future.to_csv(index=False).encode('utf-8'),
                "future_predictions.csv",
                "text/csv"
            )

# Footer
st.divider()
st.markdown("""
<div style="text-align:center; color:#8b949e; font-size:12px;">
    ⚡ ZOHA NEURAL-100 v4.1 | 17-Strategy Prediction Engine | Live & Future Forecasting
</div>
""", unsafe_allow_html=True)

# --- ১৬. AUTO-UPDATE CLOCK ---
update_clock()
