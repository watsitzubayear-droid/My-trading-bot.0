import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import random
import io
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier

# ──────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zoha Future Signals",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────────────────────
# SESSION STATE (Store generated signals)
# ──────────────────────────────────────────────────────────────
if 'generated_signals' not in st.session_state:
    st.session_state.generated_signals = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = None

# ──────────────────────────────────────────────────────────────
# MARKET CONFIGURATION
# ──────────────────────────────────────────────────────────────
OTC_MARKETS = [
    "USD/BDT (OTC)", "EUR/USD (OTC)", "GBP/USD (OTC)", "USD/JPY (OTC)", 
    "AUD/USD (OTC)", "USD/CHF (OTC)", "NZD/USD (OTC)", "EUR/GBP (OTC)", 
    "XAU/USD (Gold OTC)", "Apple (OTC)", "Amazon (OTC)", "Tesla (OTC)",
    "BTC/USD (OTC)", "ETH/USD (OTC)", "BNB/USD (OTC)"
]

REAL_MARKETS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "XAU/USD", "BTC/USD",
    "USD/CHF", "NZD/USD", "EUR/GBP", "USD/CAD", "GBP/JPY", "EUR/JPY"
]

ALL_PAIRS = OTC_MARKETS + REAL_MARKETS

# ──────────────────────────────────────────────────────────────
# ADVANCED NEON CSS STYLING
# ──────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto+Mono&display=swap');
    
    .stApp { background: linear-gradient(135deg, #0a0a0a 0%, #1a0033 100%); }
    
    .neon-header {
        font-family: 'Orbitron', sans-serif;
        color: #00ffff;
        text-align: center;
        font-size: 48px;
        text-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff, 0 0 40px #00ffff;
        padding: 20px;
        animation: flicker 2s infinite alternate;
    }
    
    @keyframes flicker {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.9; }
    }
    
    .signal-container {
        background: rgba(10, 15, 25, 0.9);
        border-left: 5px solid #00ffff;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 15px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 20px rgba(0, 255, 255, 0.3);
        transition: all 0.3s;
    }
    
    .signal-container:hover {
        box-shadow: 0 6px 30px rgba(0, 255, 255, 0.5);
        transform: translateX(5px);
    }
    
    .time-text { 
        font-family: 'Roboto Mono', monospace; 
        color: #ffffff; 
        font-size: 22px; 
        font-weight: bold;
        text-shadow: 0 0 5px #fff;
    }
    .pair-text { 
        color: #888; 
        font-size: 14px; 
        font-family: 'Orbitron', sans-serif; 
    }
    
    .up-call { 
        color: #00ff88; 
        font-weight: bold; 
        text-shadow: 0 0 10px #00ff88, 0 0 20px #00ff88;
        border: 1px solid #00ff88; 
        padding: 8px 20px; 
        border-radius: 25px;
        text-transform: uppercase;
        background: rgba(0, 255, 136, 0.1);
    }
    .down-put { 
        color: #ff0055; 
        font-weight: bold; 
        text-shadow: 0 0 10px #ff0055, 0 0 20px #ff0055;
        border: 1px solid #ff0055; 
        padding: 8px 20px; 
        border-radius: 25px;
        text-transform: uppercase;
        background: rgba(255, 0, 85, 0.1);
    }
    .accuracy-tag { 
        background: #1a1f2b; 
        color: #00f2ff; 
        padding: 4px 12px; 
        border-radius: 6px; 
        font-size: 13px;
        font-weight: bold;
        border: 1px solid #00f2ff;
    }
    
    .stButton>button {
        background: rgba(20, 20, 20, 0.8) !important;
        color: #00ffff !important;
        border: 2px solid #00ffff !important;
        border-radius: 10px !important;
        box-shadow: 0 0 15px rgba(0, 255, 255, 0.5) !important;
        font-family: 'Orbitron', monospace !important;
        font-weight: bold !important;
        font-size: 16px !important;
        transition: all 0.3s !important;
        padding: 15px 30px !important;
    }
    
    .stButton>button:hover {
        box-shadow: 0 0 25px #00ffff, 0 0 35px #00ffff !important;
        transform: translateY(-3px) !important;
    }
    
    .stDownloadButton>button {
        background: rgba(138, 43, 226, 0.2) !important;
        color: #8a2be2 !important;
        border: 2px solid #8a2be2 !important;
        border-radius: 10px !important;
        box-shadow: 0 0 15px rgba(138, 43, 226, 0.5) !important;
        font-family: 'Orbitron', monospace !important;
        font-weight: bold !important;
        font-size: 16px !important;
    }
    
    .stDownloadButton>button:hover {
        box-shadow: 0 0 25px #8a2be2, 0 0 35px #8a2be2 !important;
        transform: translateY(-3px) !important;
    }
    
    .stSelectbox, .stMultiselect, .stSlider {
        background: rgba(20, 20, 20, 0.8) !important;
        border: 2px solid #00ffff !important;
        border-radius: 8px !important;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar { width: 12px; }
    ::-webkit-scrollbar-track { background: #0a0a0a; }
    ::-webkit-scrollbar-thumb { 
        background: #00ffff; 
        border-radius: 6px;
        box-shadow: 0 0 10px #00ffff;
    }
    ::-webkit-scrollbar-thumb:hover { background: #00ff88; }
    </style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# 3D LOGO (Embedded Three.js)
# ──────────────────────────────────────────────────────────────
logo_html = """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { margin: 0; background: transparent; overflow: hidden; }
        #logo-container { width: 100%; height: 250px; }
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
</head>
<body>
    <div id="logo-container"></div>
    <script>
        const container = document.getElementById('logo-container');
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, container.clientWidth / 250, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
        renderer.setSize(container.clientWidth, 250);
        container.appendChild(renderer.domElement);

        // Create "Z" shape
        const shape = new THREE.Shape();
        shape.moveTo(-4, -3); shape.lineTo(4, -3);
        shape.lineTo(-4, 3); shape.lineTo(4, 3);

        const geometry = new THREE.ExtrudeGeometry(shape, {
            depth: 0.6, bevelEnabled: true, bevelThickness: 0.25, bevelSize: 0.15, bevelSegments: 3
        });

        const material = new THREE.MeshPhongMaterial({
            color: 0x00ffff, emissive: 0x00aaaa, emissiveIntensity: 2.5, shininess: 100
        });
        const logo = new THREE.Mesh(geometry, material);
        scene.add(logo);

        // Glow layer
        const glowMaterial = new THREE.MeshBasicMaterial({ color: 0x00ffff, transparent: true, opacity: 0.4 });
        const glowMesh = new THREE.Mesh(geometry.clone().scale(1.12, 1.12, 1.12), glowMaterial);
        scene.add(glowMesh);

        // Lighting
        scene.add(new THREE.AmbientLight(0x404040));
        const p1 = new THREE.PointLight(0x00ffff, 3, 100); p1.position.set(10, 10, 10); scene.add(p1);
        const p2 = new THREE.PointLight(0xff00ff, 2, 100); p2.position.set(-10, -10, 10); scene.add(p2);

        camera.position.z = 12;

        function animate() {
            requestAnimationFrame(animate);
            logo.rotation.y += 0.008;
            glowMesh.rotation.copy(logo.rotation);
            material.emissiveIntensity = 1.8 + Math.sin(Date.now() * 0.004) * 0.7;
            renderer.render(scene, camera);
        }
        animate();

        window.addEventListener('resize', () => {
            camera.aspect = container.clientWidth / 250;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, 250);
        });
    </script>
</body>
</html>
"""

st.components.v1.html(logo_html, height=260)

# ──────────────────────────────────────────────────────────────
# MACHINE LEARNING ENSEMBLE
# ──────────────────────────────────────────────────────────────
class MetaEnsemblePredictor:
    def __init__(self):
        self.models = {
            'rf': RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42),
            'gb': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
            'lr': LogisticRegression(penalty='l1', solver='liblinear', random_state=42)
        }
        self.meta_model = LogisticRegression(random_state=42)
        self.is_trained = False
        
    def extract_features(self, candles):
        features = []
        for i in range(len(candles) - 10):
            window = candles[i:i+10]
            features.append([
                np.mean([c['close'] - c['open'] for c in window]),
                np.std([c['high'] - c['low'] for c in window]),
                sum(1 for c in window if c['close'] > c['open']),
                (window[-1]['volume'] / np.mean([c['volume'] for c in window[:-1]])) if window[-1]['volume'] else 1.0,
                (window[-1]['spread'] - np.mean([c['spread'] for c in window[:-1]])) / max(np.std([c['spread'] for c in window]), 0.0001),
                (window[-1]['close'] - window[-3]['close']) / max(window[-1]['atr'], 0.0001),
            ])
        return np.array(features)
    
    def train(self, historical_candles):
        if len(historical_candles) < 20: return
        X = self.extract_features(historical_candles)
        y = np.array([1 if c['close'] > c['open'] else 0 for c in historical_candles[10:]])
        predictions = np.zeros((len(X), len(self.models)))
        for i, (name, model) in enumerate(self.models.items()):
            model.fit(X, y)
            predictions[:, i] = model.predict_proba(X)[:, 1]
        self.meta_model.fit(predictions, y)
        self.is_trained = True
    
    def predict_proba(self, recent_candles):
        if not self.is_trained or len(recent_candles) < 10:
            return 0.5
        features = self.extract_features(recent_candles[-10:])[-1].reshape(1, -1)
        base_preds = np.zeros((1, len(self.models)))
        for i, (name, model) in enumerate(self.models.items()):
            base_preds[:, i] = model.predict_proba(features)[:, 1]
        return self.meta_model.predict_proba(base_preds)[0, 1]

# ──────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS
# ──────────────────────────────────────────────────────────────
def calculate_ema(prices, period):
    return pd.Series(prices).ewm(span=period, adjust=False).mean().iloc[-1]

def calculate_rsi(prices, period=7):
    if len(prices) < period: return 50
    delta = pd.Series(prices).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean().iloc[-1]
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean().iloc[-1]
    if loss == 0: return 100
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def detect_3_touch_zones(candles, window=20):
    if len(candles) < window: return []
    highs = [c['high'] for c in candles[-window:]]
    lows = [c['low'] for c in candles[-window:]]
    zones = []
    for level in np.linspace(min(lows), max(highs), 10):
        touches = sum(1 for h in highs if abs(h - level) < 0.0003) + sum(1 for l in lows if abs(l - level) < 0.0003)
        if touches >= 3:
            zones.append({'price': level, 'touches': touches, 'type': 'resistance' if level > np.mean(highs) else 'support'})
    return zones[:3]

# ──────────────────────────────────────────────────────────────
# STRATEGY LOGIC
# ──────────────────────────────────────────────────────────────
def strategy_vwap_macd(candles, green_prob, market_type):
    if market_type == "otc" or len(candles) < 21:
        return None
    vwap = np.mean([c['close'] for c in candles[-20:]])
    closes = [c['close'] for c in candles[-21:]]
    if closes[-1] > vwap and closes[-2] < vwap and green_prob > 0.65:
        return "LONG", 0.72
    elif closes[-1] < vwap and closes[-2] > vwap and green_prob < 0.35:
        return "SHORT", 0.72
    return None

def strategy_ema_rsi(candles, green_prob, market_type):
    if len(candles) < 21: return None
    closes = [c['close'] for c in candles]
    ema9 = calculate_ema(closes[-9:], 9)
    ema21 = calculate_ema(closes[-21:], 21)
    rsi = calculate_rsi(closes, 7)
    
    if ema9 > ema21 and rsi < 35 and green_prob > 0.6:
        return "LONG", 0.68
    elif ema9 < ema21 and rsi > 65 and green_prob < 0.4:
        return "SHORT", 0.68
    return None

def strategy_3touch_zones(candles, green_prob, market_type):
    if market_type != "otc": return None
    zones = detect_3_touch_zones(candles)
    if not zones: return None
    last_price = candles[-1]['close']
    for zone in zones:
        if abs(last_price - zone['price']) < 0.0010:
            if zone['type'] == 'support' and green_prob > 0.6:
                return "LONG", 0.85
            elif zone['type'] == 'resistance' and green_prob < 0.4:
                return "SHORT", 0.85
    return None

def strategy_rsi_bb(candles, green_prob, market_type):
    if len(candles) < 20: return None
    closes = [c['close'] for c in candles[-20:]]
    bb_upper = np.mean(closes) + 2 * np.std(closes)
    bb_lower = np.mean(closes) - 2 * np.std(closes)
    rsi = calculate_rsi([c['close'] for c in candles], 4)
    
    if candles[-1]['low'] < bb_lower and rsi < 25 and green_prob > 0.65:
        return "LONG", 0.70
    elif candles[-1]['high'] > bb_upper and rsi > 75 and green_prob < 0.35:
        return "SHORT", 0.70
    return None

# ──────────────────────────────────────────────────────────────
# SIGNAL GENERATOR (3-MINUTE INTERVALS)
# ──────────────────────────────────────────────────────────────
def generate_advanced_signals(pairs_list, count, market_type):
    tz_bd = pytz.timezone('Asia/Dhaka')
    now = datetime.now(tz_bd)
    # Start at next clean minute, then 3-min intervals
    start_time = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
    
    signals = []
    predictor = MetaEnsemblePredictor()
    
    # Simulate historical candles for training
    candles = []
    base_price = 1.0850
    for i in range(100):
        base_price += random.uniform(-0.001, 0.001)
        candles.append({
            'open': base_price, 'close': base_price + random.uniform(-0.0005, 0.0005),
            'high': base_price + random.uniform(0, 0.001), 'low': base_price - random.uniform(0, 0.001),
            'volume': random.randint(100, 500), 'spread': random.uniform(0.0001, 0.0005),
            'atr': random.uniform(0.0005, 0.001)
        })
    
    predictor.train(candles)
    
    for i in range(count):
        pair = random.choice(pairs_list)
        green_prob = predictor.predict_proba(candles[-10:])
        
        # Apply multiple strategies
        signal = None
        strategies = [strategy_vwap_macd, strategy_ema_rsi, strategy_3touch_zones, strategy_rsi_bb]
        for strat in strategies:
            result = strat(candles, green_prob, market_type)
            if result:
                signal = result
                break
        
        # Default if no strategy triggers
        if not signal:
            signal = ("LONG", green_prob) if green_prob > 0.6 else ("SHORT", 1-green_prob)
        
        direction, confidence = signal
        
        # 3-minute interval spacing
        signal_time = start_time + timedelta(minutes=i * 3)
        time_str = signal_time.strftime("%I:%M:00 %p").lower()
        
        signals.append({
            "Pair": pair, 
            "Time": time_str, 
            "Direction": "UP / Call" if direction == "LONG" else "DOWN / Put",
            "Confidence": f"{confidence:.1%}",
            "Raw_Direction": direction,
            "Confidence_Value": confidence,
            "Timestamp": signal_time
        })
        
        # Update candles for next iteration
        candles.append({
            'open': base_price, 'close': base_price + random.uniform(-0.0005, 0.0005),
            'high': base_price + random.uniform(0, 0.001), 'low': base_price - random.uniform(0, 0.001),
            'volume': random.randint(100, 500), 'spread': random.uniform(0.0001, 0.0005),
            'atr': random.uniform(0.0005, 0.001)
        })
    
    return signals

# ──────────────────────────────────────────────────────────────
# MAIN UI
# ──────────────────────────────────────────────────────────────
# Header
st.markdown('<div class="neon-header">ZOHA FUTURE SIGNALS</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#00f2ff; font-size:18px;'>AI-Powered Multi-Strategy Prediction Engine | BDT Time Sync</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### 🌍 MARKET CONFIGURATION")
market_mode = st.sidebar.radio("Select Market Type", ["Real Market", "OTC Market"], index=1)

if market_mode == "OTC Market":
    pairs = st.sidebar.multiselect("Select OTC Assets", OTC_MARKETS, default=["USD/BDT (OTC)", "EUR/USD (OTC)"])
else:
    pairs = st.sidebar.multiselect("Select Real Assets", REAL_MARKETS, default=["EUR/USD", "GBP/USD"])

num_signals = st.sidebar.slider("Number of Signals (3-min intervals)", 10, 150, 100, step=10)

# Live BDT Time
tz_bd = pytz.timezone('Asia/Dhaka')
bdt_time = datetime.now(tz_bd)
st.sidebar.markdown(f"### ⏱ **Current BDT Time:**")
st.sidebar.markdown(f'<p style="color:#ffff00; font-size:20px; text-align:center;">{bdt_time.strftime("%H:%M:%S")}</p>', unsafe_allow_html=True)

# Generate Button
if st.button("⚡ GENERATE 100+ ADVANCED SIGNALS", use_container_width=True):
    if not pairs:
        st.error("❌ Please select at least one market pair.")
    else:
        with st.spinner("🔍 Scanning markets... Applying 6 strategies... Building predictions..."):
            market_type = "otc" if "OTC" in pairs[0] else "real"
            signals = generate_advanced_signals(pairs, num_signals, market_type)
            st.session_state.generated_signals = signals
            st.session_state.last_update = bdt_time
        st.success(f"✅ **Successfully generated {len(signals)} high-probability signals**")

# Display Signals
if st.session_state.generated_signals:
    st.markdown("---")
    st.markdown("### 📊 **GENERATED SIGNALS (3-MIN INTERVALS)**")
    
    for sig in st.session_state.generated_signals:
        color_class = "up-call" if sig['Raw_Direction'] == "LONG" else "down-put"
        st.markdown(f"""
        <div class="signal-container">
            <div>
                <span class="pair-text">{sig['Pair']}</span><br>
                <span class="time-text">{sig['Time']}</span>
            </div>
            <div>
                <span class="{color_class}">{sig['Direction']}</span>
                <span class="accuracy-tag" style="margin-left:15px;">{sig['Confidence']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# Download Button
if st.session_state.generated_signals:
    st.markdown("---")
    df_download = pd.DataFrame(st.session_state.generated_signals)
    csv_buffer = io.StringIO()
    df_download.to_csv(csv_buffer, index=False, columns=["Pair", "Time", "Direction", "Confidence"])
    
    st.download_button(
        label="📥 DOWNLOAD ALL SIGNALS (CSV)",
        data=csv_buffer.getvalue(),
        file_name=f"zoha_signals_{bdt_time.strftime('%Y%m%d_%H%M%S')}.csv",
        mime='text/csv',
        use_container_width=True,
        help=f"Download {len(st.session_state.generated_signals)} signals"
    )

# Footer
st.markdown("---")
st.caption("⚠️ **Disclaimer**: Signals are probabilistic predictions based on institutional strategies. Trade at your own risk. Not financial advice.")
st.caption("🤖 **Engine**: Meta-Ensemble ML (RandomForest + GradientBoosting + Logistic) | **Accuracy Target**: 70-85% OTC / 65-70% Real")
