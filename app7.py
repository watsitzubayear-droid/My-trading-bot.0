import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import random
import io

# ──────────────────────────────────────────────────────────────
# MARKET CONFIGURATION
# ──────────────────────────────────────────────────────────────
# MAJOR OTC MARKETS EXPANSION (40+ instruments)
ADDITIONAL_OTC = [
    # Major Forex Crosses
    "EUR/JPY (OTC)", "GBP/JPY (OTC)", "AUD/JPY (OTC)", "NZD/JPY (OTC)",
    "EUR/CHF (OTC)", "GBP/CHF (OTC)", "EUR/AUD (OTC)", "EUR/CAD (OTC)",
    "USD/CAD (OTC)", "AUD/CAD (OTC)",
    
    # Commodities
    "XAG/USD (Silver OTC)", "USOIL (Oil OTC)", "UKOIL (Brent OTC)",
    
    # Global Indices
    "S&P 500 (OTC)", "NASDAQ (OTC)", "Dow Jones (OTC)", "FTSE 100 (OTC)",
    "DAX (OTC)", "Nikkei 225 (OTC)", "CAC 40 (OTC)", "ASX 200 (OTC)",
    
    # Crypto Expansion
    "XRP/USD (OTC)", "LTC/USD (OTC)", "DOGE/USD (OTC)", "SOL/USD (OTC)",
    "ADA/USD (OTC)", "MATIC/USD (OTC)", "DOT/USD (OTC)", "AVAX/USD (OTC)",
    
    # Tech Stocks
    "Meta (OTC)", "Google (OTC)", "Microsoft (OTC)", "Netflix (OTC)",
    "Nvidia (OTC)", "AMD (OTC)", "Intel (OTC)", "GameStop (OTC)",
    
    # Meme Stocks
    "AMC (OTC)", "Blackberry (OTC)", "Nokia (OTC)", "Virgin Galactic (OTC)",
    
    # Forex Exotics
    "USD/ZAR (OTC)", "USD/MXN (OTC)", "USD/TRY (OTC)", "USD/THB (OTC)"
]

OTC_MARKETS = [
    "USD/BDT (OTC)", "EUR/USD (OTC)", "GBP/USD (OTC)", "USD/JPY (OTC)", 
    "AUD/USD (OTC)", "USD/CHF (OTC)", "NZD/USD (OTC)", "EUR/GBP (OTC)", 
    "XAU/USD (Gold OTC)", "Apple (OTC)", "Amazon (OTC)", "Tesla (OTC)",
    "BTC/USD (OTC)", "ETH/USD (OTC)", "BNB/USD (OTC)"
] + ADDITIONAL_OTC

REAL_MARKETS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "XAU/USD", "BTC/USD",
    "USD/CHF", "NZD/USD", "EUR/GBP", "USD/CAD", "GBP/JPY", "EUR/JPY",
    "XAG/USD", "USOIL", "UKOIL", "S&P 500", "NASDAQ", "Dow Jones",
    "XRP/USD", "LTC/USD", "ETH/USD"
]

# ──────────────────────────────────────────────────────────────
# TRANSPARENT STRATEGY ENGINE
# ──────────────────────────────────────────────────────────────
class TransparentStrategyEngine:
    def __init__(self):
        self.strategy_scores = {
            'vwap_macd': 0,
            'ema_rsi': 0,
            'zone_rejection': 0,
            'rsi_bb': 0,
            'volume_delta': 0
        }
    
    def explain_signal(self, signal_data):
        """Returns human-readable explanation of why signal was generated"""
        explanation = []
        
        if signal_data['vwap_cross']:
            explanation.append(f"VWAP crossover detected (prob: {signal_data['vwap_prob']:.1%})")
        if signal_data['rsi_exhaustion']:
            explanation.append(f"RSI at extreme ({signal_data['rsi_value']:.1f})")
        if signal_data['zone_quality'] > 0:
            explanation.append(f"{signal_data['zone_quality']:.0f}-touch S/R zone")
        if signal_data['volume_spike']:
            explanation.append(f"Volume spike {signal_data['volume_ratio']:.1f}x")
        
        return " + ".join(explanation) if explanation else "Momentum-based signal"

    def generate_signal(self, candles, pair, market_type):
        """
        Full transparency: This is a DEMONSTRATION implementation.
        It uses simplified math, not the full institutional-grade formulas.
        It generates PLAUSIBLE signals, not predictions based on real historical analysis.
        """
        
        if len(candles) < 20:
            return {
                'direction': random.choice(['LONG', 'SHORT']),
                'confidence': 0.55,
                'explanation': 'Insufficient data - random signal',
                'pair': pair,
                'warning': '⚠️ INSUFFICIENT CANDLE DATA - SIGNAL IS RANDOM'
            }
        
        # Simplified calculations (not full institutional math)
        closes = [c['close'] for c in candles[-20:]]
        highs = [c['high'] for c in candles[-20:]]
        lows = [c['low'] for c in candles[-20:]]
        volumes = [c['volume'] for c in candles[-20:]]
        
        # 1. VWAP Signal (simplified - no real VWAP calculation)
        vwap = np.mean(closes)
        vwap_cross = (closes[-1] > vwap and closes[-2] < vwap) or (closes[-1] < vwap and closes[-2] > vwap)
        vwap_prob = 0.72 if vwap_cross else 0.0
        
        # 2. EMA+RSI Signal (simplified - no real crossover detection)
        ema9 = calculate_ema(closes[-9:], 9)
        ema21 = calculate_ema(closes[-21:], 21)
        rsi = calculate_rsi([c['close'] for c in candles], 7)
        ema_rsi_signal = (ema9 > ema21 and rsi < 35) or (ema9 < ema21 and rsi > 65)
        ema_rsi_prob = 0.68 if ema_rsi_signal else 0.0
        
        # 3. 3-Touch Zone Signal (simplified - no real zone tracking)
        zones = detect_3_touch_zones(candles)
        zone_quality = sum(z['touches'] for z in zones) / 3 if zones else 0
        zone_signal = zone_quality > 0
        zone_prob = 0.85 if zone_quality >= 3 else (0.75 if zone_quality >= 2 else 0.0)
        
        # 4. RSI+BB Signal (simplified - no real Bollinger Bands)
        bb_upper = np.mean(closes) + 2 * np.std(closes)
        bb_lower = np.mean(closes) - 2 * np.std(closes)
        rsi_fast = calculate_rsi([c['close'] for c in candles], 4)
        bb_signal = (lows[-1] < bb_lower and rsi_fast < 25) or (highs[-1] > bb_upper and rsi_fast > 75)
        bb_prob = 0.70 if bb_signal else 0.0
        
        # 5. Volume Delta (simplified - no real tick delta)
        volume_ratio = volumes[-1] / np.mean(volumes[:-1]) if np.mean(volumes[:-1]) > 0 else 1
        volume_spike = volume_ratio > 1.5
        volume_prob = 0.65 if volume_spike else 0.0
        
        # Combine signals using weighted voting (simplified meta-learner)
        signals = [
            ('vwap_macd', vwap_prob),
            ('ema_rsi', ema_rsi_prob),
            ('zone_rejection', zone_prob),
            ('rsi_bb', bb_prob),
            ('volume_delta', volume_prob)
        ]
        
        active_signals = [s for s in signals if s[1] > 0]
        
        if len(active_signals) >= 3:
            # High confidence if multiple strategies agree
            direction = 'LONG' if sum(s[1] for s in active_signals if 'LONG' in s[0]) > sum(s[1] for s in active_signals if 'SHORT' in s[0]) else 'SHORT'
            confidence = min(np.mean([s[1] for s in active_signals]) * 1.1, 0.90)
            explanation = f"{len(active_signals)} strategies confluent"
        elif len(active_signals) == 2:
            direction = 'LONG' if active_signals[0][0] == 'vwap_macd' else 'SHORT'
            confidence = 0.70
            explanation = "2-strategy agreement"
        elif len(active_signals) == 1:
            direction = 'LONG' if active_signals[0][1] > 0 else 'SHORT'
            confidence = active_signals[0][1]
            explanation = f"Single strategy: {active_signals[0][0].replace('_', '+')}"
        else:
            # No signals - use momentum
            direction = 'LONG' if closes[-1] > closes[-2] else 'SHORT'
            confidence = 0.55
            explanation = "Momentum fallback (no pattern detected)"
        
        return {
            'direction': direction,
            'confidence': confidence,
            'explanation': explanation,
            'pair': pair,
            'vwap_cross': vwap_cross,
            'rsi_value': rsi,
            'zone_quality': zone_quality,
            'volume_ratio': volume_ratio,
            'warning': None if len(active_signals) >= 2 else '⚠️ LOW CONFLUENCE - REDUCED RELIABILITY'
        }

# ──────────────────────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────────────────────
# CSS (unchanged from previous)
st.markdown("""
    <style>
    .neon-header { font-family: 'Orbitron', sans-serif; color: #00ffff; text-align: center; font-size: 48px; text-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff, 0 0 40px #00ffff; padding: 20px; animation: flicker 2s infinite alternate; }
    @keyframes flicker { 0%, 100% { opacity: 1; } 50% { opacity: 0.9; } }
    .signal-container { background: rgba(10, 15, 25, 0.9); border-left: 5px solid #00ffff; border-radius: 8px; padding: 20px; margin-bottom: 15px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 4px 20px rgba(0, 255, 255, 0.3); transition: all 0.3s; }
    .signal-container:hover { box-shadow: 0 6px 30px rgba(0, 255, 255, 0.5); transform: translateX(5px); }
    .time-text { font-family: 'Roboto Mono', monospace; color: #ffffff; font-size: 22px; font-weight: bold; text-shadow: 0 0 5px #fff; }
    .pair-text { color: #888; font-size: 14px; font-family: 'Orbitron', sans-serif; }
    .up-call { color: #00ff88; font-weight: bold; text-shadow: 0 0 10px #00ff88, 0 0 20px #00ff88; border: 1px solid #00ff88; padding: 8px 20px; border-radius: 25px; text-transform: uppercase; background: rgba(0, 255, 136, 0.1); }
    .down-put { color: #ff0055; font-weight: bold; text-shadow: 0 0 10px #ff0055, 0 0 20px #ff0055; border: 1px solid #ff0055; padding: 8px 20px; border-radius: 25px; text-transform: uppercase; background: rgba(255, 0, 85, 0.1); }
    .accuracy-tag { background: #1a1f2b; color: #00f2ff; padding: 4px 12px; border-radius: 6px; font-size: 13px; font-weight: bold; border: 1px solid #00f2ff; }
    .explanation-box { background: rgba(30, 30, 40, 0.8); border: 1px solid #444; border-radius: 6px; padding: 10px; margin-top: 10px; font-size: 12px; color: #aaa; }
    </style>
""", unsafe_allow_html=True)

# 3D Logo (unchanged)
logo_html = """
<!DOCTYPE html><html><head><style>body{margin:0;background:transparent;overflow:hidden}</style>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script></head><body>
<div id="logo-container" style="width:100%;height:250px;"></div><script>
const container=document.getElementById('logo-container');const scene=new THREE.Scene();const camera=new THREE.PerspectiveCamera(75,container.clientWidth/250,0.1,1000);
const renderer=new THREE.WebGLRenderer({alpha:true,antialias:true});renderer.setSize(container.clientWidth,250);container.appendChild(renderer.domElement);
const shape=new THREE.Shape();shape.moveTo(-4,-3);shape.lineTo(4,-3);shape.lineTo(-4,3);shape.lineTo(4,3);
const geometry=new THREE.ExtrudeGeometry(shape,{depth:0.6,bevelEnabled:true,bevelThickness:0.25,bevelSize:0.15});
const material=new THREE.MeshPhongMaterial({color:0x00ffff,emissive:0x00aaaa,emissiveIntensity:2.5});
const logo=new THREE.Mesh(geometry,material);scene.add(logo);scene.add(new THREE.Mesh(geometry.clone().scale(1.12,1.12,1.12),new THREE.MeshBasicMaterial({color:0x00ffff,transparent:true,opacity:0.4})));
scene.add(new THREE.AmbientLight(0x404040));scene.add(new THREE.PointLight(0x00ffff,3,100,0).position.set(10,10,10));
camera.position.z=12;function animate(){requestAnimationFrame(animate);logo.rotation.y+=0.008;renderer.render(scene,camera)}animate();
</script></body></html>
"""
st.components.v1.html(logo_html, height=260)

# Sidebar
st.sidebar.markdown("### 🌍 MARKET CONFIGURATION")
market_mode = st.sidebar.radio("Select Market Type", ["Real Market", "OTC Market"], index=1)

if market_mode == "OTC Market":
    pairs = st.sidebar.multiselect("Select OTC Assets", OTC_MARKETS, default=["USD/BDT (OTC)", "EUR/USD (OTC)"])
else:
    pairs = st.sidebar.multiselect("Select Real Assets", REAL_MARKETS, default=["EUR/USD", "GBP/USD"])

num_signals = st.sidebar.slider("Number of Signals (3-min intervals)", 10, 150, 100, step=10)

# BDT Time
tz_bd = pytz.timezone('Asia/Dhaka')
bdt_time = datetime.now(tz_bd)
st.sidebar.markdown("### ⏱ **Current BDT Time:**")
st.sidebar.markdown(f'<p style="color:#ffff00; font-size:20px; text-align:center;">{bdt_time.strftime("%H:%M:%S")}</p>', unsafe_allow_html=True)

# Generate Button
if st.button("⚡ GENERATE 100+ ADVANCED SIGNALS", use_container_width=True):
    if not pairs:
        st.error("❌ Please select at least one market pair.")
    else:
        with st.spinner("🔍 Scanning markets... Building synthetic candles... Applying 5 strategies..."):
            # Build synthetic candle history (NOT real 10-day data)
            candles = []
            base_price = 1.0850
            for i in range(50):
                base_price += random.uniform(-0.001, 0.001)
                candles.append({
                    'open': base_price, 'close': base_price + random.uniform(-0.0005, 0.0005),
                    'high': base_price + random.uniform(0, 0.001), 'low': base_price - random.uniform(0, 0.001),
                    'volume': random.randint(100, 500), 'spread': random.uniform(0.0001, 0.0005),
                    'atr': random.uniform(0.0005, 0.001)
                })
            
            market_type = "otc" if "OTC" in pairs[0] else "real"
            engine = TransparentStrategyEngine()
            
            signals = []
            for i in range(num_signals):
                pair = random.choice(pairs)
                signal = engine.generate_signal(candles, pair, market_type)
                
                # 3-minute spacing
                signal_time = bdt_time + timedelta(minutes=(i+1)*3)
                time_str = signal_time.strftime("%I:%M:00 %p").lower()
                
                signals.append({
                    "Pair": pair, 
                    "Time": time_str, 
                    "Direction": "UP / Call" if signal['direction'] == "LONG" else "DOWN / Put",
                    "Confidence": f"{signal['confidence']:.1%}",
                    "Raw_Direction": signal['direction'],
                    "Confidence_Value": signal['confidence'],
                    "Timestamp": signal_time,
                    "Explanation": signal['explanation'],
                    "Warning": signal['warning']
                })
                
                # Update candles (simulating market movement)
                candles.append({
                    'open': candles[-1]['close'],
                    'close': candles[-1]['close'] + random.uniform(-0.0003, 0.0003),
                    'high': candles[-1]['close'] + random.uniform(0, 0.0008),
                    'low': candles[-1]['close'] - random.uniform(0, 0.0008),
                    'volume': random.randint(100, 500),
                    'spread': random.uniform(0.0001, 0.0005),
                    'atr': random.uniform(0.0005, 0.001)
                })
                if len(candles) > 50: candles.pop(0)
            
            st.session_state.generated_signals = signals
            st.session_state.last_update = bdt_time
        st.success(f"✅ **Generated {len(signals)} signals** (Synthetic data, not real 10-day analysis)")

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
        <div class="explanation-box">
            <strong>Logic:</strong> {sig['Explanation']}<br>
            {sig['Warning'] if sig['Warning'] else ''}
        </div>
        """, unsafe_allow_html=True)

# Download
if st.session_state.generated_signals:
    st.markdown("---")
    df_download = pd.DataFrame(st.session_state.generated_signals)
    csv_buffer = io.StringIO()
    df_download.to_csv(csv_buffer, index=False, columns=["Pair", "Time", "Direction", "Confidence", "Explanation"])
    
    st.download_button(
        label="📥 DOWNLOAD ALL SIGNALS (CSV)",
        data=csv_buffer.getvalue(),
        file_name=f"zoha_signals_{bdt_time.strftime('%Y%m%d_%H%M%S')}.csv",
        mime='text/csv',
        use_container_width=True
    )

# Transparency Notice
st.markdown("---")
st.warning("""
**⚠️ IMPORTANT TRANSPARENCY NOTICE:**

This app is a **DEMONSTRATION TOOL** for educational purposes. It does **NOT**:

- Use real 10-day historical data from any market
- Perform actual backtesting or statistical validation
- Access live broker feeds (Quotex, Forex, etc.)
- Implement full institutional-grade indicator math
- Guarantee any win rate or profitability

**What it DOES:**
- Generates **plausible signals** using simplified rule-based logic
- Demonstrates how a multi-strategy system **could** be structured
- Provides a professional UI for visualization
- Serves as a starting template for **real development**

**For a production system, you need:**
- Real-time tick data API ($500+/month)
- Historical data storage (terabytes)
- VPS with <50ms latency ($100+/month)
- Full indicator libraries (TA-Lib, QuantLib)
- Backtesting engine (Zipline, Backtrader)
- 3-6 months of development & validation

**Trade at your own risk. This is not financial advice.**
""")

st.caption("Engine: Rule-Based Composite Scoring (5 simplified strategies) | Targets: 65-85% (demonstration only)")
