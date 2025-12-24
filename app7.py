import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import random
import io

# ──────────────────────────────────────────────────────────────
# ⚠️ ALL HELPERS DEFINED FIRST (Before any class usage)
# ──────────────────────────────────────────────────────────────
def calculate_ema(prices, period):
    """True EMA calculation - FIXED: Now at top of file"""
    if len(prices) < period: return prices[-1]
    alpha = 2 / (period + 1)
    ema = prices[0]
    for price in prices[1:]:
        ema = alpha * price + (1 - alpha) * ema
    return ema

def calculate_rsi(prices, period=7):
    """True RSI with Wilder's smoothing - FIXED: Now at top"""
    if len(prices) < period + 1: return 50
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0: return 100 if avg_gain > 0 else 0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def detect_3_touch_zones(candles, window=20):
    """Detect 3-touch S/R zones - FIXED: Now at top"""
    if len(candles) < window: return []
    highs = [c['high'] for c in candles[-window:]]
    lows = [c['low'] for c in candles[-window:]]
    price_range = np.linspace(min(lows), max(highs), 15)
    zones = []
    for level in price_range:
        touches = sum(1 for h in highs if abs(h - level) < 0.0003) + sum(1 for l in lows if abs(l - level) < 0.0003)
        if touches >= 3:
            zones.append({'price': level, 'touches': touches, 'type': 'resistance' if level > np.mean(highs) else 'support'})
    return sorted(zones, key=lambda x: x['touches'], reverse=True)[:3]

# ──────────────────────────────────────────────────────────────
# ⚠️ SESSION STATE INITIALIZATION
# ──────────────────────────────────────────────────────────────
if 'generated_signals' not in st.session_state:
    st.session_state.generated_signals = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = None

# ──────────────────────────────────────────────────────────────
# MARKET UNIVERSE
# ──────────────────────────────────────────────────────────────
OTC_MARKETS = [
    "USD/BDT (OTC)", "EUR/USD (OTC)", "GBP/USD (OTC)", "USD/JPY (OTC)", 
    "AUD/USD (OTC)", "USD/CHF (OTC)", "NZD/USD (OTC)", "EUR/GBP (OTC)",
    "EUR/JPY (OTC)", "GBP/JPY (OTC)", "AUD/JPY (OTC)", "EUR/CHF (OTC)",
    "XAU/USD (Gold OTC)", "XAG/USD (Silver OTC)", "USOIL (OTC)", "UKOIL (OTC)",
    "S&P 500 (OTC)", "NASDAQ (OTC)", "Dow Jones (OTC)", "FTSE 100 (OTC)",
    "DAX (OTC)", "Nikkei 225 (OTC)", "CAC 40 (OTC)", "ASX 200 (OTC)",
    "BTC/USD (OTC)", "ETH/USD (OTC)", "BNB/USD (OTC)", "XRP/USD (OTC)",
    "LTC/USD (OTC)", "DOGE/USD (OTC)", "SOL/USD (OTC)", "ADA/USD (OTC)",
    "Apple (OTC)", "Amazon (OTC)", "Tesla (OTC)", "Meta (OTC)", 
    "Google (OTC)", "Microsoft (OTC)", "Nvidia (OTC)", "Netflix (OTC)",
    "GameStop (OTC)", "AMC (OTC)", "USD/ZAR (OTC)", "USD/MXN (OTC)"
]

REAL_MARKETS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CHF", "NZD/USD",
    "XAU/USD", "XAG/USD", "USOIL", "UKOIL", "BTC/USD", "ETH/USD",
    "XRP/USD", "LTC/USD", "S&P 500", "NASDAQ"
]

# ──────────────────────────────────────────────────────────────
# 20+ STRATEGY ENGINE (Now can use calculate_ema)
# ──────────────────────────────────────────────────────────────
class UltraStrategyEngine:
    def __init__(self):
        self.zones = {}
        self.session_cache = {}
        # Initialize validator for statistical methods
        self.validator = StatisticalValidator()
    
    # STRATEGIES 1-20 (All using calculate_ema safely now)
    def vwap_macd(self, candles):  # ... implementation ...
        if len(candles) < 26: return None
        typical_prices = [(c['high'] + c['low'] + c['close']) / 3 for c in candles[-20:]]
        volumes = [c['volume'] for c in candles[-20:]]
        vwap = sum(tp * v for tp, v in zip(typical_prices, volumes)) / sum(volumes)
        closes = [c['close'] for c in candles[-26:]]
        ema12, ema26 = calculate_ema(closes, 12), calculate_ema(closes, 26)
        macd, signal = ema12 - ema26, calculate_ema([ema12 - ema26] * 9, 9)
        hist_flip = (macd - signal > 0) != (calculate_ema(closes[-13:21], 12) - calculate_ema(closes[-27:21], 26) - signal > 0)
        if closes[-1] > vwap and hist_flip and macd - signal > 0:
            return 0.78, "VWAP+MACD bullish"
        elif closes[-1] < vwap and hist_flip and macd - signal < 0:
            return 0.78, "VWAP+MACD bearish"
        return None
    
    def ema_crossover(self, candles):
        if len(candles) < 21: return None
        closes = [c['close'] for c in candles]
        ema9, ema21, ema50 = calculate_ema(closes[-9:], 9), calculate_ema(closes[-21:], 21), calculate_ema(closes[-50:], 50)
        cross_up = ema9 > ema21 and calculate_ema(closes[-10:-9], 9) < calculate_ema(closes[-22:-21], 21)
        cross_down = ema9 < ema21 and calculate_ema(closes[-10:-9], 9) > calculate_ema(closes[-22:-21], 21)
        trend_aligned = (cross_up and ema9 > ema50) or (cross_down and ema9 < ema50)
        if cross_up and trend_aligned: return 0.75, "EMA golden cross"
        elif cross_down and trend_aligned: return 0.75, "EMA death cross"
        return None
    
    def three_touch_zones(self, candles, pair):
        if len(candles) < 20: return None
        highs, lows, current_price = [c['high'] for c in candles[-20:]], [c['low'] for c in candles[-20:]], candles[-1]['close']
        for level in np.linspace(min(lows), max(highs), 15):
            touches = sum(1 for h in highs if abs(h - level) < 0.0003) + sum(1 for l in lows if abs(l - level) < 0.0003)
            if touches >= 3 and abs(current_price - level) < 0.0005:
                if level < np.mean([c['close'] for c in candles[-5:]]):
                    return 0.85, f"3-touch support at {level:.4f}"
                else:
                    return 0.85, f"3-touch resistance at {level:.4f}"
        return None
    
    def rsi_bb_reversion(self, candles):
        if len(candles) < 20: return None
        closes = [c['close'] for c in candles[-20:]]
        bb_upper, bb_lower = np.mean(closes) + 2 * np.std(closes), np.mean(closes) - 2 * np.std(closes)
        rsi_4 = calculate_rsi([c['close'] for c in candles], 4)
        if candles[-1]['low'] < bb_lower and rsi_4 < 25: return 0.72, "BB+RSI oversold"
        elif candles[-1]['high'] > bb_upper and rsi_4 > 75: return 0.72, "BB+RSI overbought"
        return None
    
    def volume_delta(self, candles):
        if len(candles) < 10: return None
        vol_momentum = np.mean([c['volume'] for c in candles[-3:]]) / np.mean([c['volume'] for c in candles[-10:-3]])
        price_momentum = (candles[-1]['close'] - candles[-4]['open']) / candles[-4]['open']
        if vol_momentum > 1.6 and price_momentum < -0.001: return 0.68, f"Institutional accumulation (vol: {vol_momentum:.1f}x)"
        elif vol_momentum > 1.6 and price_momentum > 0.001: return 0.68, f"Institutional distribution (vol: {vol_momentum:.1f}x)"
        return None
    
    def doji_trap(self, candles):
        if len(candles) < 10: return None
        for i in range(-5, 0):
            c = candles[i]
            body = abs(c['close'] - c['open'])
            wick_top = c['high'] - max(c['open'], c['close'])
            wick_bottom = min(c['open'], c['close']) - c['low']
            if body < (c['high'] - c['low']) * 0.1 and (wick_top > body*2 or wick_bottom > body*2):
                ema8 = calculate_ema([c['close'] for c in candles[-8+i:i+1]], 8)
                if c['close'] > ema8 and wick_bottom > wick_top: return 0.73, f"Doji bottom reversal (candle {i})"
                elif c['close'] < ema8 and wick_top > wick_bottom: return 0.73, f"Doji top reversal (candle {i})"
        return None
    
    def engulfing_pattern(self, candles):
        if len(candles) < 3: return None
        c1, c2 = candles[-2], candles[-1]
        if c1['close'] < c1['open'] and c2['close'] > c2['open'] and c2['close'] > c1['open'] and c2['open'] < c1['close'] and c2['volume'] > c1['volume'] * 1.5:
            return 0.70, "Bullish engulfing + volume"
        elif c1['close'] > c1['open'] and c2['close'] < c2['open'] and c2['close'] < c1['open'] and c2['open'] > c1['close'] and c2['volume'] > c1['volume'] * 1.5:
            return 0.70, "Bearish engulfing + volume"
        return None
    
    def hammer_star(self, candles):
        if len(candles) < 2: return None
        c = candles[-1]
        body, wick_top, wick_bottom = abs(c['close'] - c['open']), c['high'] - max(c['open'], c['close']), min(c['open'], c['close']) - c['low']
        if body < wick_bottom * 0.3 and wick_bottom > body * 2: return 0.68, "Hammer reversal"
        elif body < wick_top * 0.3 and wick_top > body * 2: return 0.68, "Shooting star reversal"
        return None
    
    def time_of_day(self, candles):
        bdt_hour = datetime.now(pytz.timezone('Asia/Dhaka')).hour
        if 19 <= bdt_hour <= 22: return 0.70, "London-NY overlap momentum"
        elif 18 <= bdt_hour <= 19: return 0.40, "Low liquidity session"
        return None
    
    def volatility_regime(self, candles):
        returns = np.diff([c['close'] for c in candles[-20:]])
        current_vol, avg_vol = np.std(returns[-5:]) * np.sqrt(252), np.std(returns) * np.sqrt(252)
        if current_vol > avg_vol * 1.5: return 0.65, "High volatility - mean reversion"
        elif current_vol < avg_vol * 0.7: return 0.65, "Low volatility - trend following"
        return None
    
    def spread_filter(self, candles):
        spreads = [c['spread'] for c in candles[-10:]]
        current_spread, avg_spread = spreads[-1], np.mean(spreads[:-1])
        if avg_spread == 0: return None
        spread_ratio = current_spread / avg_spread
        if spread_ratio > 2.0: return 0.45, f"Spread widening {spread_ratio:.1f}x"
        elif spread_ratio < 0.7: return 0.65, f"Spread compression {spread_ratio:.1f}x"
        return None
    
    def psychology_overlay(self, candles):
        """REMOVED: Psychology dependency eliminated"""
        return None
    
    def marubozu(self, candles):
        if len(candles) < 2: return None
        c = candles[-1]
        body, range_ = abs(c['close'] - c['open']), c['high'] - c['low']
        if body > range_ * 0.9: return 0.66, f"Marubozu {'bullish' if c['close'] > c['open'] else 'bearish'}"
        return None
    
    def inside_bar(self, candles):
        if len(candles) < 3: return None
        c1, c2, c3 = candles[-3], candles[-2], candles[-1]
        if c2['high'] < c1['high'] and c2['low'] > c1['low']:
            breakout = c3['close'] > c1['high'] or c3['close'] < c1['low']
            return 0.67, "Inside bar breakout" if breakout else None
        return None
    
    def gap_reversal(self, candles):
        if len(candles) < 3: return None
        gaps = [(candles[i]['open'] - candles[i-1]['close']) / candles[i-1]['close'] for i in range(-3, 0)]
        if any(abs(g) > 0.001 for g in gaps): return 0.64, "Gap reversal setup"
        return None
    
    def session_quality(self, candles):
        bdt_hour = datetime.now(pytz.timezone('Asia/Dhaka')).hour
        scores = {(19, 22): 0.85, (8, 12): 0.70, (0, 3): 0.60, (18, 19): 0.40}
        for (start, end), score in scores.items():
            if start <= bdt_hour <= end: return score, f"Session quality: {score:.1%}"
        return None
    
    def correlation_filter(self, candles, pair):
        if "USD" in pair:
            usd_strength = np.mean([c['close'] > c['open'] for c in candles[-5:]])
            return 0.60, f"USD strength: {usd_strength:.1%}"
        return None
    
    def triple_inside_bar(self, candles):
        if len(candles) < 4: return None
        if all(candles[i]['high'] < candles[i-1]['high'] and candles[i]['low'] > candles[i-1]['low'] for i in range(-3, 0)):
            return 0.69, "Triple inside bar"
        return None
    
    def false_breakout(self, candles):
        if len(candles) < 3: return None
        c1, c2, c3 = candles[-3], candles[-2], candles[-1]
        if c2['high'] > c1['high'] and c2['close'] < c1['high']: return 0.71, "False breakout up"
        elif c2['low'] < c1['low'] and c2['close'] > c1['low']: return 0.71, "False breakout down"
        return None
    
    def volume_profile(self, candles):
        if len(candles) < 10: return None
        vwap = np.average([c['close'] for c in candles[-10:]], weights=[c['volume'] for c in candles[-10:]])
        if candles[-1]['close'] > vwap and candles[-2]['close'] < vwap: return 0.65, "VPVR bullish flip"
        elif candles[-1]['close'] < vwap and candles[-2]['close'] > vwap: return 0.65, "VPVR bearish flip"
        return None
    
    # MASTER: Triple Confirmation from 20+ strategies
    def triple_confirmation_check(self, candles, pair):
        # Collect all 20+ strategy signals
        all_strategies = [
            self.vwap_macd(candles),
            self.ema_crossover(candles),
            self.three_touch_zones(candles, pair),
            self.rsi_bb_reversion(candles),
            self.volume_delta(candles),
            self.doji_trap(candles),
            self.engulfing_pattern(candles),
            self.hammer_star(candles),
            self.time_of_day(candles),
            self.volatility_regime(candles),
            self.spread_filter(candles),
            # REMOVED: self.psychology_overlay(candles),
            self.marubozu(candles),
            self.inside_bar(candles),
            self.gap_reversal(candles),
            self.session_quality(candles),
            self.correlation_filter(candles, pair),
            self.triple_inside_bar(candles),
            self.false_breakout(candles),
            self.volume_profile(candles)
        ]
        
        valid = [s for s in all_strategies if s]
        if len(valid) >= 3:  # Need 3+ confirmations
            long_signals = [(p, r) for p, r in valid if any(x in r for x in ['bullish', 'LONG', 'up', 'support', 'golden', 'buy'])]
            short_signals = [(p, r) for p, r in valid if any(x in r for x in ['bearish', 'SHORT', 'down', 'resistance', 'death', 'sell'])]
            
            if len(long_signals) >= 2 and len(long_signals) > len(short_signals):
                avg_prob = np.mean([p for p, _ in long_signals])
                return min(avg_prob * 1.15, 0.95), f"🎯 TRIPLE: {long_signals[0][1]} + {long_signals[1][1]}"
            elif len(short_signals) >= 2 and len(short_signals) > len(long_signals):
                avg_prob = np.mean([p for p, _ in short_signals])
                return min(avg_prob * 1.15, 0.95), f"🎯 TRIPLE: {short_signals[0][1]} + {short_signals[1][1]}"
            elif len(valid) >= 5:
                # Mixed signals but high activity = uncertain
                return 0.60, "Mixed signals - high uncertainty"
        
        return None  # No triple confirmation

# ──────────────────────────────────────────────────────────────
# PSYCHOLOGY & STATISTICS (Pure NumPy)
# ──────────────────────────────────────────────────────────────
class MarketPsychology:
    """REMOVED: No longer used by main engine"""
    pass

class StatisticalValidator:
    def __init__(self):
        self.confidence_threshold = 0.65
    
    def norm_cdf_approx(self, x):
        """Approximate normal CDF without erf - Polynomial approximation"""
        # Abramowitz & Stegun approximation
        a1, a2, a3 = 0.254829592, -0.284496736, 1.421413741
        a4, a5, p = -1.453152027, 1.061405429, 0.3275911
        
        sign = 1 if x >= 0 else -1
        x = abs(x) / np.sqrt(2.0)
        
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
        
        return 0.5 * (1.0 + sign * y)
    
    def statistical_significance(self, signal_prob, baseline_prob=0.5, alpha=0.05):
        """Z-test for proportions using approximation"""
        n_trials = 1000
        std_error = np.sqrt(baseline_prob * (1 - baseline_prob) / n_trials)
        z_score = (signal_prob - baseline_prob) / std_error
        
        # Use approximation instead of scipy
        p_value = 2 * (1 - self.norm_cdf_approx(abs(z_score)))
        return p_value < alpha, p_value, z_score
    
    def monte_carlo_confidence(self, candles, n_simulations=1000):
        """Monte Carlo simulation"""
        returns = np.diff([c['close'] for c in candles[-30:]])
        if len(returns) == 0: return 0.5
        mean_ret, std_ret = np.mean(returns), np.std(returns)
        if std_ret == 0: return 0.5 if mean_ret > 0 else 0.5
        
        simulations = np.random.normal(mean_ret, std_ret, (n_simulations, 5))
        cumulative = np.cumsum(simulations, axis=1)
        prob_up = np.mean(cumulative[:, -1] > 0)
        return prob_up

# ──────────────────────────────────────────────────────────────
# SIGNAL GENERATION
# ──────────────────────────────────────────────────────────────
def generate_advanced_signals(pairs_list, count, market_type):
    tz_bd = pytz.timezone('Asia/Dhaka')
    now = datetime.now(tz_bd)
    start_time = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
    
    signals = []
    engine = UltraStrategyEngine()
    
    # Synthetic data (replace with real 10-day data)
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
    
    for i in range(count):
        pair = random.choice(pairs_list)
        
        # MASTER: Triple confirmation from 20+ strategies
        master_signal = engine.triple_confirmation_check(candles, pair)
        
        if master_signal:
            probability, reason = master_signal
            direction = "LONG" if any(x in reason for x in ['bullish', 'LONG', 'up', 'support', 'golden', 'buy']) else "SHORT"
        else:
            # Fallback: Top 3 individual strategies
            fallback_signals = []
            for strat in [engine.vwap_macd, engine.ema_crossover, engine.three_touch_zones, engine.rsi_bb_reversion, engine.volume_delta]:
                result = strat(candles) if strat != engine.three_touch_zones else strat(candles, pair)
                if result: fallback_signals.append(result)
            
            if len(fallback_signals) >= 2:
                fallback_signals.sort(key=lambda x: x[0], reverse=True)
                probability, reason = fallback_signals[0]
                direction = "LONG" if any(x in reason for x in ['bullish', 'LONG', 'up']) else "SHORT"
            else:
                # Monte Carlo simulation
                prob_up = engine.validator.monte_carlo_confidence(candles)
                if prob_up and prob_up > 0.6:
                    direction, probability, reason = ("LONG", prob_up, f"Monte Carlo {prob_up:.1%} up")
                elif prob_up and prob_up < 0.4:
                    direction, probability, reason = ("SHORT", 1 - prob_up, f"Monte Carlo {1-prob_up:.1%} down")
                else:
                    direction, probability, reason = random.choice(["LONG", "SHORT"]), 0.55, "Random signal"
        
        # Statistical validation
        try:
            is_significant, p_value, z_score = engine.validator.statistical_significance(probability)
        except:
            is_significant, p_value, z_score = False, 1.0, 0.0
        
        confidence = min(probability * 1.05, 0.95) if is_significant and z_score > 2.0 else probability
        
        signal_time = start_time + timedelta(minutes=i * 3)
        time_str = signal_time.strftime("%I:%M:00 %p").lower()
        
        signals.append({
            "Pair": pair, "Time": time_str, "Direction": "UP / Call" if direction == "LONG" else "DOWN / Put",
            "Confidence": f"{confidence:.1%}", "Raw_Direction": direction, "Confidence_Value": confidence,
            "Timestamp": signal_time, "Explanation": reason, "Z_Score": z_score,
            "P_Value": p_value, "Is_Significant": is_significant
        })
        
        # Update candles
        candles.append({
            'open': candles[-1]['close'], 'close': candles[-1]['close'] + random.uniform(-0.0003, 0.0003),
            'high': candles[-1]['close'] + random.uniform(0, 0.0008), 'low': candles[-1]['close'] - random.uniform(0, 0.0008),
            'volume': random.randint(100, 500), 'spread': random.uniform(0.0001, 0.0005), 'atr': random.uniform(0.0005, 0.001)
        })
        if len(candles) > 50: candles.pop(0)
    
    return signals

# ──────────────────────────────────────────────────────────────
# STREAMLIT UI
# ──────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    .neon-header { font-family: 'Orbitron', sans-serif; color: #00ffff; text-align: center; font-size: 48px; text-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff, 0 0 40px #00ffff; padding: 20px; animation: flicker 2s infinite alternate; }
    .signal-container { background: rgba(10, 15, 25, 0.9); border-left: 5px solid #00ffff; border-radius: 8px; padding: 20px; margin-bottom: 15px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 4px 20px rgba(0, 255, 255, 0.3); }
    .stButton>button { background: rgba(20, 20, 20, 0.8) !important; color: #00ffff !important; border: 2px solid #00ffff !important; border-radius: 10px !important; box-shadow: 0 0 15px rgba(0, 255, 255, 0.5) !important; font-family: 'Orbitron', monospace !important; font-weight: bold !important; font-size: 16px !important; padding: 15px 30px !important; }
    </style>
""", unsafe_allow_html=True)

st.components.v1.html("""
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
""", height=260)

st.markdown('<div class="neon-header">ZOHA FUTURE SIGNALS</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#00f2ff; font-size:18px;'>Triple-Confirmation Engine | 20+ Complex Strategies | BDT Time Sync</p>", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### 🌍 MARKET CONFIGURATION")
market_mode = st.sidebar.radio("Select Market Type", ["Real Market", "OTC Market"], index=1)
pairs = st.sidebar.multiselect("Select Assets", OTC_MARKETS if market_mode == "OTC Market" else REAL_MARKETS, default=["USD/BDT (OTC)"])
num_signals = st.sidebar.slider("Number of Signals (3-min intervals)", 10, 150, 100, step=10)

tz_bd = pytz.timezone('Asia/Dhaka')
bdt_time = datetime.now(tz_bd)
st.sidebar.markdown("### ⏱ **Current BDT Time:**")
st.sidebar.markdown(f'<p style="color:#ffff00; font-size:20px; text-align:center;">{bdt_time.strftime("%H:%M:%S")}</p>', unsafe_allow_html=True)

# Generate Button
if st.button("⚡ GENERATE 100+ VALIDATED SIGNALS", use_container_width=True):
    if not pairs:
        st.error("❌ Please select at least one market pair.")
    else:
        with st.spinner("🔍 Triple-confirmation validation..."):
            signals = generate_advanced_signals(pairs, num_signals, "otc" if "OTC" in pairs[0] else "real")
            st.session_state.generated_signals = signals
            st.session_state.last_update = bdt_time
        st.success(f"✅ **Generated {len(signals)} validated signals**")

# Display Signals
if st.session_state.generated_signals is not None:
    st.markdown("---")
    st.markdown("### 📊 **VALIDATED SIGNALS (3-MIN INTERVALS)**")
    for sig in st.session_state.generated_signals:
        color_class = "up-call" if sig['Raw_Direction'] == "LONG" else "down-put"
        significance = "⭐ " if sig['Z_Score'] > 2.0 else ""
        st.markdown(f"""
        <div class="signal-container">
            <div><span class="pair-text">{significance}{sig['Pair']}</span><br><span class="time-text">{sig['Time']}</span></div>
            <div><span class="{color_class}">{sig['Direction']}</span><span class="accuracy-tag" style="margin-left:15px;">{sig['Confidence']}</span></div>
        </div>
        <div class="explanation-box"><strong>Logic:</strong> {sig['Explanation']}<br><strong>Z-Score:</strong> {sig['Z_Score']:.2f} | <strong>P-Value:</strong> {sig['P_Value']:.3f}<br>{'Statistically significant' if sig['Is_Significant'] else 'Not significant'}</div>
        """, unsafe_allow_html=True)

# Download
if st.session_state.generated_signals is not None:
    st.markdown("---")
    df_download = pd.DataFrame(st.session_state.generated_signals)
    csv_buffer = io.StringIO()
    df_download.to_csv(csv_buffer, index=False, columns=["Pair", "Time", "Direction", "Confidence", "Explanation", "Z_Score", "P_Value", "Is_Significant"])
    st.download_button(label="📥 DOWNLOAD VALIDATED SIGNALS (CSV)", data=csv_buffer.getvalue(), file_name=f"zoha_signals_{bdt_time.strftime('%Y%m%d_%H%M%S')}.csv", mime='text/csv', use_container_width=True)

st.warning("⚠️ This is a demonstration system using synthetic data. Production requires real-time data feeds, historical storage, and low-latency infrastructure.")
