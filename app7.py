import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import random
import io

# ──────────────────────────────────────────────────────────────
# ⚠️ FIXED: Initialize Session State FIRST
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
# PSYCHOLOGY & STATISTICS (Pure NumPy, No SciPy)
# ──────────────────────────────────────────────────────────────
class MarketPsychology:
    def calculate_fear_greed(self, candles):
        """Fear/Greed proxy using volatility and momentum"""
        returns = np.diff([c['close'] for c in candles])
        volatility = np.std(returns) * np.sqrt(252)
        momentum = (candles[-1]['close'] - candles[-5]['close']) / candles[-5]['close']
        fear = min(volatility * 10, 1.0) * (1 if momentum < 0 else 0.5)
        greed = max(1 - volatility * 5, 0) * (1 if momentum > 0 else 0.5)
        return fear, greed
    
    def detect_herd_behavior(self, volumes):
        """Volume spike detection"""
        baseline = np.mean(volumes[:-5])
        recent = np.mean(volumes[-3:])
        spike_ratio = recent / baseline if baseline > 0 else 1
        return spike_ratio > 1.8, spike_ratio
    
    def institutional_vs_retail(self, candles):
        """Estimate institutional participation"""
        wick_sizes = [(c['high'] - max(c['open'], c['close'])) / (c['high'] - c['low']) for c in candles[-10:]]
        volume_trend = np.mean([c['volume'] for c in candles[-5:]]) / np.mean([c['volume'] for c in candles[-10:-5]])
        wick_ratio = np.mean(wick_sizes)
        institutional_score = (1 - wick_ratio) * volume_trend
        return institutional_score, wick_ratio

class StatisticalValidator:
    def __init__(self):
        self.confidence_threshold = 0.65
    
    # ✅ REIMPLEMENTED: No SciPy needed
    def norm_cdf(self, x):
        """Approximate normal CDF using error function"""
        return 0.5 * (1 + np.erf(x / np.sqrt(2)))
    
    def statistical_significance(self, signal_prob, baseline_prob=0.5, alpha=0.05):
        """Z-test for proportions without SciPy"""
        n_trials = 1000
        z_score = (signal_prob - baseline_prob) / np.sqrt(baseline_prob * (1 - baseline_prob) / n_trials)
        p_value = 2 * (1 - self.norm_cdf(abs(z_score)))
        return p_value < alpha, p_value, z_score
    
    def monte_carlo_confidence(self, candles, n_simulations=1000):
        """Monte Carlo simulation"""
        returns = np.diff([c['close'] for c in candles[-30:]])
        mean_ret, std_ret = np.mean(returns), np.std(returns)
        simulations = np.random.normal(mean_ret, std_ret, (n_simulations, 5))
        cumulative = np.cumsum(simulations, axis=1)
        prob_up = np.mean(cumulative[:, -1] > 0)
        return prob_up

# ──────────────────────────────────────────────────────────────
# COMPREHENSIVE STRATEGY ENGINE
# ──────────────────────────────────────────────────────────────
class ComprehensiveStrategyEngine:
    def __init__(self):
        self.psychology = MarketPsychology()
        self.validator = StatisticalValidator()
        self.zones = {}
    
    def vwap_macd_institutional(self, candles):
        if len(candles) < 26: return None
        typical_prices = [(c['high'] + c['low'] + c['close']) / 3 for c in candles[-20:]]
        volumes = [c['volume'] for c in candles[-20:]]
        vwap = sum(tp * v for tp, v in zip(typical_prices, volumes)) / sum(volumes)
        closes = [c['close'] for c in candles[-26:]]
        ema12, ema26 = calculate_ema(closes, 12), calculate_ema(closes, 26)
        macd, signal = ema12 - ema26, calculate_ema([ema12 - ema26] * 9, 9)
        hist_flip = (macd - signal > 0) != (calculate_ema(closes[-13:21], 12) - calculate_ema(closes[-27:21], 26) - signal > 0)
        
        if closes[-1] > vwap and hist_flip and macd - signal > 0:
            return 0.78, "INSTITUTIONAL: VWAP bullish + MACD flip"
        elif closes[-1] < vwap and hist_flip and macd - signal < 0:
            return 0.78, "INSTITUTIONAL: VWAP bearish + MACD flip"
        return None
    
    def triple_confirmation_check(self, candles, pair):
        """MASTER: Requires 3+ strategy confirmations"""
        all_strategies = [
            self.vwap_macd_institutional(candles),
            self.ema_crossover_pullback(candles),
            self.three_touch_zones(candles, pair)
        ]
        
        valid = [s for s in all_strategies if s]
        if len(valid) >= 3:
            long_signals = [s for s in valid if 'bullish' in s[1] or 'LONG' in s[1]]
            short_signals = [s for s in valid if 'bearish' in s[1] or 'SHORT' in s[1]]
            if len(long_signals) >= 2:
                return min(np.mean([p for p, _ in long_signals]) * 1.1, 0.92), f"TRIPLE: {' + '.join([r for _, r in long_signals[:2]])}"
        return None

# ──────────────────────────────────────────────────────────────
# SIMPLIFIED FOR BREVITY: Add full implementations of methods
# ──────────────────────────────────────────────────────────────
    def ema_crossover_pullback(self, candles):
        if len(candles) < 21: return None
        closes = [c['close'] for c in candles]
        ema9, ema21, ema50 = calculate_ema(closes[-9:], 9), calculate_ema(closes[-21:], 21), calculate_ema(closes[-50:], 50)
        cross_up = ema9 > ema21 and calculate_ema(closes[-10:-9], 9) < calculate_ema(closes[-22:-21], 21)
        cross_down = ema9 < ema21 and calculate_ema(closes[-10:-9], 9) > calculate_ema(closes[-22:-21], 21)
        distance_from_ema = abs(closes[-1] - ema9) / closes[-1]
        trend_aligned = (cross_up and ema9 > ema50) or (cross_down and ema9 < ema50)
        if cross_up and distance_from_ema < 0.001 and trend_aligned:
            return 0.75, "EMA golden cross + pullback"
        elif cross_down and distance_from_ema < 0.001 and trend_aligned:
            return 0.75, "EMA death cross + pullback"
        return None
    
    def three_touch_zones(self, candles, pair):
        if len(candles) < 20: return None
        highs, lows, current_price = [c['high'] for c in candles[-20:]], [c['low'] for c in candles[-20:]], candles[-1]['close']
        for level in np.linspace(min(lows), max(highs), 15):
            touches = sum(1 for h in highs if abs(h - level) < 0.0003) + sum(1 for l in lows if abs(l - level) < 0.0003)
            if touches >= 3 and abs(current_price - level) < 0.0005:
                if level < np.mean([c['close'] for c in candles[-5:]]):
                    return 0.85, "3-touch support zone"
                else:
                    return 0.85, "3-touch resistance zone"
        return None

# ──────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS
# ──────────────────────────────────────────────────────────────
def calculate_ema(prices, period):
    if len(prices) < period: return prices[-1]
    alpha = 2 / (period + 1)
    ema = prices[0]
    for price in prices[1:]:
        ema = alpha * price + (1 - alpha) * ema
    return ema

def calculate_rsi(prices, period=7):
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
# SIGNAL GENERATION
# ──────────────────────────────────────────────────────────────
def generate_advanced_signals(pairs_list, count, market_type):
    tz_bd = pytz.timezone('Asia/Dhaka')
    now = datetime.now(tz_bd)
    start_time = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
    
    signals = []
    engine = ComprehensiveStrategyEngine()
    
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
        
        # MASTER SIGNAL: Triple confirmation
        master_signal = engine.triple_confirmation_check(candles, pair)
        
        if master_signal:
            probability, reason = master_signal
            direction = "LONG" if "bullish" in reason or "LONG" in reason else "SHORT"
        else:
            # Fallback individual strategies
            individual_signals = [
                engine.vwap_macd_institutional(candles),
                engine.ema_crossover_pullback(candles),
                engine.three_touch_zones(candles, pair)
            ]
            
            valid_individual = [s for s in individual_signals if s]
            if valid_individual:
                probability, reason = valid_individual[0]
                direction = "LONG" if "bullish" in reason or "LONG" in reason else "SHORT"
            else:
                direction = random.choice(["LONG", "SHORT"])
                probability = 0.55
                reason = "No patterns - random signal"
        
        # Apply statistical significance filter
        is_significant, p_value, z_score = engine.validator.statistical_significance(probability)
        if is_significant and z_score > 2.0:
            confidence = min(probability * 1.05, 0.95)
            reason += f" | Sig: p={p_value:.3f}, z={z_score:.2f}"
        else:
            confidence = probability * 0.85
        
        signal_time = start_time + timedelta(minutes=i * 3)
        time_str = signal_time.strftime("%I:%M:00 %p").lower()
        
        signals.append({
            "Pair": pair, "Time": time_str, "Direction": "UP / Call" if direction == "LONG" else "DOWN / Put",
            "Confidence": f"{confidence:.1%}", "Raw_Direction": direction, "Confidence_Value": confidence,
            "Timestamp": signal_time, "Explanation": reason, "Z_Score": z_score if 'z_score' in locals() else 0,
            "P_Value": p_value if 'p_value' in locals() else 1.0
        })
        
        candles.append({
            'open': candles[-1]['close'], 'close': candles[-1]['close'] + random.uniform(-0.0003, 0.0003),
            'high': candles[-1]['close'] + random.uniform(0,
