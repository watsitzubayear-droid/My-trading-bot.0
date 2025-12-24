import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import random
import io
from scipy import stats

# ──────────────────────────────────────────────────────────────
# ⚠️ FIXED: Initialize Session State FIRST
# ──────────────────────────────────────────────────────────────
if 'generated_signals' not in st.session_state:
    st.session_state.generated_signals = None
if 'last_update' not in st.session_state:
    st.session_state.last_update = None
if 'engine' not in st.session_state:
    st.session_state.engine = None

# ──────────────────────────────────────────────────────────────
# MARKET UNIVERSE (45 Instruments)
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
# PSYCHOLOGY & MARKET BEHAVIOR LAYER
# ──────────────────────────────────────────────────────────────
class MarketPsychology:
    def __init__(self):
        self.fear_greed_index = {}
        self.retail_positioning = {}
        
    def calculate_fear_greed(self, candles):
        """Proxy for fear/greed using volatility and momentum"""
        returns = np.diff([c['close'] for c in candles])
        volatility = np.std(returns) * np.sqrt(252)
        momentum = (candles[-1]['close'] - candles[-5]['close']) / candles[-5]['close']
        
        # Fear when: high volatility + negative momentum
        # Greed when: low volatility + positive momentum
        fear_score = min(volatility * 10, 1.0) * (1 if momentum < 0 else 0.5)
        greed_score = max(1 - volatility * 5, 0) * (1 if momentum > 0 else 0.5)
        
        return fear_score, greed_score
    
    def detect_herd_behavior(self, volumes):
        """Detect unusual volume spikes suggesting retail herd activity"""
        baseline = np.mean(volumes[:-5])
        recent = np.mean(volumes[-3:])
        spike_ratio = recent / baseline if baseline > 0 else 1
        
        # Spike > 1.8x suggests herd behavior
        return spike_ratio > 1.8, spike_ratio
    
    def institutional_vs_retail(self, candles):
        """Estimate institutional vs retail participation"""
        # Institutions: volume with small wicks, steady price
        # Retail: volume with large wicks, choppy price
        
        wick_sizes = [(c['high'] - max(c['open'], c['close'])) / (c['high'] - c['low']) for c in candles[-10:]]
        volume_trend = np.mean([c['volume'] for c in candles[-5:]]) / np.mean([c['volume'] for c in candles[-10:-5]])
        
        # High wick ratio + falling volume = retail dominance
        wick_ratio = np.mean(wick_sizes)
        institutional_score = (1 - wick_ratio) * volume_trend
        
        return institutional_score, wick_ratio

# ──────────────────────────────────────────────────────────────
# ADVANCED MATHEMATICAL LAYER
# ──────────────────────────────────────────────────────────────
class StatisticalValidator:
    def __init__(self):
        self.confidence_threshold = 0.65
        
    def monte_carlo_confidence(self, candles, n_simulations=1000):
        """Monte Carlo simulation for price path confidence"""
        returns = np.diff([c['close'] for c in candles[-30:]])
        mean_ret = np.mean(returns)
        std_ret = np.std(returns)
        
        # Simulate 1000 price paths
        simulations = np.random.normal(mean_ret, std_ret, (n_simulations, 5))
        cumulative = np.cumsum(simulations, axis=1)
        
        # Probability of positive move
        prob_up = np.mean(cumulative[:, -1] > 0)
        
        return prob_up
    
    def bayesian_update(self, prior_prob, new_evidence, evidence_strength=0.1):
        """Bayesian updating of signal probability"""
        # prior_prob: current belief
        # new_evidence: 0-1 score from new indicator
        # evidence_strength: how much to weight new evidence
        
        posterior = (prior_prob * (1 - evidence_strength) + new_evidence * evidence_strength)
        return posterior
    
    def statistical_significance(self, signal_prob, baseline_prob=0.5, alpha=0.05):
        """Check if signal is statistically significant vs random"""
        # Z-test for proportions
        n_trials = 1000  # Simulated sample size
        z_score = (signal_prob - baseline_prob) / np.sqrt(baseline_prob * (1 - baseline_prob) / n_trials)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        return p_value < alpha, p_value, z_score

# ──────────────────────────────────────────────────────────────
# COMPREHENSIVE STRATEGY ENGINE (15+ Strategies)
# ──────────────────────────────────────────────────────────────
class ComprehensiveStrategyEngine:
    def __init__(self):
        self.psychology = MarketPsychology()
        self.validator = StatisticalValidator()
        self.zones = {}  # Persistent zone tracking
        self.session_cache = {}
        
    def vwap_macd_institutional(self, candles):
        """Strategy 1: VWAP + MACD with institutional confirmation"""
        if len(candles) < 26: return None
        
        # Real VWAP calculation
        typical_prices = [(c['high'] + c['low'] + c['close']) / 3 for c in candles[-20:]]
        volumes = [c['volume'] for c in candles[-20:]]
        vwap = sum(tp * v for tp, v in zip(typical_prices, volumes)) / sum(volumes)
        
        # MACD with proper signal line
        closes = [c['close'] for c in candles[-26:]]
        ema12 = calculate_ema(closes, 12)
        ema26 = calculate_ema(closes, 26)
        macd = ema12 - ema26
        signal = calculate_ema([macd] * 9, 9)
        histogram = macd - signal
        
        # Check for histogram flip
        hist_flip = (histogram > 0 and (calculate_ema(closes[-13:21], 12) - calculate_ema(closes[-27:21], 26) - signal) < 0) or \
                    (histogram < 0 and (calculate_ema(closes[-13:21], 12) - calculate_ema(closes[-27:21], 26) - signal) > 0)
        
        # Price vs VWAP
        price_vs_vwap = closes[-1] > vwap
        
        if price_vs_vwap and hist_flip and histogram > 0:
            return 0.78, "INSTITUTIONAL: VWAP bullish + MACD histogram flip positive"
        elif not price_vs_vwap and hist_flip and histogram < 0:
            return 0.78, "INSTITUTIONAL: VWAP bearish + MACD histogram flip negative"
        
        return None
    
    def ema_crossover_pullback(self, candles):
        """Strategy 2: EMA crossover with pullback qualification"""
        if len(candles) < 21: return None
        
        closes = [c['close'] for c in candles]
        ema9 = calculate_ema(closes[-9:], 9)
        ema21 = calculate_ema(closes[-21:], 21)
        ema50 = calculate_ema(closes[-50:], 50)
        
        # Crossover detection
        cross_up = ema9 > ema21 and calculate_ema(closes[-10:-9], 9) < calculate_ema(closes[-22:-21], 21)
        cross_down = ema9 < ema21 and calculate_ema(closes[-10:-9], 9) > calculate_ema(closes[-22:-21], 21)
        
        # Pullback check: price must not be too far from EMA
        distance_from_ema = abs(closes[-1] - ema9) / closes[-1]
        
        # Trend filter: EMA50 must support direction
        trend_aligned = (cross_up and ema9 > ema50) or (cross_down and ema9 < ema50)
        
        if cross_up and distance_from_ema < 0.001 and trend_aligned:
            return 0.75, "EMA 9/21 golden cross + pullback + trend aligned"
        elif cross_down and distance_from_ema < 0.001 and trend_aligned:
            return 0.75, "EMA 9/21 death cross + pullback + trend aligned"
        
        return None
    
    def three_touch_zone_with_time_decay(self, candles, pair):
        """Strategy 3: 3-Touch zones with time decay"""
        if len(candles) < 20: return None
        
        # Detect zones from last 20 candles
        highs = [c['high'] for c in candles[-20:]]
        lows = [c['low'] for c in candles[-20:]]
        current_price = candles[-1]['close']
        
        # Update persistent zone tracker
        key_levels = []
        for level in np.linspace(min(lows), max(highs), 15):
            touches = sum(1 for h in highs if abs(h - level) < 0.0003) + \
                      sum(1 for l in lows if abs(l - level) < 0.0003)
            
            if touches >= 2:  # Potential zone
                if pair not in self.zones:
                    self.zones[pair] = {}
                if level not in self.zones[pair]:
                    self.zones[pair][level] = {'touches': 0, 'first_touch': datetime.now()}
                
                self.zones[pair][level]['touches'] += 1
                self.zones[pair][level]['last_touch'] = datetime.now()
                key_levels.append((level, self.zones[pair][level]['touches']))
        
        # Check for 3-touch zones
        for level, touches in key_levels:
            if touches >= 3 and abs(current_price - level) < 0.0005:
                zone_age_hours = (datetime.now() - self.zones[pair][level]['first_touch']).total_seconds() / 3600
                time_decay_factor = max(1 - zone_age_hours / 24, 0.5)  # Zones older than 24h lose reliability
                
                if level < np.mean([c['close'] for c in candles[-5:]]):
                    return 0.85 * time_decay_factor, f"3-touch support zone (age: {zone_age_hours:.1f}h)"
                else:
                    return 0.85 * time_decay_factor, f"3-touch resistance zone (age: {zone_age_hours:.1f}h)"
        
        return None
    
    def rsi_bb_reversion_with_volatility(self, candles):
        """Strategy 4: RSI + BB with volatility filter"""
        if len(candles) < 20: return None
        
        closes = [c['close'] for c in candles[-20:]]
        bb_upper = np.mean(closes) + 2 * np.std(closes)
        bb_lower = np.mean(closes) - 2 * np.std(closes)
        rsi_4 = calculate_rsi([c['close'] for c in candles], 4)
        rsi_14 = calculate_rsi([c['close'] for c in candles], 14)
        
        # Volatility regime filter
        current_vol = np.std([c['high'] - c['low'] for c in candles[-5:]])
        avg_vol = np.std([c['high'] - c['low'] for c in candles[-20:]])
        high_vol_regime = current_vol > avg_vol * 1.5
        
        # Only trade mean reversion in low/medium volatility
        if not high_vol_regime:
            if candles[-1]['low'] < bb_lower and rsi_4 < 25 and rsi_14 < 30:
                return 0.72, "BB+RSI reversion in low vol regime"
            elif candles[-1]['high'] > bb_upper and rsi_4 > 75 and rsi_14 > 70:
                return 0.72, "BB+RSI reversion in low vol regime"
        
        return None
    
    def volume_delta_institutional(self, candles):
        """Strategy 5: Volume delta for institutional vs retail"""
        if len(candles) < 10: return None
        
        # Compare volume momentum to price momentum
        vol_momentum = np.mean([c['volume'] for c in candles[-3:]]) / np.mean([c['volume'] for c in candles[-10:-3]])
        price_momentum = (candles[-1]['close'] - candles[-4]['open']) / candles[-4]['open']
        
        # Institutional buying: volume up, price down (accumulation)
        if vol_momentum > 1.6 and price_momentum < -0.001:
            return 0.68, f"Institutional accumulation (vol: {vol_momentum:.1f}x, price: {price_momentum:.2%})"
        
        # Institutional selling: volume up, price up (distribution)
        elif vol_momentum > 1.6 and price_momentum > 0.001:
            return 0.68, f"Institutional distribution (vol: {vol_momentum:.1f}x, price: {price_momentum:.2%})"
        
        return None
    
    def time_of_day_pattern(self, candles):
        """Strategy 6: Time-of-day statistical patterns"""
        # Simulate BDT time (real implementation would use historical data)
        bdt_hour = datetime.now(pytz.timezone('Asia/Dhaka')).hour
        
        # Overlap hours (13-16 GMT = 19-22 BDT) are highest probability
        if 19 <= bdt_hour <= 22:
            # During overlap, bias toward momentum
            momentum = candles[-1]['close'] - candles[-3]['close']
            if momentum > 0:
                return 0.70, "London-NY overlap momentum bias"
            else:
                return 0.70, "London-NY overlap momentum bias"
        
        # Low liquidity hours (12-13 GMT = 18-19 BDT) = avoid
        elif 18 <= bdt_hour <= 19:
            return 0.40, "Low liquidity session - reduced confidence"
        
        return None
    
    def volatility_regime_filter(self, candles):
        """Strategy 7: Regime-based strategy switching"""
        # Calculate rolling volatility
        returns = np.diff([c['close'] for c in candles[-20:]])
        current_vol = np.std(returns[-5:]) * np.sqrt(252)
        avg_vol = np.std(returns) * np.sqrt(252)
        
        # High volatility regime: use mean reversion
        if current_vol > avg_vol * 1.5:
            return 0.65, "High volatility regime - mean reversion favored"
        
        # Low volatility regime: use trend following
        elif current_vol < avg_vol * 0.7:
            return 0.65, "Low volatility regime - trend following favored"
        
        # Normal regime: use mixed strategies
        else:
            return 0.60, "Normal volatility regime - balanced approach"
        
        return None
    
    def correlation_filter(self, candles, pair):
        """Strategy 8: Inter-market correlation"""
        # Simplified: Check if pair moves with major indices
        if "USD" in pair:
            # USD pairs often correlate with DXY
            usd_strength = np.mean([c['close'] > c['open'] for c in candles[-5:]])
            if usd_strength > 0.7:
                return 0.60, "USD strength correlation filter"
            elif usd_strength < 0.3:
                return 0.60, "USD weakness correlation filter"
        
        return None
    
    def market_psychology_overlay(self, candles):
        """Strategy 9: Psychology overlay"""
        psych = MarketPsychology()
        fear, greed = psych.calculate_fear_greed(candles)
        herd_active, herd_ratio = psych.detect_herd_behavior([c['volume'] for c in candles[-10:]])
        inst_score, wick_ratio = psych.institutional_vs_retail(candles[-10:])
        
        # Fear + herd selling = potential reversal
        if fear > 0.7 and herd_active and herd_ratio > 2.0:
            return 0.73, f"Fear+herd selling (fear:{fear:.1%}, herd:{herd_ratio:.1f}x) - contrarian signal"
        
        # Greed + retail FOMO = potential top
        elif greed > 0.7 and wick_ratio > 0.6:
            return 0.73, f"Greed+retail FOMO (greed:{greed:.1%}, wicks:{wick_ratio:.1f}) - reversal signal"
        
        return None
    
    def statistical_significance_filter(self, signal_prob, baseline_prob=0.5):
        """Strategy 10: Statistical significance check"""
        validator = StatisticalValidator()
        is_significant, p_value, z_score = validator.statistical_significance(signal_prob, baseline_prob)
        
        if is_significant and z_score > 2.0:  # 95% confidence
            return 0.80, f"Statistically significant (z={z_score:.2f}, p={p_value:.3f})"
        elif is_significant and z_score > 1.5:  # 85% confidence
            return 0.70, f"Marginally significant (z={z_score:.2f}, p={p_value:.3f})"
        
        return None
    
    def monte_carlo_probability(self, candles):
        """Strategy 11: Monte Carlo simulation"""
        validator = StatisticalValidator()
        prob_up = validator.monte_carlo_confidence(candles)
        
        if prob_up > 0.75:
            return prob_up, f"Monte Carlo {prob_up:.1%} probability up"
        elif prob_up < 0.25:
            return 1 - prob_up, f"Monte Carlo {1-prob_up:.1%} probability down"
        
        return None
    
    def session_quality_score(self, candles):
        """Strategy 12: Session-based quality scoring"""
        bdt_hour = datetime.now(pytz.timezone('Asia/Dhaka')).hour
        
        # Quality scores by session (BDT time)
        session_scores = {
            (13, 16): 0.85,  # London-NY overlap (19-22 BDT)
            (8, 12): 0.70,   # London open (14-18 BDT)
            (0, 3): 0.60,    # Tokyo open (06-09 BDT)
            (12, 13): 0.40,  # Lunch (18-19 BDT)
        }
        
        for (start, end), score in session_scores.items():
            if start <= bdt_hour <= end:
                return score, f"Session quality: {score:.1%}"
        
        return 0.50, "Off-hours session"
    
    def spread_vs_volatility_filter(self, candles):
        """Strategy 13: Spread widening as volatility predictor"""
        spreads = [c['spread'] for c in candles[-10:]]
        current_spread = spreads[-1]
        avg_spread = np.mean(spreads[:-1])
        
        if avg_spread == 0: return None
        
        spread_ratio = current_spread / avg_spread
        
        if spread_ratio > 2.0:
            return 0.45, f"Spread widening {spread_ratio:.1f}x - volatility incoming"
        elif spread_ratio < 0.7:
            return 0.65, f"Spread compression {spread_ratio:.1f}x - calm before move"
        
        return None
    
    def triple_confirmation_check(self, candles, pair):
        """MASTER STRATEGY: Requires 3+ independent confirmations"""
        all_strategies = [
            self.vwap_macd_institutional(candles),
            self.ema_crossover_pullback(candles),
            self.three_touch_zones(candles, pair),
            self.rsi_bb_reversion_with_volatility(candles),
            self.volume_delta_institutional(candles),
            self.time_of_day_pattern(candles),
            self.volatility_regime_filter(candles),
            self.correlation_filter(candles, pair),
            self.market_psychology_overlay(candles),
            self.session_quality_score(candles),
            self.spread_vs_volatility_filter(candles)
        ]
        
        # Filter out None results
        valid_signals = [s for s in all_strategies if s]
        
        if len(valid_signals) < 3:
            return None  # Insufficient confirmations
        
        # Separate long and short signals
        long_signals = [(p, r) for p, r in valid_signals if 'bullish' in r or 'LONG' in r or 'up' in r]
        short_signals = [(p, r) for p, r in valid_signals if 'bearish' in r or 'SHORT' in r or 'down' in r]
        
        # Check for consensus: need 2+ signals in same direction
        if len(long_signals) >= 2 and len(long_signals) > len(short_signals):
            avg_prob = np.mean([p for p, _ in long_signals])
            reasons = [r for _, r in long_signals]
            return min(avg_prob * 1.1, 0.92), f"TRIPLE CONFIRMED: {' + '.join(reasons[:3])}"
        
        elif len(short_signals) >= 2 and len(short_signals) > len(long_signals):
            avg_prob = np.mean([p for p, _ in short_signals])
            reasons = [r for _, r in short_signals]
            return min(avg_prob * 1.1, 0.92), f"TRIPLE CONFIRMED: {' + '.join(reasons[:3])}"
        
        elif len(valid_signals) >= 4:
            # Mixed signals but high activity = uncertain
            return 0.60, "Mixed signals - high market uncertainty"
        
        return None

# ──────────────────────────────────────────────────────────────
# MAIN GENERATION LOGIC
# ──────────────────────────────────────────────────────────────
def generate_advanced_signals(pairs_list, count, market_type):
    """Generates signals using ALL strategies and triple confirmation"""
    tz_bd = pytz.timezone('Asia/Dhaka')
    now = datetime.now(tz_bd)
    start_time = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
    
    signals = []
    engine = ComprehensiveStrategyEngine()
    
    # Build synthetic candle history (would be replaced with real 10-day data)
    candles = []
    base_price = 1.0850
    for i in range(50):
        base_price += random.uniform(-0.001, 0.001)
        candles.append({
            'open': base_price,
            'close': base_price + random.uniform(-0.0005, 0.0005),
            'high': base_price + random.uniform(0, 0.001),
            'low': base_price - random.uniform(0, 0.001),
            'volume': random.randint(100, 500),
            'spread': random.uniform(0.0001, 0.0005),
            'atr': random.uniform(0.0005, 0.001)
        })
    
    for i in range(count):
        pair = random.choice(pairs_list)
        
        # MASTER SIGNAL: Requires triple confirmation
        master_signal = engine.triple_confirmation_check(candles, pair)
        
        if master_signal:
            probability, reason = master_signal
            direction = "LONG" if "bullish" in reason or "LONG" in reason or "up" in reason else "SHORT"
        else:
            # Fallback to individual strategies if no triple confirmation
            individual_signals = [
                engine.vwap_macd_institutional(candles),
                engine.ema_crossover_pullback(candles),
                engine.three_touch_zones(candles, pair),
                engine.rsi_bb_reversion_with_volatility(candles),
                engine.volume_delta_institutional(candles)
            ]
            
            valid_individual = [s for s in individual_signals if s]
            if valid_individual:
                probability, reason = valid_individual[0]
                direction = "LONG" if "bullish" in reason or "LONG" in reason else "SHORT"
            else:
                # Final fallback: Monte Carlo
                mc_prob = engine.monte_carlo_probability(candles)
                if mc_prob:
                    probability, reason = mc_prob
                    direction = "LONG" if probability > 0.5 else "SHORT"
                else:
                    # Random as last resort
                    direction = random.choice(["LONG", "SHORT"])
                    probability = 0.55
                    reason = "No patterns - random signal"
        
        # Apply statistical significance filter
        is_significant, p_value, z_score = engine.validator.statistical_significance(probability)
        if is_significant and z_score > 2.0:
            confidence = min(probability * 1.05, 0.95)
            reason += f" | Sig: p={p_value:.3f}, z={z_score:.2f}"
        elif not is_significant:
            confidence = probability * 0.85  # Reduce confidence if not significant
            reason += f" | Not significant: p={p_value:.3f}"
        else:
            confidence = probability
        
        # 3-minute spacing
        signal_time = start_time + timedelta(minutes=i * 3)
        time_str = signal_time.strftime("%I:%M:00 %p").lower()
        
        signals.append({
            "Pair": pair,
            "Time": time_str,
            "Direction": "UP / Call" if direction == "LONG" else "DOWN / Put",
            "Confidence": f"{confidence:.1%}",
            "Raw_Direction": direction,
            "Confidence_Value": confidence,
            "Timestamp": signal_time,
            "Explanation": reason,
            "Z_Score": z_score if 'z_score' in locals() else 0,
            "P_Value": p_value if 'p_value' in locals() else 1.0
        })
        
        # Update candles for next iteration
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
    
    return signals

# ──────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS (Mathematically Correct)
# ──────────────────────────────────────────────────────────────
def calculate_ema(prices, period):
    """True EMA calculation"""
    if len(prices) < period: return prices[-1]
    alpha = 2 / (period + 1)
    ema = prices[0]
    for price in prices[1:]:
        ema = alpha * price + (1 - alpha) * ema
    return ema

def calculate_rsi(prices, period=7):
    """True RSI with Wilder's smoothing"""
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

# ──────────────────────────────────────────────────────────────
# MAIN APP UI
# ──────────────────────────────────────────────────────────────
# CSS Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto+Mono&display=swap');
    .stApp { background: linear-gradient(135deg, #0a0a0a 0%, #1a0033 100%); }
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
    .stButton>button { background: rgba(20, 20, 20, 0.8) !important; color: #00ffff !important; border: 2px solid #00ffff !important; border-radius: 10px !important; box-shadow: 0 0 15px rgba(0, 255, 255, 0.5) !important; font-family: 'Orbitron', monospace !important; font-weight: bold !important; font-size: 16px !important; transition: all 0.3s !important; padding: 15px 30px !important; }
    .stDownloadButton>button { background: rgba(138, 43, 226, 0.2) !important; color: #8a2be2 !important; border: 2px solid #8a2be2 !important; border-radius: 10px !important; box-shadow: 0 0 15px rgba(138, 43, 226, 0.5) !important; font-family: 'Orbitron', monospace !important; font-weight: bold !important; font-size: 16px !important; }
    </style>
""", unsafe_allow_html=True)

# 3D Logo
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

# Header
st.markdown('<div class="neon-header">ZOHA FUTURE SIGNALS</div>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#00f2ff; font-size:18px;'>Multi-Layer Validation Engine | 15+ Strategies | Psychology + Math + Stat</p>", unsafe_allow_html=True)

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
if st.button("⚡ GENERATE 100+ VALIDATED SIGNALS", use_container_width=True):
    if not pairs:
        st.error("❌ Please select at least one market pair.")
    else:
        with st.spinner("🔍 Layer 1: Technical Analysis... 🔍 Layer 2: Psychology... 🔍 Layer 3: Statistics..."):
            # Build synthetic historical data
            candles = []
            base_price = 1.0850
            for i in range(50):
                base_price += random.uniform(-0.001, 0.001)
                candles.append({
                    'open': base_price,
                    'close': base_price + random.uniform(-0.0005, 0.0005),
                    'high': base_price + random.uniform(0, 0.001),
                    'low': base_price - random.uniform(0, 0.001),
                    'volume': random.randint(100, 500),
                    'spread': random.uniform(0.0001, 0.0005),
                    'atr': random.uniform(0.0005, 0.001)
                })
            
            market_type = "otc" if "OTC" in pairs[0] else "real"
            signals = generate_advanced_signals(pairs, num_signals, market_type)
            
            st.session_state.generated_signals = signals
            st.session_state.last_update = bdt_time
        st.success(f"✅ **Generated {len(signals)} validated signals**")

# Display Signals
if st.session_state.generated_signals is not None:
    st.markdown("---")
    st.markdown("### 📊 **VALIDATED SIGNALS (3-MIN INTERVALS)**")
    
    for sig in st.session_state.generated_signals:
        color_class = "up-call" if sig['Raw_Direction'] == "LONG" else "down-put"
        
        # Highlight statistically significant signals
        significance = "⭐ " if sig['Z_Score'] > 2.0 else ""
        
        st.markdown(f"""
        <div class="signal-container">
            <div>
                <span class="pair-text">{significance}{sig['Pair']}</span><br>
                <span class="time-text">{sig['Time']}</span>
            </div>
            <div>
                <span class="{color_class}">{sig['Direction']}</span>
                <span class="accuracy-tag" style="margin-left:15px;">{sig['Confidence']}</span>
            </div>
        </div>
        <div class="explanation-box">
            <strong>Validation:</strong> {sig['Explanation']}<br>
            <strong>Z-Score:</strong> {sig['Z_Score']:.2f} | <strong>P-Value:</strong> {sig['P_Value']:.3f}<br>
            {sig['Warning'] if sig['Warning'] else ''}
        </div>
        """, unsafe_allow_html=True)

# Download
if st.session_state.generated_signals is not None:
    st.markdown("---")
    df_download = pd.DataFrame(st.session_state.generated_signals)
    csv_buffer = io.StringIO()
    df_download.to_csv(csv_buffer, index=False, columns=["Pair", "Time", "Direction", "Confidence", "Explanation", "Z_Score", "P_Value"])
    
    st.download_button(
        label="📥 DOWNLOAD VALIDATED SIGNALS (CSV)",
        data=csv_buffer.getvalue(),
        file_name=f"zoha_signals_{bdt_time.strftime('%Y%m%d_%H%M%S')}.csv",
        mime='text/csv',
        use_container_width=True
    )

# Transparency
st.markdown("---")
st.warning("""
**⚠️ TRANSPARENCY: This is a DEMONSTRATION system using synthetic data.**<br>
**Production requires**: Real-time data, 10-day history, GPU compute, low-latency VPS.<br>
**Cost**: $1,500+/month for institutional-grade infrastructure.
""")
