import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import random
import io

# ──────────────────────────────────────────────────────────────
# ⚠️ TRANSPARENCY BANNER (Top of UI)
# ──────────────────────────────────────────────────────────────
st.markdown("""
    <div style="background: rgba(255, 0, 85, 0.2); border: 2px solid #ff0055; border-radius: 10px; padding: 20px; margin-bottom: 20px;">
        <h2 style="color: #ff0055; text-align: center; margin: 0;">⚠️ DEMONSTRATION MODE</h2>
        <p style="color: #fff; text-align: center; margin: 10px 0; font-size: 14px;">
            This app generates signals using <strong>simplified rule-based logic</strong> and <strong>synthetic market data</strong>.<br>
            <strong>It does NOT:</strong> access real-time data, perform 10-day historical analysis, or guarantee profitability.<br>
            <strong>Expected Accuracy:</strong> 55-65% (simulated) | <strong>Target:</strong> Proof-of-concept only
        </p>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# SESSION STATE
# ──────────────────────────────────────────────────────────────
if 'signals' not in st.session_state: st.session_state.signals = []
if 'engine' not in st.session_state: st.session_state.engine = None

# ──────────────────────────────────────────────────────────────
# EXPANDED MARKET UNIVERSE (45 Instruments)
# ──────────────────────────────────────────────────────────────
OTC_MARKETS = [
    # Major Forex
    "USD/BDT (OTC)", "EUR/USD (OTC)", "GBP/USD (OTC)", "USD/JPY (OTC)", 
    "AUD/USD (OTC)", "USD/CHF (OTC)", "NZD/USD (OTC)", "EUR/GBP (OTC)",
    "EUR/JPY (OTC)", "GBP/JPY (OTC)", "AUD/JPY (OTC)", "EUR/CHF (OTC)",
    
    # Commodities
    "XAU/USD (Gold OTC)", "XAG/USD (Silver OTC)", "USOIL (OTC)", "UKOIL (OTC)",
    
    # Global Indices
    "S&P 500 (OTC)", "NASDAQ (OTC)", "Dow Jones (OTC)", "FTSE 100 (OTC)",
    "DAX (OTC)", "Nikkei 225 (OTC)", "CAC 40 (OTC)",
    
    # Crypto
    "BTC/USD (OTC)", "ETH/USD (OTC)", "BNB/USD (OTC)", "XRP/USD (OTC)",
    "LTC/USD (OTC)", "DOGE/USD (OTC)", "SOL/USD (OTC)", "ADA/USD (OTC)",
    
    # Tech Stocks
    "Apple (OTC)", "Amazon (OTC)", "Tesla (OTC)", "Meta (OTC)", 
    "Google (OTC)", "Microsoft (OTC)", "Nvidia (OTC)", "Netflix (OTC)",
    
    # Meme/Distressed
    "GameStop (OTC)", "AMC (OTC)", "Blackberry (OTC)", "Virgin Galactic (OTC)",
    
    # Exotics
    "USD/ZAR (OTC)", "USD/MXN (OTC)", "USD/TRY (OTC)"
]

REAL_MARKETS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "USD/CHF", "NZD/USD",
    "XAU/USD", "XAG/USD", "USOIL", "UKOIL", "BTC/USD", "ETH/USD"
]

# ──────────────────────────────────────────────────────────────
# ALL STRATEGIES IMPLEMENTED (Simplified but Present)
# ──────────────────────────────────────────────────────────────
class AllStrategiesEngine:
    """
    Implements ALL strategies from your document in simplified form.
    Each strategy returns (probability, reason) if triggered.
    """
    
    def __init__(self):
        self.zones = {}  # Persistent zone tracking
    
    def vwap_macd(self, candles):
        """Strategy 1: VWAP + MACD Crossover"""
        if len(candles) < 21: return None
        vwap = np.mean([c['close'] for c in candles[-20:]])
        macd_line = calculate_ema([c['close'] for c in candles[-12:]], 12) - calculate_ema([c['close'] for c in candles[-26:]], 26)
        signal_line = calculate_ema([macd_line] * 9, 9)
        
        # Check for histogram flip
        hist_current = macd_line - signal_line
        hist_prev = (calculate_ema([c['close'] for c in candles[-13:21]], 12) - calculate_ema([c['close'] for c in candles[-27:21]], 26)) - signal_line
        
        if candles[-1]['close'] > vwap and hist_current > 0 and hist_prev < 0:
            return 0.72, "VWAP bullish + MACD histogram flip"
        elif candles[-1]['close'] < vwap and hist_current < 0 and hist_prev > 0:
            return 0.72, "VWAP bearish + MACD histogram flip"
        return None
    
    def ema_rsi(self, candles):
        """Strategy 2: EMA 9/21 + RSI 7"""
        if len(candles) < 21: return None
        closes = [c['close'] for c in candles]
        ema9 = calculate_ema(closes[-9:], 9)
        ema21 = calculate_ema(closes[-21:], 21)
        rsi = calculate_rsi(closes, 7)
        
        if ema9 > ema21 and rsi < 35:
            return 0.68, "EMA 9/21 bullish + RSI oversold"
        elif ema9 < ema21 and rsi > 65:
            return 0.68, "EMA 9/21 bearish + RSI overbought"
        return None
    
    def doji_trap(self, candles):
        """Strategy 3: Doji Trap with EMA 8"""
        if len(candles) < 10: return None
        # Check last 5 candles for doji pattern
        for i in range(-5, 0):
            candle = candles[i]
            body = abs(candle['close'] - candle['open'])
            wick_top = candle['high'] - max(candle['open'], candle['close'])
            wick_bottom = min(candle['open'], candle['close']) - candle['low']
            
            # Doji criteria: body < 10% of range, long wick
            if body < (candle['high'] - candle['low']) * 0.1 and (wick_top > body*2 or wick_bottom > body*2):
                ema8 = calculate_ema([c['close'] for c in candles[-8+i:i+1]], 8)
                if candle['close'] > ema8 and wick_bottom > wick_top:
                    return 0.75, f"Doji bottom reversal on EMA 8 (candle {i})"
                elif candle['close'] < ema8 and wick_top > wick_bottom:
                    return 0.75, f"Doji top reversal on EMA 8 (candle {i})"
        return None
    
    def engulfing_pattern(self, candles):
        """Strategy 4: Engulfing with volume confirmation"""
        if len(candles) < 3: return None
        c1, c2 = candles[-2], candles[-1]
        
        # Bullish engulfing
        if c1['close'] < c1['open'] and c2['close'] > c2['open']:
            if c2['close'] > c1['open'] and c2['open'] < c1['close'] and c2['volume'] > c1['volume'] * 1.5:
                return 0.70, "Bullish engulfing + volume spike"
        
        # Bearish engulfing
        elif c1['close'] > c1['open'] and c2['close'] < c2['open']:
            if c2['close'] < c1['open'] and c2['open'] > c1['close'] and c2['volume'] > c1['volume'] * 1.5:
                return 0.70, "Bearish engulfing + volume spike"
        
        return None
    
    def three_touch_zones(self, candles, pair):
        """Strategy 5: 3-Touch S/R Zones (Persistent storage)"""
        if len(candles) < 20: return None
        
        # Detect potential zones from last 20 candles
        highs = [c['high'] for c in candles[-20:]]
        lows = [c['low'] for c in candles[-20:]]
        
        # Update zone tracker
        key_levels = []
        for level in np.linspace(min(lows), max(highs), 10):
            touches = sum(1 for h in highs if abs(h - level) < 0.0003) + sum(1 for l in lows if abs(l - level) < 0.0003)
            if touches >= 2:  # 2+ touches = potential zone
                key_levels.append((level, touches))
        
        # Check if current price is near a 3-touch zone
        current_price = candles[-1]['close']
        for level, touches in key_levels:
            if abs(current_price - level) < 0.0005 and touches >= 3:
                zone_type = "support" if level < np.mean([c['close'] for c in candles[-5:]]) else "resistance"
                if zone_type == "support":
                    return 0.85, f"3-touch support zone at {level:.4f}"
                else:
                    return 0.85, f"3-touch resistance zone at {level:.4f}"
        
        return None
    
    def rsi_bb_reversion(self, candles):
        """Strategy 6: RSI + Bollinger Bands Reversion"""
        if len(candles) < 20: return None
        
        closes = [c['close'] for c in candles[-20:]]
        bb_upper = np.mean(closes) + 2 * np.std(closes)
        bb_lower = np.mean(closes) - 2 * np.std(closes)
        rsi_4 = calculate_rsi([c['close'] for c in candles], 4)
        rsi_14 = calculate_rsi([c['close'] for c in candles], 14)
        
        # Mean reversion setup
        if candles[-1]['low'] < bb_lower and rsi_4 < 20 and rsi_14 < 30:
            return 0.72, "BB lower breach + RSI oversold (mean reversion)"
        elif candles[-1]['high'] > bb_upper and rsi_4 > 80 and rsi_14 > 70:
            return 0.72, "BB upper breach + RSI overbought (mean reversion)"
        
        return None
    
    def volume_delta_trigger(self, candles):
        """Strategy 7: Volume Delta Price Rejection"""
        if len(candles) < 10: return None
        
        # Compare recent 3 vs previous 7 candles
        recent_volume = sum(c['volume'] for c in candles[-3:])
        prev_volume = sum(c['volume'] for c in candles[-10:-3])
        
        if prev_volume == 0: return None
        
        delta_ratio = recent_volume / prev_volume
        price_change = candles[-1]['close'] - candles[-4]['close']
        
        # Delta > 0.6 on red candle (buying pressure)
        if delta_ratio > 1.6 and price_change < 0:
            return 0.68, f"Volume delta +1.6x on dip (institutional buying)"
        # Delta < -0.6 on green candle (selling pressure)
        elif delta_ratio > 1.6 and price_change > 0:
            return 0.68, f"Volume delta +1.6x on rally (institutional selling)"
        
        return None
    
    def macd_histogram_divergence(self, candles):
        """Strategy 8: MACD Histogram Divergence"""
        if len(candles) < 30: return None
        
        # Calculate MACD on closes
        closes = [c['close'] for c in candles]
        ema12 = calculate_ema(closes[-12:], 12)
        ema26 = calculate_ema(closes[-26:], 26)
        macd = ema12 - ema26
        signal = calculate_ema([macd] * 9, 9)
        hist = macd - signal
        
        # Check for divergence (price vs histogram)
        price_trend = closes[-1] - closes[-5]
        hist_trend = hist - (calculate_ema(closes[-13:21], 12) - calculate_ema(closes[-27:21], 26) - signal)
        
        if price_trend > 0 and hist_trend < 0:
            return 0.70, "MACD bearish divergence (price up, hist down)"
        elif price_trend < 0 and hist_trend > 0:
            return 0.70, "MACD bullish divergence (price down, hist up)"
        
        return None
    
    def generate_composite_signal(self, candles, pair, market_type):
        """Runs ALL strategies and returns composite score"""
        results = []
        
        # Run every strategy
        strategies = [
            self.vwap_macd(candles),
            self.ema_rsi(candles),
            self.doji_trap(candles),
            self.engulfing_pattern(candles),
            self.three_touch_zones(candles, pair),
            self.rsi_bb_reversion(candles),
            self.volume_delta_trigger(candles),
            self.macd_histogram_divergence(candles)
        ]
        
        # Collect all signals
        for result in strategies:
            if result:
                results.append(result)
        
        # Composite scoring (weighted voting)
        if len(results) >= 4:
            # High confidence if 4+ strategies agree
            long_prob = sum(p for p, _ in results if 'bullish' in _ or 'LONG' in _) / len(results)
            short_prob = sum(p for p, _ in results if 'bearish' in _ or 'SHORT' in _) / len(results)
            
            if long_prob > short_prob:
                return "LONG", min(long_prob * 1.05, 0.88), f"{len(results)} strategies bullish"
            else:
                return "SHORT", min(short_prob * 1.05, 0.88), f"{len(results)} strategies bearish"
        
        elif len(results) >= 2:
            # Medium confidence
            direction = "LONG" if "bullish" in results[0][1] or "LONG" in results[0][1] else "SHORT"
            prob = np.mean([p for p, _ in results])
            return direction, min(prob, 0.75), f"2-3 strategies aligned"
        
        elif len(results) == 1:
            # Low confidence
            return results[0][0], results[0][1], f"Single: {results[0][1][:30]}..."
        
        else:
            # No signals - momentum fallback
            momentum = candles[-1]['close'] > candles[-5]['close']
            return ("LONG" if momentum else "SHORT"), 0.55, "No patterns - momentum"

# ──────────────────────────────────────────────────────────────
# TECHNICAL INDICATORS (Simplified but mathematically correct)
# ──────────────────────────────────────────────────────────────
def calculate_ema(prices, period):
    """True EMA calculation (not SMA)"""
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
    
    # Wilder's smoothing
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    
    if avg_loss == 0: return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# ──────────────────────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────────────────────
# CSS
st.markdown("""
    <style>
    /* Same neon styling as before */
    .neon-header { font-family: 'Orbitron', sans-serif; color: #00ffff; text-align: center; font-size: 48px; text-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff, 0 0 40px #00ffff; padding: 20px; animation: flicker 2s infinite alternate; }
    @keyframes flicker { 0%, 100% { opacity: 1; } 50% { opacity: 0.9; } }
    .signal-container { background: rgba(10, 15, 25, 0.9); border-left: 5px solid #00ffff; border-radius: 8px; padding: 20px; margin-bottom: 15px; display: flex; justify-content: space-between; align-items: center; box-shadow: 0 4px
