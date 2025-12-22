import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import threading
import time
from datetime import datetime, timedelta
import pytz
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# ============ CONFIGURATION ============
MIN_CONFIDENCE_THRESHOLD = 70  # Minimum confidence % to trigger signal
BANGLADESH_TZ = pytz.timezone('Asia/Dhaka')

# ============ ENUMS & CLASSES ============
class Signal(Enum):
    BUY = 1
    SELL = -1
    HOLD = 0

class TechnicalAnalyzer:
    """Advanced technical analysis with multiple indicators"""
    
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
        
        levels = {
            '0%': recent_high,
            '23.6%': recent_high - 0.236 * diff,
            '38.2%': recent_high - 0.382 * diff,
            '50%': recent_high - 0.5 * diff,
            '61.8%': recent_high - 0.618 * diff,
            '78.6%': recent_high - 0.786 * diff,
            '100%': recent_low
        }
        return levels
    
    @staticmethod
    def calculate_vwap(data):
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        return (typical_price * data['volume']).cumsum() / data['volume'].cumsum()
    
    @staticmethod
    def calculate_atr(data, period=14):
        tr1 = data['high'] - data['low']
        tr2 = abs(data['high'] - data['close'].shift())
        tr3 = abs(data['low'] - data['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()
    
    @staticmethod
    def detect_support_resistance(data, window=20):
        """Detect key support and resistance levels"""
        recent_data = data.tail(50)
        
        # Resistance = local highs
        resistance = recent_data['high'].rolling(window=window).max().iloc[-1]
        # Support = local lows
        support = recent_data['low'].rolling(window=window).min().iloc[-1]
        
        return support, resistance

class OTCDataProvider:
    """Simulated USD/BDT OTC data provider"""
    
    def __init__(self):
        self.base_rate = 110.50
        self.volatility = 0.08
        
    def fetch_data(self, periods=200):
        """Generate realistic USD/BDT OTC data"""
        data = []
        current_time = datetime.now(BANGLADESH_TZ)
        
        for i in range(periods):
            timestamp = current_time - timedelta(minutes=(periods-i)*3)
            change = np.random.normal(0, self.volatility/100)
            self.base_rate *= (1 + change)
            
            # Bangladesh market hours pattern
            hour = timestamp.hour
            if 10 <= hour <= 16:  # Business hours
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
    """Multi-strategy engine with confidence scoring"""
    
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
        """Calculate individual strategy scores (-1 to 1)"""
        scores = {}
        
        # 1. Moving Average Crossover
        data['SMA_20'] = self.analyzer.calculate_sma(data, 20)
        data['SMA_50'] = self.analyzer.calculate_sma(data, 50)
        scores['ma_crossover'] = 1 if data['SMA_20'].iloc[-1] > data['SMA_50'].iloc[-1] else -1
        
        # 2. RSI
        data['RSI'] = self.analyzer.calculate_rsi(data)
        rsi_val = data['RSI'].iloc[-1]
        if rsi_val < 30: scores['rsi'] = 1
        elif rsi_val > 70: scores['rsi'] = -1
        else: scores['rsi'] = 0
        
        # 3. MACD
        data['MACD'], data['MACD_SIGNAL'] = self.analyzer.calculate_macd(data)
        scores['macd'] = 1 if data['MACD'].iloc[-1] > data['MACD_SIGNAL'].iloc[-1] else -1
        
        # 4. Bollinger Bands
        data['BB_UPPER'], data['BB_MIDDLE'], data['BB_LOWER'] = self.analyzer.calculate_bollinger_bands(data)
        price = data['close'].iloc[-1]
        if price <= data['BB_LOWER'].iloc[-1]: scores['bollinger'] = 1
        elif price >= data['BB_UPPER'].iloc[-1]: scores['bollinger'] = -1
        else: scores['bollinger'] = 0
        
        # 5. Stochastic Oscillator
        data['STOCH_K'], data['STOCH_D'] = self.analyzer.calculate_stochastic(data)
        k_val = data['STOCH_K'].iloc[-1]
        if k_val < 20: scores['stochastic'] = 1
        elif k_val > 80: scores['stochastic'] = -1
        else: scores['stochastic'] = 0
        
        # 6. ADX Trend Strength
        data['ADX'], data['PLUS_DI'], data['MINUS_DI'] = self.analyzer.calculate_adx(data)
        adx_val = data['ADX'].iloc[-1]
        plus_di = data['PLUS_DI'].iloc[-1]
        minus_di = data['MINUS_DI'].iloc[-1]
        
        if adx_val > 25:  # Strong trend
            scores['adx'] = 1 if plus_di > minus_di else -1
        else:
            scores['adx'] = 0
        
        # 7. Fibonacci Levels
        fib_levels = self.analyzer.calculate_fibonacci_levels(data)
        price = data['close'].iloc[-1]
        if abs(price - fib_levels['61.8%']) < 0.10: scores['fibonacci'] = 1
        elif abs(price - fib_levels['38.2%']) < 0.10: scores['fibonacci'] = -1
        else: scores['fibonacci'] = 0
        
        # 8. VWAP
        data['VWAP'] = self.analyzer.calculate_vwap(data)
        scores['vwap'] = 1 if data['close'].iloc[-1] > data['VWAP'].iloc[-1] else -1
        
        # 9. Support/Resistance
        support, resistance = self.analyzer.detect_support_resistance(data)
        price = data['close'].iloc[-1]
        if abs(price - support) < 0.10: scores['support_resistance'] = 1
        elif abs(price - resistance) < 0.10: scores['support_resistance'] = -1
        else: scores['support_resistance'] = 0
        
        return scores, data
    
    def generate_signal(self, data):
        """Generate high-accuracy signal with confidence percentage"""
        scores, data = self.calculate_strategy_scores(data)
        
        # Calculate weighted final score
        final_score = sum(scores[strategy] * weight for strategy, weight in self.weights.items())
        total_weight = sum(self.weights.values())
        normalized_score = final_score / total_weight
        
        # Calculate confidence percentage
        confidence = abs(normalized_score) * 100
        
        # Only trigger signals above confidence threshold
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

# ============ LOGIN SYSTEM ============
class LoginSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("Trading Bot Login")
        self.root.geometry("400x300")
        
        self.VALID_USERNAME = "admin"
        self.VALID_PASSWORD = "admin123"
        
        self.create_widgets()
        
    def create_widgets(self):
        frame = ttk.Frame(self.root, padding="20")
        frame.place(relx=0.5, rely=0.5, anchor="center")
        
        ttk.Label(frame, text="🇧🇩 USD/BDT Trading Bot", font=("Arial", 16, "bold")).grid(row=0, column=0, columnspan=2, pady=20)
        
        ttk.Label(frame, text="Username:").grid(row=1, column=0, sticky="e", pady=5)
        self.username_entry = ttk.Entry(frame, width=30)
        self.username_entry.grid(row=1, column=1, pady=5)
        self.username_entry.insert(0, "admin")
        
        ttk.Label(frame, text="Password:").grid(row=2, column=0, sticky="e", pady=5)
        self.password_entry = ttk.Entry(frame, width=30, show="*")
        self.password_entry.grid(row=2, column=1, pady=5)
        self.password_entry.insert(0, "admin123")
        
        ttk.Button(frame, text="🔐 Login", command=self.login, width=20).grid(row=3, column=0, columnspan=2, pady=20)
        
        self.message_label = ttk.Label(frame, text="", foreground="red")
        self.message_label.grid(row=4, column=0, columnspan=2)
        
    def login(self):
        if self.username_entry.get() == self.VALID_USERNAME and self.password_entry.get() == self.VALID_PASSWORD:
            self.message_label.config(text="✅ Login successful!", foreground="green")
            self.root.after(500, self.launch_main_app)
        else:
            self.message_label.config(text="❌ Invalid credentials!", foreground="red")
    
    def launch_main_app(self):
        self.root.destroy()
        main_root = tk.Tk()
        TradingBot(main_root)
        main_root.mainloop()

# ============ MAIN TRADING BOT ============
class TradingBot:
    def __init__(self, root):
        self.root = root
        self.root.title("🇧🇩 USD/BDT OTC Trading Bot")
        self.root.geometry("1000x800")
        
        self.data_provider = OTCDataProvider()
        self.strategy_engine = StrategyEngine()
        
        self.running = False
        self.analysis_thread = None
        self.signal_history = []
        
        self.setup_gui()
        self.update_status("Bot Stopped", "red")
        
    def setup_gui(self):
        # Header
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill="x", padx=20, pady=10)
        
        ttk.Label(header_frame, text="🇧🇩 USD/BDT OTC Trading Bot", font=("Arial", 22, "bold")).pack()
        
        current_time = datetime.now(BANGLADESH_TZ).strftime("%Y-%m-%d %H:%M:%S")
        self.time_label = ttk.Label(header_frame, text=f"🕐 Bangladesh Time: {current_time}", font=("Arial", 11))
        self.time_label.pack()
        
        # Control buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=15)
        
        self.start_btn = ttk.Button(button_frame, text="▶ START BOT", command=self.start_bot, width=20)
        self.start_btn.grid(row=0, column=0, padx=10)
        
        self.stop_btn = ttk.Button(button_frame, text="⏹ STOP BOT", command=self.stop_bot, width=20, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=10)
        
        # Status
        status_frame = ttk.LabelFrame(self.root, text="System Status", padding="10")
        status_frame.pack(fill="x", padx=20, pady=10)
        
        self.status_label = ttk.Label(status_frame, text="🟡 Waiting...", font=("Arial", 13))
        self.status_label.pack()
        
        # Current Signal
        signal_frame = ttk.LabelFrame(self.root, text="Live Trading Signal", padding="15")
        signal_frame.pack(fill="x", padx=20, pady=10)
        
        self.signal_label = ttk.Label(signal_frame, text="❌ NO SIGNAL YET", font=("Arial", 36, "bold"))
        self.signal_label.pack()
        
        # Signal details
        details_frame = ttk.LabelFrame(self.root, text="Detailed Analysis", padding="10")
        details_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        self.details_text = tk.Text(details_frame, height=18, font=("Courier", 10))
        self.details_text.pack(fill="both", expand=True)
        
        # History
        history_frame = ttk.LabelFrame(self.root, text="Signal History (Last 10)", padding="10")
        history_frame.pack(fill="x", padx=20, pady=10)
        
        self.history_text = tk.Text(history_frame, height=8, font=("Courier", 9))
        self.history_text.pack(fill="x")
        
    def update_status(self, message, color="black"):
        self.status_label.config(text=f"● {message}", foreground=color)
        
    def update_bangladesh_time(self):
        current_time = datetime.now(BANGLADESH_TZ).strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.config(text=f"🕐 Bangladesh Time: {current_time}")
        
    def start_bot(self):
        if not self.running:
            self.running = True
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.update_status("Bot Running - Analyzing every 3 minutes...", "green")
            
            self.analysis_thread = threading.Thread(target=self.analysis_loop, daemon=True)
            self.analysis_thread.start()
            
    def stop_bot(self):
        self.running = False
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.update_status("Bot Stopped", "red")
        if self.analysis_thread:
            self.analysis_thread.join(timeout=1)
            
    def analysis_loop(self):
        """Main analysis loop that runs every 3 minutes"""
        cycle_count = 0
        while self.running:
            try:
                self.update_bangladesh_time()
                cycle_count += 1
                
                # Fetch data
                self.update_status(f"Cycle #{cycle_count} | Fetching market data...", "blue")
                data = self.data_provider.fetch_data(200)
                
                # Generate signal
                self.update_status(f"Cycle #{cycle_count} | Analyzing with 9 strategies...", "blue")
                signal_type, signal_text, confidence, scores, data = self.strategy_engine.generate_signal(data)
                
                # Get current time
                current_time = datetime.now(BANGLADESH_TZ)
                
                # Update UI
                self.update_signal_display(signal_type, signal_text, confidence, scores, data, current_time)
                
                # Log to history
                self.log_signal_to_history(signal_text, current_time, data['close'].iloc[-1])
                
                # Wait for 3 minutes
                self.update_status(f"Cycle #{cycle_count} | Waiting for next cycle...", "green")
                for i in range(180):
                    if not self.running:
                        break
                    time.sleep(1)
                
            except Exception as e:
                self.update_status(f"❌ Error: {str(e)}", "red")
                time.sleep(60)
                
    def update_signal_display(self, signal_type, signal_text, confidence, scores, data, timestamp):
        """Update the main signal display with comprehensive analysis"""
        
        # Update signal label with emoji
        color = "green" if signal_type == Signal.BUY else ("red" if signal_type == Signal.SELL else "orange")
        self.signal_label.config(text=signal_text, foreground=color)
        
        # Prepare strategy breakdown
        breakdown = "\n".join([
            f"{strategy.replace('_', ' ').title():<20}: {'✓' if score > 0 else ('✗' if score < 0 else '○'):>3} | {score:>6.2f}"
            for strategy, score in scores.items()
        ])
        
        # Get Fibonacci levels
        fib_levels = self.strategy_engine.analyzer.calculate_fibonacci_levels(data)
        
        details = f"""
╔══════════════════════════════════════════════════════════════════════╗
║ ANALYSIS TIME: {timestamp.strftime('%Y-%m-%d %H:%M:%S')} (BST)                       ║
║ CURRENT RATE: {data['close'].iloc[-1]:.4f} BDT per USD                                      ║
╚══════════════════════════════════════════════════════════════════════╝

STRATEGY BREAKDOWN (Confidence: {confidence:.1f}%)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{breakdown}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FINAL SCORE: {sum(scores[s] * self.strategy_engine.weights[s] for s in scores):>6.2f}
THRESHOLD:  {MIN_CONFIDENCE_THRESHOLD / 100:>6.2f}

MARKET INDICATORS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MOVING AVERAGES:    SMA(20) = {data['SMA_20'].iloc[-1]:.4f} | SMA(50) = {data['SMA_50'].iloc[-1]:.4f}
RSI (14):           {data['RSI'].iloc[-1]:.2f} {'(OVERSOLD)' if data['RSI'].iloc[-1] < 30 else ('(OVERBOUGHT)' if data['RSI'].iloc[-1] > 70 else '(Neutral)')}
MACD:               {data['MACD'].iloc[-1]:.4f} vs Signal {data['MACD_SIGNAL'].iloc[-1]:.4f}
STOCHASTIC:         K={data['STOCH_K'].iloc[-1]:.2f} D={data['STOCH_D'].iloc[-1]:.2f}
ADX:                {data['ADX'].iloc[-1]:.2f} {'(Strong)' if data['ADX'].iloc[-1] > 25 else '(Weak)'}
VWAP:               {data['VWAP'].iloc[-1]:.4f}

KEY LEVELS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Support/Resistance: {self.strategy_engine.analyzer.detect_support_resistance(data)[0]:.4f} / {self.strategy_engine.analyzer.detect_support_resistance(data)[1]:.4f}
Fibonacci 61.8%:    {fib_levels['61.8%']:.4f}
Bollinger Upper:    {data['BB_UPPER'].iloc[-1]:.4f}
Bollinger Lower:    {data['BB_LOWER'].iloc[-1]:.4f}

OHLC DATA
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Open: {data['open'].iloc[-1]:>12.4f} | High: {data['high'].iloc[-1]:>12.4f}
Low:  {data['low'].iloc[-1]:>12.4f} | Close: {data['close'].iloc[-1]:>11.4f}
Volume: {data['volume'].iloc[-1]:>10,}
"""
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(1.0, details)
        
    def log_signal_to_history(self, signal_text, timestamp, rate):
        """Log signals to history panel"""
        time_str = timestamp.strftime('%H:%M:%S')
        entry = f"{time_str} | {signal_text:<25} | Rate: {rate:.4f}\n"
        
        self.signal_history.append((timestamp, signal_text, rate))
        if len(self.signal_history) > 10:
            self.signal_history.pop(0)
            
        self.history_text.delete(1.0, tk.END)
        header = f"{'Time':<8} | {'Signal':<25} | {'Rate':<10}\n{'='*50}\n"
        self.history_text.insert(1.0, header)
        for ts, sig, rt in self.signal_history:
            self.history_text.insert(tk.END, f"{ts.strftime('%H:%M:%S')} | {sig:<25} | {rt:.4f}\n")

# ============ ENTRY POINT ============
if __name__ == "__main__":
    login_root = tk.Tk()
    login_app = LoginSystem(login_root)
    login_root.mainloop()
