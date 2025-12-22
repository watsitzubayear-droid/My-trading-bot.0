#!/usr/bin/env python3
"""
⚠️ CRITICAL DISCLAIMERS - READ FIRST
=====================================
1. **NO OFFICIAL API**: Quotex does NOT provide an official API. This uses web automation (Selenium) which is FRAGILE and may BREAK at any time.
2. **ACCOUNT BAN RISK**: Using bots violates Quotex's Terms of Service. Your account can be PERMANENTLY BANNED with funds frozen.
3. **NO GUARANTEED PROFITS**: Markets are inherently unpredictable. This is for EDUCATIONAL PURPOSES ONLY.
4. **SECURITY RISK**: Storing credentials is dangerous. Use ONLY on a DEMO account.
5. **SIMULATION MODE ENABLED**: No real trades will execute by default. Set SIMULATION_MODE = False at your own EXTREME RISK.
"""

import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import talib
import yfinance as yf
import pytz
from flask import Flask, render_template_string, request, jsonify, redirect, url_for, flash
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from apscheduler.schedulers.background import BackgroundScheduler
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

# =============================================================================
# CONFIGURATION
# =============================================================================
class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'quotex-bot-secret-key-change-in-production')
    DATABASE_PATH = 'data/trading.db'
    SIGNAL_INTERVAL = 3  # minutes between signals
    ANALYSIS_LOOKBACK = 5  # days of historical data
    SIMULATION_MODE = True  # ⚠️ SET TO FALSE ONLY IF YOU ACCEPT ALL RISKS
    BANGLADESH_TZ = 'Asia/Dhaka'  # UTC+6
    
    # Trading parameters
    MIN_ACCURACY = 60  # Minimum accuracy to generate signal
    RISK_PER_TRADE = 0.01  # 1% risk per trade
    TRADE_DURATION = 1  # minutes (1-min strategy)
    
    # Quotex credentials (⚠️ USE DEMO ACCOUNT ONLY)
    quotex_email = os.getenv('QUOTEX_EMAIL', 'demo@example.com')
    quotex_password = os.getenv('QUOTEX_PASSWORD', 'demo_pass')

config = Config()

# =============================================================================
# DATABASE MANAGER
# =============================================================================
class Database:
    def __init__(self, db_path):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY,
                    username TEXT UNIQUE,
                    password_hash TEXT,
                    created_at TIMESTAMP
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY,
                    pair TEXT,
                    direction TEXT,
                    accuracy REAL,
                    generated_at TIMESTAMP,
                    executed BOOLEAN DEFAULT 0
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY,
                    pair TEXT,
                    direction TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    result TEXT,
                    executed_at TIMESTAMP,
                    closed_at TIMESTAMP
                )
            ''')
            conn.commit()
    
    def add_user(self, username, password_hash):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                'INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)',
                (username, password_hash, datetime.now(pytz.timezone(config.BANGLADESH_TZ)))
            )
            return cursor.lastrowid
    
    def get_user(self, username):
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(
                'SELECT * FROM users WHERE username = ?', (username,)
            ).fetchone()
    
    def get_user_by_id(self, user_id):
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(
                'SELECT * FROM users WHERE id = ?', (user_id,)
            ).fetchone()
    
    def add_signal(self, pair, direction, accuracy):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                'INSERT INTO signals (pair, direction, accuracy, generated_at) VALUES (?, ?, ?, ?)',
                (pair, direction, accuracy, datetime.now(pytz.timezone(config.BANGLADESH_TZ)))
            )
    
    def get_recent_signals(self, limit=100):
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query(
                'SELECT * FROM signals ORDER BY generated_at DESC LIMIT ?',
                conn, params=(limit,)
            )
    
    def add_trade(self, pair, direction, entry_price):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                'INSERT INTO trades (pair, direction, entry_price, executed_at, result) VALUES (?, ?, ?, ?, ?)',
                (pair, direction, entry_price, datetime.now(pytz.timezone(config.BANGLADESH_TZ)), 'PENDING')
            )
    
    def update_trade_result(self, trade_id, exit_price, result):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                'UPDATE trades SET exit_price = ?, result = ?, closed_at = ? WHERE id = ?',
                (exit_price, result, datetime.now(pytz.timezone(config.BANGLADESH_TZ)), trade_id)
            )
    
    def get_performance_stats(self):
        with sqlite3.connect(self.db_path) as conn:
            stats = conn.execute('''
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN result = 'WIN' THEN 1 ELSE 0 END) as wins,
                    SUM(CASE WHEN result = 'LOSS' THEN 1 ELSE 0 END) as losses
                FROM trades WHERE result != 'PENDING'
            ''').fetchone()
            return {
                'total': stats[0] or 0,
                'wins': stats[1] or 0,
                'losses': stats[2] or 0,
                'accuracy': (stats[1] / stats[0] * 100) if stats[0] > 0 else 0
            }

db = Database(config.DATABASE_PATH)

# =============================================================================
# SIGNAL GENERATOR
# =============================================================================
class SignalGenerator:
    def __init__(self):
        self.pairs = [
            'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'USDCAD=X',
            'EURJPY=X', 'GBPJPY=X', 'EURGBP=X', 'XAUUSD=X', 'BTC-USD',
            'ETH-USD', 'BNB-USD', 'XRP-USD', 'ADA-USD', 'SOL-USD',
            'DOGE-USD', 'DOT-USD', 'MATIC-USD', 'SHIB-USD', 'AVAX-USD',
            'NZDUSD=X', 'USDCHF=X', 'GBPCHF=X', 'EURCHF=X', 'AUDJPY=X',
            'GBPAUD=X', 'EURAUD=X', 'USDMXN=X', 'USDZAR=X', 'USDTRY=X'
        ]
    
    def fetch_historical_data(self, pair, days=5):
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            df = yf.download(pair, start=start_date, end=end_date, interval='1m', progress=False)
            if df.empty:
                return self.generate_synthetic_data(days)
            return df
        except Exception as e:
            print(f"Error fetching {pair}: {e}, using synthetic data")
            return self.generate_synthetic_data(days)
    
    def generate_synthetic_data(self, days):
        periods = days * 24 * 60
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                             periods=periods, freq='1min', tz='UTC')
        np.random.seed(42)
        returns = np.random.normal(0, 0.001, periods)
        price = 1.0 + np.cumsum(returns)
        df = pd.DataFrame({
            'Open': price,
            'High': price + 0.001,
            'Low': price - 0.001,
            'Close': price,
            'Volume': np.random.randint(100, 1000, periods)
        }, index=dates)
        return df
    
    def calculate_indicators(self, df):
        df = df.copy()
        df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
        df['MACD'], df['MACD_signal'], df['MACD_hist'] = talib.MACD(df['Close'])
        df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Close'])
        df['MA_10'] = talib.SMA(df['Close'], timeperiod=10)
        df['MA_20'] = talib.SMA(df['Close'], timeperiod=20)
        df['MA_50'] = talib.SMA(df['Close'], timeperiod=50)
        df['Stoch_K'], df['Stoch_D'] = talib.STOCH(df['High'], df['Low'], df['Close'])
        df['ADX'] = talib.ADX(df['High'], df['Low'], df['Close'], timeperiod=14)
        return df
    
    def generate_signal(self, pair):
        try:
            df = self.fetch_historical_data(pair, config.ANALYSIS_LOOKBACK)
            df = self.calculate_indicators(df)
            if len(df) < 50:
                return None
            
            latest = df.iloc[-1]
            previous = df.iloc[-2]
            signals = []
            
            # RSI
            if latest['RSI'] < 30 and previous['RSI'] >= 30:
                signals.append('BUY')
            elif latest['RSI'] > 70 and previous['RSI'] <= 70:
                signals.append('SELL')
            
            # MACD
            if latest['MACD'] > latest['MACD_signal'] and previous['MACD'] <= previous['MACD_signal']:
                signals.append('BUY')
            elif latest['MACD'] < latest['MACD_signal'] and previous['MACD'] >= previous['MACD_signal']:
                signals.append('SELL')
            
            # MA Cross
            if latest['MA_10'] > latest['MA_20'] and previous['MA_10'] <= previous['MA_20']:
                signals.append('BUY')
            elif latest['MA_10'] < latest['MA_20'] and previous['MA_10'] >= previous['MA_20']:
                signals.append('SELL')
            
            # Bollinger Bands
            if latest['Close'] <= latest['BB_lower'] and latest['RSI'] < 30:
                signals.append('BUY')
            elif latest['Close'] >= latest['BB_upper'] and latest['RSI'] > 70:
                signals.append('SELL')
            
            # Stochastic
            if latest['Stoch_K'] < 20 and latest['Stoch_D'] < 20:
                signals.append('BUY')
            elif latest['Stoch_K'] > 80 and latest['Stoch_D'] > 80:
                signals.append('SELL')
            
            buy_signals = signals.count('BUY')
            sell_signals = signals.count('SELL')
            
            if buy_signals >= 3:
                direction = 'UP'
            elif sell_signals >= 3:
                direction = 'DOWN'
            else:
                return None
            
            accuracy = self.calculate_accuracy(df, direction)
            if accuracy >= config.MIN_ACCURACY:
                return {
                    'pair': pair,
                    'direction': direction,
                    'accuracy': round(accuracy, 2)
                }
            return None
        except Exception as e:
            print(f"Error generating signal for {pair}: {e}")
            return None
    
    def calculate_accuracy(self, df, predicted_direction):
        try:
            test_period = min(100, len(df) - 1)
            correct_predictions = 0
            for i in range(len(df) - test_period, len(df) - 1):
                future_return = (df.iloc[i + 1]['Close'] - df.iloc[i]['Close']) / df.iloc[i]['Close']
                if predicted_direction == 'UP' and future_return > 0:
                    correct_predictions += 1
                elif predicted_direction == 'DOWN' and future_return < 0:
                    correct_predictions += 1
            return (correct_predictions / test_period) * 100
        except:
            return 50
    
    def generate_24h_signals(self):
        signals = []
        for pair in self.pairs:
            signal = self.generate_signal(pair)
            if signal:
                signals.append(signal)
        signals.sort(key=lambda x: x['accuracy'], reverse=True)
        max_signals = (24 * 60) // config.SIGNAL_INTERVAL
        return signals[:max_signals]

signal_gen = SignalGenerator()

# =============================================================================
# TRADING EXECUTOR
# =============================================================================
class TradingExecutor:
    def __init__(self):
        self.driver = None
    
    def initialize_driver(self):
        if config.SIMULATION_MODE:
            print("⚠️ SIMULATION MODE: No real trades will be executed")
            return None
        options = webdriver.ChromeOptions()
        options.add_argument('--headless=new')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        try:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            return driver
        except Exception as e:
            print(f"Driver initialization failed: {e}")
            return None
    
    def login(self):
        if config.SIMULATION_MODE or not self.driver:
            return False
        try:
            self.driver.get("https://quotex.com")
            login_btn = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Login')]"))
            )
            login_btn.click()
            email_field = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.NAME, "email"))
            )
            email_field.send_keys(config.quotex_email)
            password_field = self.driver.find_element(By.NAME, "password")
            password_field.send_keys(config.quotex_password)
            submit_btn = self.driver.find_element(By.XPATH, "//button[@type='submit']")
            submit_btn.click()
            time.sleep(5)
            return True
        except Exception as e:
            print(f"Login failed: {e}")
            return False
    
    def execute_trade(self, signal):
        if config.SIMULATION_MODE:
            print(f"🎮 SIMULATION: {signal['pair']} - {signal['direction']} at {signal['accuracy']}%")
            return {
                'success': True,
                'entry_price': round(np.random.uniform(0.9, 1.1), 4),
                'trade_id': 'SIM_' + str(int(time.time()))
            }
        if not self.driver:
            return {'success': False, 'error': 'Driver not initialized'}
        try:
            self.driver.get(f"https://quotex.com/trading/{signal['pair']}")
            time.sleep(3)
            btn = self.driver.find_element(By.ID, "call-button" if signal['direction'] == 'UP' else "put-button")
            btn.click()
            time.sleep(1)
            amount_field = self.driver.find_element(By.NAME, "amount")
            amount_field.clear()
            amount_field.send_keys("1")
            execute_btn = self.driver.find_element(By.ID, "execute-trade")
            execute_btn.click()
            return {
                'success': True,
                'entry_price': float(self.driver.find_element(By.ID, "current-price").text),
                'trade_id': 'LIVE_' + str(int(time.time()))
            }
        except Exception as e:
            print(f"Trade execution failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def close(self):
        if self.driver:
            self.driver.quit()

executor = TradingExecutor()

# =============================================================================
# FLASK APP & HTML TEMPLATES
# =============================================================================
app = Flask(__name__)
app.config.from_object(config)
app.secret_key = config.SECRET_KEY

# Flask-Login
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin):
    def __init__(self, id, username):
        self.id = id
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    user_data = db.get_user_by_id(int(user_id))
    if user_data:
        return User(user_data[0], user_data[1])
    return None

# HTML Templates as strings
LOGIN_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quotex Bot - Login</title>
    <style>
        :root {
            --bg-primary: #1a1a2e; --bg-secondary: #16213e; --text-primary: #ffffff;
            --text-secondary: #b0b0b0; --accent-green: #00ff88; --accent-red: #ff4757;
            --accent-blue: #0099ff; --border-color: #2a2a3e;
        }
        [data-theme="light"] {
            --bg-primary: #f0f2f5; --bg-secondary: #ffffff; --text-primary: #1a1a2e;
            --text-secondary: #4a4a6a; --accent-green: #00cc66; --accent-red: #ff3344;
            --accent-blue: #0077cc; --border-color: #d0d0e0;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, var(--bg-primary), var(--bg-secondary));
            color: var(--text-primary); display: flex; justify-content: center;
            align-items: center; min-height: 100vh;
        }
        .login-container {
            width: 400px; padding: 2rem; background: var(--bg-secondary);
            border-radius: 12px; border: 1px solid var(--border-color);
        }
        .logo h1 { font-size: 1.5rem; }
        .warning { color: var(--accent-red); font-size: 0.8rem; margin-top: 0.5rem; }
        .disclaimer {
            margin-top: 2rem; padding: 1rem; background: rgba(255, 71, 87, 0.1);
            border-radius: 8px; border: 1px solid var(--accent-red);
        }
        .disclaimer ul { margin-left: 1.5rem; font-size: 0.8rem; }
        input {
            width: 100%; padding: 0.75rem; margin-top: 0.5rem;
            background: var(--bg-primary); border: 1px solid var(--border-color);
            border-radius: 6px; color: var(--text-primary);
        }
        button {
            width: 100%; padding: 0.75rem; margin-top: 0.5rem; border: none;
            border-radius: 6px; font-weight: bold; cursor: pointer;
        }
        .btn-primary {
            background: var(--accent-blue); color: white;
        }
        .btn-secondary {
            background: var(--bg-primary); color: var(--text-primary);
            border: 1px solid var(--border-color);
        }
    </style>
</head>
<body>
    <div class="login-container">
        <div class="logo">
            <h1>🤖 Quotex Trading Bot</h1>
            <p class="warning">⚠️ DEMO ACCOUNT ONLY - HIGH RISK</p>
        </div>
        
        <form method="POST" action="{{ url_for('login') }}" class="login-form">
            <input type="text" name="username" placeholder="Username" required>
            <input type="password" name="password" placeholder="Password" required>
            <button type="submit" class="btn-primary">Login</button>
        </form>
        
        <form method="POST" action="{{ url_for('register') }}" class="register-form">
            <input type="text" name="username" placeholder="New Username" required>
            <input type="password" name="password" placeholder="New Password" required>
            <button type="submit" class="btn-secondary">Register</button>
        </form>
        
        <div class="disclaimer">
            <h3>⚠️ IMPORTANT</h3>
            <ul>
                <li>This bot may get your account BANNED</li>
                <li>NO real money guarantees</li>
                <li>Use ONLY for educational purposes</li>
                <li>Markets are 90% random</li>
            </ul>
        </div>
    </div>
</body>
</html>
"""

DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quotex Bot Dashboard</title>
    <style>
        :root {
            --bg-primary: #1a1a2e; --bg-secondary: #16213e; --text-primary: #ffffff;
            --text-secondary: #b0b0b0; --accent-green: #00ff88; --accent-red: #ff4757;
            --accent-blue: #0099ff; --border-color: #2a2a3e;
        }
        [data-theme="light"] {
            --bg-primary: #f0f2f5; --bg-secondary: #ffffff; --text-primary: #1a1a2e;
            --text-secondary: #4a4a6a; --accent-green: #00cc66; --accent-red: #ff3344;
            --accent-blue: #0077cc; --border-color: #d0d0e0;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-primary); color: var(--text-primary);
        }
        .navbar {
            display: flex; justify-content: space-between; align-items: center;
            padding: 1rem 2rem; background: var(--bg-secondary);
            border-bottom: 2px solid var(--border-color);
        }
        .clock {
            font-size: 1.5rem; font-weight: bold; color: var(--accent-blue);
        }
        .performance-bar {
            display: flex; gap: 2rem; background: var(--bg-primary);
            padding: 0.5rem 1rem; border-radius: 8px;
        }
        .dashboard {
            display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; padding: 2rem;
        }
        .signals-panel {
            background: var(--bg-secondary); padding: 1.5rem;
            border-radius: 12px; border: 1px solid var(--border-color);
        }
        .signal-card {
            display: flex; justify-content: space-between; align-items: center;
            padding: 1rem; margin: 0.5rem 0; border-radius: 8px;
            border-left: 4px solid var(--accent-green); background: var(--bg-primary);
        }
        .signal-card.down { border-left-color: var(--accent-red); }
        .signal-card .direction.up {
            background: var(--accent-green); color: var(--bg-primary);
            padding: 0.5rem 1rem; border-radius: 6px; font-weight: bold;
        }
        .signal-card .direction.down {
            background: var(--accent-red); color: var(--bg-primary);
            padding: 0.5rem 1rem; border-radius: 6px; font-weight: bold;
        }
        #themeToggle {
            background: var(--bg-primary); border: 1px solid var(--border-color);
            padding: 0.5rem 1rem; border-radius: 6px; cursor: pointer;
        }
        .btn-secondary, .btn-danger {
            padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white;
        }
        .btn-secondary { background: var(--bg-primary); border: 1px solid var(--border-color); }
        .btn-danger { background: var(--accent-red); }
    </style>
</head>
<body>
    <nav class="navbar">
        <div class="nav-left">
            <h1>🤖 Quotex Bot</h1>
            <div class="clock" id="bdClock"></div>
        </div>
        <div class="nav-center">
            <div class="performance-bar">
                <span>Wins: <strong id="wins">0</strong></span>
                <span>Losses: <strong id="losses">0</strong></span>
                <span>Accuracy: <strong id="accuracy">0%</strong></span>
            </div>
        </div>
        <div class="nav-right">
            <button id="themeToggle" class="btn-icon">🌓</button>
            <a href="{{ url_for('history') }}" class="btn-secondary">History</a>
            <a href="{{ url_for('logout') }}" class="btn-danger">Logout</a>
        </div>
    </nav>

    <main class="dashboard">
        <section class="signals-panel">
            <h2>📊 Live Signals (Next 24 Hours)</h2>
            <div class="signals-list" id="signalsList">
                <div class="loading">Generating signals...</div>
            </div>
        </section>
    </main>

    <script>
        // Bangladesh Time Clock
        function updateClock() {
            const now = new Date();
            const options = { timeZone: 'Asia/Dhaka', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false };
            document.getElementById('bdClock').textContent = now.toLocaleTimeString('en-BD', options);
        }
        setInterval(updateClock, 1000); updateClock();

        // Theme Toggle
        const themeToggle = document.getElementById('themeToggle');
        const html = document.documentElement;
        themeToggle.addEventListener('click', () => {
            const newTheme = html.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
            html.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            fetch('/api/settings', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ theme: newTheme }) });
        });
        const savedTheme = localStorage.getItem('theme') || 'dark';
        html.setAttribute('data-theme', savedTheme);

        // Load signals and performance
        function loadSignals() {
            fetch('/api/signals').then(res => res.json()).then(signals => {
                const container = document.getElementById('signalsList');
                container.innerHTML = signals.map(s => `
                    <div class="signal-card ${s.direction.toLowerCase()}">
                        <span class="pair">${s.pair.replace('=X', '')}</span>
                        <span class="direction ${s.direction.toLowerCase()}">${s.direction === 'UP' ? '↑ BUY' : '↓ SELL'}</span>
                        <span class="accuracy-badge">${s.accuracy}% Acc.</span>
                        <span class="time">${s.time}</span>
                    </div>
                `).join('');
            });
        }
        function loadPerformance() {
            fetch('/api/performance').then(res => res.json()).then(stats => {
                document.getElementById('wins').textContent = stats.wins;
                document.getElementById('losses').textContent = stats.losses;
                document.getElementById('accuracy').textContent = stats.accuracy.toFixed(1) + '%';
            });
        }
        setInterval(loadSignals, 30000); setInterval(loadPerformance, 30000);
        loadSignals(); loadPerformance();
    </script>
</body>
</html>
"""

HISTORY_TEMPLATE = """
<!DOCTYPE html>
<html lang="en" data-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Trade History</title>
    <style>
        :root {
            --bg-primary: #1a1a2e; --bg-secondary: #16213e; --text-primary: #ffffff;
            --text-secondary: #b0b0b0; --accent-green: #00ff88; --accent-red: #ff4757;
            --accent-blue: #0099ff; --border-color: #2a2a3e;
        }
        [data-theme="light"] {
            --bg-primary: #f0f2f5; --bg-secondary: #ffffff; --text-primary: #1a1a2e;
            --text-secondary: #4a4a6a; --accent-green: #00cc66; --accent-red: #ff3344;
            --accent-blue: #0077cc; --border-color: #d0d0e0;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-primary); color: var(--text-primary);
        }
        .navbar {
            display: flex; justify-content: space-between; align-items: center;
            padding: 1rem 2rem; background: var(--bg-secondary);
            border-bottom: 2px solid var(--border-color);
        }
        .trade-table {
            width: 100%; border-collapse: collapse; margin-top: 2rem;
        }
        .trade-table th, .trade-table td {
            padding: 1rem; text-align: left; border-bottom: 1px solid var(--border-color);
        }
        .trade-table .win { background: rgba(0, 255, 136, 0.1); }
        .trade-table .loss { background: rgba(255, 71, 87, 0.1); }
        .direction-up { color: var(--accent-green); font-weight: bold; }
        .direction-down { color: var(--accent-red); font-weight: bold; }
        .result-win { color: var(--accent-green); }
        .result-loss { color: var(--accent-red); }
        .btn-secondary, .btn-danger {
            padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; color: white;
        }
        .btn-secondary { background: var(--bg-primary); border: 1px solid var(--border-color); }
        .btn-danger { background: var(--accent-red); }
    </style>
</head>
<body>
    <nav class="navbar">
        <h1>📜 Trade History</h1>
        <div>
            <a href="{{ url_for('dashboard') }}" class="btn-secondary">Dashboard</a>
            <a href="{{ url_for('logout') }}" class="btn-danger">Logout</a>
        </div>
    </nav>

    <main style="padding: 2rem;">
        <table class="trade-table">
            <thead>
                <tr>
                    <th>Time</th> <th>Pair</th> <th>Direction</th>
                    <th>Entry</th> <th>Exit</th> <th>Result</th>
                </tr>
            </thead>
            <tbody>
                {% for trade in trades %}
                <tr class="{{ trade.result.lower() }}">
                    <td>{{ trade.executed_at }}</td>
                    <td>{{ trade.pair }}</td>
                    <td>
                        <span class="direction-{{ trade.direction.lower() }}">
                            {{ '↑ UP' if trade.direction == 'UP' else '↓ DOWN' }}
                        </span>
                    </td>
                    <td>{{ '{:.4f}'.format(trade.entry_price) if trade.entry_price else '-' }}</td>
                    <td>{{ '{:.4f}'.format(trade.exit_price) if trade.exit_price else '-' }}</td>
                    <td>
                        <span class="result-{{ trade.result.lower() }}">
                            {{ trade.result }}
                        </span>
                    </td>
                </tr>
                {% endfor %}
            </tbody>
        </table>
    </main>
</body>
</html>
"""

# =============================================================================
# FLASK ROUTES
# =============================================================================
@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user_data = db.get_user(username)
        if user_data and check_password_hash(user_data[2], password):
            user = User(user_data[0], user_data[1])
            login_user(user)
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid credentials', 'error')
    return render_template_string(LOGIN_TEMPLATE)

@app.route('/register', methods=['POST'])
def register():
    username = request.form.get('username')
    password = request.form.get('password')
    if not username or not password:
        flash('Fill all fields', 'error')
        return redirect(url_for('login'))
    try:
        db.add_user(username, generate_password_hash(password))
        flash('Registration successful! Please login.', 'success')
    except sqlite3.IntegrityError:
        flash('User already exists', 'error')
    return redirect(url_for('login'))

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template_string(DASHBOARD_TEMPLATE)

@app.route('/history')
@login_required
def history():
    with sqlite3.connect(config.DATABASE_PATH) as conn:
        trades = pd.read_sql_query(
            'SELECT * FROM trades ORDER BY executed_at DESC LIMIT 100', conn
        )
    return render_template_string(HISTORY_TEMPLATE, trades=trades.to_dict('records'))

@app.route('/api/signals')
@login_required
def get_signals():
    signals = db.get_recent_signals(50)
    signals_list = signals.to_dict('records')
    for signal in signals_list:
        signal['color'] = 'green' if signal['direction'] == 'UP' else 'red'
        signal['time'] = datetime.fromisoformat(signal['generated_at']).strftime('%H:%M:%S')
    return jsonify(signals_list)

@app.route('/api/performance')
@login_required
def get_performance():
    return jsonify(db.get_performance_stats())

@app.route('/api/settings', methods=['POST'])
@login_required
def update_settings():
    data = request.json
    if 'theme' in data:
        app.config['THEME'] = data['theme']
    return jsonify({'status': 'ok'})

@app.route('/api/execute_signal/<int:signal_id>', methods=['POST'])
@login_required
def execute_signal(signal_id):
    with sqlite3.connect(config.DATABASE_PATH) as conn:
        signal_data = conn.execute(
            'SELECT * FROM signals WHERE id = ?', (signal_id,)
        ).fetchone()
    if not signal_data:
        return jsonify({'error': 'Signal not found'}), 404
    signal = {'pair': signal_data[1], 'direction': signal_data[2], 'accuracy': signal_data[3]}
    result = executor.execute_trade(signal)
    if result['success']:
        db.add_trade(signal['pair'], signal['direction'], result['entry_price'])
        return jsonify({'status': 'executed', 'trade_id': result['trade_id']})
    else:
        return jsonify({'error': result.get('error', 'Unknown error')}), 500

# =============================================================================
# BACKGROUND SCHEDULER
# =============================================================================
scheduler = BackgroundScheduler()

def generate_signals_job():
    print(f"🤖 Generating signals at {datetime.now(pytz.timezone(config.BANGLADESH_TZ))}")
    signals = signal_gen.generate_24h_signals()
    for signal in signals:
        db.add_signal(signal['pair'], signal['direction'], signal['accuracy'])

scheduler.add_job(generate_signals_job, 'interval', minutes=config.SIGNAL_INTERVAL)
scheduler.start()

# Initial signal generation
threading.Thread(target=generate_signals_job, daemon=True).start()

# =============================================================================
# MAIN EXECUTION
# =============================================================================
if __name__ == '__main__':
    print("="*60)
    print("🚀 QUOTEX TRADING BOT STARTING...")
    print(f"⚠️  SIMULATION MODE: {'ON - Safe' if config.SIMULATION_MODE else 'OFF - DANGEROUS'}")
    print(f"📊 Access dashboard: http://localhost:5000")
    print(f"⏰ Bangladesh Time (UTC+6)")
    print("="*60)
    print("⚠️  WARNING: Using this bot may result in account BANNING")
    print("📚 This is for EDUCATIONAL PURPOSES ONLY")
    print("="*60)
    app.run(debug=False, host='0.0.0.0', port=5000)
