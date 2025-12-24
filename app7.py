import pandas as pd
import pandas_ta as ta
import numpy as np
from concurrent.futures import ThreadPoolExecutor

# --- PSYCHOLOGY & RISK CONFIG ---
MAX_DAILY_LOSS = -0.05  # Stop bot if 5% of balance is lost
PROFIT_TARGET = 0.08    # Stop bot if 8% profit is reached
STREAK_LIMIT = 3        # Stop if 3 consecutive losses occur

# --- 60+ MARKET LIST (SAMPLES) ---
MARKETS = ["EURUSD_OTC", "GBPUSD_OTC", "GOLD_OTC", "AAPL_OTC", "TSLA_OTC", "US30_OTC"] # Add all 60+ here

def get_signal(df):
    """Refined strategy logic for OTC markets."""
    # 1. Indicators
    df['EMA_200'] = ta.ema(df['close'], length=200)
    df['RSI'] = ta.rsi(df['close'], length=14)
    bb = ta.bbands(df['close'], length=20, std=2.5)
    
    # 2. Pattern Recognition (Pin Bar / Hammer)
    # Long wick at the bottom = rejection of lower prices
    df['lower_wick'] = (df[['open','close']].min(axis=1) - df['low'])
    df['body_size'] = abs(df['close'] - df['open'])
    is_hammer = (df['lower_wick'] > df['body_size'] * 2)

    # 3. CONFLUENCE SIGNAL
    last = df.iloc[-1]
    
    # Buy Signal: Above EMA 200 + RSI < 30 + Price at Lower BB + Hammer Pattern
    if (last['close'] > last['EMA_200'] and 
        last['RSI'] < 35 and 
        last['close'] <= bb['BBL_20_2.5'].iloc[-1] and 
        is_hammer.iloc[-1]):
        return "STRONG_BUY"
        
    return "NEUTRAL"

def scan_all_markets(market_list):
    print(f"Scanning {len(market_list)} markets...")
    with ThreadPoolExecutor(max_workers=10) as executor:
        # In a real app, replace 'fetch_data' with your broker's API call
        executor.map(process_market, market_list)

def process_market(symbol):
    # Simulated execution
    # df = fetch_data(symbol)
    # signal = get_signal(df)
    # if signal != "NEUTRAL": print(f"ALERT: {symbol} -> {signal}")
    pass
