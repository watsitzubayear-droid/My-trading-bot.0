from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import pandas_ta as ta
import time
import requests

# --- SETTINGS ---
TELEGRAM_TOKEN = "YOUR_TOKEN"
CHAT_ID = "YOUR_ID"

class QuotexInfinityBot:
    def __init__(self):
        options = Options()
        options.add_argument("--headless") # Runs in background
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/122.0.0.0")
        
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        self.price_history = {} # Stores data for all 60 markets

    def send_alert(self, msg):
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage?chat_id={CHAT_ID}&text={msg}&parse_mode=Markdown"
        requests.get(url)

    def get_market_data(self, symbol):
        """
        Logic to navigate to specific Quotex OTC pair and scrape the current price.
        Note: You must be logged in to access OTC charts.
        """
        try:
            # Navigate to the specific asset URL (Example structure)
            self.driver.get(f"https://qxbroker.com/en/trade/{symbol}")
            time.sleep(2) # Wait for chart to load
            
            # Extract price from the Quotex DOM (Inspect the element to get exact class)
            price_element = self.driver.find_element("class name", "current-price-value") 
            return float(price_element.text)
        except:
            return None

    def analyze(self, symbol, current_price):
        if symbol not in self.price_history:
            self.price_history[symbol] = []
        
        self.price_history[symbol].append(current_price)
        
        # We need at least 20 periods for Bollinger Bands
        if len(self.price_history[symbol]) > 200:
            df = pd.DataFrame(self.price_history[symbol], columns=['close'])
            
            # INDICATORS
            df['EMA_200'] = ta.ema(df['close'], length=200)
            df['RSI'] = ta.rsi(df['close'], length=14)
            bb = ta.bbands(df['close'], length=20, std=2.5)
            
            last = df.iloc[-1]
            
            # --- HIGH ACCURACY OTC RULE ---
            if last['close'] > last['EMA_200'] and last['RSI'] < 30:
                self.send_alert(f"🚀 *QUOTEX BUY ALERT*\nAsset: {symbol}\nPrice: {last['close']}")

    def run(self):
        symbols = ["EURUSD_otc", "GBPUSD_otc", "GOLD_otc", "APPLE_otc"] # List 60+ here
        while True:
            for s in symbols:
                p = self.get_market_data(s)
                if p: self.analyze(s, p)
            time.sleep(5)

if __name__ == "__main__":
    bot = QuotexInfinityBot()
    bot.run()
