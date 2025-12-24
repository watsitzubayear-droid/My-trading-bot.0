from flask import Flask, render_template, jsonify, request, send_file
from signal_engine import SignalEngine
from market_data import MarketDataSimulator
from utils import get_bdt_time, generate_future_times
import pandas as pd
import io
import csv

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate_signals', methods=['POST'])
def generate_signals():
    data = request.json
    market_type = data.get('market_type', 'real')
    pair = data.get('pair', 'EUR/USD')
    num_signals = data.get('num_signals', 100)
    
    # Initialize components
    market_data = MarketDataSimulator(market_type=market_type)
    signal_engine = SignalEngine(market_type=market_type)
    
    # Get historical data for training
    candles = market_data.generate_candles(200)
    
    # Generate future times (next 100 minutes)
    start_time = get_bdt_time()
    future_times = generate_future_times(start_time, num_signals)
    
    # Generate signals
    signals = signal_engine.generate_signals(candles, future_times, pair)
    
    return jsonify({
        'success': True,
        'market_type': market_type,
        'pair': pair,
        'generated_signals': len(signals),
        'signals': signals,
        'server_time': start_time.strftime("%Y-%m-%d %H:%M:%S BDT")
    })

@app.route('/download_signals', methods=['POST'])
def download_signals():
    data = request.json
    signals = data.get('signals', [])
    
    # Create CSV in memory
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Headers
    writer.writerow(['Pair', 'Signal Time (BDT)', 'Direction', 'Confidence', 'Formatted Signal'])
    
    # Data
    for signal in signals:
        writer.writerow([
            signal['pair'],
            signal['time'],
            signal['direction'],
            f"{signal['confidence']:.2%}",
            signal['formatted']
        ])
    
    # Prepare file
    output.seek(0)
    return send_file(
        io.BytesIO(output.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f"zoha_signals_{get_bdt_time().strftime('%Y%m%d_%H%M%S')}.csv"
    )

@app.route('/get_server_time')
def get_server_time():
    return jsonify({
        'bdt_time': get_bdt_time().strftime("%Y-%m-%d %H:%M:%S BDT")
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
