document.addEventListener('DOMContentLoaded', function() {
    const generateBtn = document.getElementById('generate-btn');
    const downloadBtn = document.getElementById('download-btn');
    const signalsTbody = document.getElementById('signals-tbody');
    const loading = document.getElementById('loading');
    const status = document.getElementById('status');
    const serverTimeEl = document.getElementById('server-time');
    
    let currentSignals = [];
    
    // Update server time every second
    function updateServerTime() {
        fetch('/get_server_time')
            .then(r => r.json())
            .then(data => {
                serverTimeEl.textContent = data.bdt_time;
            });
    }
    setInterval(updateServerTime, 1000);
    updateServerTime();
    
    // Generate signals
    generateBtn.addEventListener('click', async () => {
        loading.classList.remove('hidden');
        status.className = 'status hidden';
        
        const marketType = document.getElementById('market-type').value;
        const pair = document.getElementById('pair').value;
        const numSignals = parseInt(document.getElementById('num-signals').value);
        
        try {
            const response = await fetch('/generate_signals', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ market_type: marketType, pair, num_signals: numSignals })
            });
            
            if (!response.ok) throw new Error('Failed to generate signals');
            
            const data = await response.json();
            
            if (data.success) {
                currentSignals = data.signals;
                displaySignals(currentSignals);
                
                status.textContent = `✅ Generated ${data.generated_signals} signals for ${data.pair} (${data.market_type})`;
                status.className = 'status success';
            } else {
                throw new Error('Signal generation failed');
            }
        } catch (error) {
            status.textContent = `❌ Error: ${error.message}`;
            status.className = 'status error';
        } finally {
            loading.classList.add('hidden');
        }
    });
    
    // Display signals in table
    function displaySignals(signals) {
        signalsTbody.innerHTML = '';
        
        signals.forEach((signal, index) => {
            const row = document.createElement('tr');
            const directionClass = signal.direction === 'LONG' ? 'direction-long' : 'direction-short';
            
            row.innerHTML = `
                <td>${signal.pair}</td>
                <td>${new Date(signal.time).toLocaleTimeString('en-BD', { timeZone: 'Asia/Dhaka' })}</td>
                <td class="${directionClass}">${signal.direction}</td>
                <td>${(signal.confidence * 100).toFixed(1)}%</td>
                <td class="neon-yellow">${signal.formatted}</td>
            `;
            
            // Add flicker effect every 10th row
            if (index % 10 === 0) {
                row.style.animation = 'flicker 2s infinite';
            }
            
            signalsTbody.appendChild(row);
        });
    }
    
    // Download signals
    downloadBtn.addEventListener('click', async () => {
        if (currentSignals.length === 0) {
            status.textContent = '❌ No signals to download. Generate signals first.';
            status.className = 'status error';
            return;
        }
        
        loading.classList.remove('hidden');
        
        try {
            const response = await fetch('/download_signals', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ signals: currentSignals })
            });
            
            if (!response.ok) throw new Error('Download failed');
            
            // Create download link
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = response.headers.get('Content-Disposition').split('filename=')[1];
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(url);
            
            status.textContent = '✅ Signals downloaded successfully!';
            status.className = 'status success';
        } catch (error) {
            status.textContent = `❌ Download error: ${error.message}`;
            status.className = 'status error';
        } finally {
            loading.classList.add('hidden');
        }
    });
});

