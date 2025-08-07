FITUR TAMBAHAN YANG TELAH DITERAPKAN
✅ ML Signal via RandomForest (prediksi kenaikan harga)
✅ Candlestick Chart Interaktif menggunakan Plotly
✅ Integrasi Binance API (testnet & live mode)

📋 PANDUAN PENGGUNAAN
🔧 Instalasi
pip install -r requirements.txt

🛠️ API Binance (opsional)
Buat file .env di root:

BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
BINANCE_TESTNET=true  # atau false

Jalankan GUI Streamlit:
- buat folder data dan report 
Masukkan file CSV ke folder /data 

streamlit run app.py
Sidebar akan menyediakan tombol:

🔄 Sync Balance
📈 Place Order (testnet/live)

Format CSV harus memiliki kolom: timestamp, open, high, low, close, volume
Contoh nama file: DOGEUSDT_15m_2025-07-01_to_2025-07-14.csv
