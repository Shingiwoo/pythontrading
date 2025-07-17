1. Cara Menjalankan
Ekstrak zip:
unzip scalping_strategy_package.zip

2. cd scalping_strategy
Instal dependensi:
(Sebaiknya gunakan virtual environment)
pip install -r requirements.txt

2. Jalankan GUI Streamlit:
- buat folder data dan report
Masukkan file CSV ke folder /data 

streamlit run app.py

Format CSV harus memiliki kolom: timestamp, open, high, low, close, volume
Contoh nama file: DOGEUSDT_15m_2025-07-01_to_2025-07-14.csv
