# Bot Trading — Scalping & Swing (Binance Futures)

Repo ini berisi:

* **backtester\_scalping.py** (selaras real-trading)
* **backtester\_swing.py**
* **papertrade.py** (live data tanpa order)
* **newrealtrading.py** (real trading ke Binance)
* **engine\_core.py** (helper indikator & money management)
* **ml\_signal\_plugin.py** (opsional)
* **coin\_config.json** (per‑symbol tuning, lot/precision)

> Target strategi: WR ≥ 75%, PF > 2 untuk mode scalping/swing (dicapai lewat kombinasi filter ATR, body/ATR, ML gate, trailing, BE, time stop, dll.).

---

## 1) Persiapan

### Requirements

* Python **3.10–3.12** (disarankan 3.11)
* Paket (lihat `requirements.txt`): `pandas numpy ta scikit-learn python-binance streamlit matplotlib plotly python-dotenv ujson filelock websockets`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### ENV (contoh `.env`)

```
BINANCE_API_KEY=YOUR_KEY
BINANCE_API_SECRET=YOUR_SECRET
INSTANCE_ID=botA
# Opsional ML & scoring
USE_ML=1
SCORE_THRESHOLD=1.2
ML_MIN_TRAIN_BARS=400
ML_RETRAIN_EVERY=5000
```

> **Catatan**: `INSTANCE_ID` dipakai untuk `newClientOrderId` supaya aman multi‑bot di satu server/IP.

---

## 2) Backtester

### Scalping

```bash
streamlit run backtester_scalping.py --server.port 8501
```

* Pilih file CSV OHLCV di sidebar (kolom wajib: timestamp/open/high/low/close/volume).
* **use\_htf\_filter** default **OFF** untuk eksplorasi awal.
* Mode **Debug** tersedia untuk melihat alasan suatu bar terblokir.

### Swing

```bash
streamlit run backtester_swing.py --server.port 8502
```

---

## 3) Dry‑run (replay CSV seperti live)

```bash
python tools_dryrun_summary.py \
  --symbol ADAUSDT \
  --csv data/ADAUSDT_15m_2025-06-01_to_2025-08-09.csv \
  --coin_config coin_config.json \
  --steps 500 --balance 20 \
  --out ADA_dryrun_trades_500.csv
```

Output: ringkasan WR, PF, avg PnL, dan CSV trade detail.

---

## 4) Papertrade (live data, tanpa order)

```bash
python papertrade.py \
  --live-paper \
  --symbols ADAUSDT,DOGEUSDT,XRPUSDT \
  --interval 15m \
  --balance 20 \
  --coin_config coin_config.json \
  --instance-id PAPER01 \
  --verbose
```

* Bot menarik **klines futures** via REST, memproses sinyal, **mencatat entry/exit & PnL** ke CSV di folder `logs/<INSTANCE_ID>`.
* Ukuran posisi memperhitungkan `stepSize`, `minQty`, `minNotional` (otomatis di‑inject bila perlu).

---

## 5) Real Trading (Binance Futures USDT‑M)

> **Uji di testnet lebih dulu!**

```bash
python newrealtrading.py \
  --live \
  --symbols ADAUSDT,DOGEUSDT,XRPUSDT \
  --real-exec \
  --testnet \
  --instance-id botA \
  --account-guard \
  --verbose
```

* **Sumber balance**: `availableBalance` dari akun futures, **bukan** argumen `--balance`.
* Jika **margin tidak cukup**, bot otomatis **shrink qty** sesuai saldo & `minNotional`.
* **Account Guard**: bila posisi masih ada di akun, bot **skip entry baru** untuk simbol yang sama.
* **Time Stop** dan **ROI minimal** sesuai `coin_config.json` (`max_hold_seconds`, `min_roi_to_close_by_time`).

---

## 6) Multi‑Bot dalam 1 Server/IP

Jalankan dua kontainer/proses dengan **AKUN/APIKEY berbeda** atau sama (asal `INSTANCE_ID` beda). Pastikan direktori kerja/log terpisah.

**Contoh** (dua folder):

```
/var/www/realtrading   -> INSTANCE_ID=botA, logs: logsA/
/var/www/realtradingb  -> INSTANCE_ID=botB, logs: logsB/
```

**Contoh Docker**:

```bash
# Build
docker build -t analisa-bot:latest .

# Bot A
docker run -d --name realtradingA \
  -e BINANCE_API_KEY=KEY_A -e BINANCE_API_SECRET=SEC_A \
  -e INSTANCE_ID=botA -e USE_ML=1 -e SCORE_THRESHOLD=1.2 \
  -v $(pwd)/logsA:/app/logs \
  --restart unless-stopped \
  analisa-bot:latest \
  python newrealtrading.py --live --symbols ADAUSDT,DOGEUSDT,XRPUSDT --real-exec --account-guard --verbose

# Bot B
docker run -d --name realtradingB \
  -e BINANCE_API_KEY=KEY_B -e BINANCE_API_SECRET=SEC_B \
  -e INSTANCE_ID=botB -e USE_ML=1 -e SCORE_THRESHOLD=1.2 \
  -v $(pwd)/logsB:/app/logs \
  --restart unless-stopped \
  analisa-bot:latest \
  python newrealtrading.py --live --symbols ADAUSDT,DOGEUSDT,XRPUSDT --real-exec --account-guard --verbose
```

> Dengan mekanisme **shared `availableBalance`** di `TradingManager.run_once`, ketika menjalankan multi‑symbol, porsi modal akan **berkurang berantai** per entry: contoh saldo \$20, risk 8% → entry1 \$1.6, sisa \$18.4; entry2 \$1.47, sisa \$16.93; entry3 \$1.35, dst.

---

## 7) Tuning coin\_config.json

* **Lot/precision**: `stepSize`, `minQty`, `quantityPrecision`, `minNotional` → wajib sesuai exchange. `papertrade.py` dapat **meng‑inject otomatis** via `futures_exchange_info`.
* **Risk & Leverage**: `risk_per_trade`, `leverage`.
* **Filter**: `min_atr_pct`, `max_atr_pct`, `max_body_atr`, `use_htf_filter`.
* **Exit**: `sl_mode`, `sl_pct`/`sl_atr_mult`, `sl_min_pct`, `sl_max_pct`, `use_breakeven`, `be_trigger_pct`, `trailing_trigger`, `trailing_step`, `max_hold_seconds`, `min_roi_to_close_by_time`.
* **ML**: `ml.score_threshold` atau ENV `SCORE_THRESHOLD`.

---

## 8) Troubleshooting

* **APIError -2019 (Margin is insufficient)**: bot akan mengecilkan qty otomatis. Pastikan `minNotional` tercapai; naikkan leverage atau saldo.
* **Pylance override error**: pastikan signature override sama persis (lihat patch di bawah).
* **Duplikat orderId**: bedakan `INSTANCE_ID` tiap bot.
* **Koneksi/testnet**: set `--testnet` dan cek `FUTURES_URL` diarahkan ke testnet.

---

## 9) Dockerfile

```dockerfile
# Dockerfile (Python 3.11, slim)
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

# System deps (opsional: timezone, build tools untuk wheel fallback)
RUN apt-get update && apt-get install -y --no-install-recommends \
      tzdata build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Default: hanya shell. Jalankan per‑mode via `docker run ... python <script>.py`.
CMD ["bash"]
```

---

## 10) Patch Ringkas (lihat juga komentar PR)

* **papertrade.JournaledCoinTrader.\_size\_position** → samakan signature dengan `CoinTrader._size_position(price, sl, balance)`.
* **newrealtrading.TradingManager.run\_once** → cek `exec is not None` sebelum `get_available_balance()`.
* **engine\_core.py** → alias `body_atr` sudah ada supaya kompatibel kode lama.

---

## 11) Keamanan

* Simpan API key di `.env`/secret manager, **jangan commit**.
* Uji dengan `--testnet` sebelum real; aktifkan `--account-guard`.
* Pasang **cooldown** dan **SL min/max clamp** untuk menghindari sizing ekstrem.

Selamat membangun & happy trading! 🚀
