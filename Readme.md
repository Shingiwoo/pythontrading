# Bot Trading — Binance Futures (Scalping & Swing)

> **Tujuan:** Build bot trading modular untuk Binance Futures dengan jalur kerja: **Backtest → Dry‑Run → Papertrade (live data, tanpa order) → Real Trading**. Target strategi: WR ≥ 75% dan Profit Factor > 2 (dicapai lewat iterasi parameter & filter). File utama: `backtester_scalping.py`, `papertrade.py`, `newrealtrading.py`, `ml_signal_plugin.py`.

---

## 1) Struktur Proyek

```
.
├── backtester_scalping.py     # Streamlit backtester (scalping) — sejalan logic real-trading
├── backtester_swing.py        # Streamlit backtester (swing)
├── tools_dryrun_summary.py    # Dry‑run bar-by-bar pakai logic real (ringkasan WR/PF)
├── papertrade.py              # Live data Binance — jurnal CSV — TANPA order
├── newrealtrading.py          # Real trading ke Binance Futures (order nyata)
├── ml_signal_plugin.py        # Plugin ML (RandomForest)
├── engine_core.py             # Helper indikator, sizing/pnl util, filter & journaling
├── coin_config.json           # Konfigurasi per‑symbol (MM, filter, SL/TS/BE, presisi)
├── requirements.txt
├── Dockerfile
└── data/                      # (opsional) CSV OHLCV untuk backtest/dry‑run
```

---

## 2) Instalasi (Local)

### Python

* **Python 3.10+** disarankan (3.11 OK).
* Install deps terbaru:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -U -r requirements.txt
```

### Environment (.env)

Buat `.env` dari contoh berikut (gunakan API testnet dulu untuk uji real‑trading aman):

```env
# --- Binance ---
BINANCE_API_KEY=your_key
BINANCE_API_SECRET=your_secret
# set TRUE untuk testnet saat real trading (opsi --testnet)
# INSTANCE_ID unik per bot utk newClientOrderId
INSTANCE_ID=BOT_A

# --- ML default ---
USE_ML=1
SCORE_THRESHOLD=1.2
ML_MIN_TRAIN_BARS=400
ML_RETRAIN_EVERY=5000
ML_UP_PROB=0.55
ML_DOWN_PROB=0.45

# --- Lainnya ---
VERBOSE=1
```

> **Catatan:** Di **papertrade**, kline public API tidak membutuhkan API key. Kunci diperlukan saat **real trading**.

---

## 3) Jalur Kerja — Backtest → Dry‑Run → Papertrade → Real

### 3.1 Backtest (Scalping)

1. Taruh CSV ke folder `data/`. Contoh file: `data/trades_scalping_ADAUSDT_15M_2025-06-01_TO_2025-08-14.csv`.
2. Jalankan Streamlit:

```bash
streamlit run backtester_scalping.py --server.port 8501
```

3. Pilih file CSV → Atur parameter:

   * **use\_htf\_filter = OFF**, **ML = ON** (uji baseline)
   * Lalu uji **use\_htf\_filter = ON + ML = ON**
4. Unduh hasil trades CSV jika perlu.

> Backtester sudah selaras logic real‑trading: SL/BE/Trailing, cooldown, clamp min/max SL, dan money management berbasis **balance × risk × leverage / price** + normalisasi LOT\_SIZE.

### 3.2 Dry‑Run (CLI, cepat)

Jalankan replay bar‑by‑bar menggunakan logic `newrealtrading.py` (tanpa order):

```bash
python tools_dryrun_summary.py \
  --symbol ADAUSDT \
  --csv data/trades_scalping_ADAUSDT_15M_2025-06-01_TO_2025-08-14.csv \
  --coin_config coin_config.json \
  --steps 800 --balance 20 \
  --out ADA_dryrun_trades_800.csv
```

Output akan menampilkan **WinRate**, **ProfitFactor**, **avg PnL**, serta menyimpan CSV trade detail jika `--out` digunakan.

### 3.3 Papertrade (live data, TANPA order)

* Menarik kline futures USDⓈ‑M via public endpoint, lalu men‑drive `CoinTrader` & menyimpan jurnal CSV per simbol.

```bash
python papertrade.py \
  --live-paper \
  --symbols ADAUSDT,DOGEUSDT,XRPUSDT \
  --interval 15m \
  --balance 20 \
  --coin_config coin_config.json \
  --instance-id PT_A \
  --verbose
```

Jurnal tersimpan di folder `logs/<INSTANCE_ID>/<SYMBOL>_paper_trades.csv`.

### 3.4 Real Trading (order nyata)

> **Disarankan:** Mulai di **testnet** sampai alur terverifikasi.

```bash
# TESTNET (aman)
python newrealtrading.py \
  --live \
  --symbols ADAUSDT,DOGEUSDT,XRPUSDT \
  --interval 15m \
  --real-exec \
  --testnet \
  --instance-id REAL_A \
  --account-guard \
  --verbose

# MAINNET (modal asli, hati-hati!) — hapus --testnet
```

**Money Management multi‑coin (global balance gate)**

* Pada satu langkah pemindaian, bot menarik `availableBalance` akun futures, lalu **mengalokasikan entry berturut‑turut** per simbol.
* Contoh (balance awal \$20, risk 8%, 3 simbol):

  * DOGE: 20×0.08 = 1.60
  * XRP: (20−1.60)×0.08 = 1.49
  * ADA: (20−1.60−1.49)×0.08 ≈ 1.49 → *balance akhir ≈ 15.42*
* Jika **-2019 Margin is insufficient**, bot **mengecilkan qty** (hingga batas) atau **skip & cooldown**.

---

## 4) Menjalankan 2 Bot pada 1 Server/IP

**Direktori terpisah + ENV terpisah**:

```
/var/www/realtrading     # BOT A
/var/www/realtradingb    # BOT B
```

Langkah:

1. Duplikasi project ke dua folder di atas.
2. Masing‑masing folder punya `.env`, `coin_config.json`, `logs/`, dan **INSTANCE\_ID unik** (mis. `REAL_A`, `REAL_B`).
3. Gunakan **API key & akun Binance berbeda** (tidak akan saling bentrok). Jika **akun sama**, risiko balapan `availableBalance` tetap ada; pertimbangkan **akun terpisah** atau kurangi risk.
4. Jalankan masing‑masing:

```bash
# BOT A
cd /var/www/realtrading
python newrealtrading.py --live --symbols ADAUSDT,DOGEUSDT,XRPUSDT --interval 15m --real-exec --instance-id REAL_A --verbose

# BOT B
cd /var/www/realtradingb
python newrealtrading.py --live --symbols ADAUSDT,DOGEUSDT,XRPUSDT --interval 15m --real-exec --instance-id REAL_B --verbose
```

**Monitoring (Docker contoh):**

```bash
docker logs -f realtradingb | egrep "LIVE|DECISION|ENTRY|EXIT|COOLDOWN|Hit|FILTER BLOCK|error|WARN"
```

> **Anti‑bentrok:** Order ID memakai prefix `x-<INSTANCE_ID>-<SYMBOL>-<rand8>`. Pastikan **INSTANCE\_ID berbeda** per bot.

---

## 5) Docker (opsional)

### Build

```bash
docker build -t bot-trading:latest .
```

### Jalankan Backtester (web)

```bash
docker run --rm -it -p 8501:8501 \
  -v $PWD:/app \
  --name bt-scap \
  bot-trading:latest \
  streamlit run backtester_scalping.py --server.port 8501 --server.address 0.0.0.0
```

### Jalankan Papertrade

```bash
docker run --rm -it --name paperA \
  -v $PWD:/app \
  -e INSTANCE_ID=PT_A \
  bot-trading:latest \
  python papertrade.py --live-paper --symbols ADAUSDT,DOGEUSDT --interval 15m --balance 20 --verbose
```

### Jalankan Real Trading (testnet)

```bash
docker run -d --name realA \
  -v $PWD:/app \
  --env-file .env \
  -e INSTANCE_ID=REAL_A \
  bot-trading:latest \
  python newrealtrading.py --live --symbols ADAUSDT,DOGEUSDT --interval 15m --real-exec --testnet --verbose
```

### Jalankan Real Trading (Live)

```bash
docker run -d --name realA \
  -v $PWD:/app \
  --env-file .env \
  -e INSTANCE_ID=REAL_A \
  bot-trading:latest \
  python newrealtrading.py --live --symbols ADAUSDT,DOGEUSDT --interval 15m --real-exec --verbose
```

---

## 6) Konfigurasi per‑Symbol (`coin_config.json`)

Contoh (ringkas):

```json
{
  "ADAUSDT": {
    "leverage": 15,
    "risk_per_trade": 0.08,
    "taker_fee": 0.0005,
    "min_atr_pct": 0.003, "max_atr_pct": 0.04, "max_body_atr": 1.75,
    "use_htf_filter": 0, "cooldown_seconds": 900,
    "sl_mode": "ATR", "sl_pct": 0.008, "sl_atr_mult": 1.8,
    "sl_min_pct": 0.012, "sl_max_pct": 0.035,
    "use_breakeven": 1, "be_trigger_pct": 0.004,
    "trailing_trigger": 1.0, "trailing_step_min_pct": 0.45,
    "trailing_step_max_pct": 1.0, "trail_atr_k": 2.0,
    "max_hold_seconds": 3600,
    "stepSize": 1.0, "minQty": 1.0, "quantityPrecision": 0,
    "minNotional": 5.0,
    "SLIPPAGE_PCT": 0.02
  }
}
```

> **Tip:** `papertrade.py` otomatis **menyuntik** `stepSize/minQty/quantityPrecision/minNotional` dari `exchangeInfo` jika belum ada.

---

## 7) Troubleshooting

* **APIError -2019: Margin is insufficient**

  * Bot mengecilkan qty jika bisa; jika tetap gagal → **skip** & **cooldown**. Naikkan balance/leverage, atau turunkan risk.
  * Pastikan `minNotional` terpenuhi (`price × qty ≥ minNotional`).
* **Tidak ada entry**

  * Aktifkan `VERBOSE=1` dan cek: `FILTER BLOCK atr_ok/body_ok/htf`.
  * Di backtester, aktifkan **Mode Debug** untuk melihat alasan blokir.
* **Dua bot saling tarik balance (akun sama)**

  * Disarankan **API key/akun berbeda**. Jika harus satu akun, kurangi `risk_per_trade` dan gunakan timeframe berbeda untuk mengurangi balapan.

---

## 8) Catatan Patch Penting

* **newrealtrading.py**

  * Gate global `availableBalance` per langkah; mengurangi `used_margin` per simbol.
  * Penanganan `-2019` → shrink qty (jika bisa) atau cooldown.
  * Order ID unik: `x-<INSTANCE_ID>-<SYMBOL>-<rand8>` → aman untuk multi‑bot.
* **papertrade.py**

  * `ensure_filters_to_coin_config`: ambil & simpan `stepSize/minQty/quantityPrecision/minNotional` ke `coin_config.json` dengan file‑lock.
* **backtester\_scalping.py**

  * Default **HTF OFF** (uji baseline), opsi **ML ON**; Mode Debug untuk melihat blokir.
* **ml\_signal\_plugin.py**

  * Perbaikan \_build\_latest\_features → kembalikan **DataFrame** agar kompatibel dengan scikit‑learn.
* **engine\_core.py**

  * Filter ATR & body/ATR seragam; helper precision; time‑stop/TSL.

---

## 9) Target Kinerja

* Mulai dari parameter konservatif. Optimasi di backtester (WR & PF). Jangan langsung ke mainnet. Gunakan `tools_dryrun_summary.py` untuk sanity check sebelum papertrade & real trading.

---

**Selamat membangun dan iterasi!**
