README (Hot Reload + Real Trading)

Ringkas

Hot-reload coin_config.json otomatis tiap 5 detik (ubah via CONFIG_WATCH_INTERVAL).

Parameter yang bisa diubah live: risk_per_trade, leverage, trailing_trigger, trailing_step, max_hold_seconds, min_roi_to_close_by_time, taker_fee, maker_fee.

Exchange filters (stepSize, minQty, minNotional, tickSize) diambil & dicache; sizing dan order mematuhi aturan Binance.

PnL/ROI pakai Mark Price; fee dari config/ENV.

Persistence: posisi aktif direstore saat restart.

Telegram (opsional): set TELEGRAM_BOT_TOKEN & TELEGRAM_CHAT_ID.

.env contoh

BINANCE_API_KEY=xxx
BINANCE_API_SECRET=yyy
TAKER_FEE=0.0005
MAKER_FEE=0.0002
RISK_PERCENTAGE=1.0
USE_MULTIPLEX=1
WS_MAX_RETRY=0
CONFIG_WATCH_INTERVAL=5
STATE_FILE=/data/active_position_state.json
EXCHANGEINFO_CACHE=/data/exchange_info_cache.json
EXCHANGEINFO_TTL_SECONDS=21600
SYMBOLS=ADAUSDT,DOGEUSDT,XRPUSDT,TURBOUSDT

coin_config.json contoh

{
  "XRPUSDT": {
    "risk_per_trade": 0.08,
    "leverage": 15,
    "trailing_trigger": 0.5,
    "trailing_step": 0.3,
    "max_hold_seconds": 3600,
    "min_roi_to_close_by_time": 0.05,
    "taker_fee": 0.0005,
    "maker_fee": 0.0002
  },
  "DOGEUSDT": {
    "risk_per_trade": 0.08,
    "leverage": 15,
    "trailing_trigger": 0.5,
    "trailing_step": 0.3,
    "max_hold_seconds": 3600,
    "min_roi_to_close_by_time": 0.05,
    "taker_fee": 0.0005,
    "maker_fee": 0.0002
  },
  "TURBOUSDT": {
    "risk_per_trade": 0.08,
    "leverage": 15,
    "trailing_trigger": 0.5,
    "trailing_step": 0.3,
    "max_hold_seconds": 3600,
    "min_roi_to_close_by_time": 0.05,
    "taker_fee": 0.0005,
    "maker_fee": 0.0002
  }
}

Perkuat docker run
Contoh run yang lebih ketat:

sudo docker run -d --name realtrading \
  --env-file /var/www/realtrading/.env \
  -e SYMBOLS="ADAUSDT,DOGEUSDT,XRPUSDT,TURBOUSDT" \
  -v /var/www/realtrading/logs:/app/logs \
  -v /var/www/realtrading/data:/data \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,size=64m \
  --tmpfs /run:rw,noexec,nosuid,size=16m \
  --cap-drop ALL --security-opt no-new-privileges:true \
  --pids-limit 256 --memory 512m --memory-swap 512m \
  -p 127.0.0.1:8589:8589 \
  --restart unless-stopped \
  realtrading:latest


Penjelasan singkat:
-p 127.0.0.1:8589:8589 → hanya bisa diakses lewat Nginx (bagus, lanjutkan).
--read-only + --tmpfs → filesystem kontainer read-only, hanya logs & data yang bisa tulis.
--cap-drop ALL + no-new-privileges → minimal privilege.
Limit sumber daya (pids, memory) untuk tahan “bot scan” nakal.

Docker Compose (contoh)

services:
  bot:
    image: python:3.11
    working_dir: /app
    volumes:
      - ./:/app
      - ./data:/data
      - ./coin_config.json:/app/coin_config.json:rw
    env_file:
      - .env
    command: ["python", "newrealtrading.py"]
    restart: unless-stopped

Catatan

Ubah leverage saat posisi aktif: ukuran posisi berjalan tidak berubah (aturan Binance). Posisi baru pakai leverage baru.

SYMBOLS/TIMEFRAME saat ini statis (ubah + restart untuk meminimalkan risiko WS).

MARKET order dipakai default; limit order memerlukan pembulatan harga tickSize (sudah tersedia dari exchange filters bila nanti diperlukan).