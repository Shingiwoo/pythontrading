# ===== Base image Python 3.13 =====
FROM python:3.13-slim

ENV DEBIAN_FRONTEND=noninteractive
# Paket sistem + CA cert (fix SSL) + tzdata
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates tzdata build-essential curl \
    && rm -rf /var/lib/apt/lists/* && update-ca-certificates

# Zona waktu & SSL bundle env
ENV TZ=Asia/Jakarta \
    PYTHONUNBUFFERED=1 \
    SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
    REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt

WORKDIR /app

# Install deps dari requirements.txt (gunakan cache layer)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy project
COPY newrealtrading.py coin_config.json ml_signal_plugin.py engine_core.py /app/

# (Opsional) kalau MAU copy .env ke container, uncomment baris ini.
# Namun lebih aman pakai --env-file atau docker-compose.
# COPY .env /app/.env

# Expose port untuk Nginx reverse proxy
EXPOSE 8589

# Buat folder logs
RUN mkdir -p /app/logs

# Jalankan bot + HTTP server sederhana di 8888 (untuk Nginx proxy)
# Catatan: http.server hanya untuk health/akses sederhana.
# Bot tetap berjalan di background.
# Entry script: pilih MODE=real/paper
RUN printf '%s\n' '#!/usr/bin/env bash' \
 'set -euo pipefail' \
 'term_handler(){ echo "[entrypoint] Caught signal, stopping..."; if [[ -n "${BOT_PID:-}" ]]; then kill -TERM "$BOT_PID" 2>/dev/null || true; fi; if [[ -n "${HTTP_PID:-}" ]]; then kill -TERM "$HTTP_PID" 2>/dev/null || true; fi; wait; exit 0; }' \
 'trap term_handler SIGTERM SIGINT' \
 'APP="${MODE:-real}"' \
 'PORT="${HTTP_PORT:-8589}"' \
 'if [[ "$APP" == "paper" ]]; then' \
 '  echo "[entrypoint] Starting papertrade.py (live-paper=$LIVE_PAPER)";' \
 '  python -u /app/papertrade.py ${LIVE_PAPER:+--live-paper} --symbols "${SYMBOLS:-ADAUSDT}" --interval "${INTERVAL:-15m}" --balance "${BALANCE:-20}" --risk_pct "${RISK_PCT:-0.01}" --verbose --coin_config "${COIN_CONFIG:-/data/coin_config.json}" --logs_dir "${LOGS_DIR:-/app/logs}" --ml-thr "${ML_THR:-1.0}" --fee_bps "${FEE_BPS:-10}" --slip_bps "${SLIP_BPS:-0}" & BOT_PID=$!' \
 'else' \
 '  echo "[entrypoint] Starting newrealtrading.py";' \
 '  python -u /app/newrealtrading.py --coin_config "${COIN_CONFIG:-/data/coin_config.json}" --symbol "${SYMBOLS:-ADAUSDT}" --verbose & BOT_PID=$!' \
 'fi' \
 'python -m http.server "$PORT" --directory /app & HTTP_PID=$!' \
 'wait -n' \
 'kill -TERM "$BOT_PID" "$HTTP_PID" 2>/dev/null || true' \
 'wait' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]