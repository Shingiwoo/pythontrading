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
COPY newrealtrading.py coin_config.json ml_signal_plugin.py /app/

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
RUN printf '%s\n' '#!/usr/bin/env bash' \
 'set -euo pipefail' \
 'term_handler(){ echo "[entrypoint] Caught signal, stopping..."; if [[ -n "${BOT_PID:-}" ]]; then kill -TERM "$BOT_PID" 2>/dev/null || true; fi; if [[ -n "${HTTP_PID:-}" ]]; then kill -TERM "$HTTP_PID" 2>/dev/null || true; fi; wait; exit 0; }' \
 'trap term_handler SIGTERM SIGINT' \
 'python -u /app/newrealtrading.py & BOT_PID=$!' \
 'python -m http.server 8589 --directory /app & HTTP_PID=$!' \
 'wait -n' \
 'kill -TERM "$BOT_PID" "$HTTP_PID" 2>/dev/null || true' \
 'wait' > /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Gunakan exec-form supaya sinyal OS ditangani benar
ENTRYPOINT ["/app/entrypoint.sh"]