# Bot Trading Binance Futures

Proyek ini mengubah `app_fixscalping.py` menjadi bot trading nyata dengan backend FastAPI dan frontend Svelte.

## Persiapan

1. Salin `.env.example` menjadi `.env` lalu isi kunci API Binance.
2. Instal dependensi:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan backend:
   ```bash
   uvicorn backend.main:app --reload
   ```
4. Frontend Svelte berada pada folder `frontend/svelte`.

## Struktur

```
project/
  backend/
    main.py
    core/
      ...
  frontend/
    svelte/
      ...
```

Backend menyediakan endpoint `/api` untuk menjalankan bot, melihat status, dan laporan trading.
