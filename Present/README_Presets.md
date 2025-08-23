# Preset Runner — MTF RSI Pullback + TWAP

Tanggal: 23 August 2025

## Cara Pakai ( cepat )
```bash
python preset_runner.py --presets Experiment_Presets.json --preset EXP_C --symbols ADAUSDT,DOGEUSDT
# Output:
#   .env.from_preset
#   coin_config.from_preset.json
```

- **--preset**: EXP_A | EXP_B | EXP_C | EXP_D (atau id lain di file JSON)
- **--symbols**: daftar simbol koma (default: ADAUSDT)

## Preset Sample Output (sudah dibuat di folder ini)
- .env.EXP_A, coin_config.EXP_A.json
- .env.EXP_B, coin_config.EXP_B.json
- .env.EXP_C, coin_config.EXP_C.json
- .env.EXP_D, coin_config.EXP_D.json

> Cocok untuk langsung dicoba pada backtester/papertrade (copy/rename ke `.env` dan `coin_config.json`).

## Catatan Penting
- Nilai default lain (leverage, risk, dll) ada di `coin_config` per coin. Sesuaikan sesuai toleransi risiko.
- Log perubahan parameter sebaiknya dicatat di `Param_Change_Log.csv` agar tuning terukur.
- Pastikan backtester & engine membaca variabel sesuai nama pada `.env` & `coin_config.json` ini.
