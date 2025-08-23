# CLI Quickstart — Preset A..H (MTF RSI Pullback + TWAP)
Tanggal: 23 August 2025

## 1) Apply Preset → .env & coin_config
**Opsi 1 — prebuilt (contoh EXP_E):**
```bash
cp .env.EXP_E .env && cp coin_config.EXP_E.json coin_config.json
```
**Opsi 2 — runner:**
```bash
python preset_runner.py --presets Experiment_Presets.json --preset EXP_E --symbols ADAUSDT
cp .env.from_preset .env && cp coin_config.from_preset.json coin_config.json
```

## 2) Backtest
```bash
python backtester_scalping.py --symbols ADAUSDT --interval 15m --start 2025-01-01 --end 2025-08-19 --coin_config coin_config.json --verbose
# Optional export:
# --export_trades trades_EXP_E_ADA.csv --export_summary summary_EXP_E_ADA.json
```

## 3) Dryrun (live, no orders)
```bash
python papertrade.py --dryrun --symbols ADAUSDT --interval 15m --coin_config coin_config.json --verbose
```

## 4) Paper Trade
```bash
python papertrade.py --live-paper --symbols ADAUSDT --interval 15m --balance 20 --coin_config coin_config.json --verbose
```

## 5) NewReal
```bash
python newrealtrading.py --symbols ADAUSDT --interval 15m --coin_config coin_config.json --verbose
```

## 6) Logging Hasil
- *Experiment_Tracker.csv*: WR/PF/fee/slippage/MaxDD/notes per eksperimen.
- *Param_Change_Log.csv*: catatan perubahan parameter & dampak.
- Penamaan standar: `trades_<EXP>_<SYMBOL>.csv`, `summary_<EXP>_<SYMBOL>.json`
