#!/usr/bin/env bash
# run_preset.sh — helper untuk apply preset & menjalankan mode (backtest/dryrun/paper/newreal)
# Usage:
#   ./run_preset.sh EXP_E backtest ADAUSDT 2025-01-01 2025-08-19
#   ./run_preset.sh EXP_C paper ADAUSDT

set -e

PRESET_ID="$1"
MODE="$2"
SYMBOLS="$3"
START_DATE="$4"
END_DATE="$5"

if [[ -z "$PRESET_ID" || -z "$MODE" || -z "$SYMBOLS" ]]; then
  echo "Usage: $0 <EXP_ID> <backtest|dryrun|paper|newreal> <SYMBOLS> [START] [END]"
  exit 1
fi

# 1) Apply preset (runner)
python preset_runner.py --presets Experiment_Presets.json --preset "$PRESET_ID" --symbols "$SYMBOLS"
cp .env.from_preset .env
cp coin_config.from_preset.json coin_config.json

echo "[INFO] Applied preset $PRESET_ID for symbols: $SYMBOLS"

# 2) Run
if [[ "$MODE" == "backtest" ]]; then
  if [[ -z "$START_DATE" || -z "$END_DATE" ]]; then
    echo "For backtest, provide START & END dates. Example: 2025-01-01 2025-08-19"
    exit 2
  fi
  python backtester_scalping.py --symbols "$SYMBOLS" --interval 15m --start "$START_DATE" --end "$END_DATE" --coin_config coin_config.json --verbose
elif [[ "$MODE" == "dryrun" ]]; then
  python papertrade.py --dryrun --symbols "$SYMBOLS" --interval 15m --coin_config coin_config.json --verbose
elif [[ "$MODE" == "paper" ]]; then
  python papertrade.py --live-paper --symbols "$SYMBOLS" --interval 15m --balance 20 --coin_config coin_config.json --verbose
elif [[ "$MODE" == "newreal" ]]; then
  python newrealtrading.py --symbols "$SYMBOLS" --interval 15m --coin_config coin_config.json --verbose
else
  echo "Unknown MODE: $MODE"
  exit 3
fi
