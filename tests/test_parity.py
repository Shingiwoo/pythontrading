import os
import json
import sys
import pathlib
import pandas as pd
import pytest

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
from engine_core import compute_base_signals_backtest, compute_base_signals_live, make_decision

def test_backtest_vs_live_parity():
    csv_path = "data/ADAUSDT_15m_2025-08-01_to_2025-08-19.csv"
    if not os.path.exists(csv_path):
        pytest.skip("dataset tidak ada")
    df = pd.read_csv(csv_path)
    with open("coin_config.json") as f:
        cfg = json.load(f)
    sym = "ADAUSDT"
    decisions_bt, decisions_live = [], []
    for i in range(400, len(df)):
        d = df.iloc[: i + 1].copy()
        lb_bt, sb_bt = compute_base_signals_backtest(d, cfg.get(sym, {}))
        lb_lv, sb_lv = compute_base_signals_live(d, cfg.get(sym, {}))
        dec_bt = "LONG" if lb_bt and not sb_bt else "SHORT" if sb_bt and not lb_bt else None
        dec_lv = "LONG" if lb_lv and not sb_lv else "SHORT" if sb_lv and not lb_lv else None
        decisions_bt.append(dec_bt)
        decisions_live.append(dec_lv)
    assert decisions_bt == decisions_live
