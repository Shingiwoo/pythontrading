#!/usr/bin/env python3
"""
preset_runner.py
Gunakan untuk menerapkan preset eksperimen (EXP_A..D) menjadi file .env dan coin_config.json.
Contoh:
  python preset_runner.py --presets Experiment_Presets.json --preset EXP_C --symbols ADAUSDT,DOGEUSDT

Output:
  .env.from_preset
  coin_config.from_preset.json
"""
import argparse, json, sys

base_env_defaults = {
    "USE_CANDLE_PATTERNS": "false",
    "LTF_CONFIRM_POLICY": "loose",{'USE_MTF': 'true', 'LTF_TFS': '1m,5m', 'BTF_TF': '15m', 'HTF_TFS': '1h,4h', 'MTF_POLICY': 'soft', 'WAIT_FOR_15M_CLOSE': 'true', 'INTRABAR_ENTRY': 'false', 'SCORE_THR_LONG': '2.3', 'SCORE_THR_SHORT': '2.0', 'HTF_BONUS': '0.5', 'LTF_CONFIRM_SCORE': '1.0', 'ATR_PCT_MIN': '0.6', 'ATR_PCT_MAX': '4.0', 'USE_VWAP_CONFIRM': 'false', 'USE_TWAP_INDICATOR': 'false', 'TWAP_ANCHOR': 'rolling', 'TWAP_15M_WINDOW': '20', 'TWAP_5M_WINDOW': '36', 'TWAP_1M_WINDOW': '120', 'TWAP_ATR_K': '0.4', 'TWAP_EXEC_ENABLED': 'false', 'TWAP_CHILD_COUNT': '4', 'TWAP_DURATION_SEC': '90', 'TWAP_CHILD_TIMEOUT_SEC': '10', 'TWAP_POST_ONLY': 'true', 'TWAP_FORCE_MARKET_ON_TIMEOUT': 'true', 'USE_LTF': 'true', 'USE_ML': 'true', 'ML_UP_PROB_LONG': '0.60', 'ML_UP_PROB_SHORT': '0.45'}}
base_coin_defaults = {{'rsi_period': 25, 'rsi_long_range': [10, 45], 'rsi_short_range': [70, 90], 'ema_period': 22, 'sma_period': 22, 'use_macd': True, 'use_htf': True, 'use_ltf': True, 'htf_weight': 0.5, 'ltf_weight': 1.0, 'score_threshold_long': 2.3, 'score_threshold_short': 2.0, 'risk_per_trade': 0.08, 'leverage': 14, 'position_scale_htf': {'both': 1.0, 'one': 0.5, 'opp_4h': 0.0}, 'use_twap_indicator': False, 'twap_anchor': 'rolling', 'twap_15m_window': 20, 'twap_5m_window': 36, 'twap_1m_window': 120, 'twap_atr_k': 0.4, 'twap_exec_enabled': False, 'twap_child_count': 4, 'twap_duration_sec': 90, 'twap_child_timeout_sec': 10, 'twap_post_only': True, 'twap_force_market_on_timeout': True}}

def map_to_env(p):
    e = base_env_defaults.copy()
    def setv(k, v): e[k] = str(v).lower() if isinstance(v, bool) else str(v)
    setv("MTF_POLICY", p.get("mtf_policy", e["MTF_POLICY"]))
    setv("WAIT_FOR_15M_CLOSE", p.get("wait_for_15m_close", True))
    setv("INTRABAR_ENTRY", p.get("intrabar_entry", False))
    setv("USE_VWAP_CONFIRM", p.get("use_vwap_confirm", False))
    setv("USE_TWAP_INDICATOR", p.get("use_twap_indicator", False))
    setv("TWAP_ANCHOR", p.get("twap_anchor", e["TWAP_ANCHOR"]))
    setv("TWAP_15M_WINDOW", p.get("twap_15m_window", e["TWAP_15M_WINDOW"]))
    setv("TWAP_5M_WINDOW", p.get("twap_5m_window", e["TWAP_5M_WINDOW"]))
    setv("TWAP_1M_WINDOW", p.get("twap_1m_window", e["TWAP_1M_WINDOW"]))
    setv("TWAP_ATR_K", p.get("twap_atr_k", e["TWAP_ATR_K"]))
    setv("TWAP_EXEC_ENABLED", p.get("use_twap_exec", False))
    setv("TWAP_CHILD_COUNT", p.get("twap_child_count", e["TWAP_CHILD_COUNT"]))
    setv("TWAP_DURATION_SEC", p.get("twap_duration_sec", e["TWAP_DURATION_SEC"]))
    setv("TWAP_CHILD_TIMEOUT_SEC", p.get("twap_child_timeout_sec", e["TWAP_CHILD_TIMEOUT_SEC"]))
    setv("TWAP_POST_ONLY", p.get("twap_post_only", True))
    setv("TWAP_FORCE_MARKET_ON_TIMEOUT", p.get("twap_force_market_on_timeout", True))
    setv("SCORE_THR_LONG", p.get("score_thr_long", e["SCORE_THR_LONG"]))
    setv("SCORE_THR_SHORT", p.get("score_thr_short", e["SCORE_THR_SHORT"]))
    setv("HTF_BONUS", p.get("htf_bonus", e["HTF_BONUS"]))
    setv("LTF_CONFIRM_SCORE", p.get("ltf_confirm_score", e["LTF_CONFIRM_SCORE"]))
    setv("ATR_PCT_MIN", p.get("atr_pct_min", e["ATR_PCT_MIN"]))
    setv("ATR_PCT_MAX", p.get("atr_pct_max", e["ATR_PCT_MAX"]))
    setv("USE_LTF", p.get("use_ltf", True))
    setv("ML_UP_PROB_LONG", p.get("ml_up_prob_long", e["ML_UP_PROB_LONG"]))
    
    setv("USE_CANDLE_PATTERNS", p.get("use_candle_patterns", False))
    setv("LTF_CONFIRM_POLICY", p.get("ltf_confirm_policy", "loose"))
setv("ML_UP_PROB_SHORT", p.get("ml_up_prob_short", e["ML_UP_PROB_SHORT"]))
    return e

def env_to_text(d):
    return "\\n".join([f"{k}={v}" for k,v in d.items()]) + "\\n"

def map_to_coin_config(p, symbols):
    out = {}
    for sym in symbols:
        b = base_coin_defaults.copy()
        b["score_threshold_long"] = p.get("score_thr_long", b["score_threshold_long"])
        b["score_threshold_short"] = p.get("score_thr_short", b["score_threshold_short"])
        b["use_htf"] = p.get("use_htf", True)
        b["use_ltf"] = p.get("use_ltf", True)
        b["use_twap_indicator"] = p.get("use_twap_indicator", False)
        b["twap_anchor"] = p.get("twap_anchor", b["twap_anchor"])
        b["twap_15m_window"] = p.get("twap_15m_window", b["twap_15m_window"])
        b["twap_5m_window"] = p.get("twap_5m_window", b["twap_5m_window"])
        b["twap_1m_window"] = p.get("twap_1m_window", b["twap_1m_window"])
        b["twap_atr_k"] = p.get("twap_atr_k", b["twap_atr_k"])
        b["twap_exec_enabled"] = p.get("use_twap_exec", False)
        b["twap_child_count"] = p.get("twap_child_count", b["twap_child_count"])
        b["twap_duration_sec"] = p.get("twap_duration_sec", b["twap_duration_sec"])
        b["twap_child_timeout_sec"] = p.get("twap_child_timeout_sec", b["twap_child_timeout_sec"])
        b["twap_post_only"] = p.get("twap_post_only", b["twap_post_only"])
        b["twap_force_market_on_timeout"] = p.get("twap_force_market_on_timeout", b["twap_force_market_on_timeout"])
        out[sym] = b
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", required=True, help="Path to Experiment_Presets.json")
    ap.add_argument("--preset", required=True, help="EXP_A | EXP_B | EXP_C | EXP_D (atau id lain di file)")
    ap.add_argument("--symbols", default="ADAUSDT", help="Comma separated symbols, ex: ADAUSDT,DOGEUSDT")
    ap.add_argument("--env_out", default=".env.from_preset", help="Output .env path")
    ap.add_argument("--coin_out", default="coin_config.from_preset.json", help="Output coin_config path")
    args = ap.parse_args()

    with open(args.presets,"r") as f:
        items = json.load(f)
    pick = None
    for it in items:
        if it.get("exp_id") == args.preset:
            pick = it; break
    if pick is None:
        print("Preset tidak ditemukan:", args.preset, file=sys.stderr)
        sys.exit(1)

    env = map_to_env(pick)
    with open(args.env_out,"w") as f:
        f.write(env_to_text(env))

    syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
    coin = map_to_coin_config(pick, syms)
    with open(args.coin_out,"w") as f:
        json.dump(coin, f, indent=2)
    print("OK: wrote", args.env_out, "and", args.coin_out)

if __name__ == "__main__":
    main()
