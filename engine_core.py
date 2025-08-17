import os, json, math
from typing import Dict, Any, Tuple, Optional

import pandas as pd
from ta.trend import EMAIndicator, SMAIndicator, MACD
from ta.momentum import RSIIndicator

# -----------------------------------------------------
# Konversi tipe sederhana
# -----------------------------------------------------
def _to_float(v: Any, d: float) -> float:
    try:
        return float(v)
    except Exception:
        return float(d)

def _to_int(v: Any, d: int) -> int:
    try:
        return int(v)
    except Exception:
        return int(d)

def _to_bool(v: Any, d: bool) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(int(v))
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1","true","y","yes","on"}: return True
        if s in {"0","false","n","no","off"}: return False
    return bool(d)

# -----------------------------------------------------
# Precision helpers
# -----------------------------------------------------
def round_to_step(x: float, step: float) -> float:
    if step <= 0:
        return float(x)
    return round(x / step) * step

def floor_to_step(x: float, step: float) -> float:
    if step <= 0:
        return float(x)
    return math.floor(x / step) * step

def ceil_to_step(x: float, step: float) -> float:
    if step <= 0:
        return float(x)
    return math.ceil(x / step) * step

def enforce_precision(sym_cfg: Dict[str, Any], price: float, qty: float) -> Tuple[float, float]:
    p_step = _to_float(sym_cfg.get("tickSize", sym_cfg.get("pricePrecision", 0)), 0)
    q_step = _to_float(sym_cfg.get("stepSize", 0), 0)
    price = round_to_step(price, p_step) if p_step > 0 else price
    qty = floor_to_step(qty, q_step) if q_step > 0 else qty
    return price, qty

def meets_min_notional(sym_cfg: Dict[str, Any], price: float, qty: float) -> bool:
    min_not = _to_float(sym_cfg.get("minNotional", 0), 0)
    return (price * qty) >= min_not if min_not > 0 else True

# -----------------------------------------------------
# Config loader & merger
# -----------------------------------------------------
def load_coin_config(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def merge_config(symbol: str, base_cfg: Dict[str, Any]) -> Dict[str, Any]:
    sym_cfg = base_cfg.get(symbol, {}) if isinstance(symbol, str) else {}
    return dict(sym_cfg)

# -----------------------------------------------------
# Indikator & sinyal sederhana
# -----------------------------------------------------
def compute_indicators(df: pd.DataFrame, heikin: bool = False) -> pd.DataFrame:
    d = df.copy()
    if heikin:
        ha = d.copy()
        ha['ha_close'] = (ha['open']+ha['high']+ha['low']+ha['close'])/4
        ha['ha_open'] = ha['open'].shift(1)
        ha['ha_open'].iloc[0] = (ha['open'].iloc[0]+ha['close'].iloc[0])/2
        ha['ha_high'] = ha[['high','ha_open','ha_close']].max(axis=1)
        ha['ha_low'] = ha[['low','ha_open','ha_close']].min(axis=1)
        d[['open','high','low','close']] = ha[['ha_open','ha_high','ha_low','ha_close']]
    d['ema'] = EMAIndicator(d['close'], 22).ema_indicator()
    d['ma'] = SMAIndicator(d['close'], 20).sma_indicator()
    macd = MACD(d['close']); d['macd']=macd.macd(); d['macd_signal']=macd.macd_signal()
    rsi = RSIIndicator(d['close'], 25); d['rsi']=rsi.rsi()
    prev_close = d['close'].shift(1)
    tr = pd.DataFrame({'a': d['high']-d['low'], 'b': (d['high']-prev_close).abs(), 'c': (d['low']-prev_close).abs()})
    d['tr'] = tr.max(axis=1)
    d['atr'] = d['tr'].ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    d['atr_pct'] = d['atr'] / d['close']
    d['body_atr'] = (d['close']-d['open']).abs()/d['atr']
    d['long_base'] = (d['ema']>d['ma']) & (d['macd']>d['macd_signal']) & d['rsi'].between(10,40)
    d['short_base'] = (d['ema']<d['ma']) & (d['macd']<d['macd_signal']) & d['rsi'].between(70,90)
    return d

def htf_trend_ok(side: str, base_df: pd.DataFrame) -> bool:
    try:
        tmp = base_df.set_index('timestamp')[['close']].copy()
        htf = tmp['close'].resample('1H').last().dropna()
        if len(htf) < 210: return True
        ema50 = htf.ewm(span=50, adjust=False).mean().iloc[-1]
        ema200 = htf.ewm(span=200, adjust=False).mean().iloc[-1]
        return (ema50 >= ema200) if side=='LONG' else (ema50 <= ema200)
    except Exception:
        return True
def apply_filters(ind: pd.Series, coin_cfg: Dict[str, Any]) -> Tuple[bool, bool, Dict[str, Any]]:
    min_atr = _to_float(coin_cfg.get('min_atr_pct', 0.0), 0.0)
    max_atr = _to_float(coin_cfg.get('max_atr_pct', 1.0), 1.0)
    max_body = _to_float(coin_cfg.get('max_body_atr', 999.0), 999.0)
    atr_ok = (ind['atr_pct'] >= min_atr) and (ind['atr_pct'] <= max_atr)
    body_ok = (ind['body_atr'] <= max_body)
    return atr_ok, body_ok, {'atr_pct': float(ind['atr_pct']), 'body_atr': float(ind['body_atr'])}

def decide_base(ind: pd.Series, coin_cfg: Dict[str, Any]) -> Dict[str, bool]:
    return {'L': bool(ind.get('long_base', False)), 'S': bool(ind.get('short_base', False))}

def confirm_htf(htf_ind: pd.DataFrame, coin_cfg: Dict[str, Any]) -> bool:
    # placeholder konfirmasi HTF: true jika ema50 >= ema200
    try:
        ema50 = htf_ind['close'].ewm(span=50, adjust=False).mean().iloc[-1]
        ema200 = htf_ind['close'].ewm(span=200, adjust=False).mean().iloc[-1]
        side = coin_cfg.get('side','LONG')
        return ema50 >= ema200 if side=='LONG' else ema50 <= ema200
    except Exception:
        return True

def apply_ml_gate(up_prob: Optional[float], ml_thr: float, eps: float = 1e-9) -> bool:
    if up_prob is None:
        return True
    up_odds = up_prob / max(1 - up_prob, eps)
    return up_odds >= ml_thr

# -----------------------------------------------------
# Money management & PnL
# -----------------------------------------------------
def risk_size(balance: float, risk_pct: float, entry: float, stop: float,
              fee_bps: float, slip_bps: float, sym_cfg: Dict[str, Any]) -> float:
    lev = _to_int(sym_cfg.get('leverage', 1), 1)
    risk_val = balance * risk_pct
    diff = abs(entry - stop)
    if diff <= 0:
        return 0.0
    qty = (risk_val * lev) / diff
    step = _to_float(sym_cfg.get('stepSize', 0), 0)
    return floor_to_step(qty, step) if step>0 else qty

def pnl_net(side: str, entry: float, exit: float, qty: float,
            fee_bps: float, slip_bps: float) -> Tuple[float, float]:
    fee = (entry + exit) * qty * fee_bps/10000.0
    slip = (entry + exit) * qty * slip_bps/10000.0
    gross = (exit - entry) * qty if side=='LONG' else (entry - exit) * qty
    pnl = gross - fee - slip
    roi = (pnl / (entry*qty)) if entry*qty>0 else 0.0
    return pnl, roi*100.0

# -----------------------------------------------------
# Journaling
# -----------------------------------------------------
def journal_row(ts: str, symbol: str, side: str, entry: float, qty: float,
                sl_init: Optional[float], tsl_init: Optional[float],
                exit_price: float, exit_reason: str,
                pnl_usdt: float, roi_pct: float, balance_after: float) -> Dict[str, Any]:
    return {
        'timestamp': ts,
        'symbol': symbol,
        'side': side,
        'entry': f"{entry:.6f}",
        'qty': f"{qty:.6f}",
        'sl_init': "" if sl_init is None else f"{sl_init:.6f}",
        'tsl_init': "" if tsl_init is None else f"{tsl_init:.6f}",
        'exit_price': f"{exit_price:.6f}",
        'exit_reason': exit_reason,
        'pnl_usdt': f"{pnl_usdt:.4f}",
        'roi_pct': f"{roi_pct:.2f}",
        'balance_after': f"{balance_after:.4f}",
    }

def init_stops(side: str, entry: float, ind: pd.Series, coin_cfg: Dict[str, Any]) -> Dict[str, Optional[float]]:
    atr = float(ind.get('atr', 0.0))
    sl = entry * (1 - _to_float(coin_cfg.get('sl_pct', 0.01), 0.01)) if side=='LONG' else entry * (1 + _to_float(coin_cfg.get('sl_pct', 0.01), 0.01))
    return {'sl': sl, 'tsl': None}

def step_trailing(side: str, bar: pd.Series, prev_state: Dict[str, Optional[float]], ind: pd.Series, coin_cfg: Dict[str, Any]) -> Optional[float]:
    trigger = _to_float(coin_cfg.get('trailing_trigger', 0.7), 0.7)
    step = _to_float(coin_cfg.get('trailing_step', 0.45), 0.45)
    entry = prev_state.get('entry', 0.0)
    price = float(bar['close'])
    profit_pct = (price-entry)/entry*100 if side=='LONG' else (entry-price)/entry*100
    if profit_pct < trigger:
        return prev_state.get('tsl')
    if side=='LONG':
        new_tsl = price*(1-step/100)
        return max(prev_state.get('tsl') or prev_state.get('sl') or 0.0, new_tsl)
    else:
        new_tsl = price*(1+step/100)
        return min(prev_state.get('tsl') or prev_state.get('sl') or 1e18, new_tsl)

def maybe_move_to_BE(side: str, entry: float, tsl: Optional[float], rule: Dict[str, Any]) -> Optional[float]:
    be = _to_float(rule.get('be_trigger_pct', 0.0), 0.0)
    price = rule.get('price', entry)
    if be <= 0:
        return tsl
    if side=='LONG' and (price-entry)/entry >= be:
        return max(tsl or 0.0, entry)
    if side=='SHORT' and (entry-price)/entry >= be:
        return min(tsl or 1e18, entry)
    return tsl

def check_time_stop(entry_time: float, now: float, roi: float, rule: Dict[str, Any]) -> bool:
    max_secs = _to_int(rule.get('time_stop_secs', 0), 0)
    return (max_secs>0) and ((now-entry_time) >= max_secs)

def cooldown_until(now: float, rule: Dict[str, Any]) -> float:
    secs = _to_int(rule.get('cooldown_secs', 0), 0)
    return now + max(0, secs)

def simulate_fill_on_candle(side: str, state: Dict[str, float], bar: pd.Series, sym_cfg: Dict[str, Any], fee_bps: float, slip_bps: float) -> Optional[Tuple[float, str]]:
    o, h, l, c = bar['open'], bar['high'], bar['low'], bar['close']
    sl = state.get('sl'); tsl = state.get('tsl')
    if side=='LONG':
        if tsl is not None and l <= tsl:
            return tsl, 'Hit Trailing SL'
        if sl is not None and l <= sl:
            return sl, 'Hit Hard SL'
    else:
        if tsl is not None and h >= tsl:
            return tsl, 'Hit Trailing SL'
        if sl is not None and h >= sl:
            return sl, 'Hit Hard SL'
    return None
