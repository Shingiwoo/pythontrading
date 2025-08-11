"""
ml_signal_plugin.py — Plugin ML untuk live trading (selaras backtester)

Fungsi:
- Melatih model RandomForest berbasis fitur teknikal ringan.
- Retrain berkala per N bar.
- Memberikan probabilitas naik (up_prob) untuk bar terbaru.
- Mengonversi probabilitas menjadi skor ML (+1 long / +1 short) seperti backtester.
- Menggabungkan dengan skor dasar (base rule) dan memutuskan final sinyal memakai score_threshold.

Integrasi minimal ke newrealtrading.py:
1) `from ml_signal_plugin import MLSignal`
2) Di `CoinTrader.__init__`: `self.ml = MLSignal(self.config)`
3) Di `check_trading_signals()` setelah `df = self.calculate_indicators()` dan `last = df.iloc[-1]`:
   ```python
   up_prob = None
   if self.ml.use_ml:
       self.ml.fit_if_needed(df)
       up_prob = self.ml.predict_up_prob(df)
   long_base = bool(last.get('long_signal', False))
   short_base = bool(last.get('short_signal', False))
   long_sig, short_sig = self.ml.score_and_decide(long_base, short_base)
   ```
   Ganti seluruh penggunaan `long_sig/short_sig` setelah ini agar pakai variabel yang baru.

ENV/.env:
- USE_ML=1   # aktifkan ML di live
- SCORE_THRESHOLD=1.0  # sama seperti di backtester (1.0/1.2/1.4 dst)
- ML_MIN_TRAIN_BARS=400
- ML_LOOKAHEAD=5
- ML_RETRAIN_EVERY=120
- ML_UP_PROB=0.55
- ML_DOWN_PROB=0.45

Jika parameter juga ingin per-coin, boleh disimpan di coin_config.json dengan nama kunci yang sama — plugin akan membacanya dulu dari config coin, lalu fallback ke ENV.
"""
from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


@dataclass
class MLParams:
    use_ml: bool = False
    score_threshold: float = 1.0
    min_train_bars: int = 400
    lookahead: int = 5
    retrain_every_bars: int = 120
    up_prob_thres: float = 0.55
    down_prob_thres: float = 0.45


class MLSignal:
    def __init__(self, coin_cfg: Dict[str, Any] | None):
        cfg = coin_cfg or {}
        env = __import__('os').environ
        def _getf(key: str, default: float) -> float:
            v = cfg.get(key, env.get(key))
            try:
                return float(v)
            except Exception:
                return float(default)
        def _geti(key: str, default: int) -> int:
            v = cfg.get(key, env.get(key))
            try:
                return int(v)
            except Exception:
                return int(default)
        def _getb(key: str, default: bool) -> bool:
            v = cfg.get(key, env.get(key))
            if isinstance(v, bool):
                return v
            if isinstance(v, (int, float)):
                return bool(int(v))
            if isinstance(v, str):
                s = v.strip().lower()
                if s in {"1","true","y","yes","on"}:
                    return True
                if s in {"0","false","n","no","off"}:
                    return False
            return bool(default)

        self.params = MLParams(
            use_ml=_getb('USE_ML', False),
            score_threshold=_getf('SCORE_THRESHOLD', 1.0),
            min_train_bars=_geti('ML_MIN_TRAIN_BARS', 400),
            lookahead=_geti('ML_LOOKAHEAD', 5),
            retrain_every_bars=_geti('ML_RETRAIN_EVERY', 120),
            up_prob_thres=_getf('ML_UP_PROB', 0.55),
            down_prob_thres=_getf('ML_DOWN_PROB', 0.45)
        )

        self.model: Optional[RandomForestClassifier] = None
        self._last_fit_index: int = -1
        self._last_fit_time: float = 0.0

    # ===== Public API =====
    @property
    def use_ml(self) -> bool:
        return self.params.use_ml

    def fit_if_needed(self, df: pd.DataFrame) -> None:
        """Latih ulang model bila data cukup dan sudah melewati interval retrain."""
        feats, target = self._build_dataset(df)
        if feats is None or target is None:
            return
        n = len(feats)
        if n < self.params.min_train_bars:
            return
        # retrain guard: per X bar atau minimal 60 detik
        need = (self._last_fit_index < 0) or (n - self._last_fit_index >= self.params.retrain_every_bars) or (time.time() - self._last_fit_time > 60)
        if not need:
            return
        m = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight='balanced_subsample')
        m.fit(feats, target)
        self.model = m
        self._last_fit_index = n
        self._last_fit_time = time.time()

    def predict_up_prob(self, df: pd.DataFrame) -> Optional[float]:
        if self.model is None:
            return None
        f = self._build_latest_features(df)
        if f is None:
            return None
        proba = self.model.predict_proba(f)[0, 1]
        return float(proba)

    def score_and_decide(self, base_long: bool, base_short: bool, up_prob: Optional[float] = None) -> Tuple[bool, bool]:
        """Mimic backtester scoring: base memberi +1, ML memberi +1 ke arah probabilitasnya.
        Keputusan: sinyal valid bila skor >= score_threshold."""
        th = float(self.params.score_threshold)
        sc_long = 1.0 if base_long else 0.0
        sc_short = 1.0 if base_short else 0.0
        if self.use_ml and (up_prob is not None):
            if up_prob >= self.params.up_prob_thres:
                sc_long += 1.0
            elif up_prob <= self.params.down_prob_thres:
                sc_short += 1.0
        return (sc_long >= th, sc_short >= th)

    # ===== Internal: feature engineering =====
    def _build_dataset(self, df: pd.DataFrame) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        if df is None or df.empty:
            return None, None
        d = df.copy()
        # pastikan kolom dasar sudah ada (lihat calculate_indicators)
        req = {'close','rsi','macd','atr'}
        if not req.issubset(d.columns):
            return None, None
        # fitur tambahan ringan
        d['bb_width'] = (d['close'].rolling(20).std() * 4.0) / d['close']  # ~ Bollinger width(normalized)
        d['lag_ret'] = d['close'].pct_change().shift(1)
        d['vol'] = d['close'].rolling(20).std().shift(1)
        # target: return ke depan (lookahead) > 0
        la = int(self.params.lookahead)
        d['target'] = (d['close'].shift(-la) > d['close']).astype(int)
        feat_cols = ['rsi','macd','atr','bb_width','lag_ret','vol']
        ds = d[feat_cols + ['target']].dropna()
        if ds.empty:
            return None, None
        X = ds[feat_cols]
        y = ds['target']
        return X, y

    def _build_latest_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        if df is None or df.empty:
            return None
        d = df.copy()
        if not {'close','rsi','macd','atr'}.issubset(d.columns):
            return None
        d['bb_width'] = (d['close'].rolling(20).std() * 4.0) / d['close']
        d['lag_ret'] = d['close'].pct_change().shift(1)
        d['vol'] = d['close'].rolling(20).std().shift(1)
        feat_cols = ['rsi','macd','atr','bb_width','lag_ret','vol']
        row = d[feat_cols].iloc[[-1]].dropna()
        if row.empty:
            return None
        return row.values  # shape (1, n_feat)
