# Penjelasan Parameter Strategi

Berikut adalah penjelasan parameter yang digunakan dalam strategi trading ini, yang dapat dikonfigurasi dalam file JSON untuk menyesuaikan perilaku strategi per simbol (pair/coin).

## Indikator & Sinyal

### `ema_period`
- **Deskripsi**: Periode (jumlah bar) untuk menghitung *Exponential Moving Average* (EMA).
- **Fungsi**: Digunakan untuk mendeteksi arah tren harga utama.

### `sma_period`
- **Deskripsi**: Periode untuk menghitung *Simple Moving Average* (SMA).
- **Fungsi**: Dibandingkan dengan EMA untuk mengkonfirmasi tren (EMA > SMA = *uptrend*).

### `rsi_period`
- **Deskripsi**: Periode untuk menghitung *Relative Strength Index* (RSI).
- **Fungsi**: Mendeteksi momentum pasar dan area *overbought*/*oversold*.

### `macd_fast`, `macd_slow`, `macd_signal`
- **Deskripsi**:
  - `macd_fast`: Periode EMA cepat pada *MACD*.
  - `macd_slow`: Periode EMA lambat pada *MACD*.
  - `macd_signal`: Periode SMA pada *MACD signal line*.
- **Fungsi**: Kombinasi ini digunakan untuk mendeteksi momentum tren (*bullish*/*bearish* crossover).

## Scoring & Filter Sinyal

### `score_threshold`
- **Deskripsi**: Nilai skor total minimum (akumulasi scoring indikator & ML) agar sinyal *entry* dianggap valid.
- **Contoh**: Jika `threshold = 2.0`, hanya *entry* jika kombinasi (trend + ML, atau trend + RSI + MACD, dll.) bernilai >= 2.0.

### `ml_weight`
- **Deskripsi**: Bobot/skor kontribusi sinyal dari *machine learning* (ML).
- **Fungsi**: Jika ML memprediksi searah, skor ML = 1.0. Nilai ini dapat diubah sesuai tingkat kepercayaan pada model.

### `use_crossover_filter`
- **Deskripsi**: Jika `True`, sinyal tren hanya diberikan saat terjadi persilangan EMA/SMA (*cross up/down*) pada bar saat ini.
- **Fungsi**: Membatasi sinyal hanya pada awal tren, bukan selama tren berjalan.

### `only_trend_15m`
- **Deskripsi**: Jika `True`, *entry* hanya dilakukan jika tren pada timeframe lebih tinggi (15 menit) searah dengan sinyal.
- **Fungsi**: Jika `False`, memungkinkan *entry* counter-trend pada timeframe tinggi.

## Risk Management & Exit

### `sl_min_pct`
- **Deskripsi**: Jarak minimal *Stop Loss* (SL) dari harga *entry* (dalam persen).
- **Contoh**: `1.0` = SL minimal 1% dari harga *entry*.

### `sl_atr_multiplier`
- **Deskripsi**: Jarak *Stop Loss* berdasarkan *Average True Range* (ATR).
- **Fungsi**: SL = *entry* ± (ATR × multiplier). Membuat SL adaptif terhadap volatilitas.

### `tp_rr`
- **Deskripsi**: Rasio *Target Profit* (TP) terhadap SL (*Risk Reward Ratio*).
- **Contoh**: `2.0` berarti TP = 2 × SL.

### `trailing_enabled`
- **Deskripsi**: Mengaktifkan atau menonaktifkan *trailing stop-loss*.

### `trailing_mode`
- **Deskripsi**: Mode *trailing*:
  - `"pct"`: *Trailing offset* dalam persen.
  - `"atr"`: *Trailing* berbasis ATR.
- **Fungsi**: Menentukan cara perhitungan *trailing stop-loss*.

### `atr_multiplier`
- **Deskripsi**: Digunakan untuk *trailing stop-loss* berbasis ATR.
- **Fungsi**: *Trailing SL offset* = ATR × `atr_multiplier`.

## Trailing & Breakeven

### `breakeven_trigger_pct`
- **Deskripsi**: Persentase profit dari *entry* (misal 0.5%) yang memicu pergeseran SL ke *breakeven* (modal).

### `trailing_offset_pct`
- **Deskripsi**: Offset *trailing stop-loss* dari harga tertinggi (atau terendah), dalam persen.

### `trailing_trigger_pct`
- **Deskripsi**: *Trailing* baru aktif setelah profit mencapai persentase tertentu dari *entry*.

## Mode Swing (Trend Kuat)

### `swing_atr_multiplier`
- **Deskripsi**: Offset *trailing*/*SL* untuk mode *swing* (mengikuti tren besar), menggunakan ATR × multiplier.
- **Fungsi**: Umumnya lebih lebar dari mode *scalp*.

### `swing_breakeven_trigger_pct`
- **Deskripsi**: Persentase profit untuk memicu *breakeven* pada mode *swing*.

### `swing_trailing_offset_pct`
- **Deskripsi**: Offset *trailing stop-loss* untuk mode *swing*, dalam persen.

### `swing_trailing_trigger_pct`
- **Deskripsi**: Persentase profit yang memicu aktivasi *trailing* pada mode *swing*.

## Filter Market & Limitasi Trading

### `min_bb_width`
- **Deskripsi**: Lebar minimal *Bollinger Band* (persen dari harga).
- **Fungsi**: Jika pasar terlalu sepi (*sideways*, lebar BB sempit), sinyal tidak dikeluarkan untuk menghindari *noise*/*fake breakout*.

### `max_trades_per_day`
- **Deskripsi**: Batas maksimal jumlah trade per hari untuk simbol ini.
- **Fungsi**: Mengontrol *overtrading*.

## Catatan Umum
- Semua parameter dapat berbeda per simbol (pair/coin).
- Kombinasi parameter menentukan karakter strategi: frekuensi sinyal, selektivitas, ketat/lebarnya SL/TP, dan cara *trailing*.
- Config JSON bersifat modular, memungkinkan penyesuaian untuk satu simbol tanpa mengubah simbol lain.