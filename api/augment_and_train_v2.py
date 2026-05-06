"""
augment_and_train_v2.py — Çoklu Sensör Tahmin Modeli
-----------------------------------------------------
Her cihaz için 7 model eğitir:
  - AQI, PM2.5, CO, NO2, sıcaklık, nem, ozon

Son 24 saatin verisiyle sonraki 24 saati tahmin eder.

Kullanım:
  # Air Quality 2 için:
  python3 augment_and_train_v2.py \
    --csv_dir /home/ibrahim/hava_kalitesi_kayitlar_2 \
    --out_dir /home/ibrahim/aqm_system/model/aqm02 \
    --format aq2

  # Air Quality 0 için:
  python3 augment_and_train_v2.py \
    --csv_file /home/ibrahim/data/Air_Quality_0_log.csv \
    --out_dir /home/ibrahim/aqm_system/model/aqm00 \
    --format aq0

  # Air Quality 3 için:
  python3 augment_and_train_v2.py \
    --csv_dir /home/ibrahim/hava_kalitesi_kayitlar_3 \
    --out_dir /home/ibrahim/aqm_system/model/aqm03 \
    --format aq2
"""

import os, glob, argparse, warnings
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')
np.random.seed(42)

# ══════════════════════════════════════════
# AQI HESAPLAMA
# ══════════════════════════════════════════

def _linear(alo, ahi, blo, bhi, v):
    return (ahi - alo) / (bhi - blo) * (v - blo) + alo

def aqi_pm25(c):
    c = round(max(0, c), 1)
    for blo, bhi, alo, ahi in [
        (0.0,12.0,0,50),(12.1,35.4,51,100),(35.5,55.4,101,150),
        (55.5,150.4,151,200),(150.5,250.4,201,300),(250.5,350.4,301,400),(350.5,500.4,401,500)]:
        if blo <= c <= bhi: return _linear(alo, ahi, blo, bhi, c)
    return 500.0

def aqi_co(c):
    c = max(0, c)
    for blo, bhi, alo, ahi in [
        (0,4.4,0,50),(4.5,9.4,51,100),(9.5,12.4,101,150),
        (12.5,15.4,151,200),(15.5,30.4,201,300),(30.5,40.4,301,400),(40.5,50.4,401,500)]:
        if blo <= c <= bhi: return _linear(alo, ahi, blo, bhi, c)
    return 500.0

def aqi_no2(c):
    c = max(0, c)
    for blo, bhi, alo, ahi in [
        (0,53,0,50),(54,100,51,100),(101,360,101,150),
        (361,649,151,200),(650,1249,201,300),(1250,1649,301,400),(1650,2049,401,500)]:
        if blo <= c <= bhi: return _linear(alo, ahi, blo, bhi, c)
    return 500.0

def compute_aqi(row):
    scores = {
        'PM2.5': aqi_pm25(row['dust']),
        'CO':    aqi_co(row['co']),
        'NO2':   aqi_no2(row['no2']),
    }
    dominant = max(scores, key=scores.get)
    return round(max(scores.values()), 1), dominant

def aqi_label(v):
    if v <= 50:  return 'İyi'
    if v <= 100: return 'Orta'
    if v <= 150: return 'Hassas Gruplar'
    if v <= 200: return 'Sağlıksız'
    if v <= 300: return 'Çok Sağlıksız'
    return 'Tehlikeli'

# ══════════════════════════════════════════
# VERİ YÜKLEME
# ══════════════════════════════════════════

def load_aq2(csv_dir):
    """hava_kalitesi_*.csv formatı (AQ2 ve AQ3)"""
    files = sorted(glob.glob(os.path.join(csv_dir, 'hava_kalitesi_*.csv')))
    if not files:
        raise FileNotFoundError(f"CSV bulunamadı: {csv_dir}")
    print(f"  {len(files)} CSV dosyası yükleniyor...")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['datetime'] = pd.to_datetime(df['tarih'] + ' ' + df['saat'])
    df = df.sort_values('datetime').drop_duplicates('datetime').set_index('datetime')
    df = df[['temperature_filtered','humidity_filtered','dust_filtered',
             'co_filtered','no2_filtered','ozoneRaw_ADC']].copy()
    df.columns = ['temperature','humidity','dust','co','no2','ozone_raw']
    return df

def load_aq0(csv_file):
    """Air_Quality_0_log.csv formatı"""
    print(f"  CSV dosyası yükleniyor: {csv_file}")
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('datetime').drop_duplicates('datetime').set_index('datetime')
    df = df[['temperature_C','humidity_Pct','dustDensity_UgM3',
             'co_Ppm','no2_Ppb','ozoneRaw_ADC']].copy()
    df.columns = ['temperature','humidity','dust','co','no2','ozone_raw']
    return df

def clean_and_resample(df):
    for c in ['dust','co','no2','ozone_raw']:
        df[c] = df[c].clip(lower=0)
    df = df.resample('1h').mean().dropna()
    df[['aqi','dominant']] = pd.DataFrame(
        df.apply(compute_aqi, axis=1).tolist(),
        index=df.index, columns=['aqi','dominant'])
    print(f"  Gerçek veri: {len(df)} saat  |  AQI ort: {df['aqi'].mean():.1f}")
    return df

# ══════════════════════════════════════════
# SENTETİK VERİ ÜRETME
# ══════════════════════════════════════════

def augment(df_real, target_days=30):
    sensors = ['temperature','humidity','dust','co','no2','ozone_raw']
    stats   = {c: {'mean': df_real[c].mean(), 'std': df_real[c].std(),
                   'min':  df_real[c].min(),  'max': df_real[c].max()}
               for c in sensors}

    hourly_pattern = df_real.groupby([df_real.index.dayofweek, df_real.index.hour])[sensors].mean()
    global_mean    = df_real[sensors].mean()

    start = df_real.index[-1] + pd.Timedelta(hours=1)
    idx   = pd.date_range(start, periods=target_days*24, freq='1h')

    rows = []
    prev = {c: df_real[c].iloc[-1] for c in sensors}

    for ts in idx:
        h   = ts.hour
        dow = ts.dayofweek
        row = {}
        for c in sensors:
            key  = (dow, h)
            base = hourly_pattern.loc[key, c] if key in hourly_pattern.index else global_mean[c]
            alpha = 0.35
            noise = np.random.normal(0, stats[c]['std'] * 0.15)
            val   = alpha * prev[c] + (1 - alpha) * base + noise
            val   = float(np.clip(val, stats[c]['min']*0.7, stats[c]['max']*1.15))
            row[c]  = round(val, 3)
            prev[c] = val
        rows.append(row)

    df_syn = pd.DataFrame(rows, index=idx)
    for c in ['dust','co','no2','ozone_raw']:
        df_syn[c] = df_syn[c].clip(lower=0)
    df_syn[['aqi','dominant']] = pd.DataFrame(
        df_syn.apply(compute_aqi, axis=1).tolist(),
        index=df_syn.index, columns=['aqi','dominant'])
    print(f"  Sentetik veri: {len(df_syn)} saat  |  AQI ort: {df_syn['aqi'].mean():.1f}")
    return df_syn

# ══════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════

SENSORS = ['temperature','humidity','dust','co','no2','ozone_raw']
TARGETS = ['aqi'] + SENSORS  # 7 hedef

def build_features(df, target_col):
    """
    Belirli bir hedef için feature matrisi üretir.
    Son 24 saatin verilerini kullanır, 24 saat sonrasını tahmin eder.
    """
    d = df.copy()

    # Zaman özellikleri
    d['hour']      = d.index.hour
    d['dayofweek'] = d.index.dayofweek
    d['month']     = d.index.month
    d['hour_sin']  = np.sin(2*np.pi*d['hour']/24)
    d['hour_cos']  = np.cos(2*np.pi*d['hour']/24)
    d['dow_sin']   = np.sin(2*np.pi*d['dayofweek']/7)
    d['dow_cos']   = np.cos(2*np.pi*d['dayofweek']/7)
    d['week_phase_sin'] = np.sin(2*np.pi*(d['dayofweek']*24+d['hour'])/(7*24))
    d['week_phase_cos'] = np.cos(2*np.pi*(d['dayofweek']*24+d['hour'])/(7*24))

    # AQI lag'leri
    for lag in [1,2,3,6,12,24]:
        d[f'aqi_lag_{lag}h'] = d['aqi'].shift(lag)
    d['aqi_lag_48h']  = d['aqi'].shift(48)
    d['aqi_lag_168h'] = d['aqi'].shift(168)
    d['aqi_roll3']    = d['aqi'].rolling(3).mean()
    d['aqi_roll6']    = d['aqi'].rolling(6).mean()
    d['aqi_roll24']   = d['aqi'].rolling(24).mean()
    d['aqi_trend3']   = d['aqi'].diff(3)
    d['aqi_trend24']  = d['aqi'].diff(24)

    # Her sensör için lag'ler
    for c in SENSORS:
        for lag in [1,3,6,12,24]:
            d[f'{c}_lag_{lag}h'] = d[c].shift(lag)
        d[f'{c}_lag_48h']  = d[c].shift(48)
        d[f'{c}_roll6']    = d[c].rolling(6).mean()
        d[f'{c}_roll24']   = d[c].rolling(24).mean()
        d[f'{c}_trend6']   = d[c].diff(6)
        d[f'{c}_trend24']  = d[c].diff(24)

    # Hedef: 24 saat sonraki değer
    d['target'] = d[target_col].shift(-24)
    d = d.dropna()
    return d

# ══════════════════════════════════════════
# MODEL EĞİTİMİ
# ══════════════════════════════════════════

def train_one(df_feat, target_name):
    drop_cols = ['target','aqi','dominant'] + SENSORS
    feat_cols = [c for c in df_feat.columns if c not in drop_cols]
    X = df_feat[feat_cols]
    y = df_feat['target']

    tscv = TimeSeriesSplit(n_splits=3)
    maes, r2s = [], []
    all_preds, all_true = [], []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        m = XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=3, random_state=42, n_jobs=-1)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        preds = np.clip(m.predict(X_val), 0, None)
        mae = mean_absolute_error(y_val, preds)
        r2  = r2_score(y_val, preds)
        maes.append(mae); r2s.append(r2)
        all_preds.extend(preds); all_true.extend(y_val)
        print(f"    [{target_name}] Fold {fold+1}: MAE={mae:.2f}  R²={r2:.3f}")

    final = XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=3, random_state=42, n_jobs=-1)
    final.fit(X, y)
    return final, feat_cols, np.mean(maes), np.mean(r2s)

def train_all(df_all, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    results = {}

    for target in TARGETS:
        print(f"\n  ── {target.upper()} modeli eğitiliyor...")
        df_feat = build_features(df_all, target)
        model, feat_cols, mae, r2 = train_one(df_feat, target)

        model_path = os.path.join(out_dir, f'model_{target}.pkl')
        feat_path  = os.path.join(out_dir, f'features_{target}.pkl')
        joblib.dump(model,     model_path)
        joblib.dump(feat_cols, feat_path)
        results[target] = {'mae': mae, 'r2': r2}
        print(f"    ✓ MAE={mae:.2f}  R²={r2:.3f}  → {model_path}")

    return results

# ══════════════════════════════════════════
# GRAFİK
# ══════════════════════════════════════════

def make_plots(df_real, df_syn, results, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.patch.set_facecolor('#0a0b0d')
    for ax in axes.flat:
        ax.set_facecolor('#0f1113')
        ax.tick_params(colors='#6b7280', labelsize=8)
        ax.spines[:].set_color('#1f2125')
        ax.xaxis.label.set_color('#6b7280')
        ax.yaxis.label.set_color('#6b7280')
        ax.title.set_color('#e8eaed')

    # 1. AQI zaman serisi
    ax = axes[0, 0]
    ax.plot(df_real.index, df_real['aqi'], color='#60a5fa', lw=1.5, label='Gerçek')
    ax.plot(df_syn.index,  df_syn['aqi'],  color='#4ade80', lw=1, alpha=0.7, label='Sentetik')
    ax.axvline(df_real.index[-1], color='#fbbf24', lw=1, ls='--')
    ax.set_title('AQI — Gerçek vs Sentetik', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, facecolor='#1a1c1f', edgecolor='#3c4043', labelcolor='#e8eaed')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    fig.autofmt_xdate()

    # 2. Model performansları (MAE)
    ax = axes[0, 1]
    targets = list(results.keys())
    maes    = [results[t]['mae'] for t in targets]
    colors  = ['#f87171' if t == 'aqi' else '#60a5fa' for t in targets]
    ax.barh(targets, maes, color=colors, height=0.6)
    ax.set_title('Model Hataları (MAE)\n(kırmızı=AQI, mavi=sensör)', fontsize=10, fontweight='bold')
    ax.set_xlabel('MAE')
    for i, (t, v) in enumerate(zip(targets, maes)):
        ax.text(v + 0.1, i, f'{v:.2f}', va='center', color='#e8eaed', fontsize=8)

    # 3. R² skoru
    ax = axes[1, 0]
    r2s = [results[t]['r2'] for t in targets]
    ax.barh(targets, r2s, color='#c084fc', height=0.6)
    ax.set_title('Model R² Skoru (1.0 = mükemmel)', fontsize=10, fontweight='bold')
    ax.set_xlabel('R²')
    ax.set_xlim(0, 1.1)
    for i, (t, v) in enumerate(zip(targets, r2s)):
        ax.text(v + 0.01, i, f'{v:.3f}', va='center', color='#e8eaed', fontsize=8)

    # 4. Haftalık AQI örüntüsü
    ax = axes[1, 1]
    days = ['Pzt','Sal','Çar','Per','Cum','Cmt','Paz']
    colors_day = ['#60a5fa','#4ade80','#fbbf24','#c084fc','#f87171','#5eead4','#fb923c']
    for dow in range(7):
        mask = df_real.index.dayofweek == dow
        if mask.sum() > 0:
            hourly = df_real[mask].groupby(df_real[mask].index.hour)['aqi'].mean()
            ax.plot(hourly.index, hourly.values, label=days[dow],
                    color=colors_day[dow], lw=1.5, alpha=0.8)
    ax.set_xlabel('Saat')
    ax.set_ylabel('Ortalama AQI')
    ax.set_title('Haftalık AQI Örüntüsü', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, facecolor='#1a1c1f', edgecolor='#3c4043', labelcolor='#e8eaed', ncol=2)
    ax.set_xticks(range(0, 24, 3))

    plt.suptitle('AQI Çoklu Sensör Tahmin Modeli — Performans Raporu',
                 fontsize=13, fontweight='bold', color='#e8eaed', y=1.01)
    plt.tight_layout()
    plot_path = os.path.join(out_dir, 'model_report.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight',
                facecolor='#0a0b0d', edgecolor='none')
    plt.close()
    print(f"\n  ✓ Grafik kaydedildi → {plot_path}")

# ══════════════════════════════════════════
# ANA FONKSİYON
# ══════════════════════════════════════════

def main(csv_dir, csv_file, out_dir, fmt, synth_days):
    print("\n" + "="*50)
    print("ADIM 1 — Gerçek veri yükleniyor")
    print("="*50)

    if fmt == 'aq0':
        df_raw = load_aq0(csv_file)
    else:
        df_raw = load_aq2(csv_dir)

    df_real = clean_and_resample(df_raw)

    print("\n" + "="*50)
    print(f"ADIM 2 — {synth_days} günlük sentetik veri üretiliyor")
    print("="*50)
    df_syn = augment(df_real, target_days=synth_days)

    print("\n" + "="*50)
    print("ADIM 3 — Veri birleştiriliyor")
    print("="*50)
    df_all = pd.concat([df_real, df_syn])
    print(f"  Toplam: {len(df_all)} saat")

    print("\n" + "="*50)
    print("ADIM 4 — 7 model eğitiliyor (AQI + 6 sensör)")
    print("="*50)
    results = train_all(df_all, out_dir)

    print("\n" + "="*50)
    print("ADIM 5 — Grafikler üretiliyor")
    print("="*50)
    make_plots(df_real, df_syn, results, out_dir)

    print("\n" + "="*50)
    print("SONUÇ")
    print("="*50)
    for t, r in results.items():
        print(f"  {t:12s}: MAE={r['mae']:.2f}  R²={r['r2']:.3f}")
    print(f"\n  Model klasörü: {out_dir}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--csv_dir',    default=None)
    p.add_argument('--csv_file',   default=None)
    p.add_argument('--out_dir',    required=True)
    p.add_argument('--format',     choices=['aq0','aq2'], default='aq2')
    p.add_argument('--synth_days', type=int, default=30)
    a = p.parse_args()

    if a.format == 'aq0' and not a.csv_file:
        p.error("--format aq0 için --csv_file gerekli")
    if a.format == 'aq2' and not a.csv_dir:
        p.error("--format aq2 için --csv_dir gerekli")

    main(a.csv_dir, a.csv_file, a.out_dir, a.format, a.synth_days)
