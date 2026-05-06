"""
augment_and_train_v3.py — Haftalık Periyodik Tahmin Modeli
----------------------------------------------------------
Hoca önerisi: geçen haftanın aynı gününe bakarak bu haftayı tahmin et.
  - Hedef: 168 saat (7 gün) sonrası
  - Sentetik veri: gerçek verinin haftalık örüntüsüne yakın

Kullanım:
  # Air Quality 2:
  python3 augment_and_train_v3.py \
    --csv_dir /home/ibrahim/hava_kalitesi_kayitlar_2 \
    --out_dir /home/ibrahim/aqm_system/model/aqm02 --format aq2

  # Air Quality 0:
  python3 augment_and_train_v3.py \
    --csv_file /home/ibrahim/data/Air_Quality_0_log.csv \
    --out_dir /home/ibrahim/aqm_system/model/aqm00 --format aq0

  # Air Quality 3:
  python3 augment_and_train_v3.py \
    --csv_dir /home/ibrahim/hava_kalitesi_kayitlar_3 \
    --out_dir /home/ibrahim/aqm_system/model/aqm03 --format aq2
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

SENSORS = ['temperature','humidity','dust','co','no2','ozone_raw']
TARGETS = ['aqi'] + SENSORS
DAYS    = ['Pazartesi','Salı','Çarşamba','Perşembe','Cuma','Cumartesi','Pazar']

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
    files = sorted(glob.glob(os.path.join(csv_dir, 'hava_kalitesi_*.csv')))
    if not files:
        raise FileNotFoundError(f"CSV bulunamadı: {csv_dir}")
    print(f"  {len(files)} CSV dosyası yükleniyor...")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df['datetime'] = pd.to_datetime(df['tarih'] + ' ' + df['saat'])
    df = df.sort_values('datetime').drop_duplicates('datetime').set_index('datetime')
    df = df[['temperature_filtered','humidity_filtered','dust_filtered',
             'co_filtered','no2_filtered','ozoneRaw_ADC']].copy()
    df.columns = SENSORS
    return df

def load_aq0(csv_file):
    print(f"  CSV dosyası yükleniyor: {csv_file}")
    df = pd.read_csv(csv_file)
    df['datetime'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('datetime').drop_duplicates('datetime').set_index('datetime')
    df = df[['temperature_C','humidity_Pct','dustDensity_UgM3',
             'co_Ppm','no2_Ppb','ozoneRaw_ADC']].copy()
    df.columns = SENSORS
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
# SENTETİK VERİ — HAFTALIK ÖRÜNTÜYE YAKIN
# ══════════════════════════════════════════

def augment(df_real, target_weeks=8):
    """
    Gerçek verinin haftalık gün×saat örüntüsünü koruyarak
    sentetik veri üretir. Hoca şartı: gerçeğe yakın olmalı.
    """
    stats = {c: {
        'mean': df_real[c].mean(),
        'std':  df_real[c].std(),
        'min':  df_real[c].min(),
        'max':  df_real[c].max()
    } for c in SENSORS}

    # Gün × saat bazlı ortalama ve std (haftalık periyodisiteyi yakala)
    hourly_mean = df_real.groupby([df_real.index.dayofweek, df_real.index.hour])[SENSORS].mean()
    hourly_std  = df_real.groupby([df_real.index.dayofweek, df_real.index.hour])[SENSORS].std().fillna(0)
    global_mean = df_real[SENSORS].mean()
    global_std  = df_real[SENSORS].std()

    start = df_real.index[-1] + pd.Timedelta(hours=1)
    idx   = pd.date_range(start, periods=target_weeks*7*24, freq='1h')

    rows = []
    prev = {c: df_real[c].iloc[-1] for c in SENSORS}

    for ts in idx:
        dow = ts.dayofweek
        h   = ts.hour
        row = {}
        for c in SENSORS:
            key = (dow, h)
            if key in hourly_mean.index:
                base     = hourly_mean.loc[key, c]
                local_std = hourly_std.loc[key, c]
            else:
                base      = global_mean[c]
                local_std = global_std[c]

            # Gürültü: gerçek standart sapmanın %20'si (çok fazla sapmıyor)
            noise = np.random.normal(0, local_std * 0.20)
            # AR(1) sürekliliği: önceki değerle yumuşatma
            alpha = 0.25
            val   = alpha * prev[c] + (1 - alpha) * base + noise
            # Gerçek verinin min-max aralığında tut (±%10 tolerans)
            val   = float(np.clip(val,
                                  stats[c]['min'] * 0.90,
                                  stats[c]['max'] * 1.10))
            row[c]  = round(val, 3)
            prev[c] = val
        rows.append(row)

    df_syn = pd.DataFrame(rows, index=idx)
    for c in ['dust','co','no2','ozone_raw']:
        df_syn[c] = df_syn[c].clip(lower=0)
    df_syn[['aqi','dominant']] = pd.DataFrame(
        df_syn.apply(compute_aqi, axis=1).tolist(),
        index=df_syn.index, columns=['aqi','dominant'])

    # İstatistik karşılaştırması
    print(f"  Sentetik veri: {len(df_syn)} saat  |  AQI ort: {df_syn['aqi'].mean():.1f}")
    print(f"\n  Gerçek vs Sentetik karşılaştırması:")
    for c in ['aqi','dust','co','no2']:
        rm = df_real[c].mean() if c != 'aqi' else df_real['aqi'].mean()
        sm = df_syn[c].mean()
        print(f"    {c:12s}: gerçek={rm:.2f}  sentetik={sm:.2f}  fark={abs(rm-sm):.2f}")

    return df_syn

# ══════════════════════════════════════════
# FEATURE ENGINEERING — HAFTALİK HEDEF
# ══════════════════════════════════════════

def build_features(df, target_col):
    """
    Hedef: 168 saat (7 gün) sonraki değer
    Feature: geçen haftanın aynı günü ağırlıklı
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

    # AQI lag'leri — kısa vade
    for lag in [1, 2, 3, 6, 12, 24]:
        d[f'aqi_lag_{lag}h'] = d['aqi'].shift(lag)

    # AQI lag'leri — haftalık (EN ÖNEMLİ)
    d['aqi_lag_168h'] = d['aqi'].shift(168)   # geçen haftanın aynı günü/saati
    d['aqi_lag_336h'] = d['aqi'].shift(336)   # 2 hafta önce

    # Kayan ortalamalar
    d['aqi_roll24']   = d['aqi'].rolling(24).mean()    # günlük ort
    d['aqi_roll168']  = d['aqi'].rolling(168).mean()   # haftalık ort
    d['aqi_trend24']  = d['aqi'].diff(24)               # günlük trend
    d['aqi_trend168'] = d['aqi'].diff(168)              # haftalık trend

    # Her sensör için lag'ler
    for c in SENSORS:
        d[f'{c}_lag_24h']  = d[c].shift(24)
        d[f'{c}_lag_168h'] = d[c].shift(168)   # geçen haftanın aynı saati
        d[f'{c}_roll24']   = d[c].rolling(24).mean()
        d[f'{c}_roll168']  = d[c].rolling(168).mean()
        d[f'{c}_trend168'] = d[c].diff(168)

    # Hedef: 168 saat (7 gün) sonraki değer
    d['target'] = d[target_col].shift(-168)
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
        joblib.dump(model,     os.path.join(out_dir, f'model_{target}.pkl'))
        joblib.dump(feat_cols, os.path.join(out_dir, f'features_{target}.pkl'))
        results[target] = {'mae': mae, 'r2': r2}
        print(f"    ✓ MAE={mae:.2f}  R²={r2:.3f}")
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

    # 2. MAE karşılaştırması
    ax = axes[0, 1]
    targets = list(results.keys())
    maes    = [results[t]['mae'] for t in targets]
    colors  = ['#f87171' if t == 'aqi' else '#60a5fa' for t in targets]
    ax.barh(targets, maes, color=colors, height=0.6)
    ax.set_title('Model Hataları (MAE)\n7 günlük tahmin', fontsize=10, fontweight='bold')
    ax.set_xlabel('MAE')
    for i, (t, v) in enumerate(zip(targets, maes)):
        ax.text(v + 0.05, i, f'{v:.2f}', va='center', color='#e8eaed', fontsize=8)

    # 3. Haftalık AQI örüntüsü
    ax = axes[1, 0]
    colors_day = ['#60a5fa','#4ade80','#fbbf24','#c084fc','#f87171','#5eead4','#fb923c']
    for dow in range(7):
        mask = df_real.index.dayofweek == dow
        if mask.sum() > 0:
            hourly = df_real[mask].groupby(df_real[mask].index.hour)['aqi'].mean()
            ax.plot(hourly.index, hourly.values,
                    label=DAYS[dow], color=colors_day[dow], lw=1.5, alpha=0.8)
    ax.set_xlabel('Saat')
    ax.set_ylabel('Ortalama AQI')
    ax.set_title('Haftalık AQI Örüntüsü (Gerçek Veri)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, facecolor='#1a1c1f', edgecolor='#3c4043', labelcolor='#e8eaed', ncol=2)
    ax.set_xticks(range(0, 24, 3))

    # 4. Günlük AQI ortalaması
    ax = axes[1, 1]
    df_all = pd.concat([df_real, df_syn])
    daily  = df_all.resample('D')['aqi'].mean()
    ax.plot(daily.index, daily.values, color='#c084fc', lw=1.5)
    ax.axvline(df_real.index[-1], color='#fbbf24', lw=1, ls='--', label='Gerçek/Sentetik sınırı')
    ax.set_title('Günlük Ortalama AQI', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, facecolor='#1a1c1f', edgecolor='#3c4043', labelcolor='#e8eaed')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    fig.autofmt_xdate()

    plt.suptitle('Haftalık AQI Tahmin Modeli (Hoca Önerisi)',
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

def main(csv_dir, csv_file, out_dir, fmt, synth_weeks):
    print("\n" + "="*50)
    print("ADIM 1 — Gerçek veri yükleniyor")
    print("="*50)
    if fmt == 'aq0':
        df_raw = load_aq0(csv_file)
    else:
        df_raw = load_aq2(csv_dir)
    df_real = clean_and_resample(df_raw)

    print("\n" + "="*50)
    print(f"ADIM 2 — {synth_weeks} haftalık sentetik veri üretiliyor")
    print("  (haftalık gün×saat örüntüsüne yakın, gerçek aralıkta)")
    print("="*50)
    df_syn = augment(df_real, target_weeks=synth_weeks)

    print("\n" + "="*50)
    print("ADIM 3 — Veri birleştiriliyor")
    print("="*50)
    df_all = pd.concat([df_real, df_syn])
    print(f"  Toplam: {len(df_all)} saat ({len(df_all)//168} hafta)")

    print("\n" + "="*50)
    print("ADIM 4 — 7 model eğitiliyor (hedef: 7 gün sonrası)")
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
    p.add_argument('--csv_dir',     default=None)
    p.add_argument('--csv_file',    default=None)
    p.add_argument('--out_dir',     required=True)
    p.add_argument('--format',      choices=['aq0','aq2'], default='aq2')
    p.add_argument('--synth_weeks', type=int, default=8)
    a = p.parse_args()

    if a.format == 'aq0' and not a.csv_file:
        p.error("--format aq0 için --csv_file gerekli")
    if a.format == 'aq2' and not a.csv_dir:
        p.error("--format aq2 için --csv_dir gerekli")

    main(a.csv_dir, a.csv_file, a.out_dir, a.format, a.synth_weeks)
