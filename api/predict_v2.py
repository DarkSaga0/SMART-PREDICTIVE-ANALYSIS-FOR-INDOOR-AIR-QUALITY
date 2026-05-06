"""
predict_v2.py — Çoklu Sensör Tahmin Motoru
------------------------------------------
Her cihaz için 7 ayrı model kullanır:
  AQI, PM2.5, CO, NO2, sıcaklık, nem, ozon

Son 24 saatin verisiyle sonraki 24 saati tahmin eder.
Hangi sensörün ne kadar değişeceğini ve AQI'ye etkisini açıklar.
"""

import os, glob
import numpy as np
import pandas as pd
import joblib

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
    if v is None: return '—'
    if v <= 50:  return 'İyi'
    if v <= 100: return 'Orta'
    if v <= 150: return 'Hassas Gruplar'
    if v <= 200: return 'Sağlıksız'
    if v <= 300: return 'Çok Sağlıksız'
    return 'Tehlikeli'

# ══════════════════════════════════════════
# VERİ YÜKLEME
# ══════════════════════════════════════════

SENSORS = ['temperature','humidity','dust','co','no2','ozone_raw']
TARGETS = ['aqi'] + SENSORS

def load_recent_aq2(data_dir, hours=400):
    files = sorted(glob.glob(os.path.join(data_dir, 'hava_kalitesi_*.csv')))
    if not files:
        return None
    recent_files = files[-4:]  # son 4 gün yeterli
    dfs = []
    for f in recent_files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception:
            continue
    if not dfs:
        return None
    df = pd.concat(dfs, ignore_index=True)
    df['datetime'] = pd.to_datetime(df['tarih'] + ' ' + df['saat'])
    df = df.sort_values('datetime').drop_duplicates('datetime').set_index('datetime')
    df = df[['temperature_filtered','humidity_filtered','dust_filtered',
             'co_filtered','no2_filtered','ozoneRaw_ADC']].copy()
    df.columns = SENSORS
    for c in ['dust','co','no2','ozone_raw']:
        df[c] = df[c].clip(lower=0)
    df = df.resample('1h').mean().dropna()
    df = df.iloc[-hours:]
    df[['aqi','dominant']] = pd.DataFrame(
        df.apply(compute_aqi, axis=1).tolist(),
        index=df.index, columns=['aqi','dominant'])
    return df

def load_recent_aq0(csv_file, hours=400):
    try:
        df = pd.read_csv(csv_file)
        if len(df) == 0:
            return None
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('datetime').drop_duplicates('datetime').set_index('datetime')
        df = df[['temperature_C','humidity_Pct','dustDensity_UgM3',
                 'co_Ppm','no2_Ppb','ozoneRaw_ADC']].copy()
        df.columns = SENSORS
        for c in ['dust','co','no2','ozone_raw']:
            df[c] = df[c].clip(lower=0)
        df = df.resample('1h').mean().dropna()
        df = df.iloc[-hours:]
        df[['aqi','dominant']] = pd.DataFrame(
            df.apply(compute_aqi, axis=1).tolist(),
            index=df.index, columns=['aqi','dominant'])
        return df
    except Exception as e:
        print(f"AQ0 load hatası: {e}")
        return None

# ══════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════

def build_features_single(df, target_col):
    """Son satır için feature vektörü üretir."""
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

    # Sensör lag'leri
    for c in SENSORS:
        for lag in [1,3,6,12,24]:
            d[f'{c}_lag_{lag}h'] = d[c].shift(lag)
        d[f'{c}_lag_48h']  = d[c].shift(48)
        d[f'{c}_roll6']    = d[c].rolling(6).mean()
        d[f'{c}_roll24']   = d[c].rolling(24).mean()
        d[f'{c}_trend6']   = d[c].diff(6)
        d[f'{c}_trend24']  = d[c].diff(24)

    return d.dropna()

# ══════════════════════════════════════════
# TAHMİN AÇIKLAMA ÜRETİCİ
# ══════════════════════════════════════════

def explain_change(sensor, current_val, predicted_val, aqi_current, aqi_predicted):
    """
    Sensör değişimini ve AQI'ye etkisini açıklar.
    """
    if current_val is None or predicted_val is None:
        return None

    diff     = predicted_val - current_val
    pct      = (diff / current_val * 100) if current_val != 0 else 0
    aqi_diff = aqi_predicted - aqi_current

    sensor_labels = {
        'dust':        'PM2.5 (Toz)',
        'co':          'CO (Karbonmonoksit)',
        'no2':         'NO₂ (Nitrojendioksit)',
        'temperature': 'Sıcaklık',
        'humidity':    'Nem',
        'ozone_raw':   'Ozon',
    }

    sensor_units = {
        'dust':        'µg/m³',
        'co':          'ppm',
        'no2':         'ppb',
        'temperature': '°C',
        'humidity':    '%',
        'ozone_raw':   'ppb',
    }

    # Sadece anlamlı değişimleri raporla
    thresholds = {
        'dust': 2.0, 'co': 0.3, 'no2': 5.0,
        'temperature': 1.0, 'humidity': 3.0, 'ozone_raw': 5.0
    }

    if abs(diff) < thresholds.get(sensor, 1.0):
        return None

    label = sensor_labels.get(sensor, sensor)
    unit  = sensor_units.get(sensor, '')

    direction = '▲' if diff > 0 else '▼'
    effect    = 'artış' if diff > 0 else 'düşüş'

    return {
        'sensor':        sensor,
        'label':         label,
        'current':       round(current_val, 2),
        'predicted':     round(predicted_val, 2),
        'diff':          round(diff, 2),
        'pct':           round(pct, 1),
        'unit':          unit,
        'direction':     direction,
        'effect':        effect,
        'aqi_impact':    round(aqi_diff, 1),
        'summary':       f"{direction} {label}: {current_val:.1f} → {predicted_val:.1f} {unit} ({effect})",
    }

# ══════════════════════════════════════════
# ANA TAHMİN SINIFI
# ══════════════════════════════════════════

class PredictorV2:
    def __init__(self, model_dir):
        self.models    = {}
        self.feat_cols = {}
        self.model_dir = model_dir

        for target in TARGETS:
            model_path = os.path.join(model_dir, f'model_{target}.pkl')
            feat_path  = os.path.join(model_dir, f'features_{target}.pkl')
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model bulunamadı: {model_path}")
            self.models[target]    = joblib.load(model_path)
            self.feat_cols[target] = joblib.load(feat_path)

    def _predict_target(self, df, target):
        df_feat = build_features_single(df, target)
        if len(df_feat) == 0:
            return None
        row = df_feat.iloc[[-1]].copy()
        feat_cols = self.feat_cols[target]
        for col in feat_cols:
            if col not in row.columns:
                row[col] = 0.0
        row = row[feat_cols]
        pred = float(self.models[target].predict(row)[0])
        return max(0, round(pred, 2))

    def predict(self, data_dir=None, csv_file=None, fmt='aq2'):
        # Veriyi yükle
        if fmt == 'aq0':
            df = load_recent_aq0(csv_file)
        else:
            df = load_recent_aq2(data_dir)

        if df is None or len(df) < 25:
            return {
                'predicted_aqi':      None,
                'predicted_label':    None,
                'current_aqi':        None,
                'confidence':         'no_data',
                'dominant':           None,
                'sensor_predictions': {},
                'explanations':       [],
                'error': 'Yeterli veri yok (en az 25 saatlik veri gerekli)'
            }

        current_aqi  = float(df['aqi'].iloc[-1])
        n_hours      = len(df)
        confidence   = 'high'   if n_hours >= 168 else \
                       'medium' if n_hours >= 48  else 'low'

        # AQI tahmini
        predicted_aqi = self._predict_target(df, 'aqi')
        if predicted_aqi is None:
            predicted_aqi = current_aqi

        # Her sensör tahmini
        sensor_preds   = {}
        sensor_current = {}
        for sensor in SENSORS:
            sensor_current[sensor] = float(df[sensor].iloc[-1])
            pred = self._predict_target(df, sensor)
            sensor_preds[sensor]   = pred if pred is not None else sensor_current[sensor]

        # Açıklamalar üret
        explanations = []
        for sensor in ['dust', 'co', 'no2', 'ozone_raw', 'temperature', 'humidity']:
            exp = explain_change(
                sensor,
                sensor_current[sensor],
                sensor_preds[sensor],
                current_aqi,
                predicted_aqi
            )
            if exp:
                explanations.append(exp)

        # En etkili sensörü bul
        aqi_sensors = {
            'dust': aqi_pm25(sensor_preds['dust']),
            'co':   aqi_co(sensor_preds['co']),
            'no2':  aqi_no2(sensor_preds['no2']),
        }
        main_driver = max(aqi_sensors, key=aqi_sensors.get)

        return {
            'predicted_aqi':      round(predicted_aqi),
            'predicted_label':    aqi_label(predicted_aqi),
            'current_aqi':        round(current_aqi, 1),
            'current_label':      aqi_label(current_aqi),
            'confidence':         confidence,
            'dominant':           str(df['dominant'].iloc[-1]),
            'predicted_dominant': main_driver,
            'hours_of_data':      n_hours,
            'sensor_predictions': {
                s: round(v, 2) for s, v in sensor_preds.items()
            },
            'sensor_current': {
                s: round(v, 2) for s, v in sensor_current.items()
            },
            'explanations': explanations,
            'error': None,
        }
