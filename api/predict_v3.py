"""
predict_v3.py — Haftalık Tahmin Motoru
---------------------------------------
Geçen haftanın aynı gününe bakarak bu haftayı tahmin eder.
"Bu pazartesi AQI 75 olması bekleniyor" tarzında açıklama üretir.
"""

import os, glob
import numpy as np
import pandas as pd
import joblib

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

def load_recent_aq2(data_dir, hours=500):
    files = sorted(glob.glob(os.path.join(data_dir, 'hava_kalitesi_*.csv')))
    if not files:
        return None
    recent_files = files[-4:]
    dfs = []
    for f in recent_files:
        try: dfs.append(pd.read_csv(f))
        except Exception: continue
    if not dfs: return None
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

def load_recent_aq0(csv_file, hours=500):
    try:
        df = pd.read_csv(csv_file)
        if len(df) == 0: return None
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

def build_features_single(df):
    d = df.copy()
    d['hour']      = d.index.hour
    d['dayofweek'] = d.index.dayofweek
    d['month']     = d.index.month
    d['hour_sin']  = np.sin(2*np.pi*d['hour']/24)
    d['hour_cos']  = np.cos(2*np.pi*d['hour']/24)
    d['dow_sin']   = np.sin(2*np.pi*d['dayofweek']/7)
    d['dow_cos']   = np.cos(2*np.pi*d['dayofweek']/7)
    d['week_phase_sin'] = np.sin(2*np.pi*(d['dayofweek']*24+d['hour'])/(7*24))
    d['week_phase_cos'] = np.cos(2*np.pi*(d['dayofweek']*24+d['hour'])/(7*24))

    for lag in [1, 2, 3, 6, 12, 24]:
        d[f'aqi_lag_{lag}h'] = d['aqi'].shift(lag)
    d['aqi_lag_168h'] = d['aqi'].shift(168)
    d['aqi_lag_336h'] = d['aqi'].shift(336)
    d['aqi_roll24']   = d['aqi'].rolling(24).mean()
    d['aqi_roll168']  = d['aqi'].rolling(168).mean()
    d['aqi_trend24']  = d['aqi'].diff(24)
    d['aqi_trend168'] = d['aqi'].diff(168)

    for c in SENSORS:
        d[f'{c}_lag_24h']  = d[c].shift(24)
        d[f'{c}_lag_168h'] = d[c].shift(168)
        d[f'{c}_roll24']   = d[c].rolling(24).mean()
        d[f'{c}_roll168']  = d[c].rolling(168).mean()
        d[f'{c}_trend168'] = d[c].diff(168)

    return d.dropna()

# ══════════════════════════════════════════
# AÇIKLAMA ÜRETİCİ
# ══════════════════════════════════════════

def explain_weekly(sensor, current_val, predicted_val, aqi_current, aqi_predicted):
    sensor_labels = {
        'dust':        'PM2.5 (Toz)',
        'co':          'CO (Karbonmonoksit)',
        'no2':         'NO₂ (Nitrojendioksit)',
        'temperature': 'Sıcaklık',
        'humidity':    'Nem',
        'ozone_raw':   'Ozon',
    }
    sensor_units = {
        'dust': 'µg/m³', 'co': 'ppm', 'no2': 'ppb',
        'temperature': '°C', 'humidity': '%', 'ozone_raw': 'ppb'
    }
    thresholds = {
        'dust': 2.0, 'co': 0.2, 'no2': 3.0,
        'temperature': 1.0, 'humidity': 3.0, 'ozone_raw': 3.0
    }
    if current_val is None or predicted_val is None:
        return None
    diff = predicted_val - current_val
    if abs(diff) < thresholds.get(sensor, 1.0):
        return None
    pct       = (diff / current_val * 100) if current_val != 0 else 0
    direction = '▲' if diff > 0 else '▼'
    effect    = 'artış' if diff > 0 else 'düşüş'
    label     = sensor_labels.get(sensor, sensor)
    unit      = sensor_units.get(sensor, '')

    return {
        'sensor':     sensor,
        'label':      label,
        'current':    round(current_val, 2),
        'predicted':  round(predicted_val, 2),
        'diff':       round(diff, 2),
        'pct':        round(pct, 1),
        'unit':       unit,
        'direction':  direction,
        'effect':     effect,
        'aqi_impact': round(aqi_predicted - aqi_current, 1),
        'summary':    f"{direction} {label}: {current_val:.1f} → {predicted_val:.1f} {unit}",
    }

# ══════════════════════════════════════════
# ANA TAHMİN SINIFI
# ══════════════════════════════════════════

class PredictorV3:
    def __init__(self, model_dir):
        self.models    = {}
        self.feat_cols = {}
        for target in TARGETS:
            mp = os.path.join(model_dir, f'model_{target}.pkl')
            fp = os.path.join(model_dir, f'features_{target}.pkl')
            if not os.path.exists(mp):
                raise FileNotFoundError(f"Model bulunamadı: {mp}")
            self.models[target]    = joblib.load(mp)
            self.feat_cols[target] = joblib.load(fp)

    def _predict_target(self, df, target):
        df_feat = build_features_single(df)
        if len(df_feat) == 0:
            return None
        row = df_feat.iloc[[-1]].copy()
        fc  = self.feat_cols[target]
        for col in fc:
            if col not in row.columns:
                row[col] = 0.0
        row = row[fc]
        return max(0, round(float(self.models[target].predict(row)[0]), 2))

    def predict(self, data_dir=None, csv_file=None, fmt='aq2'):
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
                'target_day':         None,
                'target_day_tr':      None,
                'error': 'Yeterli veri yok'
            }

        current_aqi    = float(df['aqi'].iloc[-1])
        n_hours        = len(df)
        confidence     = 'high'   if n_hours >= 336 else \
                         'medium' if n_hours >= 168 else 'low'

        # Tahmin yapılacak günü hesapla (7 gün sonrası)
        last_ts       = df.index[-1]
        target_ts     = last_ts + pd.Timedelta(hours=168)
        target_dow    = target_ts.dayofweek
        target_day_tr = DAYS[target_dow]
        target_date   = target_ts.strftime('%d %B')

        # AQI tahmini
        predicted_aqi = self._predict_target(df, 'aqi')
        if predicted_aqi is None:
            predicted_aqi = current_aqi

        # Sensör tahminleri
        sensor_preds   = {}
        sensor_current = {}
        for s in SENSORS:
            sensor_current[s] = float(df[s].iloc[-1])
            pred = self._predict_target(df, s)
            sensor_preds[s]   = pred if pred is not None else sensor_current[s]

        # Açıklamalar
        explanations = []
        for s in ['dust','co','no2','ozone_raw','temperature','humidity']:
            exp = explain_weekly(
                s, sensor_current[s], sensor_preds[s],
                current_aqi, predicted_aqi
            )
            if exp:
                explanations.append(exp)

        # Baskın kirletici tahmini
        aqi_scores = {
            'dust': aqi_pm25(sensor_preds['dust']),
            'co':   aqi_co(sensor_preds['co']),
            'no2':  aqi_no2(sensor_preds['no2']),
        }
        predicted_dominant = max(aqi_scores, key=aqi_scores.get)

        # Özet cümle
        diff = predicted_aqi - current_aqi
        if abs(diff) < 3:
            summary = f"Bu {target_day_tr} ({target_date}) hava kalitesinin stabil kalması bekleniyor."
        elif diff > 0:
            summary = f"Bu {target_day_tr} ({target_date}) AQI {predicted_aqi:.0f} ile artış göstermesi bekleniyor."
        else:
            summary = f"Bu {target_day_tr} ({target_date}) AQI {predicted_aqi:.0f} ile iyileşme bekleniyor."

        return {
            'predicted_aqi':      round(predicted_aqi),
            'predicted_label':    aqi_label(predicted_aqi),
            'current_aqi':        round(current_aqi, 1),
            'current_label':      aqi_label(current_aqi),
            'confidence':         confidence,
            'dominant':           str(df['dominant'].iloc[-1]),
            'predicted_dominant': predicted_dominant,
            'hours_of_data':      n_hours,
            'target_day':         target_ts.strftime('%Y-%m-%d'),
            'target_day_tr':      target_day_tr,
            'target_date':        target_date,
            'summary':            summary,
            'sensor_predictions': {s: round(v, 2) for s, v in sensor_preds.items()},
            'sensor_current':     {s: round(v, 2) for s, v in sensor_current.items()},
            'explanations':       explanations,
            'error':              None,
        }
