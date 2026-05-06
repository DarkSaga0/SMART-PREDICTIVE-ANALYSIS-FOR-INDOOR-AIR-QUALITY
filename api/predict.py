"""
predict.py — Tahmin motoru
Her çağrıda son 24 saatlik CSV verisini okur, feature üretir, modelden tahmin alır.
"""

import os, glob
import numpy as np
import pandas as pd
import joblib

# ── AQI hesaplama ──────────────────────────────────────────

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

# ── Veri yükleme ───────────────────────────────────────────
def load_recent_aq0(csv_file, hours=48):
    """Air Quality 0 formatını okur."""
    try:
        df = pd.read_csv(csv_file)
        if df is None or len(df) == 0:
            return None
        df['datetime'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('datetime').drop_duplicates('datetime').set_index('datetime')
        df = df[['temperature_C','humidity_Pct','dustDensity_UgM3',
                 'co_Ppm','no2_Ppb','ozoneRaw_ADC']].copy()
        df.columns = ['temperature','humidity','dust','co','no2','ozone_raw']
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

def load_recent(data_dir, hours=48):
    """
    Cihazın klasöründen en son {hours} saatlik veriyi okur.
    Son 2 günün CSV'sini alarak yeterince veri sağlar.
    """
    files = sorted(glob.glob(os.path.join(data_dir, 'hava_kalitesi_*.csv')))
    if not files:
        return None
    # Son 2 dosya yeterli (her dosya = 1 gün)
    recent_files = files[-2:]
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
    df.columns = ['temperature','humidity','dust','co','no2','ozone_raw']
    for c in ['dust','co','no2','ozone_raw']:
        df[c] = df[c].clip(lower=0)
    df = df.resample('1h').mean().dropna()
    # Son {hours} saati al
    df = df.iloc[-hours:]
    df[['aqi','dominant']] = pd.DataFrame(
        df.apply(compute_aqi, axis=1).tolist(),
        index=df.index, columns=['aqi','dominant'])
    return df

# ── Feature engineering ────────────────────────────────────

def build_features_single(df):
    """Son satır için feature vektörü üretir."""
    d = df.copy()
    d['hour']      = d.index.hour
    d['dayofweek'] = d.index.dayofweek
    d['month']     = d.index.month
    d['hour_sin']  = np.sin(2 * np.pi * d['hour'] / 24)
    d['hour_cos']  = np.cos(2 * np.pi * d['hour'] / 24)
    d['dow_sin']   = np.sin(2 * np.pi * d['dayofweek'] / 7)
    d['dow_cos']   = np.cos(2 * np.pi * d['dayofweek'] / 7)

    for lag in [1, 2, 3, 6, 12, 24]:
        d[f'aqi_lag_{lag}h'] = d['aqi'].shift(lag)

    d['aqi_roll3']     = d['aqi'].rolling(3).mean()
    d['aqi_roll6']     = d['aqi'].rolling(6).mean()
    d['aqi_roll24']    = d['aqi'].rolling(24).mean()
    d['aqi_roll6_std'] = d['aqi'].rolling(6).std()
    d['aqi_trend3']    = d['aqi'].diff(3)

    for c in ['temperature','humidity','dust','co','no2','ozone_raw']:
        d[f'{c}_lag1']  = d[c].shift(1)
        d[f'{c}_lag6']  = d[c].shift(6)
        d[f'{c}_lag24'] = d[c].shift(24)

    for c in ['dust','co','no2']:
        d[f'{c}_roll6'] = d[c].rolling(6).mean()

    return d.dropna()

# ── Ana tahmin fonksiyonu ──────────────────────────────────

class Predictor:
    def __init__(self, model_dir):
        model_path = os.path.join(model_dir, 'aqi_model.pkl')
        feat_path  = os.path.join(model_dir, 'feature_cols.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model bulunamadı: {model_path}")
        self.model     = joblib.load(model_path)
        self.feat_cols = joblib.load(feat_path)

    def predict(self, data_dir):
        """
        data_dir: cihaza ait CSV klasörü
        Döner: { predicted_aqi, predicted_label, current_aqi,
                 confidence, dominant, error }
        """
        df = load_recent(data_dir)
        if df is None or len(df) < 25:
            return {
                'predicted_aqi':   None,
                'predicted_label': None,
                'current_aqi':     None,
                'confidence':      'no_data',
                'dominant':        None,
                'error': 'Yeterli veri yok (en az 25 saatlik veri gerekli)'
            }

        df_feat = build_features_single(df)
        if len(df_feat) == 0:
            return {'error': 'Feature üretilemedi', 'predicted_aqi': None}

        # Modelde olmayan feature'ları sıfırla, eksikleri doldur
        row = df_feat.iloc[[-1]].copy()
        for col in self.feat_cols:
            if col not in row.columns:
                row[col] = 0.0
        row = row[self.feat_cols]

        pred = float(self.model.predict(row)[0])
        pred = max(0, round(pred))

        current_aqi = float(df['aqi'].iloc[-1])
        n_hours     = len(df)
        confidence  = 'high' if n_hours >= 168 else \
                      'medium' if n_hours >= 48 else 'low'

        return {
            'predicted_aqi':   pred,
            'predicted_label': aqi_label(pred),
            'current_aqi':     round(current_aqi, 1),
            'current_label':   aqi_label(current_aqi),
            'confidence':      confidence,
            'dominant':        str(df['dominant'].iloc[-1]),
            'hours_of_data':   n_hours,
            'error':           None
        }

    def predict_aq0(self, csv_file):
        df = load_recent_aq0(csv_file)
        if df is None or len(df) < 25:
            return {
                'predicted_aqi':   None,
                'predicted_label': None,
                'current_aqi':     None,
                'confidence':      'no_data',
                'dominant':        None,
                'error': 'Yeterli veri yok (en az 25 saatlik veri gerekli)'
            }
        df_feat = build_features_single(df)
        if len(df_feat) == 0:
            return {'error': 'Feature üretilemedi', 'predicted_aqi': None}
        row = df_feat.iloc[[-1]].copy()
        for col in self.feat_cols:
            if col not in row.columns:
                row[col] = 0.0
        row = row[self.feat_cols]
        pred = float(self.model.predict(row)[0])
        pred = max(0, round(pred))
        current_aqi = float(df['aqi'].iloc[-1])
        n_hours = len(df)
        confidence = 'high' if n_hours >= 168 else \
                     'medium' if n_hours >= 48 else 'low'
        return {
            'predicted_aqi':   pred,
            'predicted_label': aqi_label(pred),
            'current_aqi':     round(current_aqi, 1),
            'current_label':   aqi_label(current_aqi),
            'confidence':      confidence,
            'dominant':        str(df['dominant'].iloc[-1]),
            'hours_of_data':   n_hours,
            'error':           None
        }

