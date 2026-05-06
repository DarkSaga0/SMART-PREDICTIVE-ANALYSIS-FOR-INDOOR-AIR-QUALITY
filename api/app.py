"""
app.py — Ana Flask API
----------------------
Başlatmak için:
  cd ~/aqm_system/api
  python3 app.py

Otomatik başlatma (her boot'ta):
  crontab -e  →  @reboot cd ~/aqm_system/api && python3 app.py >> ~/aqm_api.log 2>&1
"""

import os
import glob
import json
from datetime import datetime, timedelta
from flask import Flask, jsonify, Response
from flask_cors import CORS
import pandas as pd

# ── Predict modülünü import et ──
import sys
sys.path.insert(0, os.path.dirname(__file__))
from predict_v3 import PredictorV3, compute_aqi, aqi_label, load_recent_aq2, load_recent_aq0

# ══════════════════════════════════════════
# AYARLAR
# ══════════════════════════════════════════

BASE_DIR  = os.path.expanduser('~/aqm_system')
MODEL_DIR = os.path.join(BASE_DIR, 'model')

DEVICES = {
    'aqm00': {
        'name':      'AQM-00 — Air Quality 0',
        'location':  'KONUM 0',
        'data_dir':  '/home/ibrahim/hava_kalitesi_kayitlar_0',
        'csv_file':  '/home/ibrahim/data/Air_Quality_0_log.csv',
        'format':    'aq0',
        'model_dir': os.path.join(MODEL_DIR, 'aqm00'),
    },
    'aqm02': {
        'name':      'AQM-02 — Air Quality 2',
        'location':  'TEST',
        'data_dir':  '/home/ibrahim/hava_kalitesi_kayitlar_2',
        'format':    'aq2',
        'model_dir': os.path.join(MODEL_DIR, 'aqm02'),
    },
    'aqm03': {
        'name':      'AQM-03 — Air Quality 3',
        'location':  'KONUM 3',
        'data_dir':  '/home/ibrahim/hava_kalitesi_kayitlar_3',
        'format':    'aq2',
        'model_dir': os.path.join(MODEL_DIR, 'aqm03'),
    },
}

# ══════════════════════════════════════════
# FLASK KURULUMU
# ══════════════════════════════════════════

app = Flask(__name__)
CORS(app)

predictors = {}
for device_id, cfg in DEVICES.items():
    try:
        predictors[device_id] = PredictorV3(cfg['model_dir'])
        print(f"✓ Model yüklendi: {device_id} → {cfg['model_dir']}")
    except FileNotFoundError as e:
        print(f"⚠ Model bulunamadı ({device_id}): {e}")

# ══════════════════════════════════════════
# YARDIMCI FONKSİYONLAR
# ══════════════════════════════════════════

def read_latest_row_aq0(csv_file):
    try:
        df = pd.read_csv(csv_file)
        if len(df) == 0:
            return None
        row = df.iloc[-1]
        return {
            'tarih':                str(row['timestamp'])[:10],
            'saat':                 str(row['timestamp'])[11:19],
            'temperature_filtered': row['temperature_C'],
            'humidity_filtered':    row['humidity_Pct'],
            'dust_filtered':        row['dustDensity_UgM3'],
            'co_filtered':          row['co_Ppm'],
            'no2_filtered':         row['no2_Ppb'],
            'ozoneRaw_ADC':         row['ozoneRaw_ADC'],
        }
    except Exception as e:
        print(f"AQ0 okuma hatası: {e}")
        return None

def read_latest_row(data_dir):
    files = sorted(glob.glob(os.path.join(data_dir, 'hava_kalitesi_*.csv')))
    if not files:
        return None
    for f in reversed(files):
        try:
            df = pd.read_csv(f)
            if len(df) > 0:
                return df.iloc[-1]
        except Exception:
            continue
    return None

def read_history(data_dir, hours=12, fmt='aq2', csv_file=None):
    if fmt == 'aq0':
        df = load_recent_aq0(csv_file, hours=hours+2)
    else:
        df = load_recent_aq2(data_dir, hours=hours+2)
    if df is None or len(df) == 0:
        return [], []
    df = df.tail(hours)
    return (
        df['aqi'].round(1).tolist(),
        df['dust'].round(1).tolist()
    )

def read_correlation(data_dir=None, fmt='aq2', csv_file=None):
    if fmt == 'aq0':
        df = load_recent_aq0(csv_file, hours=26)
    else:
        df = load_recent_aq2(data_dir, hours=26)
    if df is None or len(df) < 5:
        return {}
    try:
        return {
            'pm_humidity': round(float(df['dust'].corr(df['humidity'])), 2),
            'co_temp':     round(float(df['co'].corr(df['temperature'])), 2),
            'no2_temp':    round(float(df['no2'].corr(df['temperature'])), 2),
            'ozone_temp':  round(float(df['ozone_raw'].corr(df['temperature'])), 2),
        }
    except Exception:
        return {}

def build_device_summary(device_id):
    cfg = DEVICES[device_id]
    if cfg.get('format') == 'aq0':
        row = read_latest_row_aq0(cfg['csv_file'])
    else:
        row = read_latest_row(cfg['data_dir'])

    if row is None:
        return {
            'id':       device_id,
            'name':     cfg['name'],
            'location': cfg['location'],
            'online':   False,
        }

    def g(col, default=0.0):
        try: return float(row.get(col, default))
        except: return default

    dust  = g('dust_filtered')
    co    = g('co_filtered')
    no2   = g('no2_filtered')
    temp  = g('temperature_filtered')
    hum   = g('humidity_filtered')
    ozone = g('ozoneRaw_ADC')

    from predict_v2 import aqi_pm25, aqi_co, aqi_no2
    scores   = {'PM2.5': aqi_pm25(dust), 'CO': aqi_co(co), 'NO2': aqi_no2(no2)}
    dominant = max(scores, key=scores.get)
    aqi_val  = round(max(scores.values()), 1)

    try:
        last_dt = datetime.strptime(str(row['tarih']) + ' ' + str(row['saat']), '%Y-%m-%d %H:%M:%S')
        online  = (datetime.now() - last_dt) < timedelta(minutes=5)
    except Exception:
        online = True

    return {
        'id':           device_id,
        'name':         cfg['name'],
        'location':     cfg['location'],
        'online':       online,
        'aqi':          aqi_val,
        'aqi_label':    aqi_label(aqi_val),
        'aqi_dominant': dominant,
        'temperature':  round(temp, 1),
        'humidity':     round(hum, 1),
        'dust':         round(dust, 1),
        'co':           round(co, 3),
        'no2':          round(no2, 1),
        'ozone_raw':    round(ozone, 1),
        'last_updated': str(row.get('tarih', '')) + 'T' + str(row.get('saat', '')),
    }

# ══════════════════════════════════════════
# ENDPOINT'LER
# ══════════════════════════════════════════

@app.route('/')
def index():
    with open(os.path.join(os.path.dirname(__file__), 'air_quality_web.html'), 'r', encoding='utf-8') as f:
        content = f.read()
    return Response(content, mimetype='text/html')

@app.route('/api/devices')
def get_devices():
    result = []
    for device_id in DEVICES:
        try:
            result.append(build_device_summary(device_id))
        except Exception as e:
            result.append({
                'id':       device_id,
                'name':     DEVICES[device_id]['name'],
                'location': DEVICES[device_id]['location'],
                'online':   False,
                'error':    str(e)
            })
    return jsonify(result)

@app.route('/api/device/<device_id>')
def get_device(device_id):
    if device_id not in DEVICES:
        return jsonify({'error': 'Cihaz bulunamadı'}), 404

    try:
        summary = build_device_summary(device_id)
        cfg     = DEVICES[device_id]
        fmt     = cfg.get('format', 'aq2')

        aqi_hist, dust_hist = read_history(
            cfg.get('data_dir'), hours=12,
            fmt=fmt, csv_file=cfg.get('csv_file')
        )

        if fmt == 'aq0':
            df_recent = load_recent_aq0(cfg['csv_file'], hours=13)
        else:
            df_recent = load_recent_aq2(cfg['data_dir'], hours=13)

        def avg_aqi(hours):
            if df_recent is None or len(df_recent) == 0:
                return summary.get('aqi')
            return round(df_recent['aqi'].tail(hours).mean(), 1)

        corr = read_correlation(
            data_dir=cfg.get('data_dir'),
            fmt=fmt,
            csv_file=cfg.get('csv_file')
        )

        summary.update({
            'history':     {'aqi': aqi_hist, 'dust': dust_hist},
            'correlation': corr,
            'aqi_1h':      avg_aqi(1),
            'aqi_6h':      avg_aqi(6),
            'aqi_12h':     avg_aqi(12),
        })
        return jsonify(summary)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/<device_id>')
def get_prediction(device_id):
    if device_id not in DEVICES:
        return jsonify({'error': 'Cihaz bulunamadı'}), 404

    if device_id not in predictors:
        return jsonify({
            'error': f'Model yüklü değil ({device_id}). augment_and_train_v2.py çalıştırın.',
            'predicted_aqi': None
        }), 503

    try:
        cfg       = DEVICES[device_id]
        predictor = predictors[device_id]
        result    = predictor.predict(
            data_dir=cfg.get('data_dir'),
            csv_file=cfg.get('csv_file'),
            fmt=cfg.get('format', 'aq2')
        )
        result['device_id'] = device_id
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health():
    return jsonify({
        'status':         'ok',
        'models_loaded':  list(predictors.keys()),
        'devices':        list(DEVICES.keys()),
        'timestamp':      datetime.now().isoformat(),
    })

# ══════════════════════════════════════════
# BAŞLAT
# ══════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "="*50)
    print("AQM Network — Flask API v2")
    print("="*50)
    print(f"  Model klasörü  : {MODEL_DIR}")
    print(f"  Cihazlar       : {', '.join(DEVICES.keys())}")
    print(f"  Yüklü modeller : {', '.join(predictors.keys())}")
    print(f"\n  API adresi     : http://localhost:5001")
    print(f"  Sağlık testi   : http://localhost:5001/api/health")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5001, debug=False)
