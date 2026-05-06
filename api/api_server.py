"""
api_server.py — Sadece ANA KUTUDA çalışır (Port 8080)

AQI hesaplama: air_quality_dashboard.py ile aynı mantık
  - Kayan ortalama tamponları (PM25, CO, NO2, O3)
  - Korelasyon tabanlı kompanzasyon (nem/sıcaklık etkisi)
  - EPA kırılım noktaları
"""

import csv
import math
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
import requests
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ─── AYARLAR ────────────────────────────────────────────────────────────────

KUTU_LISTESI = [
    {"id": "Air_Quality_0", "name": "AQM-00 — Ana Kutu",      "location": "Kütüphane", "ip": "10.1.36.144", "port": 5000},
    {"id": "Air_Quality_1", "name": "AQM-01 — Sensör Kutusu", "location": "Sensör Kutusu 1", "ip": "10.1.36.73",  "port": 5000},
    {"id": "Air_Quality_2", "name": "AQM-02 — Air Quality 2", "location": "Starbucks", "ip": "10.1.36.221",  "port": 5000},
    {"id": "Air_Quality_3", "name": "AQM-03 — Air Quality 3", "location": "Simitçi Dünyası", "ip": "10.1.36.72", "port": 5000},
]

POLL_INTERVAL   = 10
REQUEST_TIMEOUT = 5
API_PORT        = 8080

# ─── DASHBOARD İLE AYNI PENCERE BOYUTLARI ────────────────────────────────────
# @ 10s polling aralığı
AVG_WIN_PM25    = min(43200, 3600)   # ~10 saat
AVG_WIN_CO      = min(14400, 3600)
AVG_WIN_NO2     = 1800               # ~5 saat
AVG_WIN_O3      = min(14400, 3600)
FAST_FILTER_WIN = 5                  # hızlı noise bastırma
CORR_WIN        = 120                # korelasyon penceresi (~20 dakika)
CORR_MIN_STD    = 1e-6
MAX_COMP_FRAC   = 0.35
PM_HUM_GAIN     = 0.18
GAS_TEMP_GAIN   = 0.10
GAS_HUM_GAIN    = 0.08
HISTORY_LEN     = 1440                # son 1 saat grafikler için

# ─── EPA KIRILIM NOKTALARI (dashboard ile birebir) ────────────────────────────

PM25_BP = [
    (0.0,   12.0,   0,  50), (12.1,  35.4,  51, 100),
    (35.5,  55.4, 101, 150), (55.5, 150.4, 151, 200),
    (150.5,250.4, 201, 300), (250.5,350.4, 301, 400),
    (350.5,500.4, 401, 500),
]
CO_BP = [
    (0.0,  4.4,   0,  50), (4.5,  9.4,  51, 100),
    (9.5,  12.4, 101, 150), (12.5, 15.4, 151, 200),
    (15.5, 30.4, 201, 300), (30.5, 40.4, 301, 400),
    (40.5, 50.4, 401, 500),
]
NO2_BP = [
    (0,    53,    0,  50), (54,   100,  51, 100),
    (101,  360, 101, 150), (361,  649, 151, 200),
    (650, 1249, 201, 300), (1250,1649, 301, 400),
    (1650,2049, 401, 500),
]
O3_BP = [
    (0,   54,   0,  50), (55,  70,  51, 100),
    (71,  85, 101, 150), (86, 105, 151, 200),
    (106,200, 201, 300),
]
AQI_CATS = [
    (  0,  50, "İyi"),
    ( 51, 100, "Orta"),
    (101, 150, "Hassas Gruplar"),
    (151, 200, "Sağlıksız"),
    (201, 300, "Çok Sağlıksız"),
    (301, 500, "Tehlikeli"),
]

def _linear_aqi(c, bp):
    if c is None or c < 0: return None
    for lo, hi, ilo, ihi in bp:
        if lo <= c <= hi:
            return round((ihi - ilo) / (hi - lo) * (c - lo) + ilo)
    return 500

def calc_aqi_all(pm25, co, no2, ozone_ppb):
    """Dashboard ile aynı: tüm parametrelerden en yüksek AQI + baskın parametre."""
    scores = {}
    v = _linear_aqi(pm25, PM25_BP)
    if v is not None: scores["PM2.5"] = v
    v = _linear_aqi(co, CO_BP)
    if v is not None: scores["CO"] = v
    v = _linear_aqi(no2, NO2_BP)
    if v is not None: scores["NO₂"] = v
    if ozone_ppb is not None and ozone_ppb > 0:
        v = _linear_aqi(ozone_ppb, O3_BP)
        if v is not None: scores["O₃"] = v
    if not scores: return None, None
    dominant = max(scores, key=scores.get)
    return scores[dominant], dominant

def aqi_label(aqi):
    if aqi is None: return "—"
    for lo, hi, label in AQI_CATS:
        if lo <= aqi <= hi: return label
    return "Tehlikeli"

# ─── DASHBOARD İLE AYNI YARDIMCI FONKSİYONLAR ───────────────────────────────

def safe_mean(values):
    return float(sum(values) / len(values)) if values else None

def moving_average(values, win=FAST_FILTER_WIN):
    if not values: return None
    n = min(len(values), max(1, win))
    tail = list(values)[-n:]
    return float(sum(tail) / n)

def pearson_corr(x_vals, y_vals, win=CORR_WIN):
    n = min(len(x_vals), len(y_vals), max(2, win))
    if n < 2: return 0.0
    x = np.asarray(list(x_vals)[-n:], dtype=float)
    y = np.asarray(list(y_vals)[-n:], dtype=float)
    if np.std(x) < CORR_MIN_STD or np.std(y) < CORR_MIN_STD: return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    return float(np.clip(corr if not np.isnan(corr) else 0.0, -1.0, 1.0))

def compensated_value(raw_val, env_val, env_hist, corr, base_gain, hard_floor=0.0):
    if raw_val is None: return None
    if not env_hist: return max(hard_floor, raw_val)
    env_mean = safe_mean(env_hist)
    if env_mean is None: return max(hard_floor, raw_val)
    delta_env  = env_val - env_mean
    correction = base_gain * corr * delta_env
    max_delta  = abs(raw_val) * MAX_COMP_FRAC + 1e-9
    correction = float(np.clip(correction, -max_delta, max_delta))
    return max(hard_floor, raw_val - correction)

def compute_realtime_features(buf):
    """Dashboard'daki compute_realtime_features ile aynı mantık."""
    feats = {}

    # 1) Hızlı moving average
    feats["temperature_f"] = moving_average(buf["temperature"])
    feats["humidity_f"]    = moving_average(buf["humidity"])
    feats["dust_f"]        = moving_average(buf["dust_avg"])
    feats["co_f"]          = moving_average(buf["co"])
    feats["no2_f"]         = moving_average(buf["no2"])
    feats["ozone_f"]       = moving_average(buf["ozone_ppb"])

    # 2) Korelasyon
    corr_pm_h  = pearson_corr(buf["dust_avg"],  buf["humidity"])
    corr_co_t  = pearson_corr(buf["co"],        buf["temperature"])
    corr_co_h  = pearson_corr(buf["co"],        buf["humidity"])
    corr_no2_t = pearson_corr(buf["no2"],       buf["temperature"])
    corr_no2_h = pearson_corr(buf["no2"],       buf["humidity"])
    corr_o3_t  = pearson_corr(buf["ozone_ppb"], buf["temperature"])
    corr_o3_h  = pearson_corr(buf["ozone_ppb"], buf["humidity"])

    feats["corr"] = {
        "pm_humidity": corr_pm_h,
        "co_temp": corr_co_t, "no2_temp": corr_no2_t, "ozone_temp": corr_o3_t,
    }

    # 3) Kompanzasyon
    hum_hist = buf["humidity"]
    tmp_hist = buf["temperature"]

    dust_corr = compensated_value(feats["dust_f"], feats["humidity_f"],    hum_hist, corr_pm_h,  PM_HUM_GAIN)
    co_corr   = compensated_value(feats["co_f"],   feats["temperature_f"], tmp_hist, corr_co_t,  GAS_TEMP_GAIN)
    co_corr   = compensated_value(co_corr,         feats["humidity_f"],    hum_hist, corr_co_h,  GAS_HUM_GAIN)
    no2_corr  = compensated_value(feats["no2_f"],  feats["temperature_f"], tmp_hist, corr_no2_t, GAS_TEMP_GAIN)
    no2_corr  = compensated_value(no2_corr,        feats["humidity_f"],    hum_hist, corr_no2_h, GAS_HUM_GAIN)
    o3_corr   = compensated_value(feats["ozone_f"],feats["temperature_f"], tmp_hist, corr_o3_t,  GAS_TEMP_GAIN)
    o3_corr   = compensated_value(o3_corr,         feats["humidity_f"],    hum_hist, corr_o3_h,  GAS_HUM_GAIN)

    feats["dust_corr"]  = dust_corr
    feats["co_corr"]    = co_corr
    feats["no2_corr"]   = no2_corr
    feats["ozone_corr"] = o3_corr
    return feats

def avg_aqi_last_n(aqi_buf, n):
    vals = list(aqi_buf)[-n:]
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals)) if vals else None

# ─── PER-KUTU VERİ YAPISI ────────────────────────────────────────────────────

def make_kutu_state():
    return {
        "online":  False,
        "latest":  None,
        # Ham buffer (moving average + korelasyon için)
        "buf": {k: deque(maxlen=3600) for k in
                ["temperature", "humidity", "dust", "dust_avg",
                 "ozone_ppb", "co", "no2"]},
        # EPA kayan ortalama tamponları
        "avg_pm25": deque(maxlen=AVG_WIN_PM25),
        "avg_co":   deque(maxlen=AVG_WIN_CO),
        "avg_no2":  deque(maxlen=AVG_WIN_NO2),
        "avg_o3":   deque(maxlen=AVG_WIN_O3),
        # AQI geçmiş tamponları
        "aqi_buf":  deque(maxlen=4320),   # 12 saat @ 10s
        # Grafik geçmişi
        "history":  deque(maxlen=HISTORY_LEN),
        # Son hesaplanan özellikler
        "proc":     {},
    }

_lock  = threading.Lock()
_store = {k["id"]: make_kutu_state() for k in KUTU_LISTESI}

# ─── POLLING ─────────────────────────────────────────────────────────────────

def poll_kutu(kutu: dict):
    kutu_id = kutu["id"]
    url     = f"http://{kutu['ip']}:{kutu['port']}/data"
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        body = resp.json()
        if not body.get("success"): return

        raw    = body["data"]
        dust   = raw.get("dustDensity_UgM3")
        co     = raw.get("co_Ppm")
        no2    = raw.get("no2_Ppb")
        temp   = raw.get("temperature_C")
        hum    = raw.get("humidity_Pct")
        o3_ppb = raw.get("ozone_ppb")
        o3_raw = raw.get("ozoneRaw_ADC")

        with _lock:
            st  = _store[kutu_id]
            buf = st["buf"]

            # Buffer'a ekle
            for key, val in [("temperature", temp), ("humidity", hum),
                              ("dust", dust), ("ozone_ppb", o3_ppb),
                              ("co", co), ("no2", no2)]:
                if val is not None:
                    buf[key].append(float(val))

            # dust_avg
            dust_avg = moving_average(buf["dust"])
            if dust_avg is not None:
                buf["dust_avg"].append(dust_avg)

            # Realtime features (kompanzasyon + korelasyon)
            proc = compute_realtime_features(buf)
            st["proc"] = proc

            # EPA kayan ortalama tamponlarına ekle
            if proc.get("dust_corr")  is not None: st["avg_pm25"].append(proc["dust_corr"])
            if proc.get("co_corr")    is not None: st["avg_co"].append(proc["co_corr"])
            if proc.get("no2_corr")   is not None: st["avg_no2"].append(proc["no2_corr"])
            if proc.get("ozone_corr") is not None: st["avg_o3"].append(proc["ozone_corr"])

            # AQI hesapla (kayan ortalamalar üzerinden)
            aqi, dominant = calc_aqi_all(
                safe_mean(st["avg_pm25"]),
                safe_mean(st["avg_co"]),
                safe_mean(st["avg_no2"]),
                safe_mean(st["avg_o3"]),
            )
            st["aqi_buf"].append(aqi)

            # Grafik geçmişi
            st["history"].append({
                "t":    time.time(),
                "dust": proc.get("dust_corr") or dust,
                "aqi":  aqi,
            })

            # Latest snapshot
            st["latest"] = {
                "temperature": proc.get("temperature_f") or temp,
                "humidity":    proc.get("humidity_f")    or hum,
                "dust":        proc.get("dust_corr")     or dust,
                "co":          proc.get("co_corr")       or co,
                "no2":         proc.get("no2_corr")      or no2,
                "ozone_ppb":   proc.get("ozone_corr")    or o3_ppb,
                "ozone_raw":   o3_raw,
                "aqi":         aqi,
                "aqi_dominant": dominant,
            }
            st["online"] = True

    except Exception:
        with _lock:
            _store[kutu_id]["online"] = False

def load_history_from_csv(kutu_id, csv_path, hours=24):
    """Başlangıçta CSV'den geçmiş veriyi yükle."""
    try:
        import os
        if not os.path.exists(csv_path):
            return
        cutoff = time.time() - hours * 3600
        rows_loaded = 0
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # AQ-0 formatı: timestamp sütunu
                    # AQ-2/3 formatı: tarih + saat ayrı sütunlar
                    ts_str = row.get('timestamp', '')
                    if not ts_str:
                        tarih = row.get('tarih', '')
                        saat  = row.get('saat', '')
                        if tarih and saat:
                            ts_str = f"{tarih}T{saat}"
                        else:
                            continue
                    from datetime import datetime
                    dt = datetime.fromisoformat(ts_str)
                    t = dt.timestamp()
                    if t < cutoff:
                        continue
                    dust   = float(row.get('dustDensity_UgM3', 0) or 0)
                    co     = float(row.get('co_Ppm', 0) or 0)
                    no2    = float(row.get('no2_Ppb', 0) or 0)
                    temp   = float(row.get('temperature_C', 0) or 0)
                    hum    = float(row.get('humidity_Pct', 0) or 0)
                    o3_ppb = float(row.get('ozone_ppb', 0) or 0)
                    aqi, dominant = calc_aqi_all(dust, co, no2, o3_ppb)
                    point = {
                        "t": t, "temperature": temp, "humidity": hum,
                        "dust": dust, "co": co, "no2": no2,
                        "ozone_ppb": o3_ppb, "aqi": aqi,
                    }
                    with _lock:
                        _store[kutu_id]["history"].append(point)
                        if not _store[kutu_id]["latest"]:
                            _store[kutu_id]["latest"] = point
                    rows_loaded += 1
                except Exception:
                    continue
        print(f"[CSV] {kutu_id}: {rows_loaded} geçmiş nokta yüklendi")
    except Exception as e:
        print(f"[CSV] {kutu_id} yükleme hatası: {e}")

def poll_loop():
    while True:
        threads = [threading.Thread(target=poll_kutu, args=(k,), daemon=True) for k in KUTU_LISTESI]
        for t in threads: t.start()
        for t in threads: t.join(timeout=REQUEST_TIMEOUT + 1)
        time.sleep(POLL_INTERVAL)

# ─── KORELASYON (web için) ───────────────────────────────────────────────────

def get_correlation(kutu_id):
    proc = _store[kutu_id].get("proc", {})
    corr = proc.get("corr", {})
    return {
        "pm_humidity": corr.get("pm_humidity"),
        "co_temp":     corr.get("co_temp"),
        "no2_temp":    corr.get("no2_temp"),
        "ozone_temp":  corr.get("ozone_temp"),
    }

# ─── API ENDPOINT'LERİ ────────────────────────────────────────────────────────

@app.route("/api/devices")
def get_devices():
    result = []
    with _lock:
        for kutu in KUTU_LISTESI:
            kid    = kutu["id"]
            st     = _store[kid]
            p      = st["latest"]
            online = st["online"]
            if p:
                aqi       = p.get("aqi")
                dominant  = p.get("aqi_dominant")
                result.append({
                    "id": kid, "name": kutu["name"], "location": kutu["location"],
                    "online": online,
                    "aqi": aqi, "aqi_label": aqi_label(aqi), "aqi_dominant": dominant,
                    "temperature": p.get("temperature"), "humidity": p.get("humidity"),
                    "dust": p.get("dust"), "co": p.get("co"), "no2": p.get("no2"),
                    "ozone_raw": p.get("ozone_ppb"),
                })
            else:
                result.append({"id": kid, "name": kutu["name"],
                                "location": kutu["location"], "online": False})
    return jsonify(result)


@app.route("/api/device/<kutu_id>")
def get_device(kutu_id):
    kutu = next((k for k in KUTU_LISTESI if k["id"] == kutu_id), None)
    if not kutu:
        return jsonify({"error": "Cihaz bulunamadı"}), 404

    with _lock:
        st     = _store[kutu_id]
        p      = st["latest"]
        online = st["online"]
        hist   = list(st["history"])[-1440:]
        corr   = get_correlation(kutu_id)
        aqi_1h  = avg_aqi_last_n(st["aqi_buf"], 360)
        aqi_6h  = avg_aqi_last_n(st["aqi_buf"], 2160)
        aqi_12h = avg_aqi_last_n(st["aqi_buf"], 4320)

    if not p:
        return jsonify({"id": kutu_id, "name": kutu["name"],
                        "location": kutu["location"], "online": False})

    aqi      = p.get("aqi")
    dominant = p.get("aqi_dominant")

    return jsonify({
        "id": kutu_id, "name": kutu["name"], "location": kutu["location"],
        "online": online,
        "aqi": aqi, "aqi_label": aqi_label(aqi), "aqi_dominant": dominant,
        "temperature": p.get("temperature"), "humidity": p.get("humidity"),
        "dust": p.get("dust"), "co": p.get("co"), "no2": p.get("no2"),
        "ozone_raw": p.get("ozone_ppb"),
        "history": {
            "dust":        [h.get("dust")        for h in hist],
            "aqi":         [h.get("aqi")         for h in hist],
            "temperature": [h.get("temperature") for h in hist],
            "humidity":    [h.get("humidity")    for h in hist],
            "co":          [h.get("co")          for h in hist],
            "no2":         [h.get("no2")         for h in hist],
            "ozone_raw":   [h.get("ozone_ppb")   for h in hist],
            "timestamps":  [h.get("t")           for h in hist],
        },
        "correlation": corr,
        "aqi_1h": aqi_1h, "aqi_6h": aqi_6h, "aqi_12h": aqi_12h,
    })
@app.route("/api/predict/<kutu_id>")
def get_prediction(kutu_id):
    # Eski ID'yi yeni API ID'sine çevir
    id_map = {
        'Air_Quality_0': 'aqm00',
        'Air_Quality_2': 'aqm02',
        'Air_Quality_3': 'aqm03',
    }
    new_id = id_map.get(kutu_id)
    if not new_id:
        return jsonify({
            'error': 'Bu cihaz için tahmin mevcut değil',
            'predicted_aqi': None
        }), 404
    try:
        resp = requests.get(f'http://localhost:5001/api/predict/{new_id}', timeout=10)
        data = resp.json()
        data['device_id'] = kutu_id
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e), 'predicted_aqi': None}), 500
# ─── BAŞLATMA ─────────────────────────────────────────────────────────────────

CSV_PATHS = {
    "Air_Quality_0": "/home/ibrahim/data/Air_Quality_0_log.csv",
    "Air_Quality_1": "/home/ibrahim/data/Air_Quality_1_log.csv",
    "Air_Quality_2": "/home/ibrahim/data/Air_Quality_2_log.csv",
    "Air_Quality_3": "/home/ibrahim/data/Air_Quality_3_log.csv",
}

if __name__ == "__main__":
    print(f"\n[API SERVER] Port {API_PORT} — Kayan ortalama + kompanzasyon aktif")
    # CSV'den geçmiş yükle
    from datetime import datetime as _dt
    today = _dt.now().strftime("%Y-%m-%d")
    for kid, path in CSV_PATHS.items():
        if '{}' in path:
            path = path.format(today)
        load_history_from_csv(kid, path, hours=24)
    threading.Thread(target=poll_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=API_PORT, threaded=True)
