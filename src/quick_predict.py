"""
Type the current readings directly and get the forecast (no Excel needed).
Interactive:   python quick_predict.py
One line:      python quick_predict.py --time "2025-11-11 08:00" --pm25 519 --pm10 774 --no 164 --no2 175 --nox 227 \
                                       --nh3 67 --so2 53 --co 4.1 --ozone 15 --at 17 --rh 64 --ws 1.3 --wd 158 --sr 293
Required: time, PM2.5, PM10. Anything else can be skipped (press Enter / leave the flag out).
Needs quick_models.joblib in the folder you run it from (created by quick_train.py).
"""
import argparse, joblib, numpy as np, pandas as pd
from common import TARGETS, HORIZONS, category
from quick_train import quick_features, INPUT_COLS

FIELDS = [("PM2.5", "PM2.5 (ug/m3)", True), ("PM10", "PM10 (ug/m3)", True), ("NO", "NO (ug/m3)", False),
          ("NO2", "NO2 (ug/m3)", False), ("NOx", "NOx (ppb)", False), ("NH3", "NH3 (ug/m3)", False),
          ("SO2", "SO2 (ug/m3)", False), ("CO", "CO (mg/m3)", False), ("Ozone", "Ozone (ug/m3)", False),
          ("AT", "Temperature (deg C)", False), ("RH", "Humidity (%)", False), ("WS", "Wind speed (m/s)", False),
          ("WD", "Wind direction (deg)", False), ("SR", "Solar radiation (W/m2)", False)]
LIMITS = {"PM2.5": (1, 990), "PM10": (5, 990), "NO": (0, 495), "NO2": (0, 400), "NOx": (0, 495), "NH3": (0, 300),
          "SO2": (0, 300), "CO": (0, 20), "Ozone": (0, 300), "AT": (0, 50), "RH": (5, 100), "WS": (0, 15),
          "WD": (0, 360), "SR": (0, 1200)}

ap = argparse.ArgumentParser()
ap.add_argument("--time")
for k, _, _ in FIELDS: ap.add_argument("--" + k.lower().replace(".", ""))
a = vars(ap.parse_args())

def ask(label, required):
    while True:
        s = input(f"{label}{'' if required else ' [Enter to skip]'}: ").strip()
        if s == "" and not required: return np.nan
        try: return float(s)
        except ValueError: print("  please type a number")

t = a["time"] or input("Date and time, e.g. 2025-11-11 08:00 (24-hour clock): ").strip()
ts = pd.DatetimeIndex([pd.to_datetime(t, dayfirst=True)])
row = {}
for k, label, req in FIELDS:
    v = a[k.lower().replace(".", "")]
    if v is None and not any(a[x.lower().replace(".", "")] is not None for x, _, _ in FIELDS) and a["time"] is None:
        v = ask(label, req)                       # fully interactive mode
    row[k] = np.nan if v is None else float(v)
for k, (lo, hi) in LIMITS.items():
    if not np.isnan(row[k]) and not (lo <= row[k] <= hi):
        print(f"note: {k}={row[k]} looks implausible, treated as empty"); row[k] = np.nan
for k, _, req in FIELDS:
    if req and np.isnan(row[k]): raise SystemExit(f"{k} is required.")
vals = pd.DataFrame([row])[INPUT_COLS]

bundle = joblib.load("quick_models.joblib"); models, feats = bundle["models"], bundle["feats"]
X = quick_features(vals, ts)[feats]
res = {}
for t_ in TARGETS:
    now = X["AQI_now"].values if t_ == "AQI" else vals[t_].values
    res[t_] = []
    for h in HORIZONS:
        p = float(np.expm1(np.log1p(now) + models[(t_, h)].predict(X))[0])
        res[t_].append(min(max(p, 0), 500) if t_ == "AQI" else max(p, 0))
tab = pd.DataFrame(res, index=[f"+{h}h" for h in HORIZONS]).T
tab.columns = [f"+{h}h ({(ts[0] + pd.Timedelta(hours=h)):%d %b %H:%M})" for h in HORIZONS]
print(f"\nForecast from readings at {ts[0]:%d %b %Y %H:%M}\n")
print(tab.round(1).to_string())
print("\nAQI category:", ", ".join(f"+{h}h {category(pd.Series([v]))[0]}" for h, v in zip(HORIZONS, res['AQI'])))
print("(units: ug/m3, CO in mg/m3; AQI is the least accurate output in this mode)")