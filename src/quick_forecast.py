"""
Forecast from an Excel/CSV sheet of CURRENT readings (one row = one hour of readings).
Usage:  python quick_forecast.py input.xlsx [output.xlsx]
Input columns (names are matched loosely, units in brackets are ignored):
  Timestamp (date + hour), PM2.5, PM10, NO, NO2, NOx, NH3, SO2, CO, Ozone, AT, RH, WS, WD, SR
  PM2.5 and PM10 and Timestamp are required; other columns may be left empty (accuracy drops a little).
Output: the same rows plus forecast columns for +1h, +6h, +12h, +24h of AQI, PM2.5, PM10, NO2, CO, Ozone.
"""
import sys, os, joblib, numpy as np, pandas as pd
from common import TARGETS, HORIZONS, category
from quick_train import quick_features, INPUT_COLS

ALIASES = {"TIMESTAMP": "Timestamp", "DATE": "Timestamp", "DATETIME": "Timestamp", "TIME": "Timestamp",
    "PM2.5": "PM2.5", "PM25": "PM2.5", "PM10": "PM10", "NO": "NO", "NO2": "NO2", "NOX": "NOx", "NH3": "NH3", "SO2": "SO2",
    "CO": "CO", "OZONE": "Ozone", "O3": "Ozone", "AT": "AT", "TEMP": "AT", "TEMPERATURE": "AT",
    "RH": "RH", "HUMIDITY": "RH", "WS": "WS", "WIND SPEED": "WS", "WD": "WD", "WIND DIRECTION": "WD",
    "SR": "SR", "SOLAR RADIATION": "SR"}
LIMITS = {"PM2.5": (1, 990), "PM10": (5, 990), "NO": (0, 495), "NO2": (0, 400), "NOx": (0, 495), "NH3": (0, 300),
          "SO2": (0, 300), "CO": (0, 20), "Ozone": (0, 300), "AT": (0, 50), "RH": (5, 100), "WS": (0, 15),
          "WD": (0, 360), "SR": (0, 1200)}

src = sys.argv[1]
dst = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(src)[0] + "_forecast" + os.path.splitext(src)[1]
raw = pd.read_excel(src) if src.lower().endswith((".xlsx", ".xls")) else pd.read_csv(src)

# --- map column names ---
rename = {}
for c in raw.columns:
    key = str(c).split(" (")[0].strip().upper().replace("_", " ").replace("-", " ")
    if key in ALIASES and ALIASES[key] not in rename.values(): rename[c] = ALIASES[key]
df = raw.rename(columns=rename)
for need in ("Timestamp", "PM2.5", "PM10"):
    if need not in df.columns: raise SystemExit(f"Missing required column: {need}")
for c in INPUT_COLS:
    if c not in df.columns:
        df[c] = np.nan; print(f"note: column {c} not found, treated as empty")
ts = pd.DatetimeIndex(pd.to_datetime(df["Timestamp"], dayfirst=True))
vals = df[INPUT_COLS].apply(pd.to_numeric, errors="coerce")

# --- sanity limits (same idea as clean_data.py) ---
for c, (lo, hi) in LIMITS.items():
    bad = (vals[c] < lo) | (vals[c] > hi)
    if bad.any(): print(f"note: {int(bad.sum())} implausible {c} value(s) treated as empty")
    vals.loc[bad, c] = np.nan
if (vals["PM2.5"] > vals["PM10"] * 1.1).any(): print("note: some rows have PM2.5 > PM10; check your input")

# --- predict ---
bundle = joblib.load("quick_models.joblib"); models, feats = bundle["models"], bundle["feats"]
X = quick_features(vals.reset_index(drop=True), ts)[feats]
out = pd.DataFrame(index=range(len(df)))
for t in TARGETS:
    now = X["AQI_now"] if t == "AQI" else vals[t].reset_index(drop=True)
    for h in HORIZONS:
        p = np.expm1(np.log1p(now.values) + models[(t, h)].predict(X)).clip(min=0)
        if t == "AQI": p = p.clip(max=500)
        out[f"{t}_+{h}h"] = np.round(p, 2)
        if t == "AQI": out[f"AQI_category_+{h}h"] = category(pd.Series(p)).values
res = pd.concat([raw.reset_index(drop=True), out], axis=1)
if dst.lower().endswith(".xlsx"): res.to_excel(dst, index=False)
else: res.to_csv(dst, index=False)
print(f"saved {dst}  ({len(res)} rows)")
print(out[[c for c in out.columns if c.startswith("AQI_") and "category" not in c]].round(0).head(10).to_string())