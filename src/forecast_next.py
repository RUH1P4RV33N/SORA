"""
Forecast the next 1/6/12/24 hours for every station in a CLEANED recent-data file.
Usage: python forecast_next.py recent.csv [--asof "2025-12-31 23:00"] [--models models.joblib] [--out forecast_latest.csv]
recent.csv: same columns as data.csv (station, Timestamp, PM2.5 ...). Needs >= ~120 consecutive hours per station
(more is better; features look back up to 72h). Run clean_data.py on raw exports first.
"""
import argparse, joblib, numpy as np, pandas as pd
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("recent"); ap.add_argument("--asof"); ap.add_argument("--models", default="models.joblib")
ap.add_argument("--out", default="forecast_latest.csv")
a = ap.parse_args()

bundle = joblib.load(a.models); models, feats = bundle["models"], bundle["feats"]
d = pd.read_csv(a.recent, parse_dates=["Timestamp"])
if a.asof: d = d[d["Timestamp"] <= pd.Timestamp(a.asof)]
unknown = set(d["station"].unique()) - set(range(1, 5))
if unknown: raise SystemExit(f"Unknown station id(s) {unknown}; models know stations 1-4.")
short = d.groupby("station").size(); short = short[short < 120]
if len(short): print(f"WARNING: stations {list(short.index)} have <120 hours of history; forecasts may be unreliable.")

D, XS, f2 = prepare(d)
assert f2 == feats, "feature mismatch between recent data and trained models"
last = D.groupby("station")["Timestamp"].idxmax()            # newest hour per station
rows = []
for (t, h), m in models.items():
    lvl = D.groupby("station")[t].transform(lambda s: s.rolling(3, min_periods=1).mean())
    z = m.predict(XS.loc[last.values, feats])
    p = np.expm1(np.log1p(lvl.loc[last.values].values) + z).clip(min=0)
    if t == "AQI": p = np.clip(p, 0, 500)
    for st, idx, val in zip(last.index, last.values, p):
        asof = D.loc[idx, "Timestamp"]
        rows.append(dict(station=st, as_of=asof, target=t, horizon_h=h, forecast_for=asof + pd.Timedelta(hours=h), forecast=round(float(val), 2)))
out = pd.DataFrame(rows).sort_values(["station", "target", "horizon_h"])
out["category"] = np.where(out.target == "AQI", category(out.forecast.where(out.target == "AQI")), "")
out.to_csv(a.out, index=False)
print(out[out.target == "AQI"].pivot(index="station", columns="horizon_h", values="forecast").round(0).to_string())
print(f"\nfull forecast (all pollutants) saved to {a.out}")