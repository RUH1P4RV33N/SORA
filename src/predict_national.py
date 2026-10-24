"""
predict_national.py - give the CURRENT readings, get the forecast for +1, +6, +12 and +24 hours.

Run from the project root (needs models/national_models.joblib from train_national.py --mode final).

1) Type the values when asked:
       python src/predict_national.py
2) Everything on one line (leave out any value you do not have; time, PM2.5 and PM10 are required):
       python src/predict_national.py --time "2025-11-11 08:00" --pm25 519 --pm10 774 --no 164 --no2 175 \
              --nox 227 --nh3 67 --so2 53 --co 4.1 --ozone 15 --at 17 --rh 64 --ws 1.3 --wd 158 --sr 293
3) A whole sheet (one row per hour; Excel .xlsx or .csv):
       python src/predict_national.py --sheet data/my_readings.xlsx [--out results/my_forecast.xlsx]
   Sheet columns (names matched loosely, units in brackets ignored):
       Timestamp, PM2.5, PM10, NO, NO2, NOx, NH3, SO2, CO, Ozone, AT, RH, WS, WD, SR

Units: PM, NO, NO2, NH3, SO2, Ozone in ug/m3; CO in mg/m3; NOx in ppb; AT in deg C; RH in %;
       WS in m/s; WD in degrees; SR in W/m2. Use the average of the last hour.
"""
import argparse, os
import joblib, numpy as np, pandas as pd
from common import category
from national_features import build_features, GASES, WX, INPUT_COLS

ALIASES = {"TIMESTAMP": "Timestamp", "DATE": "Timestamp", "DATETIME": "Timestamp", "TIME": "Timestamp",
           "PM2.5": "PM2.5", "PM25": "PM2.5", "PM10": "PM10", "NO": "NO", "NO2": "NO2", "NOX": "NOx", "NH3": "NH3",
           "SO2": "SO2", "CO": "CO", "OZONE": "Ozone", "O3": "Ozone", "AT": "AT", "TEMP": "AT", "TEMPERATURE": "AT",
           "RH": "RH", "HUMIDITY": "RH", "WS": "WS", "WIND SPEED": "WS", "WD": "WD", "WIND DIRECTION": "WD",
           "SR": "SR", "SOLAR RADIATION": "SR", "REGION": "region", "LATITUDE": "latitude", "LONGITUDE": "longitude"}
LIMITS = {"PM2.5": (1, 990), "PM10": (5, 990), "NO": (0, 495), "NO2": (0, 400), "NOx": (0, 495), "NH3": (0, 300),
          "SO2": (0, 300), "CO": (0, 20), "Ozone": (0, 300), "AT": (0, 50), "RH": (5, 100), "WS": (0, 15),
          "WD": (0, 360), "SR": (0, 1200)}
FIELDS = [("PM2.5", "PM2.5 (ug/m3)", True), ("PM10", "PM10 (ug/m3)", True), ("NO", "NO (ug/m3)", False),
          ("NO2", "NO2 (ug/m3)", False), ("NOx", "NOx (ppb)", False), ("NH3", "NH3 (ug/m3)", False),
          ("SO2", "SO2 (ug/m3)", False), ("CO", "CO (mg/m3)", False), ("Ozone", "Ozone (ug/m3)", False),
          ("AT", "Temperature (deg C)", False), ("RH", "Humidity (%)", False), ("WS", "Wind speed (m/s)", False),
          ("WD", "Wind direction (deg)", False), ("SR", "Solar radiation (W/m2)", False)]
HORIZONS = [1, 6, 12, 24, 36, 48, 60, 72]
TARGETS = ["AQI", "PM2.5", "PM10", "NO2", "CO", "Ozone"]


def forecast(bundle, vals, ts):
    """vals: DataFrame with INPUT_COLS (+ region/latitude/longitude if the model uses them); ts: DatetimeIndex."""
    X = build_features(vals, ts, loc=bundle["loc"], region_levels=bundle["region_levels"])[bundle["feats"]]
    out = {}
    for t in TARGETS:
        now = (X["AQI_now"] if t == "AQI" else vals[t]).values.astype(float)
        for h in HORIZONS:
            p = np.expm1(np.log1p(now) + bundle["models"][(t, h)].predict(X)).clip(min=0)
            out[(t, h)] = np.clip(p, 0, 500) if t == "AQI" else p
    return out


def clean_limits(vals):
    for c, (lo, hi) in LIMITS.items():
        bad = (vals[c] < lo) | (vals[c] > hi)
        if bad.any():
            print(f"note: {int(bad.sum())} implausible {c} value(s) treated as empty")
        vals.loc[bad, c] = np.nan
    return vals


def ask(label, required):
    while True:
        s = input(f"{label}{'' if required else ' [Enter to skip]'}: ").strip()
        if s == "" and not required:
            return np.nan
        try:
            return float(s)
        except ValueError:
            print("  please type a number")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="models/national_models.joblib")
    ap.add_argument("--sheet"); ap.add_argument("--out")
    ap.add_argument("--time"); ap.add_argument("--region", help="only needed if the model was trained with --loc region")
    for k, _, _ in FIELDS:
        ap.add_argument("--" + k.lower().replace(".", ""))
    a = vars(ap.parse_args())
    bundle = joblib.load(a["models"])

    if a["sheet"]:                                                      # ---------- sheet mode
        src = a["sheet"]
        dst = a["out"] or os.path.splitext(src)[0] + "_forecast" + os.path.splitext(src)[1]
        raw = pd.read_excel(src) if src.lower().endswith((".xlsx", ".xls")) else pd.read_csv(src)
        rename = {}
        for c in raw.columns:
            key = str(c).split(" (")[0].strip().upper().replace("_", " ").replace("-", " ")
            if key in ALIASES and ALIASES[key] not in rename.values():
                rename[c] = ALIASES[key]
        df = raw.rename(columns=rename)
        for need in ("Timestamp", "PM2.5", "PM10"):
            if need not in df.columns:
                raise SystemExit(f"Missing required column: {need}")
        for c in INPUT_COLS:
            if c not in df.columns:
                df[c] = np.nan; print(f"note: column {c} not found, treated as empty")
        if "region" not in df.columns:
            df["region"] = a["region"]
        ts = pd.DatetimeIndex(pd.to_datetime(df["Timestamp"], dayfirst=True))
        vals = clean_limits(df[INPUT_COLS + ["region"]].copy())
        for c in INPUT_COLS:
            vals[c] = pd.to_numeric(vals[c], errors="coerce")
        vals = vals.reset_index(drop=True)
        res = forecast(bundle, vals, ts)
        out = pd.DataFrame({f"{t}_+{h}h": np.round(v, 2) for (t, h), v in res.items()})
        for h in HORIZONS:
            out[f"AQI_category_+{h}h"] = category(pd.Series(res[("AQI", h)])).values
        full = pd.concat([raw.reset_index(drop=True), out], axis=1)
        full.to_excel(dst, index=False) if dst.lower().endswith(".xlsx") else full.to_csv(dst, index=False)
        print(f"saved {dst} ({len(full)} rows)")
        return

    t = a["time"] or input("Date and time, e.g. 2025-11-11 08:00 (24-hour clock): ").strip()  # ---------- single reading
    ts = pd.DatetimeIndex([pd.to_datetime(t, dayfirst=True)])
    typed = any(a[k.lower().replace(".", "")] is not None for k, _, _ in FIELDS)
    row = {}
    for k, label, req in FIELDS:
        v = a[k.lower().replace(".", "")]
        if v is None and not typed and a["time"] is None:
            v = ask(label, req)
        row[k] = np.nan if v is None else float(v)
    for k, _, req in FIELDS:
        if req and np.isnan(row[k]):
            raise SystemExit(f"{k} is required.")
    vals = clean_limits(pd.DataFrame([row]))
    vals["region"] = a["region"]
    res = forecast(bundle, vals, ts)
    tab = pd.DataFrame({t_: [res[(t_, h)][0] for h in HORIZONS] for t_ in TARGETS},
                       index=[f"+{h}h ({(ts[0] + pd.Timedelta(hours=h)):%d %b %H:%M})" for h in HORIZONS]).T
    print(f"\nForecast from readings at {ts[0]:%d %b %Y %H:%M}\n")
    print(tab.round(1).to_string())
    print("\nAQI category:", ", ".join(f"+{h}h {category(pd.Series([res[('AQI', h)][0]]))[0]}" for h in HORIZONS))
    print("(units: ug/m3, CO in mg/m3; the AQI row is the least accurate output)")


if __name__ == "__main__":
    main()