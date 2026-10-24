"""
"Excel-input" model: forecast from the CURRENT reading only (no history, no station).
You give pollutants + weather (+ date/time) for one hour; it returns PM2.5, PM10, NO2, CO, Ozone and AQI
for +1, +6, +12 and +24 hours. One model for all locations (the 4 stations are pooled).

Step A (evaluate): trains on 2023-Sep 2024, validates Oct-Dec 2024, tests on 2025 -> quick_metrics.csv
Step B (final)   : retrains on everything -> quick_models.joblib
Usage: python quick_train.py data.csv
"""
import sys, joblib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from common import BP_, SRC, subindex, TARGETS, HORIZONS, category

GASES = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "SO2", "CO", "Ozone"]
WX    = ["AT", "RH", "WS", "SR"]
INPUT_COLS = GASES + WX + ["WD"]

def instant_aqi(df):
    """AQI from the current hour's concentrations only (no 24h averaging possible from one row)."""
    si = pd.DataFrame({k: subindex(df[col].values, BP_[k]) for k, col in SRC.items()}, index=df.index)
    ok = (si.notna().sum(axis=1) >= 3) & si[["PM2.5", "PM10"]].notna().any(axis=1)
    return si.max(axis=1).where(ok)

def quick_features(df, ts):
    """df: one row per reading with INPUT_COLS; ts: matching DatetimeIndex."""
    X = df[GASES + WX].copy()
    wd = np.deg2rad(df["WD"]); X["WD_sin"], X["WD_cos"] = np.sin(wd), np.cos(wd)
    X["PM_ratio"] = df["PM2.5"] / df["PM10"]
    X["NO2_NO"] = df["NO2"] / (df["NO"] + 1)
    X["AQI_now"] = instant_aqi(df)
    X["hr_sin"], X["hr_cos"] = np.sin(2*np.pi*ts.hour/24), np.cos(2*np.pi*ts.hour/24)
    doy = ts.dayofyear
    X["doy_sin"], X["doy_cos"] = np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25)
    X["dow"] = ts.dayofweek
    return X

def mk(n): return HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.06, max_iter=n, max_leaf_nodes=12,
        min_samples_leaf=60, l2_regularization=5.0, max_bins=63, early_stopping=False, random_state=0)

if __name__ == "__main__":
    from common import prepare
    d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "data.csv", parse_dates=["Timestamp"])
    D, _, _ = prepare(d)                       # D has the true (24h-average) AQI per station-hour
    ts = pd.DatetimeIndex(D["Timestamp"])
    X = quick_features(D, ts); feats = list(X.columns)
    now = {t: (X["AQI_now"] if t == "AQI" else D[t]) for t in TARGETS}   # "current value" of each target
    TRAIN_END, VAL_END = "2024-09-30 23:00", "2024-12-31 23:00"
    rows, models = [], {}
    for t in TARGETS:
        for h in HORIZONS:
            y = D.groupby("station")[t].shift(-h)
            meta = pd.DataFrame({"y": y, "now": now[t], "t": ts}).dropna()
            meta["z"] = np.log1p(meta["y"]) - np.log1p(meta["now"])
            data = X.loc[meta.index].join(meta)
            tr = data[data.t <= TRAIN_END]; va = data[(data.t > TRAIN_END) & (data.t <= VAL_END)]; te = data[data.t > VAL_END]
            m = mk(250).fit(tr[feats], tr["z"])
            errs = [mean_absolute_error(va["z"], p) for p in m.staged_predict(va[feats])]
            best = int(np.argmin(errs)) + 1
            for i, zp in enumerate(m.staged_predict(te[feats])):
                if i + 1 == best: break
            p = np.expm1(np.log1p(te["now"].values) + zp).clip(min=0)
            if t == "AQI": p = p.clip(max=500)
            r = dict(target=t, horizon_h=h, n_test=len(te), best_iter=best, MAE=mean_absolute_error(te["y"], p),
                     R2=r2_score(te["y"], p), MAE_persistence=mean_absolute_error(te["y"], te["now"]))
            if t == "AQI":
                r["cat_acc"] = (category(te["y"]) == category(pd.Series(p, index=te.index))).mean()
                r["cat_acc_persistence"] = (category(te["y"]) == category(te["now"])).mean()
            rows.append(r); print(f"eval {t} +{h}h  MAE {r['MAE']:.2f} vs persistence {r['MAE_persistence']:.2f}", flush=True)
            # final model on ALL rows, same number of rounds
            models[(t, h)] = mk(best).fit(data[feats], data["z"])
    res = pd.DataFrame(rows); res["MAE_gain_%"] = (1 - res.MAE / res.MAE_persistence) * 100
    res.round(3).to_csv("quick_metrics.csv", index=False)
    joblib.dump({"models": models, "feats": feats}, "quick_models.joblib", compress=3)
    print("saved quick_metrics.csv and quick_models.joblib")