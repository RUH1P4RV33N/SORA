"""
Multi-station AQI / pollutant forecast (hourly, 4 stations, 2023-2025).
Input : data.csv  (output of clean_data.py)
Train : 2023-01 .. 2024-09 | Validation : 2024-10 .. 2024-12 | Test : all of 2025
Usage : python aqi_forecast.py [data.csv]
"""
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CSV = sys.argv[1] if len(sys.argv) > 1 else "data.csv"
d = pd.read_csv(CSV, parse_dates=["Timestamp"])

# ---------------- AQI (CPCB, India) ----------------
BP_ = {  # (lo, hi, aqi_lo, aqi_hi); last band uses CPCB/approximate upper bound, result clipped at 500
 "PM2.5":[(0,30,0,50),(30,60,51,100),(60,90,101,200),(90,120,201,300),(120,250,301,400),(250,380,401,500)],
 "PM10": [(0,50,0,50),(50,100,51,100),(100,250,101,200),(250,350,201,300),(350,430,301,400),(430,510,401,500)],
 "NO2":  [(0,40,0,50),(40,80,51,100),(80,180,101,200),(180,280,201,300),(280,400,301,400),(400,520,401,500)],
 "SO2":  [(0,40,0,50),(40,80,51,100),(80,380,101,200),(380,800,201,300),(800,1600,301,400),(1600,2000,401,500)],
 "CO":   [(0,1,0,50),(1,2,51,100),(2,10,101,200),(10,17,201,300),(17,34,301,400),(34,50,401,500)],
 "O3":   [(0,50,0,50),(50,100,51,100),(100,168,101,200),(168,208,201,300),(208,748,301,400),(748,1000,401,500)],
 "NH3":  [(0,200,0,50),(200,400,51,100),(400,800,101,200),(800,1200,201,300),(1200,1800,301,400),(1800,2000,401,500)],
}
SRC = {"PM2.5":"PM2.5","PM10":"PM10","NO2":"NO2","SO2":"SO2","CO":"CO","O3":"Ozone","NH3":"NH3"}
def subindex(x, bands):
    x = np.asarray(x, float); out = np.full(len(x), np.nan)
    for lo, hi, a, b in bands:
        m = (x >= lo) & (x < hi) if hi < bands[-1][1] else (x >= lo)
        out[m] = a + (x[m]-lo)*(b-a)/(hi-lo)
    return np.clip(out, 0, 500)
def add_aqi(g):
    si = {}
    for k, col in SRC.items():
        win, mn = (8, 6) if k in ("CO", "O3") else (24, 16)
        si[k] = subindex(g[col].rolling(win, min_periods=mn).mean().values, BP_[k])
    si = pd.DataFrame(si, index=g.index)
    ok = (si.notna().sum(axis=1) >= 3) & si[["PM2.5", "PM10"]].notna().any(axis=1)
    g["AQI"] = si.max(axis=1).where(ok)
    return g
def category(a):
    return pd.cut(a, [-1,50,100,200,300,400,1000], labels=["Good","Satisfactory","Moderate","Poor","Very Poor","Severe"]).astype(str)

# ---------------- prep per station ----------------
frames = []
for s, g in d.groupby("station"):
    g = g.set_index("Timestamp").sort_index().asfreq("h")
    g["station"] = s
    g = g.drop(columns=["Benzene", "Toluene"])               # too sparse
    cols = [c for c in g.columns if c != "station"]
    g[cols] = g[cols].interpolate(limit=6, limit_area="inside")   # fill gaps <= 6h only
    frames.append(add_aqi(g))
D = pd.concat(frames)
# city-wide signal: mean PM2.5 over all stations at the same hour
city = D.reset_index().pivot(index="Timestamp", columns="station", values="PM2.5").mean(axis=1).rename("city_pm25")

TARGETS  = ["AQI", "PM2.5", "PM10", "NO2", "CO", "Ozone"]
HORIZONS  = [1, 6, 12, 24, 36, 48, 60, 72]
POLL = ["PM2.5", "PM10", "NO2", "NO", "NOx", "CO", "Ozone", "SO2", "NH3", "AQI"]
WX   = ["AT", "RH", "WS", "SR"]

def features(g):
    X = {}
    for c in POLL:
        for l in (0, 1, 3, 6, 12, 24, 48): X[f"{c}_l{l}"] = g[c].shift(l)
    for c in WX:
        for l in (0, 3, 12): X[f"{c}_l{l}"] = g[c].shift(l)
    for c in TARGETS:
        for w in (6, 24, 72): X[f"{c}_m{w}"] = g[c].rolling(w, min_periods=3).mean()
        X[f"{c}_s24"] = g[c].rolling(24, min_periods=6).std()
    for c in ("WS", "AT", "RH"): X[f"{c}_m24"] = g[c].rolling(24, min_periods=6).mean()
    X["BP_l0"] = g["BP"]
    wd = np.deg2rad(g["WD"]); X["WD_sin"], X["WD_cos"] = np.sin(wd), np.cos(wd)
    X["PM_ratio"] = g["PM2.5"] / g["PM10"]
    h, doy = g.index.hour, g.index.dayofyear
    X["hr_sin"], X["hr_cos"] = np.sin(2*np.pi*h/24), np.cos(2*np.pi*h/24)
    X["doy_sin"], X["doy_cos"] = np.sin(2*np.pi*doy/365.25), np.cos(2*np.pi*doy/365.25)
    X["dow"] = g.index.dayofweek
    X["station"] = g["station"]
    return pd.DataFrame(X, index=g.index)

XS = pd.concat([features(g) for _, g in D.groupby("station")])
# lagged city-wide mean PM2.5 (computed on the time index, so lags are in real hours)
cl = pd.DataFrame({f"city_pm25_l{l}": city.shift(l) for l in (0, 6, 24)})
XS = XS.join(cl)
XS["_st"] = XS["station"]; XS["_t"] = XS.index
feats = [c for c in XS.columns if c not in ("_st", "_t")]
XS = XS.reset_index(drop=True); D = D.reset_index()   # stations are stacked: use a unique row index
assert (XS["_t"].values == D["Timestamp"].values).all() and (XS["station"].values == D["station"].values).all()
print(f"feature matrix: {XS.shape}, {len(feats)} features", flush=True)

TRAIN_END, VAL_END = "2024-09-30 23:00", "2024-12-31 23:00"
def mk(n): return HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.06, max_iter=n, max_leaf_nodes=12,
        min_samples_leaf=60, l2_regularization=5.0, max_bins=63, early_stopping=False, random_state=0,
        categorical_features=[feats.index("station")])

rows, store = [], {}
for t in TARGETS:
    for h in HORIZONS:
        y   = D.groupby("station")[t].shift(-h)
        lvl = D.groupby("station")[t].transform(lambda s: s.rolling(3, min_periods=1).mean())
        meta = pd.DataFrame({"y": y, "lvl": lvl, "lag0": D[t]}).dropna()
        meta["z"] = np.log1p(meta["y"]) - np.log1p(meta["lvl"])        # predict log-change from current level
        data = XS.loc[meta.index].join(meta)
        tr = data[data._t <= TRAIN_END]; va = data[(data._t > TRAIN_END) & (data._t <= VAL_END)]; te = data[data._t > VAL_END]
        m = mk(250).fit(tr[feats], tr["z"])
        errs = [mean_absolute_error(va["z"], p) for p in m.staged_predict(va[feats])]
        best = int(np.argmin(errs)) + 1
        for i, zp in enumerate(m.staged_predict(te[feats])):
            if i + 1 == best: break
        p = np.expm1(np.log1p(te["lvl"].values) + zp).clip(min=0)
        if t == "AQI": p = p.clip(max=500)
        out = pd.DataFrame({"station": te["_st"].values, "actual": te["y"].values, "pred": p, "persistence": te["lag0"].values}, index=pd.DatetimeIndex(te["_t"]))
        store[(t, h)] = out
        r = dict(target=t, horizon_h=h, n_test=len(te), best_iter=best,
                 MAE=mean_absolute_error(out.actual, out.pred), RMSE=mean_squared_error(out.actual, out.pred)**.5,
                 R2=r2_score(out.actual, out.pred), MAE_persistence=mean_absolute_error(out.actual, out.persistence),
                 R2_persistence=r2_score(out.actual, out.persistence))
        if t == "AQI":
            r["cat_acc"] = (category(out.actual) == category(out.pred)).mean()
            r["cat_acc_persistence"] = (category(out.actual) == category(out.persistence)).mean()
        rows.append(r); print(f"done {t} +{h}h  MAE {r['MAE']:.2f} vs persistence {r['MAE_persistence']:.2f}", flush=True)

res = pd.DataFrame(rows); res["MAE_gain_%"] = (1 - res.MAE / res.MAE_persistence) * 100
res.round(3).to_csv("forecast_metrics.csv", index=False)

# breakdowns for AQI: per station, and winter (Oct-Dec) vs rest
br = []
for h in HORIZONS:
    o = store[("AQI", h)]
    for s, g in o.groupby("station"):
        br.append(dict(horizon_h=h, group=f"station {s}", MAE=mean_absolute_error(g.actual, g.pred), MAE_persistence=mean_absolute_error(g.actual, g.persistence)))
    w = o.index.month >= 10
    for name, g in (("Oct-Dec", o[w]), ("Jan-Sep", o[~w])):
        br.append(dict(horizon_h=h, group=name, MAE=mean_absolute_error(g.actual, g.pred), MAE_persistence=mean_absolute_error(g.actual, g.persistence)))
br = pd.DataFrame(br); br["MAE_gain_%"] = (1 - br.MAE / br.MAE_persistence) * 100
br.round(2).to_csv("forecast_metrics_breakdown.csv", index=False)

# save test-period predictions
pd.concat({f"{t}_+{h}h": v for (t, h), v in store.items()}, names=["series", "Timestamp"]).reset_index().to_csv("test_predictions.csv", index=False)

# plot: station 1 (Anand Vihar), Nov 2025
fig, ax = plt.subplots(2, 1, figsize=(13, 7))
for a, h in zip(ax, (6, 24)):
    o = store[("AQI", h)]; o = o[(o.station == 1) & (o.index >= "2025-11-01") & (o.index < "2025-12-01")]
    a.plot(o.index, o.actual, c="k", lw=1.2, label="Actual AQI"); a.plot(o.index, o.pred, c="tab:red", lw=1.2, label=f"Model (+{h}h)")
    a.plot(o.index, o.persistence, c="tab:blue", lw=.8, alpha=.5, label="Persistence")
    a.set_title(f"Station 1 (Anand Vihar), AQI forecast +{h}h, Nov 2025 (test year)"); a.legend(ncol=3); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig("aqi_forecast_plot.png", dpi=130)
print("ALL DONE", flush=True)