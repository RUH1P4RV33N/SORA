"""Shared code: AQI calculation, per-station preparation and feature engineering."""
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

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


TARGETS  = ["AQI", "PM2.5", "PM10", "NO2", "CO", "Ozone"]
HORIZONS = [1, 6, 12, 24, 36, 48, 60, 72]
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

def prepare(d):
    """d = cleaned data (data.csv layout). Returns D (rows), XS (features), feats (feature names)."""
    frames = []
    for s, g in d.groupby("station"):
        g = g.set_index("Timestamp").sort_index().asfreq("h")
        g["station"] = s
        g = g.drop(columns=["Benzene", "Toluene"], errors="ignore")
        cols = [c for c in g.columns if c != "station"]
        g[cols] = g[cols].interpolate(limit=6, limit_area="inside")
        frames.append(add_aqi(g))
    D = pd.concat(frames)
    city = D.reset_index().pivot(index="Timestamp", columns="station", values="PM2.5").mean(axis=1)
    XS = pd.concat([features(g) for _, g in D.groupby("station")])
    XS = XS.join(pd.DataFrame({f"city_pm25_l{l}": city.shift(l) for l in (0, 6, 24)}))
    XS["_st"] = XS["station"]; XS["_t"] = XS.index
    feats = [c for c in XS.columns if c not in ("_st", "_t")]
    XS = XS.reset_index(drop=True); D = D.reset_index()
    return D, XS, feats

def make_model(n, feats):
    return HistGradientBoostingRegressor(loss="squared_error", learning_rate=0.06, max_iter=n, max_leaf_nodes=12,
        min_samples_leaf=60, l2_regularization=5.0, max_bins=63, early_stopping=False, random_state=0,
        categorical_features=[feats.index("station")])