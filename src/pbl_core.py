"""Shared feature engineering + inference for the 72 h boundary-layer (PBL) model.

Used by train_pbl.py (training) and pbl_live.py (live API). Only variables that Open-Meteo
also publishes are used, so the model can be fed live data at inference time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TARGETS = ["blh", "u10", "v10", "t2m", "skt"]
HORIZONS = np.arange(1, 73)
THRESH = 500                      # open-mixing threshold (m)
MIN_HISTORY_H = 80                # longest lag/rolling window is 72 h

# ERA5 names (as in pbl.csv). Each has a live Open-Meteo counterpart (see pbl_live.py).
LIVE_COLS = ("u100", "v100", "u10", "v10", "fg10", "d2m", "t2m", "blh",
             "msl", "skt", "sp", "ssrd", "tcc", "tp")


def wind_dir(u, v):
    """Meteorological direction the wind blows FROM, degrees."""
    return (270 - np.degrees(np.arctan2(v, u))) % 360


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    f = {c: df[c] for c in df.columns}
    ws10, ws100 = np.hypot(df.u10, df.v10), np.hypot(df.u100, df.v100)
    wd = np.radians(wind_dir(df.u10, df.v10))
    f.update(
        ws10=ws10, ws100=ws100, shear=ws100 - ws10,
        wd_sin=np.sin(wd), wd_cos=np.cos(wd),
        dew_dep=df.t2m - df.d2m,
        stab=df.t2m - df.skt,                 # >0 : surface colder than air (stable)
        vent=df.blh * ws10,
        hour_sin=np.sin(2 * np.pi * df.index.hour / 24),
        hour_cos=np.cos(2 * np.pi * df.index.hour / 24),
        doy_sin=np.sin(2 * np.pi * df.index.dayofyear / 365.25),
        doy_cos=np.cos(2 * np.pi * df.index.dayofyear / 365.25),
    )
    for l in (1, 2, 3, 6, 12, 24, 48, 72):
        f[f"blh_l{l}"] = df.blh.shift(l)
    for c, s in (("t2m", df.t2m), ("ws10", ws10), ("ssrd", df.ssrd),
                 ("tcc", df.tcc), ("msl", df.msl), ("stab", f["stab"])):
        for l in (1, 3, 6, 24):
            f[f"{c}_l{l}"] = s.shift(l)
    f["msl_d6"] = df.msl.diff(6)
    f["msl_d24"] = df.msl.diff(24)
    f["t2m_d6"] = df.t2m.diff(6)
    for w in (24, 72):
        r = df.blh.rolling(w)
        f[f"blh_mean{w}"], f[f"blh_max{w}"], f[f"blh_min{w}"] = r.mean(), r.max(), r.min()
    f["ws10_mean24"] = ws10.rolling(24).mean()
    f["tcc_mean24"] = df.tcc.rolling(24).mean()
    f["ssrd_sum24"] = df.ssrd.rolling(24).sum()
    f["tp_sum24"] = df.tp.rolling(24).sum()
    f["tp_sum72"] = df.tp.rolling(72).sum()
    return pd.DataFrame(f).dropna()


def build_xy(F: pd.DataFrame, df: pd.DataFrame, idx, horizons=HORIZONS) -> pd.DataFrame:
    """Stack issue-time features with horizon h and the target value at issue+h."""
    frames = []
    for h in horizons:
        tt = idx + pd.Timedelta(hours=int(h))
        X = F.loc[idx].copy()
        X["h"] = h
        X["t_hour_sin"] = np.sin(2 * np.pi * tt.hour / 24)
        X["t_hour_cos"] = np.cos(2 * np.pi * tt.hour / 24)
        X["t_doy_sin"] = np.sin(2 * np.pi * tt.dayofyear / 365.25)
        X["t_doy_cos"] = np.cos(2 * np.pi * tt.dayofyear / 365.25)
        X["issue"] = np.asarray(idx)
        X["target_time"] = np.asarray(tt)
        for t in TARGETS:
            X["y_" + t] = df[t].reindex(tt).values
        back = tt - pd.Timedelta(hours=int(24 * np.ceil(h / 24)))
        X["y_naive"] = df["blh"].reindex(back).values      # same hour yesterday baseline
        frames.append(X)
    return pd.concat(frames, ignore_index=True)


def feat_cols(D: pd.DataFrame) -> list[str]:
    return [c for c in D.columns if not c.startswith("y_") and c not in ("issue", "target_time")]


def iso_utc(t) -> str:
    return pd.Timestamp(t).isoformat() + "Z"


def forecast_frame(bundle: dict, hist: pd.DataFrame):
    """72 h forecast issued at the last row of `hist` (hourly, UTC, model units).

    Returns (forecast DataFrame, dict describing the last observed hour).
    Raises ValueError if the history is too short or has gaps in the latest hours.
    """
    hist = hist[list(LIVE_COLS)]
    if len(hist) < MIN_HISTORY_H:
        raise ValueError(f"need at least {MIN_HISTORY_H} hours of history, got {len(hist)}")
    F = make_features(hist)
    if F.empty or F.index[-1] != hist.index[-1]:
        raise ValueError("latest hours contain missing values; cannot build model inputs")
    idx = F.index[[-1]]
    X = build_xy(F, hist, idx)
    Xc = X[bundle["cols"]].astype("float32")
    P = {tg: bundle["models"][tg].predict(Xc) for tg in TARGETS}

    fc = pd.DataFrame({"time": X["target_time"].values, "hour_ahead": X["h"].astype(int).values})
    fc["blh"] = np.clip(P["blh"], 0, None)
    for tg in ("u10", "v10", "t2m", "skt"):
        fc[tg] = P[tg]
    return fc


def train_stats(df: pd.DataFrame) -> dict:
    """Per-column range of the training data, used to detect unit/scale mismatches at inference."""
    return {c: {"q01": float(df[c].quantile(.01)), "q99": float(df[c].quantile(.99)),
                "median": float(df[c].median())} for c in LIVE_COLS}


def unit_check(bundle: dict, hist: pd.DataFrame) -> dict:
    """Compare the median of the last 72 live hours with the training range (1st-99th pct, widened 25 %)."""
    stats = bundle.get("train_stats")
    if not stats:
        return {"checked": False, "mismatches": [], "note": "Model bundle has no training statistics."}
    win = hist.iloc[-72:]
    bad = []
    for c in LIVE_COLS:
        s = stats[c]
        pad = 0.25 * (s["q99"] - s["q01"]) + 1e-9
        med = float(win[c].median())
        if med < s["q01"] - pad or med > s["q99"] + pad:
            bad.append({"column": c, "live_median": round(med, 4),
                        "train_q01": round(s["q01"], 4), "train_q99": round(s["q99"], 4)})
    return {"checked": True, "mismatches": bad,
            "note": "Live 72 h median compared with the training 1st-99th percentile range."}


def inversion_index(stab, hi: float):
    """0 = skin temperature at/above air temperature; 10 = stability at the 95th percentile of history."""
    return 10 * np.clip(np.asarray(stab, dtype=float) / hi, 0, 1)