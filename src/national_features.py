"""
national_features.py - the ONE place that turns a row of current readings into model inputs.
Used by train_national.py and predict_national.py so training and forecasting always agree.
"""
import numpy as np, pandas as pd
from common import BP_, SRC, subindex

GASES = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "SO2", "CO", "Ozone"]
WX = ["AT", "RH", "WS", "SR"]            # temperature, humidity, wind speed, solar radiation (+ WD, wind direction)
INPUT_COLS = GASES + WX + ["WD"]


def instant_aqi(df):
    """AQI from the current hour's concentrations only (no 24h averaging is possible from one row)."""
    si = pd.DataFrame({k: subindex(df[col].values, BP_[k]) for k, col in SRC.items()}, index=df.index)
    ok = (si.notna().sum(axis=1) >= 3) & si[["PM2.5", "PM10"]].notna().any(axis=1)
    return si.max(axis=1).where(ok)


def build_features(df, ts, loc="none", wx_suffix="", region_levels=None):
    """
    df  : one row per reading; columns GASES, AT/RH/WS/SR/WD (each + wx_suffix), and for loc modes
          'region' (name) or 'latitude','longitude'.
    ts  : DatetimeIndex of the readings (local time).
    loc : 'none' | 'region' | 'coords'   how place is described to the model.
    """
    X = pd.DataFrame(index=df.index)
    for c in GASES:
        X[c] = df[c].astype("float32")
    for c in WX:
        X[c] = df[c + wx_suffix].astype("float32")
    wd = np.deg2rad(df["WD" + wx_suffix].astype("float32"))
    X["WD_sin"], X["WD_cos"] = np.sin(wd), np.cos(wd)
    X["PM_ratio"] = df["PM2.5"] / df["PM10"]
    X["NO2_NO"] = df["NO2"] / (df["NO"] + 1)
    X["AQI_now"] = instant_aqi(df)
    X["hr_sin"], X["hr_cos"] = np.sin(2 * np.pi * ts.hour / 24), np.cos(2 * np.pi * ts.hour / 24)
    doy = ts.dayofyear
    X["doy_sin"], X["doy_cos"] = np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)
    X["dow"] = ts.dayofweek
    if loc == "region":
        lv = {r: i for i, r in enumerate(region_levels)}
        X["region_code"] = df["region"].map(lv).astype("float32")
    elif loc == "coords":
        X["latitude"], X["longitude"] = df["latitude"].astype("float32"), df["longitude"].astype("float32")
    return X.astype("float32")