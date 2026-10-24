"""
Train the 72 h boundary-layer (PBL) model.

    python src/train_pbl.py                       # uses data/raw_2/pbl.csv
    python src/train_pbl.py --csv other.csv
    python src/train_pbl.py --force               # skip the location / clock safety checks

Saves models/pbl_models.joblib (used by the /pbl/forecast API) and results/pbl_metrics.csv.
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pbl_core import (HORIZONS, LIVE_COLS, TARGETS, THRESH, build_xy, feat_cols,
                      make_features, train_stats)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw_2" / "pbl.csv"
MODEL_PATH = ROOT / "models" / "pbl_models.joblib"
RES = ROOT / "results"

# The dashboard point (same as LAT / LON in useLiveData.ts)
TARGET_LAT, TARGET_LON, TOLERANCE_DEG = 28.6364, 77.2010, 0.5
STRIDE = 5            # coprime with 24 -> issue times cover every hour of day

PARAMS = dict(
    n_estimators=3000, learning_rate=0.05, num_leaves=63, min_child_samples=50,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.7,
    reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1,
)


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def parse_time(s):
    """Parse a column that mixes '01-01-2022 00:00' (day-first) and ISO timestamps."""
    s = s.astype(str).str.strip()
    iso = s.str.match(r"^\d{4}-\d{2}-\d{2}")
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    if iso.any():
        out[iso] = pd.to_datetime(s[iso], format="ISO8601", errors="coerce")
    rest = ~iso
    if rest.any():
        r = s[rest]
        parsed = pd.to_datetime(r, format="%d-%m-%Y %H:%M", errors="coerce")
        for fmt in ("%d-%m-%Y %H:%M:%S", "%d/%m/%Y %H:%M", "%d/%m/%Y %H:%M:%S"):
            miss = parsed.isna()
            if not miss.any():
                break
            parsed[miss] = pd.to_datetime(r[miss], format=fmt, errors="coerce")
        out[rest] = parsed
    return out


def load(path):
    raw = pd.read_csv(path)
    raw.columns = [c.strip() for c in raw.columns]
    lat = float(pd.to_numeric(raw["latitude"], errors="coerce").median()) if "latitude" in raw else None
    lon = float(pd.to_numeric(raw["longitude"], errors="coerce").median()) if "longitude" in raw else None

    df = raw.drop(columns=["latitude", "longitude"], errors="ignore")
    df["valid_time"] = parse_time(df["valid_time"])
    bad = int(df["valid_time"].isna().sum())
    if bad:
        print(f"warning: dropping {bad} rows with unparseable timestamps")
        df = df.dropna(subset=["valid_time"])
    missing = [c for c in LIVE_COLS if c not in df.columns]
    if missing:
        sys.exit(f"error: {path} is missing columns {missing}")
    for c in LIVE_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates("valid_time").set_index("valid_time").sort_index()[list(LIVE_COLS)]

    gaps = df.index.to_series().diff().dropna()
    off = float((gaps != pd.Timedelta(hours=1)).mean()) if len(gaps) else 0.0
    if off > 0.01:
        print(f"warning: {off:.1%} of timestamp gaps are not 1 h - check the date column for day/month swaps.")

    df = df.asfreq("h").interpolate(limit=6, limit_direction="both")
    df["blh"] = df["blh"].clip(lower=0)
    return df, lat, lon


def safety_checks(df, lat, lon, force):
    """Refuse to train on data that cannot match the live dashboard point."""
    problems = []
    if lat is None or lon is None:
        problems.append("the CSV has no latitude/longitude columns, so the location cannot be verified")
    elif abs(lat - TARGET_LAT) > TOLERANCE_DEG or abs(lon - TARGET_LON) > TOLERANCE_DEG:
        problems.append(f"the CSV is for ({lat:.2f}, {lon:.2f}) but the dashboard point is ({TARGET_LAT}, {TARGET_LON}); "
                        "re-download ERA5 for Delhi")
    # ERA5 timestamps are UTC: mean solar radiation must peak near solar noon, 12 - lon/15 UTC
    ref_lon = lon if lon is not None else TARGET_LON
    peak = int(df.ssrd.groupby(df.index.hour).mean().idxmax())
    expect = (12 - ref_lon / 15) % 24
    d = abs(peak - expect)
    d = min(d, 24 - d)
    print(f"location in file: lat={lat}, lon={lon} | ssrd peaks at {peak:02d}:00 (expected ~{expect:.1f} UTC)")
    if d > 2.5:
        problems.append(f"solar radiation peaks at {peak:02d}:00 but solar noon for this longitude is ~{expect:.1f} UTC; "
                        "timestamps are probably not UTC")
    if problems:
        print("\nSAFETY CHECK FAILED:")
        for p in problems:
            print("  -", p)
        if not force:
            sys.exit("\nNot training. Fix the data, or pass --force if you really want this.")
        print("  (continuing because --force was given)")


def inversion_check(df, lon):
    """How often does the training data show air warmer than the ground (stab > 0)?
    The inversion index is 0 whenever skin temperature >= air temperature, so this tells you what to expect."""
    ref_lon = lon if lon is not None else TARGET_LON
    solar_hour = (df.index.hour + ref_lon / 15) % 24
    night = (solar_hour >= 20) | (solar_hour < 6)
    afternoon = (solar_hour >= 11) & (solar_hour < 16)
    stab = df.t2m - df.skt
    out = {"share_all": float((stab > 0).mean()), "share_night": float((stab[night] > 0).mean()),
           "share_afternoon": float((stab[afternoon] > 0).mean())}
    print("\nSurface-stability check (t2m - skt > 0 means a ground-based inversion is possible):")
    print(f"  all hours {out['share_all']:.0%} | night {out['share_night']:.0%} | afternoon {out['share_afternoon']:.0%}")
    print("  -> an index of 0/10 is NORMAL in the afternoon; it should be above 0 on most nights.")
    if out["share_night"] < 0.15:
        print("  WARNING: skin is warmer than air on most nights in this data. The index will sit near 0.\n"
              "  Check that the CSV is for Delhi (land); ocean grid points behave like this.")
    return out


# ----------------------------------------------------------------------------
# training / evaluation
# ----------------------------------------------------------------------------
def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))


def train(csv=None, force=False):
    df, lat, lon = load(csv or DATA)
    print(f"data: {df.index.min()} -> {df.index.max()}  ({len(df):,} hourly rows)")
    safety_checks(df, lat, lon, force)
    inv_stats = inversion_check(df, lon)

    F = make_features(df)
    t = F.index
    t_tr, t_va = t[int(.75 * len(t))], t[int(.85 * len(t))]
    pad = pd.Timedelta(hours=72)

    tr_idx = t[t < t_tr - pad][::STRIDE]
    va_idx = t[(t >= t_tr) & (t < t_va - pad)][::STRIDE]
    te_idx = t[(t >= t_va) & (t <= t[-1] - pad)][::6]

    tr, va, te = (build_xy(F, df, i) for i in (tr_idx, va_idx, te_idx))
    cols = feat_cols(tr)
    print(f"train {len(tr):,} | val {len(va):,} | test {len(te):,} rows, {len(cols)} features")

    best, preds = {}, {}
    for tg in TARGETS:
        y = "y_" + tg
        a, b = tr.dropna(subset=[y]), va.dropna(subset=[y])
        m = lgb.LGBMRegressor(**PARAMS)
        m.fit(a[cols].astype("float32"), a[y],
              eval_set=[(b[cols].astype("float32"), b[y])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        best[tg] = max(int(m.best_iteration_ or 1), 1)
        preds[tg] = m.predict(te[cols].astype("float32"))
        print(f"  {tg}: best_iter={best[tg]}")
    preds["blh"] = np.clip(preds["blh"], 0, None)

    # ---- metrics per horizon bucket -----------------------------------------
    te = te.copy()
    for tg in TARGETS:
        te["p_" + tg] = preds[tg]
    te["p_ws10"], te["y_ws10"] = np.hypot(te.p_u10, te.p_v10), np.hypot(te.y_u10, te.y_v10)
    te["p_vent"], te["y_vent"] = te.p_blh * te.p_ws10, te.y_blh * te.y_ws10
    te["p_stab"], te["y_stab"] = te.p_t2m - te.p_skt, te.y_t2m - te.y_skt
    te = te.dropna(subset=["y_blh", "y_naive", "y_u10", "y_v10", "y_t2m", "y_skt"])

    rows = []
    for name, lo, hi in [("1-6h", 1, 6), ("7-24h", 7, 24), ("25-48h", 25, 48),
                         ("49-72h", 49, 72), ("all", 1, 72)]:
        s = te[(te.h >= lo) & (te.h <= hi)]
        rows.append({
            "horizon": name,
            "BLH_MAE_m": mean_absolute_error(s.y_blh, s.p_blh),
            "BLH_RMSE_m": rmse(s.y_blh, s.p_blh),
            "BLH_R2": r2_score(s.y_blh, s.p_blh),
            "naive_MAE_m": mean_absolute_error(s.y_blh, s.y_naive),
            "naive_R2": r2_score(s.y_blh, s.y_naive),
            "windspeed_MAE": mean_absolute_error(s.y_ws10, s.p_ws10),
            "ventilation_R2": r2_score(s.y_vent, s.p_vent),
            "stability_R2": r2_score(s.y_stab, s.p_stab),
            "lid_state_acc(<500m)": float(((s.y_blh < THRESH) == (s.p_blh < THRESH)).mean()),
        })
    met = pd.DataFrame(rows).round(3)
    RES.mkdir(exist_ok=True)
    met.to_csv(RES / "pbl_metrics.csv", index=False)
    print("\n", met.to_string(index=False))

    # ---- event accuracy (next 24 h) ------------------------------------------
    ev = []
    for _, g in te[te.h <= 24].groupby("issue"):
        if len(g) < 24:
            continue
        g = g.sort_values("h")
        ya, pa = g.y_blh.values, g.p_blh.values
        r = {"peak_err_m": pa.max() - ya.max(), "peak_time_err_h": int(pa.argmax()) - int(ya.argmax())}
        ra, rp = np.where(ya >= THRESH)[0], np.where(pa >= THRESH)[0]
        if len(ra) and len(rp):
            r["recovery_time_err_h"] = int(rp[0]) - int(ra[0])
        ev.append(r)
    ev = pd.DataFrame(ev)
    if len(ev):
        print("\nNext-24h event accuracy (pred - actual):")
        print(f"  peak height MAE {ev.peak_err_m.abs().mean():.0f} m | "
              f"peak time within +-1 h: {(ev.peak_time_err_h.abs() <= 1).mean():.0%}")
        if "recovery_time_err_h" in ev:
            x = ev.recovery_time_err_h.dropna()
            print(f"  recovery time within +-1 h: {(x.abs() <= 1).mean():.0%}")

    # ---- final refit on all data --------------------------------------------
    all_idx = t[t <= t[-1] - pad][::STRIDE]
    full = build_xy(F, df, all_idx)
    models = {}
    for tg in TARGETS:
        y = "y_" + tg
        a = full.dropna(subset=[y])
        p = dict(PARAMS, n_estimators=int(best[tg] * 1.15) + 1)
        models[tg] = lgb.LGBMRegressor(**p).fit(a[cols].astype("float32"), a[y])

    stab = df.t2m - df.skt
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({
        "models": models, "cols": cols, "live_cols": list(LIVE_COLS),
        "lat": lat, "lon": lon,
        "trained_through": str(df.index.max()),
        "trained_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "inv_hi": max(float(stab.quantile(.95)), 1.0),
        "inversion_stats": inv_stats,
        "train_stats": train_stats(df),
        "metrics": met.astype(object).where(met.notna(), None).to_dict("records"),
    }, MODEL_PATH)
    print(f"\nsaved {MODEL_PATH}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--force", action="store_true", help="skip the location / UTC-clock safety checks")
    a = ap.parse_args()
    train(a.csv, a.force)