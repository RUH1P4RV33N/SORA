"""
train_national.py - train and test the national forecast model (current readings -> +1/+6/+12/+24 h).

Run from the project root:
  python src/train_national.py --mode locexp                 # does describing the place help? (small experiment)
  python src/train_national.py --mode eval  --loc none       # full tests A, B, C  -> results/*.csv
  python src/train_national.py --mode final --loc none       # train on everything -> models/national_models.joblib

Options: --data data/national_clean.csv  --weather measured|era  --stride 3  --horizons 1,6,12,24
         --targets AQI,PM2.5,PM10,NO2,CO,Ozone

The three tests (a model is only judged on data it has not been trained on):
  A  future, known places : train on all stations up to Sep 2025, test Oct-Dec 2025
  B  new places           : train without the TEST stations, test on the TEST stations (same years)
  C  new places + future  : train without the TEST stations up to Sep 2025, test on them Oct-Dec 2025
"""
import argparse, time, warnings
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from common import add_aqi, category
from national_features import build_features, INPUT_COLS

warnings.filterwarnings("ignore")
ALL_TARGETS = ["AQI", "PM2.5", "PM10", "NO2", "CO", "Ozone"]
TRAIN_END, VAL_END = pd.Timestamp("2025-06-30 23:00"), pd.Timestamp("2025-09-30 23:00")


def load(path, weather):
    d = pd.read_csv(path, parse_dates=["Timestamp"])
    meta_cols = ["site_id", "station", "city", "state", "latitude", "longitude", "region", "cluster", "role"]
    frames = []
    for s, g in d.groupby("station"):
        m = g[meta_cols].iloc[0]
        g = g.set_index("Timestamp").sort_index()
        g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="h"))
        g.index.name = "Timestamp"
        for k in meta_cols:
            g[k] = m[k]
        tmp = g[["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone", "NH3"]].interpolate(limit=6, limit_area="inside")
        g["AQI"] = add_aqi(tmp)["AQI"]                      # true CPCB AQI (24h / 8h averages) = the AQI target
        frames.append(g.reset_index())
    D = pd.concat(frames, ignore_index=True)
    if weather == "era":
        for c in ["AT", "RH", "WS", "WD", "SR"]:
            if c + "_era" not in D:
                raise SystemExit(f"--weather era needs column {c}_era (run add_weather.py first)")
    return D


def shifted_targets(D, targets, horizons):
    out = {}
    by = D.groupby("station")
    for t in targets:
        for h in horizons:
            out[(t, h)] = by[t].shift(-h).values
    return out


def fit(Xtr, ztr, n, cat_idx):
    return HistGradientBoostingRegressor(
        loss="squared_error", learning_rate=0.06, max_iter=n, max_leaf_nodes=15, min_samples_leaf=200,
        l2_regularization=5.0, max_bins=63, early_stopping=False, random_state=0,
        categorical_features=cat_idx or None).fit(Xtr, ztr)


def metrics(y, p, now):
    return dict(n=len(y), MAE=mean_absolute_error(y, p), RMSE=mean_squared_error(y, p) ** 0.5,
                R2=r2_score(y, p), MAE_now=mean_absolute_error(y, now))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["locexp", "eval", "final"], required=True)
    ap.add_argument("--loc", choices=["none", "region", "coords"], default="none")
    ap.add_argument("--weather", choices=["measured", "era"], default="measured")
    ap.add_argument("--data", default="data/national_clean.csv")
    ap.add_argument("--results", default="results")
    ap.add_argument("--models", default="models")
    ap.add_argument("--stride", type=int, default=3, help="use ~1 in N training rows (neighbouring hours are near-duplicates)")
    ap.add_argument("--horizons", default="1,6,12,24,36,48,60,72")
    ap.add_argument("--targets", default=",".join(ALL_TARGETS))
    ap.add_argument("--max-iter", type=int, default=250)
    a = ap.parse_args()
    horizons = [int(x) for x in a.horizons.split(",")]
    targets = a.targets.split(",")
    res_dir, mod_dir = Path(a.results), Path(a.models)
    res_dir.mkdir(exist_ok=True); mod_dir.mkdir(exist_ok=True)

    t0 = time.time()
    D = load(a.data, a.weather)
    suffix = "_era" if a.weather == "era" else ""
    region_levels = sorted(D["region"].dropna().unique())
    ts = pd.DatetimeIndex(D["Timestamp"])
    Xall = build_features(D, ts, loc="coords", wx_suffix=suffix)           # has latitude/longitude
    Xall["region_code"] = D["region"].map({r: i for i, r in enumerate(region_levels)}).astype("float32").values
    Y = shifted_targets(D, targets, horizons)
    rng = np.random.default_rng(0)
    keep_row = rng.random(len(D)) < 1.0 / a.stride
    is_test = (D["role"] == "TEST").values
    print(f"{len(D):,} rows, {D.station.nunique()} stations ({D[D.role == 'TEST'].station.nunique()} held out), "
          f"loaded in {time.time() - t0:.0f}s", flush=True)

    def columns(loc):
        base = [c for c in Xall.columns if c not in ("latitude", "longitude", "region_code")]
        return base + {"none": [], "region": ["region_code"], "coords": ["latitude", "longitude"]}[loc]

    def run(loc, targets, horizons, exps):
        cols = columns(loc); X = Xall[cols]
        cat_idx = [cols.index("region_code")] if loc == "region" else []
        rows, per_station, best_iters = [], [], {}
        tt = D["Timestamp"].values
        for t in targets:
            now_all = (Xall["AQI_now"] if t == "AQI" else D[t]).values
            for h in horizons:
                y = Y[(t, h)]
                ok = ~np.isnan(y) & ~np.isnan(now_all)
                z = np.log1p(np.where(ok, y, 0)) - np.log1p(np.where(ok, now_all, 0))
                t_end = tt + np.timedelta64(h, "h")
                tr_time = t_end <= np.datetime64(TRAIN_END)
                trval_time = t_end <= np.datetime64(VAL_END)
                val_time = (tt > np.datetime64(TRAIN_END)) & trval_time
                test_time = tt > np.datetime64(VAL_END)
                # choose the number of boosting rounds on the validation months (all stations)
                s_tr = ok & tr_time & keep_row; s_va = ok & val_time
                m = fit(X[s_tr], z[s_tr], a.max_iter, cat_idx)
                errs = [mean_absolute_error(z[s_va], p) for p in m.staged_predict(X[s_va])]
                best = int(np.argmin(errs)) + 1; best_iters[(t, h)] = best

                def run_exp(name, train_mask, test_mask):
                    mm = fit(X[train_mask], z[train_mask], best, cat_idx)
                    zp = mm.predict(X[test_mask])
                    p = np.expm1(np.log1p(now_all[test_mask]) + zp).clip(min=0)
                    if t == "AQI": p = p.clip(max=500)
                    r = dict(loc=loc, experiment=name, target=t, horizon_h=h, best_iter=best,
                             **metrics(y[test_mask], p, now_all[test_mask]))
                    if t == "AQI":
                        r["cat_acc"] = (category(pd.Series(y[test_mask])).values == category(pd.Series(p)).values).mean()
                        r["cat_acc_now"] = (category(pd.Series(y[test_mask])).values == category(pd.Series(now_all[test_mask])).values).mean()
                    rows.append(r)
                    if name in ("B", "C"):
                        df = pd.DataFrame({"station": D["station"].values[test_mask], "y": y[test_mask],
                                           "p": p, "now": now_all[test_mask]})
                        for s_, g in df.groupby("station"):
                            per_station.append(dict(loc=loc, experiment=name, target=t, horizon_h=h, station=s_,
                                                    n=len(g), MAE=mean_absolute_error(g.y, g.p), MAE_now=mean_absolute_error(g.y, g.now)))
                if "A" in exps: run_exp("A", ok & trval_time & keep_row, ok & test_time)
                if "B" in exps: run_exp("B", ok & ~is_test & keep_row, ok & is_test)
                if "C" in exps: run_exp("C", ok & ~is_test & trval_time & keep_row, ok & is_test & test_time)
                print(f"  [{loc}] {t} +{h}h done ({time.time() - t0:.0f}s)", flush=True)
        return pd.DataFrame(rows), pd.DataFrame(per_station), best_iters

    if a.mode == "locexp":
        allr = []
        for loc in ["none", "region", "coords"]:
            r, _, _ = run(loc, ["PM2.5"], [6, 24], ["B", "C"]); allr.append(r)
        r = pd.concat(allr); r["MAE_gain_%"] = (1 - r.MAE / r.MAE_now) * 100
        r.round(3).to_csv(res_dir / "location_experiment.csv", index=False)
        print(r.pivot_table(index=["experiment", "horizon_h"], columns="loc", values="MAE").round(2).to_string())

    elif a.mode == "eval":
        r, ps, _ = run(a.loc, targets, horizons, ["A", "B", "C"])
        r["MAE_gain_%"] = (1 - r.MAE / r.MAE_now) * 100; ps["MAE_gain_%"] = (1 - ps.MAE / ps.MAE_now) * 100
        r.round(3).to_csv(res_dir / "national_metrics.csv", index=False)
        ps.round(3).to_csv(res_dir / "national_metrics_by_station.csv", index=False)
        print(r.pivot_table(index=["target", "horizon_h"], columns="experiment", values="MAE_gain_%").round(1).to_string())

    else:  # final
        mpath = res_dir / "national_metrics.csv"
        if not mpath.exists():
            raise SystemExit("run --mode eval first (it chooses the number of boosting rounds)")
        bi = pd.read_csv(mpath).query("experiment == 'A'").set_index(["target", "horizon_h"])["best_iter"].to_dict()
        cols = columns(a.loc); X = Xall[cols]
        cat_idx = [cols.index("region_code")] if a.loc == "region" else []
        models = {}
        for t in targets:
            now_all = (Xall["AQI_now"] if t == "AQI" else D[t]).values
            for h in horizons:
                y = Y[(t, h)]; ok = ~np.isnan(y) & ~np.isnan(now_all) & keep_row
                z = np.log1p(np.where(ok, y, 0)) - np.log1p(np.where(ok, now_all, 0))
                models[(t, h)] = fit(X[ok], z[ok], int(bi[(t, h)]), cat_idx)
                print(f"  trained {t} +{h}h on {int(ok.sum()):,} rows ({time.time() - t0:.0f}s)", flush=True)
        stn = D.drop_duplicates("station")[["station", "city", "state", "region", "latitude", "longitude"]]
        joblib.dump(dict(models=models, feats=cols, loc=a.loc, weather=a.weather, region_levels=region_levels,
                         trained_through=str(D.Timestamp.max()), stations=stn), mod_dir / "national_models.joblib", compress=3)
        print("saved", mod_dir / "national_models.joblib")


if __name__ == "__main__":
    main()