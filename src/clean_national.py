"""
clean_national.py  -  clean the labelled national table, drop unusable stations, write national_clean.csv

Run from the project root:
    python src/clean_national.py [--input data/national_data.csv] [--outdir data]

Steps (all counted in cleaning_report_national.csv):
  1. read (dates are day-month-year), remove rows flagged as duplicate data of another station
  2. per station: full hourly grid, pressure hPa -> mmHg, physical limits, PM2.5 <= 1.1*PM10,
     stuck-sensor runs, single-hour spikes  (bad values become empty, never invented)
  3. choose stations: PM2.5 present for >= MIN_PM25 % of hours in BOTH 2024 and 2025
  4. mark the held-out TEST stations (edit TEST_STATIONS below)
Outputs: national_clean.csv, station_selection.csv, cleaning_report_national.csv
"""
import argparse
from pathlib import Path
import numpy as np, pandas as pd

MIN_PM25 = 75          # % of hours required in each of 2024 and 2025
TEST_STATIONS = [      # held out of training: a spread of climates
    "Velachery Res. Area, Chennai - CPCB",      # south coast
    "Doon University, Dehradun - UKPCB",        # hills
    "Victoria, Kolkata - WBPCB",                # east
    "Chhoti Gwaltoli, Indore - MPPCB",          # central
    "Maninagar, Ahmedabad - GPCB",              # west
]
LIMITS = {"PM2.5": (1, 990), "PM10": (5, 990), "NO": (0, 495), "NO2": (0, 400), "NOx": (0, 495),
          "NH3": (0, 300), "SO2": (0, 300), "CO": (0, 20), "Ozone": (0, 300),
          "AT": (0, 50), "RH": (5, 100), "WS": (0, 15), "WD": (0, 360), "SR": (0, 1200), "BP": (690, 800)}
STUCK = {"PM2.5": 12, "PM10": 12, "NO": 12, "NO2": 12, "NOx": 12, "NH3": 12, "SO2": 24,
         "CO": 48, "Ozone": 24, "AT": 12, "RH": 12, "WD": 12, "BP": 48}
SPIKE_MARGIN = {"PM2.5": 150, "PM10": 300, "NO2": 100, "SO2": 40, "NH3": 80, "Ozone": 80, "CO": 4}
VALUES = list(LIMITS)
META = ["site_id", "station", "city", "state", "latitude", "longitude", "region", "cluster"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/national_data.csv")
    ap.add_argument("--outdir", default="data")
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    log = []

    def note(step, col, st, n):
        if n:
            log.append(dict(step=step, column=col, station=st, rows_affected=int(n)))

    d = pd.read_csv(a.input, dtype={"Timestamp": str}, low_memory=False)
    ts = pd.to_datetime(d["Timestamp"], format="%d-%m-%Y %H:%M", errors="coerce")
    iso = pd.to_datetime(d["Timestamp"], errors="coerce")           # in case the file has ISO dates
    d["Timestamp"] = ts.fillna(iso)
    bad = int(d["Timestamp"].isna().sum())
    if bad:
        print(f"dropping {bad} rows with unreadable dates")
    d = d[d["Timestamp"].notna()]
    for c in VALUES:
        if c not in d:
            d[c] = np.nan
    flagged = d["dup_flag"].notna() & (d["dup_flag"].astype(str) != "")
    for s, n in d[flagged].groupby("station").size().items():
        note("rows removed: identical to another station", "-", s, n)
    d = d[~flagged]
    d = d.drop_duplicates(["station", "Timestamp"])

    floors = {c: d[c].quantile(0.02) for c in STUCK}
    parts = []
    for s, g in d.groupby("station"):
        meta = g[META].iloc[0]
        g = g.set_index("Timestamp").sort_index()[VALUES]
        g = g.reindex(pd.date_range(g.index.min(), g.index.max(), freq="h"))
        g.index.name = "Timestamp"
        hpa = g["BP"] > 850
        g.loc[hpa, "BP"] = g.loc[hpa, "BP"] * 0.750062
        note("BP converted hPa -> mmHg", "BP", s, hpa.sum())
        for c, (lo, hi) in LIMITS.items():
            b = (g[c] < lo) | (g[c] > hi)
            note(f"outside [{lo},{hi}] -> empty", c, s, b.sum()); g.loc[b, c] = np.nan
        b = g["PM2.5"] > g["PM10"] * 1.10
        note("PM2.5 > 1.1*PM10 -> PM2.5 empty", "PM2.5", s, b.sum()); g.loc[b, "PM2.5"] = np.nan
        for c, nh in STUCK.items():
            x = g[c]; run = (x != x.shift()).cumsum(); size = x.groupby(run).transform("size")
            b = x.notna() & (size >= nh) & (x > floors[c])
            note(f"same value >= {nh}h -> empty", c, s, b.sum()); g.loc[b, c] = np.nan
        for c, m in SPIKE_MARGIN.items():
            med = g[c].rolling(25, center=True, min_periods=8).median()
            b = g[c] > 6 * med + m
            note("one-hour spike -> empty", c, s, b.sum()); g.loc[b, c] = np.nan
        g = g.reset_index()
        for k in META:
            g[k] = meta[k]
        parts.append(g)
    d = pd.concat(parts, ignore_index=True)
    d["year"] = d.Timestamp.dt.year

    # ---- station selection (on cleaned values) ----
    full = {2024: 8784, 2025: 8760}
    rows = []
    for s, g in d.groupby("station"):
        r = dict(station=s, city=g.city.iloc[0], region=g.region.iloc[0], rows=len(g))
        for y in (2024, 2025):
            gy = g[g.year == y]
            r[f"PM2.5_pct_{y}"] = round(100 * gy["PM2.5"].notna().sum() / full[y], 1)
        both = g[(g.year >= 2024) & (g.year <= 2025)]
        for c in ["PM10", "NO2", "CO", "Ozone", "AT", "RH", "WS"]:
            r[f"{c}_pct_2024_25"] = round(100 * both[c].notna().sum() / 17544, 1)
        ok = (r["PM2.5_pct_2024"] >= MIN_PM25) and (r["PM2.5_pct_2025"] >= MIN_PM25)
        r["selected"] = ok
        r["reason"] = "" if ok else f"PM2.5 below {MIN_PM25}% of hours in 2024 or 2025"
        r["role"] = ("TEST" if s in TEST_STATIONS else "train") if ok else "dropped"
        rows.append(r)
    sel = pd.DataFrame(rows).sort_values(["selected", "station"], ascending=[False, True])
    chosen = set(sel[sel.selected].station)
    missing = [t for t in TEST_STATIONS if t not in chosen]
    if missing:
        print("WARNING: test station(s) not selected, pick others in TEST_STATIONS:", missing)
    sel.to_csv(out / "station_selection.csv", index=False)
    keep = d[d.station.isin(chosen)].copy()
    keep["role"] = keep.station.map(dict(zip(sel.station, sel.role)))
    cols = META + ["role", "Timestamp"] + VALUES
    keep[cols].round({c: 3 for c in VALUES + ["latitude", "longitude"]}).to_csv(out / "national_clean.csv", index=False, date_format="%Y-%m-%d %H:%M:%S")
    rep = pd.DataFrame(log)
    rep.to_csv(out / "cleaning_report_national.csv", index=False)

    print(rep.groupby("step").rows_affected.sum().sort_values(ascending=False).to_string())
    print(f"\nselected {len(chosen)} of {len(sel)} stations "
          f"({int((sel.role == 'TEST').sum())} held out for testing); {len(keep):,} rows -> {out / 'national_clean.csv'}")
    print("dropped:", "; ".join(sel[~sel.selected].station.str[:30]))


if __name__ == "__main__":
    main()