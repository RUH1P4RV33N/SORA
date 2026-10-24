"""
build_national_data.py
Merge every CPCB hourly file  Raw_Data_<year>_site_<id>_<station name>_1Hr.csv  found in a folder
(sub-folders are searched too) into ONE table, adding station, city, state, latitude, longitude,
region, cluster and role columns from stations.csv.

Usage (run from the folder that holds stations.csv, or pass --stations):
    python build_national_data.py <folder_with_the_csv_files> [--stations stations.csv] [--outdir .]

Writes in --outdir:
    national_data.csv     all rows, labelled
    station_coverage.csv  per station and year: rows and % of hours present for each column
    file_report.csv       one line per input file: what it was matched to and any warning

How a file is matched to a station (stations.csv):
  1. by the site number in the file name (e.g. site_162) - if the station's city also appears in the file name
  2. otherwise by the station name + city in the file name (works for files whose site number is not in stations.csv)
  3. if the site number matches but the city is not in the file name, you get a WARNING - check that file
"""
import argparse, hashlib, re, sys
from pathlib import Path
import numpy as np, pandas as pd

FILE_RE = re.compile(r"raw_data_(\d{4})_site_(\d+)_", re.I)
KEY_COLS = ["PM2.5", "PM10", "NO2", "CO", "Ozone", "AT", "RH", "WS"]


def slug(s):
    """'Sector-25' -> 'sector25', 'T T Nagar' -> 't_t_nagar' (same style as the portal's file names)."""
    s = str(s).lower().replace("&", "and")
    s = re.sub(r"[.\-']", "", s)
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def clean_col(c):
    return c.strip().split(" (")[0].replace(" ", "_").replace("-", "_")


def parse_ts(s):
    t = pd.to_datetime(s, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    for fmt in ("%d-%m-%Y %H:%M", "%d-%m-%Y %H:%M:%S", "%Y-%m-%d %H:%M"):
        m = t.isna() & s.notna()
        if not m.any():
            break
        t[m] = pd.to_datetime(s[m], format=fmt, errors="coerce")
    return t


def pick_station(site_id, stem, st):
    by_id = st[st.site_id == site_id] if site_id is not None else st.iloc[0:0]
    if len(by_id) and slug(by_id.iloc[0].city) in stem:
        return by_id.iloc[0], ""
    hits = st[[k in stem for k in st.key]]
    if len(hits):
        row = hits.loc[hits.key.str.len().idxmax()]
        note = ""
        if len(by_id) and row.name != by_id.index[0]:
            note = (f"WARNING: site {site_id} is listed as '{by_id.iloc[0].station}' but the file name says "
                    f"'{row.station}' - used the file name")
        return row, note
    if len(by_id):
        return by_id.iloc[0], f"WARNING: city '{by_id.iloc[0].city}' not found in the file name - check this file"
    return None, "WARNING: no station in stations.csv matches this file - rows get no coordinates"


def read_one(path):
    df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    first = df.columns[0]
    n_hdr = int((df[first] == first).sum())
    df = df[df[first] != first].copy()                      # header lines pasted inside the file
    df.columns = [clean_col(c) for c in df.columns]
    df["Timestamp"] = parse_ts(df["Timestamp"])
    bad_ts = int(df["Timestamp"].isna().sum())
    df = df[df["Timestamp"].notna()]
    for c in df.columns:
        if c != "Timestamp":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df, n_hdr, bad_ts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--stations", default="stations.csv")
    ap.add_argument("--outdir", default=".")
    a = ap.parse_args()

    st = pd.read_csv(a.stations)
    st["site_id"] = pd.to_numeric(st["site_id"], errors="coerce")
    st["key"] = [slug(n.split(",")[0]) + "_" + slug(c) for n, c in zip(st.station, st.city)]

    files = sorted(p for p in Path(a.folder).rglob("*")
                   if p.is_file() and p.suffix.lower() == ".csv" and p.name.lower().startswith("raw_data"))
    if not files:
        sys.exit(f"No Raw_Data_*.csv files found under {a.folder}")
    print(f"found {len(files)} files")

    parts, report = [], []
    for p in files:
        m = FILE_RE.match(p.name)
        sid = int(m.group(2)) if m else None
        row, note = pick_station(sid, slug(p.stem), st)
        try:
            df, n_hdr, bad_ts = read_one(p)
        except Exception as e:                              # unreadable file: report and go on
            report.append(dict(file=p.name, station="", rows=0, note=f"ERROR reading file: {e}"))
            print(f"  ! {p.name}: {e}")
            continue
        if row is not None:
            for c in ["station", "city", "state", "latitude", "longitude", "region", "cluster", "role"]:
                df[c] = row[c]
        else:
            for c in ["station", "city", "state", "latitude", "longitude", "region", "cluster", "role"]:
                df[c] = np.nan
        df["site_id"] = sid if sid is not None else (row["site_id"] if row is not None else np.nan)
        df["source_file"] = p.name
        parts.append(df)
        extra = (f" | {n_hdr} repeated header line(s) removed" if n_hdr else "") + \
                (f" | {bad_ts} rows with unreadable timestamp dropped" if bad_ts else "")
        report.append(dict(file=p.name, station=row["station"] if row is not None else "", rows=len(df),
                           first_hour=df.Timestamp.min(), last_hour=df.Timestamp.max(), note=(note + extra).strip(" |")))
        if note:
            print("  " + note + f"  [{p.name}]")

    data = pd.concat(parts, ignore_index=True)
    data["year"] = data.Timestamp.dt.year

    # ---- same station + same hour twice (e.g. a file downloaded twice) ----
    before = len(data)
    data = data.drop_duplicates(["station", "Timestamp"], keep="first")
    if before - len(data):
        print(f"removed {before - len(data)} duplicate station-hour rows")

    # ---- two DIFFERENT stations carrying identical readings in the same year ----
    data["dup_flag"] = ""
    sig = {}
    for (yr, stn), g in data.groupby(["year", "station"]):
        if g["PM2.5"].notna().sum() < 100:
            continue
        g = g.sort_values("Timestamp")
        sub = g[["PM2.5", "PM10", "NO2", "AT"]].round(2)
        sub.insert(0, "hour", g["Timestamp"].dt.strftime("%Y-%m-%d %H"))
        h = hashlib.md5(sub.to_csv(index=False).encode()).hexdigest()
        sig.setdefault((yr, h), []).append(stn)
    for (yr, _), names in sig.items():
        if len(names) > 1:
            print(f"  WARNING: identical readings in {yr} for: {' | '.join(names)}  -> only one of them can be real")
            for other in names[1:]:
                data.loc[(data.year == yr) & (data.station == other), "dup_flag"] = f"identical_to: {names[0]}"

    # ---- drop columns that are empty in every file ----
    skip = {"Timestamp", "station", "city", "state", "latitude", "longitude", "region", "cluster", "role",
            "site_id", "source_file", "year", "dup_flag"}
    empty = [c for c in data.columns if c not in skip and data[c].notna().sum() == 0]
    data = data.drop(columns=empty)
    if empty:
        print("dropped empty columns:", ", ".join(empty))

    front = ["site_id", "station", "city", "state", "latitude", "longitude", "region", "cluster", "role",
             "dup_flag", "Timestamp"]
    cols = front + [c for c in data.columns if c not in front + ["year", "source_file"]] + ["source_file"]
    data = data.sort_values(["station", "Timestamp"])
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    data[cols].to_csv(out / "national_data.csv", index=False)

    # ---- coverage per station and year (against the hours that year should have) ----
    last = data.groupby("year").Timestamp.max()
    rows = []
    for (stn, yr), g in data.groupby(["station", "year"]):
        hrs = pd.date_range(f"{yr}-01-01", last[yr], freq="h")
        exp, exp_w = len(hrs), int(hrs.month.isin([10, 11, 12, 1, 2]).sum())
        wint = g[g.Timestamp.dt.month.isin([10, 11, 12, 1, 2])]
        r = dict(station=stn, year=yr, rows=len(g), rows_pct=round(100 * len(g) / exp, 1))
        for c in KEY_COLS:
            if c in g:
                r[c + "_pct"] = round(100 * g[c].notna().sum() / exp, 1)
        r["winter_PM2.5_pct"] = round(100 * wint["PM2.5"].notna().sum() / exp_w, 1) if exp_w else np.nan
        r["pm_ok_80"] = bool(r["PM2.5_pct"] >= 80 and r["PM10_pct"] >= 80)
        r["dup_flag"] = g.dup_flag.iloc[0]
        rows.append(r)
    pd.DataFrame(rows).to_csv(out / "station_coverage.csv", index=False)
    pd.DataFrame(report).to_csv(out / "file_report.csv", index=False)

    print(f"\nSaved {len(data):,} rows from {data.station.nunique()} stations to {out / 'national_data.csv'}")
    print(f"Unmatched rows (no coordinates): {int(data.latitude.isna().sum()):,}")
    print("Also saved station_coverage.csv and file_report.csv")


if __name__ == "__main__":
    main()