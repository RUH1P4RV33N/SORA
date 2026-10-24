"""Download Delhi training data for the PBL model -> data/raw_2/pbl_delhi.csv

Uses Open-Meteo's Historical Forecast API, which is the SAME source as the live feed, so the model is
trained on the kind of data it will see at run time. Needs internet access, no account or key.

    python src/fetch_delhi_data.py
    python src/train_pbl.py --csv data/raw_2/pbl_delhi.csv
"""
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from pbl_core import LIVE_COLS
from pbl_live import HOURLY, to_model_units

LAT, LON = 28.6364, 77.2010                   # same point as the dashboard
START = date(2022, 1, 1)
END = date.today() - timedelta(days=2)        # the newest days are not always archived yet
URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OUT = Path(__file__).resolve().parents[1] / "data" / "raw_2" / "pbl_delhi.csv"


def get(start: date, end: date, variables: list[str]) -> dict:
    params = {"latitude": LAT, "longitude": LON, "start_date": start.isoformat(), "end_date": end.isoformat(),
              "hourly": ",".join(variables), "timezone": "GMT", "wind_speed_unit": "ms"}
    with urllib.request.urlopen(f"{URL}?{urllib.parse.urlencode(params, safe=',')}", timeout=90) as r:
        return json.loads(r.read())


def main() -> None:
    variables = list(HOURLY)
    frames, grid = [], (LAT, LON)
    cur = START
    while cur <= END:
        chunk_end = min(cur + timedelta(days=180), END)
        for attempt in range(5):
            try:
                payload = get(cur, chunk_end, variables)
                break
            except urllib.error.HTTPError as exc:
                if exc.code == 400 and "surface_temperature" in variables:
                    variables.remove("surface_temperature")
                    print("surface_temperature not offered here; using soil_temperature_0cm instead")
                    continue
                if exc.code == 429:
                    print("rate limit, waiting 70 s ...")
                    time.sleep(70)
                    continue
                sys.exit(f"Open-Meteo error {exc.code}: {exc.read()[:300]!r}")
            except (urllib.error.URLError, TimeoutError) as exc:
                print(f"network problem ({exc}), retrying ...")
                time.sleep(5)
        else:
            sys.exit("giving up after 5 attempts")
        grid = (payload.get("latitude", LAT), payload.get("longitude", LON))
        df = pd.DataFrame(payload["hourly"])
        df["time"] = pd.to_datetime(df["time"])
        frames.append(df.set_index("time").apply(pd.to_numeric, errors="coerce"))
        print(f"downloaded {cur} -> {chunk_end}")
        cur = chunk_end + timedelta(days=1)
        time.sleep(1)

    raw = pd.concat(frames)
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()
    model, skt_src = to_model_units(raw)

    cover = model.notna().mean()
    print("\nshare of hours with data, per column:")
    print(cover.round(3).to_string())
    print(f"skin temperature from: {skt_src}")
    bad = cover[cover < 0.8]
    if len(bad):
        sys.exit(f"\nToo many gaps in {list(bad.index)}; Open-Meteo does not archive these for Delhi. Tell me and we'll adapt.")

    model = model.interpolate(limit=6, limit_direction="both")
    out = model.copy()
    out.insert(0, "valid_time", out.index.strftime("%Y-%m-%d %H:%M:%S"))
    out["latitude"], out["longitude"] = grid
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nsaved {OUT}  ({len(out):,} hourly rows, {out.valid_time.iloc[0]} -> {out.valid_time.iloc[-1]}, grid point {grid})")
    print("next: python src/train_pbl.py --csv data/raw_2/pbl_delhi.csv")


if __name__ == "__main__":
    main()