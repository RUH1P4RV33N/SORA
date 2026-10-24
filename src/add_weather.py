"""
add_weather.py - OPTIONAL: add the same weather source for every station (fixes the stations whose own
temperature/humidity sensor is empty). Uses the free Open-Meteo historical archive (ERA5 reanalysis).

Needs internet. Run from the project root:
    python src/add_weather.py [--input data/national_clean.csv] [--out data/national_clean_wx.csv]
Then train with:  python src/train_national.py --mode eval --weather era --data data/national_clean_wx.csv

Adds columns AT_era, RH_era, WS_era, WD_era, SR_era (temperature C, humidity %, wind m/s, direction deg, solar W/m2).
Weather data: Open-Meteo.com (free for non-commercial use, please credit them).

NOTE: written against Open-Meteo's documented API but NOT run by the author (no internet where it was written).
If a request fails, the station and error are printed; check the parameters at open-meteo.com/en/docs/historical-weather-api.
"""
import argparse, json, time, urllib.parse, urllib.request
import numpy as np, pandas as pd

URL = "https://archive-api.open-meteo.com/v1/archive"
HOURLY = "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,shortwave_radiation"
RENAME = {"temperature_2m": "AT_era", "relative_humidity_2m": "RH_era", "wind_speed_10m": "WS_era",
          "wind_direction_10m": "WD_era", "shortwave_radiation": "SR_era"}


def fetch(lat, lon, start, end, tries=4):
    q = urllib.parse.urlencode(dict(latitude=lat, longitude=lon, start_date=start, end_date=end, hourly=HOURLY,
                                    wind_speed_unit="ms", timezone="Asia/Kolkata"))
    for k in range(tries):
        try:
            with urllib.request.urlopen(f"{URL}?{q}", timeout=120) as r:
                j = json.loads(r.read().decode())
            h = pd.DataFrame(j["hourly"]).rename(columns=RENAME)
            h["Timestamp"] = pd.to_datetime(h.pop("time"))
            return h
        except Exception as e:
            wait = 30 * (k + 1)
            print(f"    attempt {k + 1} failed ({e}); waiting {wait}s")
            time.sleep(wait)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/national_clean.csv")
    ap.add_argument("--out", default="data/national_clean_wx.csv")
    a = ap.parse_args()
    d = pd.read_csv(a.input, parse_dates=["Timestamp"])
    newest = (pd.Timestamp.today().normalize() - pd.Timedelta(days=7))        # the archive lags a few days
    parts = []
    for s, g in d.groupby("station"):
        start, end = g.Timestamp.min().date(), min(g.Timestamp.max(), newest).date()
        print(f"{s}: {start} -> {end}")
        w = fetch(float(g.latitude.iloc[0]), float(g.longitude.iloc[0]), start, end)
        if w is None:
            print(f"  ! no weather for {s}; its *_era columns stay empty")
            parts.append(g); continue
        parts.append(g.merge(w, on="Timestamp", how="left"))
        time.sleep(2)                                                         # be polite to the free service
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(a.out, index=False, date_format="%Y-%m-%d %H:%M:%S")
    print("saved", a.out, "| share of rows with weather:", round(out["AT_era"].notna().mean() * 100, 1), "%")


if __name__ == "__main__":
    main()