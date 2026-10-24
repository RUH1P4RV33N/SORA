"""Live service for GET /pbl/forecast.

  hour 0       = real-time values from the Open-Meteo API (the latest hour)
  hours 1..72  = produced ONLY by the trained model (models/pbl_models.joblib)

The model is fed the last 5 days of Open-Meteo data (converted to the units it was trained on).
If anything fails, the endpoint returns an error. It never fills in made-up numbers.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from pbl_core import LIVE_COLS, THRESH, forecast_frame, inversion_index, iso_utc, unit_check, wind_dir

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "pbl_models.joblib"
URL = "https://api.open-meteo.com/v1/forecast"
PAST_DAYS = 5
CACHE_SECONDS = 15 * 60
IST = timezone(timedelta(hours=5, minutes=30))
DASHBOARD_LAT, DASHBOARD_LON = 28.6364, 77.2010   # same point as useLiveData.ts
ACTIVE_MIN_INDEX = 0.5          # index below this is shown as "no inversion"
HOURLY = ["temperature_2m", "dew_point_2m", "pressure_msl", "surface_pressure", "cloud_cover",
          "wind_speed_10m", "wind_direction_10m", "wind_speed_100m", "wind_direction_100m", "wind_gusts_10m",
          "boundary_layer_height", "surface_temperature", "soil_temperature_0cm",
          "shortwave_radiation", "precipitation"]

router = APIRouter()


class PblUnavailable(RuntimeError):
    pass


_lock = threading.Lock()
_cache: dict = {"at": 0.0, "mtime": None, "payload": None}
_bundle_cache: dict = {"mtime": None, "bundle": None}


def _bundle() -> tuple[dict, float]:
    if not MODEL_PATH.exists():
        raise PblUnavailable("PBL model not trained yet. Run: python src/train_pbl.py")
    mtime = MODEL_PATH.stat().st_mtime
    if _bundle_cache["mtime"] != mtime:
        _bundle_cache.update(bundle=joblib.load(MODEL_PATH), mtime=mtime)
    bundle = _bundle_cache["bundle"]
    missing = [k for k in ("models", "cols", "lat", "lon", "inv_hi", "train_stats") if k not in bundle]
    if missing:
        raise PblUnavailable(f"models/pbl_models.joblib was trained with an older script (missing {missing}). "
                             "Retrain: python src/train_pbl.py")
    return bundle, mtime


def _fetch(lat: float, lon: float) -> pd.DataFrame:
    """Hourly Open-Meteo data, UTC, native units (past 5 days + forecast days; only past hours are used)."""
    variables = list(HOURLY)
    payload, last = None, None
    for attempt in range(4):
        params = {"latitude": lat, "longitude": lon, "hourly": ",".join(variables), "past_days": PAST_DAYS,
                  "forecast_days": 1, "timezone": "GMT", "wind_speed_unit": "ms"}
        try:
            with urllib.request.urlopen(f"{URL}?{urllib.parse.urlencode(params, safe=',')}", timeout=40) as r:
                payload = json.loads(r.read())
            break
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code == 400 and "surface_temperature" in variables:
                variables.remove("surface_temperature")       # not offered for this model: use soil_temperature_0cm
                continue
            if exc.code != 429:
                raise PblUnavailable(f"Open-Meteo returned HTTP {exc.code}") from exc
            time.sleep(5 * (attempt + 1))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last = exc
            time.sleep(2)
    if payload is None:
        raise PblUnavailable(f"Open-Meteo unreachable: {last}")
    hourly = payload.get("hourly")
    if not hourly or "time" not in hourly:
        raise PblUnavailable("Open-Meteo returned no hourly data")
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time").apply(pd.to_numeric, errors="coerce")


def to_model_units(raw: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Open-Meteo -> ERA5 names/units. Returns the frame and which variable supplied skin temperature."""
    def uv(speed: str, direction: str):
        rad = np.radians(raw[direction])
        return -raw[speed] * np.sin(rad), -raw[speed] * np.cos(rad)   # direction = where the wind comes FROM

    if "surface_temperature" in raw and raw["surface_temperature"].notna().mean() > 0.9:
        skt_c, skt_src = raw["surface_temperature"], "surface_temperature"
    else:
        skt_c, skt_src = raw["soil_temperature_0cm"], "soil_temperature_0cm"

    u10, v10 = uv("wind_speed_10m", "wind_direction_10m")
    u100, v100 = uv("wind_speed_100m", "wind_direction_100m")
    out = pd.DataFrame({
        "u100": u100, "v100": v100, "u10": u10, "v10": v10,
        "fg10": raw["wind_gusts_10m"],
        "d2m": raw["dew_point_2m"] + 273.15, "t2m": raw["temperature_2m"] + 273.15,
        "blh": raw["boundary_layer_height"],
        "msl": raw["pressure_msl"] * 100.0, "skt": skt_c + 273.15, "sp": raw["surface_pressure"] * 100.0,
        "ssrd": raw["shortwave_radiation"] * 3600.0,      # W/m2 -> J/m2 per hour
        "tcc": raw["cloud_cover"] / 100.0, "tp": raw["precipitation"] / 1000.0,
    })
    return out[list(LIVE_COLS)], skt_src


def _ist_iso(ts) -> str:
    return pd.Timestamp(ts).tz_localize("UTC").tz_convert(IST).isoformat()


def _num(x, nd=1):
    x = float(x)
    if not np.isfinite(x):
        raise PblUnavailable("non-finite value in model output")
    return round(x, nd) if nd else int(round(x))


def _compute(bundle: dict, now_utc: pd.Timestamp, raw: pd.DataFrame | None = None) -> dict:
    lat, lon = bundle["lat"], bundle["lon"]
    raw = _fetch(lat, lon) if raw is None else raw
    full, skt_src = to_model_units(raw)
    issue = now_utc.floor("h")
    hist = full.loc[:issue].interpolate(limit=3, limit_area="inside")
    if hist.empty or hist.index[-1] != issue:
        raise PblUnavailable("Open-Meteo data does not cover the current hour")
    try:
        fc = forecast_frame(bundle, hist)
    except ValueError as exc:
        raise PblUnavailable(str(exc)) from exc
    check = unit_check(bundle, hist)

    # hour 0 = live value, hours 1..72 = model output (no blending, no interpolation)
    cur = hist.iloc[-1]
    blh = np.concatenate([[cur.blh], fc.blh.values])
    u10 = np.concatenate([[cur.u10], fc.u10.values])
    v10 = np.concatenate([[cur.v10], fc.v10.values])
    t2m = np.concatenate([[cur.t2m], fc.t2m.values])
    skt = np.concatenate([[cur.skt], fc.skt.values])
    times = [issue] + list(pd.to_datetime(fc.time.values))
    ws, wd = np.hypot(u10, v10), wind_dir(u10, v10)
    vent = np.clip(blh, 0, None) * ws
    stab = t2m - skt
    inv = inversion_index(stab, bundle["inv_hi"])

    series = [{
        "hour": k, "valid_at": _ist_iso(times[k]), "observed": k == 0,
        "blh_m": _num(max(blh[k], 0), 0), "ws10_ms": _num(ws[k], 2), "wd10_deg": _num(wd[k], 0),
        "ventilation_m2s": _num(vent[k], 0), "stability_k": _num(stab[k], 1),
        "inversion_index": _num(inv[k], 1), "t2m_c": _num(t2m[k] - 273.15, 1), "skt_c": _num(skt[k] - 273.15, 1),
    } for k in range(len(times))]
    s0 = series[0]

    win = [p for p in series if 1 <= p["hour"] <= 24]
    low = min(win, key=lambda p: p["blh_m"])
    peak = max(win, key=lambda p: p["blh_m"])
    strong = max(win, key=lambda p: p["inversion_index"])

    if s0["blh_m"] < THRESH:
        r = next((p for p in series if p["hour"] > 0 and p["blh_m"] >= THRESH), None)
        recovery = ({"state": "recovers", "hour": r["hour"], "valid_at": r["valid_at"]} if r else {"state": "none_in_72h"})
    else:
        c = next((p for p in series if p["hour"] > 0 and p["blh_m"] < THRESH), None)
        recovery = {"state": "open", "next_close_hour": c["hour"] if c else None, "next_close_at": c["valid_at"] if c else None}

    events = [{"kind": "lowest", "hour": low["hour"], "valid_at": low["valid_at"], "blh_m": low["blh_m"]},
              {"kind": "peak", "hour": peak["hour"], "valid_at": peak["valid_at"], "blh_m": peak["blh_m"]}]
    if strong["inversion_index"] >= ACTIVE_MIN_INDEX:
        events.append({"kind": "strongest_inversion", "hour": strong["hour"], "valid_at": strong["valid_at"],
                       "inversion_index": strong["inversion_index"]})
    if recovery["state"] == "recovers" and recovery["hour"] <= 24:
        events.append({"kind": "recovery", "hour": recovery["hour"], "valid_at": recovery["valid_at"],
                       "blh_m": series[recovery["hour"]]["blh_m"]})
    if peak["blh_m"] >= THRESH:
        cp = next((p for p in series if peak["hour"] < p["hour"] <= 24 and p["blh_m"] < THRESH), None)
        if cp:
            events.append({"kind": "closing", "hour": cp["hour"], "valid_at": cp["valid_at"], "blh_m": cp["blh_m"]})
    events.sort(key=lambda e: e["hour"])

    flat = max(p["inversion_index"] for p in series) < ACTIVE_MIN_INDEX
    return {
        "issued_at": _ist_iso(issue), "threshold_m": THRESH,
        "now": {"blh_m": s0["blh_m"], "ws10_ms": s0["ws10_ms"], "wd10_deg": s0["wd10_deg"],
                "ventilation_m2s": s0["ventilation_m2s"], "stability_k": s0["stability_k"],
                "inversion_index": s0["inversion_index"], "inversion_active": s0["inversion_index"] >= ACTIVE_MIN_INDEX},
        "summary": {
            "lid_vs_tomorrow_m": series[24]["blh_m"] - s0["blh_m"],
            "peak": {"hour": peak["hour"], "valid_at": peak["valid_at"], "blh_m": peak["blh_m"], "ventilation_m2s": peak["ventilation_m2s"]},
            "lowest": {"hour": low["hour"], "valid_at": low["valid_at"], "blh_m": low["blh_m"]},
            "recovery": recovery,
        },
        "events": events, "series": series,
        "model": {"file": "models/pbl_models.joblib", "updated_at": bundle.get("trained_at", "unknown"),
                  "horizon_h": series[-1]["hour"], "trained_through": bundle.get("trained_through")},
        "source": {"inputs": f"Open-Meteo live data (hour 0) feeding the trained model (hours 1-72); last {PAST_DAYS} days as input",
                   "latitude": lat, "longitude": lon, "history_h": int(len(hist))},
        "validation": bundle.get("metrics"),
        "diagnostics": {
            "unit_check": check, "skin_temperature_source": skt_src,
            "location_matches_dashboard": abs(lat - DASHBOARD_LAT) <= 0.5 and abs(lon - DASHBOARD_LON) <= 0.5,
            "inversion_proxy": {**(bundle.get("inversion_stats") or {}), "flat_zero_all_72h": flat,
                                "note": "Index is 0 when skin temperature >= air temperature (normal in the afternoon)."},
        },
        "stale": False,
    }


def get_forecast() -> dict:
    with _lock:
        bundle, mtime = _bundle()
        if _cache["payload"] is not None and _cache["mtime"] == mtime and time.time() - _cache["at"] < CACHE_SECONDS:
            return _cache["payload"]
        try:
            payload = _compute(bundle, pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None))
        except PblUnavailable:
            if _cache["payload"] is not None and _cache["mtime"] == mtime:
                return {**_cache["payload"], "stale": True}
            raise
        _cache.update(at=time.time(), mtime=mtime, payload=payload)
        return payload


@router.get("/pbl/forecast")
def pbl_forecast() -> dict:
    try:
        return get_forecast()
    except PblUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc