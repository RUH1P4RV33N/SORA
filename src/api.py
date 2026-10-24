"""HTTP API for the trained national forecast model and the 72 h PBL model.

Run from the project root:
    uvicorn src.api:app --reload --port 8000

Endpoints
    GET /health
    GET /forecast             single-point forecast from readings you pass in
    GET /pbl/forecast         72 h boundary-layer / ventilation / wind forecast from the trained PBL model
    GET /stations             every station stored in the model bundle
    GET /stations/live        cached live snapshot for all stations (refreshed in the background)
    GET /stations/forecast    model forecast for one station, built from its cached live readings
"""
from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from national_features import build_features  # noqa: E402
from predict_national import HORIZONS, TARGETS, forecast  # noqa: E402
import pbl_live  # noqa: E402
from pbl_live import router as pbl_router  # noqa: E402

ROOT = SRC_DIR.parent
MODEL_PATH = ROOT / "models" / "national_models.joblib"
STATION_CATALOG_PATH = SRC_DIR / "stations.csv"
INPUTS = ("PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "SO2", "CO", "Ozone", "AT", "RH", "WS", "WD", "SR")
REQUIRED = ("PM2.5", "PM10")
IST = timezone(timedelta(hours=5, minutes=30))

AIR_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo free tier: about 600 location-calls per minute, 5,000 per hour, 10,000 per day.
# Each batch below costs BATCH_SIZE x 2 calls, so the pause keeps the rate near 500 per minute.
# Stations in the same ~11 km cell are fetched once.
REFRESH_SECONDS = 3 * 3600
BATCH_SIZE = 50
BATCH_PAUSE = 13
CELL_DECIMALS = 1

# Open-Meteo gives wind in km/h. If your training data has WS in m/s, set this to 1 / 3.6.
# Check with: python -c "import pandas as pd; print(pd.read_csv('data/national_clean.csv', usecols=['WS']).WS.describe())"
WS_SCALE = 1.0

app = FastAPI(title="National Air Forecast API", version="2.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["GET"],
    allow_headers=["*"],
)
app.include_router(pbl_router)   # GET /pbl/forecast (see pbl_live.py)

# ---------------------------------------------------------------- model bundle (reloaded if the file changes)
_bundle_cache: dict[str, object] = {"mtime": None, "bundle": None}
_bundle_lock = threading.Lock()


def _bundle() -> dict:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="National model bundle is unavailable")
    mtime = MODEL_PATH.stat().st_mtime
    with _bundle_lock:
        if _bundle_cache["mtime"] != mtime:
            _bundle_cache.update(bundle=joblib.load(MODEL_PATH), mtime=mtime)
        return _bundle_cache["bundle"]  # type: ignore[return-value]


def _value(value: Optional[float]) -> float:
    return np.nan if value is None else float(value)


def _iso(value: object) -> str:
    return pd.Timestamp(value).isoformat()


def _text(value: object) -> str:
    """Station metadata can contain NaN, which is not valid JSON, so turn it into an empty string."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


def _to_ist_naive(ts: datetime) -> datetime:
    """The model was trained on IST clock times, so convert any timezone-aware input to naive IST."""
    if ts.tzinfo is not None:
        ts = ts.astimezone(IST).replace(tzinfo=None)
    return ts


# ---------------------------------------------------------------- forecasting
def _forecast_payload(values: dict[str, Optional[float]], timestamp: datetime) -> dict[str, object]:
    bundle = _bundle()
    stamp = _to_ist_naive(timestamp)
    frame = pd.DataFrame([{name: _value(values.get(name)) for name in INPUTS}])
    frame["region"] = None
    ts = pd.DatetimeIndex([pd.Timestamp(stamp)])
    try:
        predictions = forecast(bundle, frame, ts)
        aqi_now = float(
            build_features(frame, ts, loc=bundle["loc"], region_levels=bundle["region_levels"])["AQI_now"].iloc[0]
        )
    except (KeyError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"Model input could not be evaluated: {exc}") from exc

    result: dict[str, list[dict[str, object]]] = {}
    for target in TARGETS:
        now = aqi_now if target == "AQI" else float(frame[target].iloc[0])
        points = [{"horizon_h": 0, "valid_at": _iso(stamp), "predicted": now if np.isfinite(now) else None}]
        for horizon in HORIZONS:
            predicted = float(predictions[(target, horizon)][0])
            points.append({
                "horizon_h": horizon,
                "valid_at": _iso(pd.Timestamp(stamp) + timedelta(hours=horizon)),
                "predicted": predicted if np.isfinite(predicted) else None,
            })
        result[target] = points

    return {
        "model": "national_models.joblib",
        "model_version": bundle.get("trained_through", "unknown"),
        "targets": result,
        "horizons_h": HORIZONS,
        "units": {"AQI": "CPCB AQI", "PM2.5": "µg/m³", "PM10": "µg/m³", "Ozone": "µg/m³", "NO2": "µg/m³", "CO": "mg/m³"},
        "issued_at": _iso(stamp),
    }


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model": MODEL_PATH.name,
        "available": MODEL_PATH.exists(),
        "pbl_model_available": pbl_live.MODEL_PATH.exists(),
        "stations": len(_stations),
        "live_points": len(_live["points"]),  # type: ignore[arg-type]
        "live_complete": _live["complete"],
        "live_updated_at": _live["updatedAt"],
    }


@app.get("/forecast")
def get_forecast(
    timestamp: datetime = Query(...),
    pm25: Optional[float] = Query(None),
    pm10: Optional[float] = Query(None),
    no: Optional[float] = Query(None),
    no2: Optional[float] = Query(None),
    nox: Optional[float] = Query(None),
    nh3: Optional[float] = Query(None),
    so2: Optional[float] = Query(None),
    co: Optional[float] = Query(None),
    ozone: Optional[float] = Query(None),
    at: Optional[float] = Query(None),
    rh: Optional[float] = Query(None),
    ws: Optional[float] = Query(None),
    wd: Optional[float] = Query(None),
    sr: Optional[float] = Query(None),
) -> dict[str, object]:
    values = {
        "PM2.5": pm25, "PM10": pm10, "NO": no, "NO2": no2, "NOx": nox, "NH3": nh3,
        "SO2": so2, "CO": co, "Ozone": ozone, "AT": at, "RH": rh, "WS": ws, "WD": wd, "SR": sr,
    }
    missing = [name for name in REQUIRED if values[name] is None]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required readings: {', '.join(missing)}")
    return _forecast_payload(values, timestamp)


# ---------------------------------------------------------------- stations and live snapshot
_stations: list[dict[str, object]] = []
_live: dict[str, object] = {"points": [], "inputs": {}, "updatedAt": None, "complete": False}
_live_lock = threading.Lock()
_first_attempt_done = threading.Event()


def _http_json(url: str, params: dict[str, object]) -> object:
    query = urllib.parse.urlencode(params, safe=",")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(f"{url}?{query}", timeout=40) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 2:
                raise
            print("[live] Open-Meteo rate limit hit, waiting 65 s", flush=True)
            time.sleep(65)
    raise RuntimeError("unreachable")


def _as_list(payload: object) -> list[dict]:
    return payload if isinstance(payload, list) else [payload]  # type: ignore[list-item]


def _int(value: Optional[float]) -> Optional[int]:
    """Missing stays missing (None). It is never shown as a reading of 0."""
    return int(round(value)) if value is not None else None


def _round(value: Optional[float], digits: int) -> Optional[float]:
    return round(value, digits) if value is not None else None


def _cell(s: dict[str, object]) -> tuple[float, float]:
    return (round(float(s["latitude"]), CELL_DECIMALS), round(float(s["longitude"]), CELL_DECIMALS))  # type: ignore[arg-type]


def _publish(fetched: dict[tuple[float, float], tuple[dict, dict, Optional[float]]], complete: bool) -> int:
    points: list[dict[str, object]] = []
    inputs: dict[int, dict[str, Optional[float]]] = {}
    for i, s in enumerate(_stations):
        got = fetched.get(_cell(s))
        if not got:
            continue
        ac, wc, pbl = got
        if ac.get("pm2_5") is None or ac.get("pm10") is None:
            continue
        co_ug = ac.get("carbon_monoxide")
        ws_raw = wc.get("wind_speed_10m")
        points.append({
            "id": i, "name": _text(s["station"]), "city": _text(s.get("city")),
            "state": _text(s.get("state")), "region": _text(s.get("region")),
            "latitude": float(s["latitude"]), "longitude": float(s["longitude"]),  # type: ignore[arg-type]
            "source": "Open-Meteo model grid estimate",
            "pm25": _int(ac.get("pm2_5")), "pm10": _int(ac.get("pm10")),
            "no": _int(ac.get("nitrogen_monoxide")), "no2": _int(ac.get("nitrogen_dioxide")), "nox": None,
            "nh3": _int(ac.get("ammonia")), "so2": _int(ac.get("sulphur_dioxide")),
            "co": _round(co_ug, 1), "ozone": _int(ac.get("ozone")), "aqi": _int(ac.get("us_aqi")),
            "temperature": _round(wc.get("temperature_2m"), 1),
            "humidity": _int(wc.get("relative_humidity_2m")),
            "windSpeed": _round(ws_raw, 1), "windDirection": _int(wc.get("wind_direction_10m")),
            "precipitation": _round(wc.get("precipitation"), 1),
            "boundaryLayerHeight": None if pbl is None else _int(pbl),
        })
        inputs[i] = {
            "PM2.5": ac.get("pm2_5"), "PM10": ac.get("pm10"), "NO": ac.get("nitrogen_monoxide"),
            "NO2": ac.get("nitrogen_dioxide"), "NOx": None, "NH3": ac.get("ammonia"),
            "SO2": ac.get("sulphur_dioxide"),
            "CO": None if co_ug is None else co_ug / 1000.0,  # Open-Meteo ug/m3 -> model mg/m3
            "Ozone": ac.get("ozone"), "AT": wc.get("temperature_2m"), "RH": wc.get("relative_humidity_2m"),
            "WS": None if ws_raw is None else ws_raw * WS_SCALE,
            "WD": wc.get("wind_direction_10m"), "SR": wc.get("shortwave_radiation"),
        }
    if points:
        with _live_lock:
            _live.update(points=points, inputs=inputs, updatedAt=datetime.now(IST).strftime("%H:%M:%S"), complete=complete)
        _first_attempt_done.set()
    return len(points)


def _refresh_live() -> None:
    cells: dict[tuple[float, float], list[int]] = {}
    for i, s in enumerate(_stations):
        cells.setdefault(_cell(s), []).append(i)
    keys = list(cells)

    fetched: dict[tuple[float, float], tuple[dict, dict, Optional[float]]] = {}
    for start in range(0, len(keys), BATCH_SIZE):
        batch = keys[start:start + BATCH_SIZE]
        loc = {
            "latitude": ",".join(str(k[0]) for k in batch),
            "longitude": ",".join(str(k[1]) for k in batch),
            "timezone": "Asia/Kolkata",
        }
        try:
            air = _as_list(_http_json(AIR_URL, {**loc, "current": (
                "pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,nitrogen_monoxide,ammonia,"
                "sulphur_dioxide,ozone,us_aqi")}))
            wx = _as_list(_http_json(WEATHER_URL, {**loc, "current": (
                "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,"
                "precipitation,shortwave_radiation"),
                "hourly": "boundary_layer_height", "forecast_hours": 1}))
        except Exception as exc:  # keep going; stations in this batch stay missing until the next refresh
            print(f"[live] batch starting at {start} failed: {exc}", flush=True)
            time.sleep(BATCH_PAUSE)
            continue
        for key, a, w in zip(batch, air, wx):
            pbl_list = (w.get("hourly") or {}).get("boundary_layer_height") or [None]
            fetched[key] = (a.get("current") or {}, w.get("current") or {}, pbl_list[0])
        if not _live["complete"]:  # first run only: show stations on the map as soon as they arrive
            _publish(fetched, complete=False)
        time.sleep(BATCH_PAUSE)

    count = _publish(fetched, complete=True)
    print(f"[live] {count}/{len(_stations)} stations updated from {len(keys)} grid cells", flush=True)


def _refresher() -> None:
    while True:
        try:
            _refresh_live()
        except Exception as exc:
            print(f"[live] refresh failed: {exc}", flush=True)
        finally:
            _first_attempt_done.set()
        time.sleep(REFRESH_SECONDS)


@app.on_event("startup")
def _startup() -> None:
    if not MODEL_PATH.exists():
        _first_attempt_done.set()
        return
    # The trained bundle contains only stations retained for model training.
    # The map should show the complete catalog, including all Delhi stations
    # and stations that were excluded from training for data-quality reasons.
    if STATION_CATALOG_PATH.exists():
        stn = pd.read_csv(STATION_CATALOG_PATH)
    else:
        stn = _bundle()["stations"]
    stn = stn.dropna(subset=["latitude", "longitude"])
    stn = stn.drop_duplicates(["station", "latitude", "longitude"]).reset_index(drop=True)
    _stations.extend(stn.to_dict("records"))
    source = "station catalog" if STATION_CATALOG_PATH.exists() else "model bundle"
    print(f"[live] {len(_stations)} stations loaded from the {source}", flush=True)
    threading.Thread(target=_refresher, daemon=True).start()


@app.get("/stations")
def list_stations() -> list[dict[str, object]]:
    return [{
        "id": i, "station": _text(s["station"]), "city": _text(s.get("city")), "state": _text(s.get("state")),
        "region": _text(s.get("region")), "latitude": float(s["latitude"]), "longitude": float(s["longitude"]),  # type: ignore[arg-type]
    } for i, s in enumerate(_stations)]


@app.get("/stations/live")
def stations_live() -> dict[str, object]:
    _first_attempt_done.wait(timeout=180)
    with _live_lock:
        points = _live["points"]
        updated = _live["updatedAt"]
        complete = _live["complete"]
    if not points:
        raise HTTPException(status_code=503, detail="Live station snapshot is not available yet")
    return {"points": points, "updatedAt": updated, "count": len(points), "complete": complete}  # type: ignore[arg-type]


@app.get("/stations/forecast")
def station_forecast(station_id: int = Query(..., alias="id")) -> dict[str, object]:
    with _live_lock:
        values = _live["inputs"].get(station_id)  # type: ignore[union-attr]
    if values is None:
        raise HTTPException(status_code=404, detail="No live readings for this station")
    return _forecast_payload(values, datetime.now(IST))