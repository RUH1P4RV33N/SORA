import { useEffect, useState } from 'react';
import { getAqiStatus, type AtmospherePoint, type ForecastItem, type LiveData, type ModelForecast } from '../data/dashboard';

type ApiResponse = {
  elevation?: number;
  current?: Record<string, number | null>;
  hourly?: Record<string, Array<number | null>> & { time?: string[] };
};

// Keep identical to DASHBOARD_LAT / DASHBOARD_LON in src/pbl_live.py so the PBL model's location check compares the same point.
const LAT = 28.6364;
const LON = 77.2010;
const HOURS_AHEAD = 72;
// Pressure levels used for the inversion. Levels below ground level are skipped automatically.
const LEVELS = [975, 950, 925, 900];

async function fetchJson<T>(url: string, label: string, attempts = 3): Promise<T> {
  let lastError: unknown;
  for (let attempt = 0; attempt < attempts; attempt += 1) {
    const controller = new AbortController();
    const timeout = window.setTimeout(() => controller.abort(), 15_000);
    try {
      const response = await fetch(url, { signal: controller.signal });
      if (!response.ok) throw new Error(`${label} returned HTTP ${response.status}`);
      return await response.json() as T;
    } catch (error) {
      lastError = error;
      if (attempt < attempts - 1) await new Promise((resolve) => window.setTimeout(resolve, 800 * (attempt + 1)));
    } finally {
      window.clearTimeout(timeout);
    }
  }
  throw lastError instanceof Error ? lastError : new Error(`${label} unavailable`);
}

function formatForecastTime(timestamp: string, hoursAhead: number) {
  if (hoursAhead === 0) return { day: 'NOW', time: 'LIVE' };
  const date = new Date(timestamp);
  return {
    day: `+${hoursAhead}H`,
    time: date.toLocaleString('en-IN', { month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false, timeZone: 'Asia/Kolkata' }).toUpperCase(),
  };
}

// Temperature gradient in °C per km between the surface and the pressure levels above it.
// gradient = (T2 - T1) / (z2 - z1), using geopotential height for z. Positive means temperature
// rises with height, which is an inversion. Returns the largest layer gradient, or null if unknown.
function maxLayerGradient(
  hour: number,
  surfaceTemperature: number | null | undefined,
  surfaceElevation: number | undefined,
  profile: ApiResponse,
): number | null {
  if (surfaceTemperature == null || surfaceElevation == null) return null;
  const layers: Array<{ t: number; z: number }> = [{ t: surfaceTemperature, z: surfaceElevation + 2 }];
  LEVELS.forEach((level) => {
    const t = profile.hourly?.[`temperature_${level}hPa`]?.[hour];
    const z = profile.hourly?.[`geopotential_height_${level}hPa`]?.[hour];
    // Skip levels at or below ground level, where the value is extrapolated and not real air.
    if (t != null && z != null && z > surfaceElevation + 20) layers.push({ t, z });
  });
  layers.sort((a, b) => a.z - b.z);
  if (layers.length < 2) return null;
  let max = -Infinity;
  for (let i = 1; i < layers.length; i += 1) {
    const dzKm = (layers[i].z - layers[i - 1].z) / 1000;
    if (dzKm <= 0) continue;
    max = Math.max(max, (layers[i].t - layers[i - 1].t) / dzKm);
  }
  return Number.isFinite(max) ? Number(max.toFixed(1)) : null;
}

// 0 = no inversion. 10 = gradient of 20 °C/km or more.
function inversionStrengthFrom(gradient: number | null) {
  return gradient == null ? null : Number(Math.min(10, Math.max(0, gradient) / 2).toFixed(1));
}

// Throws when the current hour is missing instead of silently falling back to index 0 (midnight data).
function indexOfHour(times: string[] | undefined, isoHour: string, label: string) {
  const found = times?.indexOf(isoHour) ?? -1;
  if (found < 0) throw new Error(`${label}: current hour ${isoHour} not found in the feed`);
  return found;
}

export function useLiveData() {
  const [data, setData] = useState<LiveData | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        // forecast_days=4 so that +72 h is always covered, even late in the day
        const params = `latitude=${LAT}&longitude=${LON}&forecast_days=4&timezone=Asia%2FKolkata`;
        const profileVars = [
          ...LEVELS.map((level) => `temperature_${level}hPa`),
          ...LEVELS.map((level) => `geopotential_height_${level}hPa`),
        ].join(',');
        const [air, weather, profile] = await Promise.all([
          fetchJson<ApiResponse>(`https://air-quality-api.open-meteo.com/v1/air-quality?${params}&current=pm2_5,pm10,ozone,nitrogen_dioxide,nitrogen_monoxide,ammonia,sulphur_dioxide,carbon_monoxide,us_aqi,aerosol_optical_depth&hourly=pm2_5,pm10,ozone,nitrogen_dioxide,us_aqi,aerosol_optical_depth`, 'Open-Meteo air feed'),
          fetchJson<ApiResponse>(`https://api.open-meteo.com/v1/forecast?${params}&current=temperature_2m,wind_speed_10m,wind_direction_10m,relative_humidity_2m,shortwave_radiation&hourly=temperature_2m,wind_speed_10m,wind_direction_10m,boundary_layer_height`, 'Open-Meteo weather feed'),
          fetchJson<ApiResponse>(`https://api.open-meteo.com/v1/forecast?${params}&hourly=${profileVars}`, 'Open-Meteo profile feed'),
        ]);
        if (!air.current || !weather.current || !air.hourly || !weather.hourly || !profile.hourly) throw new Error('Incomplete live feed');

        // Hourly arrays start at 00:00 today, so find the index of the current hour in each feed.
        const nowIso = (weather.current as unknown as { time?: string }).time ?? '';
        const nowHour = `${nowIso.slice(0, 13)}:00`;
        const weatherBase = indexOfHour(weather.hourly.time, nowHour, 'Weather feed');
        const airBase = indexOfHour(air.hourly.time, nowHour, 'Air feed');
        const profileBase = indexOfHour(profile.hourly.time, nowHour, 'Profile feed');

        const windSpeed = weather.current.wind_speed_10m ?? 0;
        const surfaceElevation = weather.elevation ?? profile.elevation;

        // Every value here, including hour 0, is the Open-Meteo value for that hour. Nothing is overridden by hand.
        const atmosphere: AtmospherePoint[] = [];
        for (let hour = 0; hour <= HOURS_AHEAD; hour += 1) {
          const wi = weatherBase + hour;
          const pi = profileBase + hour;
          if (wi >= (weather.hourly.time?.length ?? 0)) break;
          // Boundary-layer height in metres AGL. It can legitimately fall to tens of metres in a nocturnal stable layer.
          const pblValue = weather.hourly.boundary_layer_height?.[wi] ?? null;
          const pbl = pblValue != null && Number.isFinite(pblValue) && pblValue >= 0 ? pblValue : null;
          const wind = weather.hourly.wind_speed_10m?.[wi] ?? null;
          const gradient = maxLayerGradient(pi, weather.hourly.temperature_2m?.[wi], surfaceElevation, profile);
          atmosphere.push({
            hour,
            time: weather.hourly.time?.[wi] ?? '',
            pbl: pbl == null ? null : Math.round(pbl),
            inversionGradient: gradient,
            inversionStrength: inversionStrengthFrom(gradient),
            // Ventilation coefficient = mixing height (m) × wind speed (m/s); the feed gives km/h.
            ventilation: pbl == null || wind == null || !Number.isFinite(wind) ? null : Math.round(pbl * (wind / 3.6)),
          });
        }

        // Cards use hour 0 of the same series, so the dashboard, the PBL tab and the charts always agree.
        const boundaryLayerHeight = atmosphere[0]?.pbl ?? null;
        const inversionGradient = atmosphere[0]?.inversionGradient
          ?? maxLayerGradient(profileBase, weather.current.temperature_2m, surfaceElevation, profile);
        const inversionStrength = inversionStrengthFrom(inversionGradient);   // always derived from the gradient shown
        const ventilationCoefficient = boundaryLayerHeight == null ? null : Math.round(boundaryLayerHeight * (windSpeed / 3.6));
        // Stored as AOD x 100 (the Overview card divides by 100 again to show AOD).
        const aerosolOpticalDepth = air.current.aerosol_optical_depth ?? air.hourly.aerosol_optical_depth?.[airBase] ?? null;
        const aerosolSolarDimming = aerosolOpticalDepth == null ? null : Math.round(aerosolOpticalDepth * 100);
        const current = {
          aqi: Math.round(air.current.us_aqi ?? 0), // provisional; replaced below when the model responds
          pm25: Math.round(air.current.pm2_5 ?? 0),
          pm10: Math.round(air.current.pm10 ?? 0),
          ozone: Math.round(air.current.ozone ?? 0),
          no2: Math.round(air.current.nitrogen_dioxide ?? 0),
          temperature: Math.round(weather.current.temperature_2m ?? 0),
          windSpeed: Number(windSpeed.toFixed(1)),
          windDirection: Math.round(weather.current.wind_direction_10m ?? 0),
          boundaryLayerHeight,
          inversionStrength,
          inversionGradient,
          ventilationCoefficient,
          aerosolSolarDimming,
        };

        const modelApi = import.meta.env.VITE_MODEL_API_URL ?? 'http://127.0.0.1:8000';
        const carbonMonoxide = air.current.carbon_monoxide;
        const modelParams = new URLSearchParams({ timestamp: new Date().toISOString() });
        const addModelParam = (name: string, value: number | null | undefined) => {
          if (value != null && Number.isFinite(value)) modelParams.set(name, String(value));
        };
        addModelParam('pm25', air.current.pm2_5);
        addModelParam('pm10', air.current.pm10);
        addModelParam('no', air.current.nitrogen_monoxide);
        addModelParam('no2', air.current.nitrogen_dioxide);
        addModelParam('nh3', air.current.ammonia);
        addModelParam('so2', air.current.sulphur_dioxide);
        // Open-Meteo gives CO in µg/m³; the model was trained on mg/m³.
        addModelParam('co', carbonMonoxide == null ? null : carbonMonoxide / 1000);
        addModelParam('ozone', air.current.ozone);
        addModelParam('at', weather.current.temperature_2m);
        addModelParam('rh', weather.current.relative_humidity_2m);
        addModelParam('ws', weather.current.wind_speed_10m);
        addModelParam('wd', weather.current.wind_direction_10m);
        addModelParam('sr', weather.current.shortwave_radiation);
        let modelForecast: ModelForecast | null = null;
        let modelForecastError: string | null = null;
        try {
          const modelResponse = await fetch(`${modelApi.replace(/\/$/, '')}/forecast?${modelParams}`);
          if (!modelResponse.ok) throw new Error(`National model returned ${modelResponse.status}`);
          modelForecast = await modelResponse.json() as ModelForecast;
        } catch (modelError) {
          modelForecastError = modelError instanceof Error ? modelError.message : 'National model unavailable';
        }

        // ---- Canonical AQI scale resolution ----
        // The national model returns CPCB AQI. Open-Meteo returns US AQI. These are NOT
        // the same number for the same air, so we never mix them:
        //   • model up   → every AQI value comes from `modelForecast.targets.AQI` (CPCB)
        //   • model down → every AQI value comes from Open-Meteo (US AQI, marked as fallback)
        // The single field `LiveData.current.aqi` follows the same rule so no page can
        // silently read a value in the "wrong" scale.
        const modelAqiByHorizon = new Map<number, number>();
        for (const p of modelForecast?.targets.AQI ?? []) {
          if (typeof p.predicted === 'number' && Number.isFinite(p.predicted)) {
            modelAqiByHorizon.set(p.horizon_h, p.predicted);
          }
        }
        const modelCurrentAqi = modelAqiByHorizon.get(0);
        const modelAvailable = modelCurrentAqi != null;
        const displayCurrent = modelAvailable
          ? { ...current, aqi: Math.round(modelCurrentAqi as number) }
          : current;

        // Every forecast[] entry follows the same scale as displayCurrent.aqi.
        // NOTE: when the model is up but has no value for a horizon (it covers +1..+24 h), the entry
        // repeats the current AQI. Charts should plot `modelForecast.targets`, not forecast[].aqi, beyond +24 h.
        const canonicalAqi = (offset: number): number => {
          const exact = modelAqiByHorizon.get(offset);
          return exact != null ? Math.round(exact) : displayCurrent.aqi;
        };

        const airLength = air.hourly.time?.length ?? 0;
        const liveForecast: ForecastItem[] = [0, 12, 24, 36, 48, 60, 72]
          .filter((offset) => airBase + offset < airLength)
          .map((offset) => {
            const ai = airBase + offset;
            const wi = weatherBase + offset;
            const aqi = canonicalAqi(offset);
            const status = getAqiStatus(aqi);
            const time = air.hourly?.time?.[ai] ?? new Date().toISOString();
            const formatted = formatForecastTime(time, offset);
            return {
              day: formatted.day,
              time: formatted.time,
              aqi,
              status: status.label,
              pm25: Math.round(air.hourly?.pm2_5?.[ai] ?? displayCurrent.pm25),
              pm10: Math.round(air.hourly?.pm10?.[ai] ?? displayCurrent.pm10),
              ozone: Math.round(air.hourly?.ozone?.[ai] ?? displayCurrent.ozone),
              no2: Math.round(air.hourly?.nitrogen_dioxide?.[ai] ?? displayCurrent.no2),
              wind: String(Number((weather.hourly?.wind_speed_10m?.[wi] ?? displayCurrent.windSpeed).toFixed(1))),
              temp: `${Math.round(weather.hourly?.temperature_2m?.[wi] ?? displayCurrent.temperature)}°`,
              tone: status.tone,
            };
          });

        if (!cancelled) {
          setData({
            current: displayCurrent,
            forecast: liveForecast,
            atmosphere,
            modelForecast,
            modelForecastError,
            // atmosphere[] is pure Open-Meteo. The trained PBL model is drawn separately on the
            // Inversion & PBL tab (usePblForecast), and only when it was trained for Delhi.
            pblModelUsed: false,
            updatedAt: new Date().toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }),
            source: modelForecast ? 'National model + live feeds' : 'Open-Meteo live feeds (US AQI fallback)',
          });
          setError(false);
        }
      } catch (loadError) {
        if (!cancelled) {
          console.warn('Live data refresh failed; retaining the last successful feed.', loadError);
          setError(true);
        }
      }
    };
    load();
    const refresh = window.setInterval(load, 15 * 60 * 1000);
    return () => { cancelled = true; window.clearInterval(refresh); };
  }, []);

  return { data, error };
}