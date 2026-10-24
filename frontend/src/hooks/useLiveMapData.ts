import { useEffect, useState } from 'react';

export type MapPoint = {
  id: number;
  name: string;
  city: string;
  state: string;
  region: string;
  latitude: number;
  longitude: number;
  pm25: number;
  pm10: number;
  no: number;
  no2: number;
  nox: number | null;
  nh3: number;
  so2: number;
  co: number;
  ozone: number;
  temperature: number;
  humidity: number;
  aqi: number;
  windSpeed: number;
  windDirection: number;
  precipitation: number;
  boundaryLayerHeight: number | null;
  source: string;
};

export type FireHotspot = {
  latitude: number;
  longitude: number;
  brightness: number;
  satellite: string;
  acquisitionDate: string | null;
};

export type LiveMapData = {
  points: MapPoint[];
  boundaryLayerHeight: number | null;
  fires: FireHotspot[];
  fireFeedAvailable: boolean;
  fireUpdatedAt: string | null;
  updatedAt: string;
};

export const modelApiUrl = ((import.meta.env.VITE_MODEL_API_URL as string | undefined) ?? 'http://127.0.0.1:8000').replace(/\/$/, '');

// All of India: west, south, east, north
const INDIA_BBOX = '68,6,98,37';

function parseFireCsv(csv: string): FireHotspot[] {
  const [header, ...rows] = csv.trim().split(/\r?\n/);
  if (!header) return [];
  const columns = header.split(',');
  const latitudeIndex = columns.indexOf('latitude');
  const longitudeIndex = columns.indexOf('longitude');
  const brightnessIndex = columns.indexOf('bright_ti4') >= 0 ? columns.indexOf('bright_ti4') : columns.indexOf('brightness');
  const satelliteIndex = columns.indexOf('satellite');
  const dateIndex = columns.indexOf('acq_date');
  if (latitudeIndex < 0 || longitudeIndex < 0) return [];

  return rows.map((row) => row.split(',')).flatMap((values) => {
    const latitude = Number(values[latitudeIndex]);
    const longitude = Number(values[longitudeIndex]);
    if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) return [];
    return [{
      latitude,
      longitude,
      brightness: Number(values[brightnessIndex]) || 0,
      satellite: values[satelliteIndex] || 'NASA FIRMS',
      acquisitionDate: dateIndex >= 0 ? values[dateIndex] || null : null,
    }];
  });
}

export function useLiveMapData() {
  const [data, setData] = useState<LiveMapData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    let retryTimer: number | undefined;
    const load = async () => {
      try {
        const response = await fetch(`${modelApiUrl}/stations/live`);
        if (!response.ok) throw new Error(`API returned HTTP ${response.status} for /stations/live. Restart uvicorn with the new api.py.`);
        const payload = await response.json() as { points: MapPoint[]; updatedAt: string; complete?: boolean };
        const points = payload.points;
        const pbls = points.map((point) => point.boundaryLayerHeight).filter((value): value is number => value != null);
        const boundaryLayerHeight = pbls.length ? Math.round(pbls.reduce((sum, value) => sum + value, 0) / pbls.length) : null;

        let fires: FireHotspot[] = [];
        let fireFeedAvailable = false;
        let fireUpdatedAt: string | null = null;
        const fireKey = import.meta.env.VITE_NASA_FIRMS_MAP_KEY;
        if (fireKey) {
          try {
            const fireResponses = await Promise.all([
              fetch(`https://firms.modaps.eosdis.nasa.gov/api/area/csv/${fireKey}/MODIS_NRT/${INDIA_BBOX}/1`),
              fetch(`https://firms.modaps.eosdis.nasa.gov/api/area/csv/${fireKey}/VIIRS_NOAA20_NRT/${INDIA_BBOX}/1`),
            ]);
            const validFireResponses = fireResponses.filter((r) => r.ok);
            fireFeedAvailable = validFireResponses.length > 0;
            const fireRows = await Promise.all(validFireResponses.map((r) => r.text()));
            const unique = new Map<string, FireHotspot>();
            fireRows.flatMap(parseFireCsv).forEach((fire) => {
              unique.set(`${fire.latitude.toFixed(4)}:${fire.longitude.toFixed(4)}:${fire.acquisitionDate ?? ''}`, fire);
            });
            fires = [...unique.values()];
            if (fireFeedAvailable) fireUpdatedAt = new Date().toISOString();
          } catch {
            fireFeedAvailable = false;
          }
        }
        if (!cancelled) {
          setData({
            points,
            boundaryLayerHeight,
            fires,
            fireFeedAvailable,
            fireUpdatedAt,
            updatedAt: payload.updatedAt,
          });
          setError(null);
          // The first snapshot arrives in pieces; poll quickly until the backend says it is complete.
          if (payload.complete === false) retryTimer = window.setTimeout(load, 15_000);
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof TypeError
            ? `Cannot reach ${modelApiUrl}. Is uvicorn running?`
            : loadError instanceof Error ? loadError.message : 'Live station feed unavailable');
        }
      }
    };
    load();
    const refresh = window.setInterval(load, 10 * 60 * 1000);
    return () => { cancelled = true; window.clearInterval(refresh); window.clearTimeout(retryTimer); };
  }, []);

  return { data, error };
}