import type { LucideIcon } from 'lucide-react';
import { Activity, CloudFog, Command, Gauge, Layers3, ShieldAlert, Zap } from 'lucide-react';

export type PageKey = 'overview' | 'forecast' | 'inversion' | 'smoke' | 'ozone' | 'ensemble' | 'alerts';
export type Notify = (message: string) => void;
export type ForecastTone = 'critical' | 'high' | 'elevated';

export type ForecastItem = {
  day: string;
  time: string;
  aqi: number;
  status: string;
  pm25: number;
  pm10: number;
  ozone: number;
  no2: number;
  wind: string;
  temp: string;
  tone: ForecastTone;
};

export type ModelForecastPoint = {
  horizon_h: number;
  valid_at: string;
  predicted: number | null;
};

export type ModelForecast = {
  targets: Record<string, ModelForecastPoint[]>;
  horizons_h: number[];
  units: Record<string, string>;
  issued_at: string;
  model: string;
};

// Hourly atmosphere context from now to +72 h. Pollutant forecast values come from ModelForecast.
export type AtmospherePoint = {
  hour: number; // hours from now, 0 to 72
  time: string; // local ISO time from the feed
  pbl: number | null; // boundary layer height, m
  inversionGradient: number | null; // largest layer gradient near the ground, °C/km (positive = inversion)
  inversionStrength: number | null; // 0 to 10, where 10 means 20 °C/km or more
  ventilation: number | null; // PBL height x surface wind, m²/s
};

export type LiveData = {
  current: {
    aqi: number;
    pm25: number;
    pm10: number;
    ozone: number;
    no2: number;
    temperature: number;
    windSpeed: number;
    windDirection: number;
    boundaryLayerHeight: number | null;
    inversionStrength: number | null;
    inversionGradient: number | null;
    ventilationCoefficient: number | null;
    aerosolSolarDimming: number | null;
  };
  forecast: ForecastItem[];
  atmosphere: AtmospherePoint[];
  modelForecast: ModelForecast | null;
  modelForecastError: string | null;
  pblModelUsed: boolean;
  updatedAt: string;
  source: string;
};

export type Page = { id: PageKey; label: string; icon: LucideIcon };

export const pages: Page[] = [
  { id: 'overview', label: 'Command center', icon: Command },
  { id: 'forecast', label: '72H forecast', icon: Activity },
  { id: 'inversion', label: 'Inversion & PBL', icon: Layers3 },
  { id: 'smoke', label: 'Smoke dispersion', icon: CloudFog },
  { id: 'ozone', label: 'Photochemistry', icon: Zap },
  { id: 'ensemble', label: 'Model ensemble', icon: Gauge },
  { id: 'alerts', label: 'Alerts & actions', icon: ShieldAlert },
];

export function getAqiStatus(aqi: number): { label: string; tone: ForecastTone } {
  if (aqi >= 301) return { label: 'SEVERE', tone: 'critical' };
  if (aqi >= 201) return { label: 'VERY HIGH', tone: 'high' };
  if (aqi >= 101) return { label: 'POOR', tone: 'elevated' };
  return { label: 'MODERATE', tone: 'elevated' };
}