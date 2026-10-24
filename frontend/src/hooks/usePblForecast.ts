// Place at: src/hooks/usePblForecast.ts
// Fetches the trained PBL model's 72 h forecast from the API (GET /pbl/forecast). No fallback values are invented.
import { useEffect, useState } from 'react';

export type PblPoint = { hour: number; valid_at: string; observed: boolean; blh_m: number; ws10_ms: number; wd10_deg: number; ventilation_m2s: number; stability_k: number; inversion_index: number; t2m_c: number; skt_c: number };
export type PblEvent = { kind: 'lowest' | 'strongest_inversion' | 'recovery' | 'peak' | 'closing'; hour: number; valid_at: string; blh_m?: number; inversion_index?: number };
export type PblRecovery = { state: 'recovers' | 'open' | 'none_in_72h'; hour?: number; valid_at?: string; next_close_hour?: number | null; next_close_at?: string | null };
export type PblData = {
  issued_at: string; threshold_m: number; stale?: boolean;
  now: { blh_m: number; ws10_ms: number; wd10_deg: number; ventilation_m2s: number; stability_k: number; inversion_index: number; inversion_active: boolean };
  summary: { lid_vs_tomorrow_m: number; peak: { hour: number; valid_at: string; blh_m: number; ventilation_m2s: number }; lowest: { hour: number; valid_at: string; blh_m: number }; recovery: PblRecovery };
  events: PblEvent[]; series: PblPoint[];
  model: { file: string; updated_at: string; horizon_h: number; trained_through?: string };
  source: { inputs: string; latitude: number; longitude: number; history_h: number };
  validation: Record<string, number | string | null>[] | null;
  diagnostics: { unit_check: { checked: boolean; mismatches: { column: string }[]; note?: string }; skin_temperature_source?: string; location_matches_dashboard?: boolean; inversion_proxy?: { flat_zero_all_72h?: boolean; share_night?: number; share_afternoon?: number } };
};

const API = ((import.meta as unknown as { env?: Record<string, string | undefined> }).env?.VITE_MODEL_API_URL ?? 'http://127.0.0.1:8000').replace(/\/$/, '');

export function usePblForecast() {
  const [data, setData] = useState<PblData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      const controller = new AbortController();
      const timeout = window.setTimeout(() => controller.abort(), 30_000);   // the API fetches 5 days of history; do not hang forever
      try {
        const response = await fetch(`${API}/pbl/forecast`, { signal: controller.signal });
        if (!response.ok) {
          let detail = `HTTP ${response.status}`;
          try { const body = await response.json(); if (typeof body?.detail === 'string') detail = body.detail; } catch { /* keep status text */ }
          throw new Error(detail);
        }
        const payload = await response.json() as PblData;
        if (!cancelled) { setData(payload); setError(null); }
      } catch (e) {
        if (!cancelled) setError(e instanceof Error ? e.message : 'PBL model API unavailable');
      } finally {
        window.clearTimeout(timeout);
        if (!cancelled) setLoading(false);
      }
    };
    load();
    const timer = window.setInterval(load, 15 * 60 * 1000);
    return () => { cancelled = true; window.clearInterval(timer); };
  }, []);

  return { data, error, loading };
}