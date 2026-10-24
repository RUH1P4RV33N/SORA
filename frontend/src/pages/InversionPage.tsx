// Place at: src/pages/InversionPage.tsx
// Every number on this page comes from GET /pbl/forecast (trained LightGBM model fed with live Open-Meteo inputs).
// If the API is down or the model is missing, the page says so instead of drawing numbers.
import { Activity, Clock, CloudSun, Layers3, TrendingUp, Wind } from 'lucide-react';
import { MetricCard, PageShell, PanelTitle } from '../Components/dashboard/Shared';
import { InteractiveChart, type ChartSeries } from '../Components/dashboard/InteractiveChart';
import { usePblForecast, type PblEvent } from '../hooks/usePblForecast';
import type { LiveData } from '../data/dashboard';

const TZ = 'Asia/Kolkata';
const fmtTime = new Intl.DateTimeFormat('en-GB', { hour: '2-digit', minute: '2-digit', hour12: false, timeZone: TZ });
const fmtDay = new Intl.DateTimeFormat('en-GB', { weekday: 'short', timeZone: TZ });
const fmtDate = new Intl.DateTimeFormat('en-GB', { day: '2-digit', month: 'short', year: 'numeric', timeZone: TZ });
const fmtKey = new Intl.DateTimeFormat('en-CA', { timeZone: TZ });
function clock(iso: string) {
  const parsed = new Date(iso);
  const d = Number.isNaN(parsed.getTime()) ? new Date() : parsed;
  const time = fmtTime.format(d);
  const same = fmtKey.format(d) === fmtKey.format(new Date());
  return { time, label: same ? time : `${fmtDay.format(d)} ${time}`, full: `${fmtDay.format(d)} ${fmtDate.format(d)} ${time} IST` };
}
function compassDirection(degrees: number) {
  const directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];
  return directions[Math.round(((degrees % 360) + 360) % 360 / 22.5) % directions.length];
}
const EVENT_META: Record<PblEvent['kind'], { title: string; colour: string; sub: (e: PblEvent, thr: number) => string }> = {
  lowest: { title: 'Lid at its lowest', colour: '#fb7b91', sub: (e) => `Mixing height ${e.blh_m} m` },
  strongest_inversion: { title: 'Strongest inversion', colour: '#fb7b91', sub: (e) => `Index ${e.inversion_index}/10` },
  recovery: { title: 'PBL recovery', colour: '#f2be67', sub: (_e, t) => `Mixing reaches ${t} m` },
  peak: { title: 'Peak mixing', colour: '#7de3a7', sub: (e) => `Mixing height ${e.blh_m} m` },
  closing: { title: 'Lid closing', colour: '#f2be67', sub: (_e, t) => `Mixing drops below ${t} m` },
};

// `liveData` is no longer needed here; the prop is kept so existing routing code does not change.
export function InversionPage(_props: { liveData: LiveData | null }) {
  const { data, error, loading } = usePblForecast();

  if (!data) {
    return <PageShell title="Inversion & PBL" eyebrow="ATMOSPHERIC DYNAMICS">
      <div className="panel empty-state">{loading ? 'Waiting for the PBL model forecast…' : `PBL model forecast unavailable: ${error ?? 'unknown error'}`}</div>
    </PageShell>;
  }

  const { now, summary, series, events, validation } = data;
  const thr = data.threshold_m;
  const rec = summary.recovery;
  const lidClosed = now.blh_m < thr;
  const all = validation?.find((r) => r.horizon === 'all');
  const mismatches = data.diagnostics.unit_check.mismatches;

  let recoveryValue: string, recoveryUnit: string, recoveryDetail: string;
  if (rec.state === 'open') {
    recoveryValue = 'Open';
    recoveryUnit = '';
    recoveryDetail = rec.next_close_hour != null && rec.next_close_at ? `PBL ≥ ${thr} m now · closes ${clock(rec.next_close_at).label} IST` : `PBL ≥ ${thr} m through the forecast`;
  } else if (rec.state === 'recovers' && rec.valid_at) {
    recoveryValue = clock(rec.valid_at).label; recoveryUnit = 'IST'; recoveryDetail = `In ${rec.hour} h · PBL reaches ${thr} m`;
  } else { recoveryValue = 'Not in 72h'; recoveryUnit = ''; recoveryDetail = `PBL stays below ${thr} m`; }

  const lastHour = series[series.length - 1]?.hour ?? 0;
  const pblChart: ChartSeries[] = [
    { name: 'Boundary layer height (trained PBL model)', colour: '#f2be67', unit: 'm', points: series.map((p) => ({ x: p.hour, y: p.blh_m, label: clock(p.valid_at).full })) },
    { name: `Open-mixing threshold (${thr} m)`, colour: '#fb7b91', dashed: true, unit: 'm', points: [{ x: 0, y: thr, label: 'Threshold' }, { x: lastHour, y: thr, label: 'Threshold' }] },
  ];
  const invChart: ChartSeries[] = [
    { name: 'Surface stability index', colour: '#fb7b91', unit: '/10', points: series.map((p) => ({ x: p.hour, y: p.inversion_index, label: clock(p.valid_at).full })) },
  ];

  return <PageShell title="Inversion & PBL" eyebrow="ATMOSPHERIC DYNAMICS">
    <div className="page-intro">
      <div><h1>When the lid <em>closes</em></h1><p>Boundary-layer height in metres above ground, near-surface stability and ventilation for the next {data.model.horizon_h} hours from the trained PBL model. Low nighttime heights are expected during a stable inversion. Issued {clock(data.issued_at).full}.</p></div>
      <div className="status-pill"><span className="live-dot" /> {now.inversion_active ? `INVERSION ACTIVE · ${now.inversion_index}/10` : 'NO SURFACE INVERSION'}{lidClosed ? ` · LID ${Math.round(now.blh_m)} m` : ''}</div>
    </div>
    
    {mismatches.length > 0 && <div className="panel empty-state">Input check: live values are on a different scale from the training data. Treat this forecast as unreliable.</div>}
    {data.stale && <div className="panel empty-state">Showing the last successful forecast: the latest refresh failed{error ? ` (${error})` : ''}.</div>}

    <div className="metrics-grid">
      <MetricCard label="Boundary layer · inversion cap" value={String(Math.round(now.blh_m))} unit="m AGL" detail={`${summary.lid_vs_tomorrow_m > 0 ? '+' : ''}${summary.lid_vs_tomorrow_m} m at the same hour tomorrow (forecast)`} accent="#fb7b91" icon={Layers3} />
      <MetricCard label="Surface stability · T2m − skin temp" value={`${now.stability_k > 0 ? '+' : ''}${now.stability_k}`} unit="K" detail={now.stability_k > 0 ? 'Skin colder than air: near-surface inversion' : 'Skin warmer than air: convective mixing'} accent="#f2be67" icon={CloudSun} />
      <MetricCard label="PBL recovery" value={recoveryValue} unit={recoveryUnit} detail={recoveryDetail} accent="#31d6e5" icon={Clock} />
      <MetricCard label="Peak mixing height · next 24h" value={String(summary.peak.blh_m)} unit="m" detail={`Expected ${clock(summary.peak.valid_at).label} IST`} accent="#7de3a7" icon={TrendingUp} />
      <MetricCard label="Ventilation" value={now.ventilation_m2s.toLocaleString('en-IN')} unit="m²/s" detail="PBL height × 10 m wind speed" accent="#31d6e5" icon={Activity} />
      <MetricCard label="Wind direction" value={`${Math.round(now.wd10_deg)}° · ${compassDirection(now.wd10_deg)}`} detail={`10 m wind from the ${compassDirection(now.wd10_deg)} · ${now.ws10_ms.toFixed(1)} m/s`} accent="#c7a9ff" icon={Wind} />
    </div>

    <div className="two-col">
      <div className="panel chart-large">
        <PanelTitle kicker={`${data.model.horizon_h}H PBL MODEL FORECAST`} title="Boundary layer height" />
        <InteractiveChart series={pblChart} xTitle="Hours from now (forecast hour, IST)" yTitle="Boundary layer height (m AGL)" ariaLabel="Boundary layer height by forecast hour" />
        <div className="chart-legend">{pblChart.map((s) => <span key={s.name}><i style={{ background: s.colour }} />{s.name}</span>)}</div>
        <small className="chart-caption">X: hours from now · Y: boundary layer height (m above ground). Hour 0 is the current hour (live input blended with the first forecast step); hours 1+ are model forecasts. {all ? `Held-out test on the training data (not on live inputs), 1–72 h: MAE ${all.BLH_MAE_m} m, R² ${all.BLH_R2}.` : 'Held-out accuracy metrics are not stored in this model file.'}</small>
      </div>

      <div className="panel">
        <PanelTitle kicker="DIURNAL CYCLE" title="The next 24 hours" />
        {events.length ? events.map((e) => { const m = EVENT_META[e.kind]; return (
          <div key={`${e.kind}-${e.hour}`} style={{ display: 'flex', alignItems: 'center', gap: 14, padding: '12px 0', borderTop: '1px solid #1b2a36' }}>
            <span style={{ fontFamily: 'monospace', fontSize: 12, color: '#9db5c0', minWidth: 70 }}>{clock(e.valid_at).label}</span>
            <i style={{ width: 8, height: 8, borderRadius: 4, background: m.colour, display: 'inline-block' }} />
            <div><b style={{ display: 'block', fontSize: 13 }}>{m.title}</b><small style={{ color: '#9db5c0' }}>{m.sub(e, thr)}</small></div>
          </div>); }) : <div className="empty-state">No transitions predicted in the next 24 h.</div>}
        <small className="chart-caption">Times (IST) are read from the model forecast: lowest and peak mixing height, strongest stability, and when mixing crosses {thr} m.</small>
      </div>
    </div>

    <div className="panel chart-large">
      <PanelTitle kicker={`${data.model.horizon_h}H DERIVED FORECAST`} title="Surface stability (inversion) index" />
      <InteractiveChart series={invChart} yMax={10} xTitle="Hours from now (forecast hour, IST)" yTitle="Surface stability index (0–10)" ariaLabel="Surface stability index by forecast hour" />
      <small className="chart-caption">X: hours from now · Y: 0 = skin temperature at or above air temperature (no surface inversion), 10 = stability at the 95th percentile of the training history. 0/10 is a real state: it means the ground is not colder than the air (normal every afternoon). Expect it to rise above 0 on most nights. Derived from the model&apos;s predicted 2 m minus skin temperature; it is a surface-stability index, not a measured lapse rate.</small>
    </div>
    <small className="chart-caption">Inputs: {data.source.inputs} at {data.source.latitude.toFixed(2)}°N, {data.source.longitude.toFixed(2)}°E · {data.source.history_h} h of history · model {data.model.file} (trained {data.model.updated_at}, data through {data.model.trained_through ?? 'unknown'}).</small>
  </PageShell>;
}