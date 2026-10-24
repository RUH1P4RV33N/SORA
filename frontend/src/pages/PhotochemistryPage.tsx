// Place at: src/pages/PhotochemistryPage.tsx
import { useState, type CSSProperties } from 'react';
import { Activity, CloudFog, CloudSun, FlaskConical, Wind, Zap } from 'lucide-react';
import type { LiveData } from '../data/dashboard';
import { MetricCard, PageShell, PanelTitle } from '../Components/dashboard/Shared';
import { InteractiveChart, type ChartSeries } from '../Components/dashboard/InteractiveChart';

type Pollutant = 'AQI' | 'PM2.5' | 'O3';
const config: Record<Pollutant, { key: 'aqi' | 'pm25' | 'ozone'; label: string; unit: string; colour: string; model: 'AQI' | 'PM2.5' | 'Ozone' }> = {
  AQI: { key: 'aqi', label: 'AQI', unit: 'CPCB AQI', colour: '#fb4f77', model: 'AQI' },
  'PM2.5': { key: 'pm25', label: 'PM2.5', unit: 'μg/m3', colour: '#31d6e5', model: 'PM2.5' },
  'O3': { key: 'ozone', label: 'O3', unit: 'μg/m3', colour: '#c7a9ff', model: 'Ozone' },
};

const clamp = (v: number, lo = 0, hi = 1) => Math.min(hi, Math.max(lo, v));
const finite = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v);

function istHour() {
  const h = Number(new Intl.DateTimeFormat('en-GB', { hour: '2-digit', hour12: false, timeZone: 'Asia/Kolkata' }).format(new Date()));
  return h % 24;
}

function change(now: number | null | undefined, later: number | null | undefined) {
  if (!finite(now) || !finite(later) || now <= 0) return null;
  return Math.round(((later - now) / now) * 100);
}
function trendWord(pct: number | null) {
  if (pct == null) return 'trend unavailable';
  return pct > 5 ? 'rising' : pct < -5 ? 'falling' : 'steady';
}
const signed = (pct: number | null) => (pct == null ? '—' : `${pct > 0 ? '+' : ''}${pct}%`);

export function PhotochemistryPage({ liveData }: { liveData: LiveData | null }) {
  const [selected, setSelected] = useState<Pollutant>('AQI');
  const [donutHover, setDonutHover] = useState<number | null>(null);
  const current = liveData?.current;
  const cfg = config[selected];
  // `current.aqi` is CPCB when the national model is up, US AQI when it fell back.
  const currentAqiUnit = liveData?.modelForecast ? 'CPCB AQI' : 'US AQI (fallback)';

  const modelPoints = (key: 'AQI' | 'PM2.5' | 'Ozone' | 'NO2') =>
    (liveData?.modelForecast?.targets[key] ?? []).filter((p) => finite(p.predicted)).map((p) => ({ x: p.horizon_h, y: p.predicted as number, label: `+${p.horizon_h}h (national model)` }));
  const modelOzone = modelPoints('Ozone');
  const modelNo2 = modelPoints('NO2');
  const modelSelected = modelPoints(cfg.model);

  // ---- headline numbers ----
  const ozone = finite(current?.ozone) ? current!.ozone : null;
  const no2 = finite(current?.no2) ? current!.no2 : null;
  // `aerosolSolarDimming` is stored as AOD × 100 (an internal scaling artifact of the
  // Open-Meteo fetch). The physically meaningful quantity is AOD itself, dimensionless.
  const aod = finite(current?.aerosolSolarDimming) ? current!.aerosolSolarDimming! / 100 : null;
  const lastModel = modelOzone[modelOzone.length - 1];
  const ozoneLater = modelOzone.find((p) => p.x === 24)?.y ?? lastModel?.y;
  const ozoneWhen = modelOzone.find((p) => p.x === 24) ? 'in 24h' : `by +${lastModel?.x ?? 72}h`;
  const ozonePct = change(ozone, ozoneLater);
  const no2Pct = change(no2, modelNo2.find((p) => p.x === 24)?.y ?? modelNo2[modelNo2.length - 1]?.y);

  // ---- heuristics (NOT direct measurements) ----
  // `dimPct` is a 0–1 normalisation of AOD, used only inside the donut weighting.
  const dimPct = clamp(aod ?? 0);
  // `oxidative` is a normalised 0–1 index combining O3 level and (1 − AOD) as a proxy
  // for photochemical potential. It is NOT an OH concentration — the live feed has no
  // OH, HO2, or RO2 measurements. Use it as a relative indicator only.
  const oxidative = ozone == null && aod == null ? null : clamp(0.5 * clamp((ozone ?? 0) / 120) + 0.5 * (1 - dimPct));
  const oxLabel = oxidative == null ? 'Unavailable' : oxidative < 0.33 ? 'Low oxidative activity' : oxidative < 0.8 ? 'Moderate oxidative activity' : 'High oxidative activity';
  const ratio = ozone != null && no2 != null && no2 > 0 ? ozone / no2 : null;
  const regime = ratio == null ? ['—', 'UNAVAILABLE'] : ratio >= 1.5 ? ['NOx', 'LIMITED'] : ratio < 0.8 ? ['VOC', 'LIMITED'] : ['MIXED', 'REGIME'];
  const regimeColour = regime[0] === 'NOx' ? '#7de3a7' : regime[0] === 'VOC' ? '#f2be67' : '#31d6e5';

  // ---- species balance (indicative shares) ----
  // Heuristic driver weights, NOT a mass balance. See comments in the previous version
  // for the chemistry rationale; the formulas are unchanged.
  const hour = istHour();
  const daylight = clamp(Math.sin(Math.PI * (hour - 6) / 12), 0, 1);
  const ozoneLevel = clamp((ozone ?? 0) / 100);
  const no2Level = clamp((no2 ?? 0) / 60);
  const formationW = 0.25 + 0.75 * daylight * ozoneLevel * (1 - 0.8 * dimPct);
  const scavengingW = 0.15 + dimPct * (0.3 + 0.7 * daylight);
  const nightW = 0.15 + no2Level * (0.4 + 0.6 * (1 - daylight));

  const raw = [
    { name: 'Ozone formation', colour: '#c7a9ff', w: formationW },
    { name: 'Aerosol scavenging', colour: '#31d6e5', w: scavengingW },
    { name: 'Night chemistry', colour: '#f2be67', w: nightW },
  ];
  const total = raw.reduce((s, r) => s + r.w, 0) || 1;
  const parts = raw.map((r) => ({ ...r, pct: (r.w / total) * 100 }));
  const R = 60, C = 2 * Math.PI * R;
  let acc = 0;

  // ---- chart series ----
  const reactionSeries: ChartSeries[] = [
    { name: 'Ozone · national model', colour: '#c7a9ff', unit: 'μg/m3', points: modelOzone },
    { name: 'NO2 · national model', colour: '#fb4f77', dashed: true, unit: 'μg/m3', points: modelNo2 },
  ];
  const selectedSeries: ChartSeries[] = [
    { name: `${cfg.label} · national model`, colour: cfg.colour, unit: cfg.unit, points: modelSelected },
  ];

  const hovered = donutHover == null ? null : parts[donutHover];

  return <PageShell title="Photochemistry" eyebrow="CHEMICAL TRANSFORMATION">
    <div className="page-intro">
      <div><h1>The chemistry <em>of sunlight</em></h1><p>Understand where NOx, VOCs, aerosols, and sunlight drive ground-level ozone formation.</p></div>
      <div className="status-pill"><span className="live-dot" /> O3 REGIME · {regime[0]}{regime[1] ? `-${regime[1]}` : ''}</div>
    </div>

    <div className="metrics-grid">
      <MetricCard
        label="Surface ozone"
        value={String(ozone ?? '—')}
        unit="μg/m3"
        detail={`${signed(ozonePct)} ${ozoneWhen} · ${trendWord(ozonePct)}`}
        accent="#c7a9ff"
        icon={Zap}
      />
      <MetricCard
        label="NO2 column"
        value={String(no2 ?? '—')}
        unit="μg/m3"
        detail={`${signed(no2Pct)} by +24h · ${trendWord(no2Pct)}`}
        accent="#31d6e5"
        icon={Wind}
      />
      {/* Correction 1: AOD is not actinic flux, and the value is not in W/m². */}
      <MetricCard
        label="Aerosol optical depth"
        value={aod == null ? '—' : aod.toFixed(2)}
        unit="AOD"
        detail="Dimensionless CAMS column AOD · reduces sunlight to the surface · not actinic flux"
        accent="#f2be67"
        icon={CloudSun}
      />
      {/* Correction 2: normalised 0–1 index, not an OH concentration. */}
      <MetricCard
        label="Oxidative activity index"
        value={oxidative == null ? '—' : oxidative.toFixed(2)}
        unit="0–1 index"
        detail={oxidative == null ? 'Unavailable' : `${oxLabel} · model-derived from O₃ and AOD · not a direct OH measurement`}
        accent="#7de3a7"
        icon={FlaskConical}
      />
      {/* Correction 3: AQI is a composite index, not a mass. */}
      <MetricCard
        label="Current AQI"
        value={String(current?.aqi ?? '—')}
        unit={currentAqiUnit}
        detail={`Composite index · ${liveData?.source ?? 'live feed'} · computed from multiple pollutants`}
        accent="#fb4f77"
        icon={Activity}
      />
      <MetricCard
        label="Current PM2.5"
        value={String(current?.pm25 ?? '—')}
        unit="μg/m3"
        detail="Mass concentration · one input to the AQI above · same Delhi grid point & time"
        accent="#31d6e5"
        icon={CloudFog}
      />
    </div>

    <div className="two-col">
      <div className="panel chart-large">
        <PanelTitle kicker="REACTION NETWORK · 72 HOURS" title="Ozone production potential" />
        <InteractiveChart series={reactionSeries} xTitle="Forecast time (hours from now, IST)" yTitle="Concentration (μg/m3)" ariaLabel="Ozone and NO2 forecast over 72 hours" />
        <div className="chart-legend">{reactionSeries.map((s) => <span key={s.name}><i style={{ background: s.colour }} />{s.name}</span>)}</div>
        <div style={{ display: 'flex', justifyContent: 'space-around', gap: 8, flexWrap: 'wrap', marginTop: 10, fontSize: 12, color: '#9db5c0' }}>
          <span title="NOx is oxidised to NO2">NOx → NO2 · <b style={{ color: '#e6f1f5' }}>{no2 ?? '—'} μg/m3</b></span>
          <span title="No VOC measurement is available in the live feed">VOC → RO2 · <b style={{ color: '#e6f1f5' }}>not measured</b></span>
          <span title="Peroxy radicals + NO2 photolysis form ozone">RO2 → O3 · <b style={{ color: '#e6f1f5' }}>{ozone ?? '—'} μg/m3</b></span>
        </div>
        <small className="chart-caption">X: hours from now (IST checkpoints) · Y: concentration (μg/m3). Hover the chart for exact values.</small>
      </div>

      <div className="panel">
        <PanelTitle kicker="SPECIES BALANCE" title="Chemical regime" />
        <svg viewBox="0 0 200 200" role="img" aria-label="Chemical regime shares" style={{ width: '100%', maxWidth: 260, display: 'block', margin: '0 auto' }}>
          {parts.map((p, i) => {
            const len = (p.pct / 100) * C;
            const off = acc; acc += len;
            return <circle key={p.name} cx="100" cy="100" r={R} fill="none" stroke={p.colour} strokeWidth={donutHover === i ? 26 : 22}
              strokeDasharray={`${len} ${C - len}`} strokeDashoffset={-off} transform="rotate(-90 100 100)" style={{ cursor: 'pointer', transition: 'stroke-width .15s' }}
              onMouseEnter={() => setDonutHover(i)} onMouseLeave={() => setDonutHover(null)} onPointerDown={() => setDonutHover(i)} />;
          })}
          {hovered
            ? <><text x="100" y="98" textAnchor="middle" fill="#e6f1f5" fontSize="18" fontWeight="600">{hovered.pct.toFixed(0)}%</text><text x="100" y="116" textAnchor="middle" fill="#9db5c0" fontSize="9">{hovered.name}</text></>
            : <><text x="100" y="98" textAnchor="middle" fill={regimeColour} fontSize="16" fontWeight="600">{regime[0]}</text><text x="100" y="114" textAnchor="middle" fill="#9db5c0" fontSize="8">{regime[1]}</text></>}
        </svg>
        <div style={{ marginTop: 12 }}>{parts.map((p, i) => (
          <div key={p.name} onMouseEnter={() => setDonutHover(i)} onMouseLeave={() => setDonutHover(null)}
            style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '4px 0', fontSize: 12, color: '#9db5c0', opacity: donutHover == null || donutHover === i ? 1 : 0.5 }}>
            <i style={{ width: 8, height: 8, borderRadius: 4, background: p.colour, display: 'inline-block' }} /><span style={{ flex: 1 }}>{p.name}</span><b style={{ color: '#e6f1f5' }}>{p.pct.toFixed(0)}%</b>
          </div>))}
        </div>
        <small className="chart-caption">
          Driver weights from live O3, NO2, aerosol optical depth and solar elevation (IST) — not a mass balance. Photolysis is dormant at night, so ozone formation drops to a residual floor and NOx titration dominates. Regime from the O3/NO2 ratio{ratio == null ? '' : ` (${ratio.toFixed(2)})`}.
        </small>
      </div>
    </div>

    <div className="panel chart-large">
      <div className="panel-header"><PanelTitle kicker="LIVE POLLUTANT FORECAST" title={`${cfg.label} over the forecast window`} /><div className="filter-group">{(Object.keys(config) as Pollutant[]).map((p) => <button key={p} className={`filter-button ${selected === p ? 'active' : ''}`} onClick={() => setSelected(p)}>{p}</button>)}</div></div>
      <InteractiveChart series={selectedSeries} xTitle="Forecast time (hours from now, IST)" yTitle={`${cfg.label} (${cfg.unit})`} ariaLabel={`${cfg.label} forecast over 72 hours`} />
      <div className="chart-legend">{selectedSeries.map((s) => <span key={s.name}><i style={{ background: s.colour }} />{s.name}</span>)}</div>
      <small className="chart-caption">X: hours from now (IST checkpoints) · Y: {cfg.unit}. National model forecast only through +72h.</small>
      <div className="forecast-strip" style={{ marginTop: 14 }}>{modelSelected.map((point, index) => <div className={`forecast-card ${index === 0 ? 'selected' : ''}`} key={`${cfg.model}-${point.x}`} style={{ '--forecast-accent': cfg.colour } as CSSProperties}>
        <div className="forecast-top"><strong>{point.x === 0 ? 'NOW' : `+${point.x}H`}</strong><span>NATIONAL MODEL</span></div>
        <div className="aqi-number">{Math.round(point.y)}</div>
        <div className="status elevated">{cfg.label} · {cfg.unit}</div>
        <div className="forecast-data"><span><Activity size={12} /> MODEL OUTPUT</span></div>
      </div>)}</div>
      {!liveData && <div className="empty-state">Waiting for the live air-quality and weather feeds...</div>}
    </div>
  </PageShell>;
}