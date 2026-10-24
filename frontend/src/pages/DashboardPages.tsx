import { useState, type CSSProperties, type ReactNode } from 'react';
import { Activity, AlertTriangle, CalendarDays, ChevronRight, CloudFog, CloudSun, Download, Eye, Layers3, Radio, RefreshCw, Wind } from 'lucide-react';
import { getAqiStatus, type LiveData, type Notify, type PageKey } from '../data/dashboard';
import { useLiveMapData } from '../hooks/useLiveMapData';
import { MapPanel } from '../Components/dashboard/MapPanel';
import { MetricCard, PageShell, PanelTitle } from '../Components/dashboard/Shared';
import { InteractiveChart, type ChartSeries } from '../Components/dashboard/InteractiveChart';

// The Inversion & PBL page now lives in its own file; re-exported so existing imports keep working.
export { InversionPage } from './InversionPage';

type ForecastFilter = 'AQI' | 'PM2.5' | 'O₃';
// AQI is CPCB everywhere the model produces it. The Open-Meteo US-AQI value only
// appears as a fallback and is labelled dynamically so the UI never claims CPCB for a US number.
const forecastConfig: Record<ForecastFilter, { key: 'aqi' | 'pm25' | 'ozone'; model: 'AQI' | 'PM2.5' | 'Ozone'; label: string; unit: string; colour: string }> = {
  AQI: { key: 'aqi', model: 'AQI', label: 'AQI', unit: 'CPCB AQI', colour: '#fb4f77' },
  'PM2.5': { key: 'pm25', model: 'PM2.5', label: 'PM2.5', unit: 'μg/m³', colour: '#31d6e5' },
  'O₃': { key: 'ozone', model: 'Ozone', label: 'O₃', unit: 'μg/m³', colour: '#c7a9ff' },
};
function modelPoints(data: LiveData | null, target: string) {
  return (data?.modelForecast?.targets[target] ?? []).filter((point) => point.predicted != null && Number.isFinite(point.predicted));
}
// `current.aqi` is CPCB when the model is up, US AQI when it fell back. `liveData.source` carries the reason.
function currentAqiUnit(data: LiveData | null) { return data?.modelForecast ? 'CPCB AQI' : 'US AQI (fallback)'; }

function forecastCards(data: LiveData | null, filter: ForecastFilter) {
  const config = forecastConfig[filter];
  const target = config.model;
  return modelPoints(data, target).map((point, index) => {
    const value = Math.round(point.predicted as number);
    const time = new Date(point.valid_at).toLocaleString('en-IN', { month: 'short', day: '2-digit', hour: '2-digit', minute: '2-digit', hour12: false, timeZone: 'Asia/Kolkata' }).toUpperCase();
    return <div className={`forecast-card ${index === 0 ? 'selected' : ''}`} key={`${target}-${point.horizon_h}`} style={{ '--forecast-accent': config.colour } as CSSProperties}>
      <div className="forecast-top"><strong>{point.horizon_h === 0 ? 'NOW' : `+${point.horizon_h}H`}</strong><span>{time}</span></div>
      <div className="aqi-number">{value}</div>
      <div className={`status ${filter === 'AQI' ? getAqiStatus(value).tone : 'elevated'}`}>{filter === 'AQI' ? getAqiStatus(value).label : `${config.label} · ${config.unit}`}</div>
      <div className="forecast-data"><span><CloudFog size={12} /> {config.label} {value}</span><span>National model</span></div>
    </div>;
  });
}
function modelSeries(data: LiveData | null, target: string, name: string, colour: string, unit: string): ChartSeries {
  return { name, colour, unit, points: modelPoints(data, target).map((point) => ({ x: point.horizon_h, y: point.predicted as number, label: `+${point.horizon_h}h (national model)` })) };
}
function LiveRequired({ children, data }: { children: ReactNode; data: LiveData | null }) { return data ? <>{children}</> : <div className="panel empty-state">Waiting for the live air-quality and weather feeds…</div>; }

export function Overview({ setPage, liveData, notify }: { setPage: (page: PageKey) => void; liveData: LiveData | null; notify: Notify }) {
  const [mapMode, setMapMode] = useState('2D AIRSHED');
  const current = liveData?.current;
  // Aerosol optical depth is dimensionless. The hook stores it multiplied by 100.
  const aod = current?.aerosolSolarDimming == null ? null : current.aerosolSolarDimming / 100;
  return <LiveRequired data={liveData}>
    <section className="hero-row">
      <div>
        <div className="eyebrow"><span className="live-dot" /> LIVE FEED · UPDATED {liveData?.updatedAt}</div>
        <h1>Delhi NCR <em>air intelligence</em></h1>
        <p className="hero-copy">Live Open-Meteo estimates for now, and national-model forecasts for the next 72 hours.</p>
      </div>
      <div className="hero-actions">
        <span className="button ghost"><CalendarDays size={15} /> {liveData?.forecast[0]?.time ?? 'LIVE WINDOW'}</span>
        <span className="button ghost">{liveData?.modelForecast ? 'NATIONAL MODEL ONLINE' : 'NATIONAL MODEL OFFLINE · US AQI FALLBACK'}</span>
      </div>
    </section>
    <div className="metrics-grid">
      <MetricCard label="Delhi NCR composite AQI" value={String(current?.aqi ?? '—')} unit={currentAqiUnit(liveData)} detail={`${liveData?.source} · ${getAqiStatus(current?.aqi ?? 0).label}`} accent="#fb7a90" icon={Activity} />
      <MetricCard label="Boundary layer (PBL)" value={current?.boundaryLayerHeight == null ? '—' : String(current.boundaryLayerHeight)} unit="m AGL" detail="Live Open-Meteo value for the current hour" accent="#f2be67" icon={Layers3} />
      <MetricCard label="Inversion strength" value={current?.inversionStrength == null ? '—' : String(current.inversionStrength)} unit="/10" detail={current?.inversionGradient == null ? 'Profile unavailable' : `Live lapse ${current.inversionGradient}°C/km`} accent="#fb7a90" icon={CloudFog} />
      <MetricCard label="Ventilation coefficient" value={current?.ventilationCoefficient == null ? '—' : current.ventilationCoefficient.toLocaleString('en-IN')} unit="m²/s" detail="PBL height × wind speed converted to m/s" accent="#f2be67" icon={Wind} />
      <MetricCard label="Wind vector (surface)" value={String(current?.windSpeed ?? '—')} unit="km/h" detail={`${current?.windDirection ?? '—'}° bearing`} accent="#31d6e5" icon={Wind} />
      <MetricCard label="Aerosol optical depth" value={aod == null ? '—' : aod.toFixed(2)} unit="AOD" detail="Dimensionless column AOD from the live air-quality feed" accent="#c7a9ff" icon={CloudSun} />
    </div>
    <div className="main-grid">
      <div className="panel map-card">
        <div className="panel-header">
          <div><span className="panel-kicker">LIVE AIRSHED · NOWCAST</span><h2>Pollution field & plume movement</h2></div>
          <div className="segmented">{['2D AIRSHED', '3D COLUMN', 'CROSS-SECTION'].map((mode) => <button key={mode} className={mapMode === mode ? 'active' : ''} onClick={() => { setMapMode(mode); notify(`${mode} layer enabled`); }}>{mode}</button>)}</div>
        </div>
        <MapPanel mode={mapMode} />
        <div className="map-footer"><span>Live PM2.5 and AQI map layer</span><span>Updated {liveData?.updatedAt}</span></div>
      </div>
      <aside className="side-stack">
        <div className="panel outlook-panel">
          <div className="panel-header"><div><span className="panel-kicker">NATIONAL MODEL OUTLOOK</span><h2>Next model checkpoints</h2></div><Eye size={15} color="#31d6e5" /></div>
          {modelPoints(liveData, 'AQI').slice(0, 3).map((point) => {
            const pm25 = modelPoints(liveData, 'PM2.5').find((p) => p.horizon_h === point.horizon_h)?.predicted;
            return <div className="outlook-row" key={point.horizon_h}>
              <div><b>{point.horizon_h === 0 ? 'NOW' : `+${point.horizon_h}H`}</b><small>National model · PM2.5 {pm25 != null ? Math.round(pm25) : '—'}</small></div>
              <strong className={`outlook-aqi ${getAqiStatus(point.predicted as number).tone}`}>{Math.round(point.predicted as number)}</strong>
            </div>;
          })}
        </div>
        <button className="alert-card" onClick={() => setPage('alerts')}>
          <div className="alert-icon"><AlertTriangle size={17} /></div>
          <div><b>LIVE AQI RESPONSE STATUS</b><p>{getAqiStatus(current?.aqi ?? 0).label} conditions based on the current AQI ({currentAqiUnit(liveData)}).</p><span>VIEW RESPONSE PLAYBOOK <ChevronRight size={12} /></span></div>
        </button>
      </aside>
    </div>
    <section className="panel forecast-panel">
      <div className="panel-header"><div><span className="panel-kicker">NATIONAL MODEL · 72H</span><h2>AQI trajectory across the airshed</h2></div><button className="button ghost small" onClick={() => setPage('forecast')}><Download size={14} /> VIEW MODEL RUN</button></div>
      <div className="trajectory">
        <InteractiveChart series={[modelSeries(liveData, 'AQI', 'AQI · national model', '#fb7b91', 'CPCB AQI')]} xTitle="Forecast time (hours from now, IST)" yTitle="CPCB AQI (index)" ariaLabel="National model AQI by forecast time" />
        <small className="chart-caption">National model forecast only · X: forecast time (hours from now, IST).</small>
      </div>
    </section>
  </LiveRequired>;
}

export function ForecastPage({ liveData, notify }: { liveData: LiveData | null; notify: Notify }) {
  const [filter, setFilter] = useState<ForecastFilter>('AQI');
  const config = forecastConfig[filter];
  const modelTarget = config.model;
  const chartSeries: ChartSeries[] = [modelSeries(liveData, modelTarget, `${config.label} · national model`, config.colour, config.unit)];
  // When AQI is selected and the model is down, `current.aqi` carries Open-Meteo US AQI.
  // Keep the value and its label honest — the chart above is empty in that case anyway.
  const currentUnit = filter === 'AQI' ? currentAqiUnit(liveData) : config.unit;
  return <PageShell title="72H forecast" eyebrow="FORECAST OPERATIONS">
    <div className="page-intro">
      <div><h1>Air quality <em>trajectory</em></h1><p>The national CPCB model is the only forecast source shown here. Live values come from the Open-Meteo feeds.</p></div>
      <div className="hero-actions"><div className="filter-group">{(['AQI', 'PM2.5', 'O₃'] as ForecastFilter[]).map((item) => <button key={item} className={`filter-button ${filter === item ? 'active' : ''}`} onClick={() => { setFilter(item); notify(`${item} forecast layer selected`); }}>{item}</button>)}</div></div>
    </div>
    <div className="forecast-wide"><div className="forecast-strip">{forecastCards(liveData, filter)}</div></div>
    <div className="two-col">
      <div className="panel chart-large">
        <PanelTitle kicker={`NATIONAL MODEL · 72H · ${filter}`} title={`${config.label} forecast trajectory`} />
        <InteractiveChart series={chartSeries} xTitle="Forecast time (hours from now, IST)" yTitle={`${config.label} (${config.unit})`} ariaLabel={`National model ${filter} by forecast time`} />
        <div className="chart-legend"><span><i style={{ background: config.colour }} />National model ({config.unit}) · 72h</span>{liveData?.modelForecastError && <span className="muted">National model unavailable.</span>}</div>
        <small className="chart-caption">X: forecast time (hours from now, IST) · Y: {config.unit}. Values are produced by the national model.</small>
      </div>
      <div className="panel pollutant-list">
        <PanelTitle kicker="LIVE VALUE" title={`${config.label} selected · live now`} />
        <div className="pollutant">
          <div className="pollutant-name"><b>{config.label}</b><span className="cyan">LIVE</span></div>
          <strong>{liveData?.current?.[config.key] ?? '—'}<small> {currentUnit}</small></strong>
        </div>
      </div>
    </div>
  </PageShell>;
}

export function SmokePage({ notify }: { notify: Notify }) {
  const [mapLayer, setMapLayer] = useState('PLUME');
  const { data } = useLiveMapData();
  const meanPm = data ? Math.round(data.points.reduce((sum, point) => sum + point.pm25, 0) / Math.max(data.points.length, 1)) : null;
  return <PageShell title="Smoke dispersion" eyebrow="TRANSBOUNDARY PLUME INTELLIGENCE">
    <div className="page-intro">
      <div><h1>Follow the <em>plume</em></h1><p>Satellite hotspots and air-quality estimates are shown only when the live feed responds.</p></div>
      <div className="hero-actions">
        <button className="button ghost" onClick={() => notify(`${data?.fires.length ?? 0} live fire detections loaded`)}><Radio size={15} /> {data?.fires.length ?? '—'} ACTIVE FIRES</button>
        <span className="button ghost"><RefreshCw size={15} /> UPDATED {data?.updatedAt ?? '—'}</span>
      </div>
    </div>
    <div className="panel smoke-map-card">
      <div className="panel-header">
        <PanelTitle kicker="LIVE SATELLITE + AIR QUALITY" title="Source field" />
        <div className="map-tabs">{['PLUME', 'FIRES', 'WIND'].map((layer) => <button key={layer} className={mapLayer === layer ? 'active' : ''} onClick={() => setMapLayer(layer)}>{layer}</button>)}</div>
      </div>
      <MapPanel mode={mapLayer} />
      <div className="smoke-bottom">
        <div><span className="panel-kicker">MEAN MAP PM2.5</span><strong>{meanPm ?? '—'} μg/m³</strong><small>Calculated from the live map points</small></div>
        <div><span className="panel-kicker">BOUNDARY LAYER</span><strong>{data?.boundaryLayerHeight ?? '—'} m</strong><small>Live meteorological feed</small></div>
        <div><span className="panel-kicker">UPDATED</span><strong>{data?.updatedAt ?? '—'}</strong><small>Time of the last successful refresh</small></div>
      </div>
    </div>
  </PageShell>;
}

export function EnsemblePage({ liveData }: { notify: Notify; liveData: LiveData | null }) {
  const rows = modelPoints(liveData, 'AQI');
  return <PageShell title="National model" eyebrow="PREDICTION SYSTEMS">
    <div className="page-intro">
      <div><h1>National <em>model output</em></h1><p>Pollutant forecasts are produced by the single national model run.</p></div>
      <span className="button ghost">INPUTS UPDATED {liveData?.updatedAt ?? '—'}</span>
    </div>
    <div className="panel table-panel">
      <PanelTitle kicker="NATIONAL MODEL · 72H" title="AQI checkpoints" />
      <table>
        <thead><tr><th>HORIZON</th><th>AQI</th><th>PM2.5</th><th>STATUS</th></tr></thead>
        <tbody>{rows.map((row) => {
          const aqi = row.predicted as number;
          const pm25 = modelPoints(liveData, 'PM2.5').find((p) => p.horizon_h === row.horizon_h)?.predicted;
          return <tr key={row.horizon_h}>
            <td>{row.horizon_h === 0 ? 'NOW' : `+${row.horizon_h}H`}</td>
            <td>{Math.round(aqi)}</td>
            <td>{pm25 != null ? Math.round(pm25) : '—'}</td>
            <td className="table-status">{getAqiStatus(aqi).label}</td>
          </tr>;
        })}</tbody>
      </table>
      {!rows.length && <div className="empty-state">Waiting for national model output.</div>}
    </div>
  </PageShell>;
}

export function AlertsPage({ liveData }: { notify: Notify; liveData: LiveData | null }) {
  const aqi = liveData?.current.aqi;
  const status = getAqiStatus(aqi ?? 0);
  return <PageShell title="Alerts & actions" eyebrow="RESPONSE OPERATIONS">
    <div className="page-intro">
      <div><h1>Response <em>playbook</em></h1><p>Operational status is derived from the current AQI used across the dashboard.</p></div>
      <span className="button ghost">AQI UPDATED {liveData?.updatedAt ?? '—'}</span>
    </div>
    <div className="panel alert-panel">
      <div className={`status ${status.tone}`}>{liveData ? status.label : 'NO LIVE STATUS'}</div>
      <h2>{liveData ? `AQI ${aqi} · ${status.label}` : 'Waiting for live feed'}</h2>
      <p>{liveData ? `Review local response procedures for the current AQI category (${currentAqiUnit(liveData)}).` : 'No alert is asserted until a current measurement is available.'}</p>
    </div>
  </PageShell>;
}