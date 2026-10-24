import { useEffect, useMemo, useState, type CSSProperties, type ReactNode } from 'react';
import { Circle, CircleMarker, MapContainer, Marker, Polygon, Polyline, Popup, Rectangle, TileLayer, Tooltip, WMSTileLayer, useMap, useMapEvents } from 'react-leaflet';
import MarkerClusterGroup from 'react-leaflet-cluster';
import L, { type LatLngBoundsExpression } from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { modelApiUrl, useLiveMapData, type FireHotspot, type MapPoint } from '../../hooks/useLiveMapData';
import type { ModelForecast } from '../../data/dashboard';

// ------------------------------------------------------------------ constants
const indiaBounds: LatLngBoundsExpression = [[6.5, 68], [37.5, 97.5]];
const DETAIL_ZOOM = 9; // wind arrows and heat circles in the airshed view only draw at city zoom
const DELHI: [number, number] = [28.6139, 77.209];

// Each tab moves the map to the area that suits it
const modeViews: Record<string, LatLngBoundsExpression> = {
  PLUME: [[24, 70], [34, 82]],
  FIRES: [[22, 68], [35, 90]],
  'CROSS-SECTION': [[24.5, 72], [32.5, 90]],
};

// Cross-section transect: a west to east band along the Indo-Gangetic plain
const CS_LAT = 28.6;
const CS_HALF_WIDTH = 0.75; // degrees of latitude either side of the line
const CS_WEST = 74;
const CS_EAST = 88;
const CS_STEP = 0.5; // degrees of longitude per bin

const maptilerKey = import.meta.env.VITE_MAPTILER_KEY as string | undefined;
const tileUrl = maptilerKey
  ? `https://api.maptiler.com/maps/dataviz-dark/{z}/{x}/{y}.png?key=${maptilerKey}`
  : 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png';
const tileAttribution = maptilerKey
  ? '&copy; MapTiler &copy; OpenStreetMap contributors'
  : '&copy; OpenStreetMap contributors';

const overlayStyle: CSSProperties = {
  position: 'absolute', zIndex: 1000, background: 'rgba(7,15,26,.92)', border: '1px solid rgba(49,214,229,.35)',
  borderRadius: 8, padding: '10px 12px', color: '#d8e6ee', fontSize: 12, lineHeight: 1.5, pointerEvents: 'auto',
};

// ------------------------------------------------------------------ geometry helpers
const toRad = (degrees: number) => degrees * Math.PI / 180;
const toDeg = (radians: number) => radians * 180 / Math.PI;

function distanceKm(aLat: number, aLon: number, bLat: number, bLon: number) {
  const dLat = toRad(bLat - aLat);
  const dLon = toRad(bLon - aLon);
  const h = Math.sin(dLat / 2) ** 2 + Math.cos(toRad(aLat)) * Math.cos(toRad(bLat)) * Math.sin(dLon / 2) ** 2;
  return 12742 * Math.asin(Math.sqrt(h));
}

function bearingDeg(aLat: number, aLon: number, bLat: number, bLon: number) {
  const dLon = toRad(bLon - aLon);
  const y = Math.sin(dLon) * Math.cos(toRad(bLat));
  const x = Math.cos(toRad(aLat)) * Math.sin(toRad(bLat)) - Math.sin(toRad(aLat)) * Math.cos(toRad(bLat)) * Math.cos(dLon);
  return (toDeg(Math.atan2(y, x)) + 360) % 360;
}

function angleDiff(a: number, b: number) { return Math.abs(((a - b + 540) % 360) - 180); }

function destination(lat: number, lon: number, bearing: number, km: number): [number, number] {
  const b = toRad(bearing);
  const dLat = (km / 111.32) * Math.cos(b);
  const dLon = (km / (111.32 * Math.max(Math.cos(toRad(lat)), 0.1))) * Math.sin(b);
  return [lat + dLat, lon + dLon];
}

function nearestStation(points: MapPoint[], lat: number, lon: number) {
  let best: { p: MapPoint; d: number } | null = null;
  for (const p of points) {
    const d = distanceKm(lat, lon, p.latitude, p.longitude);
    if (!best || d < best.d) best = { p, d };
  }
  return best;
}

function mean(values: number[]) { return values.reduce((sum, v) => sum + v, 0) / Math.max(values.length, 1); }

function thin(points: MapPoint[], cellDegrees: number) {
  const seen = new Set<string>();
  return points.filter((p) => {
    const key = `${Math.round(p.latitude / cellDegrees)}:${Math.round(p.longitude / cellDegrees)}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

// ------------------------------------------------------------------ colours and icons
function markerColor(aqi: number) {
  if (aqi >= 301) return '#d7191c';
  if (aqi >= 201) return '#f07c18';
  if (aqi >= 101) return '#f4d313';
  return '#35c878';
}

function fireColor(brightness: number) {
  if (brightness >= 380) return '#ffd23f';
  if (brightness >= 340) return '#ff7a2f';
  return '#ff315d';
}

const iconCache = new Map<number, L.DivIcon>();
function stationIcon(aqi: number) {
  let icon = iconCache.get(aqi);
  if (!icon) {
    icon = L.divIcon({
      className: 'station-pin-wrapper',
      html: `<span class="station-pin" style="--pin-color:${markerColor(aqi)}"><i>${aqi}</i></span>`,
      iconSize: [34, 42],
      iconAnchor: [17, 42],
    });
    iconCache.set(aqi, icon);
  }
  return icon;
}

// A pseudo-3D column: height follows AQI, with a lit top and a ground shadow
function columnIcon(aqi: number) {
  const height = Math.round(Math.min(150, 10 + aqi * 0.32));
  const c = markerColor(aqi);
  return L.divIcon({
    className: 'aqi-column-icon',
    html: `<div style="position:relative;width:16px;height:${height}px">
      <div style="position:absolute;bottom:-4px;left:-5px;width:26px;height:9px;background:rgba(0,0,0,.5);border-radius:50%"></div>
      <div style="position:absolute;bottom:0;left:0;width:16px;height:${height}px;background:linear-gradient(90deg,${c}bb,${c},${c}88);border-radius:2px 2px 0 0;box-shadow:3px 0 6px rgba(0,0,0,.55)"></div>
      <div style="position:absolute;top:-3px;left:0;width:16px;height:7px;background:${c};border-radius:50%;filter:brightness(1.35)"></div>
    </div>`,
    iconSize: [16, height],
    iconAnchor: [8, height],
  });
}

function windIcon(speed: number, windDirection: number) {
  const size = Math.round(16 + Math.min(speed, 30) * 0.7);
  const color = speed >= 25 ? '#ff5d5d' : speed >= 15 ? '#f2a03d' : speed >= 8 ? '#b8e35a' : '#31d6e5';
  const bearing = (windDirection + 180) % 360; // Open-Meteo gives where the wind comes FROM
  return L.divIcon({
    className: 'wind-arrow-icon',
    html: `<svg width="${size}" height="${size}" viewBox="0 0 24 24" style="display:block;transform:rotate(${bearing}deg)"><path d="M12 2 L19 21 L12 16.5 L5 21 Z" fill="${color}" fill-opacity=".92" stroke="#06101b" stroke-width="1"/></svg>`,
    iconSize: [size, size],
    iconAnchor: [size / 2, size / 2],
  });
}

// ------------------------------------------------------------------ plume footprints from fires
type Plume = { fire: FireHotspot; speed: number; bearing: number; lengthKm: number; polygon: [number, number][]; reachesDelhi: boolean; etaHours: number | null };

// Straight-line advection: each fire's smoke is carried downwind by the nearest station's wind for 12 hours.
// This is a simple footprint, not a dispersion model.
function buildPlumes(fires: FireHotspot[], points: MapPoint[]): Plume[] {
  if (!fires.length || !points.length) return [];
  const candidates = fires
    .filter((fire) => distanceKm(fire.latitude, fire.longitude, DELHI[0], DELHI[1]) <= 700)
    .sort((a, b) => b.brightness - a.brightness)
    .slice(0, 250);
  return candidates.flatMap((fire) => {
    const station = nearestStation(points, fire.latitude, fire.longitude);
    if (!station || station.d > 300) return [];
    const speed = Math.max(station.p.windSpeed, 3);
    const bearing = (station.p.windDirection + 180) % 360;
    const lengthKm = Math.min(speed * 12, 450);
    const spread = 18;
    const polygon: [number, number][] = [
      [fire.latitude, fire.longitude],
      ...[-spread, -spread / 2, 0, spread / 2, spread].map((offset) => destination(fire.latitude, fire.longitude, bearing + offset, lengthKm)),
    ];
    const toDelhiKm = distanceKm(fire.latitude, fire.longitude, DELHI[0], DELHI[1]);
    const toDelhiBearing = bearingDeg(fire.latitude, fire.longitude, DELHI[0], DELHI[1]);
    const reachesDelhi = toDelhiKm <= lengthKm && angleDiff(toDelhiBearing, bearing) <= spread;
    return [{ fire, speed, bearing, lengthKm, polygon, reachesDelhi, etaHours: reachesDelhi ? toDelhiKm / speed : null }];
  });
}

// ------------------------------------------------------------------ small map helpers
function ZoomGate({ min, children }: { min: number; children: ReactNode }) {
  const map = useMap();
  const [zoom, setZoom] = useState(map.getZoom());
  useMapEvents({ zoomend: () => setZoom(map.getZoom()) });
  return zoom >= min ? <>{children}</> : null;
}

function ModeView({ mode }: { mode: string }) {
  const map = useMap();
  useEffect(() => {
    map.fitBounds(modeViews[mode] ?? indiaBounds, { padding: [10, 10], animate: true });
  }, [map, mode]);
  return null;
}

function WindLines({ point }: { point: MapPoint }) {
  const speed = Number.isFinite(point.windSpeed) ? Math.min(point.windSpeed, 25) : 0;
  const length = 0.025 + speed * 0.0012; // degrees of latitude
  const headSize = length * 0.3;
  const cosLat = Math.max(Math.cos(point.latitude * Math.PI / 180), 0.1);
  const bearing = (((point.windDirection + 180) % 360) * Math.PI) / 180;
  const move = (from: [number, number], b: number, dist: number): [number, number] => [
    from[0] + Math.cos(b) * dist,
    from[1] + (Math.sin(b) * dist) / cosLat,
  ];
  const start: [number, number] = [point.latitude, point.longitude];
  const tip = move(start, bearing, length);
  const wingSpread = 0.45;
  const wingA = move(tip, bearing + Math.PI - wingSpread, headSize);
  const wingB = move(tip, bearing + Math.PI + wingSpread, headSize);
  return (
    <>
      <Polyline positions={[start, tip]} pathOptions={{ color: '#00e5ff', weight: 2.5, opacity: 0.95, lineCap: 'butt' }} />
      <Polygon positions={[tip, wingA, wingB]} pathOptions={{ color: '#00e5ff', fillColor: '#00e5ff', fillOpacity: 0.95, weight: 1 }} />
    </>
  );
}

// ------------------------------------------------------------------ popups
function value(v: number, unit: string) { return `${Number.isFinite(v) ? v : '—'} ${unit}`; }

const forecastCache = new Map<number, { t: number; forecast: ModelForecast }>();

function StationForecast({ id }: { id: number }) {
  const cachedEntry = forecastCache.get(id);
  const [state, setState] = useState<'loading' | 'error' | ModelForecast>(cachedEntry?.forecast ?? 'loading');

  useEffect(() => {
    const entry = forecastCache.get(id);
    if (entry && Date.now() - entry.t < 15 * 60_000) return;
    let cancelled = false;
    fetch(`${modelApiUrl}/stations/forecast?id=${id}`)
      .then((response) => { if (!response.ok) throw new Error(String(response.status)); return response.json() as Promise<ModelForecast>; })
      .then((forecast) => { forecastCache.set(id, { t: Date.now(), forecast }); if (!cancelled) setState(forecast); })
      .catch(() => { if (!cancelled) setState('error'); });
    return () => { cancelled = true; };
  }, [id]);

  if (state === 'loading') return <div className="popup-section"><b>Model AQI forecast</b><small>Loading…</small></div>;
  if (state === 'error') return <div className="popup-section"><b>Model AQI forecast</b><small>Unavailable for this station</small></div>;
  return <div className="popup-section"><b>Model AQI forecast (CPCB)</b>{(state.targets.AQI ?? []).map((p) => <span key={p.horizon_h}>{p.horizon_h === 0 ? 'Now' : `+${p.horizon_h}h`} <em>{p.predicted == null ? '—' : Math.round(p.predicted)}</em></span>)}</div>;
}

function StationPopup({ point }: { point: MapPoint }) {
  return <div className="station-popup"><strong>{point.name} · {point.city}, {point.state}</strong><small>{point.source}</small>
    <div className="popup-section"><b>Air quality</b><span>US AQI <em>{point.aqi}</em></span><span>PM2.5 <em>{value(point.pm25, 'μg/m³')}</em></span><span>PM10 <em>{value(point.pm10, 'μg/m³')}</em></span><span>NO / NO₂ / NOx <em>{point.no} / {point.no2} / {point.nox ?? 'not provided'} μg/m³</em></span><span>NH₃ / SO₂ <em>{point.nh3} / {point.so2} μg/m³</em></span><span>CO / O₃ <em>{point.co} / {point.ozone} μg/m³</em></span></div>
    <div className="popup-section"><b>Weather</b><span>Temperature <em>{value(point.temperature, '°C')}</em></span><span>Humidity <em>{value(point.humidity, '%')}</em></span><span>Wind <em>{point.windSpeed} km/h · {point.windDirection}°</em></span><span>Precipitation <em>{value(point.precipitation, 'mm')}</em></span><span>Boundary layer <em>{point.boundaryLayerHeight == null ? '—' : `${point.boundaryLayerHeight} m`}</em></span></div>
    <StationForecast id={point.id} />
  </div>;
}

// A station drawn as a plain circle, used by the tabs that do not show pins
function StationDot({ point, radius, color, fillOpacity = 0.75, weight = 1 }: { point: MapPoint; radius: number; color?: string; fillOpacity?: number; weight?: number }) {
  const c = color ?? markerColor(point.aqi);
  return <CircleMarker center={[point.latitude, point.longitude]} radius={radius} pathOptions={{ color: c, fillColor: c, fillOpacity, weight }}>
    <Tooltip>{point.name} · AQI {point.aqi} · PM2.5 {point.pm25}</Tooltip>
    <Popup><StationPopup point={point} /></Popup>
  </CircleMarker>;
}

// ------------------------------------------------------------------ cross-section chart
type CrossBin = { lon: number; aqi: number; pbl: number | null; n: number };

function CrossSectionChart({ bins }: { bins: CrossBin[] }) {
  const width = 640, height = 170, left = 40, right = 12, top = 10, bottom = 26;
  const plotW = width - left - right, plotH = height - top - bottom;
  const x = (lon: number) => left + ((lon - CS_WEST) / (CS_EAST - CS_WEST)) * plotW;
  const aqiMax = Math.max(300, ...bins.map((b) => b.aqi));
  const pblValues = bins.filter((b) => b.pbl != null).map((b) => b.pbl as number);
  const pblMax = Math.max(...pblValues, 1);
  const yAqi = (v: number) => top + plotH - (v / aqiMax) * plotH;
  const yPbl = (v: number) => top + plotH - (v / pblMax) * plotH;
  const barW = Math.max(4, (plotW / ((CS_EAST - CS_WEST) / CS_STEP)) * 0.8);
  const pblLine = bins.filter((b) => b.pbl != null).map((b) => `${x(b.lon)},${yPbl(b.pbl as number)}`).join(' ');
  return <svg viewBox={`0 0 ${width} ${height}`} width="100%" height={height} role="img" aria-label="AQI and boundary layer height along the west to east transect">
    <line x1={left} y1={top} x2={left} y2={top + plotH} stroke="#5c7181" /><line x1={left} y1={top + plotH} x2={width - right} y2={top + plotH} stroke="#5c7181" />
    {[0, 100, 200, 300].filter((v) => v <= aqiMax).map((v) => <g key={v}><line x1={left} y1={yAqi(v)} x2={width - right} y2={yAqi(v)} stroke="#5c7181" strokeOpacity=".25" /><text x={left - 5} y={yAqi(v) + 3} fill="#9db5c0" fontSize="10" textAnchor="end">{v}</text></g>)}
    {[74, 76, 78, 80, 82, 84, 86, 88].map((lon) => <text key={lon} x={x(lon)} y={height - 8} fill="#9db5c0" fontSize="10" textAnchor="middle">{lon}°E</text>)}
    <line x1={x(77.209)} y1={top} x2={x(77.209)} y2={top + plotH} stroke="#31d6e5" strokeDasharray="3 3" /><text x={x(77.209) + 3} y={top + 9} fill="#31d6e5" fontSize="10">Delhi</text>
    {bins.map((b) => <rect key={b.lon} x={x(b.lon) - barW / 2} y={yAqi(b.aqi)} width={barW} height={top + plotH - yAqi(b.aqi)} fill={markerColor(b.aqi)} fillOpacity=".85"><title>{`${b.lon.toFixed(2)}°E · AQI ${Math.round(b.aqi)} · ${b.n} stations`}</title></rect>)}
    {pblLine && <polyline points={pblLine} fill="none" stroke="#f2be67" strokeWidth="2" strokeDasharray="5 3" />}
  </svg>;
}

// ------------------------------------------------------------------ map panel
const FIRMS_MODES = ['2D AIRSHED', 'PLUME', 'FIRES'];

export function MapPanel({ compact = false, mode = '2D AIRSHED' }: { compact?: boolean; mode?: string }) {
  const { data, error } = useLiveMapData();
  const [firmsVisible, setFirmsVisible] = useState(true);
  const [firmsOpacity, setFirmsOpacity] = useState(0.5);
  const [firmsTileError, setFirmsTileError] = useState(false);
  const points = useMemo(() => data?.points ?? [], [data]);
  const fires = useMemo(() => data?.fires ?? [], [data]);
  const firmsKey = import.meta.env.VITE_NASA_FIRMS_MAP_KEY;

  const cityColumns = useMemo(() => {
    const groups = new Map<string, MapPoint[]>();
    points.forEach((p) => { const key = `${p.city || p.name}|${p.state}`; groups.set(key, [...(groups.get(key) ?? []), p]); });
    return [...groups.entries()].map(([key, group]) => ({
      key, city: group[0].city || group[0].name, state: group[0].state, count: group.length,
      lat: mean(group.map((p) => p.latitude)), lon: mean(group.map((p) => p.longitude)), aqi: Math.round(mean(group.map((p) => p.aqi))),
    }));
  }, [points]);

  const crossBins = useMemo<CrossBin[]>(() => {
    const bins = new Map<number, { aqi: number[]; pbl: number[] }>();
    points.forEach((p) => {
      if (Math.abs(p.latitude - CS_LAT) > CS_HALF_WIDTH || p.longitude < CS_WEST || p.longitude >= CS_EAST) return;
      const k = Math.floor((p.longitude - CS_WEST) / CS_STEP);
      const bin = bins.get(k) ?? { aqi: [], pbl: [] };
      bin.aqi.push(p.aqi);
      if (p.boundaryLayerHeight != null) bin.pbl.push(p.boundaryLayerHeight);
      bins.set(k, bin);
    });
    return [...bins.entries()].sort((a, b) => a[0] - b[0]).map(([k, bin]) => ({
      lon: CS_WEST + (k + 0.5) * CS_STEP, aqi: mean(bin.aqi), pbl: bin.pbl.length ? mean(bin.pbl) : null, n: bin.aqi.length,
    }));
  }, [points]);

  const plumes = useMemo(() => (mode === 'PLUME' ? buildPlumes(fires, points) : []), [mode, fires, points]);
  const delhiPlumes = plumes.filter((p) => p.reachesDelhi);
  const earliestEta = delhiPlumes.length ? Math.min(...delhiPlumes.map((p) => p.etaHours as number)) : null;
  const windPoints = useMemo(() => (mode === 'WIND' ? thin(points, 0.5) : []), [mode, points]);
  const fireNearDelhi = fires.filter((f) => distanceKm(f.latitude, f.longitude, DELHI[0], DELHI[1]) <= 500).length;
  const strongFires = fires.filter((f) => f.brightness >= 350).length;
  const topCities = [...cityColumns].sort((a, b) => b.aqi - a.aqi).slice(0, 5);

  return <div className={`map-panel leaflet-panel ${compact ? 'map-compact' : ''}`}>
    <div className="map-mode-badge">{mode}</div>
    <MapContainer bounds={indiaBounds} minZoom={4} maxZoom={15} preferCanvas scrollWheelZoom className={`leaflet-map map-${mode.toLowerCase().replaceAll(' ', '-')}`}>
      <TileLayer attribution={tileAttribution} url={tileUrl} />
      <ModeView mode={mode} />
      {firmsKey && firmsVisible && FIRMS_MODES.includes(mode) && <WMSTileLayer
        attribution='NASA FIRMS'
        url={`https://firms.modaps.eosdis.nasa.gov/mapserver/wms/fires/${firmsKey}/fires_modis_24/`}
        layers="fires_modis_24"
        format="image/png"
        transparent
        opacity={firmsOpacity}
        version="1.1.1"
        eventHandlers={{
          tileerror: (event) => {
            const errorMessage = String((event as { error?: { message?: string } }).error?.message ?? '');
            // Leaflet aborts obsolete requests whenever the viewport changes; those are not FIRMS failures.
            if (!/abort|cancel/i.test(errorMessage)) setFirmsTileError(true);
          },
          tileload: () => setFirmsTileError(false),
        }}
      />}

      {/* 2D AIRSHED: clustered AQI pins, with heat circles and wind arrows when zoomed in */}
      {mode === '2D AIRSHED' && <>
        <ZoomGate min={DETAIL_ZOOM}>
          {points.map((point) => <Circle key={`${point.id}-heat`} center={[point.latitude, point.longitude]} radius={1200 + point.aqi * 5} pathOptions={{ color: markerColor(point.aqi), fillColor: markerColor(point.aqi), fillOpacity: .26, weight: 0 }} />)}
          {points.map((point) => <WindLines point={point} key={`${point.id}-wind`} />)}
        </ZoomGate>
        {fires.map((fire, index) => <CircleMarker center={[fire.latitude, fire.longitude]} radius={3} pathOptions={{ color: '#ff6d87', fillColor: '#ff315d', fillOpacity: .8, weight: 0 }} key={`${fire.latitude}-${fire.longitude}-${index}`} />)}
        <MarkerClusterGroup chunkedLoading>
          {points.map((point) => <Marker position={[point.latitude, point.longitude]} icon={stationIcon(point.aqi)} key={point.id}>
            <Tooltip direction="top" offset={[0, -36]}>{point.name} · AQI {point.aqi}</Tooltip>
            <Popup><StationPopup point={point} /></Popup>
          </Marker>)}
        </MarkerClusterGroup>
      </>}

      {/* 3D COLUMN: one column per city, height follows mean AQI */}
      {mode === '3D COLUMN' && cityColumns.map((city) => <Marker key={city.key} position={[city.lat, city.lon]} icon={columnIcon(city.aqi)}>
        <Tooltip direction="top" offset={[0, -8]}>{city.city}, {city.state} · mean AQI {city.aqi} · {city.count} station{city.count === 1 ? '' : 's'}</Tooltip>
      </Marker>)}

      {/* CROSS-SECTION: the transect band, with stations inside it highlighted */}
      {mode === 'CROSS-SECTION' && <>
        <Rectangle bounds={[[CS_LAT - CS_HALF_WIDTH, CS_WEST], [CS_LAT + CS_HALF_WIDTH, CS_EAST]]} pathOptions={{ color: '#31d6e5', weight: 1, dashArray: '5 5', fillColor: '#31d6e5', fillOpacity: .07 }} />
        <Polyline positions={[[CS_LAT, CS_WEST], [CS_LAT, CS_EAST]]} pathOptions={{ color: '#31d6e5', weight: 2, dashArray: '2 6' }} />
        {points.map((point) => {
          const inBand = Math.abs(point.latitude - CS_LAT) <= CS_HALF_WIDTH && point.longitude >= CS_WEST && point.longitude < CS_EAST;
          return <StationDot key={point.id} point={point} radius={inBand ? 7 : 2.5} fillOpacity={inBand ? 0.85 : 0.35} weight={inBand ? 1.5 : 0} color={inBand ? undefined : '#8aa0ad'} />;
        })}
      </>}

      {/* PLUME: PM2.5 blobs plus 12 h downwind footprints from each fire */}
      {mode === 'PLUME' && <>
        {points.map((point) => <CircleMarker key={`${point.id}-blob`} center={[point.latitude, point.longitude]} radius={Math.min(30, 6 + point.pm25 / 10)} pathOptions={{ color: markerColor(point.aqi), fillColor: markerColor(point.aqi), fillOpacity: .28, weight: 0 }}>
          <Tooltip>{point.name} · PM2.5 {point.pm25} μg/m³</Tooltip>
        </CircleMarker>)}
        {plumes.map((plume, index) => <Polygon key={`plume-${index}`} positions={plume.polygon} pathOptions={{ color: plume.reachesDelhi ? '#ffd23f' : '#ff8a3d', weight: plume.reachesDelhi ? 1 : 0, fillColor: plume.reachesDelhi ? '#ffd23f' : '#ff8a3d', fillOpacity: plume.reachesDelhi ? .22 : .1 }} />)}
        {plumes.map((plume, index) => <CircleMarker key={`src-${index}`} center={[plume.fire.latitude, plume.fire.longitude]} radius={2.5} pathOptions={{ color: '#ff315d', fillColor: '#ff315d', fillOpacity: .9, weight: 0 }} />)}
        <CircleMarker center={DELHI} radius={7} pathOptions={{ color: '#31d6e5', fillColor: '#06101b', fillOpacity: 1, weight: 2 }}><Tooltip permanent direction="right">Delhi NCR</Tooltip></CircleMarker>
      </>}

      {/* FIRES: detections sized and coloured by brightness, stations as quiet dots */}
      {mode === 'FIRES' && <>
        {points.map((point) => <StationDot key={point.id} point={point} radius={3} color="#8aa0ad" fillOpacity={0.5} weight={0} />)}
        {fires.map((fire, index) => <CircleMarker key={`${fire.latitude}-${fire.longitude}-${index}`} center={[fire.latitude, fire.longitude]} radius={3 + Math.min(6, Math.max(0, (fire.brightness - 300) / 20))} pathOptions={{ color: fireColor(fire.brightness), fillColor: fireColor(fire.brightness), fillOpacity: .8, weight: 1 }}>
          <Popup>{fire.satellite} satellite detection<br />Brightness: {fire.brightness} K{fire.acquisitionDate && <><br />Acquired: {fire.acquisitionDate}</>}</Popup>
        </CircleMarker>)}
        <CircleMarker center={DELHI} radius={6} pathOptions={{ color: '#31d6e5', fillColor: '#06101b', fillOpacity: 1, weight: 2 }}><Tooltip permanent direction="right">Delhi NCR</Tooltip></CircleMarker>
      </>}

      {/* WIND: one arrow per ~50 km, rotated to the direction the air is moving */}
      {mode === 'WIND' && windPoints.map((point) => <Marker key={point.id} position={[point.latitude, point.longitude]} icon={windIcon(point.windSpeed, point.windDirection)}>
        <Tooltip>{point.name} · {point.windSpeed} km/h, moving toward {(point.windDirection + 180) % 360}°</Tooltip>
      </Marker>)}
    </MapContainer>

    {/* Per-tab information panels */}
    {mode === 'CROSS-SECTION' && <div style={{ ...overlayStyle, left: 12, right: 12, bottom: 64 }} aria-label="AQI cross-section">
      <strong>West to east transect along {CS_LAT}°N · stations within {CS_HALF_WIDTH}° of the line</strong>
      {crossBins.length ? <CrossSectionChart bins={crossBins} /> : <div>No stations in the transect band yet.</div>}
      <div>Bars: mean US AQI per {CS_STEP}° of longitude. Dashed line: mean boundary layer height (lower means pollution is trapped nearer the ground).</div>
    </div>}
    {mode === 'PLUME' && <div style={{ ...overlayStyle, top: 52, right: 12, maxWidth: 280 }}>
      <strong>Smoke footprints toward Delhi NCR</strong>
      {!firmsKey ? <div>Add VITE_NASA_FIRMS_MAP_KEY to .env.local to draw fire-driven plumes.</div>
        : !fires.length ? <div>No fire detections loaded in the last 24 h.</div>
        : <div>{plumes.length} fires within 700 km modelled · {delhiPlumes.length} with smoke heading to Delhi{earliestEta != null && <> · earliest arrival about {Math.max(1, Math.round(earliestEta))} h</>}</div>}
      <small>Each footprint carries the fire's smoke downwind for 12 h using the nearest station's wind. It is a simple wind-advection estimate, not a dispersion model.</small>
    </div>}
    {mode === 'FIRES' && <div style={{ ...overlayStyle, top: 52, right: 12, maxWidth: 260 }}>
      <strong>Fire detections, last 24 h</strong>
      {!firmsKey ? <div>Add VITE_NASA_FIRMS_MAP_KEY to .env.local to load fire detections.</div>
        : <div>{fires.length} in India · {fireNearDelhi} within 500 km of Delhi · {strongFires} high intensity (350 K or more)</div>}
    </div>}
    {mode === '3D COLUMN' && !!topCities.length && <div style={{ ...overlayStyle, top: 52, right: 12, maxWidth: 240 }}>
      <strong>Highest mean AQI</strong>
      {topCities.map((city) => <div key={city.key}>{city.city} · {city.aqi}</div>)}
    </div>}
    {mode === 'WIND' && <div style={{ ...overlayStyle, top: 52, right: 12 }}>
      <strong>Surface wind, km/h</strong>
      <div><span style={{ color: '#31d6e5' }}>▲</span> under 8 · <span style={{ color: '#b8e35a' }}>▲</span> 8 to 15 · <span style={{ color: '#f2a03d' }}>▲</span> 15 to 25 · <span style={{ color: '#ff5d5d' }}>▲</span> 25 or more</div>
      <small>Arrows point where the air is moving.</small>
    </div>}

    {error && <div className="map-tooltip" style={{ top: 'auto', bottom: 52, left: '50%', right: 'auto', transform: 'translateX(-50%)', zIndex: 1100, maxWidth: '90%' }}><span className="live-dot" /> LIVE MAP FEED UNAVAILABLE <small>{error}</small></div>}
    {!data && !error && <div className="map-tooltip"><span className="live-dot" /> LOADING LIVE MAP FEEDS</div>}
    {!compact && FIRMS_MODES.includes(mode) && <div className="map-layer-control">
      <label><input type="checkbox" checked={firmsVisible} onChange={(event) => setFirmsVisible(event.target.checked)} /> NASA FIRMS satellite detections</label>
      <label>Opacity <input type="range" min="0.15" max="0.9" step="0.05" value={firmsOpacity} onChange={(event) => setFirmsOpacity(Number(event.target.value))} /></label>
      {firmsTileError && <small>FIRMS imagery unavailable; CSV detections remain available.</small>}
    </div>}
    {data && !compact && <div className="map-legend leaflet-legend">
      {mode === '2D AIRSHED' && <><span><i className="legend-square" /> Station colour · current AQI (zoom in for heat circles and wind arrows)</span><span><i className="legend-square pink" /> {data.fireFeedAvailable ? `Fire detections · ${data.fires.length}` : 'Fire feed unavailable'}</span></>}
      {mode === '3D COLUMN' && <span><i className="legend-square" /> Column height and colour · mean AQI per city</span>}
      {mode === 'CROSS-SECTION' && <span><i className="legend-line wind" /> Transect band · west to east along {CS_LAT}°N</span>}
      {mode === 'PLUME' && <span><i className="legend-square pink" /> Orange footprints · 12 h downwind smoke from each fire · yellow reaches Delhi</span>}
      {mode === 'FIRES' && <span><i className="legend-square pink" /> {data.fireFeedAvailable ? `Detections · ${data.fires.length}` : 'Fire feed unavailable'}{data.fireUpdatedAt && ` · refreshed ${new Date(data.fireUpdatedAt).toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit' })}`}</span>}
      {mode === 'WIND' && <span><i className="legend-line wind" /> Arrow size and colour · wind speed</span>}
    </div>}
    <div className="map-scale">INDIA · {points.length} stations <span>LIVE · {data?.updatedAt ?? 'SYNCING'}</span></div>
  </div>;
}