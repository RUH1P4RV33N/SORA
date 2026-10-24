// Place at: src/Components/dashboard/InteractiveChart.tsx
// A reusable line chart with labelled axes and a hover tooltip (works with mouse and touch).
import { useMemo, useRef, useState } from 'react';

export type ChartPoint = { x: number; y: number; label?: string };
export type ChartSeries = { name: string; colour: string; dashed?: boolean; unit?: string; points: ChartPoint[] };

type Props = {
  series: ChartSeries[];
  xTitle: string;                 // e.g. "Forecast time (hours from now, IST)"
  yTitle: string;                 // e.g. "Ozone (μg/m³)"
  ariaLabel: string;
  xMax?: number;                  // default 72
  yMax?: number;                  // default: rounded-up max of the data
  xTickEvery?: number;            // default 12
  height?: number;                // default 260
  formatX?: (x: number) => string;
};

const W = 900;
const PAD = { l: 58, r: 16, t: 16, b: 54 };

function niceMax(v: number) {
  if (!Number.isFinite(v) || v <= 0) return 1;
  const p = 10 ** Math.floor(Math.log10(v));
  const f = v / p;
  return (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10) * p;
}

export function InteractiveChart({ series, xTitle, yTitle, ariaLabel, xMax = 72, yMax, xTickEvery = 12, height = 260, formatX }: Props) {
  const H = height;
  const ref = useRef<SVGSVGElement>(null);
  const [hoverX, setHoverX] = useState<number | null>(null);
  const fx = formatX ?? ((x: number) => (x === 0 ? 'now' : `+${Math.round(x)}h`));

  const clean = useMemo(
    () => series
      .map((s) => ({ ...s, points: s.points.filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y)).sort((a, b) => a.x - b.x) }))
      .filter((s) => s.points.length),
    [series],
  );

  if (!clean.length) return <div className="empty-state">No data to plot yet.</div>;

  const top = yMax ?? niceMax(Math.max(...clean.flatMap((s) => s.points.map((p) => p.y))));
  const innerW = W - PAD.l - PAD.r;
  const innerH = H - PAD.t - PAD.b;
  const sx = (x: number) => PAD.l + (x / xMax) * innerW;
  const sy = (y: number) => PAD.t + innerH - (Math.min(y, top) / top) * innerH;
  const yTicks = [0, 1, 2, 3, 4].map((i) => (top / 4) * i);
  const xTicks: number[] = [];
  for (let x = 0; x <= xMax; x += xTickEvery) xTicks.push(x);

  const allX = Array.from(new Set(clean.flatMap((s) => s.points.map((p) => p.x))));
  const snapped = hoverX == null ? null : allX.reduce((best, x) => (Math.abs(x - hoverX) < Math.abs(best - hoverX) ? x : best), allX[0]);
  const tol = xMax / 12;
  const hits = snapped == null ? [] : clean
    .map((s) => {
      const p = s.points.reduce((best, q) => (Math.abs(q.x - snapped) < Math.abs(best.x - snapped) ? q : best), s.points[0]);
      return Math.abs(p.x - snapped) <= tol ? { s, p } : null;
    })
    .filter((h): h is { s: (typeof clean)[number]; p: ChartPoint } => h !== null);
  const heading = hits.find((h) => h.p.label)?.p.label ?? (snapped == null ? '' : fx(snapped));

  const onMove = (e: React.PointerEvent<SVGSVGElement>) => {
    const r = ref.current?.getBoundingClientRect();
    if (!r || !r.width) return;
    const px = ((e.clientX - r.left) / r.width) * W;
    setHoverX(Math.min(xMax, Math.max(0, ((px - PAD.l) / innerW) * xMax)));
  };

  const leftPct = snapped == null ? 0 : (sx(snapped) / W) * 100;
  const flip = leftPct > 60;

  return (
    <div style={{ position: 'relative' }}>
      <svg ref={ref} viewBox={`0 0 ${W} ${H}`} role="img" aria-label={ariaLabel} style={{ width: '100%', height: 'auto', display: 'block', touchAction: 'pan-y' }}
        onPointerMove={onMove} onPointerDown={onMove} onPointerLeave={() => setHoverX(null)}>
        {yTicks.map((t) => (
          <g key={t}>
            <line x1={PAD.l} x2={W - PAD.r} y1={sy(t)} y2={sy(t)} stroke="#2a3b48" strokeDasharray={t === 0 ? undefined : '2 5'} />
            <text x={PAD.l - 8} y={sy(t) + 4} fill="#9db5c0" fontSize="11" textAnchor="end">{Number(t.toFixed(1))}</text>
          </g>
        ))}
        {xTicks.map((t) => (
          <g key={t}>
            <line x1={sx(t)} x2={sx(t)} y1={H - PAD.b} y2={H - PAD.b + 5} stroke="#5c7181" />
            <text x={sx(t)} y={H - PAD.b + 19} fill="#9db5c0" fontSize="11" textAnchor="middle">{fx(t)}</text>
          </g>
        ))}
        <line x1={PAD.l} x2={PAD.l} y1={PAD.t} y2={H - PAD.b} stroke="#5c7181" />
        <line x1={PAD.l} x2={W - PAD.r} y1={H - PAD.b} y2={H - PAD.b} stroke="#5c7181" />
        <text x={PAD.l + innerW / 2} y={H - 8} fill="#9db5c0" fontSize="12" textAnchor="middle">{xTitle}</text>
        <text transform={`translate(14 ${PAD.t + innerH / 2}) rotate(-90)`} fill="#9db5c0" fontSize="12" textAnchor="middle">{yTitle}</text>

        {clean.map((s) => (
          <polyline key={s.name} fill="none" stroke={s.colour} strokeWidth="3" strokeLinejoin="round" strokeLinecap="round"
            strokeDasharray={s.dashed ? '8 5' : undefined} points={s.points.map((p) => `${sx(p.x)},${sy(p.y)}`).join(' ')} />
        ))}

        {snapped != null && <line x1={sx(snapped)} x2={sx(snapped)} y1={PAD.t} y2={H - PAD.b} stroke="#9db5c0" strokeOpacity="0.5" strokeDasharray="3 4" />}
        {hits.map(({ s, p }) => <circle key={s.name} cx={sx(p.x)} cy={sy(p.y)} r="5" fill="#0b1620" stroke={s.colour} strokeWidth="2.5" />)}
      </svg>

      {snapped != null && hits.length > 0 && (
        <div style={{
          position: 'absolute', top: 8, left: `${leftPct}%`, transform: flip ? 'translateX(calc(-100% - 12px))' : 'translateX(12px)',
          background: '#0b1620', border: '1px solid #2a3b48', borderRadius: 6, padding: '8px 10px', fontSize: 12, color: '#e6f1f5',
          pointerEvents: 'none', zIndex: 5, minWidth: 160, boxShadow: '0 6px 20px rgba(0,0,0,.45)',
        }}>
          <div style={{ fontWeight: 600, marginBottom: 4 }}>{heading}</div>
          {hits.map(({ s, p }) => (
            <div key={s.name} style={{ display: 'flex', alignItems: 'center', gap: 6, marginTop: 2 }}>
              <i style={{ width: 8, height: 8, borderRadius: 4, background: s.colour, display: 'inline-block' }} />
              <span style={{ flex: 1 }}>{s.name}</span>
              <b>{Number(p.y.toFixed(1))}{s.unit ? ` ${s.unit}` : ''}</b>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}