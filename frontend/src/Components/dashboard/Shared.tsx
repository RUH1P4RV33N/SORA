import { MoreHorizontal } from 'lucide-react';
import { Activity } from 'lucide-react';
import type { ReactNode } from 'react';

export function Sparkline({ color = '#31d6e5', second = false }: { color?: string; second?: boolean }) {
  return <svg className="sparkline" viewBox="0 0 280 72" preserveAspectRatio="none" aria-label="trend chart"><path d="M0 57 C24 49 35 54 55 42 S86 21 106 37 S133 52 151 33 S182 19 202 26 S229 48 250 32 S268 24 280 18" fill="none" stroke={color} strokeWidth="2.5" />{second && <path d="M0 22 C24 30 42 23 60 32 S94 48 117 39 S146 20 169 31 S197 49 220 42 S251 24 280 33" fill="none" stroke="#fb4f77" strokeWidth="1.5" strokeDasharray="4 4" />}</svg>;
}

export function MiniBars({ values, color = '#31d6e5' }: { values: number[]; color?: string }) {
  return <div className="mini-bars">{values.map((value, index) => <span key={index} style={{ height: `${value}%`, background: index === values.length - 1 ? '#fb4f77' : color }} />)}</div>;
}

export function MetricCard({ label, value, unit, detail, accent, icon: Icon }: { label: string; value: string; unit?: string; detail: string; accent: string; icon: typeof Activity }) {
  return <div className="metric-card"><div className="metric-label"><Icon size={14} /> {label}<MoreHorizontal size={15} /></div><div className="metric-value" style={{ color: accent }}>{value}<small>{unit}</small></div><div className="metric-detail">{detail}</div></div>;
}

export function PanelTitle({ kicker, title }: { kicker: string; title: string }) {
  return <div className="panel-title"><span className="panel-kicker">{kicker}</span><h2>{title}</h2></div>;
}

export function PageShell({ children, title, eyebrow }: { children: ReactNode; title: string; eyebrow: string }) {
  return <div aria-label={`${eyebrow} — ${title}`}>{children}</div>;
}
