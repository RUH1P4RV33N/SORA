import { ChevronRight } from 'lucide-react';
import type { Page } from '../../data/dashboard';

export function NavItem({ page, active, onClick, alertCount = 0 }: { page: Page; active: boolean; onClick: () => void; alertCount?: number }) {
  const Icon = page.icon;
  return <button className={`nav-item ${active ? 'active' : ''}`} onClick={onClick}><Icon size={16} /><span>{page.label}</span>{page.id === 'alerts' && alertCount > 0 && <i className="nav-alert">{alertCount}</i>}{active && <ChevronRight className="nav-arrow" size={14} />}</button>;
}
