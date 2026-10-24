import { useEffect, useState, type ReactNode } from 'react';
import { Activity, Bell, ChevronDown, ChevronRight, CloudFog, Menu, MoreHorizontal, X } from 'lucide-react';
import { pages, type LiveData, type PageKey, type Notify } from '../../data/dashboard';
import { NavItem } from '../../Components/layout/NavItem';

type AppShellProps = { activePage: PageKey; onNavigate: (page: PageKey) => void; liveData: LiveData | null; liveDataError: boolean; notify: Notify; toast: string; children: ReactNode };

export function AppShell({ activePage, onNavigate, liveData, liveDataError, notify, toast, children }: AppShellProps) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const [currentTime, setCurrentTime] = useState(() => new Date());
  const current = pages.find((page) => page.id === activePage) ?? pages[0];
  const Icon = current.icon;
  const navigate = (page: PageKey) => { onNavigate(page); setMobileOpen(false); };

  useEffect(() => {
    const clock = window.setInterval(() => setCurrentTime(new Date()), 1000);
    return () => window.clearInterval(clock);
  }, []);

  const clockTime = currentTime.toLocaleTimeString('en-IN', { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false, timeZone: 'Asia/Kolkata' });
  const syncLabel = liveData ? `Last sync · ${liveData.updatedAt}` : 'Last sync · waiting for live data';
  const alertCount = liveData && liveData.current.aqi >= 301 ? 1 : 0;

  return <div className="app-shell">
    <aside className={`sidebar ${mobileOpen ? 'open' : ''}`}>
      <div className="brand">
        <div className="brand-mark"><Activity size={19} /></div>
        <div>
          <b>AERO·COMMAND</b>
          <span>DELHI NCR · v4.4</span>
        </div>
        <button className="mobile-close" onClick={() => setMobileOpen(false)}><X size={18} /></button>
      </div>

      <div className="side-label">OPERATIONS</div>
      <nav>
        {pages.slice(0, 2).map((page) => (
          <NavItem key={page.id} page={page} active={activePage === page.id} onClick={() => navigate(page.id)} />
        ))}
      </nav>

      <div className="side-label">ATMOSPHERIC INTELLIGENCE</div>
      <nav>
        {pages.slice(2, 6).map((page) => (
          <NavItem key={page.id} page={page} active={activePage === page.id} onClick={() => navigate(page.id)} />
        ))}
      </nav>

      <div className="side-label">RESPONSE</div>
      <nav>
        <NavItem page={pages[6]} active={activePage === 'alerts'} onClick={() => navigate('alerts')} alertCount={alertCount} />
      </nav>

      <div className="side-bottom">
        <div className="system-health">
          <span className="live-dot" />
          <div>
            <b>{liveDataError ? 'LIVE FEED DEGRADED' : 'LIVE FEED CONNECTED'}</b>
            <small>{syncLabel}</small>
          </div>
        </div>

        <div className="operator">
          <div className="avatar">RP</div>
          <div>
            <b>RUHI PARVEEN</b>
            <small>Operational scientist</small>
          </div>
          <MoreHorizontal size={16} />
        </div>
      </div>
    </aside>

    <main className="main-content">
      <header className="topbar">
        <button className="mobile-menu" onClick={() => setMobileOpen(true)}><Menu size={20} /></button>

        <div className="breadcrumb">
          <Icon size={15} />
          <span>{current.label}</span>
          <ChevronRight size={13} />
          <b>Live view</b>
        </div>

        <div className="top-actions">
          <span className="sync">
            <span className="live-dot" /> LIVE <span>{clockTime} IST</span>
          </span>

          <button className="icon-button" onClick={() => { navigate('alerts'); notify('Opening the response alert center'); }}>
            <Bell size={16} />
            {alertCount > 0 && <i className="notification-dot" />}
          </button>

          <button className="user-button" onClick={() => notify('Commander profile controls are available in the operations console')}>
            <div className="avatar">RP</div>
            <span>COMMANDER <ChevronDown size={13} /></span>
          </button>
        </div>
      </header>

      <div className="content">
        {liveDataError && <div className="feed-notice"><CloudFog size={14} /> Live feed unavailable — waiting for the next successful refresh.</div>}
        {children}
        {toast && <div className="toast" role="status"><span className="live-dot" />{toast}</div>}
      </div>

      <footer>
        <span><span className="live-dot" /> CAQM MOSAIC · OPERATIONAL</span>
        <span>{liveData ? `LIVE DATA · UPDATED ${liveData.updatedAt} IST` : 'LIVE DATA · WAITING'}</span>
        <span>WRF-CHEM 4.4 · 1.2 KM GRID</span>
        <span>MODEL STATUS <b>{liveDataError ? 'DEGRADED' : 'HEALTHY'}</b></span>
      </footer>
    </main>
  </div>;
}