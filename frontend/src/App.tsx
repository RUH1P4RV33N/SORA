import { useState } from 'react';
import { AppShell } from './Components/layout/AppShell';
import type { Notify, PageKey } from './data/dashboard';
import { useLiveData } from './hooks/useLiveData';
import { AlertsPage, EnsemblePage, ForecastPage, InversionPage, Overview, SmokePage } from './pages/DashboardPages';
import { PhotochemistryPage } from './pages/PhotochemistryPage';

function App() {
  const [activePage, setActivePage] = useState<PageKey>('overview');
  const [toast, setToast] = useState('');
  const { data: liveData, error: liveDataError } = useLiveData();
  const notify: Notify = (message) => { setToast(message); window.setTimeout(() => setToast(''), 2800); };

  const pageProps = { liveData, notify };
  const content = activePage === 'overview' ? <Overview setPage={setActivePage} {...pageProps} />
    : activePage === 'forecast' ? <ForecastPage {...pageProps} />
      : activePage === 'inversion' ? <InversionPage liveData={liveData} />
        : activePage === 'smoke' ? <SmokePage notify={notify} />
          : activePage === 'ozone' ? <PhotochemistryPage liveData={liveData} />
            : activePage === 'ensemble' ? <EnsemblePage notify={notify} liveData={liveData} />
              : <AlertsPage notify={notify} liveData={liveData} />;

  return <AppShell activePage={activePage} onNavigate={setActivePage} liveData={liveData} liveDataError={liveDataError} notify={notify} toast={toast}>{content}</AppShell>;
}

export default App;
