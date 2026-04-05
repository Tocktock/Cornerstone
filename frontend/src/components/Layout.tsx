import { NavLink, Outlet } from 'react-router-dom'

import type { ContextSpace } from '../types/api'

type LayoutProps = {
  contextSpace: ContextSpace | null
}

const navItems = [
  { to: '/', label: 'Dashboard' },
  { to: '/glossary', label: 'Glossary' },
  { to: '/graph', label: 'Graph' },
  { to: '/decisions', label: 'Decisions' },
  { to: '/artifacts', label: 'Artifacts' },
  { to: '/review', label: 'Review' },
  { to: '/sources', label: 'Sources' },
]

export function Layout({ contextSpace }: LayoutProps) {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-card">
          <div className="eyebrow">Cornerstone</div>
          <h1>The shared organizational context layer</h1>
          <p>
            Source-backed glossary, graph, and decision context for humans and AI.
          </p>
        </div>

        <div className="context-card">
          <span className="eyebrow">Context space</span>
          <strong>{contextSpace?.name ?? 'Loading…'}</strong>
          <span className="context-meta">{contextSpace?.namespace ?? 'Preparing seeded demo context'}</span>
        </div>

        <nav className="nav-list" aria-label="Primary navigation">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}
              end={item.to === '/'}
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </aside>

      <main className="content-shell">
        <Outlet />
      </main>
    </div>
  )
}
