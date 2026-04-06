import { NavLink, Outlet } from 'react-router-dom'

import type { ActorSession, ContextSpaceRef } from '../types/api'

type LayoutProps = {
  workspace: ContextSpaceRef
  actors: ActorSession[]
  activeActor: ActorSession
  canReview: boolean
  onActorChange: (actorId: string) => void
}

const navItems = [
  { to: '/', label: 'Dashboard' },
  { to: '/glossary', label: 'Glossary' },
  { to: '/graph', label: 'Graph' },
  { to: '/decisions', label: 'Decisions' },
  { to: '/review', label: 'Review', requiresReview: true },
  { to: '/sources', label: 'Sources' },
]

export function Layout({ workspace, actors, activeActor, canReview, onActorChange }: LayoutProps) {
  const visibleNavItems = navItems.filter((item) => !item.requiresReview || canReview)

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar-top">
          <div className="brand-card">
            <div className="eyebrow">Cornerstone P0</div>
            <h1>Symptom-first workspace context</h1>
            <p>
              One canonical contract powers the UI, REST reads, and the MCP-style adapter.
            </p>
          </div>

          <div className="context-grid">
            <div className="context-card">
              <div className="context-heading">
                <span className="eyebrow">Workspace</span>
                <span className="context-meta">{workspace.context_space_kind}</span>
              </div>
              <strong className="context-value">{workspace.context_space_name}</strong>
            </div>

            <div className="context-card">
              <div className="context-heading">
                <span className="eyebrow">Persona</span>
                <span className="context-meta">{activeActor.preferred_consumer_scope}</span>
              </div>
              <strong className="context-value">{activeActor.display_name}</strong>
              <label className="persona-picker">
                <span className="muted">Switch actor</span>
                <select value={activeActor.actor_id} onChange={(event) => onActorChange(event.target.value)}>
                  {actors.map((actor) => (
                    <option key={actor.actor_id} value={actor.actor_id}>
                      {actor.display_name} ({actor.preferred_consumer_scope})
                    </option>
                  ))}
                </select>
              </label>
            </div>
          </div>
        </div>

        <nav className="nav-list" aria-label="Primary navigation">
          {visibleNavItems.map((item) => (
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
        <Outlet context={{ workspace, activeActor }} />
      </main>
    </div>
  )
}
