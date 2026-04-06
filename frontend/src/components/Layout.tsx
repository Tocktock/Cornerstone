import { NavLink, Outlet } from 'react-router-dom'

import type { ActorSession, ContextSpaceRef } from '../types/api'

type LayoutProps = {
  workspace: ContextSpaceRef
  actors: ActorSession[]
  activeActor: ActorSession
  onActorChange: (actorId: string) => void
}

const navItems = [
  { to: '/', label: 'Dashboard' },
  { to: '/glossary', label: 'Glossary' },
  { to: '/graph', label: 'Graph' },
  { to: '/decisions', label: 'Decisions' },
  { to: '/review', label: 'Review' },
  { to: '/sources', label: 'Sources' },
]

export function Layout({ workspace, actors, activeActor, onActorChange }: LayoutProps) {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-card">
          <div className="eyebrow">Cornerstone P0</div>
          <h1>Symptom-first workspace context</h1>
          <p>
            One canonical contract powers the UI, REST reads, and the MCP-style adapter.
          </p>
        </div>

        <div className="context-card">
          <span className="eyebrow">Workspace</span>
          <strong>{workspace.context_space_name}</strong>
          <span className="context-meta">{workspace.context_space_kind}</span>
        </div>

        <div className="context-card">
          <span className="eyebrow">Persona</span>
          <strong>{activeActor.display_name}</strong>
          <span className="context-meta">{activeActor.preferred_consumer_scope}</span>
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
        <Outlet context={{ workspace, activeActor }} />
      </main>
    </div>
  )
}
