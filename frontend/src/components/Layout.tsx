import { useEffect, useRef } from 'react'
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom'

import type { ActorSession, ContextSpaceRef, RuntimeBootstrapMeta } from '../types/api'
import { runtimeModeLabel, workspaceDataStateLabel } from '../viewModels'

type LayoutProps = {
  workspace: ContextSpaceRef
  runtimeInfo: RuntimeBootstrapMeta
  actors: ActorSession[]
  activeActor: ActorSession
  canReview: boolean
  canManageConnectors: boolean
  onActorChange: (actorId: string) => void
}

export function Layout({
  workspace,
  runtimeInfo,
  actors,
  activeActor,
  canReview,
  canManageConnectors,
  onActorChange,
}: LayoutProps) {
  const location = useLocation()
  const workspaceTrayRef = useRef<HTMLDetailsElement | null>(null)
  const isStudioRoute =
    location.pathname.startsWith('/review-studio') || location.pathname.startsWith('/source-studio')

  useEffect(() => {
    workspaceTrayRef.current?.removeAttribute('open')
  }, [activeActor.actor_id, location.pathname])

  return (
    <div className={`app-shell ${isStudioRoute ? 'studio-shell' : 'reader-shell'}`}>
      <header className="shell-header">
        <div className="shell-header-bar">
          <span className="mini-label">{isStudioRoute ? 'Operational surfaces' : 'Reader surfaces'}</span>
          <div className="shell-header-meta">
            <span className={`mode-indicator ${runtimeInfo.runtime_mode}`}>
              {runtimeModeLabel(runtimeInfo)}
            </span>
            <span className="shell-header-context">{workspace.context_space_name}</span>
          </div>
        </div>

        <div className="shell-brand">
          <span className="mini-label">Shared context</span>
          <Link className="brand-link" to="/">
            Cornerstone
          </Link>
        </div>

        <nav className="top-nav" aria-label="Primary navigation">
          <RouteLink to="/" label="Workspace" end />
          <RouteLink to="/explore/topics" label="Explore" activePrefixes={['/explore', '/concepts/', '/decisions/']} />
          {canReview ? <RouteLink to="/review-studio" label="Review Studio" /> : null}
          {canManageConnectors ? <RouteLink to="/source-studio" label="Source Studio" /> : null}
        </nav>

        <details ref={workspaceTrayRef} className="workspace-tray">
          <summary>
            <span className="mini-label">Workspace tray</span>
            <strong>{activeActor.display_name}</strong>
          </summary>
          <div className="tray-panel">
            <div className="tray-card">
              <span className="mini-label">Workspace</span>
              <strong>{workspace.context_space_name}</strong>
              <p>{workspace.context_space_kind}</p>
            </div>
            <div className="tray-card">
              <span className="mini-label">Active scope</span>
              <strong>{activeActor.preferred_consumer_scope}</strong>
              <p>{activeActor.base_role}</p>
            </div>
            <div className="tray-card">
              <span className="mini-label">Runtime</span>
              <strong>{runtimeModeLabel(runtimeInfo)}</strong>
              <p>{workspaceDataStateLabel(runtimeInfo)}</p>
            </div>
            <label className="form-field tray-field">
              <span className="mini-label">Switch actor</span>
              <select
                aria-label="Switch actor"
                value={activeActor.actor_id}
                onChange={(event) => {
                  workspaceTrayRef.current?.removeAttribute('open')
                  onActorChange(event.target.value)
                }}
              >
                {actors.map((actor) => (
                  <option key={actor.actor_id} value={actor.actor_id}>
                    {actor.display_name} ({actor.preferred_consumer_scope})
                  </option>
                ))}
              </select>
            </label>
          </div>
        </details>
      </header>

      <main className="content-shell">
        <Outlet context={{ workspace, activeActor }} />
      </main>
    </div>
  )
}

function RouteLink({
  to,
  label,
  end = false,
  activePrefixes = [],
}: {
  to: string
  label: string
  end?: boolean
  activePrefixes?: string[]
}) {
  const location = useLocation()

  return (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) => {
        const prefixMatch = activePrefixes.some((prefix) => location.pathname.startsWith(prefix))
        return `top-nav-link ${isActive || prefixMatch ? 'active' : ''}`
      }}
    >
      {label}
    </NavLink>
  )
}
