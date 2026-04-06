import { apiGet } from '../api/client'
import { EmptyState } from '../components/EmptyState'
import { PageHeader } from '../components/PageHeader'
import { StatusPill } from '../components/StatusPill'
import { useAsyncData } from '../hooks/useAsyncData'
import type { ActorSession, ContextSpaceRef, SourceConnectionStatus } from '../types/api'
import { formatLocalDateTime } from '../viewModels'

type SourcesPageProps = {
  workspace: ContextSpaceRef
  activeActor: ActorSession
}

export function SourcesPage({ workspace, activeActor }: SourcesPageProps) {
  const sources = useAsyncData<SourceConnectionStatus[]>(
    () => apiGet('/source-connections'),
    [workspace.context_space_id, activeActor.actor_id],
  )

  if (sources.error) {
    return <EmptyState title="Source status unavailable" description={sources.error} />
  }

  return (
    <section className="page-stack">
      <PageHeader
        title="Source status"
        description="This page is the operational symptom surface for current, stale, degraded, paused, and removed source states."
      />

      <div className="page-stack">
        {(sources.data ?? []).map((source) => (
          <article key={source.id} className="panel nested-panel">
            <div className="page-header compact-header source-card-header">
              <div>
                <span className="eyebrow">{source.provider}</span>
                <h3>{source.source_label}</h3>
                <p className="code-text source-locator">{source.source_boundary_locator}</p>
              </div>
              <div className="inline-meta">
                <StatusPill value={source.source_connection_state} />
                <StatusPill value={source.freshness_state} />
              </div>
            </div>
            <div className="meta-grid compact-columns source-meta-grid">
              <div>
                <span className="mini-label">Visibility</span>
                <p>{source.visibility_class}</p>
              </div>
              <div>
                <span className="mini-label">Last success</span>
                {source.last_successful_sync_at ? (
                  <time dateTime={source.last_successful_sync_at} title={source.last_successful_sync_at}>
                    {formatLocalDateTime(source.last_successful_sync_at)}
                  </time>
                ) : (
                  <p>Never</p>
                )}
              </div>
            </div>
            {source.last_error ? (
              <div className="alert-row error-text" role="alert">
                {source.last_error}
              </div>
            ) : null}
          </article>
        ))}
      </div>
    </section>
  )
}
