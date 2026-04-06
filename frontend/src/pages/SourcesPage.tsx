import { apiGet } from '../api/client'
import { EmptyState } from '../components/EmptyState'
import { PageHeader } from '../components/PageHeader'
import { StatusPill } from '../components/StatusPill'
import { useAsyncData } from '../hooks/useAsyncData'
import type { ContextSpaceRef, SourceConnectionStatus } from '../types/api'

type SourcesPageProps = {
  workspace: ContextSpaceRef
}

export function SourcesPage({ workspace }: SourcesPageProps) {
  const sources = useAsyncData<SourceConnectionStatus[]>(
    () => apiGet('/source-connections'),
    [workspace.context_space_id],
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
            <div className="page-header compact-header">
              <div>
                <span className="eyebrow">{source.provider}</span>
                <h3>{source.source_label}</h3>
                <p>{source.source_boundary_locator}</p>
              </div>
              <div className="inline-meta">
                <StatusPill value={source.source_connection_state} />
                <StatusPill value={source.freshness_state} />
              </div>
            </div>
            <div className="meta-grid compact-columns">
              <div>
                <strong>Visibility</strong>
                <p>{source.visibility_class}</p>
              </div>
              <div>
                <strong>Last success</strong>
                <p>{source.last_successful_sync_at ?? 'Never'}</p>
              </div>
            </div>
            {source.last_error ? <p className="error-text">{source.last_error}</p> : null}
          </article>
        ))}
      </div>
    </section>
  )
}
