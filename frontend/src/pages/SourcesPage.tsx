import { useState } from 'react'

import { apiGet, apiPost } from '../api/client'
import { EmptyState } from '../components/EmptyState'
import { PageHeader } from '../components/PageHeader'
import { StatusPill } from '../components/StatusPill'
import { useAsyncData } from '../hooks/useAsyncData'
import type { SourceConnection } from '../types/api'

type SourcesPageProps = {
  contextSpaceId: string | null
}

export function SourcesPage({ contextSpaceId }: SourcesPageProps) {
  const [isSyncingId, setIsSyncingId] = useState<string | null>(null)
  const sources = useAsyncData<SourceConnection[]>(
    () => (contextSpaceId ? apiGet('/source-connections', { context_space_id: contextSpaceId }) : Promise.resolve([])),
    [contextSpaceId],
  )

  async function syncConnection(connectionId: string) {
    setIsSyncingId(connectionId)
    try {
      await apiPost(`/source-connections/${connectionId}/sync`)
      const refreshed = await apiGet<SourceConnection[]>('/source-connections', {
        context_space_id: contextSpaceId ?? undefined,
      })
      sources.setData(refreshed)
    } finally {
      setIsSyncingId(null)
    }
  }

  if (!contextSpaceId) {
    return <EmptyState title="No sources yet" description="Wait for the backend bootstrap process to finish." />
  }

  return (
    <div className="page-stack">
      <PageHeader
        title="Source connections"
        description="Connector health, eventual sync state, and scope for source-backed ingestion."
      />

      <div className="stack-list">
        {(sources.data ?? []).map((source) => (
          <article key={source.id} className="panel list-card">
            <div className="card-row between start">
              <div>
                <span className="eyebrow">{source.provider}</span>
                <h3>{source.external_scope}</h3>
              </div>
              <StatusPill value={source.health_status} />
            </div>
            <div className="meta-grid">
              <div>
                <strong>Sync mode</strong>
                <span>{source.sync_mode}</span>
              </div>
              <div>
                <strong>Interval</strong>
                <span>{source.sync_interval_seconds}s</span>
              </div>
              <div>
                <strong>Last synced</strong>
                <span>{source.last_synced_at ?? 'Not synced yet'}</span>
              </div>
            </div>
            <div className="button-row">
              <button onClick={() => syncConnection(source.id)} disabled={isSyncingId === source.id}>
                {isSyncingId === source.id ? 'Syncing…' : 'Run sync'}
              </button>
            </div>
          </article>
        ))}
      </div>
    </div>
  )
}
