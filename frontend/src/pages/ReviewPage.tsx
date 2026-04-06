import { useState } from 'react'

import { apiGet, apiPost } from '../api/client'
import { EmptyState } from '../components/EmptyState'
import { PageHeader } from '../components/PageHeader'
import { StatusPill } from '../components/StatusPill'
import { useAsyncData } from '../hooks/useAsyncData'
import type { ActorSession, ContextSpaceRef, ReviewQueueItem } from '../types/api'

type ReviewPageProps = {
  workspace: ContextSpaceRef
  activeActor: ActorSession
}

export function ReviewPage({ workspace, activeActor }: ReviewPageProps) {
  const [refreshKey, setRefreshKey] = useState(0)
  const [actionError, setActionError] = useState<string | null>(null)
  const [actionMessage, setActionMessage] = useState<string | null>(null)
  const queue = useAsyncData<ReviewQueueItem[]>(
    () => apiGet('/review-queue'),
    [workspace.context_space_id, activeActor.actor_id, refreshKey],
  )

  async function runReview(item: ReviewQueueItem, action: 'officialize' | 'reject') {
    setActionError(null)
    setActionMessage(null)
    const basePath =
      item.resource_ref.resource_kind === 'concept'
        ? '/concepts'
        : item.resource_ref.resource_kind === 'relation'
          ? '/relations'
          : '/decisions'

    try {
      await apiPost(`${basePath}/${item.resource_ref.resource_id}/review`, { action })
      setActionMessage(`${action} succeeded for ${item.resource_ref.resource_label}.`)
      setRefreshKey((value) => value + 1)
    } catch (error) {
      setActionError(error instanceof Error ? error.message : 'Review action failed.')
    }
  }

  if (queue.error) {
    return <EmptyState title="Review scope unavailable" description={queue.error} />
  }

  return (
    <section className="page-stack">
      <PageHeader
        title="Review queue"
        description={`Acting as ${activeActor.display_name}. Cross-domain relations should reject domain-scoped review and require workspace review instead.`}
      />

      {actionError ? <EmptyState title="Review action failed" description={actionError} /> : null}
      {actionMessage ? <div className="panel success-banner">{actionMessage}</div> : null}

      {queue.data?.length ? (
        <div className="page-stack">
          {queue.data.map((item) => (
            <article key={item.resource_ref.resource_id} className="panel nested-panel">
              <span className="eyebrow">{item.review_domain}</span>
              <h3>{item.resource_ref.resource_label}</h3>
              <div className="inline-meta">
                <StatusPill value={item.resource_ref.resource_kind} />
                <StatusPill value={item.lifecycle_state} />
                <StatusPill value={item.verification_state} />
              </div>
              <p className="muted">Support disclosure: {item.support_visibility}</p>
              <div className="button-row">
                <button type="button" onClick={() => runReview(item, 'officialize')}>
                  Officialize
                </button>
                <button type="button" className="ghost-button" onClick={() => runReview(item, 'reject')}>
                  Reject
                </button>
              </div>
            </article>
          ))}
        </div>
      ) : (
        <EmptyState title="Queue is clear" description="No draft or review-required shared objects are waiting right now." />
      )}
    </section>
  )
}
