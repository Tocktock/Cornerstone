import { useState } from 'react'

import { apiGet, apiPost } from '../api/client'
import { EmptyState } from '../components/EmptyState'
import { AlertBanner, SectionIntro } from '../components/experience'
import { PageHeader } from '../components/PageHeader'
import { StatCard } from '../components/StatCard'
import { StatusPill } from '../components/StatusPill'
import { useAsyncData } from '../hooks/useAsyncData'
import type { ActorSession, ReviewQueueItem } from '../types/api'

type ReviewStudioPageProps = {
  activeActor: ActorSession
  actors: ActorSession[]
  canReview: boolean
}

export function ReviewStudioPage({ activeActor, actors, canReview }: ReviewStudioPageProps) {
  const [refreshKey, setRefreshKey] = useState(0)
  const [actionError, setActionError] = useState<string | null>(null)
  const [actionMessage, setActionMessage] = useState<string | null>(null)
  const queue = useAsyncData<ReviewQueueItem[] | null>(
    () => (canReview ? apiGet('/review-queue') : Promise.resolve(null)),
    [activeActor.actor_id, refreshKey, canReview],
  )

  const reviewActors = actors.filter(
    (actor) => actor.preferred_consumer_scope === 'review' || actor.base_role.toLowerCase() === 'admin',
  )
  const queueItems = queue.data ?? []
  const reviewRequiredCount = queueItems.filter((item) => item.lifecycle_state === 'review_required').length
  const reviewDomains = new Set(queueItems.map((item) => item.review_domain)).size
  const leadItem = queueItems[0] ?? null
  const remainingItems = leadItem ? queueItems.slice(1) : []

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
    return <EmptyState title="Review Studio unavailable" description={queue.error} />
  }

  return (
    <section className="page-stack studio-page review-studio-page">
      <PageHeader
        eyebrow="Cornerstone studio"
        title="Review Studio"
      />

      {!canReview ? (
        <EmptyState
          title="Review access required"
          description={`This actor cannot review shared objects. Use Switch actor to choose ${
            reviewActors.length ? reviewActors.map((actor) => actor.display_name).join(' or ') : 'a review-capable actor'
          }.`}
        />
      ) : null}

      {actionError ? <AlertBanner tone="danger" title="Review action failed" description={actionError} /> : null}
      {actionMessage ? <AlertBanner title="Review action completed" description={actionMessage} /> : null}

      {canReview && queueItems.length ? (
        <>
          <div className="summary-strip studio-summary-strip">
            <StatCard
              label="Queued items"
              value={queueItems.length}
              helper={`${reviewRequiredCount} review required · ${queueItems.length - reviewRequiredCount} ready to move.`}
              tone="info"
            />
            <StatCard
              label="Review domains"
              value={reviewDomains}
              helper="Queue grouping stays explicit even when visual polish increases."
              tone="sage"
            />
          </div>

          {leadItem ? (
            <div className="studio-two-column studio-stage-grid">
              <article className="studio-panel review-item-card studio-lead-card">
                <SectionIntro
                  eyebrow={leadItem.review_domain}
                  title={leadItem.resource_ref.resource_label}
                  actions={
                    <div className="artifact-status-row">
                      <StatusPill value={leadItem.resource_ref.resource_kind} />
                      <StatusPill value={leadItem.lifecycle_state} />
                      <StatusPill value={leadItem.verification_state} />
                    </div>
                  }
                />

                <div className="studio-meta-grid">
                  <div>
                    <span className="mini-label">Review domain</span>
                    <p>{leadItem.review_domain}</p>
                  </div>
                  <div>
                    <span className="mini-label">Support disclosure</span>
                    <p>{leadItem.support_visibility.replaceAll('_', ' ')}</p>
                  </div>
                </div>

                <div className="studio-action-row">
                  <button type="button" onClick={() => runReview(leadItem, 'officialize')}>
                    Officialize
                  </button>
                  <button type="button" className="destructive-button" onClick={() => runReview(leadItem, 'reject')}>
                    Reject
                  </button>
                </div>
              </article>

              <aside className="studio-panel studio-guidance-panel">
                <SectionIntro
                  eyebrow="Queue guidance"
                  title="Review posture"
                  compact
                />
                <div className="stack-list">
                  <article className="list-card compact-card">
                    <span className="mini-label">Action safety</span>
                    <p>Keep the next action obvious. Officialize and reject never share equal visual weight.</p>
                  </article>
                </div>
              </aside>
            </div>
          ) : null}

          {remainingItems.length ? (
            <section className="reader-section">
              <SectionIntro
                eyebrow="Remaining queue"
                title="Continue through the review stack"
              />
              <div className="studio-grid">
                {remainingItems.map((item) => (
                  <article key={item.resource_ref.resource_id} className="studio-panel review-item-card">
                    <div className="panel-heading panel-heading-start">
                      <div>
                        <span className="eyebrow">{item.review_domain}</span>
                        <h3>{item.resource_ref.resource_label}</h3>
                      </div>
                      <div className="artifact-status-row">
                        <StatusPill value={item.resource_ref.resource_kind} />
                        <StatusPill value={item.lifecycle_state} />
                        <StatusPill value={item.verification_state} />
                      </div>
                    </div>

                    <div className="studio-meta-grid">
                      <div>
                        <span className="mini-label">Review domain</span>
                        <p>{item.review_domain}</p>
                      </div>
                      <div>
                        <span className="mini-label">Support disclosure</span>
                        <p>{item.support_visibility.replaceAll('_', ' ')}</p>
                      </div>
                    </div>

                    <div className="studio-action-row">
                      <button type="button" onClick={() => runReview(item, 'officialize')}>
                        Officialize
                      </button>
                      <button type="button" className="destructive-button" onClick={() => runReview(item, 'reject')}>
                        Reject
                      </button>
                    </div>
                  </article>
                ))}
              </div>
            </section>
          ) : null}
        </>
      ) : canReview ? (
        <EmptyState
          title="Queue is clear"
          description="No draft or review-required shared objects are waiting right now."
          eyebrow="Review queue"
        />
      ) : null}
    </section>
  )
}
