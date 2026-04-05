import { useMemo, useState } from 'react'

import { apiGet, apiPost } from '../api/client'
import { EmptyState } from '../components/EmptyState'
import { PageHeader } from '../components/PageHeader'
import { StatusPill } from '../components/StatusPill'
import { useAsyncData } from '../hooks/useAsyncData'
import type { Concept, Decision, Relation } from '../types/api'

type ReviewPageProps = {
  contextSpaceId: string | null
  reviewerId: string | null
}

type ReviewBundle = {
  concepts: Concept[]
  relations: Relation[]
  decisions: Decision[]
}

export function ReviewPage({ contextSpaceId, reviewerId }: ReviewPageProps) {
  const [busyKey, setBusyKey] = useState<string | null>(null)
  const reviewData = useAsyncData<ReviewBundle>(
    async () => {
      const [concepts, relations, decisions] = await Promise.all([
        contextSpaceId ? apiGet<Concept[]>('/concepts', { context_space_id: contextSpaceId }) : Promise.resolve([]),
        contextSpaceId ? apiGet<Relation[]>('/relations', { context_space_id: contextSpaceId }) : Promise.resolve([]),
        contextSpaceId ? apiGet<Decision[]>('/decisions', { context_space_id: contextSpaceId }) : Promise.resolve([]),
      ])
      return { concepts, relations, decisions }
    },
    [contextSpaceId],
  )

  const draftConcepts = useMemo(
    () => (reviewData.data?.concepts ?? []).filter((item) => item.status === 'DRAFT'),
    [reviewData.data],
  )
  const draftRelations = useMemo(
    () => (reviewData.data?.relations ?? []).filter((item) => item.status === 'DRAFT'),
    [reviewData.data],
  )
  const proposedDecisions = useMemo(
    () => (reviewData.data?.decisions ?? []).filter((item) => item.status === 'PROPOSED'),
    [reviewData.data],
  )

  async function reviewItem(kind: 'concepts' | 'relations' | 'decisions', id: string, action: string) {
    if (!reviewerId) {
      return
    }
    setBusyKey(`${kind}-${id}-${action}`)
    try {
      const refreshed = await apiPost<Concept | Relation | Decision>(`/${kind}/${id}/review`, {
        actor_id: reviewerId,
        action,
      })
      reviewData.setData((current) => {
        if (!current) return current
        const updated = { ...current }
        if (kind === 'concepts') {
          updated.concepts = current.concepts.map((item) => (item.id === id ? (refreshed as Concept) : item))
        }
        if (kind === 'relations') {
          updated.relations = current.relations.map((item) => (item.id === id ? (refreshed as Relation) : item))
        }
        if (kind === 'decisions') {
          updated.decisions = current.decisions.map((item) => (item.id === id ? (refreshed as Decision) : item))
        }
        return updated
      })
    } catch (error) {
      window.alert(error instanceof Error ? error.message : 'Review action failed.')
    } finally {
      setBusyKey(null)
    }
  }

  if (!contextSpaceId) {
    return <EmptyState title="No review queue" description="Reviewable items appear after the first sync." />
  }

  return (
    <div className="page-stack">
      <PageHeader
        title="Review queue"
        description="Human approval is the gate for official concepts, relations, and decision records."
      />

      <section className="review-grid">
        <article className="panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">Concepts</span>
              <h3>Draft glossary entries</h3>
            </div>
          </div>
          <div className="stack-list">
            {draftConcepts.map((concept) => (
              <article key={concept.id} className="list-card">
                <div className="card-row between">
                  <strong>{concept.canonical_name}</strong>
                  <StatusPill value={concept.status} />
                </div>
                <p>{concept.definition}</p>
                <div className="button-row">
                  <button
                    onClick={() => reviewItem('concepts', concept.id, 'approve')}
                    disabled={busyKey === `concepts-${concept.id}-approve`}
                  >
                    Approve
                  </button>
                  <button className="ghost-button" onClick={() => reviewItem('concepts', concept.id, 'reject')}>
                    Reject
                  </button>
                </div>
              </article>
            ))}
          </div>
        </article>

        <article className="panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">Relations</span>
              <h3>Draft graph assertions</h3>
            </div>
          </div>
          <div className="stack-list">
            {draftRelations.map((relation) => (
              <article key={relation.id} className="list-card relation-card">
                <div className="card-row between">
                  <div className="relation-line">
                    <span>{relation.subject_name}</span>
                    <span className="relation-predicate">{relation.predicate}</span>
                    <span>{relation.object_name}</span>
                  </div>
                  <StatusPill value={relation.status} />
                </div>
                <p>{relation.description}</p>
                <div className="button-row">
                  <button onClick={() => reviewItem('relations', relation.id, 'approve')}>Approve</button>
                  <button className="ghost-button" onClick={() => reviewItem('relations', relation.id, 'reject')}>
                    Reject
                  </button>
                </div>
              </article>
            ))}
          </div>
        </article>
      </section>

      <section className="panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">Decisions</span>
            <h3>Proposed decision records</h3>
          </div>
        </div>
        <div className="stack-list">
          {proposedDecisions.map((decision) => (
            <article key={decision.id} className="list-card">
              <div className="card-row between">
                <strong>{decision.title}</strong>
                <StatusPill value={decision.status} />
              </div>
              <p>{decision.decision}</p>
              <div className="button-row">
                <button onClick={() => reviewItem('decisions', decision.id, 'approve')}>Approve</button>
                <button className="ghost-button" onClick={() => reviewItem('decisions', decision.id, 'reject')}>
                  Reject
                </button>
              </div>
            </article>
          ))}
        </div>
      </section>
    </div>
  )
}
