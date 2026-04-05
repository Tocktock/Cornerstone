import { apiGet } from '../api/client'
import { EmptyState } from '../components/EmptyState'
import { PageHeader } from '../components/PageHeader'
import { StatusPill } from '../components/StatusPill'
import { useAsyncData } from '../hooks/useAsyncData'
import type { Decision } from '../types/api'

type DecisionsPageProps = {
  contextSpaceId: string | null
}

export function DecisionsPage({ contextSpaceId }: DecisionsPageProps) {
  const decisions = useAsyncData<Decision[]>(
    () => (contextSpaceId ? apiGet('/decisions', { context_space_id: contextSpaceId }) : Promise.resolve([])),
    [contextSpaceId],
  )

  if (!contextSpaceId) {
    return <EmptyState title="No decisions" description="Decision records will appear once seeded data is ready." />
  }

  return (
    <div className="page-stack">
      <PageHeader
        title="Decision records"
        description="Reviewable organizational rationale connected to concepts, relations, and evidence."
      />

      <div className="stack-list">
        {(decisions.data ?? []).map((decision) => (
          <article key={decision.id} className="panel list-card large-card">
            <div className="card-row between start">
              <div>
                <span className="eyebrow">Decision record</span>
                <h3>{decision.title}</h3>
              </div>
              <StatusPill value={decision.status} />
            </div>
            <div className="detail-grid">
              <div>
                <strong>Problem</strong>
                <p>{decision.problem}</p>
              </div>
              <div>
                <strong>Decision</strong>
                <p>{decision.decision}</p>
              </div>
              <div>
                <strong>Rationale</strong>
                <p>{decision.rationale}</p>
              </div>
            </div>
            <div className="chip-row">
              {decision.concepts.map((concept) => (
                <span key={concept} className="chip">
                  {concept}
                </span>
              ))}
            </div>
            <div className="two-column-layout compact-columns">
              <div>
                <strong>Constraints</strong>
                <ul className="dense-list">
                  {decision.constraints.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
              <div>
                <strong>Impact</strong>
                <ul className="dense-list">
                  {decision.impact.map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
            </div>
          </article>
        ))}
      </div>
    </div>
  )
}
