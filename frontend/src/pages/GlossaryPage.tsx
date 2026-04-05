import { apiGet } from '../api/client'
import { EmptyState } from '../components/EmptyState'
import { PageHeader } from '../components/PageHeader'
import { StatusPill } from '../components/StatusPill'
import { useAsyncData } from '../hooks/useAsyncData'
import type { Concept } from '../types/api'

type GlossaryPageProps = {
  contextSpaceId: string | null
}

export function GlossaryPage({ contextSpaceId }: GlossaryPageProps) {
  const concepts = useAsyncData<Concept[]>(
    () => (contextSpaceId ? apiGet('/concepts', { context_space_id: contextSpaceId }) : Promise.resolve([])),
    [contextSpaceId],
  )

  if (!contextSpaceId) {
    return <EmptyState title="No context space" description="Wait for the backend to finish bootstrapping." />
  }

  return (
    <div className="page-stack">
      <PageHeader
        title="Glossary"
        description="Canonical concepts with definitions, aliases, evidence, and decision lineage."
      />

      <div className="card-grid">
        {(concepts.data ?? []).map((concept) => (
          <article key={concept.id} className="panel concept-card">
            <div className="card-row between start">
              <div>
                <span className="eyebrow">{concept.concept_type}</span>
                <h3>{concept.canonical_name}</h3>
              </div>
              <StatusPill value={concept.status} />
            </div>
            <p>{concept.definition}</p>
            <div className="chip-row">
              {concept.aliases.map((alias) => (
                <span key={alias} className="chip subtle">
                  {alias}
                </span>
              ))}
            </div>
            <div className="meta-grid">
              <div>
                <strong>Evidence</strong>
                <span>{concept.evidence.length}</span>
              </div>
              <div>
                <strong>Decisions</strong>
                <span>{concept.linked_decisions.length}</span>
              </div>
            </div>
            <div className="compact-list">
              {concept.evidence.slice(0, 2).map((evidence) => (
                <article key={evidence.id} className="evidence-card">
                  <strong>{evidence.artifact_title}</strong>
                  <p>{evidence.excerpt}</p>
                </article>
              ))}
            </div>
          </article>
        ))}
      </div>
    </div>
  )
}
