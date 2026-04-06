import { useEffect, useState } from 'react'

import { apiGet } from '../api/client'
import { EmptyState } from '../components/EmptyState'
import { PageHeader } from '../components/PageHeader'
import { StatusPill } from '../components/StatusPill'
import { useAsyncData } from '../hooks/useAsyncData'
import type { ActorSession, ConceptPayload, ContextSpaceRef, ContractEnvelope, ProvenancePayload } from '../types/api'

type GlossaryPageProps = {
  workspace: ContextSpaceRef
  activeActor: ActorSession
}

export function GlossaryPage({ workspace, activeActor }: GlossaryPageProps) {
  const concepts = useAsyncData<ContractEnvelope<ConceptPayload>[]>(
    () => apiGet('/concepts'),
    [workspace.context_space_id, activeActor.actor_id],
  )
  const [selectedConceptId, setSelectedConceptId] = useState<string | null>(null)
  const provenance = useAsyncData<ContractEnvelope<ProvenancePayload> | null>(
    () => (selectedConceptId ? apiGet(`/provenance/concept/${selectedConceptId}`) : Promise.resolve(null)),
    [selectedConceptId, activeActor.actor_id],
  )

  useEffect(() => {
    if (!selectedConceptId && concepts.data?.[0]) {
      setSelectedConceptId(concepts.data[0].payload.concept_id)
    }
  }, [concepts.data, selectedConceptId])

  return (
    <section className="page-stack glossary-page">
      <PageHeader
        title="Glossary"
        description="Each concept view comes straight from the canonical concept envelope, including trust and provenance disclosure."
      />

      {concepts.error ? <EmptyState title="Concepts unavailable" description={concepts.error} /> : null}

      <div className="two-column-layout master-detail-layout">
        <div className="page-stack master-list">
          {(concepts.data ?? []).map((envelope) => (
            <button
              key={envelope.payload.concept_id}
              type="button"
              className={`panel selectable-card glossary-card ${selectedConceptId === envelope.payload.concept_id ? 'selected' : ''}`}
              onClick={() => setSelectedConceptId(envelope.payload.concept_id)}
            >
              <span className="eyebrow">{envelope.payload.owning_domain}</span>
              <h3>{envelope.payload.canonical_name}</h3>
              <div className="inline-meta">
                <StatusPill value={envelope.payload.support_visibility} />
                <StatusPill value={envelope.payload.verification_state} />
              </div>
              <p>{envelope.payload.definition}</p>
            </button>
          ))}
        </div>

        <article className="panel detail-pane mobile-priority glossary-detail-panel">
          {provenance.data ? (
            <>
              <span className="eyebrow">Provenance</span>
              <h3>{provenance.data.payload.subject_ref.resource_label}</h3>
              <p className="muted">
                Support items: {provenance.data.payload.provenance_summary.support_item_count} · Visible:{' '}
                {provenance.data.payload.provenance_summary.visible_support_item_count}
              </p>
              <ul className="stack-list">
                {provenance.data.payload.support_items.map((item) => (
                  <li key={item.support_item_id} className="list-card">
                    <strong>{item.source_label}</strong>
                    <p>{item.excerpt_or_summary ?? 'No visible excerpt available.'}</p>
                    <div className="inline-meta">
                      <StatusPill value={item.support_item_kind} />
                      <StatusPill value={item.visibility_class} />
                      {item.origin_disclosure_level ? <StatusPill value={item.origin_disclosure_level} /> : null}
                    </div>
                  </li>
                ))}
              </ul>
              <div className="card-grid">
                {provenance.data.payload.source_summaries.map((source) => (
                  <article key={source.source_connection_id} className="panel nested-panel">
                    <strong>{source.source_label}</strong>
                    <div className="inline-meta">
                      <StatusPill value={source.source_connection_state} />
                      <StatusPill value={source.freshness_state} />
                    </div>
                  </article>
                ))}
              </div>
            </>
          ) : (
            <EmptyState title="Select a concept" description="Concept provenance will render here." />
          )}
        </article>
      </div>
    </section>
  )
}
