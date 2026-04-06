import { useEffect, useState } from 'react'

import { apiGet } from '../api/client'
import { EmptyState } from '../components/EmptyState'
import { PageHeader } from '../components/PageHeader'
import { StatusPill } from '../components/StatusPill'
import { useAsyncData } from '../hooks/useAsyncData'
import type { ActorSession, ContextSpaceRef, ContractEnvelope, DecisionPayload, ProvenancePayload } from '../types/api'

type DecisionsPageProps = {
  workspace: ContextSpaceRef
  activeActor: ActorSession
}

export function DecisionsPage({ workspace, activeActor }: DecisionsPageProps) {
  const decisions = useAsyncData<ContractEnvelope<DecisionPayload>[]>(
    () => apiGet('/decisions'),
    [workspace.context_space_id, activeActor.actor_id],
  )
  const [selectedDecisionId, setSelectedDecisionId] = useState<string | null>(null)
  const provenance = useAsyncData<ContractEnvelope<ProvenancePayload> | null>(
    () => (selectedDecisionId ? apiGet(`/provenance/decision/${selectedDecisionId}`) : Promise.resolve(null)),
    [selectedDecisionId, activeActor.actor_id],
  )

  useEffect(() => {
    if (!selectedDecisionId && decisions.data?.[0]) {
      setSelectedDecisionId(decisions.data[0].payload.decision_id)
    }
  }, [decisions.data, selectedDecisionId])

  return (
    <section className="page-stack decisions-page">
      <PageHeader
        title="Decisions"
        description="Supersession lineage stays visible, and the detail cards preserve the canonical decision payload."
      />

      {decisions.error ? <EmptyState title="Decisions unavailable" description={decisions.error} /> : null}

      <div className="two-column-layout master-detail-layout">
        <div className="page-stack master-list">
          {(decisions.data ?? []).map((envelope) => (
            <button
              key={envelope.payload.decision_id}
              type="button"
              className={`panel selectable-card decision-card ${selectedDecisionId === envelope.payload.decision_id ? 'selected' : ''}`}
              onClick={() => setSelectedDecisionId(envelope.payload.decision_id)}
            >
              <span className="eyebrow">{envelope.payload.owning_domain}</span>
              <h3>{envelope.payload.title}</h3>
              <div className="inline-meta">
                <StatusPill value={envelope.payload.lifecycle_state} />
                <StatusPill value={envelope.payload.support_visibility} />
              </div>
              <p>{envelope.payload.decision_statement}</p>
              {envelope.payload.superseded_by_ref ? (
                <div className="chip-row lineage-row">
                  <span className="chip subtle">Superseded by</span>
                  <span className="chip">{envelope.payload.superseded_by_ref.resource_label}</span>
                </div>
              ) : null}
            </button>
          ))}
        </div>

        <article className="panel detail-pane mobile-priority decisions-detail-panel">
          {provenance.data ? (
            <>
              <span className="eyebrow">Decision provenance</span>
              <h3>{provenance.data.payload.subject_ref.resource_label}</h3>
              <p className="muted">
                Support items: {provenance.data.payload.provenance_summary.support_item_count} · Promotion lineage:{' '}
                {String(provenance.data.payload.provenance_summary.promotion_lineage_present)}
              </p>
              <ul className="stack-list">
                {provenance.data.payload.support_items.map((item) => (
                  <li key={item.support_item_id} className="list-card">
                    <strong>{item.source_label}</strong>
                    <p>{item.excerpt_or_summary ?? 'No excerpt available.'}</p>
                    {item.origin_disclosure_level ? (
                      <p className="muted">Origin disclosure: {item.origin_disclosure_level}</p>
                    ) : null}
                  </li>
                ))}
              </ul>
            </>
          ) : (
            <EmptyState title="Select a decision" description="Decision provenance will appear here." />
          )}
        </article>
      </div>
    </section>
  )
}
