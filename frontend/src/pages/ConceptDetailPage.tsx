import { Link, useParams } from 'react-router-dom'

import { apiGet } from '../api/client'
import { EmptyState } from '../components/EmptyState'
import { ProvenanceStrip, RefList, SectionIntro } from '../components/experience'
import { PageHeader } from '../components/PageHeader'
import { StatusPill } from '../components/StatusPill'
import { useAsyncData } from '../hooks/useAsyncData'
import type { ActorSession, ConceptPayload, ContractEnvelope, DecisionPayload, ProvenancePayload, ResourceRef } from '../types/api'

type ConceptDetailPageProps = {
  activeActor: ActorSession
}

export function ConceptDetailPage({ activeActor }: ConceptDetailPageProps) {
  const { publicSlug } = useParams()
  const concept = useAsyncData<ContractEnvelope<ConceptPayload> | null>(
    () => (publicSlug ? apiGet(`/concepts/${publicSlug}`) : Promise.resolve(null)),
    [activeActor.actor_id, publicSlug],
  )
  const provenance = useAsyncData<ContractEnvelope<ProvenancePayload> | null>(
    () => (concept.data ? apiGet(`/provenance/concept/${concept.data.payload.concept_id}`) : Promise.resolve(null)),
    [activeActor.actor_id, concept.data?.payload.concept_id],
  )
  const decisions = useAsyncData<ContractEnvelope<DecisionPayload>[]>(
    () => apiGet('/decisions'),
    [activeActor.actor_id],
  )

  if (concept.error) {
    return <EmptyState title="Concept unavailable" description={concept.error} />
  }

  if (!concept.data) {
    return <EmptyState title="Concept is loading" description="The topic artifact will render here." />
  }

  const decisionsById = new Map((decisions.data ?? []).map((item) => [item.payload.decision_id, item.payload]))

  return (
    <section className="page-stack reader-page detail-page concept-detail-page">
      <PageHeader
        eyebrow="Cornerstone concept"
        title={concept.data.payload.canonical_name}
        actions={<Link className="ghost-link" to="/explore/topics">Back to topics</Link>}
      />

      <div className="reader-two-column detail-stage-grid">
        <article className="detail-hero detail-hero-concept">
          <span className="eyebrow">{concept.data.payload.owning_domain} topic</span>
          <h2>{concept.data.payload.canonical_name}</h2>
          <p className="artifact-lead">{concept.data.payload.definition}</p>
          {concept.data.payload.aliases.length ? (
            <div className="chip-row">
              {concept.data.payload.aliases.map((alias) => (
                <span key={alias} className="chip">
                  {alias}
                </span>
              ))}
            </div>
          ) : null}
        </article>

        <aside className="reader-secondary-panel detail-stage-rail">
          <SectionIntro
            eyebrow="Trust strip"
            title="Trust and provenance"
          />
          <ProvenanceStrip
            summary={concept.data.payload.provenance_summary}
            supportVisibility={concept.data.payload.support_visibility}
            verificationState={concept.data.payload.verification_state}
            variant="rail"
          />
          <div className="detail-stat-grid">
            <article className="list-card compact-card">
              <span className="mini-label">Visible support</span>
              <strong>{concept.data.payload.visible_support_items.length}</strong>
            </article>
            <article className="list-card compact-card">
              <span className="mini-label">Linked decisions</span>
              <strong>{concept.data.payload.linked_decision_refs.length}</strong>
            </article>
          </div>
        </aside>
      </div>

      <div className="reader-two-column detail-layout">
        <article className="reader-primary-panel detail-main-panel">
          <SectionIntro
            eyebrow="Related knowledge"
            title="Connected context"
            compact
          />

          <RefList
            title="Related decisions"
            refs={concept.data.payload.linked_decision_refs}
            resolveHref={(ref: ResourceRef) => {
              const decision = decisionsById.get(ref.resource_id)
              return decision ? `/decisions/${decision.public_slug}` : null
            }}
          />

          <RefList title="Related relations" refs={concept.data.payload.linked_relation_refs} />
        </article>

        <aside className="reader-secondary-panel detail-support-rail">
          <SectionIntro
            eyebrow="Support rail"
            title="Support and origins"
          />

          <section className="narrative-section narrative-section-panel">
            <h4>Visible support</h4>
            <div className="stack-list">
              {concept.data.payload.visible_support_items.map((item) => (
                <article key={item.support_item_id} className="list-card compact-card">
                  <div className="card-row between start">
                    <strong>{item.source_label}</strong>
                    <div className="artifact-status-row">
                      <StatusPill value={item.support_item_kind} />
                      <StatusPill value={item.visibility_class} />
                    </div>
                  </div>
                  <p>{item.excerpt_or_summary ?? 'No visible excerpt available.'}</p>
                  {item.origin_disclosure_level ? (
                    <p className="meta-copy">
                      Origin disclosure: {item.origin_disclosure_level.replaceAll('_', ' ')}
                    </p>
                  ) : null}
                </article>
              ))}
            </div>
          </section>

          {provenance.data ? (
            <>
              <section className="narrative-section narrative-section-panel">
                <h4>Source origins</h4>
                {provenance.data.payload.source_summaries.length ? (
                  <div className="stack-list">
                    {provenance.data.payload.source_summaries.map((source) => (
                      <article key={source.source_connection_id} className="list-card compact-card">
                        <strong>{source.source_label}</strong>
                        <div className="artifact-status-row">
                          <StatusPill value={source.source_connection_state} />
                          <StatusPill value={source.freshness_state} />
                        </div>
                      </article>
                    ))}
                  </div>
                ) : (
                  <p className="muted">This artifact does not expose additional source summaries in member view.</p>
                )}
              </section>
            </>
          ) : (
            <EmptyState
              title="Provenance is loading"
              description="The concept provenance summary will render here."
              eyebrow="Concept provenance"
            />
          )}
        </aside>
      </div>
    </section>
  )
}
