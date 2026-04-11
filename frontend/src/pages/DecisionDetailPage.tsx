import { Link, useParams } from 'react-router-dom'

import { apiGet } from '../api/client'
import { EmptyState } from '../components/EmptyState'
import { LineageRail, ProvenanceStrip, RefList, SectionIntro } from '../components/experience'
import { PageHeader } from '../components/PageHeader'
import { StatusPill } from '../components/StatusPill'
import { useAsyncData } from '../hooks/useAsyncData'
import type { ActorSession, ConceptPayload, ContractEnvelope, DecisionPayload, ProvenancePayload, ResourceRef } from '../types/api'

type DecisionDetailPageProps = {
  activeActor: ActorSession
}

export function DecisionDetailPage({ activeActor }: DecisionDetailPageProps) {
  const { publicSlug } = useParams()
  const decision = useAsyncData<ContractEnvelope<DecisionPayload> | null>(
    () => (publicSlug ? apiGet(`/decisions/${publicSlug}`) : Promise.resolve(null)),
    [activeActor.actor_id, publicSlug],
  )
  const provenance = useAsyncData<ContractEnvelope<ProvenancePayload> | null>(
    () => (decision.data ? apiGet(`/provenance/decision/${decision.data.payload.decision_id}`) : Promise.resolve(null)),
    [activeActor.actor_id, decision.data?.payload.decision_id],
  )
  const concepts = useAsyncData<ContractEnvelope<ConceptPayload>[]>(
    () => apiGet('/concepts'),
    [activeActor.actor_id],
  )
  const decisions = useAsyncData<ContractEnvelope<DecisionPayload>[]>(
    () => apiGet('/decisions'),
    [activeActor.actor_id],
  )

  if (decision.error) {
    return <EmptyState title="Decision unavailable" description={decision.error} />
  }

  if (!decision.data) {
    return <EmptyState title="Decision is loading" description="The decision artifact will render here." />
  }

  const conceptsById = new Map((concepts.data ?? []).map((item) => [item.payload.concept_id, item.payload]))
  const decisionsById = new Map((decisions.data ?? []).map((item) => [item.payload.decision_id, item.payload]))
  const supersedes = decision.data.payload.supersedes_ref
    ? decisionsById.get(decision.data.payload.supersedes_ref.resource_id)
    : null
  const supersededBy = decision.data.payload.superseded_by_ref
    ? decisionsById.get(decision.data.payload.superseded_by_ref.resource_id)
    : null

  return (
    <section className="page-stack reader-page detail-page decision-detail-page">
      <PageHeader
        eyebrow="Cornerstone decision"
        title={decision.data.payload.title}
        description="Presentable reader view for a canonical decision record."
        actions={<Link className="ghost-link" to="/explore/decisions">Back to decisions</Link>}
      />

      <div className="reader-two-column detail-stage-grid">
        <article className="detail-hero detail-hero-decision">
          <span className="eyebrow">{decision.data.payload.owning_domain} decision</span>
          <h2>{decision.data.payload.title}</h2>
          <p className="artifact-lead">{decision.data.payload.decision_statement}</p>
        </article>

        <aside className="reader-secondary-panel detail-stage-rail">
          <SectionIntro
            eyebrow="Lineage and trust"
            title="Decision posture"
            description="Lineage, provenance, and support state remain visible without crowding the story order."
          />
          <LineageRail
            variant="timeline"
            previous={
              supersedes
                ? { label: supersedes.title, to: `/decisions/${supersedes.public_slug}` }
                : decision.data.payload.supersedes_ref
                  ? { label: decision.data.payload.supersedes_ref.resource_label, to: null }
                  : null
            }
            next={
              supersededBy
                ? { label: supersededBy.title, to: `/decisions/${supersededBy.public_slug}` }
                : decision.data.payload.superseded_by_ref
                  ? { label: decision.data.payload.superseded_by_ref.resource_label, to: null }
                  : null
            }
          />
          <ProvenanceStrip
            summary={decision.data.payload.provenance_summary}
            supportVisibility={decision.data.payload.support_visibility}
            variant="rail"
          />
        </aside>
      </div>

      <div className="reader-two-column detail-layout">
        <article className="reader-primary-panel detail-main-panel">
          <SectionIntro
            eyebrow="Decision story"
            title="Readable narrative order"
            description="The decision statement leads, followed by rationale, constraints, impact, and linked knowledge."
            compact
          />

          <section className="narrative-section narrative-section-panel">
            <h4>Decision statement</h4>
            <p>{decision.data.payload.decision_statement}</p>
          </section>

          {decision.data.payload.problem_statement ? (
            <section className="narrative-section narrative-section-panel">
              <h4>Problem</h4>
              <p>{decision.data.payload.problem_statement}</p>
            </section>
          ) : null}

          {decision.data.payload.rationale ? (
            <section className="narrative-section narrative-section-panel">
              <h4>Rationale</h4>
              <p>{decision.data.payload.rationale}</p>
            </section>
          ) : null}

          {decision.data.payload.constraints.length ? (
            <section className="narrative-section narrative-section-panel">
              <h4>Constraints</h4>
              <div className="chip-row">
                {decision.data.payload.constraints.map((constraint) => (
                  <span key={constraint} className="chip">
                    {constraint}
                  </span>
                ))}
              </div>
            </section>
          ) : null}

          {decision.data.payload.impact_summary ? (
            <section className="narrative-section narrative-section-panel">
              <h4>Impact</h4>
              <p>{decision.data.payload.impact_summary}</p>
            </section>
          ) : null}

          <RefList
            title="Linked concepts"
            refs={decision.data.payload.linked_concept_refs}
            resolveHref={(ref: ResourceRef) => {
              const concept = conceptsById.get(ref.resource_id)
              return concept ? `/concepts/${concept.public_slug}` : null
            }}
          />

          <RefList title="Linked relations" refs={decision.data.payload.linked_relation_refs} />
        </article>

        <aside className="reader-secondary-panel detail-support-rail">
          <SectionIntro
            eyebrow="Support rail"
            title="Visible support and provenance detail"
            description="Supporting evidence remains explicit, grouped, and readable in a structured side rail."
          />

          <section className="narrative-section narrative-section-panel">
            <h4>Visible support</h4>
            <div className="stack-list">
              {decision.data.payload.visible_support_items.map((item) => (
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
            <section className="narrative-section narrative-section-panel">
              <h4>Provenance details</h4>
              <p>
                {provenance.data.payload.provenance_summary.visible_support_item_count} visible of{' '}
                {provenance.data.payload.provenance_summary.support_item_count} total support items.
              </p>
              {provenance.data.payload.support_items.length ? (
                <div className="stack-list">
                  {provenance.data.payload.support_items.map((item) => (
                    <article key={item.support_item_id} className="list-card compact-card">
                      <strong>{item.source_label}</strong>
                      <p>{item.excerpt_or_summary ?? 'No visible excerpt available.'}</p>
                    </article>
                  ))}
                </div>
                ) : null}
              </section>
          ) : (
            <EmptyState
              title="Provenance is loading"
              description="Decision provenance will render here."
              eyebrow="Decision provenance"
            />
          )}
        </aside>
      </div>
    </section>
  )
}
