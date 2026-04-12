import { Link } from 'react-router-dom'

import { apiGet } from '../api/client'
import { EmptyState } from '../components/EmptyState'
import { ArtifactCard, ExploreTabs, ProvenanceStrip, SectionIntro } from '../components/experience'
import { PageHeader } from '../components/PageHeader'
import { useAsyncData } from '../hooks/useAsyncData'
import type {
  ActorSession,
  ConceptPayload,
  ContractEnvelope,
  RuntimeBootstrapMeta,
} from '../types/api'
import {
  canActorManageConnectors,
  isProductionWorkspaceDegraded,
  isProductionWorkspacePending,
} from '../viewModels'

type ExploreTopicsPageProps = {
  activeActor: ActorSession
  runtimeInfo: RuntimeBootstrapMeta
}

export function ExploreTopicsPage({ activeActor, runtimeInfo }: ExploreTopicsPageProps) {
  const concepts = useAsyncData<ContractEnvelope<ConceptPayload>[]>(
    () => apiGet('/concepts'),
    [activeActor.actor_id],
  )

  if (concepts.error) {
    return <EmptyState title="Topics unavailable" description={concepts.error} />
  }

  const leadConcept = concepts.data?.[0] ?? null
  const supportingConcepts = leadConcept ? (concepts.data ?? []).slice(1) : concepts.data ?? []
  const railConcepts = supportingConcepts.slice(0, 3)
  const libraryConcepts = supportingConcepts.slice(3)
  const canManageConnectors = canActorManageConnectors(activeActor)
  const productionPending = isProductionWorkspacePending(runtimeInfo)
  const productionDegraded = isProductionWorkspaceDegraded(runtimeInfo)
  const sourceStudioAction = canManageConnectors ? (
    <Link className="ghost-link" to="/source-studio">
      Open Source Studio
    </Link>
  ) : null

  if (productionPending) {
    const awaitingSources = runtimeInfo.workspace_data_state === 'awaiting_sources'
    return (
      <section className="page-stack reader-page explore-page">
        <PageHeader
          eyebrow="Cornerstone explore"
          title="Explore Topics"
          description="Browse canonical concepts in an editorial view that keeps definitions first and trust disclosure close behind."
        />
        <ExploreTabs />
        <EmptyState
          eyebrow={awaitingSources ? 'Production onboarding' : 'Production sync'}
          title={awaitingSources ? 'No production topics yet' : 'Topic index is waiting for first sync'}
          description={
            awaitingSources
              ? canManageConnectors
                ? 'Production mode does not render demo topics. Connect a shared datasource before topic artifacts can appear.'
                : 'This production workspace has no linked shared datasource yet. A connector manager needs to connect one before topics can appear.'
              : `Sources are connected, but the first usable topic set is not ready yet. ${runtimeInfo.linked_source_count} linked sources are currently being prepared.`
          }
          actions={sourceStudioAction}
        />
      </section>
    )
  }

  return (
    <section className="page-stack reader-page explore-page">
      <PageHeader
        eyebrow="Cornerstone explore"
        title="Explore Topics"
        description="Browse canonical concepts in an editorial view that keeps definitions first and trust disclosure close behind."
      />
      <ExploreTabs />
      {productionDegraded ? (
        <EmptyState
          eyebrow="Production recovery"
          title="Some source health is degraded"
          description={`Published topics remain visible when available, but ${runtimeInfo.degraded_source_count} linked sources currently need recovery attention.`}
          actions={sourceStudioAction}
        />
      ) : null}

      {leadConcept ? (
        <div className="reader-two-column browse-stage-grid">
          <div className="reader-primary-panel lead-artifact browse-lead-panel">
            <ArtifactCard
              to={`/concepts/${leadConcept.payload.public_slug}`}
              eyebrow={`${leadConcept.payload.owning_domain} topic`}
              title={leadConcept.payload.canonical_name}
              summary={leadConcept.payload.definition}
              supportVisibility={leadConcept.payload.support_visibility}
              lifecycleState={leadConcept.payload.lifecycle_state}
              verificationState={leadConcept.payload.verification_state}
              variant="lead"
              ctaLabel="Read"
              meta={
                <div className="inline-meta">
                  <span className="meta-copy">{leadConcept.payload.aliases.length} aliases</span>
                  <span className="meta-copy">{leadConcept.payload.linked_decision_refs.length} linked decisions</span>
                  <span className="meta-copy">{leadConcept.payload.linked_relation_refs.length} linked relations</span>
                </div>
              }
            >
              <ProvenanceStrip
                summary={leadConcept.payload.provenance_summary}
                supportVisibility={leadConcept.payload.support_visibility}
                verificationState={leadConcept.payload.verification_state}
                variant="rail"
              />
            </ArtifactCard>
          </div>

          <aside className="reader-secondary-panel browse-support-panel">
            <SectionIntro
              eyebrow="Browse guide"
              title={`${(concepts.data ?? []).length} official topics`}
              description="Use the lead artifact for orientation, then move through the compact topic library."
            />
            <div className="stack-list">
              {railConcepts.map((envelope) => (
                <ArtifactCard
                  key={envelope.payload.concept_id}
                  to={`/concepts/${envelope.payload.public_slug}`}
                  eyebrow={`${envelope.payload.owning_domain} topic`}
                  title={envelope.payload.canonical_name}
                  summary={envelope.payload.definition}
                  supportVisibility={envelope.payload.support_visibility}
                  lifecycleState={envelope.payload.lifecycle_state}
                  verificationState={envelope.payload.verification_state}
                  variant="compact"
                  ctaLabel="Read"
                />
              ))}
            </div>
          </aside>
        </div>
      ) : null}

      {libraryConcepts.length ? (
        <section className="reader-section">
          <SectionIntro
            eyebrow="Topic library"
            title="Definitions with trust in line"
            description="The browse surface stays compact and readable while preserving direct continuity into presentable detail routes."
          />
          <div className="card-grid browse-card-grid">
            {libraryConcepts.map((envelope, index) => (
              <ArtifactCard
                key={envelope.payload.concept_id}
                to={`/concepts/${envelope.payload.public_slug}`}
                eyebrow={`${envelope.payload.owning_domain} topic`}
                title={envelope.payload.canonical_name}
                summary={envelope.payload.definition}
                supportVisibility={envelope.payload.support_visibility}
                lifecycleState={envelope.payload.lifecycle_state}
                verificationState={envelope.payload.verification_state}
                variant={index < 2 ? 'standard' : 'compact'}
                ctaLabel="Read"
                meta={<span className="meta-copy">{envelope.payload.linked_decision_refs.length} linked decisions</span>}
              />
            ))}
          </div>
        </section>
      ) : null}

      {!leadConcept && !productionDegraded ? (
        <EmptyState
          eyebrow="Published topics"
          title="No published topics yet"
          description="The workspace has source connectivity, but no member-facing topic artifacts are published yet."
          actions={sourceStudioAction}
        />
      ) : null}
    </section>
  )
}
