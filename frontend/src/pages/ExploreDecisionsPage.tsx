import { Link } from 'react-router-dom'

import { apiGet } from '../api/client'
import { EmptyState } from '../components/EmptyState'
import { ArtifactCard, ExploreTabs, LineageRail, ProvenanceStrip, SectionIntro } from '../components/experience'
import { PageHeader } from '../components/PageHeader'
import { useAsyncData } from '../hooks/useAsyncData'
import type {
  ActorSession,
  ContractEnvelope,
  DecisionPayload,
  RuntimeBootstrapMeta,
} from '../types/api'
import {
  canActorManageConnectors,
  isProductionWorkspaceDegraded,
  isProductionWorkspacePending,
} from '../viewModels'

type ExploreDecisionsPageProps = {
  activeActor: ActorSession
  runtimeInfo: RuntimeBootstrapMeta
}

export function ExploreDecisionsPage({ activeActor, runtimeInfo }: ExploreDecisionsPageProps) {
  const decisions = useAsyncData<ContractEnvelope<DecisionPayload>[]>(
    () => apiGet('/decisions'),
    [activeActor.actor_id],
  )

  if (decisions.error) {
    return <EmptyState title="Decisions unavailable" description={decisions.error} />
  }

  const leadDecision = decisions.data?.[0] ?? null
  const supportingDecisions = leadDecision ? (decisions.data ?? []).slice(1) : decisions.data ?? []
  const railDecisions = supportingDecisions.slice(0, 3)
  const libraryDecisions = supportingDecisions.slice(3)
  const decisionsById = new Map((decisions.data ?? []).map((item) => [item.payload.decision_id, item.payload]))
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
          title="Explore Decisions"
          description="Browse decision records as readable artifacts with explicit lineage instead of buried operator metadata."
        />
        <ExploreTabs />
        <EmptyState
          eyebrow={awaitingSources ? 'Production onboarding' : 'Production sync'}
          title={awaitingSources ? 'No production decisions yet' : 'Decision records are waiting for first sync'}
          description={
            awaitingSources
              ? canManageConnectors
                ? 'Production mode does not show demo decision records. Connect a shared datasource before decision artifacts can appear.'
                : 'This production workspace has no linked shared datasource yet. A connector manager needs to connect one before decision records can appear.'
              : `Sources are connected, but the first usable decision set is not ready yet. ${runtimeInfo.linked_source_count} linked sources are currently being prepared.`
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
        title="Explore Decisions"
        description="Browse decision records as readable artifacts with explicit lineage instead of buried operator metadata."
      />
      <ExploreTabs />
      {productionDegraded ? (
        <EmptyState
          eyebrow="Production recovery"
          title="Some source health is degraded"
          description={`Published decision records remain visible when available, but ${runtimeInfo.degraded_source_count} linked sources currently need recovery attention.`}
          actions={sourceStudioAction}
        />
      ) : null}

      {leadDecision ? (
        <div className="reader-two-column browse-stage-grid">
          <div className="reader-primary-panel lead-artifact browse-lead-panel">
            <ArtifactCard
              to={`/decisions/${leadDecision.payload.public_slug}`}
              eyebrow={`${leadDecision.payload.owning_domain} decision`}
              title={leadDecision.payload.title}
              summary={leadDecision.payload.decision_statement}
              supportVisibility={leadDecision.payload.support_visibility}
              lifecycleState={leadDecision.payload.lifecycle_state}
              verificationState={leadDecision.payload.provenance_summary.verification_state}
              variant="lead"
              ctaLabel="Read"
            >
              <LineageRail
                variant="timeline"
                previous={
                  leadDecision.payload.supersedes_ref
                    ? { label: leadDecision.payload.supersedes_ref.resource_label, to: null }
                    : null
                }
                next={
                  leadDecision.payload.superseded_by_ref
                    ? { label: leadDecision.payload.superseded_by_ref.resource_label, to: null }
                    : null
                }
              />
              <ProvenanceStrip
                summary={leadDecision.payload.provenance_summary}
                supportVisibility={leadDecision.payload.support_visibility}
                variant="rail"
              />
            </ArtifactCard>
          </div>

          <aside className="reader-secondary-panel browse-support-panel">
            <SectionIntro
              eyebrow="Decision browse"
              title={`${(decisions.data ?? []).length} current records`}
              description="Use the lead record for orientation, then scan compact lineage-aware entries."
            />
            <div className="stack-list">
              {railDecisions.map((envelope) => (
                <ArtifactCard
                  key={envelope.payload.decision_id}
                  to={`/decisions/${envelope.payload.public_slug}`}
                  eyebrow={`${envelope.payload.owning_domain} decision`}
                  title={envelope.payload.title}
                  summary={envelope.payload.decision_statement}
                  supportVisibility={envelope.payload.support_visibility}
                  lifecycleState={envelope.payload.lifecycle_state}
                  verificationState={envelope.payload.provenance_summary.verification_state}
                  variant="compact"
                  ctaLabel="Read"
                >
                  <LineageRail
                    variant="timeline"
                    previous={
                      envelope.payload.supersedes_ref
                        ? {
                            label:
                              decisionsById.get(envelope.payload.supersedes_ref.resource_id)?.title ??
                              envelope.payload.supersedes_ref.resource_label,
                            to: decisionsById.get(envelope.payload.supersedes_ref.resource_id)
                              ? `/decisions/${decisionsById.get(envelope.payload.supersedes_ref.resource_id)?.public_slug}`
                              : null,
                          }
                        : null
                    }
                    next={
                      envelope.payload.superseded_by_ref
                        ? {
                            label:
                              decisionsById.get(envelope.payload.superseded_by_ref.resource_id)?.title ??
                              envelope.payload.superseded_by_ref.resource_label,
                            to: decisionsById.get(envelope.payload.superseded_by_ref.resource_id)
                              ? `/decisions/${decisionsById.get(envelope.payload.superseded_by_ref.resource_id)?.public_slug}`
                              : null,
                          }
                        : null
                    }
                  />
                </ArtifactCard>
              ))}
            </div>
          </aside>
        </div>
      ) : null}

      {libraryDecisions.length ? (
        <section className="reader-section">
          <SectionIntro
            eyebrow="Decision library"
            title="Readable records with lineage in-line"
            description="The browse surface keeps lineage visible without turning decisions back into operator cards."
          />
          <div className="card-grid browse-card-grid">
            {libraryDecisions.map((envelope, index) => {
              const supersedes = envelope.payload.supersedes_ref
                ? decisionsById.get(envelope.payload.supersedes_ref.resource_id)
                : null
              const supersededBy = envelope.payload.superseded_by_ref
                ? decisionsById.get(envelope.payload.superseded_by_ref.resource_id)
                : null

              return (
                <ArtifactCard
                  key={envelope.payload.decision_id}
                  to={`/decisions/${envelope.payload.public_slug}`}
                  eyebrow={`${envelope.payload.owning_domain} decision`}
                  title={envelope.payload.title}
                  summary={envelope.payload.decision_statement}
                  supportVisibility={envelope.payload.support_visibility}
                  lifecycleState={envelope.payload.lifecycle_state}
                  verificationState={envelope.payload.provenance_summary.verification_state}
                  variant={index < 2 ? 'standard' : 'compact'}
                  ctaLabel="Read"
                >
                  <LineageRail
                    variant="timeline"
                    previous={
                      supersedes
                        ? {
                            label: supersedes.title,
                            to: `/decisions/${supersedes.public_slug}`,
                          }
                        : envelope.payload.supersedes_ref
                          ? { label: envelope.payload.supersedes_ref.resource_label, to: null }
                          : null
                    }
                    next={
                      supersededBy
                        ? {
                            label: supersededBy.title,
                            to: `/decisions/${supersededBy.public_slug}`,
                          }
                        : envelope.payload.superseded_by_ref
                          ? { label: envelope.payload.superseded_by_ref.resource_label, to: null }
                          : null
                    }
                  />
                </ArtifactCard>
              )
            })}
          </div>
        </section>
      ) : null}

      {!leadDecision && !productionDegraded ? (
        <EmptyState
          eyebrow="Published decisions"
          title="No published decisions yet"
          description="The workspace has source connectivity, but no member-facing decision artifacts are published yet."
          actions={sourceStudioAction}
        />
      ) : null}
    </section>
  )
}
