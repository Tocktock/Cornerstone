import { Link } from 'react-router-dom'
import { useMemo } from 'react'
import { useNavigate, useParams } from 'react-router-dom'

import { apiGet } from '../api/client'
import { EmptyState } from '../components/EmptyState'
import { ExploreTabs, SectionIntro } from '../components/experience'
import { PageHeader } from '../components/PageHeader'
import { StatusPill } from '../components/StatusPill'
import { useAsyncData } from '../hooks/useAsyncData'
import type {
  ActorSession,
  ContractEnvelope,
  GraphSlicePayload,
  RuntimeBootstrapMeta,
} from '../types/api'
import {
  buildGraphExplorer,
  canActorManageConnectors,
  isProductionWorkspaceDegraded,
  isProductionWorkspacePending,
} from '../viewModels'

type ExploreMapPageProps = {
  activeActor: ActorSession
  runtimeInfo: RuntimeBootstrapMeta
}

export function ExploreMapPage({ activeActor, runtimeInfo }: ExploreMapPageProps) {
  const { conceptId } = useParams()
  const navigate = useNavigate()
  const graph = useAsyncData<ContractEnvelope<GraphSlicePayload>>(
    () => apiGet('/graph', conceptId ? { root: conceptId } : undefined),
    [activeActor.actor_id, conceptId],
  )

  const explorer = useMemo(
    () => (graph.data ? buildGraphExplorer(graph.data.payload, conceptId ?? null) : null),
    [graph.data, conceptId],
  )
  const rootNodes = explorer?.nodes.filter((node) => node.isRoot) ?? []
  const linkedNodes = explorer?.nodes.filter((node) => !node.isRoot) ?? []
  const canManageConnectors = canActorManageConnectors(activeActor)
  const productionPending = isProductionWorkspacePending(runtimeInfo)
  const productionDegraded = isProductionWorkspaceDegraded(runtimeInfo)
  const sourceStudioAction = canManageConnectors ? (
    <Link className="ghost-link" to="/source-studio">
      Open Source Studio
    </Link>
  ) : null

  if (graph.error) {
    return <EmptyState title="Map unavailable" description={graph.error} />
  }

  if (productionPending) {
    const awaitingSources = runtimeInfo.workspace_data_state === 'awaiting_sources'
    return (
      <section className="page-stack reader-page explore-page">
        <PageHeader
          eyebrow="Cornerstone explore"
          title="Explore Map"
        />
        <ExploreTabs />
        <EmptyState
          eyebrow={awaitingSources ? 'Production onboarding' : 'Production sync'}
          title={awaitingSources ? 'Map data is not connected yet' : 'Map data is still synchronizing'}
          description={
            awaitingSources
              ? canManageConnectors
                ? 'Connect a shared datasource before the production concept map can be built.'
                : 'A connector manager needs to connect a shared datasource before the concept map can appear.'
              : `${runtimeInfo.linked_source_count} linked sources are still preparing the first map slice.`
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
        title="Explore Map"
      />
      <ExploreTabs />
      {productionDegraded ? (
        <EmptyState
          eyebrow="Production recovery"
          title="Some source health is degraded"
          description={`${runtimeInfo.degraded_source_count} linked sources currently need recovery attention.`}
          actions={sourceStudioAction}
        />
      ) : null}

      {explorer ? (
        <div className="reader-two-column map-layout map-stage-grid">
          <article className="reader-primary-panel graph-detail-panel map-focus-stage">
            <SectionIntro
              eyebrow="Map focus"
              title={explorer.selectedNode.resource_label}
              actions={
                <div className="artifact-status-row">
                  <StatusPill value={`${explorer.selectedNode.totalRelationCount} links`} />
                </div>
              }
            />

            <div className="chip-row root-jump-row">
              {explorer.rootNodes.map((root) => (
                <button
                  key={root.resource_id}
                  type="button"
                  className={`chip-button ${explorer.selectedNode.resource_id === root.resource_id ? 'selected' : ''}`}
                  aria-label={`Jump to ${root.resource_label}`}
                  onClick={() => navigate(`/explore/map/${root.resource_id}`)}
                >
                  {root.resource_label}
                </button>
              ))}
            </div>

            <div className="map-focus-band">
              <article className="list-card map-focus-card">
                <span className="eyebrow">{explorer.selectedNode.isRoot ? 'Root concept' : 'Linked concept'}</span>
                <p className="map-focus-title">{explorer.selectedNode.resource_label}</p>
                <div className="artifact-status-row">
                  <StatusPill value={`${explorer.selectedNode.outboundCount} outbound`} />
                  <StatusPill value={`${explorer.selectedNode.inboundCount} inbound`} />
                </div>
              </article>

              <div className="map-memory-rail">
                <div className="map-memory-stop">
                  <span className="mini-label">Root concepts</span>
                  <strong>{rootNodes.length}</strong>
                  <p>{rootNodes.map((node) => node.resource_label).join(' · ')}</p>
                </div>
                <div className="map-memory-stop">
                  <span className="mini-label">Linked concepts</span>
                  <strong>{linkedNodes.length}</strong>
                  <p>
                    {linkedNodes.length
                      ? linkedNodes.map((node) => node.resource_label).join(' · ')
                      : 'No linked concepts in the current slice.'}
                  </p>
                </div>
              </div>
            </div>

            <div className="graph-relation-columns">
              <section className="narrative-section outbound-relations">
                <SectionIntro
                  eyebrow="Outbound"
                  title="Outbound relations"
                  compact
                />
                {explorer.outboundRelations.length ? (
                  <div className="stack-list">
                    {explorer.outboundRelations.map((relation) => (
                      <button
                        key={relation.relation_ref.resource_id}
                        type="button"
                        className="list-card compact-card relation-card"
                        onClick={() => navigate(`/explore/map/${relation.counterpart.resource_id}`)}
                      >
                        <p className="relation-title">
                          <strong>{explorer.selectedNode.resource_label}</strong>
                          <span className="relation-predicate">{relation.predicate}</span>
                          <strong>{relation.counterpart.resource_label}</strong>
                        </p>
                        <div className="artifact-status-row">
                          <StatusPill value={relation.support_visibility} />
                          <StatusPill value={relation.verification_state} />
                        </div>
                      </button>
                    ))}
                  </div>
                ) : (
                  <div className="list-card empty-substate">No outbound relations in the current slice.</div>
                )}
              </section>

              <section className="narrative-section inbound-relations">
                <SectionIntro
                  eyebrow="Inbound"
                  title="Inbound relations"
                  compact
                />
                {explorer.inboundRelations.length ? (
                  <div className="stack-list">
                    {explorer.inboundRelations.map((relation) => (
                      <button
                        key={relation.relation_ref.resource_id}
                        type="button"
                        className="list-card compact-card relation-card"
                        onClick={() => navigate(`/explore/map/${relation.counterpart.resource_id}`)}
                      >
                        <p className="relation-title">
                          <strong>{relation.counterpart.resource_label}</strong>
                          <span className="relation-predicate">{relation.predicate}</span>
                          <strong>{explorer.selectedNode.resource_label}</strong>
                        </p>
                        <div className="artifact-status-row">
                          <StatusPill value={relation.support_visibility} />
                          <StatusPill value={relation.verification_state} />
                        </div>
                      </button>
                    ))}
                  </div>
                ) : (
                  <div className="list-card empty-substate">No inbound relations in the current slice.</div>
                )}
              </section>
            </div>
          </article>

          <aside className="reader-secondary-panel">
            <SectionIntro
              eyebrow="Concept index"
              title="Navigate the slice"
            />

            <section className="narrative-section compact-section">
              <h4>Root concepts</h4>
              <div className="graph-node-list">
                {rootNodes.map((node) => (
                  <button
                    key={node.resource_id}
                    type="button"
                    className={`artifact-card artifact-card-compact graph-node-button ${explorer.selectedNode.resource_id === node.resource_id ? 'selected' : ''}`}
                    aria-label={`Explore ${node.resource_label}`}
                    onClick={() => navigate(`/explore/map/${node.resource_id}`)}
                  >
                    <div className="card-row between start">
                      <div>
                        <span className="eyebrow">root concept</span>
                        <h3>{node.resource_label}</h3>
                      </div>
                      <span className="count-badge">{node.totalRelationCount}</span>
                    </div>
                    <p>{node.outboundCount} outbound · {node.inboundCount} inbound</p>
                  </button>
                ))}
              </div>
            </section>

            <section className="narrative-section compact-section">
              <h4>Linked concepts</h4>
              <div className="graph-node-list">
                {linkedNodes.map((node) => (
                  <button
                    key={node.resource_id}
                    type="button"
                    className={`artifact-card artifact-card-rail graph-node-button ${explorer.selectedNode.resource_id === node.resource_id ? 'selected' : ''}`}
                    aria-label={`Explore ${node.resource_label}`}
                    onClick={() => navigate(`/explore/map/${node.resource_id}`)}
                  >
                    <div className="card-row between start">
                      <div>
                        <span className="eyebrow">linked concept</span>
                        <h3>{node.resource_label}</h3>
                      </div>
                      <span className="count-badge">{node.totalRelationCount}</span>
                    </div>
                    <p>{node.outboundCount} outbound · {node.inboundCount} inbound</p>
                  </button>
                ))}
              </div>
            </section>
          </aside>
        </div>
      ) : (
        <EmptyState
          title="No published map yet"
          description="The workspace has connectivity, but no member-facing concept map is published yet."
          actions={sourceStudioAction}
        />
      )}
    </section>
  )
}
