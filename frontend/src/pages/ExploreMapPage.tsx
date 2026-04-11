import { useMemo } from 'react'
import { useNavigate, useParams } from 'react-router-dom'

import { apiGet } from '../api/client'
import { EmptyState } from '../components/EmptyState'
import { ExploreTabs, SectionIntro } from '../components/experience'
import { PageHeader } from '../components/PageHeader'
import { StatusPill } from '../components/StatusPill'
import { useAsyncData } from '../hooks/useAsyncData'
import type { ActorSession, ContractEnvelope, GraphSlicePayload } from '../types/api'
import { buildGraphExplorer } from '../viewModels'

type ExploreMapPageProps = {
  activeActor: ActorSession
}

export function ExploreMapPage({ activeActor }: ExploreMapPageProps) {
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

  if (graph.error) {
    return <EmptyState title="Map unavailable" description={graph.error} />
  }

  return (
    <section className="page-stack reader-page explore-page">
      <PageHeader
        eyebrow="Cornerstone explore"
        title="Explore Map"
        description="Trace concept relationships through a URL-addressable map with clearer relation storytelling."
      />
      <ExploreTabs />

      {explorer ? (
        <div className="reader-two-column map-layout map-stage-grid">
          <article className="reader-primary-panel graph-detail-panel map-focus-stage">
            <SectionIntro
              eyebrow="Map focus"
              title={explorer.selectedNode.resource_label}
              description={`${explorer.selectedNode.totalRelationCount} direct relations remain visible in the current slice.`}
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
                <p className="panel-copy">
                  The active object stays visually dominant while the surrounding relation lanes update around it.
                </p>
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
                  description="Follow what the current concept points toward."
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
                  description="See what currently resolves back into the active concept."
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
              description="Roots and linked concepts stay visually distinct so the current path remains legible."
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
        <EmptyState title="Map is loading" description="The explore map will render here." />
      )}
    </section>
  )
}
