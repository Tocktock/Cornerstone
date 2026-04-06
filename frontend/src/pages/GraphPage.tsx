import { useEffect, useMemo, useState } from 'react'

import { apiGet } from '../api/client'
import { EmptyState } from '../components/EmptyState'
import { PageHeader } from '../components/PageHeader'
import { StatusPill } from '../components/StatusPill'
import { useAsyncData } from '../hooks/useAsyncData'
import type { ActorSession, ContextSpaceRef, ContractEnvelope, GraphSlicePayload } from '../types/api'
import { buildGraphExplorer } from '../viewModels'

type GraphPageProps = {
  workspace: ContextSpaceRef
  activeActor: ActorSession
}

export function GraphPage({ workspace, activeActor }: GraphPageProps) {
  const graph = useAsyncData<ContractEnvelope<GraphSlicePayload>>(
    () => apiGet('/graph'),
    [workspace.context_space_id, activeActor.actor_id],
  )
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null)

  const explorer = useMemo(
    () => (graph.data ? buildGraphExplorer(graph.data.payload, selectedNodeId) : null),
    [graph.data, selectedNodeId],
  )

  useEffect(() => {
    if (!graph.data) {
      return
    }
    const nextDefault = buildGraphExplorer(graph.data.payload, selectedNodeId)?.selectedNode.resource_id ?? null
    if (nextDefault && nextDefault !== selectedNodeId) {
      setSelectedNodeId(nextDefault)
    }
  }, [graph.data, selectedNodeId])

  if (graph.error) {
    return <EmptyState title="Graph unavailable" description={graph.error} />
  }

  return (
    <section className="page-stack graph-page">
      <PageHeader
        title="Graph slice"
        description="Explore the current graph slice by selecting a concept and tracing the inbound and outbound relations around it."
      />

      {explorer ? (
        <>
          <article className="panel graph-root-panel">
            <div className="panel-heading panel-heading-start">
              <div>
                <span className="eyebrow">Root concepts</span>
                <h3>Jump to a starting point</h3>
              </div>
              <p className="panel-copy">Root concepts anchor the curated graph slice for this workspace.</p>
            </div>
            <div className="chip-row">
              {explorer.rootNodes.map((root) => (
                <button
                  key={root.resource_id}
                  type="button"
                  className={`chip-button ${explorer.selectedNode.resource_id === root.resource_id ? 'selected' : ''}`}
                  aria-label={`Jump to ${root.resource_label}`}
                  onClick={() => setSelectedNodeId(root.resource_id)}
                >
                  {root.resource_label}
                </button>
              ))}
            </div>
          </article>

          <div className="graph-explorer-layout">
            <article className="panel graph-detail-panel">
              <span className="eyebrow">{explorer.selectedNode.resource_kind}</span>
              <h3>{explorer.selectedNode.resource_label}</h3>
              <p className="muted">
                {explorer.selectedNode.totalRelationCount} direct relation
                {explorer.selectedNode.totalRelationCount === 1 ? '' : 's'} in this slice.
              </p>

              <div className="graph-relation-columns">
                <section className="panel nested-panel relation-panel outbound-relations">
                  <div className="panel-heading panel-heading-start">
                    <div>
                      <span className="mini-label">Outbound</span>
                      <h4>{explorer.outboundRelations.length} relation{explorer.outboundRelations.length === 1 ? '' : 's'}</h4>
                    </div>
                  </div>
                  {explorer.outboundRelations.length ? (
                    <div className="stack-list">
                      {explorer.outboundRelations.map((relation) => (
                        <article key={relation.relation_ref.resource_id} className="list-card compact-card">
                          <p className="relation-title">
                            <strong>{explorer.selectedNode.resource_label}</strong>
                            <span className="relation-predicate">{relation.predicate}</span>
                            <strong>{relation.counterpart.resource_label}</strong>
                          </p>
                          <div className="inline-meta">
                            <StatusPill value={relation.support_visibility} />
                            <StatusPill value={relation.verification_state} />
                          </div>
                        </article>
                      ))}
                    </div>
                  ) : (
                    <div className="list-card empty-substate">No outbound relations in the current slice.</div>
                  )}
                </section>

                <section className="panel nested-panel relation-panel inbound-relations">
                  <div className="panel-heading panel-heading-start">
                    <div>
                      <span className="mini-label">Inbound</span>
                      <h4>{explorer.inboundRelations.length} relation{explorer.inboundRelations.length === 1 ? '' : 's'}</h4>
                    </div>
                  </div>
                  {explorer.inboundRelations.length ? (
                    <div className="stack-list">
                      {explorer.inboundRelations.map((relation) => (
                        <article key={relation.relation_ref.resource_id} className="list-card compact-card">
                          <p className="relation-title">
                            <strong>{relation.counterpart.resource_label}</strong>
                            <span className="relation-predicate">{relation.predicate}</span>
                            <strong>{explorer.selectedNode.resource_label}</strong>
                          </p>
                          <div className="inline-meta">
                            <StatusPill value={relation.support_visibility} />
                            <StatusPill value={relation.verification_state} />
                          </div>
                        </article>
                      ))}
                    </div>
                  ) : (
                    <div className="list-card empty-substate">No inbound relations in the current slice.</div>
                  )}
                </section>
              </div>
            </article>

            <div className="page-stack">
              <article className="panel graph-index-panel">
                <div className="panel-heading panel-heading-start">
                  <div>
                    <span className="eyebrow">Concept index</span>
                    <h3>Explore the slice</h3>
                  </div>
                  <p className="panel-copy">Select any concept to inspect its direct relation path.</p>
                </div>
                <div className="graph-node-list">
                  {explorer.nodes.map((node) => (
                    <button
                      key={node.resource_id}
                      type="button"
                      className={`panel selectable-card graph-node-button ${
                        explorer.selectedNode.resource_id === node.resource_id ? 'selected' : ''
                      }`}
                      aria-label={`Explore ${node.resource_label}`}
                      onClick={() => setSelectedNodeId(node.resource_id)}
                    >
                      <div className="card-row between start">
                        <div>
                          <span className="eyebrow">{node.resource_kind}</span>
                          <h4>{node.resource_label}</h4>
                        </div>
                        <span className="count-badge">{node.totalRelationCount}</span>
                      </div>
                      <div className="inline-meta">
                        {node.isRoot ? <span className="chip subtle">root</span> : null}
                        <span className="muted">{node.outboundCount} outbound</span>
                        <span className="muted">{node.inboundCount} inbound</span>
                      </div>
                    </button>
                  ))}
                </div>
              </article>
            </div>
          </div>
        </>
      ) : (
        <EmptyState title="Graph is loading" description="No graph data yet." />
      )}
    </section>
  )
}
