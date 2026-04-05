import { apiGet } from '../api/client'
import { EmptyState } from '../components/EmptyState'
import { PageHeader } from '../components/PageHeader'
import { StatusPill } from '../components/StatusPill'
import { useAsyncData } from '../hooks/useAsyncData'
import type { GraphResponse, Relation } from '../types/api'

type GraphPageProps = {
  contextSpaceId: string | null
}

export function GraphPage({ contextSpaceId }: GraphPageProps) {
  const graph = useAsyncData<GraphResponse>(
    () => (contextSpaceId ? apiGet('/graph', { context_space_id: contextSpaceId }) : Promise.resolve({ nodes: [], edges: [] })),
    [contextSpaceId],
  )
  const relations = useAsyncData<Relation[]>(
    () => (contextSpaceId ? apiGet('/relations', { context_space_id: contextSpaceId }) : Promise.resolve([])),
    [contextSpaceId],
  )

  if (!contextSpaceId) {
    return <EmptyState title="No graph yet" description="The graph view appears after the first context space is ready." />
  }

  return (
    <div className="page-stack">
      <PageHeader
        title="Ontology and context graph"
        description="Official concept relations expressed as reviewable assertions with provenance."
      />

      <section className="two-column-layout">
        <article className="panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">Nodes</span>
              <h3>Concept map</h3>
            </div>
          </div>
          <div className="graph-node-grid">
            {(graph.data?.nodes ?? []).map((node) => (
              <div key={node.id} className="graph-node-card">
                <span className="eyebrow">{node.type}</span>
                <strong>{node.label}</strong>
                <StatusPill value={node.status} />
              </div>
            ))}
          </div>
        </article>

        <article className="panel">
          <div className="panel-heading">
            <div>
              <span className="eyebrow">Edges</span>
              <h3>Relation flow</h3>
            </div>
          </div>
          <div className="stack-list">
            {(relations.data ?? []).map((relation) => (
              <article key={relation.id} className="list-card relation-card">
                <div className="card-row between">
                  <div className="relation-line">
                    <span>{relation.subject_name}</span>
                    <span className="relation-predicate">{relation.predicate}</span>
                    <span>{relation.object_name}</span>
                  </div>
                  <StatusPill value={relation.status} />
                </div>
                <p>{relation.description}</p>
              </article>
            ))}
          </div>
        </article>
      </section>
    </div>
  )
}
