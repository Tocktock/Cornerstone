import { apiGet } from '../api/client'
import { EmptyState } from '../components/EmptyState'
import { PageHeader } from '../components/PageHeader'
import { StatusPill } from '../components/StatusPill'
import { useAsyncData } from '../hooks/useAsyncData'
import type { ContextSpaceRef, ContractEnvelope, GraphSlicePayload } from '../types/api'

type GraphPageProps = {
  workspace: ContextSpaceRef
}

export function GraphPage({ workspace }: GraphPageProps) {
  const graph = useAsyncData<ContractEnvelope<GraphSlicePayload>>(() => apiGet('/graph'), [workspace.context_space_id])

  if (graph.error) {
    return <EmptyState title="Graph unavailable" description={graph.error} />
  }

  return (
    <section className="page-stack">
      <PageHeader
        title="Graph slice"
        description="The graph surface shows the same relation semantics the API and MCP-style adapter expose."
      />

      {graph.data ? (
        <>
          <article className="panel">
            <span className="eyebrow">Roots</span>
            <div className="inline-meta">
              {graph.data.payload.root_concept_refs.map((root) => (
                <StatusPill key={root.resource_id} value={root.resource_label} />
              ))}
            </div>
          </article>

          <div className="card-grid">
            {graph.data.payload.nodes.map((node) => (
              <article key={node.resource_id} className="graph-node-card">
                <span className="eyebrow">{node.resource_kind}</span>
                <h3>{node.resource_label}</h3>
              </article>
            ))}
          </div>

          <div className="page-stack">
            {graph.data.payload.edges.map((edge) => (
              <article key={edge.relation_ref.resource_id} className="panel nested-panel">
                <strong>
                  {edge.subject_concept_ref.resource_label} {edge.predicate} {edge.object_concept_ref.resource_label}
                </strong>
                <div className="inline-meta">
                  <StatusPill value={edge.support_visibility} />
                  <StatusPill value={edge.verification_state} />
                </div>
              </article>
            ))}
          </div>
        </>
      ) : (
        <EmptyState title="Graph is loading" description="No graph data yet." />
      )}
    </section>
  )
}
