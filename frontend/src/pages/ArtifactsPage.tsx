import { apiGet } from '../api/client'
import { EmptyState } from '../components/EmptyState'
import { PageHeader } from '../components/PageHeader'
import { StatusPill } from '../components/StatusPill'
import { useAsyncData } from '../hooks/useAsyncData'
import type { Artifact } from '../types/api'

type ArtifactsPageProps = {
  contextSpaceId: string | null
}

export function ArtifactsPage({ contextSpaceId }: ArtifactsPageProps) {
  const artifacts = useAsyncData<Artifact[]>(
    () => (contextSpaceId ? apiGet('/artifacts', { context_space_id: contextSpaceId }) : Promise.resolve([])),
    [contextSpaceId],
  )

  if (!contextSpaceId) {
    return <EmptyState title="No artifacts" description="Artifacts appear after the connector syncs source files." />
  }

  return (
    <div className="page-stack">
      <PageHeader
        title="Artifacts and provenance"
        description="Persistent mirrored artifacts from source systems, used as the provenance base for curation."
      />

      <div className="table-panel panel">
        <table className="data-table">
          <thead>
            <tr>
              <th>Artifact</th>
              <th>Type</th>
              <th>Evidence</th>
              <th>Status</th>
              <th>Source</th>
            </tr>
          </thead>
          <tbody>
            {(artifacts.data ?? []).map((artifact) => (
              <tr key={artifact.id}>
                <td>
                  <strong>{artifact.title}</strong>
                  <div className="table-meta">{artifact.external_id}</div>
                </td>
                <td>{artifact.artifact_type}</td>
                <td>{artifact.evidence_count}</td>
                <td>
                  <StatusPill value={artifact.status} />
                </td>
                <td>
                  <a href={artifact.canonical_url} target="_blank" rel="noreferrer">
                    Open file
                  </a>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
