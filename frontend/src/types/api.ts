export type ContextSpace = {
  id: string
  name: string
  namespace: string
  status: string
}

export type Actor = {
  id: string
  context_space_id: string
  actor_type: string
  display_name: string
  roles: string[]
  status: string
}

export type Evidence = {
  id: string
  selector: string
  excerpt: string
  normalized_claim: string
  verification_status: string
  artifact_id: string
  artifact_title: string
  artifact_url: string
}

export type Concept = {
  id: string
  context_space_id: string
  concept_type: string
  canonical_name: string
  aliases: string[]
  definition: string
  status: string
  evidence: Evidence[]
  linked_decisions: string[]
}

export type Relation = {
  id: string
  context_space_id: string
  subject_concept_id: string
  subject_name: string
  predicate: string
  object_concept_id: string
  object_name: string
  description: string
  status: string
  evidence: Evidence[]
  linked_decisions: string[]
}

export type Decision = {
  id: string
  context_space_id: string
  title: string
  problem: string
  decision: string
  rationale: string
  constraints: string[]
  impact: string[]
  status: string
  evidence: Evidence[]
  concepts: string[]
  relations: string[]
}

export type Artifact = {
  id: string
  context_space_id: string
  source_connection_id: string
  external_id: string
  artifact_type: string
  title: string
  canonical_url: string
  status: string
  evidence_count: number
  metadata_json: Record<string, unknown>
}

export type SourceConnection = {
  id: string
  context_space_id: string
  provider: string
  external_scope: string
  sync_mode: string
  sync_interval_seconds: number
  health_status: string
  last_synced_at: string | null
  last_error: string | null
}

export type GraphResponse = {
  nodes: Array<{ id: string; label: string; type: string; status: string }>
  edges: Array<{ id: string; source: string; target: string; label: string; status: string }>
}

export type Stats = {
  concept_count: number
  relation_count: number
  decision_count: number
  artifact_count: number
  evidence_count: number
}

export type Answer = {
  query: string
  summary: string
  concepts: Concept[]
  relations: Relation[]
  decisions: Decision[]
  evidence: Evidence[]
}
