export type ContextSpaceRef = {
  context_space_id: string
  context_space_kind: 'workspace' | 'personal'
  context_space_name: string
}

export type ResourceRef = {
  resource_kind: 'concept' | 'relation' | 'decision' | 'artifact' | 'support_item'
  resource_id: string
  resource_label: string
}

export type SupportItemSummary = {
  support_item_id: string
  support_item_kind: 'evidence_fragment' | 'promoted_support'
  visibility_class: 'member_visible' | 'evidence_only'
  source_label: string
  excerpt_or_summary?: string | null
  origin_disclosure_level?: 'named_origin' | 'redacted_origin' | 'hidden_origin' | null
  source_locator?: string | null
}

export type ProvenanceSummary = {
  support_item_count: number
  visible_support_item_count: number
  restricted_support_present: boolean
  freshness_state: string
  verification_state?: string | null
  promotion_lineage_present: boolean
}

export type ConceptPayload = {
  concept_id: string
  public_slug: string
  canonical_name: string
  aliases: string[]
  definition: string
  owning_domain: string
  review_domain: string
  lifecycle_state: string
  verification_state: string
  support_visibility: string
  visible_support_items: SupportItemSummary[]
  linked_relation_refs: ResourceRef[]
  linked_decision_refs: ResourceRef[]
  provenance_summary: ProvenanceSummary
}

export type RelationPayload = {
  relation_id: string
  subject_concept_ref: ResourceRef
  predicate: string
  object_concept_ref: ResourceRef
  description?: string | null
  review_domain: string
  lifecycle_state: string
  verification_state: string
  support_visibility: string
  visible_support_items: SupportItemSummary[]
  linked_decision_refs: ResourceRef[]
  provenance_summary: ProvenanceSummary
}

export type DecisionPayload = {
  decision_id: string
  title: string
  decision_statement: string
  problem_statement?: string | null
  rationale?: string | null
  constraints: string[]
  impact_summary?: string | null
  owning_domain: string
  review_domain: string
  lifecycle_state: string
  support_visibility: string
  visible_support_items: SupportItemSummary[]
  linked_concept_refs: ResourceRef[]
  linked_relation_refs: ResourceRef[]
  supersedes_ref?: ResourceRef | null
  superseded_by_ref?: ResourceRef | null
  provenance_summary: ProvenanceSummary
}

export type AnswerPayload = {
  answer_status: string
  answer_text: string
  answer_sections: Array<{ heading: string; body: string }>
  support_visibility: string
  verification_state: string
  visible_support_items: SupportItemSummary[]
  cited_concept_refs: ResourceRef[]
  cited_relation_refs: ResourceRef[]
  cited_decision_refs: ResourceRef[]
  provenance_summary: ProvenanceSummary
  follow_up_refs: ResourceRef[]
}

export type SearchResultItem = {
  resource_ref: ResourceRef
  match_reason_summary: string
  support_visibility?: string | null
  lifecycle_state?: string | null
  verification_state?: string | null
  provenance_summary?: ProvenanceSummary | null
}

export type SearchResultsPayload = {
  results: SearchResultItem[]
  result_count: number
}

export type GraphEdgePayload = {
  relation_ref: ResourceRef
  subject_concept_ref: ResourceRef
  predicate: string
  object_concept_ref: ResourceRef
  support_visibility: string
  verification_state: string
}

export type GraphSlicePayload = {
  root_concept_refs: ResourceRef[]
  nodes: ResourceRef[]
  edges: GraphEdgePayload[]
}

export type SourceSummary = {
  source_connection_id: string
  source_label: string
  source_connection_state: string
  freshness_state: string
  visibility_class: string
  last_attempted_sync_at?: string | null
  last_successful_sync_at?: string | null
  effective_sync_policy: Record<string, unknown>
  last_error?: string | null
}

export type ProvenancePayload = {
  subject_ref: ResourceRef
  support_items: SupportItemSummary[]
  source_summaries: SourceSummary[]
  provenance_summary: ProvenanceSummary
}

export type NoMatchPayload = {
  reason: string
  request_rewrite_hint?: string | null
  suggested_follow_up: Array<{ label: string; resource_ref?: ResourceRef | null }>
}

export type ContractEnvelope<T> = {
  contract_version: string
  response_kind: string
  request_intent: string
  context_space_ref: ContextSpaceRef
  consumer_scope: 'member' | 'review' | 'admin'
  payload: T
  related_refs: ResourceRef[]
  warnings: string[]
}

export type SourceConnectionStatus = {
  id: string
  context_space_id: string
  provider: string
  source_label: string
  source_boundary_locator: string
  template_key: string
  visibility_class: string
  sync_mode: string
  sync_interval_seconds: number
  source_connection_state: string
  freshness_state: string
  last_attempted_sync_at?: string | null
  last_successful_sync_at?: string | null
  last_error?: string | null
  effective_sync_policy: Record<string, unknown>
  next_scheduled_sync_at?: string | null
  last_run_at?: string | null
  last_run_status?: string | null
  can_manage: boolean
  removed_at?: string | null
}

export type SourceConnectionDetail = SourceConnectionStatus & {
  provider_credential_ref?: string | null
  selected_scope_json: Record<string, unknown>
  sync_checkpoint_json: Record<string, unknown>
}

export type ConnectorTemplateSummary = {
  template_key: string
  provider: string
  label: string
  description: string
  scope_kind: string
  default_visibility_class: 'member_visible' | 'evidence_only'
  recommended_sync_interval_seconds: number
  preview_required: boolean
}

export type ProviderBindingStartResponse = {
  provider: string
  provider_credential_ref?: string | null
  authorization_url?: string | null
  binding_state?: string | null
  account_label?: string | null
  demo_mode: boolean
}

export type ProviderBindingSummary = {
  provider: string
  provider_credential_ref: string
  account_label?: string | null
}

export type SourcePreviewItem = {
  upstream_id: string
  title: string
  artifact_type: string
  source_locator?: string | null
  excerpt?: string | null
  source_updated_at?: string | null
}

export type SourceConnectionPreviewResponse = {
  provider: string
  template_key: string
  resolved_source_boundary_locator: string
  selected_scope_json: Record<string, unknown>
  suggested_sync_mode: string
  suggested_sync_interval_seconds: number
  preview_items: SourcePreviewItem[]
  visibility_class: 'member_visible' | 'evidence_only'
  effective_sync_policy: Record<string, unknown>
}

export type SourceConnectionCreatePayload = {
  template_key: string
  provider_credential_ref?: string | null
  source_label: string
  selected_scope_input: string
  visibility_class: 'member_visible' | 'evidence_only'
  sync_interval_seconds?: number | null
}

export type SourceConnectionUpdatePayload = {
  source_label?: string
  visibility_class?: 'member_visible' | 'evidence_only'
  sync_interval_seconds?: number
  provider_credential_ref?: string | null
  selected_scope_input?: string
}

export type SyncRunSummary = {
  id: string
  source_connection_id: string
  trigger_kind: string
  run_status: string
  started_at: string
  finished_at?: string | null
  artifact_count: number
  support_item_count: number
  error_summary?: string | null
}

export type ReviewQueueItem = {
  resource_ref: ResourceRef
  review_domain: string
  lifecycle_state: string
  verification_state: string
  support_visibility: string
  suggested_actions: string[]
}

export type ActorSession = {
  actor_id: string
  display_name: string
  base_role: string
  token: string
  scoped_capabilities: Array<{ capability: string; scope: string }>
  preferred_consumer_scope: 'member' | 'review' | 'admin'
}

export type ViewerBootstrap = {
  workspace: ContextSpaceRef
  personal_context: ContextSpaceRef
  actors: ActorSession[]
}

export type SyncRunResult = {
  source_connection_id: string
  artifact_count: number
  support_item_count: number
  source_connection_state: string
  freshness_state: string
}
