from __future__ import annotations

from datetime import UTC, datetime
from typing import TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_serializer

from cornerstone.domain.enums import (
    AnswerStatus,
    ConceptKind,
    ConsumerScope,
    ContextSpaceKind,
    CuratedLifecycleState,
    DecisionLifecycleState,
    FreshnessState,
    NoMatchReason,
    OriginDisclosureLevel,
    RequestIntent,
    ResourceKind,
    ResponseKind,
    ReviewAction,
    SharedSelectionKind,
    SourceConnectionState,
    SupportItemKind,
    SupportVisibility,
    SyncMode,
    SyncRunStatus,
    SyncTriggerKind,
    RuntimeMode,
    VerificationState,
    VisibilityClass,
    WorkspaceDataState,
)


def _encode_contract_datetime(value: datetime) -> str:
    normalized = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return normalized.isoformat().replace("+00:00", "Z")


class ContractModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    @field_serializer("*", when_used="json", check_fields=False)
    def serialize_datetimes(self, value):
        if isinstance(value, datetime):
            return _encode_contract_datetime(value)
        return value


class ContextSpaceRef(ContractModel):
    context_space_id: str
    context_space_kind: ContextSpaceKind
    context_space_name: str


class ResourceRef(ContractModel):
    resource_kind: ResourceKind
    resource_id: str
    resource_label: str


class SupportItemSummary(ContractModel):
    support_item_id: str
    support_item_kind: SupportItemKind
    visibility_class: VisibilityClass
    source_label: str
    excerpt_or_summary: str | None = None
    origin_disclosure_level: OriginDisclosureLevel | None = None
    source_locator: str | None = None


class ProvenanceSummary(ContractModel):
    support_item_count: int
    visible_support_item_count: int
    restricted_support_present: bool
    freshness_state: FreshnessState
    verification_state: VerificationState | None = None
    promotion_lineage_present: bool


class ConceptPayload(ContractModel):
    concept_id: str
    public_slug: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    definition: str
    owning_domain: str
    review_domain: str
    lifecycle_state: CuratedLifecycleState
    verification_state: VerificationState
    support_visibility: SupportVisibility
    visible_support_items: list[SupportItemSummary] = Field(default_factory=list)
    linked_relation_refs: list[ResourceRef] = Field(default_factory=list)
    linked_decision_refs: list[ResourceRef] = Field(default_factory=list)
    provenance_summary: ProvenanceSummary


class RelationPayload(ContractModel):
    relation_id: str
    subject_concept_ref: ResourceRef
    predicate: str
    object_concept_ref: ResourceRef
    description: str | None = None
    review_domain: str
    lifecycle_state: CuratedLifecycleState
    verification_state: VerificationState
    support_visibility: SupportVisibility
    visible_support_items: list[SupportItemSummary] = Field(default_factory=list)
    linked_decision_refs: list[ResourceRef] = Field(default_factory=list)
    provenance_summary: ProvenanceSummary


class DecisionPayload(ContractModel):
    decision_id: str
    public_slug: str
    title: str
    decision_statement: str
    problem_statement: str | None = None
    rationale: str | None = None
    constraints: list[str] = Field(default_factory=list)
    impact_summary: str | None = None
    owning_domain: str
    review_domain: str
    lifecycle_state: DecisionLifecycleState
    support_visibility: SupportVisibility
    visible_support_items: list[SupportItemSummary] = Field(default_factory=list)
    linked_concept_refs: list[ResourceRef] = Field(default_factory=list)
    linked_relation_refs: list[ResourceRef] = Field(default_factory=list)
    supersedes_ref: ResourceRef | None = None
    superseded_by_ref: ResourceRef | None = None
    provenance_summary: ProvenanceSummary


class AnswerSection(ContractModel):
    heading: str
    body: str


class AnswerPayload(ContractModel):
    answer_status: AnswerStatus
    answer_text: str
    answer_sections: list[AnswerSection] = Field(default_factory=list)
    support_visibility: SupportVisibility
    verification_state: VerificationState
    visible_support_items: list[SupportItemSummary] = Field(default_factory=list)
    cited_concept_refs: list[ResourceRef] = Field(default_factory=list)
    cited_relation_refs: list[ResourceRef] = Field(default_factory=list)
    cited_decision_refs: list[ResourceRef] = Field(default_factory=list)
    provenance_summary: ProvenanceSummary
    follow_up_refs: list[ResourceRef] = Field(default_factory=list)


class SearchResultItem(ContractModel):
    resource_ref: ResourceRef
    match_reason_summary: str
    support_visibility: SupportVisibility | None = None
    lifecycle_state: CuratedLifecycleState | DecisionLifecycleState | None = None
    verification_state: VerificationState | None = None
    provenance_summary: ProvenanceSummary | None = None


class SearchResultsPayload(ContractModel):
    results: list[SearchResultItem] = Field(default_factory=list)
    result_count: int


class GraphEdgePayload(ContractModel):
    relation_ref: ResourceRef
    subject_concept_ref: ResourceRef
    predicate: str
    object_concept_ref: ResourceRef
    support_visibility: SupportVisibility
    verification_state: VerificationState


class GraphSlicePayload(ContractModel):
    root_concept_refs: list[ResourceRef]
    nodes: list[ResourceRef]
    edges: list[GraphEdgePayload] = Field(default_factory=list)


class SourceSummary(ContractModel):
    source_connection_id: str
    source_label: str
    source_connection_state: SourceConnectionState
    freshness_state: FreshnessState
    visibility_class: VisibilityClass
    last_attempted_sync_at: datetime | None = None
    last_successful_sync_at: datetime | None = None
    effective_sync_policy: dict[str, object] = Field(default_factory=dict)
    last_error: str | None = None


class ProvenancePayload(ContractModel):
    subject_ref: ResourceRef
    support_items: list[SupportItemSummary] = Field(default_factory=list)
    source_summaries: list[SourceSummary] = Field(default_factory=list)
    provenance_summary: ProvenanceSummary


class SuggestedFollowUp(ContractModel):
    label: str
    resource_ref: ResourceRef | None = None


class NoMatchPayload(ContractModel):
    reason: NoMatchReason
    request_rewrite_hint: str | None = None
    suggested_follow_up: list[SuggestedFollowUp] = Field(default_factory=list)


PayloadT = TypeVar("PayloadT", bound=BaseModel)


class ContractEnvelope[PayloadT](ContractModel):
    contract_version: str
    response_kind: ResponseKind
    request_intent: RequestIntent
    context_space_ref: ContextSpaceRef
    consumer_scope: ConsumerScope
    payload: PayloadT
    related_refs: list[ResourceRef] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class WorkspaceHomeFeaturedCard(ContractModel):
    resource_ref: ResourceRef
    public_slug: str
    title: str
    eyebrow: str
    summary: str
    support_visibility: SupportVisibility
    lifecycle_state: CuratedLifecycleState | DecisionLifecycleState
    verification_state: VerificationState | None = None
    provenance_summary: ProvenanceSummary


class WorkspaceHomeRecentChange(ContractModel):
    resource_ref: ResourceRef
    public_slug: str
    change_summary: str
    changed_at: datetime
    support_visibility: SupportVisibility
    lifecycle_state: CuratedLifecycleState | DecisionLifecycleState
    verification_state: VerificationState | None = None


class WorkspaceHomeFreshnessAlert(ContractModel):
    source_connection_id: str
    source_label: str
    source_connection_state: SourceConnectionState
    freshness_state: FreshnessState
    last_successful_sync_at: datetime | None = None
    note: str


class ReviewQueueSummary(ContractModel):
    pending_count: int
    review_required_count: int
    officialize_ready_count: int


class SourceHealthSummary(ContractModel):
    total_count: int
    active_count: int
    monitoring_count: int
    stale_count: int
    degraded_count: int
    paused_count: int
    removed_count: int


class WorkspaceHomePayload(ContractModel):
    hero_prompt: str
    featured_answer: ContractEnvelope[AnswerPayload | NoMatchPayload] | None = None
    featured_cards: list[WorkspaceHomeFeaturedCard] = Field(default_factory=list)
    recent_changes: list[WorkspaceHomeRecentChange] = Field(default_factory=list)
    freshness_alerts: list[WorkspaceHomeFreshnessAlert] = Field(default_factory=list)
    review_queue_summary: ReviewQueueSummary
    source_health_summary: SourceHealthSummary


class SourceConnectionStatus(ContractModel):
    id: str
    context_space_id: str
    provider: str
    source_label: str
    source_boundary_locator: str
    template_key: str
    visibility_class: VisibilityClass
    sync_mode: SyncMode
    sync_interval_seconds: int
    source_connection_state: SourceConnectionState
    freshness_state: FreshnessState
    last_attempted_sync_at: datetime | None = None
    last_successful_sync_at: datetime | None = None
    last_error: str | None = None
    effective_sync_policy: dict[str, object] = Field(default_factory=dict)
    next_scheduled_sync_at: datetime | None = None
    last_run_at: datetime | None = None
    last_run_status: SyncRunStatus | None = None
    can_manage: bool = False
    removed_at: datetime | None = None


class SourceConnectionDetail(SourceConnectionStatus):
    provider_credential_ref: str | None = None
    selected_scope_json: dict[str, object] = Field(default_factory=dict)
    sync_checkpoint_json: dict[str, object] = Field(default_factory=dict)


class ConnectorTemplateSummary(ContractModel):
    template_key: str
    provider: str
    label: str
    description: str
    scope_kind: str
    default_visibility_class: VisibilityClass
    recommended_sync_interval_seconds: int
    preview_required: bool = True


class ProviderBindingStartRequest(ContractModel):
    provider: str


class ProviderBindingStartResponse(ContractModel):
    provider: str
    provider_credential_ref: str | None = None
    authorization_url: str | None = None
    binding_state: str | None = None
    account_label: str | None = None
    demo_mode: bool = False


class ProviderBindingCompleteRequest(ContractModel):
    provider: str
    binding_state: str
    code: str


class ProviderBindingSummary(ContractModel):
    provider: str
    provider_credential_ref: str
    account_label: str | None = None


class SourcePreviewItem(ContractModel):
    upstream_id: str
    title: str
    artifact_type: str
    source_locator: str | None = None
    excerpt: str | None = None
    source_updated_at: datetime | None = None


class SourceConnectionPreviewRequest(ContractModel):
    template_key: str
    provider_credential_ref: str | None = None
    source_label: str
    selected_scope_input: str
    visibility_class: VisibilityClass = VisibilityClass.MEMBER_VISIBLE


class SourceConnectionPreviewResponse(ContractModel):
    provider: str
    template_key: str
    resolved_source_boundary_locator: str
    selected_scope_json: dict[str, object] = Field(default_factory=dict)
    suggested_sync_mode: SyncMode
    suggested_sync_interval_seconds: int
    preview_items: list[SourcePreviewItem] = Field(default_factory=list)
    visibility_class: VisibilityClass
    effective_sync_policy: dict[str, object] = Field(default_factory=dict)


class SourceConnectionCreate(ContractModel):
    template_key: str
    provider_credential_ref: str | None = None
    source_label: str
    selected_scope_input: str
    visibility_class: VisibilityClass = VisibilityClass.MEMBER_VISIBLE
    sync_interval_seconds: int | None = None


class SourceConnectionUpdate(ContractModel):
    source_label: str | None = None
    visibility_class: VisibilityClass | None = None
    sync_interval_seconds: int | None = None
    provider_credential_ref: str | None = None
    selected_scope_input: str | None = None


class SyncRunSummary(ContractModel):
    id: str
    source_connection_id: str
    trigger_kind: SyncTriggerKind
    run_status: SyncRunStatus
    started_at: datetime
    finished_at: datetime | None = None
    artifact_count: int = 0
    support_item_count: int = 0
    error_summary: str | None = None


class ReviewQueueItem(ContractModel):
    resource_ref: ResourceRef
    review_domain: str
    lifecycle_state: CuratedLifecycleState | DecisionLifecycleState
    verification_state: VerificationState
    support_visibility: SupportVisibility
    suggested_actions: list[ReviewAction] = Field(default_factory=list)


class ActorSession(ContractModel):
    actor_id: str
    display_name: str
    base_role: str
    token: str
    scoped_capabilities: list[dict[str, str]] = Field(default_factory=list)
    preferred_consumer_scope: ConsumerScope


class ViewerBootstrap(ContractModel):
    workspace: ContextSpaceRef
    personal_context: ContextSpaceRef
    actors: list[ActorSession] = Field(default_factory=list)
    runtime_mode: RuntimeMode
    workspace_data_state: WorkspaceDataState
    linked_source_count: int = 0
    active_source_count: int = 0
    degraded_source_count: int = 0


class SyncRunResult(ContractModel):
    source_connection_id: str
    artifact_count: int
    support_item_count: int
    source_connection_state: SourceConnectionState
    freshness_state: FreshnessState


class DraftConceptCreate(ContractModel):
    context_space_id: str
    canonical_name: str
    definition: str
    owning_domain: str
    concept_kind: ConceptKind = ConceptKind.TERM
    aliases: list[str] = Field(default_factory=list)
    support_item_ids: list[str] = Field(default_factory=list)
    linked_decision_ids: list[str] = Field(default_factory=list)


class DraftRelationCreate(ContractModel):
    context_space_id: str
    subject_concept_id: str
    predicate: str
    object_concept_id: str
    description: str | None = None
    support_item_ids: list[str] = Field(default_factory=list)
    linked_decision_ids: list[str] = Field(default_factory=list)
    workspace_wide: bool = False


class DraftDecisionCreate(ContractModel):
    context_space_id: str
    title: str
    decision_statement: str
    owning_domain: str
    problem_statement: str | None = None
    rationale: str | None = None
    constraints: list[str] = Field(default_factory=list)
    impact_summary: str | None = None
    support_item_ids: list[str] = Field(default_factory=list)
    linked_concept_ids: list[str] = Field(default_factory=list)
    linked_relation_ids: list[str] = Field(default_factory=list)
    supersedes_decision_id: str | None = None


class PromotionRequest(ContractModel):
    personal_support_item_id: str
    workspace_context_id: str
    shared_selection_kind: SharedSelectionKind
    shared_payload: str
    visibility_class: VisibilityClass = VisibilityClass.MEMBER_VISIBLE
    origin_disclosure_level: OriginDisclosureLevel = OriginDisclosureLevel.REDACTED_ORIGIN


class ReviewActionRequest(ContractModel):
    action: ReviewAction
    supersedes_decision_id: str | None = None
