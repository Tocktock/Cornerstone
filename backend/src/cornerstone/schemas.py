from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> str:
    return str(uuid4())


def to_camel(value: str) -> str:
    parts = value.split("_")
    return parts[0] + "".join(part.capitalize() for part in parts[1:])


def normalize_concept_term(value: str) -> str:
    """Normalize user-facing ontology terms for alias matching.

    This intentionally keeps punctuation and domain-specific symbols intact while
    making whitespace and case stable. v1.3.0 uses this for exact/partial
    ontology search only; it is not a semantic or embedding normalizer.
    """

    return " ".join(value.strip().casefold().split())


def clean_concept_aliases(aliases: list[str], *, primary_name: str | None = None) -> list[str]:
    """Return display aliases deduplicated by normalized term."""

    normalized_primary = normalize_concept_term(primary_name) if primary_name else None
    seen: set[str] = set()
    cleaned: list[str] = []
    for alias in aliases:
        display_alias = " ".join(alias.strip().split())
        if not display_alias:
            continue
        normalized = normalize_concept_term(display_alias)
        if normalized_primary is not None and normalized == normalized_primary:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        cleaned.append(display_alias)
    return cleaned


class APIModel(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        use_enum_values=True,
        arbitrary_types_allowed=False,
    )


class DataSourceType(StrEnum):
    NOTION = "notion"
    SLACK = "slack"
    GOOGLE_DOCS = "google_docs"
    GOOGLE_DRIVE = "google_drive"
    GITHUB = "github"
    MANUAL = "manual"


class DataSourceStatus(StrEnum):
    # Legacy aggregate status kept for backward-compatible API consumers.
    # Source Studio should prefer auth_status, connection_status, sync_status,
    # freshness_state, and next_action.
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    PENDING_AUTH = "pending_auth"
    CONNECTED = "connected"
    SYNC_PENDING = "sync_pending"
    SYNCING = "syncing"
    DEGRADED = "degraded"
    FAILED = "failed"
    STALE = "stale"


class DataSourceAuthStatus(StrEnum):
    NOT_STARTED = "not_started"
    INTENT_CREATED = "intent_created"
    OAUTH_REDIRECTED = "oauth_redirected"
    AUTHORIZED = "authorized"
    AUTH_FAILED = "auth_failed"
    REVOKED = "revoked"


class DataSourceConnectionStatus(StrEnum):
    UNTESTED = "untested"
    TEST_PASSED = "test_passed"
    TEST_FAILED = "test_failed"
    PERMISSION_LIMITED = "permission_limited"


class DataSourceSyncStatus(StrEnum):
    NEVER_SYNCED = "never_synced"
    QUEUED = "queued"
    SYNCING = "syncing"
    WAITING_RETRY = "waiting_retry"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    DEGRADED = "degraded"
    CANCELLED = "cancelled"


class DataSourceNextAction(StrEnum):
    CONNECT = "connect"
    COMPLETE_OAUTH = "complete_oauth"
    TEST_CONNECTION = "test_connection"
    DISCOVER_SOURCES = "discover_sources"
    SELECT_SOURCES = "select_sources"
    GRANT_PERMISSION = "grant_permission"
    RUN_FIRST_SYNC = "run_first_sync"
    REVIEW_EVIDENCE = "review_evidence"
    RECONNECT = "reconnect"
    RETRY_SYNC = "retry_sync"
    NONE = "none"


class FreshnessState(StrEnum):
    FRESH = "fresh"
    AGING = "aging"
    STALE = "stale"
    UNKNOWN = "unknown"
    MIXED = "mixed"


class ExtractionStatus(StrEnum):
    PENDING = "pending"
    COMPLETE = "complete"
    FAILED = "failed"


class EvidenceFragmentType(StrEnum):
    DEFINITION = "definition"
    DECISION = "decision"
    POLICY = "policy"
    REQUIREMENT = "requirement"
    EXAMPLE = "example"
    CLAIM = "claim"
    OPEN_QUESTION = "open_question"


class TrustState(StrEnum):
    UNREVIEWED = "unreviewed"
    REVIEWED = "reviewed"
    REJECTED = "rejected"
    CONFLICTED = "conflicted"


class ConceptStatus(StrEnum):
    CANDIDATE = "candidate"
    REVIEWING = "reviewing"
    OFFICIAL = "official"
    CONFLICTED = "conflicted"
    DEPRECATED = "deprecated"
    SUPERSEDED = "superseded"


class TrustLabel(StrEnum):
    OFFICIAL = "official"
    EVIDENCE_SUPPORTED = "evidence_supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    STALE = "stale"
    CONFLICTED = "conflicted"
    UNSUPPORTED = "unsupported"


class RelationType(StrEnum):
    IS_A = "is_a"
    PART_OF = "part_of"
    DEPENDS_ON = "depends_on"
    PRECEDES = "precedes"
    FOLLOWS = "follows"
    PRODUCES = "produces"
    CONSUMES = "consumes"
    UPDATES = "updates"
    VALIDATES = "validates"
    TRIGGERS = "triggers"
    BLOCKS = "blocks"
    CONFLICTS_WITH = "conflicts_with"
    SUPERSEDES = "supersedes"
    OWNED_BY = "owned_by"
    USED_BY = "used_by"
    CREATED_BY = "created_by"
    GOVERNED_BY = "governed_by"
    SOURCE_OF_TRUTH_FOR = "source_of_truth_for"
    RELATED_TO = "related_to"


class RelationStatus(StrEnum):
    CANDIDATE = "candidate"
    REVIEWING = "reviewing"
    OFFICIAL = "official"
    REJECTED = "rejected"


class OntologyGraphMode(StrEnum):
    OFFICIAL = "official"
    CANDIDATE = "candidate"
    MIXED = "mixed"


class OntologySearchMatchType(StrEnum):
    NAME = "name"
    ALIAS = "alias"
    PARTIAL = "partial"


class OntologyExtractionProvider(StrEnum):
    LOCAL_RULE_BASED = "local_rule_based"
    LIVE_LLM = "live_llm"


class OntologyExtractionRunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class OntologyReExtractionRunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"


class OntologyReExtractionTrigger(StrEnum):
    CONNECTOR_SYNC = "connector_sync"
    SCHEDULED_SYNC = "scheduled_sync"
    MANUAL_UPLOAD = "manual_upload"
    MANUAL_SYNC = "manual_sync"
    WEBHOOK = "webhook"
    MANUAL_REQUEST = "manual_request"


class OntologyProofStepStatus(StrEnum):
    PLANNED = "planned"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


class OntologyProofRunStatus(StrEnum):
    PLANNED = "planned"
    PASSED = "passed"
    FAILED = "failed"


class OntologyCandidateStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    MERGED = "merged"


class ErrorInfo(APIModel):
    code: str
    message: str
    occurred_at: datetime = Field(default_factory=utc_now)


class DataSource(APIModel):
    id: str = Field(default_factory=new_id)
    type: DataSourceType
    name: str
    status: DataSourceStatus
    production_enabled: bool = True
    created_at: datetime = Field(default_factory=utc_now)
    auth_status: DataSourceAuthStatus = DataSourceAuthStatus.NOT_STARTED
    connection_status: DataSourceConnectionStatus = DataSourceConnectionStatus.UNTESTED
    sync_status: DataSourceSyncStatus = DataSourceSyncStatus.NEVER_SYNCED
    next_action: DataSourceNextAction = DataSourceNextAction.CONNECT
    last_connection_test_at: datetime | None = None
    last_discovery_at: datetime | None = None
    last_sync_at: datetime | None = None
    last_successful_sync_at: datetime | None = None
    last_error: ErrorInfo | None = None
    freshness_state: FreshnessState = FreshnessState.UNKNOWN
    sync_freshness_state: FreshnessState = FreshnessState.UNKNOWN
    content_freshness_state: FreshnessState = FreshnessState.UNKNOWN
    artifact_count: int = 0
    evidence_fragment_count: int = 0
    discovered_object_count: int = 0
    selected_object_count: int = 0


class CreateDataSourceRequest(APIModel):
    type: DataSourceType
    name: str = Field(min_length=1)
    production_enabled: bool = True


class SourceStudioResponse(APIModel):
    production_enabled: bool
    has_real_sources: bool
    onboarding_required: bool
    message: str
    sources: list[DataSource]


class SourceObject(APIModel):
    """Provider-normalized source object ready for Artifact ingestion.

    Connectors may parse provider-specific payloads however they need, but they must cross
    this boundary before Artifact/Evidence creation. Core sync logic must not inspect
    Notion/Slack/Google/GitHub-specific shapes.
    """

    source_external_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    content: str = ""
    source_url: HttpUrl | str | None = None
    source_updated_at: datetime | None = None
    source_object_type: str = "unknown"
    provider_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_url")
    @classmethod
    def stringify_source_url(cls, value: HttpUrl | str | None) -> str | None:
        if value is None:
            return None
        return str(value)


class SyncSourceRequest(APIModel):
    objects: list[SourceObject] = Field(default_factory=list)
    queue_ontology_reextraction: bool = Field(
        default=True,
        validation_alias=AliasChoices("queueOntologyReextraction", "queueOntologyReExtraction"),
    )
    run_ontology_reextraction_inline: bool = Field(
        default=False,
        validation_alias=AliasChoices("runOntologyReextractionInline", "runOntologyReExtractionInline"),
    )
    ontology_focus_concept: str | None = Field(default=None, min_length=1)


class ManualTextUpload(APIModel):
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    source_external_id: str | None = None
    source_url: HttpUrl | str | None = None
    source_updated_at: datetime | None = None
    provider_metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("source_url")
    @classmethod
    def stringify_source_url(cls, value: HttpUrl | str | None) -> str | None:
        if value is None:
            return None
        return str(value)


class ManualTextUploadRequest(APIModel):
    objects: list[ManualTextUpload] = Field(min_length=1)
    queue_ontology_reextraction: bool = Field(
        default=True,
        validation_alias=AliasChoices("queueOntologyReextraction", "queueOntologyReExtraction"),
    )
    run_ontology_reextraction_inline: bool = Field(
        default=False,
        validation_alias=AliasChoices("runOntologyReextractionInline", "runOntologyReExtractionInline"),
    )
    ontology_focus_concept: str | None = Field(default=None, min_length=1)


class Artifact(APIModel):
    id: str = Field(default_factory=new_id)
    datasource_id: str
    source_type: DataSourceType
    source_external_id: str
    source_url: str | None = None
    source_object_type: str = "unknown"
    title: str
    raw_content_hash: str
    captured_at: datetime = Field(default_factory=utc_now)
    source_updated_at: datetime | None = None
    freshness_state: FreshnessState = FreshnessState.UNKNOWN
    extraction_status: ExtractionStatus = ExtractionStatus.PENDING
    provider_metadata: dict[str, Any] = Field(default_factory=dict)


class QuoteRange(APIModel):
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_offsets(self) -> QuoteRange:
        if self.end_offset < self.start_offset:
            raise ValueError("end_offset must be greater than or equal to start_offset")
        return self


class Provenance(APIModel):
    data_source_id: str
    source_type: DataSourceType
    source_external_id: str
    source_url: str | None = None
    artifact_title: str
    captured_at: datetime
    source_updated_at: datetime | None = None
    quote_range: QuoteRange | None = None


class EvidenceFragment(APIModel):
    id: str = Field(default_factory=new_id)
    artifact_id: str
    text: str = Field(min_length=1)
    fragment_type: EvidenceFragmentType
    provenance: Provenance
    trust_state: TrustState = TrustState.UNREVIEWED
    freshness_state: FreshnessState = FreshnessState.UNKNOWN
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_required_provenance(self) -> EvidenceFragment:
        if not self.artifact_id:
            raise ValueError("EvidenceFragment requires artifact_id")
        if not self.provenance.data_source_id:
            raise ValueError("EvidenceFragment requires provenance.data_source_id")
        if not self.provenance.source_external_id:
            raise ValueError("EvidenceFragment requires provenance.source_external_id")
        if not self.provenance.artifact_title:
            raise ValueError("EvidenceFragment requires provenance.artifact_title")
        if not self.provenance.captured_at:
            raise ValueError("EvidenceFragment requires provenance.captured_at")
        return self


class ReviewEvidenceRequest(APIModel):
    trust_state: TrustState
    reviewed_by: str = Field(min_length=1)
    review_note: str | None = None

    @model_validator(mode="after")
    def validate_review_target_state(self) -> ReviewEvidenceRequest:
        if self.trust_state == TrustState.UNREVIEWED:
            raise ValueError("Evidence review must mark the fragment as reviewed, rejected, or conflicted.")
        return self


class EvidenceReviewQueueItem(APIModel):
    evidence_fragment: EvidenceFragment
    artifact: Artifact
    data_source: DataSource
    linked_concept_ids: list[str] = Field(default_factory=list)
    linked_decision_record_ids: list[str] = Field(default_factory=list)
    suggested_next_actions: list[str] = Field(default_factory=list)


class EvidenceReviewQueueResponse(APIModel):
    items: list[EvidenceReviewQueueItem]
    total_count: int
    unreviewed_count: int = 0
    reviewed_count: int = 0
    rejected_count: int = 0
    conflicted_count: int = 0


class CreateConceptFromEvidenceRequest(APIModel):
    name: str = Field(min_length=1)
    short_definition: str = Field(min_length=1)
    body: str | None = None
    owner: str | None = None
    created_by: str = Field(min_length=1)
    status: ConceptStatus = ConceptStatus.REVIEWING

    @model_validator(mode="after")
    def validate_candidate_status(self) -> CreateConceptFromEvidenceRequest:
        if self.status not in {ConceptStatus.CANDIDATE, ConceptStatus.REVIEWING}:
            raise ValueError("Concept candidates created from evidence must be candidate or reviewing.")
        return self


class SyncSourceResponse(APIModel):
    data_source: DataSource
    artifacts: list[Artifact]
    evidence_fragments: list[EvidenceFragment]
    artifact_created_count: int = 0
    artifact_reused_count: int = 0
    artifact_changed_count: int = 0
    evidence_created_count: int = 0
    created_artifact_ids: list[str] = Field(default_factory=list)
    reused_artifact_ids: list[str] = Field(default_factory=list)
    changed_artifact_ids: list[str] = Field(default_factory=list)
    ontology_reextraction_run_id: str | None = None
    ontology_reextraction_status: str | None = None


class DecisionRecord(APIModel):
    id: str = Field(default_factory=new_id)
    title: str
    decision: str
    reason: str
    alternatives_considered: list[str] = Field(default_factory=list)
    decided_by: str
    decided_at: datetime = Field(default_factory=utc_now)
    evidence_fragment_ids: list[str] = Field(default_factory=list)
    affected_concept_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class CreateDecisionRecordRequest(APIModel):
    title: str = Field(min_length=1)
    decision: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    alternatives_considered: list[str] = Field(default_factory=list)
    decided_by: str = Field(min_length=1)
    evidence_fragment_ids: list[str] = Field(default_factory=list)
    affected_concept_ids: list[str] = Field(default_factory=list)


class Concept(APIModel):
    id: str = Field(default_factory=new_id)
    name: str
    aliases: list[str] = Field(default_factory=list)
    short_definition: str
    body: str | None = None
    status: ConceptStatus = ConceptStatus.CANDIDATE
    owner: str | None = None
    evidence_fragment_ids: list[str] = Field(default_factory=list)
    decision_record_ids: list[str] = Field(default_factory=list)
    created_by: str
    officialized_by: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    last_reviewed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_aliases(self) -> Concept:
        self.aliases = clean_concept_aliases(self.aliases, primary_name=self.name)
        return self


class CreateConceptRequest(APIModel):
    name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    short_definition: str = Field(min_length=1)
    body: str | None = None
    owner: str | None = None
    evidence_fragment_ids: list[str] = Field(default_factory=list)
    decision_record_ids: list[str] = Field(default_factory=list)
    created_by: str = Field(default="system")

    @model_validator(mode="after")
    def validate_aliases(self) -> CreateConceptRequest:
        self.aliases = clean_concept_aliases(self.aliases, primary_name=self.name)
        return self


class OfficializeConceptRequest(APIModel):
    reviewed_by: str = Field(min_length=1)


class ConceptRelation(APIModel):
    id: str = Field(default_factory=new_id)
    source_concept_id: str
    target_concept_id: str
    relation_type: RelationType
    status: RelationStatus = RelationStatus.CANDIDATE
    evidence_fragment_ids: list[str] = Field(default_factory=list)
    decision_record_id: str | None = None
    created_by: str
    officialized_by: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    last_reviewed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_distinct_concepts(self) -> ConceptRelation:
        if self.source_concept_id == self.target_concept_id:
            raise ValueError("ConceptRelation source and target Concepts must differ.")
        return self


class CreateConceptRelationRequest(APIModel):
    source_concept_id: str = Field(min_length=1)
    target_concept_id: str = Field(min_length=1)
    relation_type: RelationType
    evidence_fragment_ids: list[str] = Field(default_factory=list)
    decision_record_id: str | None = None
    created_by: str = Field(min_length=1)
    status: RelationStatus = RelationStatus.CANDIDATE

    @model_validator(mode="after")
    def validate_candidate_status(self) -> CreateConceptRelationRequest:
        if self.source_concept_id == self.target_concept_id:
            raise ValueError("ConceptRelation source and target Concepts must differ.")
        if self.status not in {RelationStatus.CANDIDATE, RelationStatus.REVIEWING}:
            raise ValueError("Created ConceptRelations must be candidate or reviewing.")
        return self


class OfficializeConceptRelationRequest(APIModel):
    reviewed_by: str = Field(min_length=1)


class ConceptRef(APIModel):
    id: str
    name: str
    status: ConceptStatus


class OntologySearchResult(APIModel):
    id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    short_definition: str
    status: ConceptStatus
    matched_by: OntologySearchMatchType
    matched_value: str
    score: float = Field(ge=0, le=1)


class OntologySearchResponse(APIModel):
    query: str
    mode: OntologyGraphMode = OntologyGraphMode.OFFICIAL
    results: list[OntologySearchResult] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utc_now)


class ReviewProvenance(APIModel):
    created_by: str | None = None
    officialized_by: str | None = None
    reviewed_by: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_reviewed_at: datetime | None = None
    status: str | None = None


class GraphEntitySupportSummary(APIModel):
    evidence_count: int = 0
    reviewed_evidence_count: int = 0
    unreviewed_evidence_count: int = 0
    conflicted_evidence_count: int = 0
    stale_evidence_count: int = 0
    unknown_freshness_evidence_count: int = 0
    invalid_evidence_reference_count: int = 0


class OntologyGraphSupportSummary(APIModel):
    node_count: int = 0
    edge_count: int = 0
    evidence_count: int = 0
    reviewed_evidence_count: int = 0
    unreviewed_evidence_count: int = 0
    conflicted_evidence_count: int = 0
    stale_evidence_count: int = 0
    unknown_freshness_evidence_count: int = 0
    official_node_count: int = 0
    non_official_node_count: int = 0
    official_edge_count: int = 0
    non_official_edge_count: int = 0
    invalid_evidence_reference_count: int = 0
    invalid_relation_count: int = 0


class OntologyCandidateGraphSummary(APIModel):
    pending_concept_candidate_count: int = 0
    pending_relation_candidate_count: int = 0
    approved_concept_candidate_count: int = 0
    approved_relation_candidate_count: int = 0
    rejected_concept_candidate_count: int = 0
    rejected_relation_candidate_count: int = 0
    merged_concept_candidate_count: int = 0
    merged_relation_candidate_count: int = 0
    has_pending_candidates: bool = False
    note: str | None = None


class OntologyGraphExplanation(APIModel):
    summary: str
    ssot_status: str
    trust_reason: str
    graph_scope: str
    evidence_policy: str
    candidate_boundary: str
    review_summary: str
    recommended_next_actions: list[str] = Field(default_factory=list)


class OntologyGraphNode(APIModel):
    id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    short_definition: str
    status: ConceptStatus
    owner: str | None = None
    is_focus: bool = False
    evidence_fragment_ids: list[str] = Field(default_factory=list)
    decision_record_ids: list[str] = Field(default_factory=list)
    review_provenance: ReviewProvenance | None = None
    support_summary: GraphEntitySupportSummary = Field(default_factory=GraphEntitySupportSummary)
    explanation: str | None = None


class OntologyGraphEdge(APIModel):
    id: str
    source_concept_id: str
    target_concept_id: str
    source_concept_name: str | None = None
    target_concept_name: str | None = None
    focus_direction: str | None = None
    relation_type: RelationType
    status: RelationStatus
    evidence_fragment_ids: list[str] = Field(default_factory=list)
    decision_record_id: str | None = None
    review_provenance: ReviewProvenance | None = None
    support_summary: GraphEntitySupportSummary = Field(default_factory=GraphEntitySupportSummary)
    explanation: str | None = None


class OntologyGraphVisualizationNode(APIModel):
    id: str
    label: str
    category: str
    display_state: str
    is_focus: bool = False
    group: str = "neighbor"
    citation_panel: dict[str, Any] = Field(default_factory=dict)
    review_provenance_panel: dict[str, Any] = Field(default_factory=dict)


class OntologyGraphVisualizationEdge(APIModel):
    id: str
    source_id: str
    target_id: str
    label: str
    direction: str
    display_state: str
    citation_panel: dict[str, Any] = Field(default_factory=dict)
    review_provenance_panel: dict[str, Any] = Field(default_factory=dict)


class OntologyGraphVisualization(APIModel):
    focus_concept_id: str | None = None
    empty_state: str | None = None
    nodes: list[OntologyGraphVisualizationNode] = Field(default_factory=list)
    edges: list[OntologyGraphVisualizationEdge] = Field(default_factory=list)
    state_legend: dict[str, str] = Field(default_factory=dict)
    layout_hints: dict[str, Any] = Field(default_factory=dict)
    citation_panel: dict[str, Any] = Field(default_factory=dict)
    review_provenance_panel: dict[str, Any] = Field(default_factory=dict)


class ConceptRelationRef(APIModel):
    id: str
    source_concept_id: str
    target_concept_id: str
    relation_type: RelationType
    status: RelationStatus


class DecisionRecordRef(APIModel):
    id: str
    title: str


class CitationSupportType(StrEnum):
    CONCEPT = "concept"
    CONCEPT_RELATION = "concept_relation"
    DECISION_RECORD = "decision_record"
    EVIDENCE_FRAGMENT = "evidence_fragment"


class CitationSupportRef(APIModel):
    entity_type: CitationSupportType
    entity_id: str
    relationship: str = "supports"


class EvidenceCitation(APIModel):
    evidence_fragment_id: str
    artifact_id: str
    text: str
    data_source_id: str | None = None
    source_type: DataSourceType | None = None
    source_external_id: str | None = None
    source_url: str | None = None
    artifact_title: str
    captured_at: datetime
    source_updated_at: datetime | None = None
    freshness_state: FreshnessState
    trust_state: TrustState
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    supports: list[CitationSupportRef] = Field(default_factory=list)
    is_valid: bool = True
    validity_errors: list[str] = Field(default_factory=list)


class FreshnessSummary(APIModel):
    state: FreshnessState
    stale_evidence_count: int = 0
    unknown_evidence_count: int = 0

class OntologyGraphResponse(APIModel):
    response_id: str = Field(default_factory=new_id)
    query: str
    mode: OntologyGraphMode = OntologyGraphMode.OFFICIAL
    depth: int = Field(ge=0)
    focus_concept: OntologyGraphNode | None = None
    nodes: list[OntologyGraphNode] = Field(default_factory=list)
    edges: list[OntologyGraphEdge] = Field(default_factory=list)
    evidence: list[EvidenceCitation] = Field(default_factory=list)
    freshness: FreshnessSummary
    trust_label: TrustLabel
    support_summary: OntologyGraphSupportSummary = Field(default_factory=OntologyGraphSupportSummary)
    candidate_summary: OntologyCandidateGraphSummary = Field(default_factory=OntologyCandidateGraphSummary)
    explanation: OntologyGraphExplanation | None = None
    visualization: OntologyGraphVisualization = Field(default_factory=OntologyGraphVisualization)
    limitations: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utc_now)
    official_graph_available: bool = False


class CreateOntologyExtractionRunRequest(APIModel):
    evidence_fragment_ids: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    focus_concept: str | None = Field(default=None, min_length=1)
    provider: OntologyExtractionProvider = OntologyExtractionProvider.LOCAL_RULE_BASED
    model_name: str | None = Field(default=None, min_length=1)
    prompt_version: str | None = Field(default=None, min_length=1)
    requested_by: str = Field(default="system", min_length=1)
    max_evidence_fragments: int = Field(default=50, ge=1, le=200)

    @model_validator(mode="after")
    def validate_scope(self) -> CreateOntologyExtractionRunRequest:
        if not self.evidence_fragment_ids and not self.artifact_ids:
            raise ValueError("Ontology extraction requires evidence_fragment_ids or artifact_ids.")
        return self


class OntologyExtractionRun(APIModel):
    id: str = Field(default_factory=new_id)
    provider: OntologyExtractionProvider = OntologyExtractionProvider.LOCAL_RULE_BASED
    model_name: str
    prompt_version: str
    status: OntologyExtractionRunStatus = OntologyExtractionRunStatus.QUEUED
    requested_by: str
    focus_concept: str | None = None
    evidence_fragment_ids: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    concept_candidate_count: int = 0
    relation_candidate_count: int = 0
    warning_count: int = 0
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None


class ConceptCandidate(APIModel):
    id: str = Field(default_factory=new_id)
    extraction_run_id: str
    name: str = Field(min_length=1)
    normalized_name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    proposed_definition: str = Field(min_length=1)
    concept_type: str = Field(default="unknown", min_length=1)
    evidence_fragment_ids: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    status: OntologyCandidateStatus = OntologyCandidateStatus.PENDING
    matched_existing_concept_id: str | None = None
    rationale: str | None = None
    validation_errors: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    review_note: str | None = None
    promoted_concept_id: str | None = None
    merged_into_concept_id: str | None = None

    @model_validator(mode="after")
    def validate_candidate(self) -> ConceptCandidate:
        self.normalized_name = normalize_concept_term(self.normalized_name or self.name)
        self.aliases = clean_concept_aliases(self.aliases, primary_name=self.name)
        if not self.evidence_fragment_ids:
            raise ValueError("ConceptCandidate requires at least one evidence fragment.")
        return self


class RelationCandidate(APIModel):
    id: str = Field(default_factory=new_id)
    extraction_run_id: str
    source_name: str = Field(min_length=1)
    target_name: str = Field(min_length=1)
    normalized_source_name: str = Field(min_length=1)
    normalized_target_name: str = Field(min_length=1)
    source_candidate_id: str | None = None
    target_candidate_id: str | None = None
    source_concept_id: str | None = None
    target_concept_id: str | None = None
    relation_type: RelationType
    evidence_fragment_ids: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1)
    status: OntologyCandidateStatus = OntologyCandidateStatus.PENDING
    validation_errors: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    review_note: str | None = None
    promoted_relation_id: str | None = None
    merged_into_relation_id: str | None = None

    @model_validator(mode="after")
    def validate_relation_candidate(self) -> RelationCandidate:
        self.normalized_source_name = normalize_concept_term(self.normalized_source_name or self.source_name)
        self.normalized_target_name = normalize_concept_term(self.normalized_target_name or self.target_name)
        if self.normalized_source_name == self.normalized_target_name:
            raise ValueError("RelationCandidate source and target must differ.")
        if not self.evidence_fragment_ids:
            raise ValueError("RelationCandidate requires at least one evidence fragment.")
        return self




class EditConceptCandidateRequest(APIModel):
    edited_by: str = Field(min_length=1)
    name: str | None = Field(default=None, min_length=1)
    aliases: list[str] | None = None
    proposed_definition: str | None = Field(default=None, min_length=1)
    concept_type: str | None = Field(default=None, min_length=1)
    evidence_fragment_ids: list[str] | None = Field(default=None, min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    rationale: str | None = None


class ApproveConceptCandidateRequest(APIModel):
    reviewed_by: str = Field(min_length=1)
    name: str | None = Field(default=None, min_length=1)
    aliases: list[str] | None = None
    short_definition: str | None = Field(default=None, min_length=1)
    body: str | None = None
    owner: str | None = None
    review_note: str | None = None


class RejectOntologyCandidateRequest(APIModel):
    reviewed_by: str = Field(min_length=1)
    review_note: str | None = None


class MergeConceptCandidateRequest(APIModel):
    reviewed_by: str = Field(min_length=1)
    target_concept_id: str = Field(min_length=1)
    aliases: list[str] | None = None
    short_definition: str | None = Field(default=None, min_length=1)
    body: str | None = None
    append_evidence: bool = True
    review_note: str | None = None


class EditRelationCandidateRequest(APIModel):
    edited_by: str = Field(min_length=1)
    source_name: str | None = Field(default=None, min_length=1)
    target_name: str | None = Field(default=None, min_length=1)
    source_concept_id: str | None = Field(default=None, min_length=1)
    target_concept_id: str | None = Field(default=None, min_length=1)
    relation_type: RelationType | None = None
    evidence_fragment_ids: list[str] | None = Field(default=None, min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    rationale: str | None = Field(default=None, min_length=1)


class ApproveRelationCandidateRequest(APIModel):
    reviewed_by: str = Field(min_length=1)
    source_concept_id: str | None = Field(default=None, min_length=1)
    target_concept_id: str | None = Field(default=None, min_length=1)
    relation_type: RelationType | None = None
    review_note: str | None = None


class MergeRelationCandidateRequest(APIModel):
    reviewed_by: str = Field(min_length=1)
    target_relation_id: str = Field(min_length=1)
    append_evidence: bool = True
    review_note: str | None = None


class ConceptCandidateReviewResponse(APIModel):
    candidate: ConceptCandidate
    concept: Concept | None = None
    audit_event_ids: list[str] = Field(default_factory=list)


class RelationCandidateReviewResponse(APIModel):
    candidate: RelationCandidate
    relation: ConceptRelation | None = None
    audit_event_ids: list[str] = Field(default_factory=list)


class OntologyExtractionRunResponse(APIModel):
    run: OntologyExtractionRun
    concept_candidates: list[ConceptCandidate] = Field(default_factory=list)
    relation_candidates: list[RelationCandidate] = Field(default_factory=list)


class OntologyExtractionRunListResponse(APIModel):
    runs: list[OntologyExtractionRun] = Field(default_factory=list)


class ConceptCandidateListResponse(APIModel):
    candidates: list[ConceptCandidate] = Field(default_factory=list)


class RelationCandidateListResponse(APIModel):
    candidates: list[RelationCandidate] = Field(default_factory=list)


class CandidateReviewBlocker(APIModel):
    code: str
    message: str
    evidence_fragment_id: str | None = None


class CandidateReviewPreview(APIModel):
    candidate_id: str
    candidate_type: str
    action: str
    can_apply: bool
    official_graph_will_change: bool
    mutation_summary: str
    target_concept_id: str | None = None
    target_relation_id: str | None = None
    evidence_preserved: bool = True
    blocker_reasons: list[CandidateReviewBlocker] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)


class CandidateReviewQueueGroup(APIModel):
    focus_concept: str
    pending_concept_candidate_count: int = 0
    pending_relation_candidate_count: int = 0
    high_confidence_count: int = 0
    low_confidence_count: int = 0
    blocked_count: int = 0
    source_ids: list[str] = Field(default_factory=list)
    extraction_run_ids: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)


class CandidateReviewQueueSummary(APIModel):
    total_pending_concept_candidates: int = 0
    total_pending_relation_candidates: int = 0
    grouped_by_focus_concept: list[CandidateReviewQueueGroup] = Field(default_factory=list)
    counts_by_status: dict[str, int] = Field(default_factory=dict)
    counts_by_run: dict[str, int] = Field(default_factory=dict)
    counts_by_source: dict[str, int] = Field(default_factory=dict)
    machine_readable_next_actions: list[str] = Field(default_factory=list)


class CreateOntologyReExtractionRunRequest(APIModel):
    datasource_id: str | None = Field(default=None, min_length=1)
    sync_job_id: str | None = Field(default=None, min_length=1)
    artifact_ids: list[str] = Field(default_factory=list)
    focus_concept: str | None = Field(default=None, min_length=1)
    trigger: OntologyReExtractionTrigger = OntologyReExtractionTrigger.MANUAL_REQUEST
    created_by: str = Field(default="system", min_length=1)
    reason: str | None = None
    run_inline: bool = False

    @model_validator(mode="after")
    def validate_scope(self) -> CreateOntologyReExtractionRunRequest:
        if not self.datasource_id and not self.sync_job_id and not self.artifact_ids:
            raise ValueError("Ontology re-extraction requires datasource_id, sync_job_id, or artifact_ids.")
        return self


class RunOntologyReExtractionRunRequest(APIModel):
    requested_by: str = Field(default="system", min_length=1)


class OntologyReExtractionRun(APIModel):
    id: str = Field(default_factory=new_id)
    datasource_id: str
    provider: DataSourceType
    trigger: OntologyReExtractionTrigger = OntologyReExtractionTrigger.MANUAL_REQUEST
    status: OntologyReExtractionRunStatus = OntologyReExtractionRunStatus.QUEUED
    created_by: str = Field(default="system", min_length=1)
    sync_job_id: str | None = None
    focus_concept: str | None = None
    reason: str | None = None
    source_external_ids: list[str] = Field(default_factory=list)
    artifact_ids: list[str] = Field(default_factory=list)
    changed_artifact_ids: list[str] = Field(default_factory=list)
    evidence_fragment_ids: list[str] = Field(default_factory=list)
    extraction_run_ids: list[str] = Field(default_factory=list)
    concept_candidate_count: int = 0
    relation_candidate_count: int = 0
    warning_count: int = 0
    official_graph_mutated: bool = False
    error: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @model_validator(mode="after")
    def validate_artifacts(self) -> OntologyReExtractionRun:
        if not self.artifact_ids and self.status not in {OntologyReExtractionRunStatus.SKIPPED, OntologyReExtractionRunStatus.FAILED}:
            raise ValueError("OntologyReExtractionRun requires at least one artifact id unless skipped or failed.")
        return self


class OntologyReExtractionRunResponse(APIModel):
    run: OntologyReExtractionRun
    extraction_runs: list[OntologyExtractionRun] = Field(default_factory=list)
    concept_candidates: list[ConceptCandidate] = Field(default_factory=list)
    relation_candidates: list[RelationCandidate] = Field(default_factory=list)


class OntologyReExtractionRunListResponse(APIModel):
    runs: list[OntologyReExtractionRun] = Field(default_factory=list)


class CreateOntologyProofRunRequest(APIModel):
    focus_concept: str = Field(default="Settlement", min_length=1)
    reviewer: str = Field(default="reviewer@example.com", min_length=1)
    created_by: str = Field(default="operator", min_length=1)
    source_name: str | None = Field(default=None, min_length=1)
    seed_content: str | None = Field(default=None, min_length=1)
    dry_run: bool = False
    confirm_mutation: bool = False
    run_evaluation: bool = True


class OntologyProofChecklistStep(APIModel):
    key: str = Field(min_length=1)
    title: str = Field(min_length=1)
    category: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    required: bool = True
    status: OntologyProofStepStatus = OntologyProofStepStatus.PLANNED
    detail: str | None = None
    checks: list[str] = Field(default_factory=list)
    object_ids: dict[str, Any] = Field(default_factory=dict)
    next_actions: list[str] = Field(default_factory=list)


class OntologyProofSummary(APIModel):
    status: OntologyProofRunStatus = OntologyProofRunStatus.PLANNED
    required_total: int = 0
    required_passed: int = 0
    required_failed: int = 0
    optional_total: int = 0
    optional_passed: int = 0
    optional_failed: int = 0
    planned_count: int = 0
    skipped_count: int = 0
    official_graph_available: bool = False
    official_graph_mutated: bool = False
    candidate_count: int = 0
    approved_concept_count: int = 0
    approved_relation_count: int = 0
    evaluation_success: bool | None = None


class OntologyProofRunResponse(APIModel):
    run_id: str = Field(default_factory=new_id)
    status: OntologyProofRunStatus
    focus_concept: str
    reviewer: str
    dry_run: bool = False
    source_id: str | None = None
    artifact_ids: list[str] = Field(default_factory=list)
    evidence_fragment_ids: list[str] = Field(default_factory=list)
    reextraction_run_id: str | None = None
    extraction_run_ids: list[str] = Field(default_factory=list)
    concept_candidate_ids: list[str] = Field(default_factory=list)
    relation_candidate_ids: list[str] = Field(default_factory=list)
    approved_concept_ids: list[str] = Field(default_factory=list)
    approved_relation_ids: list[str] = Field(default_factory=list)
    graph_response_id: str | None = None
    evaluation_task_id: str | None = None
    evaluation_result_id: str | None = None
    checklist: list[OntologyProofChecklistStep] = Field(default_factory=list)
    summary: OntologyProofSummary
    limitations: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utc_now)


class OntologySsotReadinessCheck(OntologyProofChecklistStep):
    """A v2.0.0 release-readiness checklist item for the ontology SSOT loop."""


class OntologySsotReadinessResponse(APIModel):
    response_id: str = Field(default_factory=new_id)
    release_version: str = "2.5.0"
    focus_concept: str = Field(min_length=1)
    mode: OntologyGraphMode = OntologyGraphMode.OFFICIAL
    depth: int = Field(default=1, ge=0, le=1)
    status: OntologyProofRunStatus
    official_graph_available: bool = False
    official_graph_safe: bool = False
    trust_label: TrustLabel | None = None
    graph_response_id: str | None = None
    node_count: int = 0
    edge_count: int = 0
    evidence_count: int = 0
    pending_concept_candidate_count: int = 0
    pending_relation_candidate_count: int = 0
    latest_evaluation_result_id: str | None = None
    latest_evaluation_success: bool | None = None
    evaluation_summary: OntologyGraphEvalMetricSummary | None = None
    checks: list[OntologySsotReadinessCheck] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    graph: OntologyGraphResponse | None = None
    generated_at: datetime = Field(default_factory=utc_now)


class GroundedContextResponse(APIModel):
    response_id: str = Field(default_factory=new_id)
    query: str
    answer: str
    trust_label: TrustLabel
    concepts: list[ConceptRef] = Field(default_factory=list)
    relations: list[ConceptRelationRef] = Field(default_factory=list)
    decisions: list[DecisionRecordRef] = Field(default_factory=list)
    evidence: list[EvidenceCitation] = Field(default_factory=list)
    freshness: FreshnessSummary
    limitations: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utc_now)
    official_answer_available: bool = False




class GroundedContextEvalTask(APIModel):
    id: str = Field(default_factory=new_id)
    name: str = Field(min_length=1)
    query: str = Field(min_length=1)
    expected_answer_contains: list[str] = Field(default_factory=list)
    expected_trust_label: TrustLabel | None = None
    expected_freshness_state: FreshnessState | None = None
    required_evidence_fragment_ids: list[str] = Field(default_factory=list)
    required_concept_ids: list[str] = Field(default_factory=list)
    required_decision_record_ids: list[str] = Field(default_factory=list)
    require_official_answer: bool = False
    require_evidence: bool = True
    min_evidence_count: int = Field(default=1, ge=0)
    expected_clarification_reduced: bool | None = None
    tags: list[str] = Field(default_factory=list)
    created_by: str = Field(default="system", min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_eval_task_contract(self) -> GroundedContextEvalTask:
        return _validate_grounded_eval_model(self)


class CreateGroundedContextEvalTaskRequest(APIModel):
    name: str = Field(min_length=1)
    query: str = Field(min_length=1)
    expected_answer_contains: list[str] = Field(default_factory=list)
    expected_trust_label: TrustLabel | None = None
    expected_freshness_state: FreshnessState | None = None
    required_evidence_fragment_ids: list[str] = Field(default_factory=list)
    required_concept_ids: list[str] = Field(default_factory=list)
    required_decision_record_ids: list[str] = Field(default_factory=list)
    require_official_answer: bool = False
    require_evidence: bool = True
    min_evidence_count: int = Field(default=1, ge=0)
    expected_clarification_reduced: bool | None = None
    tags: list[str] = Field(default_factory=list)
    created_by: str = Field(default="system", min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_eval_task_contract(self) -> CreateGroundedContextEvalTaskRequest:
        return _validate_grounded_eval_model(self)


def _validate_grounded_eval_model(model: Any):
    _validate_grounded_eval_task_contract(
        expected_answer_contains=model.expected_answer_contains,
        expected_trust_label=model.expected_trust_label,
        expected_freshness_state=model.expected_freshness_state,
        required_evidence_fragment_ids=model.required_evidence_fragment_ids,
        required_concept_ids=model.required_concept_ids,
        required_decision_record_ids=model.required_decision_record_ids,
        require_official_answer=model.require_official_answer,
        require_evidence=model.require_evidence,
        min_evidence_count=model.min_evidence_count,
        expected_clarification_reduced=model.expected_clarification_reduced,
    )
    return model



def _validate_grounded_eval_task_contract(
    *,
    expected_answer_contains: list[str],
    expected_trust_label: TrustLabel | None,
    expected_freshness_state: FreshnessState | None,
    required_evidence_fragment_ids: list[str],
    required_concept_ids: list[str],
    required_decision_record_ids: list[str],
    require_official_answer: bool,
    require_evidence: bool,
    min_evidence_count: int,
    expected_clarification_reduced: bool | None,
) -> None:
    has_explicit_success_condition = any(
        [
            bool(expected_answer_contains),
            expected_trust_label is not None,
            expected_freshness_state is not None,
            bool(required_evidence_fragment_ids),
            bool(required_concept_ids),
            bool(required_decision_record_ids),
            require_official_answer,
        ]
    )
    if not has_explicit_success_condition:
        raise ValueError(
            "Evaluation tasks must define at least one explicit success condition: "
            "expectedTrustLabel, expectedAnswerContains, requiredEvidenceFragmentIds, "
            "requiredConceptIds, requiredDecisionRecordIds, requireOfficialAnswer, "
            "or expectedFreshnessState."
        )
    if expected_trust_label == TrustLabel.UNSUPPORTED:
        if require_evidence or min_evidence_count != 0 or require_official_answer:
            raise ValueError(
                "Unsupported evaluation tasks must set requireEvidence=false, "
                "minEvidenceCount=0, and requireOfficialAnswer=false."
            )
        if required_evidence_fragment_ids or required_concept_ids or required_decision_record_ids:
            raise ValueError(
                "Unsupported evaluation tasks cannot require evidence, Concept, or DecisionRecord IDs."
            )
        return
    if not require_evidence and min_evidence_count == 0:
        raise ValueError(
            "Evaluation tasks that do not expect unsupported must require evidence "
            "or set minEvidenceCount greater than zero."
        )


class RunGroundedContextEvalTaskRequest(APIModel):
    evaluated_by: str = Field(default="system", min_length=1)
    clarification_reduced: bool | None = None


class RunGroundedContextEvalRequest(APIModel):
    task_ids: list[str] = Field(default_factory=list)
    evaluated_by: str = Field(default="system", min_length=1)


class GroundedContextEvalResult(APIModel):
    id: str = Field(default_factory=new_id)
    task_id: str
    response_id: str
    query: str
    answer: str
    trust_label: TrustLabel
    response: GroundedContextResponse
    answer_correct: bool
    evidence_valid: bool
    provenance_present: bool
    trust_label_correct: bool
    freshness_policy_respected: bool
    unsupported_official_claim: bool
    citation_validity_rate: float = Field(ge=0.0, le=1.0)
    clarification_reduced: bool | None = None
    success: bool
    failure_reasons: list[str] = Field(default_factory=list)
    evaluated_at: datetime = Field(default_factory=utc_now)
    evaluated_by: str = Field(default="system", min_length=1)


class GroundedContextEvalRunResponse(APIModel):
    results: list[GroundedContextEvalResult]
    total_count: int
    success_count: int
    grounded_context_task_success_rate: float = Field(ge=0.0, le=1.0)


class GroundedContextEvalMetricSummary(APIModel):
    total_task_count: int = 0
    evaluated_result_count: int = 0
    successful_result_count: int = 0
    grounded_context_task_success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    provenance_coverage_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    citation_validity_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    freshness_compliance_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    trust_label_correctness_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    unsupported_answer_correctness_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    generated_at: datetime = Field(default_factory=utc_now)


# Backward-compatible alias inside v0.12.0 service code.
GroundedContextEvalSummary = GroundedContextEvalMetricSummary


class OntologyGraphEvalTask(APIModel):
    id: str = Field(default_factory=new_id)
    name: str = Field(min_length=1)
    concept_query: str = Field(min_length=1)
    mode: OntologyGraphMode = OntologyGraphMode.OFFICIAL
    depth: int = Field(default=1, ge=0, le=1)
    expected_trust_label: TrustLabel | None = None
    expected_freshness_state: FreshnessState | None = None
    required_concept_ids: list[str] = Field(default_factory=list)
    required_relation_ids: list[str] = Field(default_factory=list)
    required_evidence_fragment_ids: list[str] = Field(default_factory=list)
    require_official_graph: bool = True
    require_evidence: bool = True
    min_evidence_count: int = Field(default=1, ge=0)
    min_node_count: int = Field(default=1, ge=0)
    min_edge_count: int = Field(default=0, ge=0)
    require_review_provenance: bool = True
    max_pending_candidate_count: int | None = Field(default=None, ge=0)
    tags: list[str] = Field(default_factory=list)
    created_by: str = Field(default="system", min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_ontology_eval_task_contract(self) -> OntologyGraphEvalTask:
        return _validate_ontology_graph_eval_model(self)


class CreateOntologyGraphEvalTaskRequest(APIModel):
    name: str = Field(min_length=1)
    concept_query: str = Field(min_length=1)
    mode: OntologyGraphMode = OntologyGraphMode.OFFICIAL
    depth: int = Field(default=1, ge=0, le=1)
    expected_trust_label: TrustLabel | None = None
    expected_freshness_state: FreshnessState | None = None
    required_concept_ids: list[str] = Field(default_factory=list)
    required_relation_ids: list[str] = Field(default_factory=list)
    required_evidence_fragment_ids: list[str] = Field(default_factory=list)
    require_official_graph: bool = True
    require_evidence: bool = True
    min_evidence_count: int = Field(default=1, ge=0)
    min_node_count: int = Field(default=1, ge=0)
    min_edge_count: int = Field(default=0, ge=0)
    require_review_provenance: bool = True
    max_pending_candidate_count: int | None = Field(default=None, ge=0)
    tags: list[str] = Field(default_factory=list)
    created_by: str = Field(default="system", min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_ontology_eval_task_contract(self) -> CreateOntologyGraphEvalTaskRequest:
        return _validate_ontology_graph_eval_model(self)


def _validate_ontology_graph_eval_model(model: Any):
    _validate_ontology_graph_eval_task_contract(
        expected_trust_label=model.expected_trust_label,
        expected_freshness_state=model.expected_freshness_state,
        required_concept_ids=model.required_concept_ids,
        required_relation_ids=model.required_relation_ids,
        required_evidence_fragment_ids=model.required_evidence_fragment_ids,
        require_official_graph=model.require_official_graph,
        require_evidence=model.require_evidence,
        min_evidence_count=model.min_evidence_count,
        min_node_count=model.min_node_count,
        min_edge_count=model.min_edge_count,
        require_review_provenance=model.require_review_provenance,
        max_pending_candidate_count=model.max_pending_candidate_count,
    )
    return model



def _validate_ontology_graph_eval_task_contract(
    *,
    expected_trust_label: TrustLabel | None,
    expected_freshness_state: FreshnessState | None,
    required_concept_ids: list[str],
    required_relation_ids: list[str],
    required_evidence_fragment_ids: list[str],
    require_official_graph: bool,
    require_evidence: bool,
    min_evidence_count: int,
    min_node_count: int,
    min_edge_count: int,
    require_review_provenance: bool,
    max_pending_candidate_count: int | None,
) -> None:
    has_explicit_success_condition = any(
        [
            expected_trust_label is not None,
            expected_freshness_state is not None,
            bool(required_concept_ids),
            bool(required_relation_ids),
            bool(required_evidence_fragment_ids),
            require_official_graph,
            require_evidence,
            min_evidence_count > 0,
            min_node_count > 0,
            min_edge_count > 0,
            require_review_provenance,
            max_pending_candidate_count is not None,
        ]
    )
    if not has_explicit_success_condition:
        raise ValueError(
            "Ontology graph evaluation tasks must define at least one explicit success condition: "
            "expectedTrustLabel, expectedFreshnessState, requiredConceptIds, "
            "requiredRelationIds, requiredEvidenceFragmentIds, requireOfficialGraph, "
            "requireEvidence, min counts, requireReviewProvenance, or maxPendingCandidateCount."
        )
    if expected_trust_label == TrustLabel.UNSUPPORTED:
        if (
            require_evidence
            or min_evidence_count != 0
            or require_official_graph
            or min_node_count != 0
            or min_edge_count != 0
            or require_review_provenance
        ):
            raise ValueError(
                "Unsupported ontology graph evaluation tasks must set requireEvidence=false, "
                "minEvidenceCount=0, requireOfficialGraph=false, minNodeCount=0, "
                "minEdgeCount=0, and requireReviewProvenance=false."
            )
        if required_evidence_fragment_ids or required_concept_ids or required_relation_ids:
            raise ValueError(
                "Unsupported ontology graph evaluation tasks cannot require evidence, Concept, or Relation IDs."
            )


class RunOntologyGraphEvalTaskRequest(APIModel):
    evaluated_by: str = Field(default="system", min_length=1)


class RunOntologyGraphEvalRequest(APIModel):
    task_ids: list[str] = Field(default_factory=list)
    evaluated_by: str = Field(default="system", min_length=1)


class OntologyGraphEvalResult(APIModel):
    id: str = Field(default_factory=new_id)
    task_id: str
    response_id: str
    concept_query: str
    mode: OntologyGraphMode
    depth: int = Field(ge=0, le=1)
    trust_label: TrustLabel
    response: OntologyGraphResponse
    graph_found: bool
    graph_depth_respected: bool
    node_requirements_met: bool
    edge_requirements_met: bool
    evidence_valid: bool
    provenance_present: bool
    trust_label_correct: bool
    freshness_policy_respected: bool
    official_graph_safe: bool
    candidate_boundary_respected: bool
    relation_integrity_valid: bool
    citation_validity_rate: float = Field(ge=0.0, le=1.0)
    success: bool
    failure_reasons: list[str] = Field(default_factory=list)
    evaluated_at: datetime = Field(default_factory=utc_now)
    evaluated_by: str = Field(default="system", min_length=1)


class OntologyGraphEvalRunResponse(APIModel):
    results: list[OntologyGraphEvalResult]
    total_count: int
    success_count: int
    ontology_graph_task_success_rate: float = Field(ge=0.0, le=1.0)


class OntologyGraphEvalMetricSummary(APIModel):
    total_task_count: int = 0
    evaluated_result_count: int = 0
    successful_result_count: int = 0
    ontology_graph_task_success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_validity_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    provenance_coverage_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    citation_validity_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    official_graph_safety_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    candidate_boundary_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    relation_integrity_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    trust_label_correctness_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    freshness_compliance_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    generated_at: datetime = Field(default_factory=utc_now)


class HealthResponse(APIModel):
    status: str
    service: str
    version: str


class AuditEvent(APIModel):
    id: str = Field(default_factory=new_id)
    event_type: str
    actor: str
    entity_type: str
    entity_id: str
    occurred_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


# Connector framework schemas (v0.4.0)
class ConnectorAuthType(StrEnum):
    OAUTH2 = "oauth2"
    API_TOKEN = "api_token"
    GITHUB_APP = "github_app"
    MANUAL = "manual"


AuthType = ConnectorAuthType


class ConnectorAvailability(StrEnum):
    AVAILABLE = "available"
    COMING_SOON = "coming_soon"
    DISABLED = "disabled"


class ConnectorScope(APIModel):
    key: str
    label: str
    description: str
    required: bool = True


class ConnectorSetupStep(APIModel):
    key: str
    title: str
    description: str
    order: int = Field(ge=1)


class ConnectorCapabilities(APIModel):
    supports_oauth: bool = False
    supports_api_token: bool = False
    supports_discovery: bool = False
    supports_selection: bool = False
    supports_incremental_sync: bool = False
    supports_webhooks: bool = False
    supports_source_updated_at: bool = False
    supports_source_urls: bool = False
    supports_author_metadata: bool = False
    supports_permission_snapshots: bool = False


class ConnectorDefinition(APIModel):
    provider: DataSourceType
    display_name: str
    description: str
    auth_type: ConnectorAuthType
    availability: ConnectorAvailability = ConnectorAvailability.AVAILABLE
    production_ready: bool = False
    required_scopes: list[ConnectorScope] = Field(default_factory=list)
    optional_scopes: list[ConnectorScope] = Field(default_factory=list)
    supported_objects: list[str] = Field(default_factory=list)
    discoverable_objects: list[str] = Field(default_factory=list)
    ingestible_objects: list[str] = Field(default_factory=list)
    setup_steps: list[ConnectorSetupStep] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    docs_url: str | None = None
    capabilities: ConnectorCapabilities = Field(default_factory=ConnectorCapabilities)


class ConnectorSupportMatrixItem(APIModel):
    provider: DataSourceType
    object_type: str
    support_state: str
    proof_state: str
    creates_evidence: bool
    queues_candidate_reextraction: bool
    mutates_official_graph: bool = False
    limitations: list[str] = Field(default_factory=list)


class ConnectorSupportMatrixResponse(APIModel):
    items: list[ConnectorSupportMatrixItem] = Field(default_factory=list)
    secret_redaction_policy: str
    live_proof_env_guards: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=utc_now)


class IntegrationPackageManifest(APIModel):
    chosen_path: str = "external_integration_package"
    job_to_be_done: str
    stable_endpoints: list[str] = Field(default_factory=list)
    trust_boundary: str
    non_chosen_path: str = "frontend_mvp"
    non_chosen_reason: str
    quickstart: list[str] = Field(default_factory=list)


class IntegrationOntologyResponse(APIModel):
    concept: str
    official_graph: OntologyGraphResponse
    ssot_readiness: OntologySsotReadinessResponse
    candidate_vs_official_boundary: str
    evidence_citations: list[EvidenceCitation] = Field(default_factory=list)
    trust_state: TrustLabel
    unsupported_state: str | None = None
    review_gate_bypass_allowed: bool = False


class ConnectionIntentStatus(StrEnum):
    CREATED = "created"
    OAUTH_REDIRECTED = "oauth_redirected"
    COMPLETED = "completed"
    EXPIRED = "expired"
    FAILED = "failed"


class ConnectorErrorCode(StrEnum):
    OAUTH_FAILED = "oauth_failed"
    TOKEN_EXPIRED = "token_expired"
    TOKEN_REVOKED = "token_revoked"
    INSUFFICIENT_SCOPE = "insufficient_scope"
    PERMISSION_DENIED = "permission_denied"
    RATE_LIMITED = "rate_limited"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    OBJECT_NOT_FOUND = "object_not_found"
    CONTENT_PARSE_FAILED = "content_parse_failed"
    SYNC_TIMEOUT = "sync_timeout"
    CONNECTION_TEST_FAILED = "connection_test_failed"
    SOURCE_SELECTION_REQUIRED = "source_selection_required"
    UNSUPPORTED_OBJECT_TYPE = "unsupported_object_type"
    NO_CREDENTIAL = "no_credential"
    UNSUPPORTED_PROVIDER = "unsupported_provider"
    UNKNOWN = "unknown"


class ConnectorNextAction(StrEnum):
    RECONNECT = "reconnect"
    GRANT_PERMISSION = "grant_permission"
    RETRY = "retry"
    CONTACT_ADMIN = "contact_admin"
    WAIT_AND_RETRY = "wait_and_retry"
    TEST_CONNECTION = "test_connection"
    SELECT_SOURCES = "select_sources"
    NONE = "none"


class ConnectorError(APIModel):
    code: ConnectorErrorCode
    user_message: str
    technical_message: str | None = None
    retryable: bool = False
    next_action: ConnectorNextAction = ConnectorNextAction.NONE
    retry_after_seconds: int | None = Field(default=None, ge=0)
    occurred_at: datetime = Field(default_factory=utc_now)


class CreateConnectionIntentRequest(APIModel):
    provider: DataSourceType
    source_name: str = Field(min_length=1)
    created_by: str = Field(default="system", min_length=1)
    return_url: str | None = None
    requested_scopes: list[str] = Field(default_factory=list)


class ConnectionIntent(APIModel):
    id: str = Field(default_factory=new_id)
    provider: DataSourceType
    status: ConnectionIntentStatus = ConnectionIntentStatus.CREATED
    auth_type: ConnectorAuthType = ConnectorAuthType.OAUTH2
    source_name: str = Field(min_length=1)
    created_by: str = Field(min_length=1)
    requested_scopes: list[str] = Field(default_factory=list)
    authorization_url: str | None = None
    redirect_uri: str
    return_url: str | None = None
    state_nonce: str = Field(min_length=16)
    expires_at: datetime
    created_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime | None = None
    datasource_id: str | None = None
    failure_error: ConnectorError | None = None


class CredentialStatus(StrEnum):
    ACTIVE = "active"
    REVOKED = "revoked"


class ConnectorCredential(APIModel):
    id: str = Field(default_factory=new_id)
    datasource_id: str
    provider: DataSourceType
    auth_type: ConnectorAuthType
    encrypted_access_token: str = Field(min_length=1)
    encrypted_refresh_token: str | None = None
    granted_scopes: list[str] = Field(default_factory=list)
    external_account_id: str | None = None
    external_workspace_id: str | None = None
    external_workspace_name: str | None = None
    external_bot_id: str | None = None
    token_expires_at: datetime | None = None
    status: CredentialStatus = CredentialStatus.ACTIVE
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    revoked_at: datetime | None = None


class ConnectorCredentialPublic(APIModel):
    id: str
    datasource_id: str
    provider: DataSourceType
    auth_type: ConnectorAuthType
    status: CredentialStatus
    granted_scopes: list[str] = Field(default_factory=list)
    external_account_id: str | None = None
    external_workspace_id: str | None = None
    external_workspace_name: str | None = None
    token_expires_at: datetime | None = None
    created_at: datetime
    updated_at: datetime
    revoked_at: datetime | None = None


class OAuthCallbackResponse(APIModel):
    intent: ConnectionIntent
    data_source: DataSource | None = None
    credential: ConnectorCredentialPublic | None = None
    next_action: ConnectorNextAction = ConnectorNextAction.NONE


class SelectionRule(APIModel):
    field: str
    operator: str
    value: str


class SourceSelectionMode(StrEnum):
    SELECTED_ONLY = "selected_only"
    WORKSPACE_LIMITED = "workspace_limited"
    ALL_ACCESSIBLE = "all_accessible"


class SourceSelection(APIModel):
    id: str = Field(default_factory=new_id)
    datasource_id: str
    sync_mode: SourceSelectionMode = SourceSelectionMode.SELECTED_ONLY
    include_rules: list[SelectionRule] = Field(default_factory=list)
    exclude_rules: list[SelectionRule] = Field(default_factory=list)
    selected_external_object_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class ProviderObjectType(StrEnum):
    PAGE = "page"
    DATABASE = "database"
    DATA_SOURCE = "data_source"
    BLOCK = "block"
    DOCUMENT = "document"
    SPREADSHEET = "spreadsheet"
    PRESENTATION = "presentation"
    PDF = "pdf"
    FOLDER = "folder"
    FILE = "file"
    TEXT_FILE = "text_file"
    UNKNOWN = "unknown"


class ProviderObjectAccessState(StrEnum):
    ACCESSIBLE = "accessible"
    INACCESSIBLE = "inaccessible"
    DELETED = "deleted"
    UNKNOWN = "unknown"


class ProviderObjectSnapshot(APIModel):
    id: str = Field(default_factory=new_id)
    datasource_id: str
    provider: DataSourceType
    external_id: str = Field(min_length=1)
    external_url: str | None = None
    object_type: ProviderObjectType = ProviderObjectType.UNKNOWN
    title: str | None = None
    parent_external_id: str | None = None
    last_edited_time: datetime | None = None
    discovered_at: datetime = Field(default_factory=utc_now)
    selected_for_sync: bool = False
    access_state: ProviderObjectAccessState = ProviderObjectAccessState.UNKNOWN
    ingestion_supported: bool = True
    ingestion_unsupported_reason: str | None = None
    raw_metadata_hash: str = Field(min_length=16)
    provider_metadata: dict[str, Any] = Field(default_factory=dict)


class DiscoverProviderObjectsRequest(APIModel):
    page_size: int = Field(default=25, ge=1, le=100)
    cursor: str | None = None


class DiscoverProviderObjectsResponse(APIModel):
    data_source: DataSource
    objects: list[ProviderObjectSnapshot]
    next_cursor: str | None = None
    has_more: bool = False


class ProviderObjectSnapshotListResponse(APIModel):
    datasource_id: str
    objects: list[ProviderObjectSnapshot]
    total_count: int
    accessible_count: int
    syncable_count: int
    selected_count: int


class UpsertSourceSelectionRequest(APIModel):
    sync_mode: SourceSelectionMode = SourceSelectionMode.SELECTED_ONLY
    include_rules: list[SelectionRule] = Field(default_factory=list)
    exclude_rules: list[SelectionRule] = Field(default_factory=list)
    selected_external_object_ids: list[str] = Field(default_factory=list)


class ConnectionTestStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"


class ConnectionTestResult(APIModel):
    status: ConnectionTestStatus
    datasource_id: str
    provider: DataSourceType
    external_workspace_name: str | None = None
    granted_scopes: list[str] = Field(default_factory=list)
    can_read_objects: bool = False
    sample_object_count: int = 0
    limitations: list[str] = Field(default_factory=list)
    error: ConnectorError | None = None
    tested_at: datetime = Field(default_factory=utc_now)


class SyncJobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    RETRY_WAITING = "retry_waiting"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SyncJobTrigger(StrEnum):
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    WEBHOOK = "webhook"
    RETRY = "retry"


class SyncJob(APIModel):
    id: str = Field(default_factory=new_id)
    datasource_id: str
    provider: DataSourceType
    status: SyncJobStatus = SyncJobStatus.QUEUED
    trigger: SyncJobTrigger = SyncJobTrigger.MANUAL
    created_by: str = Field(default="system", min_length=1)
    selection_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    artifact_created_count: int = 0
    artifact_reused_count: int = 0
    evidence_created_count: int = 0
    error: ConnectorError | None = None
    cursor: str | None = None
    attempt_count: int = Field(default=0, ge=0)
    max_attempts: int = Field(default=3, ge=1)
    next_attempt_at: datetime | None = None
    cancel_requested_at: datetime | None = None
    cancelled_by: str | None = None
    lease_owner: str | None = None
    lease_acquired_at: datetime | None = None
    lease_expires_at: datetime | None = None
    lease_heartbeat_at: datetime | None = None
    schedule_id: str | None = None
    enqueue_key: str | None = None
    queue_ontology_reextraction: bool = True
    run_ontology_reextraction_inline: bool = False
    ontology_focus_concept: str | None = None


class SyncCursor(APIModel):
    id: str = Field(default_factory=new_id)
    datasource_id: str
    provider: DataSourceType
    cursor_key: str = Field(default="default", min_length=1)
    last_cursor: str | None = None
    last_successful_sync_job_id: str | None = None
    processed_external_object_ids: list[str] = Field(default_factory=list)
    artifact_created_count: int = 0
    artifact_reused_count: int = 0
    evidence_created_count: int = 0
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    advanced_at: datetime | None = None


class SyncScheduleStatus(StrEnum):
    ACTIVE = "active"
    PAUSED = "paused"


class SyncSchedule(APIModel):
    id: str = Field(default_factory=new_id)
    datasource_id: str
    provider: DataSourceType
    status: SyncScheduleStatus = SyncScheduleStatus.ACTIVE
    interval_minutes: int = Field(default=60, ge=5, le=10080)
    next_run_at: datetime
    last_enqueued_at: datetime | None = None
    last_enqueued_sync_job_id: str | None = None
    max_attempts: int = Field(default=3, ge=1, le=10)
    created_by: str = Field(default="system", min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class UpsertSyncScheduleRequest(APIModel):
    enabled: bool = True
    interval_minutes: int = Field(default=60, ge=5, le=10080)
    start_at: datetime | None = None
    max_attempts: int = Field(default=3, ge=1, le=10)
    created_by: str = Field(default="system", min_length=1)


class SyncSchedulerRunResult(APIModel):
    enqueued_job_count: int = 0
    skipped_schedule_count: int = 0
    schedules_checked_count: int = 0
    jobs: list[SyncJob] = Field(default_factory=list)
    schedules: list[SyncSchedule] = Field(default_factory=list)


class SyncJobEvent(APIModel):
    id: str = Field(default_factory=new_id)
    sync_job_id: str
    datasource_id: str
    event_type: str
    message: str
    occurred_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CreateSyncJobRequest(APIModel):
    trigger: SyncJobTrigger = SyncJobTrigger.MANUAL
    created_by: str = Field(default="system", min_length=1)
    run_inline: bool = False
    max_attempts: int = Field(default=3, ge=1, le=10)
    objects: list[SourceObject] | None = None
    queue_ontology_reextraction: bool = Field(
        default=True,
        validation_alias=AliasChoices("queueOntologyReextraction", "queueOntologyReExtraction"),
    )
    run_ontology_reextraction_inline: bool = Field(
        default=False,
        validation_alias=AliasChoices("runOntologyReextractionInline", "runOntologyReExtractionInline"),
    )
    ontology_focus_concept: str | None = Field(default=None, min_length=1)


class SyncJobDetail(APIModel):
    job: SyncJob
    events: list[SyncJobEvent] = Field(default_factory=list)


class HeartbeatSyncJobLeaseRequest(APIModel):
    worker_id: str = Field(default="api-worker", min_length=1, max_length=128)
    lease_seconds: int = Field(default=300, ge=30, le=3600)


class RunSyncWorkerRequest(APIModel):
    max_jobs: int = Field(default=1, ge=1, le=50)
    include_not_ready: bool = False
    job_id: str | None = None
    worker_id: str = Field(default="api-worker", min_length=1, max_length=128)
    lease_seconds: int = Field(default=300, ge=30, le=3600)


class RunSyncSchedulerRequest(APIModel):
    max_schedules: int = Field(default=50, ge=1, le=500)
    include_not_due: bool = False


class SyncWorkerRunResult(APIModel):
    processed_job_count: int = 0
    skipped_job_count: int = 0
    jobs: list[SyncJobDetail] = Field(default_factory=list)
