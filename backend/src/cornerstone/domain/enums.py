from __future__ import annotations

from enum import StrEnum


class ContextSpaceKind(StrEnum):
    WORKSPACE = "workspace"
    PERSONAL = "personal"


class ActorKind(StrEnum):
    HUMAN = "human"
    TEAM = "team"
    SERVICE = "service"
    AI = "ai"


class BaseRole(StrEnum):
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"


class Capability(StrEnum):
    MANAGE_CONNECTORS = "manage_connectors"
    OPERATE = "operate"
    REVIEW = "review"
    OWN_DOMAIN = "own_domain"


class ConsumerScope(StrEnum):
    MEMBER = "member"
    REVIEW = "review"
    ADMIN = "admin"


class ReviewAction(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    OFFICIALIZE = "officialize"
    SUPERSEDE = "supersede"
    MARK_FOR_REVALIDATION = "mark_for_revalidation"
    RESOLVE_REVIEW_REQUIRED = "resolve_review_required"


class ConnectorAction(StrEnum):
    CREATE = "create"
    UPDATE = "update"
    PAUSE = "pause"
    RESUME = "resume"
    REMOVE = "remove"
    REBIND = "rebind"
    SYNC = "sync"


class SyncMode(StrEnum):
    POLLING = "polling"
    SCHEDULED_SYNC = "scheduled_sync"
    WEBHOOK = "webhook"
    HYBRID = "hybrid"
    SNAPSHOT_UPLOAD = "snapshot_upload"


class SyncRunStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


class SyncTriggerKind(StrEnum):
    INITIAL = "initial"
    MANUAL = "manual"
    SCHEDULED = "scheduled"
    RECOVERY = "recovery"


class SourceConnectionState(StrEnum):
    PENDING_SETUP = "pending_setup"
    ACTIVE = "active"
    SYNCING = "syncing"
    DEGRADED = "degraded"
    PAUSED = "paused"
    REMOVED = "removed"


class FreshnessState(StrEnum):
    CURRENT = "current"
    MONITORING = "monitoring"
    STALE = "stale"
    DRIFT_DETECTED = "drift_detected"
    UNKNOWN = "unknown"


class SupportVisibility(StrEnum):
    SOURCE_BACKED = "source_backed"
    RESTRICTED_SUPPORT = "restricted_support"
    INSUFFICIENT_SUPPORT = "insufficient_support"


class VerificationState(StrEnum):
    UNVERIFIED = "unverified"
    VERIFIED = "verified"
    MONITORING = "monitoring"
    REVIEW_REQUIRED = "review_required"
    SUPPORT_INSUFFICIENT = "support_insufficient"
    DRIFT_DETECTED = "drift_detected"


class CuratedLifecycleState(StrEnum):
    SUGGESTED = "suggested"
    DRAFT = "draft"
    IN_REVIEW = "in_review"
    OFFICIAL = "official"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"


class DecisionLifecycleState(StrEnum):
    PROPOSED = "proposed"
    IN_REVIEW = "in_review"
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    ARCHIVED = "archived"


class VisibilityClass(StrEnum):
    MEMBER_VISIBLE = "member_visible"
    EVIDENCE_ONLY = "evidence_only"


class SupportItemKind(StrEnum):
    EVIDENCE_FRAGMENT = "evidence_fragment"
    PROMOTED_SUPPORT = "promoted_support"


class ConceptKind(StrEnum):
    TERM = "term"
    DOMAIN = "domain"
    SYSTEM = "system"
    POLICY = "policy"
    WORKFLOW = "workflow"
    ROLE = "role"
    METRIC = "metric"
    EVENT = "event"
    ARTIFACT_TYPE = "artifact_type"
    STATUS = "status"


class CanonicalPredicate(StrEnum):
    IS_A = "is_a"
    PART_OF = "part_of"
    INSTANCE_OF = "instance_of"
    DEPENDS_ON = "depends_on"
    USED_IN = "used_in"
    INPUT_TO = "input_to"
    OUTPUT_OF = "output_of"
    PRECEDES = "precedes"
    TRIGGERS = "triggers"
    RESULTS_IN = "results_in"
    OWNED_BY = "owned_by"
    GOVERNED_BY = "governed_by"
    DEFINED_BY = "defined_by"
    APPLIES_TO = "applies_to"
    CONFLICTS_WITH = "conflicts_with"
    SUPERSEDES = "supersedes"


class SharedSelectionKind(StrEnum):
    ARTIFACT_EXCERPT = "artifact_excerpt"
    SECTION_EXCERPT = "section_excerpt"
    FRAGMENT_EXCERPT = "fragment_excerpt"
    SUMMARY_CLAIM = "summary_claim"


class OriginDisclosureLevel(StrEnum):
    NAMED_ORIGIN = "named_origin"
    REDACTED_ORIGIN = "redacted_origin"
    HIDDEN_ORIGIN = "hidden_origin"


class ResponseKind(StrEnum):
    CONCEPT = "concept"
    RELATION = "relation"
    DECISION = "decision"
    ANSWER = "answer"
    SEARCH_RESULTS = "search_results"
    GRAPH_SLICE = "graph_slice"
    PROVENANCE = "provenance"
    NO_MATCH = "no_match"


class RequestIntent(StrEnum):
    SEARCH_CONTEXT = "search_context"
    GET_CONCEPT = "get_concept"
    GET_RELATION = "get_relation"
    GET_DECISION = "get_decision"
    GET_ANSWER = "get_answer"
    GET_GRAPH_SLICE = "get_graph_slice"
    FOLLOW_PROVENANCE = "follow_provenance"


class ResourceKind(StrEnum):
    CONCEPT = "concept"
    RELATION = "relation"
    DECISION = "decision"
    ARTIFACT = "artifact"
    SUPPORT_ITEM = "support_item"


class AnswerStatus(StrEnum):
    OFFICIAL = "official"
    PARTIAL = "partial"
    REVIEW_REQUIRED = "review_required"


class NoMatchReason(StrEnum):
    NO_OFFICIAL_MATCH = "no_official_match"
    NO_VISIBLE_MATCH = "no_visible_match"
    OUTSIDE_SCOPE = "outside_scope"
    NOT_AVAILABLE_IN_WORKSPACE = "not_available_in_workspace"
