from __future__ import annotations

from enum import StrEnum


class ContextSpaceStatus(StrEnum):
    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"


class ActorType(StrEnum):
    HUMAN = "HUMAN"
    TEAM = "TEAM"
    SERVICE = "SERVICE"
    AI = "AI"


class ActorStatus(StrEnum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"


class SyncMode(StrEnum):
    MANUAL = "MANUAL"
    POLLING = "POLLING"
    WEBHOOK = "WEBHOOK"
    HYBRID = "HYBRID"


class ConnectionHealthStatus(StrEnum):
    PENDING = "PENDING"
    HEALTHY = "HEALTHY"
    ERROR = "ERROR"


class ArtifactStatus(StrEnum):
    ACTIVE = "ACTIVE"
    REMOVED = "REMOVED"


class EvidenceVerificationStatus(StrEnum):
    UNVERIFIED = "UNVERIFIED"
    VERIFIED = "VERIFIED"


class ConceptType(StrEnum):
    TERM = "TERM"
    SYSTEM = "SYSTEM"
    POLICY = "POLICY"
    WORKFLOW = "WORKFLOW"
    ROLE = "ROLE"
    METRIC = "METRIC"
    EVENT = "EVENT"


class ConceptStatus(StrEnum):
    DRAFT = "DRAFT"
    OFFICIAL = "OFFICIAL"
    REJECTED = "REJECTED"
    DEPRECATED = "DEPRECATED"


class RelationStatus(StrEnum):
    DRAFT = "DRAFT"
    OFFICIAL = "OFFICIAL"
    REJECTED = "REJECTED"
    DEPRECATED = "DEPRECATED"


class Directionality(StrEnum):
    DIRECTED = "DIRECTED"
    UNDIRECTED = "UNDIRECTED"


class DecisionStatus(StrEnum):
    PROPOSED = "PROPOSED"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    SUPERSEDED = "SUPERSEDED"


class DecisionActorRole(StrEnum):
    PROPOSED_BY = "PROPOSED_BY"
    APPROVED_BY = "APPROVED_BY"
    REVIEWED_BY = "REVIEWED_BY"


class DecisionConceptRole(StrEnum):
    ABOUT = "ABOUT"
    MODIFIES = "MODIFIES"
    DEPRECATES = "DEPRECATES"


class DecisionRelationRole(StrEnum):
    ABOUT = "ABOUT"
    INTRODUCES = "INTRODUCES"
    DEPRECATES = "DEPRECATES"
