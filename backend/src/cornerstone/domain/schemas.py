from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cornerstone.domain.enums import (
    ActorType,
    ConceptStatus,
    ConceptType,
    DecisionStatus,
    RelationStatus,
    SyncMode,
)


class ORMModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class ContextSpaceRead(ORMModel):
    id: str
    name: str
    namespace: str
    status: str
    created_at: datetime
    updated_at: datetime


class ActorRead(ORMModel):
    id: str
    context_space_id: str
    actor_type: ActorType
    display_name: str
    roles: list[str]
    status: str


class SourceConnectionRead(ORMModel):
    id: str
    context_space_id: str
    provider: str
    external_scope: str
    sync_mode: SyncMode
    sync_interval_seconds: int
    health_status: str
    last_synced_at: datetime | None
    last_error: str | None


class EvidenceRead(BaseModel):
    id: str
    selector: str
    excerpt: str
    normalized_claim: str
    verification_status: str
    artifact_id: str
    artifact_title: str
    artifact_url: str


class ConceptRead(BaseModel):
    id: str
    context_space_id: str
    concept_type: ConceptType
    canonical_name: str
    aliases: list[str]
    definition: str
    status: ConceptStatus
    evidence: list[EvidenceRead]
    linked_decisions: list[str]


class RelationRead(BaseModel):
    id: str
    context_space_id: str
    subject_concept_id: str
    subject_name: str
    predicate: str
    object_concept_id: str
    object_name: str
    description: str
    status: RelationStatus
    evidence: list[EvidenceRead]
    linked_decisions: list[str]


class DecisionRead(BaseModel):
    id: str
    context_space_id: str
    title: str
    problem: str
    decision: str
    rationale: str
    constraints: list[str]
    impact: list[str]
    status: DecisionStatus
    evidence: list[EvidenceRead]
    concepts: list[str]
    relations: list[str]


class ArtifactRead(BaseModel):
    id: str
    context_space_id: str
    source_connection_id: str
    external_id: str
    artifact_type: str
    title: str
    canonical_url: str
    status: str
    evidence_count: int
    metadata_json: dict[str, Any]


class GraphNode(BaseModel):
    id: str
    label: str
    type: str
    status: str


class GraphEdge(BaseModel):
    id: str
    source: str
    target: str
    label: str
    status: str


class GraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


class StructuredAnswerResponse(BaseModel):
    query: str
    summary: str
    concepts: list[ConceptRead]
    relations: list[RelationRead]
    decisions: list[DecisionRead]
    evidence: list[EvidenceRead]


class SyncRunResult(BaseModel):
    source_connection_id: str
    artifact_count: int
    evidence_count: int
    concept_count: int
    relation_count: int
    decision_count: int


class ConceptCreate(BaseModel):
    context_space_id: str
    concept_type: ConceptType
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    definition: str
    owner_actor_id: str | None = None
    evidence_fragment_ids: list[str] = Field(default_factory=list)
    linked_decision_ids: list[str] = Field(default_factory=list)


class RelationCreate(BaseModel):
    context_space_id: str
    subject_concept_id: str
    predicate: str
    object_concept_id: str
    description: str = ""
    evidence_fragment_ids: list[str] = Field(default_factory=list)
    linked_decision_ids: list[str] = Field(default_factory=list)


class DecisionCreate(BaseModel):
    context_space_id: str
    title: str
    problem: str
    decision: str
    rationale: str
    constraints: list[str] = Field(default_factory=list)
    impact: list[str] = Field(default_factory=list)
    evidence_fragment_ids: list[str] = Field(default_factory=list)
    linked_concept_ids: list[str] = Field(default_factory=list)
    linked_relation_ids: list[str] = Field(default_factory=list)


class ReviewActionRequest(BaseModel):
    actor_id: str
    action: str
