from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    MetaData,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from cornerstone.domain.enums import (
    ActorStatus,
    ActorType,
    ArtifactStatus,
    ConceptStatus,
    ConceptType,
    ConnectionHealthStatus,
    ContextSpaceStatus,
    DecisionActorRole,
    DecisionConceptRole,
    DecisionRelationRole,
    DecisionStatus,
    Directionality,
    EvidenceVerificationStatus,
    RelationStatus,
    SyncMode,
)

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class TimestampedUUIDMixin:
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        onupdate=lambda: datetime.now(UTC),
    )


class ContextSpace(TimestampedUUIDMixin, Base):
    __tablename__ = "context_spaces"

    name: Mapped[str] = mapped_column(String(255), nullable=False)
    namespace: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    status: Mapped[ContextSpaceStatus] = mapped_column(
        Enum(ContextSpaceStatus), nullable=False, default=ContextSpaceStatus.ACTIVE
    )
    review_policy: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    visibility_policy: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    actors: Mapped[list["Actor"]] = relationship(back_populates="context_space", cascade="all, delete-orphan")
    source_connections: Mapped[list["SourceConnection"]] = relationship(
        back_populates="context_space", cascade="all, delete-orphan"
    )
    artifacts: Mapped[list["Artifact"]] = relationship(
        back_populates="context_space", cascade="all, delete-orphan"
    )
    concepts: Mapped[list["Concept"]] = relationship(back_populates="context_space", cascade="all, delete-orphan")
    relations: Mapped[list["ConceptRelation"]] = relationship(
        back_populates="context_space", cascade="all, delete-orphan"
    )
    decisions: Mapped[list["DecisionRecord"]] = relationship(
        back_populates="context_space", cascade="all, delete-orphan"
    )


class Actor(TimestampedUUIDMixin, Base):
    __tablename__ = "actors"
    __table_args__ = (UniqueConstraint("context_space_id", "display_name"),)

    context_space_id: Mapped[str] = mapped_column(ForeignKey("context_spaces.id"), index=True, nullable=False)
    actor_type: Mapped[ActorType] = mapped_column(Enum(ActorType), nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    external_identities: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    roles: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    status: Mapped[ActorStatus] = mapped_column(Enum(ActorStatus), nullable=False, default=ActorStatus.ACTIVE)

    context_space: Mapped[ContextSpace] = relationship(back_populates="actors")
    owned_concepts: Mapped[list["Concept"]] = relationship(back_populates="owner_actor")
    decision_links: Mapped[list["DecisionActorLink"]] = relationship(
        back_populates="actor", cascade="all, delete-orphan"
    )


class SourceConnection(TimestampedUUIDMixin, Base):
    __tablename__ = "source_connections"
    __table_args__ = (UniqueConstraint("context_space_id", "provider", "external_scope"),)

    context_space_id: Mapped[str] = mapped_column(ForeignKey("context_spaces.id"), index=True, nullable=False)
    provider: Mapped[str] = mapped_column(String(100), nullable=False)
    external_scope: Mapped[str] = mapped_column(String(1024), nullable=False)
    sync_mode: Mapped[SyncMode] = mapped_column(Enum(SyncMode), nullable=False, default=SyncMode.POLLING)
    sync_cursor: Mapped[str | None] = mapped_column(Text)
    sync_interval_seconds: Mapped[int] = mapped_column(nullable=False, default=300)
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    health_status: Mapped[ConnectionHealthStatus] = mapped_column(
        Enum(ConnectionHealthStatus), nullable=False, default=ConnectionHealthStatus.PENDING
    )
    last_error: Mapped[str | None] = mapped_column(Text)
    settings: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    context_space: Mapped[ContextSpace] = relationship(back_populates="source_connections")
    artifacts: Mapped[list["Artifact"]] = relationship(
        back_populates="source_connection", cascade="all, delete-orphan"
    )


class Artifact(TimestampedUUIDMixin, Base):
    __tablename__ = "artifacts"
    __table_args__ = (UniqueConstraint("source_connection_id", "external_id"),)

    context_space_id: Mapped[str] = mapped_column(ForeignKey("context_spaces.id"), index=True, nullable=False)
    source_connection_id: Mapped[str] = mapped_column(ForeignKey("source_connections.id"), index=True, nullable=False)
    external_id: Mapped[str] = mapped_column(String(1024), nullable=False)
    artifact_type: Mapped[str] = mapped_column(String(64), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    canonical_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    source_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[ArtifactStatus] = mapped_column(Enum(ArtifactStatus), nullable=False, default=ArtifactStatus.ACTIVE)
    content_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)

    context_space: Mapped[ContextSpace] = relationship(back_populates="artifacts")
    source_connection: Mapped[SourceConnection] = relationship(back_populates="artifacts")
    evidence_fragments: Mapped[list["EvidenceFragment"]] = relationship(
        back_populates="artifact", cascade="all, delete-orphan"
    )


class EvidenceFragment(TimestampedUUIDMixin, Base):
    __tablename__ = "evidence_fragments"
    __table_args__ = (UniqueConstraint("artifact_id", "selector"),)

    artifact_id: Mapped[str] = mapped_column(ForeignKey("artifacts.id"), index=True, nullable=False)
    selector: Mapped[str] = mapped_column(String(255), nullable=False)
    excerpt: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_claim: Mapped[str] = mapped_column(Text, nullable=False, default="")
    extracted_by: Mapped[str] = mapped_column(String(64), nullable=False, default="system")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    verification_status: Mapped[EvidenceVerificationStatus] = mapped_column(
        Enum(EvidenceVerificationStatus), nullable=False, default=EvidenceVerificationStatus.UNVERIFIED
    )

    artifact: Mapped[Artifact] = relationship(back_populates="evidence_fragments")
    concept_links: Mapped[list["ConceptEvidenceLink"]] = relationship(
        back_populates="evidence_fragment", cascade="all, delete-orphan"
    )
    relation_links: Mapped[list["RelationEvidenceLink"]] = relationship(
        back_populates="evidence_fragment", cascade="all, delete-orphan"
    )
    decision_links: Mapped[list["DecisionEvidenceLink"]] = relationship(
        back_populates="evidence_fragment", cascade="all, delete-orphan"
    )


class Concept(TimestampedUUIDMixin, Base):
    __tablename__ = "concepts"
    __table_args__ = (UniqueConstraint("context_space_id", "canonical_key"),)

    context_space_id: Mapped[str] = mapped_column(ForeignKey("context_spaces.id"), index=True, nullable=False)
    concept_type: Mapped[ConceptType] = mapped_column(Enum(ConceptType), nullable=False)
    canonical_name: Mapped[str] = mapped_column(String(255), nullable=False)
    canonical_key: Mapped[str] = mapped_column(String(255), nullable=False)
    aliases: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    definition: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[ConceptStatus] = mapped_column(Enum(ConceptStatus), nullable=False, default=ConceptStatus.DRAFT)
    owner_actor_id: Mapped[str | None] = mapped_column(ForeignKey("actors.id"), index=True)
    effective_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    effective_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    context_space: Mapped[ContextSpace] = relationship(back_populates="concepts")
    owner_actor: Mapped[Actor | None] = relationship(back_populates="owned_concepts")
    outgoing_relations: Mapped[list["ConceptRelation"]] = relationship(
        back_populates="subject_concept",
        foreign_keys="ConceptRelation.subject_concept_id",
        cascade="all, delete-orphan",
    )
    incoming_relations: Mapped[list["ConceptRelation"]] = relationship(
        back_populates="object_concept",
        foreign_keys="ConceptRelation.object_concept_id",
        cascade="all, delete-orphan",
    )
    evidence_links: Mapped[list["ConceptEvidenceLink"]] = relationship(
        back_populates="concept", cascade="all, delete-orphan"
    )
    decision_links: Mapped[list["DecisionConceptLink"]] = relationship(
        back_populates="concept", cascade="all, delete-orphan"
    )


class DecisionRecord(TimestampedUUIDMixin, Base):
    __tablename__ = "decision_records"
    __table_args__ = (UniqueConstraint("context_space_id", "title_key"),)

    context_space_id: Mapped[str] = mapped_column(ForeignKey("context_spaces.id"), index=True, nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    title_key: Mapped[str] = mapped_column(String(255), nullable=False)
    problem: Mapped[str] = mapped_column(Text, nullable=False, default="")
    decision: Mapped[str] = mapped_column(Text, nullable=False, default="")
    rationale: Mapped[str] = mapped_column(Text, nullable=False, default="")
    constraints: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    impact: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    status: Mapped[DecisionStatus] = mapped_column(Enum(DecisionStatus), nullable=False, default=DecisionStatus.PROPOSED)
    effective_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    review_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    supersedes_decision_id: Mapped[str | None] = mapped_column(ForeignKey("decision_records.id"), index=True)
    alternatives_considered: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    assumptions: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    trade_offs: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    outcome_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")

    context_space: Mapped[ContextSpace] = relationship(back_populates="decisions")
    supersedes_decision: Mapped["DecisionRecord | None"] = relationship(remote_side="DecisionRecord.id")
    actor_links: Mapped[list["DecisionActorLink"]] = relationship(
        back_populates="decision_record", cascade="all, delete-orphan"
    )
    evidence_links: Mapped[list["DecisionEvidenceLink"]] = relationship(
        back_populates="decision_record", cascade="all, delete-orphan"
    )
    concept_links: Mapped[list["DecisionConceptLink"]] = relationship(
        back_populates="decision_record", cascade="all, delete-orphan"
    )
    relation_links: Mapped[list["DecisionRelationLink"]] = relationship(
        back_populates="decision_record", cascade="all, delete-orphan"
    )
    introduced_relations: Mapped[list["ConceptRelation"]] = relationship(back_populates="introduced_by_decision")


class ConceptRelation(TimestampedUUIDMixin, Base):
    __tablename__ = "concept_relations"
    __table_args__ = (
        UniqueConstraint("context_space_id", "subject_concept_id", "predicate", "object_concept_id"),
    )

    context_space_id: Mapped[str] = mapped_column(ForeignKey("context_spaces.id"), index=True, nullable=False)
    subject_concept_id: Mapped[str] = mapped_column(ForeignKey("concepts.id"), index=True, nullable=False)
    predicate: Mapped[str] = mapped_column(String(128), nullable=False)
    object_concept_id: Mapped[str] = mapped_column(ForeignKey("concepts.id"), index=True, nullable=False)
    description: Mapped[str] = mapped_column(Text, nullable=False, default="")
    status: Mapped[RelationStatus] = mapped_column(Enum(RelationStatus), nullable=False, default=RelationStatus.DRAFT)
    directionality: Mapped[Directionality] = mapped_column(
        Enum(Directionality), nullable=False, default=Directionality.DIRECTED
    )
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=1.0)
    effective_from: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    effective_to: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    introduced_by_decision_id: Mapped[str | None] = mapped_column(ForeignKey("decision_records.id"), index=True)

    context_space: Mapped[ContextSpace] = relationship(back_populates="relations")
    subject_concept: Mapped[Concept] = relationship(
        back_populates="outgoing_relations", foreign_keys=[subject_concept_id]
    )
    object_concept: Mapped[Concept] = relationship(
        back_populates="incoming_relations", foreign_keys=[object_concept_id]
    )
    evidence_links: Mapped[list["RelationEvidenceLink"]] = relationship(
        back_populates="concept_relation", cascade="all, delete-orphan"
    )
    decision_links: Mapped[list["DecisionRelationLink"]] = relationship(
        back_populates="concept_relation", cascade="all, delete-orphan"
    )
    introduced_by_decision: Mapped[DecisionRecord | None] = relationship(back_populates="introduced_relations")


class ConceptEvidenceLink(Base):
    __tablename__ = "concept_evidence_links"

    concept_id: Mapped[str] = mapped_column(ForeignKey("concepts.id"), primary_key=True)
    evidence_fragment_id: Mapped[str] = mapped_column(ForeignKey("evidence_fragments.id"), primary_key=True)

    concept: Mapped[Concept] = relationship(back_populates="evidence_links")
    evidence_fragment: Mapped[EvidenceFragment] = relationship(back_populates="concept_links")


class RelationEvidenceLink(Base):
    __tablename__ = "relation_evidence_links"

    relation_id: Mapped[str] = mapped_column(ForeignKey("concept_relations.id"), primary_key=True)
    evidence_fragment_id: Mapped[str] = mapped_column(ForeignKey("evidence_fragments.id"), primary_key=True)

    concept_relation: Mapped[ConceptRelation] = relationship(back_populates="evidence_links")
    evidence_fragment: Mapped[EvidenceFragment] = relationship(back_populates="relation_links")


class DecisionEvidenceLink(Base):
    __tablename__ = "decision_evidence_links"

    decision_id: Mapped[str] = mapped_column(ForeignKey("decision_records.id"), primary_key=True)
    evidence_fragment_id: Mapped[str] = mapped_column(ForeignKey("evidence_fragments.id"), primary_key=True)

    decision_record: Mapped[DecisionRecord] = relationship(back_populates="evidence_links")
    evidence_fragment: Mapped[EvidenceFragment] = relationship(back_populates="decision_links")


class DecisionConceptLink(Base):
    __tablename__ = "decision_concept_links"

    decision_id: Mapped[str] = mapped_column(ForeignKey("decision_records.id"), primary_key=True)
    concept_id: Mapped[str] = mapped_column(ForeignKey("concepts.id"), primary_key=True)
    relationship_type: Mapped[DecisionConceptRole] = mapped_column(Enum(DecisionConceptRole), nullable=False)

    decision_record: Mapped[DecisionRecord] = relationship(back_populates="concept_links")
    concept: Mapped[Concept] = relationship(back_populates="decision_links")


class DecisionRelationLink(Base):
    __tablename__ = "decision_relation_links"

    decision_id: Mapped[str] = mapped_column(ForeignKey("decision_records.id"), primary_key=True)
    relation_id: Mapped[str] = mapped_column(ForeignKey("concept_relations.id"), primary_key=True)
    relationship_type: Mapped[DecisionRelationRole] = mapped_column(Enum(DecisionRelationRole), nullable=False)

    decision_record: Mapped[DecisionRecord] = relationship(back_populates="relation_links")
    concept_relation: Mapped[ConceptRelation] = relationship(back_populates="decision_links")


class DecisionActorLink(Base):
    __tablename__ = "decision_actor_links"

    decision_id: Mapped[str] = mapped_column(ForeignKey("decision_records.id"), primary_key=True)
    actor_id: Mapped[str] = mapped_column(ForeignKey("actors.id"), primary_key=True)
    role: Mapped[DecisionActorRole] = mapped_column(Enum(DecisionActorRole), nullable=False)

    decision_record: Mapped[DecisionRecord] = relationship(back_populates="actor_links")
    actor: Mapped[Actor] = relationship(back_populates="decision_links")
