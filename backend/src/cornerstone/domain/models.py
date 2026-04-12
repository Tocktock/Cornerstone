from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from cornerstone.clock import utcnow
from cornerstone.domain.enums import (
    ActorKind,
    BaseRole,
    ConceptKind,
    ConsumerScope,
    ContextSpaceKind,
    CuratedLifecycleState,
    DecisionLifecycleState,
    FreshnessState,
    OriginDisclosureLevel,
    SharedSelectionKind,
    SourceConnectionState,
    SupportItemKind,
    SupportVisibility,
    SyncMode,
    SyncRunStatus,
    SyncTriggerKind,
    VerificationState,
    VisibilityClass,
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


class TimestampedMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=utcnow,
        onupdate=utcnow,
        nullable=False,
    )


class ContextSpace(Base, TimestampedMixin):
    __tablename__ = "context_spaces"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    kind: Mapped[ContextSpaceKind] = mapped_column(
        Enum(ContextSpaceKind), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    slug: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    membership_boundary: Mapped[str] = mapped_column(String(255), nullable=False)
    default_visibility_class: Mapped[VisibilityClass] = mapped_column(
        Enum(VisibilityClass),
        nullable=False,
        default=VisibilityClass.MEMBER_VISIBLE,
    )
    visibility_defaults: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    is_default: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    verification_policy: Mapped[VerificationPolicy | None] = relationship(
        back_populates="context_space",
        cascade="all, delete-orphan",
        uselist=False,
    )
    actors: Mapped[list[Actor]] = relationship(
        back_populates="context_space", cascade="all, delete-orphan"
    )
    review_scope_grants: Mapped[list[ReviewScopeGrant]] = relationship(
        back_populates="context_space",
        cascade="all, delete-orphan",
    )
    connector_scope_grants: Mapped[list[ConnectorScopeGrant]] = relationship(
        back_populates="context_space",
        cascade="all, delete-orphan",
    )
    provider_credentials: Mapped[list[ProviderCredential]] = relationship(
        back_populates="context_space",
        cascade="all, delete-orphan",
    )
    source_connections: Mapped[list[SourceConnection]] = relationship(
        back_populates="context_space",
        cascade="all, delete-orphan",
    )
    sync_runs: Mapped[list[SyncRun]] = relationship(
        back_populates="context_space",
        cascade="all, delete-orphan",
    )
    artifacts: Mapped[list[Artifact]] = relationship(
        back_populates="context_space", cascade="all, delete-orphan"
    )
    support_items: Mapped[list[SupportItem]] = relationship(
        back_populates="context_space",
        cascade="all, delete-orphan",
    )
    concepts: Mapped[list[Concept]] = relationship(
        back_populates="context_space", cascade="all, delete-orphan"
    )
    relations: Mapped[list[ConceptRelation]] = relationship(
        back_populates="context_space",
        cascade="all, delete-orphan",
    )
    decisions: Mapped[list[DecisionRecord]] = relationship(
        back_populates="context_space",
        cascade="all, delete-orphan",
    )


class VerificationPolicy(Base, TimestampedMixin):
    __tablename__ = "verification_policies"
    __table_args__ = (UniqueConstraint("context_space_id"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    context_space_id: Mapped[str] = mapped_column(
        ForeignKey("context_spaces.id"), nullable=False, index=True
    )
    label: Mapped[str] = mapped_column(String(255), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    minimum_support_items: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    minimum_durable_support_items: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    minimum_visible_support_items_for_source_backed: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
    )
    allow_restricted_support_for_officialization: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )
    allow_member_restricted_support_publication: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )
    freshness_target_hours: Mapped[int] = mapped_column(Integer, nullable=False, default=24)
    continuous_revalidation_enabled: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=True
    )
    allow_accepted_decision_lineage_as_support: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
    )

    context_space: Mapped[ContextSpace] = relationship(back_populates="verification_policy")


class Actor(Base, TimestampedMixin):
    __tablename__ = "actors"
    __table_args__ = (UniqueConstraint("context_space_id", "display_name"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    context_space_id: Mapped[str] = mapped_column(
        ForeignKey("context_spaces.id"), nullable=False, index=True
    )
    principal_key: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    actor_kind: Mapped[ActorKind] = mapped_column(Enum(ActorKind), nullable=False)
    display_name: Mapped[str] = mapped_column(String(255), nullable=False)
    base_role: Mapped[BaseRole] = mapped_column(Enum(BaseRole), nullable=False)
    auth_token: Mapped[str] = mapped_column(String(255), nullable=False, unique=True, index=True)
    scoped_capabilities: Mapped[list[dict[str, str]]] = mapped_column(
        JSON, nullable=False, default=list
    )
    preferred_consumer_scope: Mapped[ConsumerScope] = mapped_column(
        Enum(ConsumerScope),
        nullable=False,
        default=ConsumerScope.MEMBER,
    )

    context_space: Mapped[ContextSpace] = relationship(back_populates="actors")
    review_scope_grants: Mapped[list[ReviewScopeGrant]] = relationship(
        back_populates="actor",
        cascade="all, delete-orphan",
    )
    connector_scope_grants: Mapped[list[ConnectorScopeGrant]] = relationship(
        back_populates="actor",
        cascade="all, delete-orphan",
    )
    provider_credentials: Mapped[list[ProviderCredential]] = relationship(
        back_populates="created_by_actor",
        cascade="all, delete-orphan",
        foreign_keys="ProviderCredential.created_by_actor_id",
    )
    promoted_support_items: Mapped[list[SupportItem]] = relationship(
        back_populates="promoter",
        foreign_keys="SupportItem.promoter_id",
    )


class ReviewScopeGrant(Base, TimestampedMixin):
    __tablename__ = "review_scope_grants"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    context_space_id: Mapped[str] = mapped_column(
        ForeignKey("context_spaces.id"), nullable=False, index=True
    )
    actor_id: Mapped[str] = mapped_column(ForeignKey("actors.id"), nullable=False, index=True)
    review_domains: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    allowed_review_actions: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    target_object_kinds: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)

    context_space: Mapped[ContextSpace] = relationship(back_populates="review_scope_grants")
    actor: Mapped[Actor] = relationship(back_populates="review_scope_grants")


class ConnectorScopeGrant(Base, TimestampedMixin):
    __tablename__ = "connector_scope_grants"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    context_space_id: Mapped[str] = mapped_column(
        ForeignKey("context_spaces.id"), nullable=False, index=True
    )
    actor_id: Mapped[str] = mapped_column(ForeignKey("actors.id"), nullable=False, index=True)
    allowed_connector_actions: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)

    context_space: Mapped[ContextSpace] = relationship(back_populates="connector_scope_grants")
    actor: Mapped[Actor] = relationship(back_populates="connector_scope_grants")


class ProviderCredential(Base, TimestampedMixin):
    __tablename__ = "provider_credentials"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    context_space_id: Mapped[str] = mapped_column(
        ForeignKey("context_spaces.id"), nullable=False, index=True
    )
    provider: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    credential_reference: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    binding_state: Mapped[str | None] = mapped_column(String(255), unique=True, index=True)
    account_label: Mapped[str | None] = mapped_column(String(255))
    auth_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_by_actor_id: Mapped[str | None] = mapped_column(ForeignKey("actors.id"), index=True)
    last_validated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    context_space: Mapped[ContextSpace] = relationship(back_populates="provider_credentials")
    created_by_actor: Mapped[Actor | None] = relationship(
        back_populates="provider_credentials",
        foreign_keys=[created_by_actor_id],
    )
    source_connections: Mapped[list[SourceConnection]] = relationship(
        back_populates="provider_credential"
    )


class SourceConnection(Base, TimestampedMixin):
    __tablename__ = "source_connections"
    __table_args__ = (UniqueConstraint("context_space_id", "provider", "source_boundary_locator"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    context_space_id: Mapped[str] = mapped_column(
        ForeignKey("context_spaces.id"), nullable=False, index=True
    )
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    source_label: Mapped[str] = mapped_column(String(255), nullable=False)
    source_boundary_locator: Mapped[str] = mapped_column(String(2048), nullable=False)
    template_key: Mapped[str] = mapped_column(String(255), nullable=False)
    provider_credential_ref: Mapped[str | None] = mapped_column(
        ForeignKey("provider_credentials.credential_reference"), index=True
    )
    selected_scope_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    sync_checkpoint_json: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    visibility_class: Mapped[VisibilityClass] = mapped_column(
        Enum(VisibilityClass),
        nullable=False,
        default=VisibilityClass.MEMBER_VISIBLE,
    )
    sync_mode: Mapped[SyncMode] = mapped_column(
        Enum(SyncMode), nullable=False, default=SyncMode.POLLING
    )
    sync_interval_seconds: Mapped[int] = mapped_column(Integer, nullable=False, default=300)
    source_connection_state: Mapped[SourceConnectionState] = mapped_column(
        Enum(SourceConnectionState),
        nullable=False,
        default=SourceConnectionState.PENDING_SETUP,
    )
    last_attempted_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_successful_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    freshness_state: Mapped[FreshnessState] = mapped_column(
        Enum(FreshnessState),
        nullable=False,
        default=FreshnessState.UNKNOWN,
    )
    last_error: Mapped[str | None] = mapped_column(Text)
    effective_sync_policy: Mapped[dict[str, Any]] = mapped_column(
        JSON, nullable=False, default=dict
    )
    next_scheduled_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    removed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    context_space: Mapped[ContextSpace] = relationship(back_populates="source_connections")
    provider_credential: Mapped[ProviderCredential | None] = relationship(
        back_populates="source_connections"
    )
    artifacts: Mapped[list[Artifact]] = relationship(
        back_populates="source_connection", cascade="all, delete-orphan"
    )
    sync_runs: Mapped[list[SyncRun]] = relationship(
        back_populates="source_connection", cascade="all, delete-orphan"
    )


class SyncRun(Base, TimestampedMixin):
    __tablename__ = "sync_runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    context_space_id: Mapped[str] = mapped_column(
        ForeignKey("context_spaces.id"), nullable=False, index=True
    )
    source_connection_id: Mapped[str] = mapped_column(
        ForeignKey("source_connections.id"), nullable=False, index=True
    )
    trigger_kind: Mapped[SyncTriggerKind] = mapped_column(Enum(SyncTriggerKind), nullable=False)
    run_status: Mapped[SyncRunStatus] = mapped_column(
        Enum(SyncRunStatus), nullable=False, default=SyncRunStatus.RUNNING
    )
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    artifact_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    support_item_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_summary: Mapped[str | None] = mapped_column(Text)

    context_space: Mapped[ContextSpace] = relationship(back_populates="sync_runs")
    source_connection: Mapped[SourceConnection] = relationship(back_populates="sync_runs")


class Artifact(Base, TimestampedMixin):
    __tablename__ = "artifacts"
    __table_args__ = (UniqueConstraint("source_connection_id", "external_id"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    context_space_id: Mapped[str] = mapped_column(
        ForeignKey("context_spaces.id"), nullable=False, index=True
    )
    source_connection_id: Mapped[str] = mapped_column(
        ForeignKey("source_connections.id"), nullable=False, index=True
    )
    external_id: Mapped[str] = mapped_column(String(2048), nullable=False)
    artifact_type: Mapped[str] = mapped_column(String(128), nullable=False)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    source_locator: Mapped[str] = mapped_column(String(2048), nullable=False)
    source_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_refreshed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    content_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    content_text: Mapped[str] = mapped_column(Text, nullable=False, default="")
    freshness_state: Mapped[FreshnessState] = mapped_column(
        Enum(FreshnessState),
        nullable=False,
        default=FreshnessState.UNKNOWN,
    )
    visibility_class: Mapped[VisibilityClass] = mapped_column(
        Enum(VisibilityClass),
        nullable=False,
        default=VisibilityClass.MEMBER_VISIBLE,
    )
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)

    context_space: Mapped[ContextSpace] = relationship(back_populates="artifacts")
    source_connection: Mapped[SourceConnection] = relationship(back_populates="artifacts")
    support_items: Mapped[list[SupportItem]] = relationship(
        back_populates="artifact", cascade="all, delete-orphan"
    )


class PromotionLineage(Base, TimestampedMixin):
    __tablename__ = "promotion_lineages"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_context_kind: Mapped[ContextSpaceKind] = mapped_column(
        Enum(ContextSpaceKind), nullable=False
    )
    personal_source_owner_principal_key: Mapped[str] = mapped_column(String(255), nullable=False)
    private_origin_ref: Mapped[str] = mapped_column(String(2048), nullable=False)
    selection_method: Mapped[str] = mapped_column(String(255), nullable=False)
    selection_scope_summary: Mapped[str] = mapped_column(Text, nullable=False)
    workspace_disclosure_note: Mapped[str] = mapped_column(Text, nullable=False)
    origin_disclosure_level: Mapped[OriginDisclosureLevel] = mapped_column(
        Enum(OriginDisclosureLevel),
        nullable=False,
    )

    support_items: Mapped[list[SupportItem]] = relationship(back_populates="promotion_lineage")


class SupportItem(Base, TimestampedMixin):
    __tablename__ = "support_items"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    context_space_id: Mapped[str] = mapped_column(
        ForeignKey("context_spaces.id"), nullable=False, index=True
    )
    support_item_kind: Mapped[SupportItemKind] = mapped_column(
        Enum(SupportItemKind), nullable=False
    )
    visibility_class: Mapped[VisibilityClass] = mapped_column(Enum(VisibilityClass), nullable=False)
    source_label: Mapped[str] = mapped_column(String(255), nullable=False)
    excerpt_or_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    source_locator: Mapped[str | None] = mapped_column(String(2048))
    freshness_state: Mapped[FreshnessState] = mapped_column(
        Enum(FreshnessState),
        nullable=False,
        default=FreshnessState.UNKNOWN,
    )
    artifact_id: Mapped[str | None] = mapped_column(ForeignKey("artifacts.id"), index=True)
    selector: Mapped[str | None] = mapped_column(String(255))
    normalized_claim: Mapped[str | None] = mapped_column(Text)
    promoter_id: Mapped[str | None] = mapped_column(ForeignKey("actors.id"), index=True)
    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    shared_selection_kind: Mapped[SharedSelectionKind | None] = mapped_column(
        Enum(SharedSelectionKind)
    )
    shared_payload: Mapped[str | None] = mapped_column(Text)
    origin_disclosure_level: Mapped[OriginDisclosureLevel | None] = mapped_column(
        Enum(OriginDisclosureLevel)
    )
    promotion_lineage_id: Mapped[str | None] = mapped_column(
        ForeignKey("promotion_lineages.id"), index=True
    )

    context_space: Mapped[ContextSpace] = relationship(back_populates="support_items")
    artifact: Mapped[Artifact | None] = relationship(back_populates="support_items")
    promoter: Mapped[Actor | None] = relationship(
        back_populates="promoted_support_items", foreign_keys=[promoter_id]
    )
    promotion_lineage: Mapped[PromotionLineage | None] = relationship(
        back_populates="support_items"
    )
    concept_links: Mapped[list[ConceptSupportLink]] = relationship(
        back_populates="support_item",
        cascade="all, delete-orphan",
    )
    relation_links: Mapped[list[RelationSupportLink]] = relationship(
        back_populates="support_item",
        cascade="all, delete-orphan",
    )
    decision_links: Mapped[list[DecisionSupportLink]] = relationship(
        back_populates="support_item",
        cascade="all, delete-orphan",
    )


class Concept(Base, TimestampedMixin):
    __tablename__ = "concepts"
    __table_args__ = (
        UniqueConstraint(
            "context_space_id", "public_slug", name="uq_concepts_context_space_public_slug"
        ),
        UniqueConstraint(
            "context_space_id", "canonical_name", name="uq_concepts_context_space_canonical_name"
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    context_space_id: Mapped[str] = mapped_column(
        ForeignKey("context_spaces.id"), nullable=False, index=True
    )
    public_slug: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    canonical_name: Mapped[str] = mapped_column(String(255), nullable=False)
    aliases: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    definition: Mapped[str] = mapped_column(Text, nullable=False)
    concept_kind: Mapped[ConceptKind] = mapped_column(Enum(ConceptKind), nullable=False)
    owning_domain: Mapped[str] = mapped_column(String(255), nullable=False)
    review_domain: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    lifecycle_state: Mapped[CuratedLifecycleState] = mapped_column(
        Enum(CuratedLifecycleState),
        nullable=False,
        default=CuratedLifecycleState.DRAFT,
    )
    verification_state: Mapped[VerificationState] = mapped_column(
        Enum(VerificationState),
        nullable=False,
        default=VerificationState.UNVERIFIED,
    )
    support_visibility: Mapped[SupportVisibility] = mapped_column(
        Enum(SupportVisibility),
        nullable=False,
        default=SupportVisibility.INSUFFICIENT_SUPPORT,
    )

    context_space: Mapped[ContextSpace] = relationship(back_populates="concepts")
    outgoing_relations: Mapped[list[ConceptRelation]] = relationship(
        back_populates="subject_concept",
        foreign_keys="ConceptRelation.subject_concept_id",
        cascade="all, delete-orphan",
    )
    incoming_relations: Mapped[list[ConceptRelation]] = relationship(
        back_populates="object_concept",
        foreign_keys="ConceptRelation.object_concept_id",
        cascade="all, delete-orphan",
    )
    support_links: Mapped[list[ConceptSupportLink]] = relationship(
        back_populates="concept",
        cascade="all, delete-orphan",
    )
    decision_links: Mapped[list[DecisionConceptLink]] = relationship(
        back_populates="concept",
        cascade="all, delete-orphan",
    )


class ConceptRelation(Base, TimestampedMixin):
    __tablename__ = "concept_relations"
    __table_args__ = (
        UniqueConstraint(
            "context_space_id", "subject_concept_id", "predicate", "object_concept_id"
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    context_space_id: Mapped[str] = mapped_column(
        ForeignKey("context_spaces.id"), nullable=False, index=True
    )
    subject_concept_id: Mapped[str] = mapped_column(
        ForeignKey("concepts.id"), nullable=False, index=True
    )
    predicate: Mapped[str] = mapped_column(String(128), nullable=False)
    object_concept_id: Mapped[str] = mapped_column(
        ForeignKey("concepts.id"), nullable=False, index=True
    )
    description: Mapped[str | None] = mapped_column(Text)
    review_domain: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    lifecycle_state: Mapped[CuratedLifecycleState] = mapped_column(
        Enum(CuratedLifecycleState),
        nullable=False,
        default=CuratedLifecycleState.DRAFT,
    )
    verification_state: Mapped[VerificationState] = mapped_column(
        Enum(VerificationState),
        nullable=False,
        default=VerificationState.UNVERIFIED,
    )
    support_visibility: Mapped[SupportVisibility] = mapped_column(
        Enum(SupportVisibility),
        nullable=False,
        default=SupportVisibility.INSUFFICIENT_SUPPORT,
    )

    context_space: Mapped[ContextSpace] = relationship(back_populates="relations")
    subject_concept: Mapped[Concept] = relationship(
        back_populates="outgoing_relations",
        foreign_keys=[subject_concept_id],
    )
    object_concept: Mapped[Concept] = relationship(
        back_populates="incoming_relations",
        foreign_keys=[object_concept_id],
    )
    support_links: Mapped[list[RelationSupportLink]] = relationship(
        back_populates="relation",
        cascade="all, delete-orphan",
    )
    decision_links: Mapped[list[DecisionRelationLink]] = relationship(
        back_populates="relation",
        cascade="all, delete-orphan",
    )


class DecisionRecord(Base, TimestampedMixin):
    __tablename__ = "decision_records"
    __table_args__ = (
        UniqueConstraint("context_space_id", "public_slug", name="uq_decisions_context_space_public_slug"),
        UniqueConstraint("context_space_id", "title"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    context_space_id: Mapped[str] = mapped_column(
        ForeignKey("context_spaces.id"), nullable=False, index=True
    )
    public_slug: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    problem_statement: Mapped[str | None] = mapped_column(Text)
    decision_statement: Mapped[str] = mapped_column(Text, nullable=False)
    rationale: Mapped[str | None] = mapped_column(Text)
    constraints: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    impact_summary: Mapped[str | None] = mapped_column(Text)
    owning_domain: Mapped[str] = mapped_column(String(255), nullable=False)
    review_domain: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    lifecycle_state: Mapped[DecisionLifecycleState] = mapped_column(
        Enum(DecisionLifecycleState),
        nullable=False,
        default=DecisionLifecycleState.PROPOSED,
    )
    verification_state: Mapped[VerificationState] = mapped_column(
        Enum(VerificationState),
        nullable=False,
        default=VerificationState.UNVERIFIED,
    )
    support_visibility: Mapped[SupportVisibility] = mapped_column(
        Enum(SupportVisibility),
        nullable=False,
        default=SupportVisibility.INSUFFICIENT_SUPPORT,
    )
    supersedes_decision_id: Mapped[str | None] = mapped_column(
        ForeignKey("decision_records.id"),
        index=True,
    )

    context_space: Mapped[ContextSpace] = relationship(back_populates="decisions")
    supersedes_decision: Mapped[DecisionRecord | None] = relationship(
        remote_side="DecisionRecord.id"
    )
    support_links: Mapped[list[DecisionSupportLink]] = relationship(
        back_populates="decision",
        cascade="all, delete-orphan",
    )
    concept_links: Mapped[list[DecisionConceptLink]] = relationship(
        back_populates="decision",
        cascade="all, delete-orphan",
    )
    relation_links: Mapped[list[DecisionRelationLink]] = relationship(
        back_populates="decision",
        cascade="all, delete-orphan",
    )


class ConceptSupportLink(Base):
    __tablename__ = "concept_support_links"

    concept_id: Mapped[str] = mapped_column(ForeignKey("concepts.id"), primary_key=True)
    support_item_id: Mapped[str] = mapped_column(ForeignKey("support_items.id"), primary_key=True)

    concept: Mapped[Concept] = relationship(back_populates="support_links")
    support_item: Mapped[SupportItem] = relationship(back_populates="concept_links")


class RelationSupportLink(Base):
    __tablename__ = "relation_support_links"

    relation_id: Mapped[str] = mapped_column(ForeignKey("concept_relations.id"), primary_key=True)
    support_item_id: Mapped[str] = mapped_column(ForeignKey("support_items.id"), primary_key=True)

    relation: Mapped[ConceptRelation] = relationship(back_populates="support_links")
    support_item: Mapped[SupportItem] = relationship(back_populates="relation_links")


class DecisionSupportLink(Base):
    __tablename__ = "decision_support_links"

    decision_id: Mapped[str] = mapped_column(ForeignKey("decision_records.id"), primary_key=True)
    support_item_id: Mapped[str] = mapped_column(ForeignKey("support_items.id"), primary_key=True)

    decision: Mapped[DecisionRecord] = relationship(back_populates="support_links")
    support_item: Mapped[SupportItem] = relationship(back_populates="decision_links")


class DecisionConceptLink(Base):
    __tablename__ = "decision_concept_links"

    decision_id: Mapped[str] = mapped_column(ForeignKey("decision_records.id"), primary_key=True)
    concept_id: Mapped[str] = mapped_column(ForeignKey("concepts.id"), primary_key=True)

    decision: Mapped[DecisionRecord] = relationship(back_populates="concept_links")
    concept: Mapped[Concept] = relationship(back_populates="decision_links")


class DecisionRelationLink(Base):
    __tablename__ = "decision_relation_links"

    decision_id: Mapped[str] = mapped_column(ForeignKey("decision_records.id"), primary_key=True)
    relation_id: Mapped[str] = mapped_column(ForeignKey("concept_relations.id"), primary_key=True)

    decision: Mapped[DecisionRecord] = relationship(back_populates="relation_links")
    relation: Mapped[ConceptRelation] = relationship(back_populates="decision_links")
