from __future__ import annotations

from datetime import datetime
from typing import Any

import sqlalchemy as sa
from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.types import CHAR, TypeDecorator

from cornerstone.schemas import new_id

JSONB_TYPE = sa.JSON().with_variant(postgresql.JSONB(), "postgresql")
CITEXT_TYPE = sa.String(255).with_variant(postgresql.CITEXT(), "postgresql")
TZ_DATETIME = sa.DateTime(timezone=True)


class UUIDString(TypeDecorator[str]):
    """Portable UUID storage.

    PostgreSQL stores real UUID values; lightweight tests can use SQLite without replacing models.
    Application DTOs still expose UUIDs as strings so route contracts remain stable.
    """

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect: sa.engine.Dialect) -> sa.types.TypeEngine[Any]:
        if dialect.name == "postgresql":
            return dialect.type_descriptor(postgresql.UUID(as_uuid=False))
        return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value: str | None, dialect: sa.engine.Dialect) -> str | None:
        if value is None:
            return None
        return str(value)

    def process_result_value(self, value: object | None, dialect: sa.engine.Dialect) -> str | None:
        if value is None:
            return None
        return str(value)


class Base(DeclarativeBase):
    pass


concept_evidence_fragments = sa.Table(
    "concept_evidence_fragments",
    Base.metadata,
    sa.Column("concept_id", UUIDString(), ForeignKey("concepts.id", ondelete="CASCADE"), primary_key=True),
    sa.Column(
        "evidence_fragment_id",
        UUIDString(),
        ForeignKey("evidence_fragments.id", ondelete="RESTRICT"),
        primary_key=True,
    ),
    sa.Index("ix_concept_evidence_fragments_evidence", "evidence_fragment_id"),
)

concept_decision_records = sa.Table(
    "concept_decision_records",
    Base.metadata,
    sa.Column("concept_id", UUIDString(), ForeignKey("concepts.id", ondelete="CASCADE"), primary_key=True),
    sa.Column(
        "decision_record_id",
        UUIDString(),
        ForeignKey("decision_records.id", ondelete="RESTRICT"),
        primary_key=True,
    ),
    sa.Index("ix_concept_decision_records_decision", "decision_record_id"),
)

concept_relation_evidence_fragments = sa.Table(
    "concept_relation_evidence_fragments",
    Base.metadata,
    sa.Column(
        "concept_relation_id",
        UUIDString(),
        ForeignKey("concept_relations.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    sa.Column(
        "evidence_fragment_id",
        UUIDString(),
        ForeignKey("evidence_fragments.id", ondelete="RESTRICT"),
        primary_key=True,
    ),
    sa.Index("ix_concept_relation_evidence_fragments_evidence", "evidence_fragment_id"),
)

decision_record_evidence_fragments = sa.Table(
    "decision_record_evidence_fragments",
    Base.metadata,
    sa.Column(
        "decision_record_id",
        UUIDString(),
        ForeignKey("decision_records.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    sa.Column(
        "evidence_fragment_id",
        UUIDString(),
        ForeignKey("evidence_fragments.id", ondelete="RESTRICT"),
        primary_key=True,
    ),
    sa.Index("ix_decision_record_evidence_fragments_evidence", "evidence_fragment_id"),
)

decision_record_affected_concepts = sa.Table(
    "decision_record_affected_concepts",
    Base.metadata,
    sa.Column(
        "decision_record_id",
        UUIDString(),
        ForeignKey("decision_records.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    sa.Column("concept_id", UUIDString(), ForeignKey("concepts.id", ondelete="RESTRICT"), primary_key=True),
    sa.Index("ix_decision_record_affected_concepts_concept", "concept_id"),
)


class DataSourceRow(Base):
    __tablename__ = "data_sources"

    id: Mapped[str] = mapped_column(UUIDString(), primary_key=True, default=new_id)
    type: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    name: Mapped[str] = mapped_column(CITEXT_TYPE, nullable=False)
    status: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    production_enabled: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(TZ_DATETIME, nullable=False)
    auth_status: Mapped[str] = mapped_column(sa.String(32), nullable=False, default="not_started")
    connection_status: Mapped[str] = mapped_column(sa.String(32), nullable=False, default="untested")
    sync_status: Mapped[str] = mapped_column(sa.String(32), nullable=False, default="never_synced")
    next_action: Mapped[str] = mapped_column(sa.String(32), nullable=False, default="connect")
    last_connection_test_at: Mapped[datetime | None] = mapped_column(TZ_DATETIME, nullable=True)
    last_discovery_at: Mapped[datetime | None] = mapped_column(TZ_DATETIME, nullable=True)
    last_sync_at: Mapped[datetime | None] = mapped_column(TZ_DATETIME, nullable=True)
    last_successful_sync_at: Mapped[datetime | None] = mapped_column(TZ_DATETIME, nullable=True)
    last_error: Mapped[dict[str, Any] | None] = mapped_column(JSONB_TYPE, nullable=True)
    freshness_state: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    sync_freshness_state: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    content_freshness_state: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    artifact_count: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    evidence_fragment_count: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    discovered_object_count: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    selected_object_count: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)

    __table_args__ = (
        sa.Index("ix_data_sources_status", "status"),
        sa.Index("ix_data_sources_type", "type"),
        sa.Index("ix_data_sources_production_status", "production_enabled", "status"),
        sa.Index("ix_data_sources_runtime_state", "auth_status", "connection_status", "sync_status", "next_action"),
    )


class ArtifactRow(Base):
    __tablename__ = "artifacts"

    id: Mapped[str] = mapped_column(UUIDString(), primary_key=True, default=new_id)
    datasource_id: Mapped[str] = mapped_column(
        UUIDString(), ForeignKey("data_sources.id", ondelete="RESTRICT"), nullable=False
    )
    source_type: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    source_external_id: Mapped[str] = mapped_column(sa.String(512), nullable=False)
    source_url: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    source_object_type: Mapped[str] = mapped_column(sa.String(128), nullable=False, default="unknown")
    title: Mapped[str] = mapped_column(sa.Text, nullable=False)
    raw_content_hash: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    captured_at: Mapped[datetime] = mapped_column(TZ_DATETIME, nullable=False)
    source_updated_at: Mapped[datetime | None] = mapped_column(TZ_DATETIME, nullable=True)
    freshness_state: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    extraction_status: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    provider_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB_TYPE, nullable=False, default=dict)

    __table_args__ = (
        UniqueConstraint(
            "datasource_id",
            "source_external_id",
            "raw_content_hash",
            name="uq_artifacts_source_identity_hash",
        ),
        sa.Index("ix_artifacts_datasource_id", "datasource_id"),
        sa.Index("ix_artifacts_source_external_id", "source_external_id"),
        sa.Index("ix_artifacts_raw_content_hash", "raw_content_hash"),
        sa.Index("ix_artifacts_freshness_state", "freshness_state"),
    )


class EvidenceFragmentRow(Base):
    __tablename__ = "evidence_fragments"

    id: Mapped[str] = mapped_column(UUIDString(), primary_key=True, default=new_id)
    artifact_id: Mapped[str] = mapped_column(
        UUIDString(), ForeignKey("artifacts.id", ondelete="CASCADE"), nullable=False
    )
    text: Mapped[str] = mapped_column(sa.Text, nullable=False)
    fragment_type: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    provenance: Mapped[dict[str, Any]] = mapped_column(JSONB_TYPE, nullable=False)
    trust_state: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    freshness_state: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    reviewed_by: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    reviewed_at: Mapped[datetime | None] = mapped_column(TZ_DATETIME, nullable=True)

    __table_args__ = (
        sa.Index("ix_evidence_fragments_artifact_id", "artifact_id"),
        sa.Index("ix_evidence_fragments_trust_state", "trust_state"),
        sa.Index("ix_evidence_fragments_freshness_state", "freshness_state"),
    )


class DecisionRecordRow(Base):
    __tablename__ = "decision_records"

    id: Mapped[str] = mapped_column(UUIDString(), primary_key=True, default=new_id)
    title: Mapped[str] = mapped_column(sa.Text, nullable=False)
    decision: Mapped[str] = mapped_column(sa.Text, nullable=False)
    reason: Mapped[str] = mapped_column(sa.Text, nullable=False)
    alternatives_considered: Mapped[list[str]] = mapped_column(JSONB_TYPE, nullable=False, default=list)
    decided_by: Mapped[str] = mapped_column(sa.Text, nullable=False)
    decided_at: Mapped[datetime] = mapped_column(TZ_DATETIME, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TZ_DATETIME, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(TZ_DATETIME, nullable=False)

    __table_args__ = (
        sa.Index("ix_decision_records_decided_by", "decided_by"),
        sa.Index("ix_decision_records_decided_at", "decided_at"),
    )


class ConceptRow(Base):
    __tablename__ = "concepts"

    id: Mapped[str] = mapped_column(UUIDString(), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(CITEXT_TYPE, nullable=False)
    short_definition: Mapped[str] = mapped_column(sa.Text, nullable=False)
    body: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    status: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    owner: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    created_by: Mapped[str] = mapped_column(sa.Text, nullable=False)
    officialized_by: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TZ_DATETIME, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(TZ_DATETIME, nullable=False)
    last_reviewed_at: Mapped[datetime | None] = mapped_column(TZ_DATETIME, nullable=True)

    __table_args__ = (
        UniqueConstraint("name", name="uq_concepts_name"),
        sa.Index("ix_concepts_status", "status"),
        sa.Index("ix_concepts_owner", "owner"),
    )


class ConceptRelationRow(Base):
    __tablename__ = "concept_relations"

    id: Mapped[str] = mapped_column(UUIDString(), primary_key=True, default=new_id)
    source_concept_id: Mapped[str] = mapped_column(
        UUIDString(), ForeignKey("concepts.id", ondelete="RESTRICT"), nullable=False
    )
    target_concept_id: Mapped[str] = mapped_column(
        UUIDString(), ForeignKey("concepts.id", ondelete="RESTRICT"), nullable=False
    )
    relation_type: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    status: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    decision_record_id: Mapped[str | None] = mapped_column(
        UUIDString(), ForeignKey("decision_records.id", ondelete="RESTRICT"), nullable=True
    )
    created_by: Mapped[str] = mapped_column(sa.Text, nullable=False)
    officialized_by: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TZ_DATETIME, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(TZ_DATETIME, nullable=False)
    last_reviewed_at: Mapped[datetime | None] = mapped_column(TZ_DATETIME, nullable=True)

    __table_args__ = (
        sa.Index("ix_concept_relations_source", "source_concept_id"),
        sa.Index("ix_concept_relations_target", "target_concept_id"),
        sa.Index("ix_concept_relations_status", "status"),
        sa.Index("ix_concept_relations_type", "relation_type"),
        sa.CheckConstraint("source_concept_id <> target_concept_id", name="ck_concept_relations_distinct_concepts"),
    )


class AuditEventRow(Base):
    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(UUIDString(), primary_key=True, default=new_id)
    event_type: Mapped[str] = mapped_column(sa.Text, nullable=False)
    actor: Mapped[str] = mapped_column(sa.Text, nullable=False)
    entity_type: Mapped[str] = mapped_column(sa.Text, nullable=False)
    entity_id: Mapped[str] = mapped_column(UUIDString(), nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(TZ_DATETIME, nullable=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB_TYPE, nullable=False, default=dict)

    __table_args__ = (
        sa.Index("ix_audit_events_entity", "entity_type", "entity_id"),
        sa.Index("ix_audit_events_event_type", "event_type"),
        sa.Index("ix_audit_events_occurred_at", "occurred_at"),
    )


class ConnectionIntentRow(Base):
    __tablename__ = "connection_intents"

    id: Mapped[str] = mapped_column(UUIDString(), primary_key=True, default=new_id)
    provider: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    status: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    auth_type: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    source_name: Mapped[str] = mapped_column(sa.Text, nullable=False)
    created_by: Mapped[str] = mapped_column(sa.Text, nullable=False)
    requested_scopes: Mapped[list[str]] = mapped_column(JSONB_TYPE, nullable=False, default=list)
    authorization_url: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    redirect_uri: Mapped[str] = mapped_column(sa.Text, nullable=False)
    return_url: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    state_nonce: Mapped[str] = mapped_column(sa.String(256), nullable=False, unique=True)
    expires_at: Mapped[datetime] = mapped_column(TZ_DATETIME, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TZ_DATETIME, nullable=False)
    completed_at: Mapped[datetime | None] = mapped_column(TZ_DATETIME, nullable=True)
    datasource_id: Mapped[str | None] = mapped_column(
        UUIDString(), ForeignKey("data_sources.id", ondelete="SET NULL"), nullable=True
    )
    failure_error: Mapped[dict[str, Any] | None] = mapped_column(JSONB_TYPE, nullable=True)

    __table_args__ = (
        sa.Index("ix_connection_intents_provider_status", "provider", "status"),
        sa.Index("ix_connection_intents_created_by", "created_by"),
        sa.Index("ix_connection_intents_expires_at", "expires_at"),
    )


class ConnectorCredentialRow(Base):
    __tablename__ = "connector_credentials"

    id: Mapped[str] = mapped_column(UUIDString(), primary_key=True, default=new_id)
    datasource_id: Mapped[str] = mapped_column(
        UUIDString(), ForeignKey("data_sources.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    auth_type: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    encrypted_access_token: Mapped[str] = mapped_column(sa.Text, nullable=False)
    encrypted_refresh_token: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    granted_scopes: Mapped[list[str]] = mapped_column(JSONB_TYPE, nullable=False, default=list)
    external_account_id: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    external_workspace_id: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    external_workspace_name: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    external_bot_id: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    token_expires_at: Mapped[datetime | None] = mapped_column(TZ_DATETIME, nullable=True)
    status: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(TZ_DATETIME, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(TZ_DATETIME, nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(TZ_DATETIME, nullable=True)

    __table_args__ = (
        sa.Index("ix_connector_credentials_source_status", "datasource_id", "status"),
        sa.Index("ix_connector_credentials_provider", "provider"),
    )


class SourceSelectionRow(Base):
    __tablename__ = "source_selections"

    id: Mapped[str] = mapped_column(UUIDString(), primary_key=True, default=new_id)
    datasource_id: Mapped[str] = mapped_column(
        UUIDString(), ForeignKey("data_sources.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    sync_mode: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    include_rules: Mapped[list[dict[str, Any]]] = mapped_column(JSONB_TYPE, nullable=False, default=list)
    exclude_rules: Mapped[list[dict[str, Any]]] = mapped_column(JSONB_TYPE, nullable=False, default=list)
    selected_external_object_ids: Mapped[list[str]] = mapped_column(JSONB_TYPE, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(TZ_DATETIME, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(TZ_DATETIME, nullable=False)

    __table_args__ = (sa.Index("ix_source_selections_datasource_id", "datasource_id"),)


class ProviderObjectSnapshotRow(Base):
    __tablename__ = "provider_object_snapshots"

    id: Mapped[str] = mapped_column(UUIDString(), primary_key=True, default=new_id)
    datasource_id: Mapped[str] = mapped_column(
        UUIDString(), ForeignKey("data_sources.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    external_id: Mapped[str] = mapped_column(sa.Text, nullable=False)
    external_url: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    object_type: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    title: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    parent_external_id: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    last_edited_time: Mapped[datetime | None] = mapped_column(TZ_DATETIME, nullable=True)
    discovered_at: Mapped[datetime] = mapped_column(TZ_DATETIME, nullable=False)
    selected_for_sync: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, default=False)
    access_state: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    ingestion_supported: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, default=True)
    ingestion_unsupported_reason: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    raw_metadata_hash: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    provider_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB_TYPE, nullable=False, default=dict)

    __table_args__ = (
        UniqueConstraint("datasource_id", "external_id", name="uq_provider_object_snapshots_source_external"),
        sa.Index("ix_provider_object_snapshots_datasource", "datasource_id"),
        sa.Index("ix_provider_object_snapshots_access", "datasource_id", "access_state"),
        sa.Index("ix_provider_object_snapshots_ingestion", "datasource_id", "ingestion_supported"),
        sa.Index("ix_provider_object_snapshots_selected", "datasource_id", "selected_for_sync"),
        sa.Index("ix_provider_object_snapshots_type", "provider", "object_type"),
    )


class SyncJobRow(Base):
    __tablename__ = "sync_jobs"

    id: Mapped[str] = mapped_column(UUIDString(), primary_key=True, default=new_id)
    datasource_id: Mapped[str] = mapped_column(
        UUIDString(), ForeignKey("data_sources.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    status: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    trigger: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    created_by: Mapped[str] = mapped_column(sa.Text, nullable=False)
    selection_id: Mapped[str | None] = mapped_column(UUIDString(), nullable=True)
    created_at: Mapped[datetime] = mapped_column(TZ_DATETIME, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(TZ_DATETIME, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(TZ_DATETIME, nullable=True)
    artifact_created_count: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    artifact_reused_count: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    evidence_created_count: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    error: Mapped[dict[str, Any] | None] = mapped_column(JSONB_TYPE, nullable=True)
    cursor: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    attempt_count: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=3)
    next_attempt_at: Mapped[datetime | None] = mapped_column(TZ_DATETIME, nullable=True)
    cancel_requested_at: Mapped[datetime | None] = mapped_column(TZ_DATETIME, nullable=True)
    cancelled_by: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    lease_owner: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    lease_acquired_at: Mapped[datetime | None] = mapped_column(TZ_DATETIME, nullable=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(TZ_DATETIME, nullable=True)
    lease_heartbeat_at: Mapped[datetime | None] = mapped_column(TZ_DATETIME, nullable=True)
    schedule_id: Mapped[str | None] = mapped_column(UUIDString(), ForeignKey("sync_schedules.id", ondelete="SET NULL"), nullable=True)
    enqueue_key: Mapped[str | None] = mapped_column(sa.Text, nullable=True)

    __table_args__ = (
        sa.Index("ix_sync_jobs_datasource_status", "datasource_id", "status"),
        sa.Index("ix_sync_jobs_created_at", "created_at"),
        sa.Index("ix_sync_jobs_retry_ready", "status", "next_attempt_at"),
        sa.Index("ix_sync_jobs_lease_expiry", "status", "lease_expires_at"),
        sa.Index("ix_sync_jobs_claimable", "status", "next_attempt_at", "lease_expires_at"),
        sa.Index("ix_sync_jobs_schedule", "schedule_id"),
        sa.Index("ix_sync_jobs_enqueue_key", "enqueue_key", unique=True),
    )


class SyncCursorRow(Base):
    __tablename__ = "sync_cursors"

    id: Mapped[str] = mapped_column(UUIDString(), primary_key=True, default=new_id)
    datasource_id: Mapped[str] = mapped_column(
        UUIDString(), ForeignKey("data_sources.id", ondelete="CASCADE"), nullable=False
    )
    provider: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    cursor_key: Mapped[str] = mapped_column(sa.String(128), nullable=False, default="default")
    last_cursor: Mapped[str | None] = mapped_column(sa.Text, nullable=True)
    last_successful_sync_job_id: Mapped[str | None] = mapped_column(
        UUIDString(), ForeignKey("sync_jobs.id", ondelete="SET NULL"), nullable=True
    )
    processed_external_object_ids: Mapped[list[str]] = mapped_column(JSONB_TYPE, nullable=False, default=list)
    artifact_created_count: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    artifact_reused_count: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    evidence_created_count: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(TZ_DATETIME, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(TZ_DATETIME, nullable=False)
    advanced_at: Mapped[datetime | None] = mapped_column(TZ_DATETIME, nullable=True)

    __table_args__ = (
        UniqueConstraint("datasource_id", "cursor_key", name="uq_sync_cursors_datasource_key"),
        sa.Index("ix_sync_cursors_datasource", "datasource_id"),
        sa.Index("ix_sync_cursors_provider", "provider"),
    )


class SyncScheduleRow(Base):
    __tablename__ = "sync_schedules"

    id: Mapped[str] = mapped_column(UUIDString(), primary_key=True, default=new_id)
    datasource_id: Mapped[str] = mapped_column(
        UUIDString(), ForeignKey("data_sources.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    provider: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    status: Mapped[str] = mapped_column(sa.String(32), nullable=False)
    interval_minutes: Mapped[int] = mapped_column(sa.Integer, nullable=False)
    next_run_at: Mapped[datetime] = mapped_column(TZ_DATETIME, nullable=False)
    last_enqueued_at: Mapped[datetime | None] = mapped_column(TZ_DATETIME, nullable=True)
    last_enqueued_sync_job_id: Mapped[str | None] = mapped_column(
        UUIDString(), ForeignKey("sync_jobs.id", ondelete="SET NULL"), nullable=True
    )
    max_attempts: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=3)
    created_by: Mapped[str] = mapped_column(sa.Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TZ_DATETIME, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(TZ_DATETIME, nullable=False)

    __table_args__ = (
        sa.Index("ix_sync_schedules_status_next_run", "status", "next_run_at"),
        sa.Index("ix_sync_schedules_datasource", "datasource_id"),
        sa.Index("ix_sync_schedules_provider", "provider"),
    )


class SyncJobEventRow(Base):
    __tablename__ = "sync_job_events"

    id: Mapped[str] = mapped_column(UUIDString(), primary_key=True, default=new_id)
    sync_job_id: Mapped[str] = mapped_column(
        UUIDString(), ForeignKey("sync_jobs.id", ondelete="CASCADE"), nullable=False
    )
    datasource_id: Mapped[str] = mapped_column(UUIDString(), nullable=False)
    event_type: Mapped[str] = mapped_column(sa.Text, nullable=False)
    message: Mapped[str] = mapped_column(sa.Text, nullable=False)
    occurred_at: Mapped[datetime] = mapped_column(TZ_DATETIME, nullable=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB_TYPE, nullable=False, default=dict)

    __table_args__ = (
        sa.Index("ix_sync_job_events_job_time", "sync_job_id", "occurred_at"),
        sa.Index("ix_sync_job_events_datasource", "datasource_id"),
        sa.Index("ix_sync_job_events_event_type", "event_type"),
    )


class GroundedContextEvalTaskRow(Base):
    __tablename__ = "grounded_context_eval_tasks"

    id: Mapped[str] = mapped_column(UUIDString(), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(sa.Text, nullable=False)
    query: Mapped[str] = mapped_column(sa.Text, nullable=False)
    expected_answer_contains: Mapped[list[str]] = mapped_column(JSONB_TYPE, nullable=False, default=list)
    expected_trust_label: Mapped[str | None] = mapped_column(sa.String(64), nullable=True)
    expected_freshness_state: Mapped[str | None] = mapped_column(sa.String(32), nullable=True)
    required_evidence_fragment_ids: Mapped[list[str]] = mapped_column(JSONB_TYPE, nullable=False, default=list)
    required_concept_ids: Mapped[list[str]] = mapped_column(JSONB_TYPE, nullable=False, default=list)
    required_decision_record_ids: Mapped[list[str]] = mapped_column(JSONB_TYPE, nullable=False, default=list)
    require_official_answer: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, default=False)
    require_evidence: Mapped[bool] = mapped_column(sa.Boolean, nullable=False, default=True)
    min_evidence_count: Mapped[int] = mapped_column(sa.Integer, nullable=False, default=1)
    expected_clarification_reduced: Mapped[bool | None] = mapped_column(sa.Boolean, nullable=True)
    tags: Mapped[list[str]] = mapped_column(JSONB_TYPE, nullable=False, default=list)
    created_by: Mapped[str] = mapped_column(sa.Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TZ_DATETIME, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(TZ_DATETIME, nullable=False)
    metadata_: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB_TYPE, nullable=False, default=dict)

    __table_args__ = (
        sa.Index("ix_grounded_context_eval_tasks_created_at", "created_at"),
        sa.Index("ix_grounded_context_eval_tasks_expected_trust", "expected_trust_label"),
        sa.Index("ix_grounded_context_eval_tasks_query", "query"),
    )


class GroundedContextEvalResultRow(Base):
    __tablename__ = "grounded_context_eval_results"

    id: Mapped[str] = mapped_column(UUIDString(), primary_key=True, default=new_id)
    task_id: Mapped[str] = mapped_column(
        UUIDString(), ForeignKey("grounded_context_eval_tasks.id", ondelete="CASCADE"), nullable=False
    )
    response_id: Mapped[str] = mapped_column(UUIDString(), nullable=False)
    query: Mapped[str] = mapped_column(sa.Text, nullable=False)
    answer: Mapped[str] = mapped_column(sa.Text, nullable=False)
    trust_label: Mapped[str] = mapped_column(sa.String(64), nullable=False)
    response: Mapped[dict[str, Any]] = mapped_column(JSONB_TYPE, nullable=False)
    answer_correct: Mapped[bool] = mapped_column(sa.Boolean, nullable=False)
    evidence_valid: Mapped[bool] = mapped_column(sa.Boolean, nullable=False)
    provenance_present: Mapped[bool] = mapped_column(sa.Boolean, nullable=False)
    trust_label_correct: Mapped[bool] = mapped_column(sa.Boolean, nullable=False)
    freshness_policy_respected: Mapped[bool] = mapped_column(sa.Boolean, nullable=False)
    unsupported_official_claim: Mapped[bool] = mapped_column(sa.Boolean, nullable=False)
    citation_validity_rate: Mapped[float] = mapped_column(sa.Float, nullable=False)
    clarification_reduced: Mapped[bool | None] = mapped_column(sa.Boolean, nullable=True)
    success: Mapped[bool] = mapped_column(sa.Boolean, nullable=False)
    failure_reasons: Mapped[list[str]] = mapped_column(JSONB_TYPE, nullable=False, default=list)
    evaluated_at: Mapped[datetime] = mapped_column(TZ_DATETIME, nullable=False)
    evaluated_by: Mapped[str] = mapped_column(sa.Text, nullable=False)

    __table_args__ = (
        sa.Index("ix_grounded_context_eval_results_task", "task_id"),
        sa.Index("ix_grounded_context_eval_results_success", "success"),
        sa.Index("ix_grounded_context_eval_results_evaluated_at", "evaluated_at"),
        sa.Index("ix_grounded_context_eval_results_trust", "trust_label"),
    )
