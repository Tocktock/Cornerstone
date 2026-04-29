"""postgres persistence foundation

Revision ID: 0001_postgres_persistence
Revises:
Create Date: 2026-04-25
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0001_postgres_persistence"
down_revision = None
branch_labels = None
depends_on = None

UUID = postgresql.UUID(as_uuid=False)
JSONB = postgresql.JSONB


def upgrade() -> None:
    # Required extensions for production persistence:
    # - pgcrypto: UUID generation via gen_random_uuid()
    # - citext: case-insensitive concept/source natural names
    # - vector: future evidence embedding and semantic retrieval storage
    op.execute('CREATE EXTENSION IF NOT EXISTS "pgcrypto";')
    op.execute('CREATE EXTENSION IF NOT EXISTS "citext";')
    op.execute('CREATE EXTENSION IF NOT EXISTS "vector";')

    # Alembic creates alembic_version.version_num as VARCHAR(32) by default.
    # Cornerstone uses descriptive revision IDs longer than 32 characters
    # (for example, 0007_provider_object_ingestion_support), so widen the
    # column during the first migration before later revision IDs are stored.
    op.execute(
        "ALTER TABLE IF EXISTS alembic_version "
        "ALTER COLUMN version_num TYPE VARCHAR(128);"
    )

    op.create_table(
        "data_sources",
        sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("type", sa.String(length=32), nullable=False),
        sa.Column("name", postgresql.CITEXT(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("production_enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("last_sync_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_successful_sync_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", JSONB, nullable=True),
        sa.Column("freshness_state", sa.String(length=32), nullable=False),
        sa.Column("sync_freshness_state", sa.String(length=32), nullable=False),
        sa.Column("content_freshness_state", sa.String(length=32), nullable=False),
        sa.Column("artifact_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("evidence_fragment_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.CheckConstraint(
            "status in ('disconnected','connecting','pending_auth','connected','sync_pending','syncing','degraded','failed','stale')",
            name="ck_data_sources_status",
        ),
        sa.CheckConstraint(
            "freshness_state in ('fresh','aging','stale','unknown','mixed')",
            name="ck_data_sources_freshness_state",
        ),
        sa.CheckConstraint(
            "sync_freshness_state in ('fresh','aging','stale','unknown','mixed')",
            name="ck_data_sources_sync_freshness_state",
        ),
        sa.CheckConstraint(
            "content_freshness_state in ('fresh','aging','stale','unknown','mixed')",
            name="ck_data_sources_content_freshness_state",
        ),
    )
    op.create_index("ix_data_sources_status", "data_sources", ["status"])
    op.create_index("ix_data_sources_type", "data_sources", ["type"])
    op.create_index(
        "ix_data_sources_production_status", "data_sources", ["production_enabled", "status"]
    )

    op.create_table(
        "artifacts",
        sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("datasource_id", UUID, sa.ForeignKey("data_sources.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("source_type", sa.String(length=32), nullable=False),
        sa.Column("source_external_id", sa.String(length=512), nullable=False),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("raw_content_hash", sa.String(length=64), nullable=False),
        sa.Column("captured_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("source_updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("freshness_state", sa.String(length=32), nullable=False),
        sa.Column("extraction_status", sa.String(length=32), nullable=False),
        sa.UniqueConstraint(
            "datasource_id",
            "source_external_id",
            "raw_content_hash",
            name="uq_artifacts_source_identity_hash",
        ),
        sa.CheckConstraint(
            "freshness_state in ('fresh','aging','stale','unknown','mixed')",
            name="ck_artifacts_freshness_state",
        ),
        sa.CheckConstraint(
            "extraction_status in ('pending','complete','failed')",
            name="ck_artifacts_extraction_status",
        ),
    )
    op.create_index("ix_artifacts_datasource_id", "artifacts", ["datasource_id"])
    op.create_index("ix_artifacts_source_external_id", "artifacts", ["source_external_id"])
    op.create_index("ix_artifacts_raw_content_hash", "artifacts", ["raw_content_hash"])
    op.create_index("ix_artifacts_freshness_state", "artifacts", ["freshness_state"])

    op.create_table(
        "evidence_fragments",
        sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("artifact_id", UUID, sa.ForeignKey("artifacts.id", ondelete="CASCADE"), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("fragment_type", sa.String(length=32), nullable=False),
        sa.Column("provenance", JSONB, nullable=False),
        sa.Column("trust_state", sa.String(length=32), nullable=False),
        sa.Column("freshness_state", sa.String(length=32), nullable=False),
        sa.Column("reviewed_by", sa.Text(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint("length(text) > 0", name="ck_evidence_fragments_text_non_empty"),
        sa.CheckConstraint(
            "fragment_type in ('definition','decision','policy','requirement','example','claim','open_question')",
            name="ck_evidence_fragments_fragment_type",
        ),
        sa.CheckConstraint(
            "trust_state in ('unreviewed','reviewed','rejected')",
            name="ck_evidence_fragments_trust_state",
        ),
        sa.CheckConstraint(
            "freshness_state in ('fresh','aging','stale','unknown','mixed')",
            name="ck_evidence_fragments_freshness_state",
        ),
        sa.CheckConstraint(
            "provenance ? 'data_source_id' AND provenance ? 'source_external_id' AND provenance ? 'artifact_title' AND provenance ? 'captured_at'",
            name="ck_evidence_fragments_required_provenance_keys",
        ),
    )
    op.create_index("ix_evidence_fragments_artifact_id", "evidence_fragments", ["artifact_id"])
    op.create_index("ix_evidence_fragments_trust_state", "evidence_fragments", ["trust_state"])
    op.create_index("ix_evidence_fragments_freshness_state", "evidence_fragments", ["freshness_state"])
    op.create_index(
        "ix_evidence_fragments_provenance_gin",
        "evidence_fragments",
        ["provenance"],
        postgresql_using="gin",
    )

    op.create_table(
        "decision_records",
        sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("decision", sa.Text(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("alternatives_considered", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("decided_by", sa.Text(), nullable=False),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.CheckConstraint("length(title) > 0", name="ck_decision_records_title_non_empty"),
        sa.CheckConstraint("length(decision) > 0", name="ck_decision_records_decision_non_empty"),
        sa.CheckConstraint("length(reason) > 0", name="ck_decision_records_reason_non_empty"),
    )
    op.create_index("ix_decision_records_decided_by", "decision_records", ["decided_by"])
    op.create_index("ix_decision_records_decided_at", "decision_records", ["decided_at"])

    op.create_table(
        "concepts",
        sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("name", postgresql.CITEXT(), nullable=False),
        sa.Column("short_definition", sa.Text(), nullable=False),
        sa.Column("body", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("owner", sa.Text(), nullable=True),
        sa.Column("created_by", sa.Text(), nullable=False),
        sa.Column("officialized_by", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("last_reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("name", name="uq_concepts_name"),
        sa.CheckConstraint("length(name) > 0", name="ck_concepts_name_non_empty"),
        sa.CheckConstraint("length(short_definition) > 0", name="ck_concepts_short_definition_non_empty"),
        sa.CheckConstraint(
            "status in ('candidate','reviewing','official','conflicted','deprecated','superseded')",
            name="ck_concepts_status",
        ),
    )
    op.create_index("ix_concepts_status", "concepts", ["status"])
    op.create_index("ix_concepts_owner", "concepts", ["owner"])

    op.create_table(
        "concept_evidence_fragments",
        sa.Column("concept_id", UUID, sa.ForeignKey("concepts.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("evidence_fragment_id", UUID, sa.ForeignKey("evidence_fragments.id", ondelete="RESTRICT"), primary_key=True),
    )
    op.create_index(
        "ix_concept_evidence_fragments_evidence",
        "concept_evidence_fragments",
        ["evidence_fragment_id"],
    )

    op.create_table(
        "concept_decision_records",
        sa.Column("concept_id", UUID, sa.ForeignKey("concepts.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("decision_record_id", UUID, sa.ForeignKey("decision_records.id", ondelete="RESTRICT"), primary_key=True),
    )
    op.create_index(
        "ix_concept_decision_records_decision",
        "concept_decision_records",
        ["decision_record_id"],
    )

    op.create_table(
        "decision_record_evidence_fragments",
        sa.Column("decision_record_id", UUID, sa.ForeignKey("decision_records.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("evidence_fragment_id", UUID, sa.ForeignKey("evidence_fragments.id", ondelete="RESTRICT"), primary_key=True),
    )
    op.create_index(
        "ix_decision_record_evidence_fragments_evidence",
        "decision_record_evidence_fragments",
        ["evidence_fragment_id"],
    )

    op.create_table(
        "decision_record_affected_concepts",
        sa.Column("decision_record_id", UUID, sa.ForeignKey("decision_records.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("concept_id", UUID, sa.ForeignKey("concepts.id", ondelete="RESTRICT"), primary_key=True),
    )
    op.create_index(
        "ix_decision_record_affected_concepts_concept",
        "decision_record_affected_concepts",
        ["concept_id"],
    )

    op.create_table(
        "audit_events",
        sa.Column("id", UUID, primary_key=True, server_default=sa.text("gen_random_uuid()")),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("actor", sa.Text(), nullable=False),
        sa.Column("entity_type", sa.Text(), nullable=False),
        sa.Column("entity_id", UUID, nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.text("now()")),
        sa.Column("metadata", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
    )
    op.create_index("ix_audit_events_entity", "audit_events", ["entity_type", "entity_id"])
    op.create_index("ix_audit_events_event_type", "audit_events", ["event_type"])
    op.create_index("ix_audit_events_occurred_at", "audit_events", ["occurred_at"])
    op.create_index(
        "ix_audit_events_metadata_gin",
        "audit_events",
        ["metadata"],
        postgresql_using="gin",
    )

    # Prepared table for later semantic retrieval. No API writes to this yet, but having it in the
    # persistence foundation lets the DB contract reserve pgvector-backed storage cleanly.
    op.execute(
        """
        CREATE TABLE evidence_embeddings (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            evidence_fragment_id UUID NOT NULL REFERENCES evidence_fragments(id) ON DELETE CASCADE,
            embedding_model TEXT NOT NULL,
            dimensions INTEGER NOT NULL CHECK (dimensions > 0),
            embedding VECTOR(1536) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (evidence_fragment_id, embedding_model)
        );
        """
    )
    op.execute(
        "CREATE INDEX ix_evidence_embeddings_fragment ON evidence_embeddings (evidence_fragment_id);"
    )
    op.execute(
        "CREATE INDEX ix_evidence_embeddings_vector_hnsw ON evidence_embeddings USING hnsw (embedding vector_cosine_ops);"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_evidence_embeddings_vector_hnsw;")
    op.execute("DROP TABLE IF EXISTS evidence_embeddings;")
    op.drop_table("audit_events")
    op.drop_table("decision_record_affected_concepts")
    op.drop_table("decision_record_evidence_fragments")
    op.drop_table("concept_decision_records")
    op.drop_table("concept_evidence_fragments")
    op.drop_table("concepts")
    op.drop_table("decision_records")
    op.drop_table("evidence_fragments")
    op.drop_table("artifacts")
    op.drop_table("data_sources")
