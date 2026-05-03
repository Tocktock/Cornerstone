BEGIN;

CREATE TABLE alembic_version (
    version_num VARCHAR(32) NOT NULL, 
    CONSTRAINT alembic_version_pkc PRIMARY KEY (version_num)
);

-- Running upgrade  -> 0001_postgres_persistence

CREATE EXTENSION IF NOT EXISTS "pgcrypto";;

CREATE EXTENSION IF NOT EXISTS "citext";;

CREATE EXTENSION IF NOT EXISTS "vector";;

ALTER TABLE IF EXISTS alembic_version ALTER COLUMN version_num TYPE VARCHAR(128);;

CREATE TABLE data_sources (
    id UUID DEFAULT gen_random_uuid() NOT NULL, 
    type VARCHAR(32) NOT NULL, 
    name CITEXT NOT NULL, 
    status VARCHAR(32) NOT NULL, 
    production_enabled BOOLEAN DEFAULT true NOT NULL, 
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    last_sync_at TIMESTAMP WITH TIME ZONE, 
    last_successful_sync_at TIMESTAMP WITH TIME ZONE, 
    last_error JSONB, 
    freshness_state VARCHAR(32) NOT NULL, 
    sync_freshness_state VARCHAR(32) NOT NULL, 
    content_freshness_state VARCHAR(32) NOT NULL, 
    artifact_count INTEGER DEFAULT 0 NOT NULL, 
    evidence_fragment_count INTEGER DEFAULT 0 NOT NULL, 
    PRIMARY KEY (id), 
    CONSTRAINT ck_data_sources_status CHECK (status in ('disconnected','connecting','pending_auth','connected','sync_pending','syncing','degraded','failed','stale')), 
    CONSTRAINT ck_data_sources_freshness_state CHECK (freshness_state in ('fresh','aging','stale','unknown','mixed')), 
    CONSTRAINT ck_data_sources_sync_freshness_state CHECK (sync_freshness_state in ('fresh','aging','stale','unknown','mixed')), 
    CONSTRAINT ck_data_sources_content_freshness_state CHECK (content_freshness_state in ('fresh','aging','stale','unknown','mixed'))
);

CREATE INDEX ix_data_sources_status ON data_sources (status);

CREATE INDEX ix_data_sources_type ON data_sources (type);

CREATE INDEX ix_data_sources_production_status ON data_sources (production_enabled, status);

CREATE TABLE artifacts (
    id UUID DEFAULT gen_random_uuid() NOT NULL, 
    datasource_id UUID NOT NULL, 
    source_type VARCHAR(32) NOT NULL, 
    source_external_id VARCHAR(512) NOT NULL, 
    source_url TEXT, 
    title TEXT NOT NULL, 
    raw_content_hash VARCHAR(64) NOT NULL, 
    captured_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    source_updated_at TIMESTAMP WITH TIME ZONE, 
    freshness_state VARCHAR(32) NOT NULL, 
    extraction_status VARCHAR(32) NOT NULL, 
    PRIMARY KEY (id), 
    CONSTRAINT uq_artifacts_source_identity_hash UNIQUE (datasource_id, source_external_id, raw_content_hash), 
    CONSTRAINT ck_artifacts_freshness_state CHECK (freshness_state in ('fresh','aging','stale','unknown','mixed')), 
    CONSTRAINT ck_artifacts_extraction_status CHECK (extraction_status in ('pending','complete','failed')), 
    FOREIGN KEY(datasource_id) REFERENCES data_sources (id) ON DELETE RESTRICT
);

CREATE INDEX ix_artifacts_datasource_id ON artifacts (datasource_id);

CREATE INDEX ix_artifacts_source_external_id ON artifacts (source_external_id);

CREATE INDEX ix_artifacts_raw_content_hash ON artifacts (raw_content_hash);

CREATE INDEX ix_artifacts_freshness_state ON artifacts (freshness_state);

CREATE TABLE evidence_fragments (
    id UUID DEFAULT gen_random_uuid() NOT NULL, 
    artifact_id UUID NOT NULL, 
    text TEXT NOT NULL, 
    fragment_type VARCHAR(32) NOT NULL, 
    provenance JSONB NOT NULL, 
    trust_state VARCHAR(32) NOT NULL, 
    freshness_state VARCHAR(32) NOT NULL, 
    reviewed_by TEXT, 
    reviewed_at TIMESTAMP WITH TIME ZONE, 
    PRIMARY KEY (id), 
    CONSTRAINT ck_evidence_fragments_text_non_empty CHECK (length(text) > 0), 
    CONSTRAINT ck_evidence_fragments_fragment_type CHECK (fragment_type in ('definition','decision','policy','requirement','example','claim','open_question')), 
    CONSTRAINT ck_evidence_fragments_trust_state CHECK (trust_state in ('unreviewed','reviewed','rejected')), 
    CONSTRAINT ck_evidence_fragments_freshness_state CHECK (freshness_state in ('fresh','aging','stale','unknown','mixed')), 
    CONSTRAINT ck_evidence_fragments_required_provenance_keys CHECK (provenance ? 'data_source_id' AND provenance ? 'source_external_id' AND provenance ? 'artifact_title' AND provenance ? 'captured_at'), 
    FOREIGN KEY(artifact_id) REFERENCES artifacts (id) ON DELETE CASCADE
);

CREATE INDEX ix_evidence_fragments_artifact_id ON evidence_fragments (artifact_id);

CREATE INDEX ix_evidence_fragments_trust_state ON evidence_fragments (trust_state);

CREATE INDEX ix_evidence_fragments_freshness_state ON evidence_fragments (freshness_state);

CREATE INDEX ix_evidence_fragments_provenance_gin ON evidence_fragments USING gin (provenance);

CREATE TABLE decision_records (
    id UUID DEFAULT gen_random_uuid() NOT NULL, 
    title TEXT NOT NULL, 
    decision TEXT NOT NULL, 
    reason TEXT NOT NULL, 
    alternatives_considered JSONB DEFAULT '[]'::jsonb NOT NULL, 
    decided_by TEXT NOT NULL, 
    decided_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    PRIMARY KEY (id), 
    CONSTRAINT ck_decision_records_title_non_empty CHECK (length(title) > 0), 
    CONSTRAINT ck_decision_records_decision_non_empty CHECK (length(decision) > 0), 
    CONSTRAINT ck_decision_records_reason_non_empty CHECK (length(reason) > 0)
);

CREATE INDEX ix_decision_records_decided_by ON decision_records (decided_by);

CREATE INDEX ix_decision_records_decided_at ON decision_records (decided_at);

CREATE TABLE concepts (
    id UUID DEFAULT gen_random_uuid() NOT NULL, 
    name CITEXT NOT NULL, 
    short_definition TEXT NOT NULL, 
    body TEXT, 
    status VARCHAR(32) NOT NULL, 
    owner TEXT, 
    created_by TEXT NOT NULL, 
    officialized_by TEXT, 
    created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    last_reviewed_at TIMESTAMP WITH TIME ZONE, 
    PRIMARY KEY (id), 
    CONSTRAINT uq_concepts_name UNIQUE (name), 
    CONSTRAINT ck_concepts_name_non_empty CHECK (length(name) > 0), 
    CONSTRAINT ck_concepts_short_definition_non_empty CHECK (length(short_definition) > 0), 
    CONSTRAINT ck_concepts_status CHECK (status in ('candidate','reviewing','official','conflicted','deprecated','superseded'))
);

CREATE INDEX ix_concepts_status ON concepts (status);

CREATE INDEX ix_concepts_owner ON concepts (owner);

CREATE TABLE concept_evidence_fragments (
    concept_id UUID NOT NULL, 
    evidence_fragment_id UUID NOT NULL, 
    PRIMARY KEY (concept_id, evidence_fragment_id), 
    FOREIGN KEY(concept_id) REFERENCES concepts (id) ON DELETE CASCADE, 
    FOREIGN KEY(evidence_fragment_id) REFERENCES evidence_fragments (id) ON DELETE RESTRICT
);

CREATE INDEX ix_concept_evidence_fragments_evidence ON concept_evidence_fragments (evidence_fragment_id);

CREATE TABLE concept_decision_records (
    concept_id UUID NOT NULL, 
    decision_record_id UUID NOT NULL, 
    PRIMARY KEY (concept_id, decision_record_id), 
    FOREIGN KEY(concept_id) REFERENCES concepts (id) ON DELETE CASCADE, 
    FOREIGN KEY(decision_record_id) REFERENCES decision_records (id) ON DELETE RESTRICT
);

CREATE INDEX ix_concept_decision_records_decision ON concept_decision_records (decision_record_id);

CREATE TABLE decision_record_evidence_fragments (
    decision_record_id UUID NOT NULL, 
    evidence_fragment_id UUID NOT NULL, 
    PRIMARY KEY (decision_record_id, evidence_fragment_id), 
    FOREIGN KEY(decision_record_id) REFERENCES decision_records (id) ON DELETE CASCADE, 
    FOREIGN KEY(evidence_fragment_id) REFERENCES evidence_fragments (id) ON DELETE RESTRICT
);

CREATE INDEX ix_decision_record_evidence_fragments_evidence ON decision_record_evidence_fragments (evidence_fragment_id);

CREATE TABLE decision_record_affected_concepts (
    decision_record_id UUID NOT NULL, 
    concept_id UUID NOT NULL, 
    PRIMARY KEY (decision_record_id, concept_id), 
    FOREIGN KEY(decision_record_id) REFERENCES decision_records (id) ON DELETE CASCADE, 
    FOREIGN KEY(concept_id) REFERENCES concepts (id) ON DELETE RESTRICT
);

CREATE INDEX ix_decision_record_affected_concepts_concept ON decision_record_affected_concepts (concept_id);

CREATE TABLE audit_events (
    id UUID DEFAULT gen_random_uuid() NOT NULL, 
    event_type TEXT NOT NULL, 
    actor TEXT NOT NULL, 
    entity_type TEXT NOT NULL, 
    entity_id UUID NOT NULL, 
    occurred_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL, 
    metadata JSONB DEFAULT '{}'::jsonb NOT NULL, 
    PRIMARY KEY (id)
);

CREATE INDEX ix_audit_events_entity ON audit_events (entity_type, entity_id);

CREATE INDEX ix_audit_events_event_type ON audit_events (event_type);

CREATE INDEX ix_audit_events_occurred_at ON audit_events (occurred_at);

CREATE INDEX ix_audit_events_metadata_gin ON audit_events USING gin (metadata);

CREATE TABLE evidence_embeddings (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            evidence_fragment_id UUID NOT NULL REFERENCES evidence_fragments(id) ON DELETE CASCADE,
            embedding_model TEXT NOT NULL,
            dimensions INTEGER NOT NULL CHECK (dimensions > 0),
            embedding VECTOR(1536) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (evidence_fragment_id, embedding_model)
        );;

CREATE INDEX ix_evidence_embeddings_fragment ON evidence_embeddings (evidence_fragment_id);;

CREATE INDEX ix_evidence_embeddings_vector_hnsw ON evidence_embeddings USING hnsw (embedding vector_cosine_ops);;

INSERT INTO alembic_version (version_num) VALUES ('0001_postgres_persistence') RETURNING alembic_version.version_num;

-- Running upgrade 0001_postgres_persistence -> 0002_connector_framework

CREATE TABLE connection_intents (
    id UUID NOT NULL, 
    provider VARCHAR(32) NOT NULL, 
    status VARCHAR(32) NOT NULL, 
    auth_type VARCHAR(32) NOT NULL, 
    source_name TEXT NOT NULL, 
    created_by TEXT NOT NULL, 
    requested_scopes JSONB NOT NULL, 
    authorization_url TEXT, 
    redirect_uri TEXT NOT NULL, 
    return_url TEXT, 
    state_nonce VARCHAR(256) NOT NULL, 
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    completed_at TIMESTAMP WITH TIME ZONE, 
    datasource_id UUID, 
    failure_error JSONB, 
    PRIMARY KEY (id), 
    FOREIGN KEY(datasource_id) REFERENCES data_sources (id) ON DELETE SET NULL, 
    UNIQUE (state_nonce)
);

CREATE INDEX ix_connection_intents_created_by ON connection_intents (created_by);

CREATE INDEX ix_connection_intents_expires_at ON connection_intents (expires_at);

CREATE INDEX ix_connection_intents_provider_status ON connection_intents (provider, status);

CREATE TABLE connector_credentials (
    id UUID NOT NULL, 
    datasource_id UUID NOT NULL, 
    provider VARCHAR(32) NOT NULL, 
    auth_type VARCHAR(32) NOT NULL, 
    encrypted_access_token TEXT NOT NULL, 
    encrypted_refresh_token TEXT, 
    granted_scopes JSONB NOT NULL, 
    external_account_id TEXT, 
    external_workspace_id TEXT, 
    external_workspace_name TEXT, 
    external_bot_id TEXT, 
    token_expires_at TIMESTAMP WITH TIME ZONE, 
    status VARCHAR(32) NOT NULL, 
    created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    revoked_at TIMESTAMP WITH TIME ZONE, 
    PRIMARY KEY (id), 
    FOREIGN KEY(datasource_id) REFERENCES data_sources (id) ON DELETE CASCADE
);

CREATE INDEX ix_connector_credentials_provider ON connector_credentials (provider);

CREATE INDEX ix_connector_credentials_source_status ON connector_credentials (datasource_id, status);

CREATE TABLE source_selections (
    id UUID NOT NULL, 
    datasource_id UUID NOT NULL, 
    sync_mode VARCHAR(64) NOT NULL, 
    include_rules JSONB NOT NULL, 
    exclude_rules JSONB NOT NULL, 
    selected_external_object_ids JSONB NOT NULL, 
    created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    PRIMARY KEY (id), 
    FOREIGN KEY(datasource_id) REFERENCES data_sources (id) ON DELETE CASCADE, 
    UNIQUE (datasource_id)
);

CREATE INDEX ix_source_selections_datasource_id ON source_selections (datasource_id);

CREATE TABLE sync_jobs (
    id UUID NOT NULL, 
    datasource_id UUID NOT NULL, 
    provider VARCHAR(32) NOT NULL, 
    status VARCHAR(32) NOT NULL, 
    trigger VARCHAR(32) NOT NULL, 
    created_by TEXT NOT NULL, 
    selection_id UUID, 
    created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    started_at TIMESTAMP WITH TIME ZONE, 
    finished_at TIMESTAMP WITH TIME ZONE, 
    artifact_created_count INTEGER NOT NULL, 
    artifact_reused_count INTEGER NOT NULL, 
    evidence_created_count INTEGER NOT NULL, 
    error JSONB, 
    cursor TEXT, 
    PRIMARY KEY (id), 
    FOREIGN KEY(datasource_id) REFERENCES data_sources (id) ON DELETE CASCADE
);

CREATE INDEX ix_sync_jobs_created_at ON sync_jobs (created_at);

CREATE INDEX ix_sync_jobs_datasource_status ON sync_jobs (datasource_id, status);

CREATE TABLE sync_job_events (
    id UUID NOT NULL, 
    sync_job_id UUID NOT NULL, 
    datasource_id UUID NOT NULL, 
    event_type TEXT NOT NULL, 
    message TEXT NOT NULL, 
    occurred_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    metadata JSONB NOT NULL, 
    PRIMARY KEY (id), 
    FOREIGN KEY(sync_job_id) REFERENCES sync_jobs (id) ON DELETE CASCADE
);

CREATE INDEX ix_sync_job_events_datasource ON sync_job_events (datasource_id);

CREATE INDEX ix_sync_job_events_event_type ON sync_job_events (event_type);

CREATE INDEX ix_sync_job_events_job_time ON sync_job_events (sync_job_id, occurred_at);

UPDATE alembic_version SET version_num='0002_connector_framework' WHERE alembic_version.version_num = '0001_postgres_persistence';

-- Running upgrade 0002_connector_framework -> 0003_source_state_discovery

ALTER TABLE data_sources ADD COLUMN auth_status VARCHAR(32) DEFAULT 'not_started' NOT NULL;

ALTER TABLE data_sources ADD COLUMN connection_status VARCHAR(32) DEFAULT 'untested' NOT NULL;

ALTER TABLE data_sources ADD COLUMN sync_status VARCHAR(32) DEFAULT 'never_synced' NOT NULL;

ALTER TABLE data_sources ADD COLUMN next_action VARCHAR(32) DEFAULT 'connect' NOT NULL;

ALTER TABLE data_sources ADD COLUMN last_connection_test_at TIMESTAMP WITH TIME ZONE;

ALTER TABLE data_sources ADD COLUMN last_discovery_at TIMESTAMP WITH TIME ZONE;

ALTER TABLE data_sources ADD COLUMN discovered_object_count INTEGER DEFAULT '0' NOT NULL;

ALTER TABLE data_sources ADD COLUMN selected_object_count INTEGER DEFAULT '0' NOT NULL;

CREATE INDEX ix_data_sources_runtime_state ON data_sources (auth_status, connection_status, sync_status, next_action);

CREATE TABLE provider_object_snapshots (
    id UUID NOT NULL, 
    datasource_id UUID NOT NULL, 
    provider VARCHAR(32) NOT NULL, 
    external_id TEXT NOT NULL, 
    external_url TEXT, 
    object_type VARCHAR(32) NOT NULL, 
    title TEXT, 
    parent_external_id TEXT, 
    last_edited_time TIMESTAMP WITH TIME ZONE, 
    discovered_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    selected_for_sync BOOLEAN DEFAULT false NOT NULL, 
    access_state VARCHAR(32) NOT NULL, 
    raw_metadata_hash VARCHAR(64) NOT NULL, 
    PRIMARY KEY (id), 
    FOREIGN KEY(datasource_id) REFERENCES data_sources (id) ON DELETE CASCADE, 
    CONSTRAINT uq_provider_object_snapshots_source_external UNIQUE (datasource_id, external_id)
);

CREATE INDEX ix_provider_object_snapshots_datasource ON provider_object_snapshots (datasource_id);

CREATE INDEX ix_provider_object_snapshots_access ON provider_object_snapshots (datasource_id, access_state);

CREATE INDEX ix_provider_object_snapshots_selected ON provider_object_snapshots (datasource_id, selected_for_sync);

CREATE INDEX ix_provider_object_snapshots_type ON provider_object_snapshots (provider, object_type);

ALTER TABLE data_sources ALTER COLUMN auth_status DROP DEFAULT;

ALTER TABLE data_sources ALTER COLUMN connection_status DROP DEFAULT;

ALTER TABLE data_sources ALTER COLUMN sync_status DROP DEFAULT;

ALTER TABLE data_sources ALTER COLUMN next_action DROP DEFAULT;

ALTER TABLE data_sources ALTER COLUMN discovered_object_count DROP DEFAULT;

ALTER TABLE data_sources ALTER COLUMN selected_object_count DROP DEFAULT;

UPDATE alembic_version SET version_num='0003_source_state_discovery' WHERE alembic_version.version_num = '0002_connector_framework';

-- Running upgrade 0003_source_state_discovery -> 0004_generic_ingestion_contract

ALTER TABLE artifacts ADD COLUMN source_object_type VARCHAR(128) DEFAULT 'unknown' NOT NULL;

ALTER TABLE artifacts ADD COLUMN provider_metadata JSONB DEFAULT '{}'::jsonb NOT NULL;

ALTER TABLE provider_object_snapshots ADD COLUMN provider_metadata JSONB DEFAULT '{}'::jsonb NOT NULL;

CREATE INDEX ix_artifacts_source_object_type ON artifacts (source_type, source_object_type);

ALTER TABLE artifacts ALTER COLUMN source_object_type DROP DEFAULT;

ALTER TABLE artifacts ALTER COLUMN provider_metadata DROP DEFAULT;

ALTER TABLE provider_object_snapshots ALTER COLUMN provider_metadata DROP DEFAULT;

UPDATE alembic_version SET version_num='0004_generic_ingestion_contract' WHERE alembic_version.version_num = '0003_source_state_discovery';

-- Running upgrade 0004_generic_ingestion_contract -> 0005_durable_sync_worker

ALTER TABLE sync_jobs ADD COLUMN attempt_count INTEGER DEFAULT '0' NOT NULL;

ALTER TABLE sync_jobs ADD COLUMN max_attempts INTEGER DEFAULT '3' NOT NULL;

ALTER TABLE sync_jobs ADD COLUMN next_attempt_at TIMESTAMP WITH TIME ZONE;

ALTER TABLE sync_jobs ADD COLUMN cancel_requested_at TIMESTAMP WITH TIME ZONE;

ALTER TABLE sync_jobs ADD COLUMN cancelled_by TEXT;

CREATE INDEX ix_sync_jobs_retry_ready ON sync_jobs (status, next_attempt_at);

CREATE TABLE sync_cursors (
    id UUID NOT NULL, 
    datasource_id UUID NOT NULL, 
    provider VARCHAR(32) NOT NULL, 
    cursor_key VARCHAR(128) DEFAULT 'default' NOT NULL, 
    last_cursor TEXT, 
    last_successful_sync_job_id UUID, 
    processed_external_object_ids JSONB DEFAULT '[]'::jsonb NOT NULL, 
    artifact_created_count INTEGER DEFAULT '0' NOT NULL, 
    artifact_reused_count INTEGER DEFAULT '0' NOT NULL, 
    evidence_created_count INTEGER DEFAULT '0' NOT NULL, 
    created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    advanced_at TIMESTAMP WITH TIME ZONE, 
    PRIMARY KEY (id), 
    FOREIGN KEY(datasource_id) REFERENCES data_sources (id) ON DELETE CASCADE, 
    FOREIGN KEY(last_successful_sync_job_id) REFERENCES sync_jobs (id) ON DELETE SET NULL, 
    CONSTRAINT uq_sync_cursors_datasource_key UNIQUE (datasource_id, cursor_key)
);

CREATE INDEX ix_sync_cursors_datasource ON sync_cursors (datasource_id);

CREATE INDEX ix_sync_cursors_provider ON sync_cursors (provider);

ALTER TABLE sync_jobs ALTER COLUMN attempt_count DROP DEFAULT;

ALTER TABLE sync_jobs ALTER COLUMN max_attempts DROP DEFAULT;

ALTER TABLE sync_cursors ALTER COLUMN cursor_key DROP DEFAULT;

ALTER TABLE sync_cursors ALTER COLUMN processed_external_object_ids DROP DEFAULT;

ALTER TABLE sync_cursors ALTER COLUMN artifact_created_count DROP DEFAULT;

ALTER TABLE sync_cursors ALTER COLUMN artifact_reused_count DROP DEFAULT;

ALTER TABLE sync_cursors ALTER COLUMN evidence_created_count DROP DEFAULT;

UPDATE alembic_version SET version_num='0005_durable_sync_worker' WHERE alembic_version.version_num = '0004_generic_ingestion_contract';

-- Running upgrade 0005_durable_sync_worker -> 0006_scheduled_sync_runtime

CREATE TABLE sync_schedules (
    id UUID NOT NULL, 
    datasource_id UUID NOT NULL, 
    provider VARCHAR(32) NOT NULL, 
    status VARCHAR(32) NOT NULL, 
    interval_minutes INTEGER NOT NULL, 
    next_run_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    last_enqueued_at TIMESTAMP WITH TIME ZONE, 
    last_enqueued_sync_job_id UUID, 
    max_attempts INTEGER DEFAULT '3' NOT NULL, 
    created_by TEXT NOT NULL, 
    created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    PRIMARY KEY (id), 
    FOREIGN KEY(datasource_id) REFERENCES data_sources (id) ON DELETE CASCADE, 
    FOREIGN KEY(last_enqueued_sync_job_id) REFERENCES sync_jobs (id) ON DELETE SET NULL, 
    CONSTRAINT uq_sync_schedules_datasource UNIQUE (datasource_id)
);

CREATE INDEX ix_sync_schedules_status_next_run ON sync_schedules (status, next_run_at);

CREATE INDEX ix_sync_schedules_datasource ON sync_schedules (datasource_id);

CREATE INDEX ix_sync_schedules_provider ON sync_schedules (provider);

ALTER TABLE sync_schedules ALTER COLUMN max_attempts DROP DEFAULT;

UPDATE alembic_version SET version_num='0006_scheduled_sync_runtime' WHERE alembic_version.version_num = '0005_durable_sync_worker';

-- Running upgrade 0006_scheduled_sync_runtime -> 0007_provider_object_ingestion_support

ALTER TABLE provider_object_snapshots ADD COLUMN ingestion_supported BOOLEAN DEFAULT true NOT NULL;

ALTER TABLE provider_object_snapshots ADD COLUMN ingestion_unsupported_reason TEXT;

CREATE INDEX ix_provider_object_snapshots_ingestion ON provider_object_snapshots (datasource_id, ingestion_supported);

ALTER TABLE provider_object_snapshots ALTER COLUMN ingestion_supported DROP DEFAULT;

UPDATE alembic_version SET version_num='0007_provider_object_ingestion_support' WHERE alembic_version.version_num = '0006_scheduled_sync_runtime';

-- Running upgrade 0007_provider_object_ingestion_support -> 0008_worker_lease_primitives

ALTER TABLE sync_jobs ADD COLUMN lease_owner TEXT;

ALTER TABLE sync_jobs ADD COLUMN lease_acquired_at TIMESTAMP WITH TIME ZONE;

ALTER TABLE sync_jobs ADD COLUMN lease_expires_at TIMESTAMP WITH TIME ZONE;

ALTER TABLE sync_jobs ADD COLUMN schedule_id UUID;

ALTER TABLE sync_jobs ADD COLUMN enqueue_key TEXT;

ALTER TABLE sync_jobs ADD CONSTRAINT fk_sync_jobs_schedule_id_sync_schedules FOREIGN KEY(schedule_id) REFERENCES sync_schedules (id) ON DELETE SET NULL;

CREATE INDEX ix_sync_jobs_lease_expiry ON sync_jobs (status, lease_expires_at);

CREATE INDEX ix_sync_jobs_schedule ON sync_jobs (schedule_id);

CREATE UNIQUE INDEX ix_sync_jobs_enqueue_key ON sync_jobs (enqueue_key);

UPDATE alembic_version SET version_num='0008_worker_lease_primitives' WHERE alembic_version.version_num = '0007_provider_object_ingestion_support';

-- Running upgrade 0008_worker_lease_primitives -> 0009_live_postgres_worker_concurrency

ALTER TABLE sync_jobs ADD COLUMN lease_heartbeat_at TIMESTAMP WITH TIME ZONE;

CREATE INDEX ix_sync_jobs_claimable ON sync_jobs (status, next_attempt_at, lease_expires_at);

UPDATE alembic_version SET version_num='0009_live_postgres_worker_concurrency' WHERE alembic_version.version_num = '0008_worker_lease_primitives';

-- Running upgrade 0009_live_postgres_worker_concurrency -> 0010_officialization_workbench

CREATE TABLE concept_relations (
    id UUID NOT NULL, 
    source_concept_id UUID NOT NULL, 
    target_concept_id UUID NOT NULL, 
    relation_type VARCHAR(64) NOT NULL, 
    status VARCHAR(32) NOT NULL, 
    decision_record_id UUID, 
    created_by TEXT NOT NULL, 
    officialized_by TEXT, 
    created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    last_reviewed_at TIMESTAMP WITH TIME ZONE, 
    PRIMARY KEY (id), 
    CONSTRAINT ck_concept_relations_distinct_concepts CHECK (source_concept_id <> target_concept_id), 
    FOREIGN KEY(source_concept_id) REFERENCES concepts (id) ON DELETE RESTRICT, 
    FOREIGN KEY(target_concept_id) REFERENCES concepts (id) ON DELETE RESTRICT, 
    FOREIGN KEY(decision_record_id) REFERENCES decision_records (id) ON DELETE RESTRICT
);

CREATE INDEX ix_concept_relations_source ON concept_relations (source_concept_id);

CREATE INDEX ix_concept_relations_target ON concept_relations (target_concept_id);

CREATE INDEX ix_concept_relations_status ON concept_relations (status);

CREATE INDEX ix_concept_relations_type ON concept_relations (relation_type);

CREATE TABLE concept_relation_evidence_fragments (
    concept_relation_id UUID NOT NULL, 
    evidence_fragment_id UUID NOT NULL, 
    PRIMARY KEY (concept_relation_id, evidence_fragment_id), 
    FOREIGN KEY(concept_relation_id) REFERENCES concept_relations (id) ON DELETE CASCADE, 
    FOREIGN KEY(evidence_fragment_id) REFERENCES evidence_fragments (id) ON DELETE RESTRICT
);

CREATE INDEX ix_concept_relation_evidence_fragments_evidence ON concept_relation_evidence_fragments (evidence_fragment_id);

UPDATE alembic_version SET version_num='0010_officialization_workbench' WHERE alembic_version.version_num = '0009_live_postgres_worker_concurrency';

-- Running upgrade 0010_officialization_workbench -> 0011_grounded_context_evaluation

CREATE TABLE grounded_context_eval_tasks (
    id UUID NOT NULL, 
    name TEXT NOT NULL, 
    query TEXT NOT NULL, 
    expected_answer_contains JSONB DEFAULT '[]' NOT NULL, 
    expected_trust_label VARCHAR(64), 
    expected_freshness_state VARCHAR(32), 
    required_evidence_fragment_ids JSONB DEFAULT '[]' NOT NULL, 
    required_concept_ids JSONB DEFAULT '[]' NOT NULL, 
    required_decision_record_ids JSONB DEFAULT '[]' NOT NULL, 
    require_official_answer BOOLEAN DEFAULT false NOT NULL, 
    require_evidence BOOLEAN DEFAULT true NOT NULL, 
    min_evidence_count INTEGER DEFAULT '1' NOT NULL, 
    expected_clarification_reduced BOOLEAN, 
    tags JSONB DEFAULT '[]' NOT NULL, 
    created_by TEXT NOT NULL, 
    created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    metadata JSONB DEFAULT '{}' NOT NULL, 
    PRIMARY KEY (id)
);

CREATE INDEX ix_grounded_context_eval_tasks_created_at ON grounded_context_eval_tasks (created_at);

CREATE INDEX ix_grounded_context_eval_tasks_expected_trust ON grounded_context_eval_tasks (expected_trust_label);

CREATE INDEX ix_grounded_context_eval_tasks_query ON grounded_context_eval_tasks (query);

CREATE TABLE grounded_context_eval_results (
    id UUID NOT NULL, 
    task_id UUID NOT NULL, 
    response_id UUID NOT NULL, 
    query TEXT NOT NULL, 
    answer TEXT NOT NULL, 
    trust_label VARCHAR(64) NOT NULL, 
    response JSONB NOT NULL, 
    answer_correct BOOLEAN NOT NULL, 
    evidence_valid BOOLEAN NOT NULL, 
    provenance_present BOOLEAN NOT NULL, 
    trust_label_correct BOOLEAN NOT NULL, 
    freshness_policy_respected BOOLEAN NOT NULL, 
    unsupported_official_claim BOOLEAN NOT NULL, 
    citation_validity_rate FLOAT NOT NULL, 
    clarification_reduced BOOLEAN, 
    success BOOLEAN NOT NULL, 
    failure_reasons JSONB DEFAULT '[]' NOT NULL, 
    evaluated_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    evaluated_by TEXT NOT NULL, 
    PRIMARY KEY (id), 
    FOREIGN KEY(task_id) REFERENCES grounded_context_eval_tasks (id) ON DELETE CASCADE
);

CREATE INDEX ix_grounded_context_eval_results_task ON grounded_context_eval_results (task_id);

CREATE INDEX ix_grounded_context_eval_results_success ON grounded_context_eval_results (success);

CREATE INDEX ix_grounded_context_eval_results_evaluated_at ON grounded_context_eval_results (evaluated_at);

CREATE INDEX ix_grounded_context_eval_results_trust ON grounded_context_eval_results (trust_label);

UPDATE alembic_version SET version_num='0011_grounded_context_evaluation' WHERE alembic_version.version_num = '0010_officialization_workbench';

-- Running upgrade 0011_grounded_context_evaluation -> 0012_concept_aliases_ontology_graph

CREATE TABLE concept_aliases (
    id UUID NOT NULL, 
    concept_id UUID NOT NULL, 
    alias CITEXT NOT NULL, 
    normalized_alias CITEXT NOT NULL, 
    created_by TEXT NOT NULL, 
    created_at TIMESTAMP WITH TIME ZONE NOT NULL, 
    PRIMARY KEY (id), 
    CONSTRAINT uq_concept_aliases_normalized_alias UNIQUE (normalized_alias), 
    CONSTRAINT uq_concept_aliases_concept_normalized_alias UNIQUE (concept_id, normalized_alias), 
    FOREIGN KEY(concept_id) REFERENCES concepts (id) ON DELETE CASCADE
);

CREATE INDEX ix_concept_aliases_concept_id ON concept_aliases (concept_id);

CREATE INDEX ix_concept_aliases_normalized_alias ON concept_aliases (normalized_alias);

UPDATE alembic_version SET version_num='0012_concept_aliases_ontology_graph' WHERE alembic_version.version_num = '0011_grounded_context_evaluation';

COMMIT;

