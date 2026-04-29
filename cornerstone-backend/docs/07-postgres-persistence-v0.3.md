# 07 — PostgreSQL Persistence v0.3

## Goal

Move Cornerstone from an in-memory trust-contract prototype to a durable backend persistence foundation while preserving the PRD's trust gates: evidence, provenance, freshness, review state, officialization gating, and auditability.

## Persistence decision

PostgreSQL is the production persistence database. The in-memory repository remains only for fast backend contract tests.

Required extensions:

| Extension | Purpose |
| --- | --- |
| `pgcrypto` | Database-side UUID generation through `gen_random_uuid()` in migrations. |
| `citext` | Case-insensitive natural names such as Concept names. |
| `vector` | Future evidence embedding storage and semantic retrieval. |

The first migration creates these extensions with `CREATE EXTENSION IF NOT EXISTS` and the app can verify that they are installed before startup when `VERIFY_POSTGRES_EXTENSIONS_ON_STARTUP=true`.

## Architecture changes

```text
FastAPI routes/services
  ↓
Repository contract methods
  ↓
SqlAlchemyStore
  ↓
SQLAlchemy 2.x Engine/Session
  ↓
PostgreSQL + pgcrypto + citext + vector
```

The repository interface intentionally mirrors the previous `InMemoryStore` methods so backend trust logic does not change when switching persistence backends.

## Tables

Core tables:

```text
data_sources
artifacts
evidence_fragments
decision_records
concepts
audit_events
```

Join tables:

```text
concept_evidence_fragments
concept_decision_records
decision_record_evidence_fragments
decision_record_affected_concepts
```

Prepared future retrieval table:

```text
evidence_embeddings
```

The `evidence_embeddings` table uses `VECTOR(1536)` and an HNSW cosine index. No API writes to this table yet; it is included so persistence is ready for the later grounded retrieval slice without changing the database extension baseline.

## Trust-related constraints

| Rule | Persistence mechanism |
| --- | --- |
| Artifacts are idempotent for unchanged source content | Unique constraint on `datasource_id`, `source_external_id`, `raw_content_hash`. |
| Evidence must reference a real Artifact | Foreign key from `evidence_fragments.artifact_id` to `artifacts.id`. |
| Concepts reference supporting EvidenceFragments | Join table with foreign keys. |
| Concepts reference DecisionRecords | Join table with foreign keys. |
| DecisionRecords reference reviewed evidence | Join table with foreign keys; review validation remains in service layer. |
| Audit events are durable | `audit_events` table with entity and event indexes. |
| Provenance remains queryable | JSONB `provenance` column with GIN index in migration. |

## Transaction behavior

`SqlAlchemyStore.transaction()` uses a request-local SQLAlchemy `Session`. During source sync, all Artifact and EvidenceFragment writes are inside one transaction. If extraction fails, the partial Artifact/Evidence writes roll back and the source is updated separately to `failed` or `degraded` with `lastError`.

Officialization continues to use service-level validation and then persists Concept status plus audit event. The next hardening step should wrap Concept update and audit write inside one explicit transaction in the route/service boundary.

## Local PostgreSQL

The included `docker-compose.yml` uses:

```text
pgvector/pgvector:pg17
```

Run:

```bash
docker compose up -d postgres
export PERSISTENCE_BACKEND=postgres
export DATABASE_URL=postgresql+psycopg://cornerstone:cornerstone@localhost:5432/cornerstone
alembic upgrade head
python scripts/check_postgres_extensions.py
uvicorn cornerstone.main:app --reload
```

## Testing added in v0.3

New tests cover:

```text
- Required extension list and migration contents.
- Migration creates pgcrypto, citext, vector, and evidence_embeddings.
- SQLAlchemy metadata contains normalized core and join tables.
- PostgreSQL DDL compiles Concept names as CITEXT.
- EvidenceFragment provenance compiles as JSONB on PostgreSQL.
- SQLAlchemy-backed API persists source/artifact/evidence/concept data across app/store instances.
- SQLAlchemy-backed sync idempotency reuses Artifacts for the same source identity/hash.
- SQLAlchemy-backed transaction rollback prevents partial sync writes.
```

## Current limitation

The sandbox test run validates SQLAlchemy repository behavior with SQLite and validates PostgreSQL-specific DDL/migration output offline. It does not spin up a live PostgreSQL server in the sandbox. The package includes Docker Compose and extension checks so a live PostgreSQL integration test can be run in a developer or CI environment with Docker/Postgres available.

## Next backend slice

```text
1. Add CI job with live PostgreSQL + pgvector service.
2. Run alembic upgrade head against live PostgreSQL.
3. Run the same API integration tests against PostgreSQL, not SQLite.
4. Add transactional officialization wrapper for Concept update + AuditEvent write.
5. Add ConceptRelation persistence and officialization gate.
6. Add Notion connector token persistence and encrypted secret boundary.
```
