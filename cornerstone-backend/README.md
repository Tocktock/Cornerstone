# Cornerstone Backend

## v1.0.0 Backend MVP release

This package is the full backend MVP release artifact for Cornerstone. It promotes the verified `v1.0.0-rc.1` / `v0.13.1` release-candidate line to final backend `v1.0.0` with release metadata updated so runtime/package version markers report `1.0.0`.

The release is backend-only and preserves the live-proven MVP loop:

```text
Live PostgreSQL
→ live Notion page
→ Artifact
→ EvidenceFragment
→ evidence review
→ official Concept
→ grounded context response
→ evaluation result
→ grounded_context_task_success_rate
```

Final release documents:

- `docs/29-backend-v1.0.0.md`
- `docs/release/v1.0.0-release-notes.md`
- `docs/release/v1.0.0-readiness.md`
- `docs/release/backend-operator-runbook.md`
- `docs/release/backend-release-checklist.md`
- `docs/release/known-limitations.md`
- `docs/release/production-deployment-checklist.md`
- `docs/release/secrets-and-credential-handling.md`

The release proof remains recorded in:

```text
docs/live-proof-records/2026-04-28-v0.13.1-blocker-fix.md
```

Backend-only FastAPI implementation for Cornerstone's evidence-backed organizational context layer.

This package focuses on the PRD's backend trust foundation:

```text
Real or manual source
→ Artifact snapshot
→ EvidenceFragment with required provenance
→ Evidence review
→ Concept candidate or DecisionRecord
→ Officialization gate
→ Grounded context response with trust/freshness/limitations
→ Durable PostgreSQL persistence
→ Connector intent / credential / sync-job framework
```

## Status: v1.0.0 backend MVP release

Implemented in this backend slice:

- FastAPI app factory and `/v1` API routes.
- PostgreSQL persistence mode behind the same repository contract as the in-memory test store.
- SQLAlchemy-backed `SqlAlchemyStore` with transaction context support.
- Alembic migration `0001_postgres_persistence` plus `0002_connector_framework`.
- Required PostgreSQL extensions in migration and runtime verification:
  - `pgcrypto` for DB-side UUID generation.
  - `citext` for case-insensitive Concept/source natural names.
  - `vector` for future evidence embedding / semantic retrieval storage.
- `docker-compose.yml` using the `pgvector/pgvector:pg17` image for local Postgres with `vector` support.
- Normalized persistence tables for Concepts, DecisionRecords, and supporting EvidenceFragments.
- Artifact idempotency constraint on `dataSourceId + sourceExternalId + rawContentHash`.
- JSONB persistence for EvidenceFragment provenance and audit metadata.
- Transactional sync rollback in the SQLAlchemy repository.
- Persistence tests that verify SQLAlchemy-backed API durability across app/store instances.
- Offline Alembic SQL rendering as part of the test/report script.
- Connector catalog, Notion adapter, and Manual smoke-test adapter.
- Stateful connection intents for OAuth redirect/callback handling.
- Encrypted connector credential persistence with public-safe API responses.
- Source connection test endpoint.
- Source selection contract before first connector sync.
- Provider object selection safety: only accessible and ingestion-supported objects can be selected.
- Production-mode fail-closed checks for persistence, provider mock mode, secrets, reviewers, OAuth callback URL, database URL, and required PostgreSQL extensions.
- Atomic worker success writes across Artifact/Evidence/source/cursor/job/event state.
- Worker lease/claim primitives for queued and retry-waiting SyncJobs.
- Scheduled SyncJob `scheduleId`/`enqueueKey` idempotency primitives.
- Worker CLI/API support for `workerId` and `leaseSeconds`.
- Provider-normalized ingestion into Artifacts through `SourceObject`.
- Selected Notion page sync into Artifacts and EvidenceFragments.
- Notion markdown retrieval with block-children fallback.
- Durable sync jobs, retry-waiting state, cancellation metadata, sync cursors, and sync job events.
- Per-source sync schedules and scheduler endpoint that enqueues due scheduled jobs.
- External worker CLI entrypoint for process/container execution.
- PostgreSQL CI workflow/script assets for migration, extension, and test verification.
- Disconnect flow that revokes stored credentials.
- Notion mock external API mode for deterministic backend contract tests.
- Worker endpoints for running a single sync job or draining queued jobs.
- Sync cursor endpoint for Source Studio/admin visibility.
- SDK-backed `NotionGateway` boundary for live Notion API calls.
- Evaluation tasks/results and `grounded_context_task_success_rate` summary endpoint.
- Ruff, mypy, compile, pytest, coverage, JUnit, and migration SQL reports.
- Strict live PostgreSQL verification runner that fails if live tests skip when explicitly requested.
- Stable mypy report generation with non-incremental execution and timeout guard.
- Evidence review queue with trust/freshness/source/type filters.
- Evidence-to-Concept candidate creation.
- ConceptRelation API/persistence and officialization quality gates.
- Audit-backed officialization workflow events for Concepts, Relations, Decisions, and Evidence reviews.
- Grounded context evaluation tasks, persisted results, metric summary, and `grounded_context_task_success_rate`.

Still intentionally not implemented:

- Notion database/data_source ingestion semantics. Databases/data_sources are discoverable but not selectable for ingestion yet.
- Distributed queue infrastructure.
- Notion webhooks and full incremental provider cursor integration.
- Full RBAC provider; the backend still uses a configurable reviewer allow-list.
- Full review workbench UI and batch review operations.
- Runtime vector retrieval; `evidence_embeddings` is prepared in the migration but no API writes to it yet.
- LLM-graded evaluation and production clarification-reduction measurement.

## Quickstart — tests

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
./scripts/run_tests.sh
```

Generated reports:

```text
reports/test-report.txt
reports/coverage-summary.txt
reports/junit.xml
reports/ruff-report.txt
reports/mypy-report.txt
reports/compile-report.txt
reports/alembic-offline.sql
reports/alembic-offline.log
reports/live-postgres-report.txt
reports/live-postgres-summary.txt
```

Verified for v1.0.0 packaging in this sandbox:

```text
compileall passed
release-candidate static check passed for v1.0.0
package hygiene passed
compileall passed
full dependency-based pytest/coverage/ruff/mypy checks were not rerun in this sandbox because runtime dependencies are not installed here
v0.13.1 live PostgreSQL and live Notion proof remains recorded in docs/live-proof-records/2026-04-28-v0.13.1-blocker-fix.md
```

Run the full release-candidate gate locally or in CI:

```bash
./scripts/run_tests.sh
python -m mypy src --show-error-codes --no-color-output --no-incremental
python scripts/check_release_candidate.py
python scripts/run_live_postgres_tests.py --min-passed 5
PYTHONPATH=src python scripts/run_live_notion_e2e.py
```

Live PostgreSQL tests are intentionally excluded from the default local run. To run them, set `RUN_POSTGRES_TESTS=1`, provide a PostgreSQL `DATABASE_URL`, and run `python scripts/run_live_postgres_tests.py`.

## Quickstart — PostgreSQL persistence

Start local PostgreSQL with pgvector support:

```bash
docker compose up -d postgres
```

Install dependencies and run migrations:

```bash
pip install -e '.[dev]'
export PERSISTENCE_BACKEND=postgres
export DATABASE_URL=postgresql+psycopg://cornerstone:cornerstone@localhost:5432/cornerstone
alembic upgrade head
python scripts/check_postgres_extensions.py
```

Run the API against PostgreSQL:

```bash
export PERSISTENCE_BACKEND=postgres
uvicorn cornerstone.main:app --reload
```

Open:

```text
http://localhost:8000/docs
```

## Production-mode runtime safety

Local defaults are intentionally non-production. To start the API or worker with production mode enabled, configure safe runtime values first:

```bash
export APP_ENV=production
export PRODUCTION_MODE=true
export PERSISTENCE_BACKEND=postgres
export DATABASE_URL=postgresql+psycopg://svc_cornerstone:strong-password@postgres.internal:5432/cornerstone
export CONNECTOR_ENCRYPTION_SECRET='replace-with-a-long-production-secret'
export CONNECTOR_OAUTH_CALLBACK_URL=https://cornerstone.example.com/v1/oauth/notion/callback
export NOTION_MOCK_EXTERNAL_API=false
export NOTION_CLIENT_ID='<real-notion-client-id>'
export NOTION_CLIENT_SECRET='<real-notion-client-secret>'
export AUTHORIZED_REVIEWERS_RAW='reviewer@company.internal,ops@company.internal'
```

If any production guard fails, `uvicorn cornerstone.main:app` and `scripts/run_sync_worker.py` refuse to start.

## Connector API examples

Create a Notion connection intent:

```bash
curl -X POST http://localhost:8000/v1/connections/intents \
  -H 'Content-Type: application/json' \
  -d '{"provider":"notion","sourceName":"Team Notion","createdBy":"admin@example.com"}'
```

Complete mocked OAuth locally using the returned `stateNonce`:

```bash
curl 'http://localhost:8000/v1/oauth/notion/callback?state={stateNonce}&code=dev-code'
```

Test the connected source:

```bash
curl -X POST http://localhost:8000/v1/sources/{sourceId}/test
```

Select Notion objects before first sync:

```bash
curl -X PUT http://localhost:8000/v1/sources/{sourceId}/selections \
  -H 'Content-Type: application/json' \
  -d '{"syncMode":"selected_only","selectedExternalObjectIds":["notion-page-1"]}'
```

Start a tracked sync job:

```bash
curl -X POST http://localhost:8000/v1/sources/{sourceId}/sync-jobs \
  -H 'Content-Type: application/json' \
  -d '{"createdBy":"admin@example.com","runInline":true}'
```

Run a queued sync job with an explicit worker lease:

```bash
curl -X POST http://localhost:8000/v1/sync-worker/run \
  -H 'Content-Type: application/json' \
  -d '{"maxJobs":1,"workerId":"worker-a","leaseSeconds":300}'
```

## Manual trust-loop API examples

Create a manual source:

```bash
curl -X POST http://localhost:8000/v1/sources \
  -H 'Content-Type: application/json' \
  -d '{"type":"manual","name":"Pilot Domain","productionEnabled":true}'
```

Sync one source object:

```bash
curl -X POST http://localhost:8000/v1/manual-sources/{sourceId}/sync \
  -H 'Content-Type: application/json' \
  -d '{
    "objects": [
      {
        "sourceExternalId": "doc-1",
        "title": "Cornerstone Overview",
        "content": "Cornerstone is the shared organizational context layer. It must preserve provenance.",
        "sourceUrl": "https://example.internal/doc-1"
      }
    ]
  }'
```

Review evidence:

```bash
curl -X POST http://localhost:8000/v1/evidence/{evidenceId}/review \
  -H 'Content-Type: application/json' \
  -d '{"trustState":"reviewed","reviewedBy":"reviewer@example.com"}'
```

Create a DecisionRecord:

```bash
curl -X POST http://localhost:8000/v1/decision-records \
  -H 'Content-Type: application/json' \
  -d '{
    "title":"Define Cornerstone",
    "decision":"Cornerstone is the shared organizational context layer.",
    "reason":"This definition appears in reviewed source evidence.",
    "decidedBy":"reviewer@example.com",
    "evidenceFragmentIds":["<evidence-id>"]
  }'
```

Create a Concept from evidence:

```bash
curl -X POST http://localhost:8000/v1/concepts \
  -H 'Content-Type: application/json' \
  -d '{
    "name":"Cornerstone",
    "shortDefinition":"Shared organizational context layer.",
    "evidenceFragmentIds":["<evidence-id>"],
    "createdBy":"reviewer@example.com"
  }'
```

Officialize the Concept:

```bash
curl -X POST http://localhost:8000/v1/concepts/{conceptId}/officialize \
  -H 'Content-Type: application/json' \
  -d '{"reviewedBy":"reviewer@example.com"}'
```

Query grounded context:

```bash
curl 'http://localhost:8000/v1/context/query?q=What%20is%20Cornerstone%3F'
```

Read audit events:

```bash
curl http://localhost:8000/v1/audit-events
```

## Documentation

- [Backend architecture](docs/00-backend-architecture.md)
- [API contract](docs/01-api-contract.md)
- [Testing strategy](docs/02-testing-strategy.md)
- [Feature test matrix](docs/03-feature-test-matrix.md)
- [Implementation plan](docs/04-implementation-plan.md)
- [Development standards](docs/05-development-standards.md)
- [v0.2 backend hardening notes](docs/06-backend-hardening-v0.2.md)
- [v0.3 PostgreSQL persistence notes](docs/07-postgres-persistence-v0.3.md)
- [v0.4 connector framework + Notion skeleton notes](docs/08-connector-framework-notion-v0.4.md)
- [v0.5 source state + Notion discovery notes](docs/09-source-state-notion-discovery-v0.5.md)
- [v0.6 generic ingestion + Notion page ingestion notes](docs/10-generic-ingestion-notion-v0.6.md)
- [v0.6.1 SDK-backed Notion gateway notes](docs/11-notion-sdk-gateway-v0.6.1.md)
- [v0.7 durable sync worker notes](docs/12-durable-sync-worker-v0.7.md)
- [v0.8 external worker / scheduled sync / PostgreSQL CI notes](docs/13-external-worker-scheduled-postgres-v0.8.md)
- [v0.8.1 manual source sync safety notes](docs/14-manual-source-sync-safety-v0.8.1.md)
- [v0.8.2 provider object ingestion safety notes](docs/15-provider-object-ingestion-safety-v0.8.2.md)
- [v0.8.3 production config fail-closed notes](docs/16-production-config-fail-closed-v0.8.3.md)
- [v0.8.4 atomic sync write boundaries notes](docs/17-atomic-sync-write-boundaries-v0.8.4.md)
- [v0.8.5 worker lease primitives notes](docs/18-worker-lease-primitives-v0.8.5.md)
- [v0.9.0 live PostgreSQL + multi-worker safety notes](docs/19-live-postgres-multi-worker-v0.9.md)
- [v0.9.1 live PostgreSQL verification notes](docs/20-live-postgres-verification-v0.9.1.md)

- [v0.9.2 live Notion E2E notes](docs/21-live-notion-e2e-v0.9.2.md)
- [v0.10.0 evidence review + officialization hardening notes](docs/22-evidence-review-officialization-v0.10.0.md)
