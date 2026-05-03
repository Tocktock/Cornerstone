# Backend v0.8.0 — External Worker, Scheduled Sync, and PostgreSQL CI Assets

v0.8.0 continues the backend connector hardening work after v0.7.0.

The goal is to make sync operations durable and operationally visible without pretending that a full distributed queue has already been implemented.

## Scope

Implemented in this version:

```text
1. Per-source sync schedule model
2. Sync schedule API endpoints
3. Scheduler runner that enqueues due scheduled jobs
4. External worker CLI entrypoint
5. Docker Compose worker profile
6. PostgreSQL CI workflow and local CI script assets
7. Alembic migration 0006_scheduled_sync_runtime
8. Tests for schedule gating, due/not-due behavior, duplicate prevention, and external worker iteration
```

Not implemented in this version:

```text
1. Distributed queue infrastructure
2. Cron/Kubernetes scheduler deployment
3. Webhook-driven Notion incremental sync
4. Notion database/data_source ingestion semantics
5. Production KMS integration
```

## Why this slice matters

The PRD requires Source Admins to understand connection state, sync state, freshness state, errors, and next actions. It also requires sync failures not to silently mark data as fresh.

v0.7.0 made sync jobs durable. v0.8.0 adds the next operational layer:

```text
Schedule due work
→ enqueue durable SyncJob
→ external worker drains queued jobs
→ cursor advances only after success
→ failed/retry/cancelled jobs never fake freshness
```

## New API surface

```http
GET  /v1/sources/{source_id}/sync-schedule
PUT  /v1/sources/{source_id}/sync-schedule
POST /v1/sync-scheduler/run
```

Existing worker endpoints remain:

```http
POST /v1/sync-worker/run
POST /v1/sync-jobs/{sync_job_id}/run
POST /v1/sync-jobs/{sync_job_id}/cancel
GET  /v1/sources/{source_id}/sync-cursor
```

## SyncSchedule model

```ts
type SyncSchedule = {
  id: string;
  datasourceId: string;
  provider: "notion" | "slack" | "google_docs" | "github" | "manual";
  status: "active" | "paused";
  intervalMinutes: number;
  nextRunAt: string;
  lastEnqueuedAt?: string;
  lastEnqueuedSyncJobId?: string;
  maxAttempts: number;
  createdBy: string;
  createdAt: string;
  updatedAt: string;
};
```

## Scheduling rules

The scheduler only enqueues work when:

```text
1. The schedule is active.
2. nextRunAt is due, unless includeNotDue is explicitly set.
3. The source has a saved selection.
4. The selection contains selected external object IDs.
5. The source does not already have an active queued/running/retry-waiting job.
```

The scheduler does not run the job itself. It creates a durable `SyncJob` with `trigger=scheduled`.

## External worker CLI

Run one iteration:

```bash
python scripts/run_sync_worker.py --once --run-scheduler --max-jobs 10
```

Run a bounded local loop:

```bash
python scripts/run_sync_worker.py --run-scheduler --iterations 5 --sleep-seconds 5 --max-jobs 10
```

The CLI intentionally calls the same service functions as the API endpoints:

```text
run_sync_scheduler()
run_sync_worker()
```

This prevents drift between local tests, admin-triggered runs, and process/container worker execution.

## Docker Compose worker profile

Run PostgreSQL:

```bash
docker compose up -d postgres
```

Run one worker iteration using the profile:

```bash
docker compose --profile worker run --rm sync-worker
```

This is a local operational smoke path, not a production deployment.

## PostgreSQL CI assets

New files:

```text
.github/workflows/backend-ci.yml
scripts/run_postgres_ci.sh
```

The CI path is designed to verify:

```text
1. Python dependency installation
2. Alembic upgrade head against PostgreSQL + pgvector image
3. Required PostgreSQL extensions
4. Backend tests and reports
```

Local PostgreSQL CI path:

```bash
docker compose up -d postgres
export DATABASE_URL=postgresql+psycopg://cornerstone:cornerstone@localhost:5432/cornerstone
export PERSISTENCE_BACKEND=postgres
./scripts/run_postgres_ci.sh
```

## Migration

Migration added:

```text
0006_scheduled_sync_runtime
```

It creates:

```text
sync_schedules
```

Indexes:

```text
ix_sync_schedules_status_next_run
ix_sync_schedules_datasource
ix_sync_schedules_provider
```

Unique constraint:

```text
uq_sync_schedules_datasource
```

## Trust invariants protected

v0.8.0 preserves these invariants:

```text
1. Scheduled sync only enqueues work; it does not create Artifacts by itself.
2. External workers run the same durable worker service as the API.
3. SyncCursor advances only after successful ingestion.
4. Retry-waiting, failed, cancelled, and skipped schedules do not mark source data fresh.
5. A source with an active job does not receive duplicate scheduled jobs.
6. A schedule cannot be enabled without a saved source selection.
```

## Tests added

```text
test_schedule_requires_selection_before_enabling
test_scheduler_enqueues_due_schedule_and_worker_processes_it
test_scheduler_skips_not_due_schedule_until_forced
test_scheduler_does_not_enqueue_duplicate_when_source_has_active_job
test_paused_schedule_is_persisted_but_not_enqueued
test_sync_schedule_table_is_provider_agnostic_and_due_indexed
test_build_schedule_preserves_identity_and_computes_next_run
test_next_schedule_run_at_uses_interval_minutes
test_scheduler_only_advances_schedule_when_job_is_enqueued
test_scheduler_skips_source_without_selection
test_external_worker_iteration_runs_scheduler_then_worker
```

## Remaining work

Next likely backend phase:

```text
v0.9.0 — Live PostgreSQL verification hardening + production worker deployment contract
```

Recommended scope:

```text
1. Testcontainers or equivalent live PostgreSQL integration tests
2. Worker lease/lock semantics for multi-worker safety
3. Idempotent scheduler enqueue under concurrency
4. Production secret/KMS boundary
5. Scheduled sync admin visibility endpoints
6. Provider cursor integration for Notion incremental sync
```
