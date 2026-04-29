# 19 — Live PostgreSQL + Multi-worker Safety v0.9.0

## Goal

v0.9.0 hardens the durable sync runtime so worker execution is safer under concurrent workers and ready for live PostgreSQL verification.

The product invariant is unchanged:

```text
A sync job may create trusted Artifact/Evidence state only when job ownership, source state, cursor advancement, and audit events remain consistent.
```

## Implemented changes

### Worker lease heartbeat

`SyncJob` now includes:

```text
leaseOwner
leaseAcquiredAt
leaseExpiresAt
leaseHeartbeatAt
```

Workers can refresh an active lease through:

```http
POST /v1/sync-jobs/{sync_job_id}/heartbeat
```

Request:

```json
{
  "workerId": "worker-a",
  "leaseSeconds": 300
}
```

Only the current lease owner can heartbeat the job.

### Expired lease recovery

A job in `running` state is claimable only when:

```text
leaseExpiresAt <= now
```

This allows recovery from crashed workers without making active jobs duplicate-processable.

### Active lease protection

A job in `running` state with a non-expired lease is not runnable by another worker.

### PostgreSQL row-lock claim path

The SQLAlchemy store claims jobs with:

```text
SELECT ... FOR UPDATE SKIP LOCKED
```

That prepares live PostgreSQL concurrency safety by ensuring two workers cannot claim the same row at the same time.

### Scheduler idempotency

Scheduled jobs keep a stable enqueue key:

```text
sync-schedule:{scheduleId}:{nextRunAt}
```

Duplicate enqueue key writes are treated as skipped scheduler work rather than a crash path.

## New migration

```text
0009_live_postgres_worker_concurrency
```

Adds:

```text
sync_jobs.lease_heartbeat_at
ix_sync_jobs_claimable(status, next_attempt_at, lease_expires_at)
```

## New tests

Local deterministic tests:

```text
test_expired_running_job_lease_can_be_reclaimed_by_second_worker
test_sync_job_lease_heartbeat_extends_active_worker_lease
test_sync_job_schema_tracks_lease_heartbeat
test_sync_job_table_has_live_postgres_claim_columns_and_indexes
test_worker_concurrency_migration_adds_heartbeat_and_claimable_index
test_in_memory_store_reclaims_expired_running_job_lease
test_in_memory_store_rejects_active_running_job_claim_by_second_worker
test_in_memory_store_lease_heartbeat_requires_owner
```

Live PostgreSQL tests:

```text
tests/postgres/test_live_postgres_worker_concurrency.py
```

These are gated by:

```bash
RUN_POSTGRES_TESTS=1
DATABASE_URL=postgresql+psycopg://cornerstone:cornerstone@localhost:5432/cornerstone
```

## Verification commands

Local non-PostgreSQL verification:

```bash
./scripts/run_tests.sh
```

Live PostgreSQL verification:

```bash
docker compose up -d postgres
RUN_POSTGRES_TESTS=1 ./scripts/run_postgres_ci.sh
```

## Remaining risks

```text
1. Production KMS/secret-manager integration is still deferred.
2. Notion database/data_source ingestion remains deferred.
3. Deployment-specific queue orchestration is not implemented.
4. Live PostgreSQL tests require a PostgreSQL server and are skipped in local non-PostgreSQL runs.
```
