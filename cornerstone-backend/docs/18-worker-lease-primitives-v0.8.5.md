# v0.8.5 — Worker Lease and Enqueue Idempotency Primitives

v0.8.5 prepares the backend for v0.9.0 multi-worker safety without claiming that distributed execution is fully production-proven yet.

The goal is to stop the worker runtime from using only "list queued jobs, then update one" semantics. That pattern is unsafe once multiple worker processes can observe the same queue. v0.8.5 adds explicit lease/claim fields, claim methods, and scheduled enqueue idempotency keys so the next release can verify the same behavior against live PostgreSQL concurrency.

## Why this matters

A sync job is part of the source trust path. If two workers process the same job, Cornerstone can duplicate provider calls, duplicate evidence, or create confusing audit trails. If two schedulers enqueue the same due run, Source Studio can show duplicated jobs for one schedule window.

The product principle is unchanged:

```text
No fake freshness. No ambiguous sync state. No duplicated source trust work.
```

## New SyncJob fields

`SyncJob` now includes worker lease and scheduled enqueue fields:

```text
leaseOwner
leaseAcquiredAt
leaseExpiresAt
scheduleId
enqueueKey
```

These are persisted in PostgreSQL through Alembic migration `0008_worker_lease_primitives`.

## Claim behavior

Workers now claim jobs through the store boundary before doing provider work.

Claimable statuses:

```text
queued
retry_waiting, if nextAttemptAt is due or includeNotReady=true
```

A successful claim atomically changes the job to:

```text
status=running
attemptCount += 1
startedAt=now
leaseOwner=<worker id>
leaseAcquiredAt=now
leaseExpiresAt=now + leaseSeconds
nextAttemptAt=null
```

Then the worker records:

```text
sync.job_claimed
sync.job_started
```

Terminal states clear the lease fields:

```text
succeeded
failed
retry_waiting
cancelled
```

## API / CLI changes

`POST /v1/sync-worker/run` accepts:

```json
{
  "maxJobs": 10,
  "workerId": "worker-a",
  "leaseSeconds": 300
}
```

`POST /v1/sync-jobs/{syncJobId}/run` accepts query parameters:

```text
workerId=api-job-runner
leaseSeconds=300
```

The external worker CLI accepts:

```bash
python scripts/run_sync_worker.py --once --worker-id worker-a --lease-seconds 300
```

## Scheduled enqueue idempotency

Scheduled jobs now carry:

```text
scheduleId
enqueueKey
```

The enqueue key is stable for one schedule due window:

```text
sync-schedule:{scheduleId}:{nextRunAt}
```

The database migration adds a unique index on `sync_jobs.enqueue_key`. This gives v0.9.0 a concrete constraint to validate with live PostgreSQL concurrent scheduler tests.

## New tests

```text
test_sync_job_table_has_worker_lease_and_enqueue_primitives
test_in_memory_store_claims_sync_job_once_with_worker_lease
test_schedule_enqueue_key_is_stable_and_duplicate_jobs_are_rejected
test_worker_claim_event_records_worker_identity_and_clears_lease_on_success
test_already_claimed_job_is_skipped_by_second_worker
```

## What this patch does not claim

v0.8.5 is not yet full distributed worker safety.

Still pending for v0.9.0:

```text
- Live PostgreSQL concurrent claim tests.
- `SELECT ... FOR UPDATE SKIP LOCKED` verification under multiple workers.
- Concurrent scheduled enqueue race tests.
- Expired lease recovery/heartbeat semantics.
- Production queue deployment contract.
```

## Exit criteria

```text
The backend has explicit worker claim/lease fields, claim behavior, claim observability, and scheduled enqueue idempotency primitives ready for live PostgreSQL concurrency verification.
```
