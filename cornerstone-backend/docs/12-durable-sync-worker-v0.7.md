# Backend v0.7.0 — Durable Sync Worker and Checkpoint Runtime

v0.7.0 moves connector sync from request-time execution toward a durable worker model.

The goal is to protect the PRD trust contract: source sync failure must not silently mark data as fresh, previous data must be shown with degraded/stale state when appropriate, and generated Artifacts/EvidenceFragments must remain provenance-backed.

## What changed

### Queued by default

`POST /v1/sources/{source_id}/sync-jobs` now creates a queued job by default.

Inline execution remains available only when explicitly requested with:

```json
{
  "runInline": true
}
```

This preserves deterministic tests and local demos while making queue-first behavior the product default.

### Worker endpoints

```http
POST /v1/sync-jobs/{sync_job_id}/run
POST /v1/sync-worker/run
```

The single-job endpoint runs one job.

The worker endpoint drains queued jobs up to `maxJobs` and can optionally force retry-waiting jobs with `includeNotReady`.

### Retry waiting

Retryable connector errors move a job to:

```text
retry_waiting
```

The job records:

```text
attemptCount
maxAttempts
nextAttemptAt
lastError
```

The DataSource moves to:

```text
syncStatus=waiting_retry
nextAction=retry_sync
freshnessState=unknown
```

This prevents rate-limit or provider downtime from pretending that data is fresh.

### Cancellation

`POST /v1/sync-jobs/{sync_job_id}/cancel` records cancellation intent.

Queued and retry-waiting jobs become cancelled immediately.

Running jobs record `cancelRequestedAt` and the worker checks cancellation before provider listing, before ingestion, and before cursor advancement.

Cancelled jobs do not create Artifacts/EvidenceFragments and do not advance cursors.

### Sync cursor

A durable `SyncCursor` tracks successful processing only after a job succeeds.

```text
datasourceId
provider
cursorKey
lastCursor
lastSuccessfulSyncJobId
processedExternalObjectIds
artifactCreatedCount
artifactReusedCount
evidenceCreatedCount
advancedAt
```

Cursor advancement is intentionally success-only. A failed, cancelled, or retry-waiting job must not update the cursor.

## New / updated endpoints

```http
POST /v1/sync-jobs/{sync_job_id}/run
POST /v1/sync-worker/run
GET  /v1/sources/{source_id}/sync-cursor
POST /v1/sync-jobs/{sync_job_id}/cancel
```

## Database changes

Alembic migration:

```text
0005_durable_sync_worker
```

Adds:

```text
sync_jobs.attempt_count
sync_jobs.max_attempts
sync_jobs.next_attempt_at
sync_jobs.cancel_requested_at
sync_jobs.cancelled_by
sync_cursors
```

The cursor table is unique by:

```text
datasource_id + cursor_key
```

## Worker state transitions

```text
queued
→ running
→ succeeded
```

Retryable failure before max attempts:

```text
running
→ retry_waiting
→ running
```

Final failure:

```text
running
→ failed
```

Cancellation:

```text
queued/retry_waiting/running
→ cancelled
```

## Trust invariants

v0.7.0 explicitly protects these invariants:

```text
1. Sync failure does not mark source data fresh.
2. Retry-waiting jobs do not advance sync cursors.
3. Cancelled jobs do not ingest content.
4. Cursor advancement happens only after successful ingestion.
5. Prior successful data remains available but source state becomes degraded/failed when later sync fails.
6. Provider rate limits become actionable retry-waiting state rather than generic failure.
```

## Tests added

```text
test_sync_job_is_queued_by_default_and_worker_advances_cursor
test_cancelled_queued_sync_job_does_not_ingest_content
test_rate_limited_sync_job_waits_for_retry_and_does_not_advance_cursor
test_retry_waiting_job_is_skipped_until_due
```

Existing connector tests were updated to account for the new `sync.cursor_advanced` event.
