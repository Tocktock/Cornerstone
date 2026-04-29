# v0.8.4 — Atomic Sync Write Boundaries Patch

v0.8.4 fixes the fourth pre-v0.9.0 safety issue: connector worker success writes were not committed as one service-level unit.

Before this patch, a worker could:

```text
1. Fetch selected provider objects.
2. Write Artifacts and EvidenceFragments.
3. Update source freshness/counters.
4. Then fail while advancing the SyncCursor or marking the SyncJob succeeded.
```

That could leave durable evidence without the matching cursor/job audit trail. The retry path could then see ambiguous state.

## Principle

Provider/network work should happen outside the database transaction.

Product-state writes that make a sync successful should happen inside one transaction:

```text
Artifact writes
EvidenceFragment writes
source freshness/counter update
SyncCursor advancement
SyncJob succeeded update
terminal SyncJobEvent records
```

If any of those writes fail, they all roll back. The worker then records a retry/failure state through the existing failure handler.

## Implementation

The worker now calls:

```python
_commit_sync_success_atomically(...)
```

This function opens `store.transaction()` and performs:

```text
sync_source_objects(..., emit_logs=False)
advance SyncCursor
update SyncJob to succeeded
update DataSource next action/error state
write sync.cursor_advanced event
write sync.job_succeeded event
```

The generic `sync_source_objects` service now accepts:

```python
emit_logs: bool = True
```

Manual source sync still emits source/artifact logs directly. Worker-driven sync suppresses those lower-level logs until the atomic success path commits, avoiding misleading success logs for rolled-back worker attempts.

## New Tests

```text
test_sync_success_writes_roll_back_when_cursor_advance_fails
test_persistent_sync_success_writes_roll_back_when_cursor_advance_fails
```

These tests intentionally fail cursor advancement after Artifact/Evidence extraction has started. They verify:

```text
- SyncJob moves to retry_waiting.
- SyncCursor is not created.
- Artifacts are rolled back.
- EvidenceFragments are rolled back.
- DataSource artifact/evidence counters remain zero.
- DataSource sync freshness remains unknown.
```

The second test runs through the SQLAlchemy store using SQLite so repository-level transaction semantics are tested beyond the in-memory rollback implementation.

## What This Patch Does Not Solve

This patch does not add multi-worker locking or PostgreSQL row-level claim semantics. That remains v0.9.0 scope.

It also does not replace live PostgreSQL CI. v0.9.0 should still verify the same atomicity and concurrency behavior against a real PostgreSQL database.

## Exit Criteria

```text
A worker success path can no longer persist Artifacts/Evidence without also advancing the cursor and marking the job succeeded.
```
