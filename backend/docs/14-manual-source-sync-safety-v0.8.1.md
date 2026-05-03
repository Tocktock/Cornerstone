# v0.8.1 — Manual Source Sync Safety Patch

## Goal

v0.8.1 removes the legacy generic source sync route and replaces it with an explicit
manual-source-only sync route.

This matters because provider-backed sources such as Notion must not be able to ingest
arbitrary caller-supplied objects. Provider sources must pass through the connector flow:

```text
credential → connection test → discovery → selection → sync job → normalized SourceObject
```

Manual sources remain useful, but their ingestion path is now explicit:

```http
POST /v1/manual-sources/{source_id}/sync
```

## API change

Removed:

```http
POST /v1/sources/{source_id}/sync
```

Added:

```http
POST /v1/manual-sources/{source_id}/sync
```

## Behavior

```text
Manual source:
POST /v1/manual-sources/{source_id}/sync → accepted when source state is syncable

Provider source:
POST /v1/manual-sources/{source_id}/sync → 409 Conflict

Removed legacy route:
POST /v1/sources/{source_id}/sync → 404 Not Found
```

## Why compatibility was intentionally not preserved

Cornerstone is a new project. Keeping a generic legacy sync endpoint would create a trust bypass
and make the connector model harder to reason about. The safer behavior is to remove the old route
instead of keeping compatibility.

## Quality gate protected

This patch protects the PRD rule:

```text
The system only knows what it can ground in real sources.
```

Provider-backed sources can no longer skip connector credential, discovery, selection, and sync-job
state checks through a generic source sync endpoint.

## Tests added / updated

```text
test_legacy_source_sync_route_is_removed
test_oauth_completion_does_not_allow_manual_sync_for_notion
manual-source sync tests now use /v1/manual-sources/{source_id}/sync
```

## Remaining follow-up before v0.9.0

```text
1. ✅ Done in v0.8.2: reject unsupported Notion object selection before ingestion.
2. Add production config fail-closed checks.
3. Add live PostgreSQL worker/concurrency verification.
4. Add multi-worker lease/lock semantics.
```
