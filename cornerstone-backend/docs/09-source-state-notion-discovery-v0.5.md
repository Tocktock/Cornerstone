# v0.5.0 — Source Runtime State + Notion Discovery/Selection

## Purpose

This slice completes the connector Phase A/B backend foundation:

1. Split source runtime state into UI-friendly dimensions.
2. Add provider object discovery snapshots.
3. Add Notion discovery and selection before ingestion.
4. Keep sync jobs honest: discovery metadata does not fabricate Artifacts or EvidenceFragments.

This keeps Source Studio aligned with the product principle: no fake confidence.

## Refined source state model

`DataSource.status` remains as a backward-compatible aggregate field, but Source Studio should prefer the split state fields.

```ts
type DataSourceRuntimeState = {
  authStatus:
    | "not_started"
    | "intent_created"
    | "oauth_redirected"
    | "authorized"
    | "auth_failed"
    | "revoked";

  connectionStatus:
    | "untested"
    | "test_passed"
    | "test_failed"
    | "permission_limited";

  syncStatus:
    | "never_synced"
    | "queued"
    | "syncing"
    | "succeeded"
    | "failed"
    | "degraded"
    | "cancelled";

  freshnessState:
    | "fresh"
    | "aging"
    | "stale"
    | "unknown";

  nextAction:
    | "connect"
    | "complete_oauth"
    | "test_connection"
    | "discover_sources"
    | "select_sources"
    | "grant_permission"
    | "run_first_sync"
    | "review_evidence"
    | "reconnect"
    | "retry_sync"
    | "none";
};
```

## State transitions

### OAuth success

```text
authStatus: authorized
connectionStatus: untested
syncStatus: never_synced
freshnessState: unknown
nextAction: test_connection
```

OAuth success is not content freshness and not evidence trust.

### Connection test success

```text
connectionStatus: test_passed
nextAction: discover_sources
```

### Discovery success with accessible objects

```text
lastDiscoveryAt: now
discoveredObjectCount: N
connectionStatus: test_passed
nextAction: select_sources
```

### Discovery success with no accessible objects

```text
connectionStatus: permission_limited
nextAction: grant_permission
```

### Selection saved

```text
selectedObjectCount: N
nextAction: run_first_sync
```

### Sync job success with no content ingestion

```text
syncStatus: succeeded
freshnessState: unknown
nextAction: none
```

v0.5.0 intentionally does not create Artifacts from discovery metadata alone.

## ProviderObjectSnapshot

Discovery produces snapshots that are safe to show in Source Studio before ingestion.

```ts
type ProviderObjectSnapshot = {
  id: string;
  datasourceId: string;
  provider: "notion";
  externalId: string;
  externalUrl?: string;
  objectType: "page" | "database" | "data_source" | "block" | "unknown";
  title?: string;
  parentExternalId?: string;
  lastEditedTime?: string;
  discoveredAt: string;
  selectedForSync: boolean;
  accessState: "accessible" | "inaccessible" | "deleted" | "unknown";
  rawMetadataHash: string;
};
```

## New API endpoints

```http
POST /v1/sources/{source_id}/discover
GET  /v1/sources/{source_id}/objects
PUT  /v1/sources/{source_id}/selections
GET  /v1/sources/{source_id}/selections
```

## Notion discovery behavior

The Notion connector now supports discovery through the provider adapter interface.

Mock mode returns two accessible fixture objects:

```text
notion-page-1
notion-database-1
```

Live mode calls Notion search and maps results into `ProviderObjectSnapshot` rows:

```text
page      → page
database  → database
data_source → data_source
block     → block
unknown   → unknown
```

The connector preserves:

```text
external ID
external URL
object type
title
parent external ID
last edited time
access/deleted state
metadata hash
```

## Selection behavior

Selections are validated against discovered snapshots. Since v0.8.2, selection also requires `ingestionSupported=true`; Notion databases/data_sources remain discoverable but cannot be selected for sync yet.

The backend blocks:

```text
selection before discovery
empty selected_only selection
unknown object IDs
inaccessible object IDs
```

`all_accessible` expands only to currently accessible discovered objects with `ingestionSupported=true`.

## Persistence

Migration `0003_source_state_discovery.py` adds:

```text
data_sources.auth_status
data_sources.connection_status
data_sources.sync_status
data_sources.next_action
data_sources.last_connection_test_at
data_sources.last_discovery_at
data_sources.discovered_object_count
data_sources.selected_object_count
provider_object_snapshots
```

Indexes:

```text
ix_data_sources_runtime_state
ix_provider_object_snapshots_datasource
ix_provider_object_snapshots_access
ix_provider_object_snapshots_selected
ix_provider_object_snapshots_type
```

## Test coverage

v0.5.0 adds tests for:

```text
OAuth success runtime state
connection test next action
discovery requires connection test
discovery persists snapshots
provider object listing
selection requires discovery
selection rejects unknown/inaccessible IDs
selection marks snapshots selected
all_accessible expansion
sync jobs require selection
sync job does not fake content artifacts
disconnect revokes source runtime state
SQLAlchemy persistence across store instances
live Notion token exchange request shape
live Notion connection test headers
live Notion discovery mapping
Notion rate-limit error mapping
```

## Remaining scope

Not implemented in v0.5.0:

```text
live Notion page/block content ingestion
background worker queue
retry scheduler
provider webhook handling
KMS-backed token encryption
Artifact/Evidence extraction from Notion blocks
```

## Recommended next slice

```text
v0.6.0 — Live Notion Page/Block Ingestion
```

Minimum scope:

```text
1. Fetch selected Notion pages/databases/data sources.
2. Fetch child blocks with pagination.
3. Normalize blocks into SourceObject.contentText.
4. Preserve block/page URLs, last edited time, and metadata hashes.
5. Create Artifacts from selected content only.
6. Extract EvidenceFragments with provenance.
7. Add rate-limit retry/backoff in sync job execution.
8. Keep sync transactional and idempotent.
```
