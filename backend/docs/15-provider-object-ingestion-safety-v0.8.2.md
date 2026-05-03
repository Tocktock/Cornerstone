# v0.8.2 — Provider Object Ingestion Safety Patch

## Purpose

v0.8.2 fixes the second pre-v0.9.0 review issue: Notion discovery could expose databases and data_sources, while the ingestion adapter only supports pages. The previous behavior allowed `all_accessible` selection to include unsupported objects, which could make sync appear successful while silently skipping selected content.

Cornerstone must not create fake confidence. A discovered provider object is not automatically safe to ingest.

## Product rule

```text
Discovered ≠ selectable ≠ ingested evidence.
```

A provider object can be:

```text
discovered: visible to Source Studio
accessible: readable by the provider credential
ingestion_supported: supported by the current connector ingestion implementation
selected_for_sync: selected by the Source Admin for connector sync jobs
```

Only objects that are both accessible and ingestion-supported can be selected for sync.

## API/model changes

### ProviderObjectSnapshot

New fields:

```ts
ingestionSupported: boolean;
ingestionUnsupportedReason?: string | null;
```

Example Notion database snapshot:

```json
{
  "externalId": "notion-database-1",
  "objectType": "database",
  "accessState": "accessible",
  "ingestionSupported": false,
  "ingestionUnsupportedReason": "Notion database ingestion is not implemented in this backend slice. Select Notion pages instead."
}
```

### ProviderObjectSnapshotListResponse

New count:

```ts
syncableCount: number;
```

`syncableCount` counts only accessible and ingestion-supported objects.

### ConnectorDefinition

New fields:

```ts
discoverableObjects: string[];
ingestibleObjects: string[];
```

For Notion v0.8.2:

```json
{
  "supportedObjects": ["page"],
  "discoverableObjects": ["page", "database", "data_source"],
  "ingestibleObjects": ["page"]
}
```

## Selection behavior

### Allowed

```text
selected_only + Notion page → accepted
all_accessible → expands only to ingestion-supported Notion pages
```

### Rejected

```text
selected_only + Notion database → 409 unsupported_object_type
selected_only + Notion data_source → 409 unsupported_object_type
workspace_limited → 409 unsupported_object_type until workspace semantics are implemented
unknown object ID → 409 permission_denied
inaccessible object ID → 409 permission_denied
```

## Persistence changes

Alembic migration:

```text
0007_provider_object_ingestion_support
```

Adds:

```text
provider_object_snapshots.ingestion_supported boolean not null
provider_object_snapshots.ingestion_unsupported_reason text null
ix_provider_object_snapshots_ingestion(datasource_id, ingestion_supported)
```

## Why this matters

Without this patch, a Source Admin could select a Notion database, run sync, and get a successful job with no Artifacts or EvidenceFragments from that selected object. That breaks Source Studio trust because the UI implies a selected object was synced when the connector did not ingest it.

v0.8.2 makes unsupported ingestion explicit before the sync job starts.

## Tests added/updated

```text
test_source_selection_rejects_unsupported_notion_database_object
test_workspace_limited_selection_is_rejected_until_implemented
test_all_accessible_selection_expands_only_to_syncable_objects
test_provider_object_ingestion_support_migration_adds_selection_guard_columns
test_notion_mock_discovery_returns_snapshots_for_selection
test_notion_live_discovery_maps_pages_databases_and_data_sources
```

## Verification

```text
140 passed
coverage threshold >=85%
ruff passed
mypy passed
compileall passed
Alembic offline SQL rendered
```

## Deferred

```text
Notion database/data_source ingestion semantics
Live PostgreSQL concurrency verification
Multi-worker lock/lease semantics
Production config fail-closed checks
```
