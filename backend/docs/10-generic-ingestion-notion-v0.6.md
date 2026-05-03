# v0.6.0 — Generic Ingestion Contract + Selected Notion Page Ingestion

## Goal

Complete the next connector phases without letting the backend become Notion-only.

The connector boundary is now:

```text
Provider API payload
→ provider adapter
→ SourceObject
→ Artifact
→ EvidenceFragment
→ review / officialization / grounded serving
```

Core Artifact/Evidence services do not inspect Notion-specific page, database, block, or API payload shapes.

## Implemented

### Generic ingestion contract

`SourceObject` is now the normalized input for Artifact creation:

```text
sourceExternalId
sourceObjectType
title
content
sourceUrl
sourceUpdatedAt
providerMetadata
```

`Artifact` now persists:

```text
sourceObjectType
providerMetadata
```

`ProviderObjectSnapshot` now persists:

```text
providerMetadata
```

This supports non-Notion providers such as Slack threads, Google Docs, GitHub files, and manual notes without adding provider-specific columns.

### Notion page ingestion

The Notion adapter now supports selected page ingestion:

```text
selected ProviderObjectSnapshot(page)
→ Notion page metadata
→ Notion page markdown
→ fallback to block children text
→ normalized SourceObject
→ Artifact / EvidenceFragment pipeline
```

Notion database and data_source objects remain discoverable, but v0.8.2 blocks them from source selection until their ingestion semantics are implemented.

### Anti–Notion-lock-in guardrail

The package includes a `ManualConnector` smoke-test adapter that can:

```text
discover → select → normalize SourceObject
```

This proves the connector registry and ingestion boundary are not Notion-only.

## Persistence changes

Migration:

```text
0004_generic_ingestion_contract.py
```

Adds:

```text
artifacts.source_object_type
artifacts.provider_metadata
provider_object_snapshots.provider_metadata
ix_artifacts_source_object_type
```

## Important behavior

- Discovery metadata still does not count as evidence.
- Only selected provider objects are passed to connector ingestion.
- Notion page content becomes Artifacts only after sync.
- EvidenceFragments are still unreviewed by default.
- Officialization gates remain unchanged.
- Duplicate unchanged content still reuses Artifacts through source identity + content hash.

## Tests added

```text
test_core_artifact_schema_is_provider_agnostic
test_provider_object_snapshot_supports_provider_metadata_without_notion_lock_in
test_connector_registry_can_resolve_non_notion_manual_connector
test_manual_connector_can_discover_select_and_normalize_source_objects
test_notion_live_list_objects_fetches_selected_page_markdown
test_notion_live_list_objects_falls_back_to_block_children
test_sync_job_lifecycle_ingests_selected_notion_page_into_artifacts
```

## Deferred

```text
- Durable background worker queue
- Live PostgreSQL CI
- Notion database/data_source ingestion semantics
- Notion webhooks
- Runtime rate-limit scheduling/backoff
- Incremental sync cursors
- Production KMS/secret manager integration
- Full RBAC provider
```
