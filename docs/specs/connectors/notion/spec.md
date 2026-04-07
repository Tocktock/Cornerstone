# Notion Shared Connector

## Summary

This spec defines the P0 Notion connector behavior for shared workspace sources.

The Notion connector is template-driven and ships two templates:
- `notion_shared_page_tree`
- `notion_shared_database`

## Status

Draft

## Why this exists

Notion is a high-value source of organizational definitions, workflows, rationale, and structured records.

P0 needs one real provider implementation that proves the connector platform beyond local fixture sources while preserving the existing Cornerstone trust model.

## Scope and owned behavior

This spec owns:
- the first Notion templates
- shared-source setup inputs
- Notion scope selection behavior
- preview behavior
- initial and recurring sync behavior
- Notion-specific normalization expectations
- manager-only binding behavior

This spec does not make Notion the canonical source of truth. It remains an upstream source system.

## Templates

### `notion_shared_page_tree`

Use when a connector manager wants to ingest one root page and the descendant pages contained under that tree.

The template must preserve:
- one selected root page
- descendant page discovery under that root
- page content normalized into one `Artifact` per page
- `EvidenceFragment` derivation from visible block or paragraph text

### `notion_shared_database`

Use when a connector manager wants to ingest one Notion database and its entries.

The template must preserve:
- one selected database
- one `Artifact` per database entry page
- row or property summary fragments suitable for evidence and review

## Setup contract

Shared setup is manager-only and must require:
- one bound Notion credential reference
- one template choice
- one target scope input or selection
- one `visibility_class`
- one human-readable source label

Before save, Cornerstone must:
- validate the credential reference against the workspace
- resolve the requested page or database identity
- produce a preview sample
- disclose the default visibility and sync behavior

## Binding contract

- Binding is initiated by a workspace actor allowed to manage connectors.
- Binding uses a user OAuth flow when available.
- Cornerstone stores the resulting provider credential internally and surfaces only a workspace-bound credential reference in connection setup or management APIs.
- Non-manager actors must not be able to inspect provider auth payloads, tokens, or rebind controls.

## Sync contract

- Initial sync is full.
- Recurring sync is scheduled and may use incremental checkpoints when the adapter can do so safely.
- If incremental continuation is not safe or the checkpoint becomes invalid, the connector must fall back to a safe scan without inventing new consumer-facing state vocabulary.
- Manual resync is always allowed for authorized actors.

Loss of upstream visibility or deleted upstream content must:
- keep prior synced memory visible for provenance and review honesty
- move affected artifacts and support to `monitoring`
- move the source connection to `degraded` only when the connection itself is unhealthy or access is broken

## Normalization contract

- One upstream logical page becomes one `Artifact`.
- Artifact provenance must preserve:
  - upstream page or database entry identity
  - source locator
  - last edited time
  - connector template key
  - relevant scope metadata
- `EvidenceFragment` extraction should prefer readable paragraph or block-level text over implementation-specific block JSON.
- Database entries may summarize property values into readable evidence fragments, but they should remain provenance-first rather than schema-heavy in P0.

## Constraints and non-goals

- P0 does not support workspace-wide Notion crawl.
- P0 does not require every block type to become a first-class normalized structure.
- P0 does not change personal-source behavior.
- P0 does not add Google Drive in the same milestone.

## Related docs

- [../spec.md](../spec.md)
- [../../p0/001-source-ingestion-and-sync/spec.md](../../p0/001-source-ingestion-and-sync/spec.md)
- [../../workspace-and-access/spec.md](../../workspace-and-access/spec.md)
- [../../sync-and-provenance/spec.md](../../sync-and-provenance/spec.md)
