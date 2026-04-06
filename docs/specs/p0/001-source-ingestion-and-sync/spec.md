# P0-001 Source Ingestion and Sync

## Status

Draft

## Summary

Cornerstone must allow a workspace to connect shared sources, ingest source memory into the workspace boundary, keep that source memory eventually current, and preserve provenance well enough for later curation and review.

## Goals

- allow workspace-level source setup for shared organizational context
- preserve durable source memory inside the workspace boundary
- support eventual correctness rather than requiring strict real-time updates
- preserve provenance strongly enough that later review can trace official outputs back to support
- separate source ingestion from official meaning approval

## Functional requirements

### Shared-source setup

The product must allow an authorized connector manager to:
- add a shared source to a workspace
- assign source intent or template
- assign visibility as `member_visible` or `evidence_only`
- confirm which workspace the source belongs to
- review current source status after setup

### Personal-source setup

When personal context is enabled, the product must allow a personal-source owner to:
- add a personal source to a personal context
- inspect the private source in that personal context
- select specific material for later creation of `PromotedSupport`

### Eventual sync

The product must keep source memory eventually current and expose:
- new source material after sync
- changed source material after sync
- missing or inaccessible source material in source status
- stale, drifted, failed, paused, or removed conditions through canonical state vocabulary

### Provenance

Every artifact and evidence fragment created through this feature must preserve enough provenance to answer:
- where it came from
- which source produced it
- when it was last refreshed
- how a reviewer can navigate back to the original source when that is visible and allowed

### Promotion boundary

P0 must support explicit creation of `PromotedSupport` from personal context into a workspace.

Promotion must:
- create one workspace-scoped support item
- preserve promoter identity and promotion time
- preserve disclosure level and private origin lineage
- avoid exposing the rest of the private document by implication

## Acceptance criteria

- A workspace can add and manage shared sources through authorized connector managers.
- Source memory is persisted for each connected source.
- Source status makes stale, failed, or degraded conditions visible.
- Source-derived artifacts and evidence retain provenance and freshness context.
- Source changes do not directly officialize curated outputs.
- `member_visible` and `evidence_only` behavior is clearly separated.
- Explicit personal-to-workspace promotion creates `PromotedSupport` rather than exposing personal source content directly.

## Linked canonical specs

- [../../connectors/spec.md](../../connectors/spec.md)
- [../../sync-and-provenance/spec.md](../../sync-and-provenance/spec.md)
- [../../workspace-and-access/spec.md](../../workspace-and-access/spec.md)
- [../../state-vocabulary/spec.md](../../state-vocabulary/spec.md)
