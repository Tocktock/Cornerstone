# Sync and Provenance

## Summary

Sync and provenance define how Cornerstone stays aligned with source systems, exposes freshness honestly, and preserves the lineage required for trust.

## Why this exists

Cornerstone cannot claim grounded organizational context if it cannot answer:
- where a support item came from
- when it was last refreshed
- whether it came from a shared connector, a snapshot, or promoted personal material
- whether the current trust posture still reflects source reality

## Scope and owned behavior

This spec owns:
- sync semantics
- provenance display and storage contract
- freshness and source-connection state semantics
- support-item lineage expectations
- promotion lineage from personal context into workspace context
- sync observability principles

## Current behavior

- Source systems remain the upstream source of raw operational truth.
- Cornerstone persists normalized source memory so it can rehydrate provenance and review state over time.
- Sync is eventual rather than strongly real-time.
- Sync failure must be observable and recoverable.
- A stale or drifted source does not automatically destroy official context, but it may reopen review.
- Provenance should be visible not only to operators but also to members and AI consumers when it affects trust.

## Canonical support-item types

Cornerstone recognizes two canonical support-item types for shared context:
- `EvidenceFragment`
- `PromotedSupport`

Shared official outputs may cite either type.

Shared official outputs may **not** cite personal-context artifacts or personal-context evidence directly.

## Provenance contract

Every surfaced support item should preserve:
- source label
- support-item kind
- visibility class
- freshness state where relevant
- last successful sync summary where relevant
- evidence count where relevant
- promotion lineage when the support item originated in personal context and was later shared

## Promoted-support contract

`PromotedSupport` is the canonical workspace-scoped object created when private personal material is explicitly shared into a workspace.

A promoted-support item must preserve the following required fields:

| Field | Cardinality | Meaning |
| --- | --- | --- |
| `promoted_support_id` | `1` | Stable identity of the promoted support item |
| `workspace_ref` | `1` | Target workspace that now contains the shared support item |
| `promoter_ref` | `1` | Actor who created the promotion |
| `promoted_at` | `1` | When the item was promoted into shared context |
| `shared_selection_kind` | `1` | One of `artifact_excerpt`, `section_excerpt`, `fragment_excerpt`, `summary_claim` |
| `shared_payload` | `1` | The exact shared content available to workspace review and retrieval according to visibility rules |
| `origin_disclosure_level` | `1` | One of `named_origin`, `redacted_origin`, `hidden_origin` |
| `private_origin_ref` | `1` | Private lineage reference back to the personal origin |
| `visibility_class` | `1` | `member_visible` or `evidence_only` |

### Partial-document promotion

Promotion may apply to:
- a whole artifact excerpt intended to be shared as-is
- one section of a larger artifact
- one specific fragment or range
- a summary claim derived from private material

Only the selected shared portion becomes part of the workspace. Promotion does not expose the rest of the private document by implication.

### Lineage inspection rules

- Members may inspect a promoted-support item only when it is `member_visible` and linked to a member-visible official output.
- Reviewers and admins may inspect the workspace-visible shared payload, promoter identity, promotion time, and origin disclosure level needed for review.
- No workspace actor other than the promoter automatically gains access to the underlying personal artifact, connector, or full personal origin reference.

## Sync guarantees

- Polling, scheduled sync, webhook, and hybrid models are all valid.
- Snapshot uploads do not behave like live sources and should not promise live refresh.
- Every source should expose enough metadata to understand:
  - last successful sync
  - last attempted sync
  - current `source_connection_state`
  - current `freshness_state` where relevant
  - effective sync policy
  - recoverability path
- Repeated sync should be idempotent with respect to source identity and version semantics.
- Revalidation should be able to operate on already ingested support when a full live sync is not required.

## Constraints and non-goals

- This spec does not require a single sync engine implementation.
- This spec does not require strict real-time updates for every provider.
- This spec does not define review policy; it defines the fidelity and traceability that review depends on.

## Related docs

- [`../ontology/spec.md`](../ontology/spec.md)

- [../connectors/spec.md](../connectors/spec.md)
- [../workspace-and-access/spec.md](../workspace-and-access/spec.md)
- [../state-vocabulary/spec.md](../state-vocabulary/spec.md)
- [../review-and-validation/spec.md](../review-and-validation/spec.md)
- [../../decisions/0007-eventual-correctness-and-provenance-over-instant-sync.md](../../decisions/0007-eventual-correctness-and-provenance-over-instant-sync.md)
- [../../decisions/0011-personal-sources-remain-separate-until-explicitly-promoted-into-shared-context.md](../../decisions/0011-personal-sources-remain-separate-until-explicitly-promoted-into-shared-context.md)
