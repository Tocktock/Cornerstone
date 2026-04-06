# State Vocabulary

## Summary

This spec defines the **canonical state and enum vocabulary** exposed by Cornerstone across human UI, programmatic API, MCP, and other model-facing surfaces.

Implementations may use richer internal state models, but anything that crosses the canonical contract boundary must map to the values defined here.

## Why this exists

Consumers cannot reliably integrate with Cornerstone if lifecycle, verification, trust, or freshness values change by surface or by implementation.

Stable meaning requires stable exposed vocabulary.

## Scope and owned behavior

This spec owns:
- canonical exposed enum values
- mapping rules from internal states to canonical states
- the distinction between surface vocabulary and internal implementation vocabulary

## Contract rules

- The values in this spec are normative for any consumer-facing surface.
- A surface may not invent a different exposed enum for the same semantic field.
- An implementation may maintain additional internal states only if it maps them losslessly to one canonical exposed value before the result leaves the implementation boundary.
- If a field is defined here and surfaced in the serving contract, the surface must use the exact enum value defined here.

## Canonical enums

### Support visibility

| Field | Allowed values | Meaning |
| --- | --- | --- |
| `support_visibility` | `source_backed`, `restricted_support`, `insufficient_support` | What the current consumer can honestly claim about the inspectability and sufficiency of support |

### Curated object lifecycle

Used by concepts and relations.

| Field | Allowed values | Meaning |
| --- | --- | --- |
| `lifecycle_state` | `suggested`, `draft`, `in_review`, `official`, `deprecated`, `archived` | Publication state of a concept or relation |

### Decision lifecycle

Used by decision records.

| Field | Allowed values | Meaning |
| --- | --- | --- |
| `lifecycle_state` | `proposed`, `in_review`, `accepted`, `rejected`, `superseded`, `archived` | Publication state of a decision record |

### Verification state

| Field | Allowed values | Meaning |
| --- | --- | --- |
| `verification_state` | `unverified`, `verified`, `monitoring`, `review_required`, `support_insufficient`, `drift_detected` | Current trust and revalidation posture |

### Source connection state

| Field | Allowed values | Meaning |
| --- | --- | --- |
| `source_connection_state` | `pending_setup`, `active`, `syncing`, `degraded`, `paused`, `removed` | Operational state of a source connection |

### Freshness state

| Field | Allowed values | Meaning |
| --- | --- | --- |
| `freshness_state` | `current`, `monitoring`, `stale`, `drift_detected`, `unknown` | Currentness of support or source memory |

### Context-space kind

| Field | Allowed values | Meaning |
| --- | --- | --- |
| `context_space_kind` | `workspace`, `personal` | Whether the boundary is shared organizational context or personal context |

### Visibility class

| Field | Allowed values | Meaning |
| --- | --- | --- |
| `visibility_class` | `member_visible`, `evidence_only` | Whether source or support may appear in normal member-facing surfaces |

## Mapping rules

- Human-facing prose may render friendly labels such as “source-backed,” but the canonical exposed enum remains `source_backed`.
- Internal implementations may track richer freshness or health detail, but the exposed contract must map it to one `freshness_state` and, where relevant, one `source_connection_state`.
- A surface may omit a state field only when the serving contract explicitly marks it as optional for that response kind.

## Related docs

- [../serving-contract/spec.md](../serving-contract/spec.md)
- [../review-and-validation/spec.md](../review-and-validation/spec.md)
- [../sync-and-provenance/spec.md](../sync-and-provenance/spec.md)
- [../decision-context/spec.md](../decision-context/spec.md)
- [../../decisions/0013-canonical-state-vocabulary-is-stable-across-surfaces.md](../../decisions/0013-canonical-state-vocabulary-is-stable-across-surfaces.md)
