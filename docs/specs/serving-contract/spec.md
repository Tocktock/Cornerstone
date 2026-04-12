# Serving Contract

## Summary

This spec defines the **canonical consumer-facing contract** for Cornerstone across:

The semantic meaning of the payloads in this contract is owned by [`../ontology/spec.md`](../ontology/spec.md). This spec owns only the consumer-facing envelope and payload shapes.
- human UI
- programmatic API
- MCP and other model-facing surfaces

Transport details are replaceable. Response kinds, field semantics, cardinality, and exposed enum values are not.

## Why this exists

Cornerstone serves the same truth layer to multiple kinds of consumers. Without one canonical contract:
- UI would invent one answer shape
- API would invent another
- MCP tools would invent a third
- trust and provenance labels would drift between surfaces

This spec prevents that drift.

## Contract conventions

- The field names below are the canonical semantic names.
- Cardinality uses the notation `1`, `0..1`, `1..*`, and `0..*`.
- A surface may wrap or package these fields differently only when a lossless mapping to the canonical fields exists.
- Canonical enum values are defined in [../state-vocabulary/spec.md](../state-vocabulary/spec.md) and must be used exactly where referenced here.

## Canonical operation intents

Allowed `request_intent` values are:
- `get_workspace_home`
- `search_context`
- `get_concept`
- `get_relation`
- `get_decision`
- `get_answer`
- `get_graph_slice`
- `follow_provenance`

## Canonical response kinds

Allowed `response_kind` values are:
- `workspace_home`
- `concept`
- `relation`
- `decision`
- `answer`
- `search_results`
- `graph_slice`
- `provenance`
- `no_match`

## Common response envelope

Every consumer-facing response must preserve the following fields:

| Field | Cardinality | Meaning |
| --- | --- | --- |
| `contract_version` | `1` | Canonical contract version identifier |
| `response_kind` | `1` | One canonical response kind |
| `request_intent` | `1` | One canonical request intent |
| `context_space_ref` | `1` | Current workspace or personal context reference |
| `consumer_scope` | `1` | One of `member`, `review`, `admin` |
| `payload` | `1` | Response-kind-specific body |
| `related_refs` | `0..*` | Linked resources that may be followed next |
| `warnings` | `0..*` | Non-fatal disclosure or caution labels |

## Bootstrap contract

The frontend bootstrap payload is additive and must preserve:

| Field | Cardinality | Meaning |
| --- | --- | --- |
| `workspace` | `1` | Primary shared workspace context reference |
| `personal_context` | `1` | Default personal context reference |
| `actors` | `0..*` | Available actor sessions for the current app bootstrap |
| `runtime_mode` | `1` | One of `mock` or `production` |
| `workspace_data_state` | `1` | One of `demo_seeded`, `awaiting_sources`, `syncing_sources`, `ready`, `degraded` |
| `linked_source_count` | `1` | Count of currently linked shared workspace sources |
| `active_source_count` | `1` | Count of currently active shared workspace sources |
| `degraded_source_count` | `1` | Count of linked sources currently requiring recovery attention |

Bootstrap runtime metadata is additive only. It must not redefine the semantic meaning of concept, decision, answer, graph, provenance, or workspace-home payloads.

### Context-space reference

| Field | Cardinality | Meaning |
| --- | --- | --- |
| `context_space_id` | `1` | Stable context-space identity |
| `context_space_kind` | `1` | Canonical context-space kind |
| `context_space_name` | `1` | Human-readable name |

### Resource reference

| Field | Cardinality | Meaning |
| --- | --- | --- |
| `resource_kind` | `1` | One of `concept`, `relation`, `decision`, `artifact`, `support_item` |
| `resource_id` | `1` | Stable resource identity |
| `resource_label` | `1` | Human-readable label |

### Support-item summary

| Field | Cardinality | Meaning |
| --- | --- | --- |
| `support_item_id` | `1` | Stable support-item identity |
| `support_item_kind` | `1` | `evidence_fragment` or `promoted_support` |
| `visibility_class` | `1` | `member_visible` or `evidence_only` |
| `source_label` | `1` | Source or origin summary |
| `excerpt_or_summary` | `0..1` | Shared visible excerpt or summary |
| `origin_disclosure_level` | `0..1` | Present for promoted support when disclosure is relevant |
| `source_locator` | `0..1` | Link or locator when visible to the current consumer |

### Provenance summary

| Field | Cardinality | Meaning |
| --- | --- | --- |
| `support_item_count` | `1` | Count of support items considered for the payload |
| `visible_support_item_count` | `1` | Count of support items visible to the current consumer |
| `restricted_support_present` | `1` | Whether hidden decisive support exists |
| `freshness_state` | `1` | Canonical freshness state |
| `verification_state` | `0..1` | Canonical verification state when the payload represents curated meaning |
| `promotion_lineage_present` | `1` | Whether promoted personal support contributes to the payload |

## Resource payload shapes

### `concept` payload

| Field | Cardinality | Meaning |
| --- | --- | --- |
| `concept_id` | `1` | Stable concept identity |
| `public_slug` | `1` | Stable member-facing identifier |
| `canonical_name` | `1` | Canonical term name |
| `aliases` | `0..*` | Alternate labels |
| `definition` | `1` | Canonical definition |
| `owning_domain` | `1` | Responsible domain |
| `review_domain` | `1` | Canonical review domain |
| `lifecycle_state` | `1` | Canonical curated-object lifecycle |
| `verification_state` | `1` | Canonical verification state |
| `support_visibility` | `1` | Canonical support-visibility value |
| `visible_support_items` | `0..*` | Support items visible to the current consumer |
| `linked_relation_refs` | `0..*` | Related relations |
| `linked_decision_refs` | `0..*` | Related decisions |
| `provenance_summary` | `1` | Provenance summary |

### `relation` payload

| Field | Cardinality | Meaning |
| --- | --- | --- |
| `relation_id` | `1` | Stable relation identity |
| `subject_concept_ref` | `1` | Subject concept |
| `predicate` | `1` | Canonical relation predicate |
| `object_concept_ref` | `1` | Object concept |
| `description` | `0..1` | Human-readable explanation |
| `review_domain` | `1` | Canonical review domain |
| `lifecycle_state` | `1` | Canonical curated-object lifecycle |
| `verification_state` | `1` | Canonical verification state |
| `support_visibility` | `1` | Canonical support-visibility value |
| `visible_support_items` | `0..*` | Support items visible to the current consumer |
| `linked_decision_refs` | `0..*` | Decisions explaining the relation |
| `provenance_summary` | `1` | Provenance summary |

### `decision` payload

| Field | Cardinality | Meaning |
| --- | --- | --- |
| `decision_id` | `1` | Stable decision identity |
| `public_slug` | `1` | Stable member-facing decision identifier |
| `title` | `1` | Decision title |
| `decision_statement` | `1` | Explicit decision statement |
| `problem_statement` | `0..1` | Trigger or problem |
| `rationale` | `0..1` | Why the decision was chosen |
| `constraints` | `0..*` | Constraints that shaped the decision |
| `impact_summary` | `0..1` | Expected or realized impact |
| `owning_domain` | `1` | Responsible domain |
| `review_domain` | `1` | Canonical review domain |
| `lifecycle_state` | `1` | Canonical decision lifecycle |
| `support_visibility` | `1` | Canonical support-visibility value |
| `visible_support_items` | `0..*` | Support items visible to the current consumer |
| `linked_concept_refs` | `0..*` | Affected concepts |
| `linked_relation_refs` | `0..*` | Affected relations |
| `supersedes_ref` | `0..1` | Older decision replaced or narrowed |
| `superseded_by_ref` | `0..1` | Newer decision that replaced this one |
| `provenance_summary` | `1` | Provenance summary |

### `workspace_home` payload

| Field | Cardinality | Meaning |
| --- | --- | --- |
| `hero_prompt` | `1` | Primary prompt copy for the workspace ask surface |
| `featured_answer` | `0..1` | Featured answer envelope or no-match envelope for the default home prompt |
| `featured_cards` | `0..*` | Editorial summaries for featured concepts and decisions |
| `recent_changes` | `0..*` | Lightweight summaries of recently changed presentable artifacts |
| `freshness_alerts` | `0..*` | Quiet operational cues about source freshness and degraded state |
| `review_queue_summary` | `1` | Aggregate review pressure summary for the current workspace |
| `source_health_summary` | `1` | Aggregate source-connection health summary for the current workspace |

Each featured card must expose:

| Field | Cardinality | Meaning |
| --- | --- | --- |
| `resource_ref` | `1` | Referenced concept or decision |
| `public_slug` | `1` | Stable presentable slug for the card detail route |
| `title` | `1` | Reader-facing artifact title |
| `eyebrow` | `1` | Short editorial label for the artifact type and domain |
| `summary` | `1` | Short presentable summary text |
| `support_visibility` | `1` | Canonical support-visibility value |
| `lifecycle_state` | `1` | Canonical lifecycle state |
| `verification_state` | `0..1` | Canonical verification state when available |
| `provenance_summary` | `1` | Provenance summary |

Each recent-change item must expose:

| Field | Cardinality | Meaning |
| --- | --- | --- |
| `resource_ref` | `1` | Changed concept or decision |
| `public_slug` | `1` | Stable presentable slug for the detail route |
| `change_summary` | `1` | Reader-facing description of the current artifact state |
| `changed_at` | `1` | Timestamp for the most recent material artifact change |
| `support_visibility` | `1` | Canonical support-visibility value |
| `lifecycle_state` | `1` | Canonical lifecycle state |
| `verification_state` | `0..1` | Canonical verification state when available |

Each freshness-alert item must expose:

| Field | Cardinality | Meaning |
| --- | --- | --- |
| `source_connection_id` | `1` | Stable source-connection identity |
| `source_label` | `1` | Human-readable source label |
| `source_connection_state` | `1` | Canonical source-connection state |
| `freshness_state` | `1` | Canonical freshness state |
| `last_successful_sync_at` | `0..1` | Timestamp for the most recent successful sync |
| `note` | `1` | Human-readable monitoring or intervention note |

The review-queue summary must expose:

| Field | Cardinality | Meaning |
| --- | --- | --- |
| `pending_count` | `1` | Number of shared objects currently needing review attention |
| `review_required_count` | `1` | Number of items blocked on explicit review-required state |
| `officialize_ready_count` | `1` | Number of items that can be moved forward without review-required blocking state |

The source-health summary must expose:

| Field | Cardinality | Meaning |
| --- | --- | --- |
| `total_count` | `1` | Number of source connections in the workspace |
| `active_count` | `1` | Number of active source connections |
| `monitoring_count` | `1` | Number of connections in monitoring freshness state |
| `stale_count` | `1` | Number of stale connections |
| `degraded_count` | `1` | Number of degraded connections |
| `paused_count` | `1` | Number of paused connections |
| `removed_count` | `1` | Number of removed connections |

### `answer` payload

| Field | Cardinality | Meaning |
| --- | --- | --- |
| `answer_status` | `1` | One of `official`, `partial`, `review_required` |
| `answer_text` | `1` | Grounded answer text |
| `answer_sections` | `0..*` | Structured answer sections |
| `support_visibility` | `1` | Canonical support-visibility value |
| `verification_state` | `1` | Canonical verification state |
| `visible_support_items` | `0..*` | Support items visible to the current consumer |
| `cited_concept_refs` | `0..*` | Concepts cited by the answer |
| `cited_relation_refs` | `0..*` | Relations cited by the answer |
| `cited_decision_refs` | `0..*` | Decisions cited by the answer |
| `provenance_summary` | `1` | Provenance summary |
| `follow_up_refs` | `0..*` | Recommended next hops |

### `search_results` payload

| Field | Cardinality | Meaning |
| --- | --- | --- |
| `results` | `0..*` | Ordered search-result items |
| `result_count` | `1` | Total result count returned in this response |

Each search-result item must expose:

| Field | Cardinality | Meaning |
| --- | --- | --- |
| `resource_ref` | `1` | Resource returned by the search |
| `match_reason_summary` | `1` | Why the result matched |
| `support_visibility` | `0..1` | Present when the result is a curated object or answerable support summary |
| `lifecycle_state` | `0..1` | Present when the result is a curated object |
| `verification_state` | `0..1` | Present when the result is a curated object |
| `provenance_summary` | `0..1` | Present when provenance matters for the result |

### `graph_slice` payload

| Field | Cardinality | Meaning |
| --- | --- | --- |
| `root_concept_refs` | `1..*` | Concepts that anchor the slice |
| `nodes` | `1..*` | Concepts included in the slice |
| `edges` | `0..*` | Relations included in the slice |

Each graph edge must expose:

| Field | Cardinality | Meaning |
| --- | --- | --- |
| `relation_ref` | `1` | Relation represented by the edge |
| `subject_concept_ref` | `1` | Edge subject |
| `predicate` | `1` | Edge predicate |
| `object_concept_ref` | `1` | Edge object |
| `support_visibility` | `1` | Canonical support-visibility value |
| `verification_state` | `1` | Canonical verification state |

### `provenance` payload

| Field | Cardinality | Meaning |
| --- | --- | --- |
| `subject_ref` | `1` | Resource whose provenance is being followed |
| `support_items` | `0..*` | Support items visible to the current consumer |
| `source_summaries` | `0..*` | Source-origin summaries |
| `provenance_summary` | `1` | Provenance summary |

### `no_match` payload

| Field | Cardinality | Meaning |
| --- | --- | --- |
| `reason` | `1` | One of `no_official_match`, `no_visible_match`, `outside_scope`, `not_available_in_workspace` |
| `request_rewrite_hint` | `0..1` | Suggested clarification or query rewrite |
| `suggested_follow_up` | `0..*` | Recommended next actions or resource refs |

## Non-goals

- This spec does not define transport syntax.
- This spec does not define one endpoint layout or tool naming scheme.
- This spec does not replace the behavior specs that define when a result should exist.
- This spec does not authorize surfaces to hide trust-state differences behind prose.

## Related docs

- [../ai-operator-surfaces/spec.md](../ai-operator-surfaces/spec.md)
- [../state-vocabulary/spec.md](../state-vocabulary/spec.md)
- [../retrieval-and-answers/spec.md](../retrieval-and-answers/spec.md)
- [../workspace-and-access/spec.md](../workspace-and-access/spec.md)
- [../review-and-validation/spec.md](../review-and-validation/spec.md)
- [../../decisions/0012-serving-contract-is-canonical-across-ui-api-and-mcp-surfaces.md](../../decisions/0012-serving-contract-is-canonical-across-ui-api-and-mcp-surfaces.md)
- [../../decisions/0010-member-facing-source-backed-claims-require-visible-support-or-restricted-support-disclosure.md](../../decisions/0010-member-facing-source-backed-claims-require-visible-support-or-restricted-support-disclosure.md)
- [../../decisions/0013-canonical-state-vocabulary-is-stable-across-surfaces.md](../../decisions/0013-canonical-state-vocabulary-is-stable-across-surfaces.md)
