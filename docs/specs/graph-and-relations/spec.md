# Graph and Relations

## Summary

Relations are the first-class semantic connections that turn isolated concepts into organizational structure.

The graph surface is a projection over concepts, concept relations, linked decisions, and support summaries inside one workspace boundary.

## Scope and owned behavior

This spec owns:
- relation behavior on human-facing and model-facing graph surfaces
- graph-facing discovery and navigation behavior
- relation support-visibility rules
- relation review-domain rules
- the distinction between semantic graph predicates and non-graph structural links

## Canonical relation semantics

A `ConceptRelation` is a concrete `CuratedObject` defined by the ontology layer.

Every relation must expose:
- stable identity
- exactly one subject concept
- exactly one canonical predicate
- exactly one object concept
- exactly one review domain
- canonical lifecycle state
- canonical verification state
- canonical support-visibility state
- visible support items available to the current consumer
- linked decisions where relevant
- provenance summary

The canonical predicate taxonomy is owned by [`../ontology/spec.md`](../ontology/spec.md).

## Semantic graph vs structural links

The graph surface uses only **semantic graph predicates** carried by `ConceptRelation`.

The following are **not** semantic graph predicates and must not replace them:
- support links from `EvidenceFragment` or `PromotedSupport`
- decision applicability links from `DecisionRecord`
- review-scope grants or connector-scope grants
- promotion lineage
- containment and provenance links

A graph surface may show those as overlays, badges, side panels, or linked context, but they are not canonical semantic edges.

## Canonical predicate families

Cornerstone’s canonical graph supports at least these predicate families:
- ontological
  - `is_a`, `part_of`, `instance_of`
- operational
  - `depends_on`, `used_in`, `input_to`, `output_of`
- flow and temporal
  - `precedes`, `triggers`, `results_in`
- governance and semantic
  - `owned_by`, `governed_by`, `defined_by`, `applies_to`, `conflicts_with`, `supersedes`

A new canonical predicate requires an ontology update before it may appear as a stable graph edge.

## Review-domain rule

- If both endpoint concepts have the same `owning_domain`, the relation uses that domain as `review_domain`.
- Otherwise the relation uses `workspace` as `review_domain` and requires workspace-wide review.

## Current behavior

- `ConceptRelation` is first-class and reviewable.
- Member-facing graph views default to `member_visible` support.
- Review views may expose `evidence_only` support when needed for diagnosis or approval.
- A member-facing relation may be labeled `source_backed` only when the current consumer can inspect at least one visible support item.
- Cross-workspace semantic graph edges are not allowed.

## Constraints and non-goals

- The graph is not a separate truth store.
- The graph should not expose raw implementation edges without support context.
- This spec does not require one specific graph visualization implementation.

## Related docs

- [`../ontology/spec.md`](../ontology/spec.md)
- [`../domain-model/spec.md`](../domain-model/spec.md)
- [`../concepts/spec.md`](../concepts/spec.md)
- [`../decision-context/spec.md`](../decision-context/spec.md)
- [`../review-and-validation/spec.md`](../review-and-validation/spec.md)
- [`../../decisions/0005-glossary-and-graph-are-projections-over-curated-entities.md`](../../decisions/0005-glossary-and-graph-are-projections-over-curated-entities.md)
- [`../../decisions/0015-ontology-layer-is-the-canonical-semantic-reference.md`](../../decisions/0015-ontology-layer-is-the-canonical-semantic-reference.md)
