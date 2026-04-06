# Decision Context

## Summary

Decision context preserves the rationale layer of organizational meaning.

It explains:
- what was decided
- why it was decided
- which constraints were in play
- what concepts or relations were affected
- which support grounded the decision
- whether a later decision superseded it

## Scope and owned behavior

This spec owns:
- the product definition of decision context
- the required structure of decision records
- how decision records link to concepts, relations, and support
- the behavior expected from decision-facing surfaces

## Canonical decision semantics

A `DecisionRecord` is a concrete `CuratedObject` defined by the ontology layer.

Every decision record must preserve:
- stable identity
- title
- problem or trigger
- decision statement
- rationale
- zero or more constraints
- zero or one impact summary
- exactly one owning domain
- exactly one review domain
- canonical lifecycle state
- canonical verification state
- canonical support-visibility state
- zero or more linked support items
- zero or more linked concept refs
- zero or more linked relation refs
- optional supersession lineage

## Decision links vs graph predicates

A decision record may affect concepts and concept relations, but it is **not** itself a semantic graph predicate.

That means:
- support links remain support links
- decision applicability remains a decision link family
- semantic graph edges remain `ConceptRelation` predicates

A graph or passport view may surface decision overlays or linked rationale panels, but it must not collapse decision applicability into a fake semantic predicate.

## Current behavior

- Decision context is a first-class requirement of Cornerstone.
- The canonical system object is `DecisionRecord`.
- Decision records may apply to:
  - concepts
  - concept relations
  - systems
  - policies
  - workflows
  - roles
- Decision records must be reviewable and traceable.
- Not every meeting, note, or thread becomes a decision record.
- A good decision record lets a reader reconstruct most of the rationale without reopening many raw sources.

## State model

Decision lifecycle and verification states are owned by [`../state-vocabulary/spec.md`](../state-vocabulary/spec.md).

Accepted records may remain readable even after they are later superseded, as long as the newer lineage is clearly shown.

## Permissions and visibility

- Decision records may be readable more broadly once accepted and relevant to an official concept, policy, or workflow.
- Draft or proposed decision records remain operator- or reviewer-limited until reviewed.
- Creating or refining a decision draft requires `operate`.
- Approval, rejection, supersession, and revalidation require `review` within allowed scope.
- Domain ownership may route responsibility but does not replace review capability.

## Constraints and non-goals

- Decision context is not raw meeting-note storage.
- Decision context is not hidden model chain-of-thought.
- This spec does not require every curated output to have its own standalone decision record, but it does require Cornerstone to support them as first-class objects.

## Related docs

- [`../ontology/spec.md`](../ontology/spec.md)
- [`../domain-model/spec.md`](../domain-model/spec.md)
- [`../review-and-validation/spec.md`](../review-and-validation/spec.md)
- [`../workspace-and-access/spec.md`](../workspace-and-access/spec.md)
- [`../../decisions/0004-decision-records-are-first-class.md`](../../decisions/0004-decision-records-are-first-class.md)
- [`../../decisions/0015-ontology-layer-is-the-canonical-semantic-reference.md`](../../decisions/0015-ontology-layer-is-the-canonical-semantic-reference.md)
