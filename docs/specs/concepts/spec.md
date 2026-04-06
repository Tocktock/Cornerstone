# Concepts

## Summary

Concepts are the member-facing glossary surface of Cornerstone.

They expose canonical organizational meanings backed by support context, linked relations, and linked decision context. Concepts are how the product turns raw source memory into stable, consumable organizational meaning.

## Scope and owned behavior

This spec owns:
- concept discovery and detail behavior
- stable concept identity on member-facing surfaces
- concept request intake
- concept-facing support visibility rules
- concept-facing use of ontology kinds and relation links

## Canonical concept semantics

A concept is a concrete `CuratedObject` defined by the ontology layer.

Every concept must preserve:
- stable identity
- stable `public_slug`
- canonical name
- definition
- exactly one `concept_kind`
- exactly one `owning_domain`
- exactly one `review_domain`
- canonical lifecycle state
- canonical verification state
- canonical support-visibility state
- linked visible support available to the current consumer
- linked semantic relations
- linked decision records
- provenance summary

## Concept kinds

Concept kinds are owned by [`../ontology/spec.md`](../ontology/spec.md).

Cornerstone supports canonical concept kinds such as:
- `term`
- `domain`
- `system`
- `policy`
- `workflow`
- `role`
- `metric`
- `event`
- `artifact_type`
- `status`

A surface may choose whether to render the kind explicitly, but implementations must preserve it semantically.

## Current behavior

Additional rules:
- Each concept owns a stable workspace-scoped `public_slug` so URLs and references do not depend on the latest label wording.
- A concept may be supported by both `member_visible` and `evidence_only` support items.
- Member-facing views show only visible support items.
- If a concept’s approval depends partly or wholly on hidden support, the concept must disclose `restricted_support`.
- A member-facing concept may be labeled `source_backed` only when the current member can inspect at least one visible support item.
- Admin and review views may reveal `evidence_only` support rows when needed to explain or approve the output.
- A missing-term request creates or updates a suggested concept candidate rather than bypassing review and publishing.

## Permissions and visibility

- Members can read official concepts.
- Members may submit requests for missing concepts.
- Members do not directly approve or publish concept changes.
- Actors with `operate` can create and refine concept drafts.
- Actors with matching `review` scope can officialize and revalidate concepts.

## Constraints and non-goals

- Concepts are not the admin review queue.
- Concepts are not a second storage root beside the curated domain model.
- Concepts do not own relation or decision approval logic; those belong to review and validation.

## Related docs

- [`../ontology/spec.md`](../ontology/spec.md)
- [`../domain-model/spec.md`](../domain-model/spec.md)
- [`../graph-and-relations/spec.md`](../graph-and-relations/spec.md)
- [`../review-and-validation/spec.md`](../review-and-validation/spec.md)
- [`../retrieval-and-answers/spec.md`](../retrieval-and-answers/spec.md)
- [`../serving-contract/spec.md`](../serving-contract/spec.md)
- [`../../decisions/0010-member-facing-source-backed-claims-require-visible-support-or-restricted-support-disclosure.md`](../../decisions/0010-member-facing-source-backed-claims-require-visible-support-or-restricted-support-disclosure.md)
