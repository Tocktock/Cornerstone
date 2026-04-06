# Cornerstone Docs

This folder is the canonical documentation home for Cornerstone.

## What Cornerstone is

Cornerstone continuously reconstructs organizational context from scattered documents, conversations, and systems, then serves that context back to humans and AI in a source-backed, reviewable form.

Internally, Cornerstone is the **semantic system of record** for organizational meaning, relationships, decision context, support visibility, and provenance disclosure.

Externally, Cornerstone should be described as the **shared organizational context layer for humans and AI**.

## Documentation model

- [specs/](./specs/): canonical product behavior and feature contracts
- [decisions/](./decisions/): durable product and architecture choices
- [memories/](./memories/): rationale history and context notes

## Source of Truth rules

- Feature behavior belongs in `docs/specs/`.
- Durable boundaries belong in `docs/decisions/`.
- Historical notes belong in `docs/memories/`.
- Files outside `docs/` may summarize these rules, but they must not restate detailed behavior as a second Source of Truth.

## Current canonical specs

- [system-overview](./specs/system-overview/spec.md)
- [foundations](./specs/foundations/spec.md)
- [ontology](./specs/ontology/spec.md)
- [state-vocabulary](./specs/state-vocabulary/spec.md)
- [workspace-and-access](./specs/workspace-and-access/spec.md)
- [connectors](./specs/connectors/spec.md)
- [sync-and-provenance](./specs/sync-and-provenance/spec.md)
- [domain-model](./specs/domain-model/spec.md)
- [concepts](./specs/concepts/spec.md)
- [graph-and-relations](./specs/graph-and-relations/spec.md)
- [decision-context](./specs/decision-context/spec.md)
- [review-and-validation](./specs/review-and-validation/spec.md)
- [retrieval-and-answers](./specs/retrieval-and-answers/spec.md)
- [serving-contract](./specs/serving-contract/spec.md)
- [authoring-and-curation](./specs/authoring-and-curation/spec.md)
- [p0 implementation specs](./specs/p0/README.md)

## Maintenance workflow

1. Update or create the relevant feature spec before implementation.
2. Update the ontology spec when a new canonical object kind, concept kind, or semantic predicate is introduced.
3. Add or update a decision record if a durable boundary changes.
4. Add a memory note only when the why matters later, and keep it subordinate to the specs and decisions.
5. Keep summaries short and link back here instead of duplicating detail.
