# Cornerstone Documentation

This package contains the **canonical product documentation** for Cornerstone.

Cornerstone is the **shared organizational context layer for humans and AI**. These docs define what Cornerstone is, what it is not, which product foundations are fixed, and which functional contracts must exist before implementation begins.

## Reading order

Read in this order if you are new to the project:

1. [docs/specs/system-overview/spec.md](./docs/specs/system-overview/spec.md)
2. [docs/specs/foundations/spec.md](./docs/specs/foundations/spec.md)
3. [docs/specs/ontology/spec.md](./docs/specs/ontology/spec.md)
4. [docs/specs/state-vocabulary/spec.md](./docs/specs/state-vocabulary/spec.md)
5. [docs/specs/workspace-and-access/spec.md](./docs/specs/workspace-and-access/spec.md)
6. [docs/specs/connectors/spec.md](./docs/specs/connectors/spec.md)
7. [docs/specs/sync-and-provenance/spec.md](./docs/specs/sync-and-provenance/spec.md)
8. [docs/specs/domain-model/spec.md](./docs/specs/domain-model/spec.md)
9. [docs/specs/concepts/spec.md](./docs/specs/concepts/spec.md)
10. [docs/specs/graph-and-relations/spec.md](./docs/specs/graph-and-relations/spec.md)
11. [docs/specs/decision-context/spec.md](./docs/specs/decision-context/spec.md)
12. [docs/specs/review-and-validation/spec.md](./docs/specs/review-and-validation/spec.md)
13. [docs/specs/retrieval-and-answers/spec.md](./docs/specs/retrieval-and-answers/spec.md)
14. [docs/specs/serving-contract/spec.md](./docs/specs/serving-contract/spec.md)
15. [docs/specs/authoring-and-curation/spec.md](./docs/specs/authoring-and-curation/spec.md)
16. [docs/specs/p0/README.md](./docs/specs/p0/README.md)

## Documentation model

- [docs/specs/](./docs/specs/): canonical product and feature behavior
- [docs/decisions/](./docs/decisions/): durable product and architecture choices
- [docs/memories/](./docs/memories/): rationale and historical context that should not override specs or decisions

## Source of Truth rules

- Product and feature behavior must be defined in `docs/specs/`.
- Durable product and architecture choices belong in `docs/decisions/`.
- Rationale and historical notes belong in `docs/memories/`.
- Code, tickets, and presentations may reference these docs, but they must not become a second Source of Truth.

## Operating rule

Cornerstone is **docs-first and spec-driven**.

That means:
- no meaningful feature work starts without an owning spec
- no change to a non-replaceable foundation lands without an updated decision record
- no transport, framework, or implementation choice may redefine the canonical product contract

## Command surface

- `./run-all.sh up`: start the local product stack.
- `./run-all.sh down`: stop the local product stack.
- `./run-all.sh check`: run the default local quality gate in one shot.
- `./run-all.sh check --with-corpus`: run the default gate plus the opt-in full corpus smoke.

The default quality gate currently runs:

- `make lint`
- `make typecheck`
- `make backend-fast`
- `make backend-integration`
- `make symptoms`
