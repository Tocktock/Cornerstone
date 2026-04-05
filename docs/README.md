# Cornerstone Documentation

This folder is the canonical documentation home for the repository.

## Documentation model

- [`specs/`](./specs/): canonical feature behavior and product requirements for new feature-level work
- [`decisions/`](./decisions/): architectural decisions and durable technical rules
- [`memories/`](./memories/): conversations, intentions, deprecations, and rationale history
- [`sot/`](./sot/): product identity, goals, and non-negotiable framing
- [`spec/`](./spec/): legacy high-level product and domain references that remain useful until migrated

## Source of Truth rules

- Feature behavior must be documented in `docs/specs/`.
- Architectural choices and long-lived invariants belong in `docs/decisions/`.
- Traceability and rationale belong in `docs/memories/`.
- `docs/sot/` and existing `docs/spec/` files remain reference inputs, but new feature-level Source of Truth content should not be added there.
- Files outside `docs/` may summarize and link, but they must not restate detailed feature behavior as a second Source of Truth.

## Current feature specs

- [`connector-auth-foundation`](./specs/connector-auth-foundation/spec.md)

## Existing high-level references

- [`sot/PROJECT_SOT.md`](./sot/PROJECT_SOT.md)
- [`sot/WHY_AND_GOALS.md`](./sot/WHY_AND_GOALS.md)
- [`spec/PRODUCT_SPEC.md`](./spec/PRODUCT_SPEC.md)
- [`spec/DOMAIN_MODEL.md`](./spec/DOMAIN_MODEL.md)
- [`spec/SPEC_DRIVEN_DEVELOPMENT.md`](./spec/SPEC_DRIVEN_DEVELOPMENT.md)

## Required maintenance workflow

1. Update the relevant feature spec before or during implementation.
2. Record the intent, rationale, conversation outcome, or deprecation note in `docs/memories/` when behavior or direction changes.
3. Add or update a decision record in `docs/decisions/` when the change introduces a durable architectural or governance rule.
4. Keep high-level READMEs short and link back here instead of duplicating feature detail.
