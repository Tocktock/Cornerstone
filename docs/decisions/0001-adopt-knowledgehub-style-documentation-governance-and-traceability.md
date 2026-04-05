# 0001 - Adopt KnowledgeHub-style documentation governance and traceability

## Status

Accepted

## Decision

Cornerstone adopts the same documentation governance model used in `KnowledgeHub`.

The durable repository rule is:

- `docs/specs/` is the canonical home for feature behavior and requirements
- `docs/decisions/` stores durable architectural choices and technical invariants
- `docs/memories/` stores conversations, intentions, deprecations, rationale, and implementation notes
- `docs/sot/` and the existing `docs/spec/` files remain valid high-level references until they are migrated, but new feature-level Source of Truth content should not be added there

## Why

Cornerstone already has a spec-driven development document and several high-level product and domain references, but it does not yet have a formal place for:

- conversation outcomes
- rationale history
- durable architectural decisions
- feature-first specs for new work

The repository also needs an explicit rule that today’s conversation and future design decisions are written into repository documents rather than living only in chat history.

Using the same governance model as `KnowledgeHub` avoids inventing a second documentation standard for closely related work.

## Consequences

- New feature work must create or update `docs/specs/<feature>/spec.md`.
- Durable architectural or repository rules must be documented in `docs/decisions/`.
- Design conversations, implementation rationale, and deprecations must be recorded in `docs/memories/`.
- Existing `docs/sot/` and `docs/spec/` content can continue to inform the system until migrated, but they are no longer the place for new feature-level Source of Truth documents.
