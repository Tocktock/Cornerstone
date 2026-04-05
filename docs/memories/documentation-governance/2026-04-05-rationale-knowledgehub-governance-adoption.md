---
date: 2026-04-05
feature: documentation-governance
type: rationale
related_specs:
  - /docs/spec/SPEC_DRIVEN_DEVELOPMENT.md
related_decisions:
  - /docs/decisions/0001-adopt-knowledgehub-style-documentation-governance-and-traceability.md
status: active
---

# KnowledgeHub-style documentation governance adoption

## Context

Cornerstone already had spec-driven intent in `docs/spec/SPEC_DRIVEN_DEVELOPMENT.md`, but it did not have a repository-level rule that captured where feature specs, durable decisions, and rationale history must live.

The immediate trigger was a request that all conversation and decision outcomes be written into repository documents rather than living only in chat history.

## Decision or observation

Cornerstone now adopts the same documentation governance pattern used in `KnowledgeHub`:

- `docs/specs/` for feature behavior
- `docs/decisions/` for durable architecture and repository rules
- `docs/memories/` for traceability, conversations, and rationale

`docs/sot/` and the existing `docs/spec/` files remain valid legacy references until migrated, but new feature-level Source of Truth content should move into the new structure.

## Impact

- The repository now has a stable place to record conversation outcomes and design rationale.
- Future implementation work can be tied to specs and decisions without overloading README files or chat logs.
- The repo-level `AGENTS.md` now matches the documented folder model instead of relying on informal process expectations.

## Follow-up

- Migrate legacy feature-level material from `docs/spec/` into `docs/specs/` when those areas are actively changed.
- Keep `docs/README.md` up to date as new feature specs and decisions are added.
