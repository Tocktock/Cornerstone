# Decisions and Project Memory

This folder stores architectural decisions and durable technical memory.

Feature behavior lives in [`../specs/`](../specs/), and traceability history lives in [`../memories/`](../memories/).

## Current decisions

- [0001 - Adopt KnowledgeHub-style documentation governance and traceability](./0001-adopt-knowledgehub-style-documentation-governance-and-traceability.md)
- [0002 - Selectively adopt KnowledgeHub connector and auth foundation](./0002-selectively-adopt-knowledgehub-connector-and-auth-foundation.md)

## How to use this folder

- Add a new numbered file for any lasting architectural or repository-governance choice.
- Prefer writing down why a rule exists, not just what the rule is.
- Do not use this folder as the canonical home for feature behavior. Put feature behavior in `docs/specs/` instead.
