# 0011 - Personal sources remain separate until explicitly promoted into shared context

- **Status:** Accepted

## Context

Personal sources can be valuable preparation inputs, but they are a privacy boundary and cannot be treated as shared workspace truth by default.

## Decision

Personal source content remains in personal context until a personal-source owner explicitly creates `PromotedSupport` for a target workspace. `PromotedSupport` is the only shared object that may be created directly from personal context.

## Consequences

- Shared official outputs may cite `PromotedSupport`, not personal artifacts or evidence directly.
- Workspace actors do not automatically gain access to the underlying personal source.
- Promotion must preserve disclosure level and lineage.
