# 0002 - Context is workspace-scoped

- **Status:** Accepted

## Context

Organizational meaning changes by team and organization. A global unbounded truth layer would break provenance, ownership, and review responsibility.

## Decision

Shared Cornerstone context is always workspace-scoped. Personal context is allowed only as a secondary private layer and does not directly publish shared official outputs.

## Consequences

- All shared canonical objects belong to one workspace.
- Cross-workspace leakage is not allowed by default.
- Personal context must remain separate until explicit promotion.
