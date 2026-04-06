# 0009 - Access uses base roles plus scoped capabilities

- **Status:** Accepted

## Context

The product needs one permission model that can explain connector setup, authoring, review, and domain responsibility without inventing a second hidden set of actors.

## Decision

Cornerstone uses base roles (`owner`, `admin`, `member`) plus scoped capabilities (`manage_connectors`, `operate`, `review`, `own_domain`). Human-facing labels are derived from those capabilities.

## Consequences

- Owners and admins implicitly hold workspace-wide connector, authoring, and review capabilities.
- Shared connector management is distinct from general authoring.
- Domain ownership routes accountability but does not replace review permission.
