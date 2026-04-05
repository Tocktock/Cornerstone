# 0002 - Selectively adopt KnowledgeHub connector and auth foundation

## Status

Accepted

## Decision

Cornerstone will selectively adopt the connector and authentication foundation from `KnowledgeHub` rather than copying the full `KnowledgeHub` product model.

The durable direction is:

- bring explicit user and session handling into Cornerstone
- bring OAuth state handling and secure connector token storage
- bring provider readiness and connector account state for Google Drive and Notion
- keep the existing filesystem connector and current curation model operational during the migration
- map the initial adoption onto Cornerstone’s current `ContextSpace` boundary
- do not import the full multi-workspace, invitation, and admin-console product model unless a later accepted decision requires it

## Why

Cornerstone’s current sync path is effectively filesystem-only and lacks first-class authenticated user context, OAuth state handling, and secure connector token storage.

That is the direct reason Google Drive and Notion connectivity is currently hard to support safely and predictably.

`KnowledgeHub` already has a working architectural direction for:

- session-backed user identity
- provider OAuth state
- encrypted connector tokens
- provider readiness and connection status

Those pieces are valuable to Cornerstone. The full multi-workspace product model is not automatically valuable and would create avoidable complexity if copied wholesale.

## Consequences

- Cornerstone will need backend schema and service changes for users, sessions, OAuth state, and connector accounts.
- Secure token storage becomes a first-class requirement rather than an optional future improvement.
- Frontend work should expose signed-in user state and connector readiness rather than only filesystem source rows.
- Further provider browse, resource selection, and deeper sync behavior can follow after the auth and connection foundation is in place.
