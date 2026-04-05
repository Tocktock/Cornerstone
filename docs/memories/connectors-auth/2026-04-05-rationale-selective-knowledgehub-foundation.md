---
date: 2026-04-05
feature: connector-auth-foundation
type: rationale
related_specs:
  - /docs/specs/connector-auth-foundation/spec.md
related_decisions:
  - /docs/decisions/0002-selectively-adopt-knowledgehub-connector-and-auth-foundation.md
status: active
---

# Selective KnowledgeHub connector and auth foundation adoption

## Context

Cornerstone needs better Google and Notion connector support, plus stronger handling of authentication and user information.

A comparison against `KnowledgeHub` showed that the current Cornerstone implementation is still filesystem-only in practice and does not yet define first-class user, session, or OAuth-state records. `KnowledgeHub` already has a stronger connector and auth foundation, but it also carries broader workspace, invitation, and admin-console behavior that Cornerstone may not need.

## Decision or observation

The chosen direction is to adopt the `KnowledgeHub` foundation selectively:

- reuse the auth, session, OAuth-state, and secure-token concepts
- reuse provider readiness and connector account state for Google Drive and Notion
- keep Cornerstone’s current filesystem connector and curation model in place during the migration
- avoid copying the full `KnowledgeHub` workspace product model unless a later decision explicitly requires it

## Impact

- Connector work in Cornerstone now has a documented architectural target instead of an ad hoc “add Notion and Google somehow” direction.
- Secure token handling is explicitly part of the required scope.
- Future implementation should add backend schema, services, tests, and frontend session surfaces in line with the new feature spec.

## Follow-up

- Implement the backend user/session and connector foundation described in `/docs/specs/connector-auth-foundation/spec.md`.
- Add secure token encryption support before persisting live Google Drive or Notion credentials.
