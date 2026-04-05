# Connector Auth Foundation

**Status:** Draft  
**Type:** Feature Specification  
**Last Updated:** 2026-04-05

---

## Summary

Cornerstone will add a first-class authenticated connector foundation for Google Drive and Notion, backed by explicit user identity, session handling, secure OAuth token storage, and provider-specific connection state.

The goal is to reuse the proven architectural direction from `KnowledgeHub` without importing its full multi-workspace product surface into Cornerstone.

---

## Background / Problem

Cornerstone currently supports a filesystem-only source connection model and seeds demo data directly from a local path.

That is enough for demo ingestion, but it does not provide:

- authenticated user identity
- session-backed permissions
- OAuth state management
- secure connector token storage
- provider readiness or re-auth state
- a stable way to connect Google Drive or Notion

This leaves Notion and Google integration fragile and makes user/account handling effectively implicit rather than explicit.

---

## Goals

- Add explicit authenticated user and session handling to the backend.
- Provide a provider-neutral connector account model for Google Drive and Notion.
- Persist connector OAuth state and account metadata safely.
- Encrypt provider access and refresh tokens at rest.
- Expose connector readiness and connection status through stable backend APIs.
- Preserve the current filesystem connector while authenticated connectors are introduced.
- Keep the initial adoption compatible with Cornerstone's existing `ContextSpace` boundary.

---

## Non-goals

- Rebuilding all of `KnowledgeHub`'s multi-workspace, invitation, and admin-console product model in the first step.
- Replacing the existing filesystem connector immediately.
- Achieving full browse, selection, and sync parity with `KnowledgeHub` in the same change as the auth foundation.
- Making Notion an application login provider.

---

## Primary users

- knowledge operators who connect and maintain source systems
- reviewers and domain owners who need correct user identity and permissions
- administrators who need auditable connector ownership and health state
- downstream AI consumers that depend on stable provenance and user-scoped operations

---

## Key workflows

### 1. Sign in

A human user signs in, receives a session token, and the backend can identify the current user on future requests.

### 2. Start connector OAuth

An authenticated user starts a Google Drive or Notion OAuth flow from Cornerstone. The backend records state, intended return path, and ownership context before redirecting the user to the provider.

### 3. Complete connector OAuth

The provider callback is validated against stored OAuth state. Cornerstone exchanges the authorization code, records provider account metadata, encrypts tokens, and stores the connector connection.

### 4. View readiness and connection state

The frontend can ask which providers are configured, which connections already exist, and whether a connection needs attention or re-authentication.

### 5. Continue using existing filesystem ingestion

The current filesystem connection remains available during the migration so demo ingestion and current review workflows do not break while authenticated connectors are being introduced.

---

## Permissions and visibility

- Only authenticated users may start or complete connector OAuth flows.
- Shared connector management must be limited to authorized operators or administrators.
- Regular consumers may read synced knowledge without handling provider credentials directly.
- User-facing UI should show provider labels and connection state, not raw OAuth or token internals.

---

## Requirements

1. The backend must define first-class user and user-session records.
2. The backend must hash session tokens server-side and never persist raw session tokens.
3. The backend must store connector OAuth state records with expiration and intended return path.
4. The backend must define connection rows that hold provider, owner, account identity, status, scopes, and token expiry metadata.
5. Provider access and refresh tokens must be encrypted at rest.
6. The backend must provide a stable authenticated-user endpoint for frontend session hydration.
7. The backend must provide a readiness endpoint that reports provider configuration and connection setup state.
8. The initial provider set must include `google_drive` and `notion`.
9. The existing filesystem connector path must remain operable while the new foundation is introduced.

---

## Non-replaceable invariants

- Notion is a connector provider, not an application login provider.
- Connector tokens must be encrypted at rest.
- Session tokens must be hashed server-side.
- Connector state must be explicit and queryable rather than inferred from ad hoc settings blobs.
- The provider abstraction must stay replaceable so additional connector providers can be added later.
- The initial Cornerstone adoption should map onto the existing `ContextSpace` boundary unless a separate accepted decision introduces a broader workspace model.

---

## Replaceable choices

- Whether Google app login and Google Drive connector ownership share one account row or separate rows.
- Whether shared connector ownership is mapped directly to `ContextSpace` or later generalized to a separate workspace model.
- The exact frontend information architecture for connection setup and connection detail pages.
- The specific sync scheduling defaults for authenticated providers.

---

## Risks

- Pulling too much of `KnowledgeHub` wholesale would add product complexity Cornerstone may not need.
- Introducing OAuth without proper encryption would create security debt instead of solving the current problem.
- Mapping roles from existing `Actor` records to authenticated human users may require a deliberate migration rule.
- Database migrations will need to preserve current demo and filesystem behavior.

---

## Open questions

1. Should Cornerstone stay `ContextSpace`-scoped or later adopt a separate workspace model?
2. What is the cleanest mapping between existing reviewer/operator `Actor` rows and authenticated users?
3. Should Google be both the app-login provider and the Google Drive connector account, or should those concerns stay separate?
4. What is the minimum provider browse or selection UX required for the first useful Google Drive and Notion release?

---

## Rollout / Verification

- Add schema support for users, sessions, OAuth state, and connector connections.
- Add backend tests for auth session resolution, OAuth start/callback behavior, and provider readiness.
- Add frontend session hydration and connector status rendering.
- Keep filesystem sync integration tests green during the migration.
- Verify that callback and re-auth failure states are explicit and user-visible.
