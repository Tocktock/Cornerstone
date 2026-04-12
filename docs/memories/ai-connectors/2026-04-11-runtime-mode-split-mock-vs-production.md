# Runtime Mode Split: Mock vs Production

## Summary

Cornerstone now has one backend-owned runtime mode with two values:
- `mock`
- `production`

This change does not alter concept, decision, provenance, lineage, access, review, or trust semantics. It changes only boot behavior, connector demo fallback policy, and the frontend empty-state posture for production workspaces.

## Why this was introduced

The prior local-first behavior mixed two separate concerns:
- minimal workspace bootstrap so the app can start
- demo knowledge and demo datasource behavior so the product can be explored without setup

That was acceptable for mock environments, but it became unsafe for production because the UI could appear healthy while silently relying on demo bootstrap or demo provider binding.

## Implemented policy

- `runtime_mode=mock`
  - minimal workspace bootstrap is created
  - demo content seeding may run
  - demo Notion binding and fixture sync remain available
- `runtime_mode=production`
  - only minimal workspace bootstrap is created
  - demo content seeding is disabled
  - demo Notion binding is disabled
  - missing OAuth configuration returns a clear configuration error

## Frontend implication

The frontend now reads additive bootstrap metadata:
- `runtime_mode`
- `workspace_data_state`
- `linked_source_count`
- `active_source_count`
- `degraded_source_count`

That metadata controls the production UX:
- `awaiting_sources` shows guided empty states instead of demo content
- `syncing_sources` shows first-sync guidance
- `degraded` preserves visible content but surfaces recovery cues

## Why demo fallback is blocked in production

Production must never present synthetic confidence.

If production silently falls back to demo content or demo provider credentials:
- operators can believe a workspace is connected when it is not
- members can read fabricated placeholder knowledge as if it were real shared context
- connector setup failures become harder to diagnose because the failure is masked by fixture behavior

Blocking demo fallback in production keeps operational failures explicit and keeps member-facing knowledge honest.

## Local Docker operation

The local Docker stack now forwards `CORNERSTONE_RUNTIME_MODE` through `compose.yml`, and `.env.example` exposes `CORNERSTONE_RUNTIME_MODE=mock` as the default local posture.

That means local mode switching no longer needs to depend on hand-editing `.env` for normal use:
- `./run-dev.sh` forces the mock/dev runtime profile
- `./run-prod.sh` forces the production-like runtime profile

The generic `./run-all.sh` launcher still exists for environment-driven operation, but no frontend toggle exists and the backend remains the canonical source of truth.

The operator-facing local files now also enumerate the valid runtime-mode options explicitly:
- `mock`
- `production`
