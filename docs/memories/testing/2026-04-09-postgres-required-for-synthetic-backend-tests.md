# 2026-04-09 - Postgres required for synthetic backend tests

## Context

Cornerstone’s production database is PostgreSQL, and the repository test stack already provisions PostgreSQL for synthetic backend verification.

Some local test paths still tolerated SQLite:
- the shared pytest fixture accepted non-Postgres URLs
- one domain test module created in-memory SQLite databases directly

That drift made it too easy to validate transport and domain behavior against a different SQL dialect than the one the product actually runs.

## Decision

Synthetic backend tests must use PostgreSQL.

Concretely:
- the shared pytest test database fixture now requires a PostgreSQL URL
- the default local pytest database URL matches the Compose test stack port
- SQLite is no longer accepted as a synthetic backend test database
- domain tests that previously created in-memory SQLite engines now run on the shared PostgreSQL-backed test fixture

## Why it matters

- contract and integration verification should exercise the same database family as production
- backend synthetic tests should fail clearly when PostgreSQL is unavailable instead of silently drifting to a different backend
- a single Postgres-backed path keeps local commands, CI expectations, and root testing guidance aligned
