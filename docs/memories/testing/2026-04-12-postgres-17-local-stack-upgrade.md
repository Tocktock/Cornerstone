# 2026-04-12 - Postgres 17 local stack upgrade

## Context

Cornerstone already treats PostgreSQL as the canonical database for both local synthetic verification and normal local stack operation.

The Docker definitions were still pinned to `postgres:16-alpine` in:
- `compose.yml`
- `compose.test.yml`

## Decision

Both local Docker stacks now target `postgres:17-alpine`.

This applies to:
- the normal local app stack
- the synthetic test stack used by `make backend-fast`, `make backend-integration`, and `make symptoms`

## Operator impact

This is a major Postgres image upgrade.

Existing Docker volumes initialized by Postgres 16 are not migrated in place by Cornerstone. The local operator recovery path remains explicit:
- `./run-dev.sh up --reset-db`
- `./run-prod.sh up --reset-db`
- `make test-stack-down` before recreating the test stack

That keeps the upgrade reversible and avoids hiding destructive volume resets behind normal startup.

The launcher now also removes stale service containers that clearly drifted away from the active Compose project state, such as:
- a DB container still pinned to `postgres:16-alpine`
- a service container that no longer belongs to the project default network

That repair is container-only. Volume reset remains explicit.

## Failure modes observed during verification

Two concrete local failure modes were reproduced after the upgrade:

- a Postgres volume initialized by version 16 failed with the expected compatibility error from PostgreSQL 17
- a stale Compose-labelled container that had lost the project network caused backend startup to fail because the `db` hostname could not be resolved

The launcher repair now handles the stale-container case automatically before startup. The major-version data-directory mismatch remains an explicit operator reset path because Cornerstone does not silently delete local database volumes.

## Verified recovery

The upgrade path was re-verified with:
- `make test-stack-down`
- `make test-stack-up`
- `make test-stack-down`
- `./run-prod.sh up --reset-db -d`

After the reset path, the production-profile stack recreated the DB on `postgres:17-alpine`, `/api/v1/health` returned `{\"status\":\"ok\"}`, and `/api/v1/bootstrap` returned a clean production onboarding posture.
