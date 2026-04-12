# Cornerstone Documentation

This package contains the **canonical product documentation** for Cornerstone.

Cornerstone is the **shared organizational context layer for humans and AI**. These docs define what Cornerstone is, what it is not, which product foundations are fixed, and which functional contracts must exist before implementation begins.

## Reading order

Read in this order if you are new to the project:

1. [docs/specs/system-overview/spec.md](./docs/specs/system-overview/spec.md)
2. [docs/specs/foundations/spec.md](./docs/specs/foundations/spec.md)
3. [docs/specs/ontology/spec.md](./docs/specs/ontology/spec.md)
4. [docs/specs/state-vocabulary/spec.md](./docs/specs/state-vocabulary/spec.md)
5. [docs/specs/workspace-and-access/spec.md](./docs/specs/workspace-and-access/spec.md)
6. [docs/specs/connectors/spec.md](./docs/specs/connectors/spec.md)
7. [docs/specs/sync-and-provenance/spec.md](./docs/specs/sync-and-provenance/spec.md)
8. [docs/specs/domain-model/spec.md](./docs/specs/domain-model/spec.md)
9. [docs/specs/concepts/spec.md](./docs/specs/concepts/spec.md)
10. [docs/specs/graph-and-relations/spec.md](./docs/specs/graph-and-relations/spec.md)
11. [docs/specs/decision-context/spec.md](./docs/specs/decision-context/spec.md)
12. [docs/specs/review-and-validation/spec.md](./docs/specs/review-and-validation/spec.md)
13. [docs/specs/retrieval-and-answers/spec.md](./docs/specs/retrieval-and-answers/spec.md)
14. [docs/specs/serving-contract/spec.md](./docs/specs/serving-contract/spec.md)
15. [docs/specs/ai-operator-surfaces/spec.md](./docs/specs/ai-operator-surfaces/spec.md)
16. [docs/specs/authoring-and-curation/spec.md](./docs/specs/authoring-and-curation/spec.md)
17. [docs/specs/p0/README.md](./docs/specs/p0/README.md)

## Documentation model

- [docs/specs/](./docs/specs/): canonical product and feature behavior
- [docs/decisions/](./docs/decisions/): durable product and architecture choices
- [docs/memories/](./docs/memories/): rationale and historical context that should not override specs or decisions

## Source of Truth rules

- Product and feature behavior must be defined in `docs/specs/`.
- Durable product and architecture choices belong in `docs/decisions/`.
- Rationale and historical notes belong in `docs/memories/`.
- Code, tickets, and presentations may reference these docs, but they must not become a second Source of Truth.

## Operating rule

Cornerstone is **docs-first and spec-driven**.

That means:
- no meaningful feature work starts without an owning spec
- no change to a non-replaceable foundation lands without an updated decision record
- no transport, framework, or implementation choice may redefine the canonical product contract

## Command surface

- `./run-dev.sh up`: start the local mock/dev stack with demo-friendly defaults.
- `./run-prod.sh up`: start the local production-like stack with demo fallback disabled.
- `./run-all.sh up`: start the local stack through the generic launcher. This respects the current environment and is lower-level than the explicit dev/prod launchers.
- `./run-dev.sh up --reset-db`: recreate the local dev database volume before startup. Use this after pulling a schema-breaking change such as the P0 model rewrite.
- `./run-prod.sh up --reset-db`: recreate the local production-profile database volume before startup. Use this after the Postgres 17 image upgrade if an older local prod volume already exists.
- `./run-all.sh down`: stop the local product stack.
- `./run-all.sh check`: run the default local quality gate in one shot.
- `./run-all.sh check --with-corpus`: run the default gate plus the opt-in full corpus smoke.

The backend now backfills additive local schema changes such as `decision_records.public_slug` on startup, but `--reset-db` remains the recovery path for older volumes that drift beyond those compatibility repairs or still contain Postgres 16 data files after the Postgres 17 image upgrade.

## Runtime modes

The local stack supports two backend-owned runtime mode values through `CORNERSTONE_RUNTIME_MODE`:

- `mock`: default local posture; demo content seeding and demo connector fallback may remain available
- `production`: disables demo content seeding and demo connector fallback, so shared workspaces must be populated by real linked sources

For normal local use, prefer the dedicated launchers:

- `./run-dev.sh up`
- `./run-prod.sh up`

These launchers now use separate local Compose project names and separate Postgres volumes:

- `./run-dev.sh` -> `cornerstone-dev`
- `./run-prod.sh` -> `cornerstone-prod`

That separation prevents `run-prod.sh` from inheriting demo-seeded state from the mock/dev stack.

Both the local app stack and the synthetic test stack now target `postgres:17-alpine`. The launcher removes obviously stale service containers that lost their Compose network or still pin the old Postgres 16 DB container, but if you already have a local `cornerstone-dev`, `cornerstone-prod`, or `cornerstone-test` volume from the previous Postgres 16 image, you still need to reset that stack volume before restarting it.

Use `.env` or direct environment exports only when you intentionally need the generic launcher behavior from `./run-all.sh`.

Example:

```dotenv
CORNERSTONE_RUNTIME_MODE=mock
```

```dotenv
CORNERSTONE_RUNTIME_MODE=production
```

The runtime mode is deployment-controlled and is not user-switchable in the UI. The dedicated launchers are local operator conveniences only; canonical behavior remains defined in [docs/specs/connectors/spec.md](./docs/specs/connectors/spec.md) and [docs/specs/workspace-and-access/spec.md](./docs/specs/workspace-and-access/spec.md).

The default quality gate currently runs:

- `make lint`
- `make typecheck`
- `make backend-fast`
- `make backend-integration`
- `make symptoms`

## Test backend rule

- All synthetic backend tests in this repository are Postgres-backed.
- `make backend-fast`, `make backend-integration`, and `make corpus-smoke` provision and run against the local Postgres test database via `CORNERSTONE_TEST_DATABASE_URL`.
- `./run-all.sh check` is therefore a Postgres-backed quality gate for backend synthetic verification.
- SQLite may still be useful as an ad hoc local debugging fallback, but it is not the canonical or release-signoff backend for synthetic tests in Cornerstone.

The local dev stack now uses a P0-specific Postgres volume so `./run-all.sh up` does not reuse the legacy pre-P0 schema volume by accident.
