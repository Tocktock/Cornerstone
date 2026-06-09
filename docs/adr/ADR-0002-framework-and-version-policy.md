# ADR-0002 - Framework and Version Policy for VS-0 Scaffold

**Date:** 2026-06-09
**Status:** Accepted as setup-planning authority; no runtime implementation yet.
**Owner:** JiYong / Tars

## Context

CornerStone needs a zero-base scaffold before product feature coding starts. The scaffold must remain scenario-first, CLI-native-first, local/on-prem-friendly, and compatible with the product SoT.

The setup decision must not be "latest at any cost." It must be latest compatible, pinned, and verifiable.

## Decision

Target this baseline for the VS-0 scaffold:

| Area | Target |
|---|---|
| Backend language | Python 3.14.x |
| Frontend runtime | Node.js 24.x LTS |
| Database | PostgreSQL 18.x |
| Backend API | FastAPI 0.136.3 |
| CLI framework | Typer 0.26.7 |
| Data validation | Pydantic 2.13.4 |
| ORM / DB layer | SQLAlchemy 2.0.50 |
| DB migrations | Alembic 1.18.4 |
| Postgres driver | psycopg 3.3.4 |
| Python package manager | uv 0.11.19 |
| Python lint/format | Ruff 0.15.16 |
| Python test | pytest 9.0.3 |
| Python typing | mypy 2.1.0 |
| Frontend framework | Next.js 16.2.7 |
| UI library | React 19.2.7 |
| TypeScript | 6.0.3 candidate; fallback to 5.9.x if Next typecheck/build fails |
| JS package manager | pnpm 11.5.2 |
| Keyword search | PostgreSQL full-text search |
| Vector search | pgvector 0.8.2 |
| Policy | OPA/Rego-compatible policy engine, OPA 1.17.1 target |
| Tool sandbox direction | Wasmtime 45.0.1 target after VS-0 boundaries stabilize |

## Version Policy

- Runtime pins: Python 3.14.x, Node 24.x LTS, PostgreSQL 18.x, pgvector 0.8.2.
- Commit lockfiles when scaffold files are introduced: `uv.lock` and `pnpm-lock.yaml`.
- Use exact lockfile versions for release verification.
- Major upgrades require a scenario contract, CLI transcript verification, and owner approval.
- New production dependencies require explicit approval before implementation.
- TypeScript 6.0.3 is allowed only if `pnpm typecheck` and `pnpm build` pass. Otherwise, pin TypeScript 5.9.x for the first scaffold.

## Evidence

This ADR is based on the attached setup research and a fresh source check on 2026-06-09:

- Python Developer's Guide lists Python 3.14 as bugfix/stable and Python 3.15 as prerelease.
- Node.js release guidance says production applications should use Active LTS or Maintenance LTS; Node 24 is LTS while Node 26 is Current.
- PostgreSQL current docs are PostgreSQL 18.4.
- PyPI, npm registry, pgvector tags, OPA releases, and Wasmtime releases were checked for the package targets above.

## Consequences

Positive:

- The scaffold starts from modern versions without using prerelease runtimes as the production baseline.
- Lockfiles become the release truth for dependency versions.
- Version upgrades remain scenario-verifiable instead of ad hoc.

Costs:

- Some package targets may require fallback if local build/typecheck evidence fails.
- Python 3.14 and Node 24 require developer machines and CI images that can support them.

## Non-Decision

This ADR does not add dependencies, lockfiles, runtime code, Docker services, or generated scaffold files. It is setup-planning authority only.
