# 20 — Live PostgreSQL Verification and Mypy Stability v0.9.1

## Goal

v0.9.1 turns the v0.9.0 PostgreSQL/concurrency assets into a stricter verification release.

The product invariant is unchanged:

```text
A sync job may create trusted Artifact/Evidence state only when database persistence, worker ownership, cursor advancement, and audit state are proven consistent.
```

## Implemented changes

### Strict live PostgreSQL runner

Added:

```bash
python scripts/run_live_postgres_tests.py
```

The runner fails closed unless:

```text
RUN_POSTGRES_TESTS=1
DATABASE_URL uses a PostgreSQL scheme
```

It runs:

```text
alembic upgrade head
scripts/check_postgres_extensions.py
pytest tests/postgres -m postgres
```

It then parses the pytest terminal report and rejects a supposedly live PostgreSQL run when tests are skipped, failed, errored, or below the expected pass count.

### Local test script no longer reports intentional PostgreSQL skips

`./scripts/run_tests.sh` now ignores `tests/postgres` unless `RUN_POSTGRES_TESTS=1` is explicitly set.

This makes local reports cleaner:

```text
normal local run: non-PostgreSQL tests only
explicit live run: PostgreSQL tests must run and must not skip
```

### Stable mypy execution

`./scripts/run_tests.sh` now runs mypy with:

```text
--no-incremental
--cache-dir reports/.mypy_cache
timeout guard
```

This avoids stale cache effects and makes the mypy report deterministic in CI/local environments where mypy is installed.

### Live PostgreSQL coverage expanded

Added live PostgreSQL tests for:

```text
worker heartbeat owner enforcement
worker heartbeat lease extension
duplicate scheduled enqueue key rejection
```

Existing live tests still cover:

```text
required extensions
migration head
concurrent single-claim behavior
expired lease reclaim behavior
```

### CI workflow tightened

GitHub Actions now runs the strict live PostgreSQL runner first, then runs the non-PostgreSQL backend checks separately.

## Verification commands

Local backend verification without PostgreSQL:

```bash
./scripts/run_tests.sh
```

Strict live PostgreSQL verification:

```bash
docker compose up -d postgres
export RUN_POSTGRES_TESTS=1
export DATABASE_URL=postgresql+psycopg://cornerstone:cornerstone@localhost:5432/cornerstone
python scripts/run_live_postgres_tests.py
```

Combined PostgreSQL CI path:

```bash
./scripts/run_postgres_ci.sh
```

## Sandbox note

This environment does not provide a running PostgreSQL server or Docker daemon, so the strict live PostgreSQL runner is included but not executed to success here. The non-PostgreSQL test suite, mypy, ruff, compile, coverage, and offline Alembic checks are executable in this sandbox.

## Remaining risks

```text
1. Live PostgreSQL tests still need to be executed in Docker/GitHub Actions before production confidence.
2. Production KMS/secret-manager integration remains deferred.
3. Notion database/data_source ingestion remains deferred.
4. Deployment-specific queue orchestration is still outside the backend package.
```
