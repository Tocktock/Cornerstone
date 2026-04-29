# Backend Release Checklist

Use this checklist before tagging `v1.0.0-rc.1` or `v1.0.0`.

## Version and package

```text
[ ] pyproject.toml version matches release version.
[ ] src/cornerstone/__init__.py version matches release version.
[ ] /healthz returns the expected version.
[ ] Release ZIP contains source, migrations, scripts, docs, tests, and sanitized reports only.
[ ] Release ZIP contains no __pycache__, .pyc, .pytest_cache, .mypy_cache, .ruff_cache, .coverage, .venv, or .env files.
```

## Static checks

```text
[ ] ./scripts/run_tests.sh passes.
[ ] python -m mypy src --show-error-codes --no-color-output --no-incremental passes.
[ ] python scripts/check_release_candidate.py passes.
[ ] Alembic offline SQL renders.
[ ] OpenAPI snapshot tests pass.
```

## Live PostgreSQL proof

```text
[ ] Docker PostgreSQL with pgvector starts.
[ ] Alembic upgrade head succeeds on live PostgreSQL.
[ ] pgcrypto extension exists.
[ ] citext extension exists.
[ ] vector extension exists.
[ ] Live PostgreSQL worker concurrency tests pass.
[ ] Live PostgreSQL summary shows skipped=0, failed=0, errors=0.
```

## Live Notion proof

```text
[ ] Notion token is provided through environment only.
[ ] Test page is shared with the integration.
[ ] Live Notion E2E runner passes.
[ ] Live Notion gated pytest passes with 0 skipped tests.
[ ] At least one Artifact is created.
[ ] At least one EvidenceFragment is created.
[ ] Source next action is review_evidence.
```

## Product loop proof

```text
[ ] Evidence review queue shows unreviewed evidence.
[ ] Authorized reviewer can mark evidence reviewed.
[ ] Unauthorized reviewer is rejected with 403.
[ ] Concept candidate can be created from reviewed evidence.
[ ] Concept can become official.
[ ] Unauthorized officialization is rejected with 403.
[ ] Grounded query returns trustLabel=official with valid citation.
[ ] Unsupported query returns trustLabel=unsupported.
[ ] Evaluation task succeeds.
[ ] groundedContextTaskSuccessRate is computed and maps to grounded_context_task_success_rate.
```

## Source API safety

```text
[ ] Direct Notion source creation returns 409.
[ ] Fake OAuth completion route returns 404.
[ ] Legacy generic source sync returns 404.
[ ] Manual sync on Notion source returns 409.
[ ] Weak evaluation task returns 422.
```

## Documentation

```text
[ ] Operator runbook is current.
[ ] Known limitations are current.
[ ] Production deployment checklist is current.
[ ] Secrets and credential handling checklist is current.
[ ] API freeze review is current.
[ ] Live proof artifact template is current.
[ ] v1.0.0 readiness doc is current.
```

## Release decision

```text
[ ] All P0 release gates passed.
[ ] Any P1 gaps are documented as known limitations.
[ ] No new feature work is required to prove backend MVP.
[ ] Release owner approves v1.0.0-rc.1 or v1.0.0 tag.
```

