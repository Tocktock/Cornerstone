# Cornerstone Backend v0.12.2 Test Run Summary

## Scope

v0.12.2 packages the fixes discovered during manual live PostgreSQL and live Notion proof.

## Sandbox verification

```text
pytest non-live suite: 226 passed
coverage total: 85%
ruff: All checks passed
mypy: Success: no issues found in 55 source files
compileall: passed
Alembic offline SQL: rendered
```

## Live proof record

The user-run live proof passed before this package was created and is recorded in:

```text
docs/live-proof-records/2026-04-27-change-log.md
```

That proof included:

```text
Live PostgreSQL verification: 5 passed, 0 skipped
Live Notion page E2E: status=passed, artifact_count=1, evidence_fragment_count=4
Grounded context: trustLabel=official, invalidCitationCount=0
Evaluation: success=true, groundedContextTaskSuccessRate=1.0
Safety negatives: fake provider source=409, fake OAuth=404, legacy sync=404, Notion manual sync=409
```

## Live tests in this sandbox

Live PostgreSQL and live Notion were not rerun in this sandbox. They require Docker/PostgreSQL and a real Notion token/page ID.
