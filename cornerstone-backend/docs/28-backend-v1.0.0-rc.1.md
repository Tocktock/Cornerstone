# v1.0.0-rc.1 — Backend MVP Release Candidate

## Purpose

`v1.0.0-rc.1` is the first backend MVP release-candidate tag for Cornerstone.

This release candidate is intentionally based on the **same verified commit as v0.13.1**. The code version markers remain `0.13.1` because v0.13.1 is the blocker-fixed implementation that passed the release gate. The `v1.0.0-rc.1` tag is the release-candidate label applied to that verified commit.

## No new backend feature work

No new backend feature work is introduced for RC-1. The release candidate freezes the live-proven backend MVP loop:

```text
Live PostgreSQL
→ live Notion page
→ Artifact
→ EvidenceFragment
→ evidence review
→ official Concept
→ grounded context response
→ evaluation result
→ grounded_context_task_success_rate
```

## Evidence used for RC-1

RC-1 is supported by:

```text
docs/live-proof-records/2026-04-27-change-log.md
docs/live-proof-records/2026-04-28-v0.13.1-blocker-fix.md
```

The v0.13.1 blocker-fix record confirms:

```text
local non-live gate: 230 passed
coverage: 85%
ruff: passed
mypy: passed, 55 source files
compileall: passed
release-candidate check: passed for v0.13.1
live PostgreSQL: 5 passed, 0 skipped
live Notion: passed
runnerArtifactCount: 1
runnerEvidenceFragmentCount: 5
groundedContextTaskSuccessRate: 1.0
safety negative checks: passed
secret scan: no Notion token pattern found
```

## RC-1 acceptance criteria

RC-1 is acceptable only if:

```text
1. v1.0.0-rc.1 points to the same verified commit as v0.13.1.
2. The worktree is clean before tagging.
3. Clean checkout verification passes.
4. Live PostgreSQL verification passes with zero skipped tests.
5. Live Notion E2E passes with a real shared Notion page.
6. The product-loop API proof passes.
7. Safety negative checks pass.
8. Secret scan passes.
9. Human sign-off is recorded.
```

## Known limitations carried into RC-1

RC-1 is backend MVP scope, not full product GA. These remain intentionally deferred:

```text
Slack connector
Google Docs connector
GitHub connector
Notion database/data_source ingestion
runtime vector retrieval
frontend UI
enterprise SSO/RBAC
LLM-graded evaluation
```

## Decision rule

If RC-1 verification finds no P0/P1 blocker, proceed to backend `v1.0.0`.

If a blocker appears, patch narrowly and tag `v1.0.0-rc.2`.
