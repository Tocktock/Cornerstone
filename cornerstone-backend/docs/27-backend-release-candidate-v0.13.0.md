# v0.13.1 — Backend Release-Candidate Cleanup

## Goal

v0.13.1 is a release-candidate cleanup version. It does not add new product features. It packages the operational material needed to move from the live-proof backend to a backend MVP release candidate.

The backend release gate remains:

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

This release-candidate cleanup is grounded in the live proof recorded in:

```text
docs/live-proof-records/2026-04-27-change-log.md
```

## What changed

Added release-oriented documentation and checks:

```text
docs/release/backend-operator-runbook.md
docs/release/backend-release-checklist.md
docs/release/known-limitations.md
docs/release/production-deployment-checklist.md
docs/release/secrets-and-credential-handling.md
docs/release/api-freeze-review.md
docs/release/live-proof-artifact-template.md
docs/release/v1.0.0-readiness.md
scripts/check_release_candidate.py
```

Updated version markers:

```text
pyproject.toml → 0.13.1
src/cornerstone/__init__.py → 0.13.1
README.md → v0.13.1 focus and release-candidate guidance
```

## Why this version exists

v0.12.2 packaged the live-proof fixes. v0.13.1 makes the proof repeatable and reviewable by a backend operator or release reviewer.

This is intentionally a cleanup version because the product has already proven the first backend MVP loop with one live Notion page and live PostgreSQL. The remaining work before v1.0.0 is mostly release discipline:

```text
- repeatable runbook
- explicit limitations
- API freeze review
- secret handling expectations
- production deployment checks
- final release checklist
```

## Non-goals

v0.13.1 does not implement:

```text
- Slack connector
- Google Docs connector
- GitHub connector
- Notion database/data_source ingestion
- vector retrieval runtime
- full enterprise RBAC/SSO
- frontend UI
- LLM-graded evaluation
```

Those are post-backend-MVP items unless release requirements change.

## Release-candidate acceptance criteria

v0.13.1 is ready to become a v1.0.0 backend release candidate only if:

```text
1. Local non-live checks pass.
2. Live PostgreSQL verification passes with zero skipped tests.
3. Live Notion E2E proof passes with a real shared Notion page.
4. Evidence review and officialization pass.
5. Grounded query returns official with valid citation.
6. Evaluation task succeeds.
7. Safety negative checks pass.
8. Known limitations are documented and accepted.
9. API freeze review is complete.
10. No secrets are included in reports, docs, commits, or release ZIPs.
```

## Recommended next step

Run the backend release checklist in:

```text
docs/release/backend-release-checklist.md
```

If the checklist passes without code changes, tag:

```text
v1.0.0-rc.1
```

If any proof-blocking issue appears, patch narrowly as:

```text
v0.13.1
```



## v0.13.1 blocker patch and RC-1 note

The original v0.13.0 checkout did not pass the static release-candidate checker unchanged after the documented local verification workflow generated runtime artifacts. v0.13.1 is a narrow release-tooling patch that keeps the backend product behavior unchanged and allows local-only generated artifacts while still rejecting secrets and packaged cache/build noise.

`v1.0.0-rc.1` should be tagged on the same verified commit as v0.13.1 after human release sign-off.
