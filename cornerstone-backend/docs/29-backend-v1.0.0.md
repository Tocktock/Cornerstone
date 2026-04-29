# v1.0.0 — Backend MVP Release

Cornerstone Backend v1.0.0 is the backend MVP release.

## Release Basis

This release promotes the verified `v1.0.0-rc.1` / `v0.13.1` line to final `v1.0.0` after clean checks, live PostgreSQL proof, live Notion proof, product-loop proof, safety negative checks, secret scan, and human release approval.

## Proven Product Loop

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

## Scope

Included:

- FastAPI backend.
- PostgreSQL persistence and Alembic migrations.
- Notion page connector path and manual source path.
- Artifact and EvidenceFragment creation.
- Evidence review queue.
- Concept, ConceptRelation, and DecisionRecord officialization gates.
- Grounded serving contract.
- Evaluation framework and `grounded_context_task_success_rate`.
- Release runbooks, live proof template, known limitations, and production deployment checklist.

Deferred:

- Notion database/data_source ingestion.
- Slack, Google Docs, and GitHub connectors.
- Frontend UI.
- Runtime vector retrieval/ranking.
- Enterprise SSO/RBAC.
- Production KMS/secret-manager integration unless adopted by the deployment owner.

## Release Gate

Before deploying, run the release checklist in `docs/release/backend-release-checklist.md`.
