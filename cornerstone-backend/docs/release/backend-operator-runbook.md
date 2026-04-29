# Backend Operator Runbook

## Purpose

This runbook describes how to operate and verify the backend MVP loop for Cornerstone.

The required proof loop is:

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

## 1. Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## 2. Local non-live gate

```bash
./scripts/run_tests.sh
python -m mypy src --show-error-codes --no-color-output --no-incremental
python scripts/check_release_candidate.py
```

Expected:

```text
pytest passes
coverage threshold passes
ruff passes
mypy passes
compileall passes
release-candidate check passes
```

## 3. Live PostgreSQL gate

```bash
docker compose up -d postgres

export RUN_POSTGRES_TESTS=1
export PERSISTENCE_BACKEND=postgres
export DATABASE_URL='postgresql+psycopg://cornerstone:cornerstone@localhost:5432/cornerstone_live_proof'

python scripts/run_live_postgres_tests.py --min-passed 5
```

Expected:

```text
Live PostgreSQL verification passed: 5 tests, 0 skipped.
```

Required reports:

```text
reports/live-postgres-summary.txt
reports/live-postgres-report.txt
```

## 4. Live Notion E2E gate

Use a safe test page shared with the Notion integration. Do not commit or share tokens.

```bash
export RUN_NOTION_E2E=1
export NOTION_MOCK_EXTERNAL_API=false
export CONNECTOR_ENCRYPTION_SECRET='replace-with-a-long-local-proof-secret-32chars-plus'
export NOTION_E2E_ACCESS_TOKEN='<your-notion-access-token>'
export NOTION_E2E_PAGE_ID='<your-shared-notion-page-id>'
export NOTION_E2E_REQUIRE_EVIDENCE=1
export PERSISTENCE_BACKEND=postgres
export DATABASE_URL='postgresql+psycopg://cornerstone:cornerstone@localhost:5432/cornerstone_live_proof'

alembic upgrade head
PYTHONPATH=src python scripts/run_live_notion_e2e.py | tee reports/manual-live-notion-e2e.json
RUN_NOTION_E2E=1 python -m pytest tests/live_notion -m live_notion -vv --color=no
```

Pass criteria:

```text
status=passed
sync_job_status=succeeded
artifact_count >= 1
evidence_fragment_count >= 1
0 skipped live Notion tests
```

## 5. API release-gate proof

Start the API:

```bash
export AUTHORIZED_REVIEWERS_RAW='reviewer@example.com'
export PERSISTENCE_BACKEND=postgres
export DATABASE_URL='postgresql+psycopg://cornerstone:cornerstone@localhost:5432/cornerstone_live_proof'
uvicorn cornerstone.main:app --reload
```

Confirm:

```bash
curl -s http://localhost:8000/healthz | jq
```

Expected version:

```text
0.13.1
```

Then verify:

```text
1. /v1/sources shows a real Notion source.
2. /v1/artifacts has at least one Artifact.
3. /v1/evidence has at least one EvidenceFragment.
4. Evidence review by authorized reviewer succeeds.
5. Unauthorized review fails with 403.
6. Concept candidate can be created from reviewed evidence.
7. Concept officialization succeeds.
8. Unauthorized officialization fails with 403.
9. /v1/context/query returns trustLabel=official for the pilot query.
10. Unsupported query returns trustLabel=unsupported.
11. Evaluation task run returns success=true.
12. /v1/evaluations/summary returns groundedContextTaskSuccessRate=1.0 for the pilot task.
```

## 6. Safety negative checks

Run these checks before release:

```text
POST /v1/sources with type=notion returns 409.
POST /v1/sources/{sourceId}/oauth/complete returns 404.
POST /v1/sources/{sourceId}/sync returns 404.
POST /v1/manual-sources/{notionSourceId}/sync returns 409.
Weak evaluation task returns 422.
```

## 7. Evidence package

Save sanitized proof artifacts:

```text
reports/live-postgres-summary.txt
reports/live-postgres-report.txt
reports/manual-live-notion-e2e.json
reports/manual-reviewed-evidence.json
reports/manual-concept-candidate.json
reports/manual-official-concept.json
reports/manual-grounded-context.json
reports/manual-unsupported-context.json
reports/manual-eval-result.json
reports/manual-eval-summary.json
```

Do not include:

```text
Notion access tokens
.env files with secrets
raw logs that include tokens
private source content beyond the safe pilot snippets
```

