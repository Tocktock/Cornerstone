# 2026-04-27 Live Proof Change Log

## Goal

Use the manual live proof as a backend release gate for this product loop:

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

## Fixes packaged in v0.12.2

### 1. Live PostgreSQL migration blocker

File: `migrations/versions/0001_postgres_persistence.py`

Problem: Alembic creates `alembic_version.version_num` as `VARCHAR(32)` by default, but Cornerstone uses descriptive revision IDs longer than 32 characters, such as `0007_provider_object_ingestion_support`.

Fix: widen `alembic_version.version_num` to `VARCHAR(128)` in the first migration.

Regression coverage: `tests/unit/test_postgres_extensions.py` asserts the DDL exists.

### 2. Live PostgreSQL UUID fixture

File: `tests/postgres/test_live_postgres_worker_concurrency.py`

Problem: live PostgreSQL tests inserted `DataSource(id="source-1")`, but the PostgreSQL schema uses UUID columns.

Fix: use a valid UUID string fixture for the live source ID.

### 3. Live Notion settings loading

Files:

- `scripts/run_live_notion_e2e.py`
- `tests/live_notion/test_live_notion_page_e2e.py`

Problem: the live Notion runner and gated live pytest path instantiated `Settings()` directly, so sourced environment variables such as `NOTION_MOCK_EXTERNAL_API=false` and `PERSISTENCE_BACKEND=postgres` were ignored.

Fix: use `Settings.from_env()` in both live Notion entrypoints.

### 4. Mock Notion freshness

File: `src/cornerstone/connectors/providers/notion/gateway.py`

Problem: `MockNotionGateway` hard-coded `last_edited_time=2026-04-20T12:00:00.000Z`; by 2026-04-27 UTC this exceeded the 7-day fresh threshold and made the local connector lifecycle test time-dependent.

Fix: mock Notion pages now emit a current UTC Notion-style timestamp via `_mock_last_edited_time()`.

## User-run verification evidence

### Local package gate

```text
226 passed
coverage total 85%
ruff: All checks passed
mypy: Success: no issues found in 55 source files
compileall passed
```

### Explicit mypy gate

```text
Success: no issues found in 55 source files
```

### Live PostgreSQL gate

```text
Live PostgreSQL verification passed: 5 tests, 0 skipped.
passed=5
skipped=0
failed=0
errors=0
```

### Live Notion discovery

```text
status=passed
mode=live
total_count=272
page=268
data_source=4
ingestible=268
not_ingestible=4
```

### Live Notion page E2E

```text
status=passed
sync_job_status=succeeded
artifact_count=1
evidence_fragment_count=4
source_next_action=review_evidence
```

Gated pytest path:

```text
tests/live_notion/test_live_notion_page_e2e.py::test_live_notion_page_to_artifact_and_evidence_e2e PASSED
1 passed
0 skipped
```

### Live API release-gate proof

```text
health.version=0.12.1
hasRealSources=true
onboardingRequired=false
source.type=notion
source.productionEnabled=true
source.authStatus=authorized
source.connectionStatus=test_passed
source.artifactCount=1
source.evidenceFragmentCount=4
```

Artifact/evidence:

```text
artifact_count=1
freshnessState=fresh
extractionStatus=complete
evidence_fragment_count=4
definition_evidence="Cornerstone is a shared organizational context layer."
definition_trustState=unreviewed
definition_freshnessState=fresh
```

Review/officialization:

```text
reviewed evidence: trustState=reviewed, reviewedBy=reviewer@example.com
unauthorized review: 403
concept candidate: name=Cornerstone, status=reviewing
official concept: status=official, officializedBy=reviewer@example.com
unauthorized officialization: 403
```

Grounded context:

```text
trustLabel=official
officialAnswerAvailable=true
conceptCount=1
evidenceCount=1
invalidCitationCount=0
freshness.state=fresh
limitations=[]
```

Unsupported context:

```text
trustLabel=unsupported
officialAnswerAvailable=false
limitations contains "No matching Concept or EvidenceFragment was found."
```

Evaluation:

```text
success=true
answerCorrect=true
evidenceValid=true
provenancePresent=true
trustLabelCorrect=true
freshnessPolicyRespected=true
unsupportedOfficialClaim=false
citationValidityRate=1.0
groundedContextTaskSuccessRate=1.0
provenanceCoverageRate=1.0
citationValidityRate=1.0
freshnessComplianceRate=1.0
trustLabelCorrectnessRate=1.0
weak evaluation task: 422
```

Source API safety negatives:

```text
fake provider source creation: 409
fake OAuth completion: 404
legacy sync route: 404
manual sync on Notion source: 409
```

## Current status

The backend release gate passed for the configured pilot Notion page. v0.12.2 packages the local fixes that made that proof pass.

## Explicit limitation

This proof validates Notion page ingestion. Notion database/data_source ingestion remains intentionally deferred.
