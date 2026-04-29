# 02 — Testing Strategy

## Success criteria

The backend is acceptable for v0.2 when tests prove the P0 trust rules:

1. Production mode has an honest empty state.
2. Pending OAuth sources do not count as real connected sources.
3. Source status transitions are explicit.
4. Sync creates Artifacts from source objects.
5. EvidenceFragments always include required provenance.
6. Sync is idempotent for unchanged source objects.
7. Failed sync rolls back partial writes.
8. Failed sync after prior success marks the source degraded and does not imply freshness.
9. Evidence review requires an authorized reviewer.
10. Concepts cannot become official without reviewed support.
11. Concepts cannot become official from non-production evidence in production mode.
12. Fake DecisionRecord IDs cannot support officialization.
13. DecisionRecords require reviewed evidence.
14. Grounded context returns `unsupported` when support is missing.
15. Official context is not labeled official when freshness is stale, mixed, or unknown.
16. Structured operational logs are emitted for sync, review, officialization, serving, and HTTP request paths.
17. Audit events are exposed for review, blocked officialization, successful officialization, and DecisionRecord creation.
18. Worker success writes commit or roll back as a single unit across Artifact, EvidenceFragment, source, SyncCursor, SyncJob, and terminal SyncJobEvent state.

## Test pyramid

```text
Unit tests
  Domain services:
    freshness
    extraction/provenance
    officialization gates
    trust label selection

Integration tests
  FastAPI request/response contracts:
    source onboarding
    source sync
    artifact/evidence creation
    evidence review
    DecisionRecord creation
    concept creation and officialization
    grounded context query
    audit event listing
    structured log assertions with caplog

Persistence tests, next slice
  SQLAlchemy repository
  Alembic migrations
  database transaction rollback
  worker success write atomicity
  idempotency constraints
```

## Unit tests

### FreshnessPolicy

| Case | Expected |
| --- | --- |
| No timestamp | `unknown` |
| Timestamp within fresh threshold | `fresh` |
| Timestamp after fresh threshold and before stale threshold | `aging` |
| Timestamp after stale threshold | `stale` |
| Future timestamp from clock skew | `fresh` |

### EvidenceExtractor

| Case | Expected |
| --- | --- |
| Non-empty artifact content | At least one EvidenceFragment. |
| Extracted fragment | Has artifact ID, dataSourceId, sourceType, sourceExternalId, source URL when available, artifact title, capturedAt, quoteRange. |
| Requirement sentence | Classified as `requirement`. |
| Decision sentence | Classified as `decision`. |
| Question sentence | Classified as `open_question`. |

### OfficializationService

| Case | Expected |
| --- | --- |
| Concept with no evidence and no decision | Block officialization. |
| Concept with unreviewed evidence | Block officialization. |
| Concept with non-production evidence in production mode | Block officialization. |
| Unauthorized reviewer | Block officialization with authorization error. |
| Concept with reviewed production evidence | Mark official and emit audit event payload. |

### GroundedContextService

| Case | Expected |
| --- | --- |
| No matching Concept | `unsupported`. |
| Candidate with reviewed evidence | `evidence_supported`. |
| Candidate with unreviewed evidence | `partially_supported`. |
| Official with fresh reviewed evidence | `official`. |
| Official with stale evidence | `stale`. |
| Official with unknown freshness | `partially_supported`, not `official`. |
| Non-production source evidence in production mode | Excluded, usually `unsupported`. |
| Conflicted Concept | `conflicted`. |

## Integration tests

Integration tests exercise the full API contract through FastAPI `TestClient`.

Important failure-path tests:

```text
test_notion_source_starts_pending_auth_and_sync_is_blocked
test_pending_auth_source_does_not_count_as_real_connected_source
test_officialization_without_support_returns_conflict
test_officialization_with_unreviewed_evidence_returns_conflict
test_non_production_source_evidence_cannot_officialize
test_unauthorized_reviewer_cannot_officialize
test_create_concept_with_fake_decision_record_id_returns_not_found
test_decision_record_creation_requires_reviewed_evidence
test_unauthorized_reviewer_cannot_review_evidence
test_failed_sync_after_prior_success_marks_degraded_and_rolls_back
```

## Observability tests

Structured logs are intentionally asserted, not treated as incidental output.

Current tested event families:

```text
source.*
artifact.*
evidence.*
decision_record.*
concept.*
context.*
http.request.*
```

## Current verified report

```text
49 tests passed
91% total coverage
```

The test report includes full test names and structured JSON log events. The coverage report includes branch coverage.


## v0.9.0 live PostgreSQL and multi-worker checks

v0.9.0 adds deterministic local tests plus gated live PostgreSQL tests.

Local checks prove:

```text
1. Active running leases cannot be claimed by another worker.
2. Expired running leases can be reclaimed.
3. Worker heartbeat requires the current lease owner.
4. `leaseHeartbeatAt` is persisted in schemas and SQLAlchemy metadata.
5. Migration `0009_live_postgres_worker_concurrency` adds claimable index and heartbeat column.
```

Live PostgreSQL checks are located in `tests/postgres/` and require:

```bash
RUN_POSTGRES_TESTS=1
DATABASE_URL=postgresql+psycopg://cornerstone:cornerstone@localhost:5432/cornerstone
```

They verify:

```text
1. Required PostgreSQL extensions and migration head.
2. Concurrent workers claim one queued job only once.
3. Concurrent workers reclaim one expired running job only once.
```

## v0.9.1 strict live PostgreSQL verification

v0.9.1 separates normal local verification from explicit live PostgreSQL verification.

Local runs use:

```bash
./scripts/run_tests.sh
```

When `RUN_POSTGRES_TESTS` is not set to `1`, this script ignores `tests/postgres` rather than reporting intentional skipped tests. This keeps local reports clean and avoids treating skipped live tests as normal success evidence.

Strict live PostgreSQL runs use:

```bash
export RUN_POSTGRES_TESTS=1
export DATABASE_URL=postgresql+psycopg://cornerstone:cornerstone@localhost:5432/cornerstone
python scripts/run_live_postgres_tests.py
```

The strict runner fails when:

```text
- RUN_POSTGRES_TESTS is not explicitly enabled.
- DATABASE_URL is missing or not PostgreSQL.
- PostgreSQL migrations fail.
- Required extensions are missing.
- Live PostgreSQL tests skip.
- Live PostgreSQL tests fail or error.
- Too few live PostgreSQL tests pass.
```

v0.9.1 also stabilizes mypy reporting by running mypy with non-incremental mode, a report-local cache, and a timeout guard.
## v0.9.2 Live Notion E2E Testing

v0.9.2 adds `tests/live_notion` and `scripts/run_live_notion_e2e.py`. These tests are skipped from the default suite and require `RUN_NOTION_E2E=1`, `NOTION_MOCK_EXTERNAL_API=false`, `NOTION_E2E_ACCESS_TOKEN`, `NOTION_E2E_PAGE_ID`, and a non-default `CONNECTOR_ENCRYPTION_SECRET`.


## v0.10.0 testing additions

v0.10.0 adds tests for the reviewer workflow and officialization hardening layer:

```text
- Evidence review queue filtering and source context.
- Evidence conflicted trust state.
- Concept candidate creation from EvidenceFragments.
- Unauthorized reviewer blocking.
- ConceptRelation creation and read/list APIs.
- ConceptRelation officialization with reviewed evidence.
- ConceptRelation officialization blocked without support.
- ConceptRelation officialization blocked unless both Concepts are official.
- SQLAlchemy persistence of official ConceptRelations across app/store reloads.
```

These tests keep the backend aligned with the rule that official context must be reviewer-authorized and backed by reviewed evidence or a valid DecisionRecord.

## v0.11.0 Grounded serving contract tests

v0.11.0 adds contract and behavior tests for the shared human/AI serving response.

Coverage includes:

```text
- Evidence-only response when reviewed EvidenceFragments match but no Concept exists.
- Rejected evidence is excluded from grounded support.
- Conflicted evidence and conflicted Concepts return `conflicted`.
- Official Concept responses include official ConceptRelations.
- DecisionRecords appear in grounded responses.
- Evidence citations expose support metadata.
- Citation validity fields are present and clean for valid citations.
- OpenAPI snapshot protects `/v1/context/query`.
```

The snapshot intentionally covers only the grounded-serving path and relevant schema components so it remains stable and reviewable.


## v0.12.0 Evaluation tests

Evaluation tests cover rule-based task scoring, metric summary calculation, API task/run/result/summary flows, persistence metadata, and migration coverage for evaluation tables.


## v0.12.0 Evaluation framework tests

v0.12.0 adds evaluation tests for the PRD's `grounded_context_task_success_rate` success metric.

Coverage includes:

```text
- Evaluation task creation and validation.
- Unsupported task success when unsupported is expected.
- Detection of unsupported official claims.
- Required answer substring checks.
- Required evidence/concept/decision support checks.
- Metric summary calculation.
- API run-one and run-many behavior.
- In-memory and SQLAlchemy persistence for tasks/results.
- Alembic migration coverage for evaluation tables and indexes.
```

The default test suite still excludes gated live PostgreSQL and live Notion E2E tests unless their explicit environment flags are set.


## v0.12.1 Product trust cleanup tests

- Direct provider-backed source creation is rejected.
- Fake source OAuth completion route is removed.
- Degraded-source serving keeps reviewed captured evidence available with explicit limitations.
- Officialization remains stricter than serving.
- Vague evaluation tasks are rejected.
- Evaluation OpenAPI surface is snapshot-protected.

## v0.13.0 release-candidate checks

The release-candidate cleanup adds one static check that does not import runtime dependencies:

```bash
python scripts/check_release_candidate.py
```

This check verifies:

```text
- version markers are consistent
- release docs exist
- README references release-candidate docs
- API freeze review documents unsafe removed routes
- live proof record is present
- package hygiene excludes local caches and compiled Python files
```

This check complements, but does not replace:

```bash
./scripts/run_tests.sh
python -m mypy src --show-error-codes --no-color-output --no-incremental
python scripts/run_live_postgres_tests.py --min-passed 5
PYTHONPATH=src python scripts/run_live_notion_e2e.py
```
