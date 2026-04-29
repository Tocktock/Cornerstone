# 04 — Implementation Plan

## Backend-first sequence

### Slice 1a — Contract and trust-loop skeleton

Status: completed before v0.2.

Scope:

- FastAPI app factory.
- Pydantic v2 API schemas.
- In-memory repository for fast validation.
- Source state endpoints.
- Sync endpoint for source objects.
- Artifact and EvidenceFragment creation.
- Freshness calculation.
- Concept candidate creation.
- Initial officialization gate.
- Grounded context query endpoint.
- Unit and integration tests.

### Slice 1b — Trust gate correctness

Status: implemented in v0.2.

Scope:

- Production/demo isolation in officialization and serving.
- Pending auth source does not count as real connected production source.
- Evidence review endpoint.
- Authorized reviewer allow-list.
- Reviewed evidence required for officialization.
- Fresh/aging evidence required for officialization.
- DecisionRecord model and API.
- Fake DecisionRecord IDs blocked.
- DecisionRecord requires reviewed evidence.
- Sync idempotency for unchanged source object hashes.
- In-memory rollback for failed sync.
- Degraded source state after failed sync following previous success.
- Audit events for review, blocked officialization, successful officialization, and DecisionRecord creation.
- Structured log assertions.

Exit criteria:

```text
49 tests passed
91% total coverage
```

### Slice 1c — Persistence and migrations

Status: next backend slice.

Scope:

- Add repository protocol/interface.
- Add SQLAlchemy models mirroring domain entities.
- Add PostgreSQL connection settings.
- Add Alembic environment and initial migration.
- Add uniqueness/index rules:
  - DataSource ID
  - Artifact `datasource_id + source_external_id + raw_content_hash`
  - EvidenceFragment artifact foreign key
  - Concept support references
  - DecisionRecord support references
- Implement DB transactions for sync and officialization.
- Add database integration tests.

Exit criteria:

- All v0.2 trust-gate tests pass against persistent repository.
- Migration can create schema from scratch.
- Failed sync cannot persist partial Artifact/Evidence writes.
- Officialization is atomic with audit event creation.

### Slice 1d — Real Notion connector

Status: planned.

Scope:

- OAuth callback handling.
- Secure token persistence boundary.
- Notion API client adapter.
- Page/database discovery.
- Incremental sync cursor.
- Permission-aware source object capture.
- Connector failure/degraded/stale behavior.
- Mocked Notion API integration tests.

Exit criteria:

- A real Notion source can produce Artifacts and EvidenceFragments.
- OAuth failure and sync failure states are visible.
- No demo data can enter production official context.

### Slice 1e — Evaluation framework backend

Status: planned.

Scope:

- GroundedContextEvalTask.
- GroundedContextEvalResult.
- Metric calculation for `grounded_context_task_success_rate`.
- Citation validity checks.
- Freshness compliance checks.
- Trust-label correctness checks.
- Unsupported official-claim detection.

Exit criteria:

- Evaluation can score grounded context responses.
- Metric includes correctness, evidence validity, provenance, trust label, freshness, and unsupported official claims.

## Current risk ordering

1. Persistence and transactions.
2. Real Notion connector/token handling.
3. Full RBAC/authorization provider.
4. Evaluation framework.
5. Review workflow UI/batch operations.
6. Retrieval quality.

## Rule for future implementation

Do not add AI/serving sophistication before persistence, connector, and trust gate correctness are stable. A more fluent answer is harmful if it weakens provenance, freshness, or unsupported-state behavior.


### Slice v0.9.0 — Live PostgreSQL and multi-worker safety

Status: implemented.

Scope:

- Add `leaseHeartbeatAt` to SyncJob runtime state.
- Add sync job heartbeat endpoint.
- Allow expired running leases to be reclaimed.
- Block active running leases from duplicate worker claims.
- Use PostgreSQL `FOR UPDATE SKIP LOCKED` in the SQLAlchemy claim path.
- Add scheduled enqueue duplicate handling through stable enqueue keys.
- Add live PostgreSQL tests gated by `RUN_POSTGRES_TESTS=1`.
- Add migration `0009_live_postgres_worker_concurrency`.

Exit criteria:

```text
A single queued job cannot be claimed by two workers.
An expired running lease can be recovered.
A wrong worker cannot heartbeat another worker's lease.
Live PostgreSQL tests exist for extension/migration and concurrent claim behavior.
```
## v0.9.2 Completed

- Added live Notion E2E pilot runner.
- Added page snapshot retrieval for operator-specified Notion page IDs.
- Added gated live Notion tests excluded from default local runs.


### Slice v0.10.0 — Evidence review queue and officialization backend hardening

Status: implemented.

Scope:

- Add Evidence review queue endpoint with source/artifact context.
- Add conflicted EvidenceFragment trust state.
- Add reviewer action to create Concept candidates from EvidenceFragments.
- Add ConceptRelation schema, persistence, API, and audit events.
- Add officialization gate for ConceptRelations.
- Require official source/target Concepts for official Relations.
- Require reviewed evidence or valid DecisionRecord support for official Relations.
- Add transaction boundaries for reviewer workflow writes plus audit events.

Exit criteria:

```text
A reviewer can inspect unreviewed evidence, mark it reviewed/rejected/conflicted,
create a Concept candidate from evidence, create a typed ConceptRelation,
and officialize that relation only when trust gates are satisfied.
```

## v0.11.0 implementation note

Grounded serving contract hardening is complete for the backend contract slice.

Implemented:

```text
- Shared response shape for humans and AI.
- Trust-label handling for official/evidence_supported/partially_supported/stale/conflicted/unsupported.
- Evidence-only serving when no Concept exists.
- ConceptRelation and DecisionRecord support metadata in citations.
- Citation validity guardrails.
- OpenAPI snapshot coverage for `/v1/context/query`.
```

Next implementation area:

```text
v0.12.0 — Evaluation framework and grounded_context_task_success_rate.
```


## v0.12.0 Completed — Evaluation framework

The backend now persists evaluation tasks/results and reports `grounded_context_task_success_rate`. Remaining backend release-candidate work is live PostgreSQL execution, live Notion E2E execution, credential/KMS hardening, and release runbook cleanup.


## v0.12.0 implementation note

Evaluation framework and `grounded_context_task_success_rate` are implemented for the backend contract slice.

Implemented:

```text
- GroundedContextEvalTask schema and API.
- GroundedContextEvalResult schema and persisted response snapshots.
- Rule-based evaluator for correctness, evidence validity, provenance, trust label, freshness, and unsupported official claims.
- Evaluation metric summary API.
- SQLAlchemy and in-memory store support.
- Alembic migration 0011_grounded_context_evaluation.
```

Next implementation areas:

```text
v0.12.1 — product trust cleanup before live proof.
v0.12.2 — live PostgreSQL execution proof with zero skipped live tests.
v0.12.3 — live Notion E2E execution proof with a real shared page.
v0.12.4 — credential/security hardening for pilot release.
v0.12.5 — release-candidate cleanup and operator runbook.
```


## v0.12.1 Completed — Product trust cleanup before live proof

- Removed fake/provider-backed source creation through `POST /v1/sources`.
- Removed fake OAuth completion route from the API surface.
- Confirmed officialization and serving use separate source-eligibility policies.
- Strengthened evaluation task validation so metrics cannot be gamed by vague tasks.
- Added evaluation OpenAPI snapshot coverage.
- Cleaned release packaging to exclude `__pycache__` and `.pyc` files.

## v0.13.0 — Backend release-candidate cleanup

Status: implemented.

Scope:

```text
- Add operator runbook.
- Add backend release checklist.
- Add known limitations.
- Add production deployment checklist.
- Add secrets and credential handling checklist.
- Add API freeze review.
- Add live proof artifact template.
- Add v1.0.0 readiness summary.
- Add static release-candidate check script.
```

Exit criteria:

```text
A backend operator can repeat the live PostgreSQL, live Notion, review, officialization, grounded serving, and evaluation proof from docs alone.
```

Next recommended tag after the checklist passes:

```text
v1.0.0-rc.1
```
