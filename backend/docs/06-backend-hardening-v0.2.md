# 06 — Backend Hardening v0.2

## Why this pass was needed

The previous backend scaffold had useful API shape, but the trust model was too weak. It could imply production readiness while leaving gaps around demo contamination, unreviewed evidence, fake DecisionRecord references, and incomplete operational logs.

v0.2 focuses only on backend correctness.

## Main improvements

```text
Production/demo isolation
Reviewed evidence gate
DecisionRecord validation
Reviewer authorization placeholder
Sync idempotency
Sync rollback/degraded state
Audit events for blocked paths
Grounded serving trust-label hardening
Verbose reproducible reports
```

## Trust gates now enforced

A Concept cannot become official unless:

```text
reviewer is authorized
AND support exists
AND all referenced support exists
AND EvidenceFragments are reviewed
AND EvidenceFragments are fresh or aging
AND EvidenceFragments have valid Artifact/DataSource provenance
AND production mode excludes non-production sources
AND source status is connected
```

## Evidence provenance now required

EvidenceFragments require:

```text
artifactId
dataSourceId
sourceType
sourceExternalId
artifactTitle
capturedAt
quoteRange when extracted from text
```

## Sync behavior now safer

- Re-syncing identical content reuses the existing Artifact and EvidenceFragments.
- Failed extraction rolls back newly written Artifact/Evidence rows in the in-memory store.
- Failure after a previous successful sync marks the source `degraded` and freshness `unknown`.

## Serving behavior now safer

Grounded context no longer labels a response `official` when evidence freshness is stale, mixed, or unknown.

Production mode excludes non-production or unhealthy source evidence from serving.

## Test evidence

```text
49 tests passed
91% total coverage
```

Important added tests:

```text
test_pending_auth_source_does_not_count_as_real_connected_source
test_sync_is_idempotent_for_same_source_external_id_and_hash
test_failed_sync_after_prior_success_marks_degraded_and_rolls_back
test_officialization_with_unreviewed_evidence_returns_conflict
test_non_production_source_evidence_cannot_officialize
test_unauthorized_reviewer_cannot_officialize
test_create_concept_with_fake_decision_record_id_returns_not_found
test_decision_record_creation_requires_reviewed_evidence
test_decision_record_can_support_concept_officialization
test_official_concept_with_unknown_freshness_is_not_labeled_official
```

## Remaining backend risks

- In-memory repository only; no durable persistence yet.
- No real Notion connector yet.
- Reviewer authorization is an allow-list, not enterprise RBAC.
- No ConceptRelation route yet.
- No evaluation runner yet.
- Query matching is deliberately simple and should be replaced by a retrieval interface after trust foundations are durable.

## Next backend action

Build the SQLAlchemy/PostgreSQL repository and migrations while keeping the v0.2 test suite as a regression safety net.
