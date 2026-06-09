# VS-0 Artifact Runtime Batch 3 Report - 2026-06-09

Status: PASS for the first artifact/archive product-runtime slice only.
Scope: `CS-ARCH-001` through `CS-ARCH-005`.

This report does not mark search, evidence bundles, briefs, claims, actions, UI, API, namespace policy, or the full audit/security suite as complete.

## Scenario Table

| ID | Type | Status | Evidence |
|---|---|---|---|
| CS-ARCH-001 | MUST_PASS | PASS | `reports/scenario/vs0-artifacts-2026-06-09.json`, `cornerstone artifact ingest fixtures/vs0/packs/01_artifact_basic/input.txt --json` |
| CS-ARCH-002 | MUST_PASS | PASS | `reports/scenario/vs0-artifacts-2026-06-09.json`, `cornerstone artifact ingest ... --derived-mode fail --json`, `cornerstone artifact show <artifact_id> --json` |
| CS-ARCH-003 | MUST_PASS | PASS | `reports/scenario/vs0-artifacts-2026-06-09.json`, duplicate and changed-content ingest transcripts |
| CS-ARCH-004 | MUST_PASS | PASS | `reports/scenario/vs0-artifacts-2026-06-09.json`, `cornerstone artifact show <artifact_id> --json` |
| CS-ARCH-005 | MUST_PASS | PASS | `reports/scenario/vs0-artifacts-2026-06-09.json`, unknown-format ingest/show transcripts |

## Command Evidence

```sh
PATH="$PWD:$PATH" cornerstone scenario verify vs0-artifacts --json --output reports/scenario/vs0-artifacts-2026-06-09.json
# status: success
# scenario_set: vs0-artifacts
# summary.pass: 5
# summary.blocking: 0
# summary.product_feature_claims: PARTIAL_VS0_ARTIFACTS_ONLY
# negative_evidence.lost_originals: 0
# negative_evidence.conflicting_duplicate_truth_records: 0
```

```sh
PATH="$PWD:$PATH" cornerstone scenario gate reports/scenario/vs0-artifacts-2026-06-09.json --json
# status: success
# scenario_count: 5
# blocking_count: 0
```

```sh
python3 scripts/generate_scenario_verification_matrix.py --check
python3 scripts/verify_scenario_matrix.py
# PASS: scenario verification matrix is current.
# PASS: scenario verification matrix verified (206 scenarios; no missing rows; no unevidenced PASS claims).
```

```sh
python3 -m unittest discover -s tests -p 'test_*.py'
# Ran 10 tests
# OK
```

```sh
make verify-local-fast
# PASS: CornerStone scaffold CLI verified (version, health, honest ready, scenario list, coverage, vs0-scaffold verify, vs0-fixtures verify, vs0-artifacts verify, unittest).
```

The generated report embeds CLI transcripts for:

- `cornerstone artifact ingest fixtures/vs0/packs/01_artifact_basic/input.txt --state-dir tmp/scenario/vs0-artifacts --json`
- `cornerstone artifact ingest fixtures/vs0/packs/03_unknown_and_failed_extraction/fail.txt --derived-mode fail --json`
- `cornerstone artifact ingest fixtures/vs0/packs/03_unknown_and_failed_extraction/unknown.bin --media-type application/octet-stream --derived-mode unsupported --json`
- duplicate and changed-content `cornerstone artifact ingest` commands for `pack_02_dedup_versioning`
- `cornerstone artifact show <artifact_id> --json`
- `cornerstone audit verify --state-dir tmp/scenario/vs0-artifacts --json`

## Evidence Summary

- Original artifact records include stable artifact IDs, SHA-256 checksums, original storage refs, source paths, timestamps, tenant/owner/namespace/workspace scope, trust state, and provenance transformations.
- Failed derived processing records keep the original artifact and mark `derived.status: failed`.
- Unknown-format records keep the original artifact and mark `derived.status: deferred` with `reason: unsupported_format`.
- Identical content re-ingestion returns the same artifact ID with `deduplicated: true`.
- Changed content creates a distinct artifact ID with `provenance.lineage_from` set to the prior artifact.
- Artifact ingest/show commands emit evidence refs and audit refs.
- `cornerstone audit verify` reports hash-chain integrity success for the scenario run.

## Matrix Update

`docs/scenario-contracts/SCENARIO_VERIFICATION_MATRIX.csv` marks only `CS-ARCH-001` through `CS-ARCH-005` as `PASS`. All other product rows remain at their prior status.

## Gaps

- Full 206-scenario product PASS remains incomplete.
- VS-0 search, evidence bundle, brief, claim, action, approval/execution, UI, API, namespace policy, prompt-injection product behavior, secret-redaction product behavior, and full audit lifecycle scenarios remain `NOT_VERIFIED`.
- This batch uses a standard-library local runtime state directory for deterministic verification. It does not introduce the approved Postgres/FastAPI/UI stack.
