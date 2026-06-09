# VS-0 Search Evidence Batch 5 Report - 2026-06-09

Status: PASS for the first search snapshot, claim evidence-bundle, and evidence-viewer runtime slice only.
Scope: `CS-ARCH-008`, `CS-ARCH-009`, and `CS-UND-001`.

This report does not mark semantic retrieval, brief generation, full claim lifecycle, UI evidence viewer, API parity, or action workflows as complete.

## Scenario Table

| ID | Type | Status | Evidence |
|---|---|---|---|
| CS-ARCH-008 | MUST_PASS | PASS | `reports/scenario/vs0-search-evidence-2026-06-09.json`, search snapshot, evidence bundle, and draft claim transcripts |
| CS-ARCH-009 | MUST_PASS | PASS | `reports/scenario/vs0-search-evidence-2026-06-09.json`, evidence viewer and artifact show transcripts |
| CS-UND-001 | MUST_PASS | PASS | `reports/scenario/vs0-search-evidence-2026-06-09.json`, timed ingest-plus-search transcript with snippet and evidence refs |

## Command Evidence

```sh
PATH="$PWD:$PATH" cornerstone scenario verify vs0-search-evidence --json --output reports/scenario/vs0-search-evidence-2026-06-09.json
# status: success
# scenario_set: vs0-search-evidence
# summary.pass: 3
# summary.blocking: 0
# summary.product_feature_claims: PARTIAL_VS0_SEARCH_EVIDENCE_ONLY
# search_evidence.result_count: 1
# search_evidence.first_snippet: alpha-evidence-anchor.
# search_evidence.claim_id: claim_c45b0823f53ae7a1
# search_evidence.evidence_viewer_id: viewer_9e9a9bd9324b5100
# search_evidence.first_use_duration_ms: 90.843
```

The generated report embeds CLI transcripts for:

- `cornerstone artifact ingest fixtures/vs0/packs/01_artifact_basic/input.txt --json`
- `cornerstone search query alpha-evidence-anchor --json`
- `cornerstone evidence bundle create --search-snapshot-id <id> --json`
- `cornerstone evidence bundle show <bundle_id> --json`
- `cornerstone evidence view <bundle_id> --json`
- `cornerstone claim create --evidence-bundle-id <bundle_id> --json`
- `cornerstone claim show <claim_id> --json`
- `cornerstone artifact show <artifact_id> --json`
- `cornerstone audit verify --json`

## Evidence Summary

- Search snapshots store query, filters, result count, result snippets, original storage refs, derived text refs, duration, evidence refs, and audit refs.
- Evidence bundles store the search snapshot ID, query, filters, result snapshot, and artifact-backed evidence items.
- The draft claim stores a claim-to-evidence-bundle link, search snapshot reference, query, filters, and artifact refs.
- The evidence viewer exposes original source metadata plus derived text/metadata for the cited artifact.
- Evidence bundle reads, evidence viewer opens, and claim reads are audit-recorded.
- The search result includes `alpha-evidence-anchor`; ingest-plus-search completed within the local first-use threshold.

## Matrix Update

`docs/scenario-contracts/SCENARIO_VERIFICATION_MATRIX.csv` marks only `CS-ARCH-008`, `CS-ARCH-009`, and `CS-UND-001` as `PASS` in this batch.

## Gaps

- `CS-UND-002` remains `NOT_VERIFIED`; semantic retrieval is not implemented or evidenced in this batch.
- Full claim lifecycle, claim review/approval, and claim-to-evidence UI/API behavior remain `NOT_VERIFIED`.
- Full VS-0 and full 206-scenario product PASS remain incomplete.
