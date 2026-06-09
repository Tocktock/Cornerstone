# VS-0 Search Understanding Batch 6 Report - 2026-06-09

Status: PASS for the first deterministic search-understanding runtime slice only.
Scope: `CS-UND-002` and `CS-UND-003`.

This report does not mark dense-vector retrieval, semantic model integration, artifact UI/API detail, ontology suggestions, brief generation, missions, UI, API parity, or production search infrastructure as complete.

## Scenario Table

| ID | Type | Status | Evidence |
|---|---|---|---|
| CS-UND-002 | MUST_PASS | PASS | `reports/scenario/vs0-search-understanding-2026-06-09.json`, exact and semantic-alias search transcripts |
| CS-UND-003 | MUST_PASS | PASS | `reports/scenario/vs0-search-understanding-2026-06-09.json`, personal, organization, and project workspace search comparison transcripts |

## Command Evidence

```sh
PATH="$PWD:$PATH" cornerstone scenario verify vs0-search-understanding --json --output reports/scenario/vs0-search-understanding-2026-06-09.json
# status: success
# scenario_set: vs0-search-understanding
# summary.pass: 2
# summary.blocking: 0
# summary.product_feature_claims: PARTIAL_VS0_SEARCH_UNDERSTANDING_ONLY
# search_understanding.semantic_artifact_id: art_2735313e0cb92563
# search_understanding.project_result_count: 1
# negative_evidence.cross_workspace_results: 0
# negative_evidence.same_content_scope_collisions: 0
```

## Evidence Summary

- Exact search for `alpha-evidence-anchor` returns the ingested artifact with exact and keyword match reasons.
- Semantic-alias search for `retain raw proof` returns the same artifact through deterministic alias reasons such as `retain -> keep`, `raw -> original/source`, and `proof -> evidence/source/material`.
- Personal workspace search returns only `personal-only-alpha`; organization workspace search returns only `org-visible-beta`; project workspace search returns only `distinct artifact`.
- Cross-workspace searches return zero results across personal, organization, and project scopes.
- Same-content ingestion into personal and organization scopes preserves one result per active scope with the correct owner/namespace/workspace metadata.

## Matrix Update

`docs/scenario-contracts/SCENARIO_VERIFICATION_MATRIX.csv` marks only `CS-UND-002` and `CS-UND-003` as `PASS` in this batch.

## Gaps

- `CS-UND-004` remains `NOT_VERIFIED`; UI/API artifact detail evidence is not complete in this batch.
- `CS-UND-005` and later ontology/brief scenarios remain `NOT_VERIFIED`.
- Dense-vector retrieval and local Ollama semantic smoke remain outside this batch.
- UI/API artifact viewers and mission relationships remain `NOT_VERIFIED`.
