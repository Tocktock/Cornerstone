# VS3 Scenario Gate Row Classification Consistency Guard Checkpoint

**Date:** 2026-06-30 KST
**Owner:** JiYong / Tars
**Slice:** VS3 scenario-gate coverage metadata hardening
**Status:** Local deterministic checkpoint; VS3-P, production/on-prem readiness, live-provider readiness, real-IdP readiness, migration/restore readiness, security acceptance, and human acceptance remain unclaimed.

## Scope

This checkpoint covers one small VS3 verifier slice:

- `VS3-GATE-004` - native `cornerstone scenario verify vs3-onprem-trusted-extension --json` and `cornerstone scenario gate <report> --json` must emit and validate status, counts, per-row evidence, human rows, and gate metadata.
- `VS3-REG-004` - scenario coverage and audit coverage cannot silently drop or mutate before a release/local-dev assurance claim.

All other VS3 AI rows remain mapped for later implementation slices. `VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`.

## Gap Found

The existing VS3 scenario gate validated row identity, row refs, aggregate refs, traceability, source-tree freshness, source command transcripts, human rows, and proof boundaries.

However, a frozen scenario row could still have classification metadata changed while preserving row ID, status counts, evidence refs, audit refs, policy refs, and source report refs.

Pre-patch mutation probe:

```text
source: tmp/vs3-probe-source.json
target row: VS3-GATE-001

owner_human:
before {'scenario_id': 'VS3-GATE-001', 'type': 'MUST_PASS', 'phase': 'VS3-0', 'owner': 'AI', 'status': 'PASS'}
after  {'scenario_id': 'VS3-GATE-001', 'type': 'MUST_PASS', 'phase': 'VS3-0', 'owner': 'Human', 'status': 'PASS'}
exit 0 status success errors []

type_human_required:
before {'scenario_id': 'VS3-GATE-001', 'type': 'MUST_PASS', 'phase': 'VS3-0', 'owner': 'AI', 'status': 'PASS'}
after  {'scenario_id': 'VS3-GATE-001', 'type': 'HUMAN_REQUIRED', 'phase': 'VS3-0', 'owner': 'AI', 'status': 'PASS'}
exit 0 status success errors []

phase_wrong:
before {'scenario_id': 'VS3-GATE-001', 'type': 'MUST_PASS', 'phase': 'VS3-0', 'owner': 'AI', 'status': 'PASS'}
after  {'scenario_id': 'VS3-GATE-001', 'type': 'MUST_PASS', 'phase': 'Human gate', 'owner': 'AI', 'status': 'PASS'}
exit 0 status success errors []
```

Why this matters:

- `owner=Human` can hide an AI row from AI PASS enforcement logic.
- `type=HUMAN_REQUIRED` can make the report contradict the frozen matrix while retaining passing counts.
- `phase=Human gate` can blur implementation slice and human-gate boundaries.
- The VS3 contract requires the matrix to remain frozen and status-neutral; reports may carry execution status, but not mutate row classification.

## Implementation

Changed `packages/cornerstone_cli/main.py`:

- Load the frozen VS3 matrix through `list_scenarios(root, "vs3-onprem-trusted-extension")`.
- Build `expected_scenarios_by_id`.
- Add `row_classification_issues` to `coverage_validation`.
- Compare each report row against the frozen matrix for:
  - `type`
  - `phase`
  - `owner`
  - `initial_status`
- Fail `coverage_validation` and emit `CS_VS3_SCENARIO_COVERAGE_INVALID` when any classification mismatch is present.

Changed `tests/scenario/test_scaffold_cli.py`:

- Added `test_vs3_scenario_gate_rejects_local_dev_claim_with_row_classification_mismatch`.
- The test mutates `VS3-GATE-001` from `type=MUST_PASS`, `phase=VS3-0`, `owner=AI` to `type=HUMAN_REQUIRED`, `phase=Human gate`, `owner=Human`.
- Expected result: gate exits `4`, reports `CS_VS3_SCENARIO_COVERAGE_INVALID`, and records all row classification mismatches while keeping human-required, claim-boundary, row-ref, aggregate-ref, and source-transcript validations isolated as passed.

## Verification

Syntax:

```text
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
exit 0
```

Focused unittest:

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_row_classification_mismatch \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_row_id_scenario_id_mismatch \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_preserves_source_proof_boundary_and_transcript

Ran 3 tests in 1.754s
OK
```

Fresh post-patch mutation probe:

```text
source: tmp/vs3-classification-post-source.json

owner_human exit 4 status failed errors ['CS_VS3_SCENARIO_COVERAGE_INVALID']
source_tree_current passed
classification_issues [{'actual': 'Human', 'expected': 'AI', 'field': 'owner', 'issue': 'row_classification_mismatch', 'row_index': 0, 'scenario_id': 'VS3-GATE-001'}]

type_human_required exit 4 status failed errors ['CS_VS3_SCENARIO_COVERAGE_INVALID']
source_tree_current passed
classification_issues [{'actual': 'HUMAN_REQUIRED', 'expected': 'MUST_PASS', 'field': 'type', 'issue': 'row_classification_mismatch', 'row_index': 0, 'scenario_id': 'VS3-GATE-001'}]

phase_wrong exit 4 status failed errors ['CS_VS3_SCENARIO_COVERAGE_INVALID']
source_tree_current passed
classification_issues [{'actual': 'Human gate', 'expected': 'VS3-0', 'field': 'phase', 'issue': 'row_classification_mismatch', 'row_index': 0, 'scenario_id': 'VS3-GATE-001'}]

all_three exit 4 status failed errors ['CS_VS3_SCENARIO_COVERAGE_INVALID']
source_tree_current passed
classification_issues [
  {'actual': 'HUMAN_REQUIRED', 'expected': 'MUST_PASS', 'field': 'type', 'issue': 'row_classification_mismatch', 'row_index': 0, 'scenario_id': 'VS3-GATE-001'},
  {'actual': 'Human gate', 'expected': 'VS3-0', 'field': 'phase', 'issue': 'row_classification_mismatch', 'row_index': 0, 'scenario_id': 'VS3-GATE-001'},
  {'actual': 'Human', 'expected': 'AI', 'field': 'owner', 'issue': 'row_classification_mismatch', 'row_index': 0, 'scenario_id': 'VS3-GATE-001'}
]

baseline exit 0 status success source_tree_current passed classification_issues []
```

## Decision

This slice strengthens `VS3-GATE-004` and `VS3-REG-004` by preventing VS3 local-dev assurance reports from silently changing frozen row classification metadata.

This is not a VS3-P claim. Human/on-prem evidence remains required for production/on-prem readiness, live provider readiness, real IdP readiness, migration/restore readiness, independent security acceptance, and human UX acceptance.
