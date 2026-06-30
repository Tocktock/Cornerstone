# VS3 Scenario Gate Downstream Generated Evidence Ignore Guard Checkpoint

**Date:** 2026-06-30 KST
**Scope:** VS3 local/dev scenario gate generated-evidence stability guard.
**Status:** Local/dev slice verified.

## Slice Contract

Goal:
- Keep `cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json` stable when known downstream generated evidence artifacts are produced after the scenario report.
- Preserve the existing guard that rejects source-bearing drift and non-ignored generated evidence drift.

In this slice:
- `VS3-GATE-004` - native VS3 scenario gate remains the product-level local verification entry point.
- `VS3-REG-004` - scenario/audit/source coverage gates still detect omitted or mutated non-ignored evidence.

Full VS3 scenario mapping:
- `in_this_slice`: `VS3-GATE-004`, `VS3-REG-004`.
- `later_slice`: `VS3-GATE-001`, `VS3-GATE-002`, `VS3-GATE-003`, `VS3-CTX-001` through `VS3-CTX-005`, `VS3-RLS-001` through `VS3-RLS-006`, `VS3-OPA-001` through `VS3-OPA-005`, `VS3-EGR-001` through `VS3-EGR-006`, `VS3-CON-001` through `VS3-CON-006`, `VS3-TOOL-001` through `VS3-TOOL-007`, `VS3-OBS-001` through `VS3-OBS-003`, `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-003`, `VS3-REG-005`, `VS3-REG-006`, `VS3-REG-007`, `VS3-REG-008`.
- `HUMAN_REQUIRED`: `VS3-H01` through `VS3-H07`.
- `blocked`: none for this local/dev slice.
- `out_of_scope`: VS3-P, production/on-prem readiness, real IdP, real network, live provider, migration/restore readiness, independent security acceptance, and human UX acceptance.

## Current Behavior

Before this slice, the VS3 scenario gate treated `reports/security/vs3-local-checkpoint.json` as ordinary generated dirty evidence. That file is a downstream local checkpoint artifact generated after the scenario report, so its byte/hash drift could make the gate fail even when the local checkpoint itself reported:

```text
status success
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
```

This created a self-invalidating evidence loop for the downstream checkpoint artifact. The gate already ignored the checked scenario report path for generated dirty snapshot validation, but it did not apply the same narrow downstream-generated ignore rule to generated dirty path validation or to the local checkpoint artifact.

## Implementation

Changed:
- `packages/cornerstone_cli/main.py`
  - Added `VS3_SCENARIO_GATE_DOWNSTREAM_GENERATED_EVIDENCE_PATHS`.
  - Added `ignored_paths` support to `_vs3_source_tree_generated_dirty_path_validation`.
  - Reused the same narrow ignore set for generated dirty path and generated dirty snapshot validation.
  - Restricted ignored paths to safe relative generated-evidence paths, so source-bearing paths cannot be hidden by the ignore mechanism.
  - Limited the downstream set to the VS3 human-gate derived envelopes and VS3 local checkpoint envelopes that are generated after the aggregate scenario report.
- `tests/scenario/test_scaffold_cli.py`
  - Added a positive regression test for downstream local checkpoint drift after scenario report generation.
  - Kept the existing negative tests for source-bearing generated dirty paths, missing non-ignored generated paths, and changed non-ignored generated evidence bytes.

## Proof Boundary

This checkpoint proves only a local/dev VS3 scenario-gate evidence-loop guard.

It does not claim:
- VS3-P readiness;
- production/on-prem readiness;
- real IdP readiness;
- real network readiness;
- live provider readiness;
- migration/restore readiness;
- independent security acceptance;
- human UX acceptance.

## Evidence

Commands run before final evidence refresh:

```text
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
exit 0
```

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_accepts_local_dev_claim_with_downstream_checkpoint_drift \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_changed_generated_dirty_bytes \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_missing_generated_dirty_paths \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_source_path_in_generated_dirty_paths
Ran 4 tests in 102.595s
OK
```

Final evidence refresh:

```text
./cornerstone scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
exit 0
status success
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
summary pass=50 human_required=7 blocking=0 scenario_count=57
```

```text
./cornerstone human-gate evidence-status --scope vs3 --record-dir reports/human-gates/vs3/record-templates --use-existing --json --output reports/human-gates/vs3/evidence-status.json
exit 0
status success
final_verdict HUMAN_REQUIRED
summary pass=50 human_required=7 blocking=0 vs3_l_claim=LOCAL_DEV_ASSURANCE_VERIFIED vs3_p_claim=NOT_CLAIMED
```

```text
./cornerstone human-gate review-kit --scope vs3 --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --use-existing --json --output reports/human-gates/vs3/review-kit.json
exit 0
status success
final_verdict HUMAN_REQUIRED
summary pass=50 human_required=7 blocking=0 vs3_l_claim=LOCAL_DEV_ASSURANCE_VERIFIED vs3_p_claim=NOT_CLAIMED
```

```text
./cornerstone human-gate vs3-p-gate --scope vs3 --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --output reports/human-gates/vs3/vs3-p-gate.json --json
exit 4
status blocked
final_verdict HUMAN_REQUIRED
expected error CS_VS3_P_GATE_HUMAN_EVIDENCE_REQUIRED
unresolved_human_required_rows=VS3-H01..VS3-H07
```

```text
./cornerstone security vs3-local-checkpoint --json --output reports/security/vs3-local-checkpoint.json
exit 0
status success
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
summary pass=50 human_required=7 blocking=0 vs3_l_claim=LOCAL_DEV_ASSURANCE_VERIFIED vs3_p_claim=NOT_CLAIMED
```

```text
./cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json
exit 0
status success
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
summary pass=50 human_required=7 fail=0 not_run=0 not_verified=0 blocking=0 scenario_count=57
generated_dirty_path_validation.status=passed
generated_dirty_snapshot_validation.status=passed
generated_dirty_snapshot_validation.hash_mismatch=false
ignored_paths=[
  reports/human-gates/vs3/evidence-status.json,
  reports/human-gates/vs3/review-kit.json,
  reports/human-gates/vs3/vs3-local-checkpoint.json,
  reports/human-gates/vs3/vs3-p-gate.json,
  reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json,
  reports/security/vs3-local-checkpoint.json
]
```

## Decision

The slice is verified for the local/dev scenario-gate evidence-loop boundary. Continue to the next small VS3 slice only after preserving the proof distinction: VS3-L local/dev assurance may be reported, while VS3-P and human/on-prem readiness remain blocked by `VS3-H01` through `VS3-H07`.
