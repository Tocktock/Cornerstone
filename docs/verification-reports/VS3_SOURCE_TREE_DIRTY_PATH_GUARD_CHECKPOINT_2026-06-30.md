# VS3 Source Tree Dirty Path Guard Checkpoint

**Date:** 2026-06-30 KST
**Owner:** JiYong / Tars
**Status:** Local deterministic checkpoint for one VS3 gate-hardening slice.
**Scope:** VS3-L local/dev scenario-gate evidence only.

## Slice Contract

Goal:

- Harden the VS3 scenario gate so `source_tree.dirty_paths` is safe, unique, and consistent with the current source-bearing dirty paths.

In this slice:

- `VS3-GATE-003` - local/dev reports cannot overclaim by omitting source-bearing dirty paths from source-tree metadata.
- `VS3-GATE-004` - the native `cornerstone scenario gate ... --json` path validates the report.
- `VS3-REG-004` - evidence and coverage gates detect malformed, duplicate, stale, or incomplete source-tree dirty path metadata.
- `VS3-REG-005` - claim boundaries remain no stronger than local/dev evidence.

Full VS3 scenario mapping remains active:

- `MUST_PASS`: 42 rows, `VS3-GATE-001` through `VS3-OBS-003`.
- `REGRESSION`: 8 rows, `VS3-REG-001` through `VS3-REG-008`.
- `HUMAN_REQUIRED`: 7 rows, `VS3-H01` through `VS3-H07`.

Out of scope:

- VS3-P production/on-prem readiness.
- Real IdP, real network, live-provider, migration/restore, independent security review, or human UX acceptance.
- Any broad refactor outside the source-tree dirty path guard.

Done criteria:

- A tampered local/dev report missing a current source-bearing dirty path fails the gate.
- A valid local/dev report passes with `duplicate_paths=[]`, `stale_source_dirty_paths=[]`, `missing_current_source_dirty_paths=[]`, and `invalid_entries=[]`.
- Generated evidence files may be newly added after the report without invalidating source-bearing dirty path consistency.
- Aggregate VS3 verify/gate evidence remains local/dev only with all human gates explicit.

## Baseline Gap

Before this slice, a tampered report could remove a source-bearing dirty path:

```json
{
  "source_tree": {
    "dirty_paths": [
      "... all original entries except packages/cornerstone_cli/main.py ..."
    ]
  }
}
```

while keeping the source snapshot, command transcript, and self-command transcript source-tree copies aligned.

Observed baseline:

```text
returncode 0
status success
errors []
source_tree_current_validation.status passed
generated_dirty_path_validation.status passed
dirty_path_validation None
```

A second tamper that appended `../outside` to `source_tree.dirty_paths` also returned:

```text
returncode 0
status success
errors []
```

That meant the gate validated source snapshot hashes and generated dirty path classification, but did not validate the dirty path list itself.

## Implementation Evidence

Code and tests:

- `packages/cornerstone_cli/main.py` adds `_vs3_source_tree_dirty_path_validation`.
- `packages/cornerstone_cli/main.py` includes `dirty_path_validation` in the gate payload and source report summary.
- `packages/cornerstone_cli/main.py` fails local/dev assurance reports with `CS_VS3_SOURCE_TREE_DIRTY_PATHS_INVALID` when dirty path validation fails.
- `tests/scenario/test_scaffold_cli.py` asserts the valid gate path keeps dirty path validation clean.
- `tests/scenario/test_scaffold_cli.py` adds `test_vs3_scenario_gate_rejects_local_dev_claim_with_missing_source_dirty_path`.

Focused regression command:

```bash
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_preserves_source_proof_boundary_and_transcript \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_missing_source_dirty_path
```

Observed result:

```text
Ran 2 tests in 46.020s
OK
```

Adjacent source-tree guard regression command:

```bash
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_source_path_in_generated_dirty_paths \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_generated_evidence_source_snapshot_path \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_stale_source_snapshot_path_metadata
```

Observed result:

```text
Ran 3 tests in 69.079s
OK
```

Syntax check:

```bash
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
```

Observed result:

```text
exit code 0
```

Aggregate VS3 report regeneration:

```bash
./cornerstone scenario verify vs3-onprem-trusted-extension --json \
  --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json \
  > /tmp/cs-vs3-dirty-paths-verify.json
```

Observed result:

- `status`: `success`
- `final_verdict`: `VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED`
- `summary.scenario_count`: `57`
- `summary.pass`: `50`
- `summary.human_required`: `7`
- `summary.product_feature_claims`: `VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_GATES_REQUIRED`
- `source_tree.dirty_paths`: `138`
- `source_tree.generated_dirty_paths`: `129`
- `source_tree.verified_source_snapshot_paths`: `16`
- `proof_boundary.vs3_l`: `LOCAL_DEV_ASSURANCE_VERIFIED`
- `proof_boundary.vs3_p`: `NOT_CLAIMED`
- `proof_boundary.production_onprem`: `NOT_CLAIMED`
- `proof_boundary.live_provider`: `NOT_CLAIMED`
- `proof_boundary.real_idp`: `NOT_CLAIMED`
- `proof_boundary.real_network`: `NOT_CLAIMED`
- `proof_boundary.migration_restore`: `NOT_CLAIMED`
- `proof_boundary.security_acceptance`: `NOT_CLAIMED`
- `proof_boundary.human_acceptance`: `NOT_CLAIMED`

Aggregate VS3 gate:

```bash
./cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json \
  > /tmp/cs-vs3-dirty-paths-gate.json
```

Observed result:

- `status`: `success`
- `scenario_count`: `57`
- `blocking_count`: `0`
- `errors`: `[]`
- `source_tree_current_validation.status`: `passed`
- `source_tree_current_validation.mismatches`: `[]`
- `source_tree_snapshot_path_validation.status`: `passed`
- `source_tree_snapshot_path_validation.path_count`: `16`
- `source_tree_snapshot_path_validation.generated_evidence_paths`: `[]`
- `source_tree_snapshot_path_validation.duplicate_paths`: `[]`
- `source_tree_snapshot_path_validation.invalid_entries`: `[]`
- `dirty_path_validation.status`: `passed`
- `dirty_path_validation.path_count`: `138`
- `dirty_path_validation.source_path_count`: `9`
- `dirty_path_validation.duplicate_paths`: `[]`
- `dirty_path_validation.stale_source_dirty_paths`: `[]`
- `dirty_path_validation.missing_current_source_dirty_paths`: `[]`
- `dirty_path_validation.invalid_entries`: `[]`
- `generated_dirty_path_validation.status`: `passed`
- `generated_dirty_path_validation.path_count`: `129`
- `generated_dirty_path_validation.non_generated_paths`: `[]`
- `generated_dirty_path_validation.duplicate_paths`: `[]`
- `generated_dirty_path_validation.stale_paths`: `[]`
- `generated_dirty_path_validation.dirty_path_missing_paths`: `[]`
- `generated_dirty_path_validation.invalid_entries`: `[]`
- `component_proof_validation.status`: `passed`
- `coverage_validation.status`: `passed`
- `claim_boundary_validation.status`: `passed`
- `human_required_validation.status`: `passed`
- `source_transcript_validation.status`: `passed`

## Decision

This slice is locally verified.

The gate now rejects unsafe, duplicate, stale, or incomplete source-bearing entries in `source_tree.dirty_paths`. This strengthens the VS3-L local/dev source evidence boundary without making any VS3-P, production, external-provider, migration, security-acceptance, or human-acceptance claim.

## Remaining Gates

`VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`.

This checkpoint does not prove:

- VS3-P production/on-prem readiness.
- Real IdP readiness.
- Real network readiness.
- Live-provider readiness.
- Migration/restore readiness.
- Independent security acceptance.
- Human UX acceptance.
- PR, merge, deployment, or release state.
