# VS3 Source Tree Generated Dirty Path Guard Checkpoint

**Date:** 2026-06-30 KST
**Owner:** JiYong / Tars
**Status:** Local deterministic checkpoint for one VS3 gate-hardening slice.
**Scope:** VS3-L local/dev scenario-gate evidence only.

## Slice Contract

Goal:

- Harden the VS3 scenario gate so `source_tree.generated_dirty_paths` can only contain generated evidence paths that are still dirty in the current source tree.

In this slice:

- `VS3-GATE-003` - local/dev reports cannot overclaim by hiding source-bearing dirty paths inside generated-evidence bookkeeping.
- `VS3-GATE-004` - the native `cornerstone scenario gate ... --json` path validates the report.
- `VS3-REG-004` - evidence and coverage gates detect malformed, stale, duplicate, or source-bearing generated dirty path entries.
- `VS3-REG-005` - claim boundaries remain no stronger than local/dev evidence.

Full VS3 scenario mapping remains active:

- `MUST_PASS`: 42 rows, `VS3-GATE-001` through `VS3-OBS-003`.
- `REGRESSION`: 8 rows, `VS3-REG-001` through `VS3-REG-008`.
- `HUMAN_REQUIRED`: 7 rows, `VS3-H01` through `VS3-H07`.

Out of scope:

- VS3-P production/on-prem readiness.
- Real IdP, real network, live-provider, migration/restore, independent security review, or human UX acceptance.
- Any broad refactor outside the generated dirty path guard.

Done criteria:

- A tampered local/dev report with a source-bearing path in `generated_dirty_paths` fails the gate.
- A valid local/dev report passes with `non_generated_paths=[]`, `duplicate_paths=[]`, `stale_paths=[]`, `dirty_path_missing_paths=[]`, and `invalid_entries=[]`.
- Source snapshot path taxonomy and metadata guards still pass for a valid report.
- Aggregate VS3 verify/gate evidence remains local/dev only with all human gates explicit.

## Baseline Gap

Before this slice, a tampered report could set:

```json
{
  "source_tree": {
    "generated_dirty_paths": ["packages/cornerstone_cli/main.py"]
  }
}
```

while keeping the command transcript and self-command transcript source-tree copies aligned.

Observed baseline:

```text
returncode 0
status success
errors []
source_tree_current_validation.status passed
source_tree_snapshot_path_validation.status passed
```

That meant the gate validated the source-tree fingerprint and source snapshot path taxonomy, but not whether `generated_dirty_paths` was restricted to generated evidence files.

## Implementation Evidence

Code and tests:

- `packages/cornerstone_cli/main.py` adds `_vs3_is_safe_relative_path`.
- `packages/cornerstone_cli/main.py` adds `_vs3_source_tree_generated_dirty_path_validation`.
- `packages/cornerstone_cli/main.py` exposes `generated_dirty_paths` in the current source-tree snapshot.
- `packages/cornerstone_cli/main.py` includes `generated_dirty_path_validation` in the gate payload and source report summary.
- `packages/cornerstone_cli/main.py` fails local/dev assurance reports with `CS_VS3_SOURCE_TREE_GENERATED_DIRTY_PATHS_INVALID` when generated dirty path validation fails.
- `tests/scenario/test_scaffold_cli.py` asserts the valid gate path keeps generated dirty path validation clean.
- `tests/scenario/test_scaffold_cli.py` adds `test_vs3_scenario_gate_rejects_local_dev_claim_with_source_path_in_generated_dirty_paths`.

Focused regression command:

```bash
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_preserves_source_proof_boundary_and_transcript \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_source_path_in_generated_dirty_paths
```

Observed result:

```text
Ran 2 tests in 46.154s
OK
```

Adjacent source-tree guard regression command:

```bash
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_generated_evidence_source_snapshot_path \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_stale_source_snapshot_path_metadata
```

Observed result:

```text
Ran 2 tests in 46.089s
OK
```

Aggregate VS3 report regeneration:

```bash
./cornerstone scenario verify vs3-onprem-trusted-extension --json \
  --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json \
  > /tmp/cs-vs3-generated-dirty-paths-verify.json
```

Observed result:

- `status`: `success`
- `final_verdict`: `VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED`
- `summary.scenario_count`: `57`
- `summary.pass`: `50`
- `summary.human_required`: `7`
- `summary.product_feature_claims`: `VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_GATES_REQUIRED`
- `source_tree.dirty_paths`: `137`
- `source_tree.generated_dirty_paths`: `128`

Aggregate VS3 gate:

```bash
./cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json \
  > /tmp/cs-vs3-generated-dirty-paths-gate.json
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
- `generated_dirty_path_validation.status`: `passed`
- `generated_dirty_path_validation.path_count`: `128`
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

The gate now rejects source-bearing, unsafe, duplicate, stale, or no-longer-dirty entries in `source_tree.generated_dirty_paths`. This strengthens the VS3-L local/dev source evidence boundary without making any VS3-P, production, external-provider, migration, security-acceptance, or human-acceptance claim.

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
