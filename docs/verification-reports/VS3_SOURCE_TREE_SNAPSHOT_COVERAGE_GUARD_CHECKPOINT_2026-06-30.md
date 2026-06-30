# VS3 Source Tree Snapshot Coverage Guard Checkpoint

**Date:** 2026-06-30 KST
**Owner:** JiYong / Tars
**Status:** Local deterministic checkpoint for one VS3 gate-hardening slice.
**Scope:** VS3-L local/dev scenario-gate evidence only.

## Slice Contract

Goal:

- Harden the VS3 scenario gate so `source_tree.verified_source_snapshot_paths` cannot silently omit current source-bearing snapshot entries while retaining the same source-tree hash and dirty path lists.

In this slice:

- `VS3-GATE-003` - local/dev reports cannot overclaim by dropping source snapshot path-level evidence.
- `VS3-GATE-004` - the native `cornerstone scenario gate ... --json` path validates snapshot coverage.
- `VS3-REG-004` - evidence and coverage gates detect source snapshot omissions before a release claim.
- `VS3-REG-005` - claim boundaries remain no stronger than local/dev evidence.

Full VS3 scenario mapping remains active:

- `in_this_slice`: `VS3-GATE-003`, `VS3-GATE-004`, `VS3-REG-004`, `VS3-REG-005`.
- `later_slice`: `VS3-GATE-001`, `VS3-GATE-002`, `VS3-CTX-001`, `VS3-CTX-002`, `VS3-CTX-003`, `VS3-CTX-004`, `VS3-CTX-005`, `VS3-RLS-001`, `VS3-RLS-002`, `VS3-RLS-003`, `VS3-RLS-004`, `VS3-RLS-005`, `VS3-RLS-006`, `VS3-OPA-001`, `VS3-OPA-002`, `VS3-OPA-003`, `VS3-OPA-004`, `VS3-OPA-005`, `VS3-EGR-001`, `VS3-EGR-002`, `VS3-EGR-003`, `VS3-EGR-004`, `VS3-EGR-005`, `VS3-EGR-006`, `VS3-CON-001`, `VS3-CON-002`, `VS3-CON-003`, `VS3-CON-004`, `VS3-CON-005`, `VS3-CON-006`, `VS3-TOOL-001`, `VS3-TOOL-002`, `VS3-TOOL-003`, `VS3-TOOL-004`, `VS3-TOOL-005`, `VS3-TOOL-006`, `VS3-TOOL-007`, `VS3-OBS-001`, `VS3-OBS-002`, `VS3-OBS-003`, `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-003`, `VS3-REG-006`, `VS3-REG-007`, `VS3-REG-008`.
- `HUMAN_REQUIRED`: `VS3-H01`, `VS3-H02`, `VS3-H03`, `VS3-H04`, `VS3-H05`, `VS3-H06`, `VS3-H07`.
- `blocked`: none for this slice.
- `out_of_scope`: VS3-P production/on-prem readiness, real IdP, real network, live-provider, migration/restore, independent security review, human UX acceptance, PR/merge/deployment/release state.

Done criteria:

- A tampered local/dev report missing a current source snapshot path fails the gate.
- The older source-tree validators still pass for that tamper, proving this guard covers a distinct gap.
- A valid local/dev report passes with `missing_current_snapshot_paths=[]`, `stale_snapshot_paths=[]`, `duplicate_paths=[]`, `current_duplicate_paths=[]`, `invalid_entries=[]`, and `current_invalid_entries=[]`.
- Aggregate VS3 verify/gate evidence remains local/dev only with all human gates explicit.

## Baseline Gap

Before this slice, this mutation passed:

```text
verify_returncode 0
target_removed packages/cornerstone_cli/main.py
gate_returncode 0
gate_status success
error_codes []
source_tree_current_validation.status passed
source_tree_snapshot_path_validation.status passed
dirty_path_validation.status passed
generated_dirty_path_validation.status passed
```

The tamper removed `packages/cornerstone_cli/main.py` from `source_tree.verified_source_snapshot_paths`, but preserved `verified_source_worktree_hash`, `dirty_paths`, `generated_dirty_paths`, command transcripts, and self-command transcript source-tree copies.

That meant the gate could prove source snapshot entries were individually well-shaped, but could not prove the recorded list covered all current source-bearing snapshot entries.

## Implementation Evidence

Code and tests:

- `packages/cornerstone_cli/main.py` adds `_vs3_source_tree_snapshot_coverage_validation`.
- `packages/cornerstone_cli/main.py` includes `verified_source_snapshot_paths` in `_vs3_current_source_tree`.
- `packages/cornerstone_cli/main.py` includes `source_tree_snapshot_coverage_validation` in the gate payload and source report summary.
- `packages/cornerstone_cli/main.py` fails local/dev assurance reports with `CS_VS3_SOURCE_TREE_SNAPSHOT_COVERAGE_INVALID` when recorded snapshot paths omit or add source-bearing paths relative to the current source tree.
- `tests/scenario/test_scaffold_cli.py` asserts the valid gate path keeps snapshot coverage clean.
- `tests/scenario/test_scaffold_cli.py` adds `test_vs3_scenario_gate_rejects_local_dev_claim_with_missing_source_snapshot_path`.

Syntax and focused regression command:

```bash
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py && \
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_preserves_source_proof_boundary_and_transcript \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_missing_source_snapshot_path
```

Observed result:

```text
Ran 2 tests in 45.981s
OK
```

Adjacent source-tree guard regression command:

```bash
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_missing_source_dirty_path \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_source_path_in_generated_dirty_paths \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_generated_evidence_source_snapshot_path \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_stale_source_snapshot_path_metadata
```

Observed result:

```text
Ran 4 tests in 92.006s
OK
```

Aggregate VS3 report regeneration:

```bash
./cornerstone scenario verify vs3-onprem-trusted-extension --json \
  --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json \
  > /tmp/cs-vs3-snapshot-coverage-verify.json
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

Aggregate VS3 gate:

```bash
./cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json \
  > /tmp/cs-vs3-snapshot-coverage-gate.json
```

Observed result:

- `status`: `success`
- `scenario_count`: `57`
- `blocking_count`: `0`
- `errors`: `[]`
- `source_tree_current_validation.status`: `passed`
- `source_tree_current_validation.mismatches`: `[]`
- `source_tree_snapshot_path_validation.status`: `passed`
- `source_tree_snapshot_coverage_validation.status`: `passed`
- `source_tree_snapshot_coverage_validation.recorded_path_count`: `16`
- `source_tree_snapshot_coverage_validation.current_path_count`: `16`
- `source_tree_snapshot_coverage_validation.missing_current_snapshot_paths`: `[]`
- `source_tree_snapshot_coverage_validation.stale_snapshot_paths`: `[]`
- `source_tree_snapshot_coverage_validation.duplicate_paths`: `[]`
- `source_tree_snapshot_coverage_validation.current_duplicate_paths`: `[]`
- `source_tree_snapshot_coverage_validation.invalid_entries`: `[]`
- `source_tree_snapshot_coverage_validation.current_invalid_entries`: `[]`
- `dirty_path_validation.status`: `passed`
- `generated_dirty_path_validation.status`: `passed`
- `component_proof_validation.status`: `passed`
- `coverage_validation.status`: `passed`
- `claim_boundary_validation.status`: `passed`
- `human_required_validation.status`: `passed`
- `scenario_gate_conditions.source_tree_snapshot_coverage`: `true`
- `scenario_gate_summary.source_tree_snapshot_coverage_failures`: `0`

Post-checkpoint gate and docs checks:

```bash
./cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json \
  > /tmp/cs-vs3-snapshot-coverage-post-checkpoint-gate.json

scripts/verify_sot_docs.sh
git diff --check
```

Observed result:

- post-checkpoint `status`: `success`
- post-checkpoint `scenario_count`: `57`
- post-checkpoint `blocking_count`: `0`
- post-checkpoint `errors`: `[]`
- post-checkpoint `source_tree_snapshot_coverage_validation.status`: `passed`
- post-checkpoint `source_tree_snapshot_coverage_validation.recorded_path_count`: `16`
- post-checkpoint `source_tree_snapshot_coverage_validation.current_path_count`: `16`
- `scripts/verify_sot_docs.sh`: `PASS: CornerStone SoT docs verified`
- `git diff --check`: exit code `0`

## Decision

This slice is locally verified.

The VS3 gate now rejects source snapshot path omissions that previously passed while preserving source-tree hash, dirty path, generated dirty path, transcript, claim-boundary, and human-required validations. This strengthens the VS3-L local/dev source evidence boundary without making any VS3-P, production, external-provider, migration, security-acceptance, or human-acceptance claim.

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
