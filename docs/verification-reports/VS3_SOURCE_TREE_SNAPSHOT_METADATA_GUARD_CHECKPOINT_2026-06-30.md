# VS3 Source Tree Snapshot Metadata Guard Checkpoint

**Date:** 2026-06-30 KST
**Owner:** JiYong / Tars
**Status:** Local deterministic checkpoint for one VS3 gate-hardening slice.
**Scope:** VS3-L local/dev scenario-gate evidence only.

## Slice Contract

Goal:

- Harden the VS3 scenario gate so each source-bearing entry in `source_tree.verified_source_snapshot_paths` carries valid and current per-file metadata.

In this slice:

- `VS3-GATE-003` - local/dev reports cannot overclaim with stale evidence metadata.
- `VS3-GATE-004` - the native `cornerstone scenario gate ... --json` path validates the report.
- `VS3-REG-004` - evidence and coverage gates detect malformed or stale source snapshot metadata.
- `VS3-REG-005` - claim boundaries remain no stronger than local/dev evidence.

Full VS3 scenario mapping remains active:

- `MUST_PASS`: 42 rows, `VS3-GATE-001` through `VS3-OBS-003`.
- `REGRESSION`: 8 rows, `VS3-REG-001` through `VS3-REG-008`.
- `HUMAN_REQUIRED`: 7 rows, `VS3-H01` through `VS3-H07`.

Out of scope:

- VS3-P production/on-prem readiness.
- Real IdP, real network, live-provider, migration/restore, independent security review, or human UX acceptance.
- Any broad refactor outside the source snapshot metadata guard.

Done criteria:

- A tampered local/dev report with stale `sha256` and `bytes` inside a source snapshot entry fails the gate.
- A valid local/dev report passes with `metadata_mismatches=[]`.
- Generated evidence paths remain rejected by taxonomy and do not produce noisy metadata mismatches.
- Aggregate VS3 verify/gate evidence remains local/dev only with all human gates explicit.

## Baseline Gap

Before this slice, a tampered report could change the first source snapshot entry to:

```json
{
  "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
  "bytes": 0
}
```

while keeping the top-level source-tree fingerprint and transcript copies aligned.

Observed baseline:

```text
returncode 0
status success
errors []
snapshot_validation {'duplicate_paths': [], 'generated_evidence_paths': [], 'invalid_entries': [], 'path_count': 16, 'schema_version': 'cs.vs3_source_tree_snapshot_paths.v0', 'status': 'passed'}
```

That meant the gate validated the source-tree fingerprint and path taxonomy, but not the current per-file metadata of each snapshot entry.

## Implementation Evidence

Code and tests:

- `packages/cornerstone_cli/main.py` adds per-entry validation for safe relative paths, `status`, `state`, `sha256`, `bytes`, and current file metadata.
- `packages/cornerstone_cli/main.py` reports `metadata_mismatches` inside `source_tree_snapshot_path_validation`.
- `packages/cornerstone_cli/main.py` includes `metadata_mismatches` in `CS_VS3_SOURCE_TREE_SNAPSHOT_PATHS_INVALID`.
- `tests/scenario/test_scaffold_cli.py` adds `test_vs3_scenario_gate_rejects_local_dev_claim_with_stale_source_snapshot_path_metadata`.
- The successful and generated-path tamper tests now assert `metadata_mismatches=[]` where applicable.

Focused regression command:

```bash
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_preserves_source_proof_boundary_and_transcript \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_generated_evidence_source_snapshot_path \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_stale_source_snapshot_path_metadata
```

Observed result:

```text
Ran 3 tests in 68.984s
OK
```

Aggregate VS3 report regeneration:

```bash
./cornerstone scenario verify vs3-onprem-trusted-extension --json \
  --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json \
  > /tmp/cs-vs3-snapshot-metadata-verify.json
```

Observed result:

- `status`: `success`
- `scenario_count`: `57`
- `pass`: `50`
- `human_required`: `7`
- `product_feature_claims`: `VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_GATES_REQUIRED`
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
  > /tmp/cs-vs3-snapshot-metadata-gate.json
```

Observed result:

- `status`: `success`
- `errors`: `[]`
- `source_tree_current_validation.status`: `passed`
- `source_tree_snapshot_path_validation.status`: `passed`
- `source_tree_snapshot_path_validation.path_count`: `16`
- `source_tree_snapshot_path_validation.generated_evidence_paths`: `[]`
- `source_tree_snapshot_path_validation.duplicate_paths`: `[]`
- `source_tree_snapshot_path_validation.invalid_entries`: `[]`
- `source_tree_snapshot_path_validation.metadata_mismatches`: `[]`
- `component_proof_validation.status`: `passed`
- `coverage_validation.status`: `passed`
- `negative_evidence_validation.status`: `passed`
- `claim_boundary_validation.status`: `passed`
- `traceability_validation.status`: `passed`
- `completion_claim_validation.status`: `passed`

## Decision

This slice is locally verified.

The gate now rejects stale or malformed source snapshot entry metadata while preserving the previous generated-evidence path taxonomy guard. This strengthens the VS3-L source evidence boundary without making any VS3-P, production, external-provider, migration, security-acceptance, or human-acceptance claim.

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
