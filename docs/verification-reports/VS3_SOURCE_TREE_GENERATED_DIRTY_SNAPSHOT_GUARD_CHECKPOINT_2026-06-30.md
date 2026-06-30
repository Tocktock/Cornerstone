# VS3 Source Tree Generated Dirty Snapshot Guard Checkpoint

**Date:** 2026-06-30 KST
**Owner:** JiYong / Tars
**Status:** Local/dev generated evidence byte-identity guard slice verified.
**Scope:** VS3 local/dev scenario-gate hardening only.

## Slice Contract

Goal:

- Make the VS3 local/dev scenario gate reject reports when a generated evidence dirty-path file changes after report generation.
- Preserve the existing source-bearing snapshot guard and generated dirty-path taxonomy guard.
- Avoid a self-referential report hash loop by excluding only the checked scenario report path from generated-evidence byte validation.

In this slice:

- `VS3-GATE-003` - local/dev reports must not overclaim unsupported readiness.
- `VS3-GATE-004` - native `cornerstone scenario verify/gate ... --json` must expose reliable gate metadata.
- `VS3-REG-004` - missing or stale evidence coverage must fail before release claims.
- `VS3-REG-005` - report wording must stay no stronger than evidence.

Full VS3 mapping:

- In this slice: `VS3-GATE-003`, `VS3-GATE-004`, `VS3-REG-004`, `VS3-REG-005`.
- Later AI slices: `VS3-GATE-001`, `VS3-GATE-002`, `VS3-CTX-001`, `VS3-CTX-002`, `VS3-CTX-003`, `VS3-CTX-004`, `VS3-CTX-005`, `VS3-RLS-001`, `VS3-RLS-002`, `VS3-RLS-003`, `VS3-RLS-004`, `VS3-RLS-005`, `VS3-RLS-006`, `VS3-OPA-001`, `VS3-OPA-002`, `VS3-OPA-003`, `VS3-OPA-004`, `VS3-OPA-005`, `VS3-EGR-001`, `VS3-EGR-002`, `VS3-EGR-003`, `VS3-EGR-004`, `VS3-EGR-005`, `VS3-EGR-006`, `VS3-CON-001`, `VS3-CON-002`, `VS3-CON-003`, `VS3-CON-004`, `VS3-CON-005`, `VS3-CON-006`, `VS3-TOOL-001`, `VS3-TOOL-002`, `VS3-TOOL-003`, `VS3-TOOL-004`, `VS3-TOOL-005`, `VS3-TOOL-006`, `VS3-TOOL-007`, `VS3-OBS-001`, `VS3-OBS-002`, `VS3-OBS-003`, `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-003`, `VS3-REG-006`, `VS3-REG-007`, `VS3-REG-008`.
- Human-required rows: `VS3-H01`, `VS3-H02`, `VS3-H03`, `VS3-H04`, `VS3-H05`, `VS3-H06`, `VS3-H07`.

Non-scope:

- No VS3-P production/on-prem readiness claim.
- No real IdP, real network, live provider, independent security review, human UX acceptance, production migration, or restore readiness claim.
- No product feature expansion beyond current generated-evidence byte-identity validation.

## Baseline Gap

Before this slice, a fresh VS3 report still passed the gate after a generated evidence checkpoint file was changed on disk:

```text
target docs/verification-reports/VS3_SOURCE_TREE_GENERATED_DIRTY_PATH_COVERAGE_GUARD_CHECKPOINT_2026-06-30.md
exit 0
status success
errors []
generated_dirty passed missing 0 stale []
source_current passed []
```

That meant path coverage alone could hide changed generated evidence bytes while still preserving a local/dev assurance claim.

## Implementation

Changed report metadata:

- `packages/cornerstone_cli/acceptance.py` now records `generated_dirty_snapshot_hash`.
- `packages/cornerstone_cli/acceptance.py` now records `generated_dirty_snapshot_paths` with `path`, `status`, `state`, `sha256`, and `bytes` for generated evidence dirty paths.

Changed gate validation:

- `packages/cornerstone_cli/main.py` now validates generated evidence snapshot entries against the current generated evidence snapshot.
- `packages/cornerstone_cli/main.py` now rejects missing, stale, duplicate, invalid, or byte-mismatched generated evidence snapshot entries.
- `packages/cornerstone_cli/main.py` now emits `CS_VS3_SOURCE_TREE_GENERATED_DIRTY_SNAPSHOT_INVALID`.
- The checked scenario report path is ignored for this byte-identity validation because the report writes its final bytes after the source tree metadata is captured. The path still remains visible in `generated_dirty_paths`; only its self-hash comparison is excluded.

Changed regression coverage:

- `tests/scenario/test_scaffold_cli.py` adds `test_vs3_scenario_gate_rejects_local_dev_claim_with_changed_generated_dirty_bytes`.
- The VS3 local/dev success-path test now asserts `generated_dirty_snapshot_validation.status == passed`.

## Verification Evidence

Syntax:

```text
python3 -m py_compile packages/cornerstone_cli/acceptance.py packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
exit: 0
```

Focused tests:

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_preserves_source_proof_boundary_and_transcript \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_changed_generated_dirty_bytes

Ran 2 tests in 47.483s
OK
```

Neighboring source-tree and generated-dirty-path tests:

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_preserves_source_proof_boundary_and_transcript \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_changed_generated_dirty_bytes \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_missing_generated_dirty_paths \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_source_path_in_generated_dirty_paths \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_missing_source_dirty_path \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_stale_source_tree \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_stale_post_commit_source_tree \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_tampered_source_snapshot_status

Ran 8 tests in 190.200s
OK
```

Component proof refresh after source changes:

```text
./cornerstone security vs3-request-context --json
./cornerstone security vs3-postgres-rls --json
./cornerstone security vs3-opa-policy --json
./cornerstone security vs3-egress-sandbox --json
./cornerstone security vs3-connectorhub-source --json
./cornerstone security vs3-tool-registry --json
./cornerstone security vs3-observability --json
exit: 0
```

Refreshed component source tree hash:

```text
711b6d1fff29f2b7873b7133020d11c997d0f59bebc6d7d747b3045d60fa59ba
```

Pre-checkpoint canonical VS3 local/dev report and gate:

```text
./cornerstone scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
./cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json
exit: 0
```

Gate summary:

```text
verify_status success VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
gate_status success VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
summary {'blocking': 0, 'fail': 0, 'human_required': 7, 'not_run': 0, 'not_verified': 0, 'pass': 50, 'product_feature_claims': 'VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_GATES_REQUIRED', 'scenario_count': 57}
generated_dirty_snapshot passed 133 False
ignored_paths ['reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json']
errors []
source_tree_hash 711b6d1fff29f2b7873b7133020d11c997d0f59bebc6d7d747b3045d60fa59ba
generated_snapshot_paths 134
```

Post-checkpoint stale-report guard:

```text
./cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json
exit: 4

status failed
errors ['CS_VS3_SOURCE_TREE_GENERATED_DIRTY_PATHS_INVALID', 'CS_VS3_SOURCE_TREE_GENERATED_DIRTY_SNAPSHOT_INVALID']
generated_dirty_path failed missing ['docs/verification-reports/VS3_SOURCE_TREE_GENERATED_DIRTY_SNAPSHOT_GUARD_CHECKPOINT_2026-06-30.md']
generated_dirty_snapshot failed missing ['docs/verification-reports/VS3_SOURCE_TREE_GENERATED_DIRTY_SNAPSHOT_GUARD_CHECKPOINT_2026-06-30.md'] hash_mismatch True
```

Final canonical VS3 local/dev report and gate after this checkpoint path was included:

```text
./cornerstone scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
./cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json
exit: 0

verify_status success VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
gate_status success VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
summary {'blocking': 0, 'fail': 0, 'human_required': 7, 'not_run': 0, 'not_verified': 0, 'pass': 50, 'product_feature_claims': 'VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_GATES_REQUIRED', 'scenario_count': 57}
generated_dirty_path passed count 136 missing 0
generated_dirty_snapshot passed recorded 135 current 135 hash_mismatch False
ignored_paths ['reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json']
errors []
has_checkpoint True
source_tree_hash 711b6d1fff29f2b7873b7133020d11c997d0f59bebc6d7d747b3045d60fa59ba
```

## Proof Boundary

This checkpoint proves only that the local/dev VS3 scenario gate rejects changed generated evidence dirty-path bytes, excluding the checked scenario report file's unavoidable self-write.

It does not prove:

- production/on-prem readiness;
- real IdP readiness;
- live provider readiness;
- independent security acceptance;
- human UX acceptance;
- migration/restore readiness.

Those remain governed by `VS3-H01` through `VS3-H07`.

## Decision

The generated dirty snapshot guard slice is ready to keep as part of the VS3 local/dev gate.
