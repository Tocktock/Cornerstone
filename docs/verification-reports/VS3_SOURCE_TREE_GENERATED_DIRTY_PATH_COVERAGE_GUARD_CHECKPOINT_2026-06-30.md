# VS3 Source Tree Generated Dirty Path Coverage Guard Checkpoint

**Date:** 2026-06-30 KST
**Owner:** JiYong / Tars
**Status:** Local/dev generated dirty path coverage guard slice verified.
**Scope:** VS3 local/dev scenario-gate hardening only.

## Slice Contract

Goal:

- Make the VS3 local/dev scenario gate reject reports that omit current generated evidence dirty paths from `source_tree.generated_dirty_paths`.
- Preserve the existing guard that rejects source-bearing paths incorrectly listed as generated dirty paths.

In this slice:

- `VS3-GATE-003` - local/dev reports must not overclaim unsupported readiness.
- `VS3-GATE-004` - native `cornerstone scenario verify/gate ... --json` must expose reliable gate metadata.
- `VS3-REG-004` - missing or stale source/evidence coverage must fail before release claims.
- `VS3-REG-005` - report wording must stay no stronger than evidence.

Full VS3 mapping:

- In this slice: `VS3-GATE-003`, `VS3-GATE-004`, `VS3-REG-004`, `VS3-REG-005`.
- Later AI slices: `VS3-GATE-001`, `VS3-GATE-002`, `VS3-CTX-001`, `VS3-CTX-002`, `VS3-CTX-003`, `VS3-CTX-004`, `VS3-CTX-005`, `VS3-RLS-001`, `VS3-RLS-002`, `VS3-RLS-003`, `VS3-RLS-004`, `VS3-RLS-005`, `VS3-RLS-006`, `VS3-OPA-001`, `VS3-OPA-002`, `VS3-OPA-003`, `VS3-OPA-004`, `VS3-OPA-005`, `VS3-EGR-001`, `VS3-EGR-002`, `VS3-EGR-003`, `VS3-EGR-004`, `VS3-EGR-005`, `VS3-EGR-006`, `VS3-CON-001`, `VS3-CON-002`, `VS3-CON-003`, `VS3-CON-004`, `VS3-CON-005`, `VS3-CON-006`, `VS3-TOOL-001`, `VS3-TOOL-002`, `VS3-TOOL-003`, `VS3-TOOL-004`, `VS3-TOOL-005`, `VS3-TOOL-006`, `VS3-TOOL-007`, `VS3-OBS-001`, `VS3-OBS-002`, `VS3-OBS-003`, `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-003`, `VS3-REG-006`, `VS3-REG-007`, `VS3-REG-008`.
- Human-required rows: `VS3-H01`, `VS3-H02`, `VS3-H03`, `VS3-H04`, `VS3-H05`, `VS3-H06`, `VS3-H07`.

Non-scope:

- No VS3-P production/on-prem readiness claim.
- No real IdP, real network, live provider, independent security review, human UX acceptance, production migration, or restore readiness claim.
- No product feature expansion beyond current generated-evidence boundary validation.

## Baseline Gap

Before this slice, the VS3 gate accepted a local/dev report where `source_tree.generated_dirty_paths` was replaced with an empty list even though the current tree had generated evidence dirty paths:

```text
current report generated_dirty_paths count 133
empty return 0 status success errors [] generated_validation {'dirty_path_missing_paths': [], 'duplicate_paths': [], 'invalid_entries': [], 'non_generated_paths': [], 'path_count': 0, 'schema_version': 'cs.vs3_generated_dirty_paths.v0', 'stale_paths': [], 'status': 'passed'}
```

That meant a report could hide generated evidence churn while still claiming local/dev assurance.

## Implementation

Changed generated dirty path validation:

- `packages/cornerstone_cli/main.py` now computes `missing_current_generated_dirty_paths` by comparing recorded generated evidence dirty paths to the current source tree.
- `packages/cornerstone_cli/main.py` now fails `generated_dirty_path_validation` when current generated evidence dirty paths are omitted.
- `CS_VS3_SOURCE_TREE_GENERATED_DIRTY_PATHS_INVALID` now includes `missing_current_generated_dirty_paths` in its error payload.

Changed regression coverage:

- `tests/scenario/test_scaffold_cli.py` adds `test_vs3_scenario_gate_rejects_local_dev_claim_with_missing_generated_dirty_paths`.
- The existing generated-dirty-path taxonomy test now also accepts the stronger missing-current evidence detail.

## Verification Evidence

Syntax and focused tests:

```text
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
exit: 0

python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_preserves_source_proof_boundary_and_transcript \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_missing_generated_dirty_paths

Ran 2 tests in 45.529s
OK
```

Neighboring source-tree and dirty-path tests:

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_preserves_source_proof_boundary_and_transcript \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_missing_generated_dirty_paths \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_source_path_in_generated_dirty_paths \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_missing_source_dirty_path \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_stale_source_tree \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_stale_post_commit_source_tree

Ran 6 tests in 139.124s
OK
```

Component proof refresh:

```text
./cornerstone security vs3-request-context --json > /tmp/cs-vs3-gen-dirty-refresh-request-context.json
./cornerstone security vs3-postgres-rls --json > /tmp/cs-vs3-gen-dirty-refresh-postgres-rls.json
./cornerstone security vs3-opa-policy --json > /tmp/cs-vs3-gen-dirty-refresh-opa-policy.json
./cornerstone security vs3-egress-sandbox --json > /tmp/cs-vs3-gen-dirty-refresh-egress-sandbox.json
./cornerstone security vs3-connectorhub-source --json > /tmp/cs-vs3-gen-dirty-refresh-connectorhub-source.json
./cornerstone security vs3-tool-registry --json > /tmp/cs-vs3-gen-dirty-refresh-tool-registry.json
./cornerstone security vs3-observability --json > /tmp/cs-vs3-gen-dirty-refresh-observability.json
exit: 0
```

Canonical VS3 local/dev report and gate:

```text
./cornerstone scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json > /tmp/cs-vs3-generated-dirty-coverage-verify.json
./cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json > /tmp/cs-vs3-generated-dirty-coverage-gate.json
exit: 0
```

Gate summary:

```text
status success
scenario_count 57
summary {'blocking': 0, 'fail': 0, 'human_required': 7, 'not_run': 0, 'not_verified': 0, 'pass': 50, 'product_feature_claims': 'VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_GATES_REQUIRED', 'scenario_count': 57}
component_status passed
source_tree_current passed []
generated_dirty passed count 133 missing 0
errors []
```

Tamper rehearsal:

```text
./cornerstone scenario gate /tmp/cs-vs3-generated-dirty-coverage-tampered-report.json --json > /tmp/cs-vs3-generated-dirty-coverage-tampered-gate.json
exit: 4
```

Tamper result:

```text
status failed
errors[0].code CS_VS3_SOURCE_TREE_GENERATED_DIRTY_PATHS_INVALID
generated_dirty_status failed
path_count 0
non_generated_paths []
duplicate_paths []
stale_paths []
dirty_path_missing_paths []
missing_current_generated_dirty_paths_count 133
missing_current_generated_dirty_paths_sample ['docs/verification-reports/VS1_ONTOLOGY_AUTO_SUGGEST_PROMOTE_REPORT_2026-06-15.md', 'docs/verification-reports/VS3_CONNECTORHUB_SOURCE_CHECKPOINT_2026-06-29.md', 'docs/verification-reports/VS3_CONTRACT_MATRIX_CONSISTENCY_CHECKPOINT_2026-06-29.md', 'docs/verification-reports/VS3_EGRESS_SANDBOX_CHECKPOINT_2026-06-29.md', 'docs/verification-reports/VS3_EVIDENCE_RECONCILIATION_AND_OVERCLAIM_CHECKPOINT_2026-06-29.md', 'docs/verification-reports/VS3_HUMAN_GATE_DEPENDENCY_ORDER_GUARD_CHECKPOINT_2026-06-29.md', 'docs/verification-reports/VS3_HUMAN_GATE_DERIVED_REPORT_SELF_TRANSCRIPT_GUARD_CHECKPOINT_2026-06-29.md', 'docs/verification-reports/VS3_HUMAN_GATE_DERIVED_SCENARIO_TRACEABILITY_GUARD_CHECKPOINT_2026-06-30.md', 'docs/verification-reports/VS3_HUMAN_GATE_EVIDENCE_INTAKE_CHECKPOINT_2026-06-29.md', 'docs/verification-reports/VS3_HUMAN_GATE_TEMPLATE_STATUS_CHECKPOINT_2026-06-29.md']
invalid_entries []
source_tree_current passed []
component_status passed
```

Post-checkpoint stale-report guard:

```text
./cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json > /tmp/cs-vs3-generated-dirty-coverage-post-checkpoint-gate.json
exit: 4

status failed
errors ['CS_VS3_SOURCE_TREE_GENERATED_DIRTY_PATHS_INVALID']
generated_dirty failed missing_count 1
missing_sample ['docs/verification-reports/VS3_SOURCE_TREE_GENERATED_DIRTY_PATH_COVERAGE_GUARD_CHECKPOINT_2026-06-30.md']
```

This was the expected self-consistency failure after adding this checkpoint file. The canonical report was then regenerated.

Final canonical VS3 local/dev report and gate:

```text
./cornerstone scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json > /tmp/cs-vs3-generated-dirty-coverage-final-verify.json
./cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json > /tmp/cs-vs3-generated-dirty-coverage-final-gate.json
exit: 0

status success
scenario_count 57
summary {'blocking': 0, 'fail': 0, 'human_required': 7, 'not_run': 0, 'not_verified': 0, 'pass': 50, 'product_feature_claims': 'VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_GATES_REQUIRED', 'scenario_count': 57}
component_status passed
source_tree_current passed []
generated_dirty passed count 135 missing 0
errors []
```

## Proof Boundary

This checkpoint proves only that the local/dev VS3 scenario gate rejects omission of current generated evidence dirty paths.

It does not prove:

- production/on-prem readiness;
- real IdP readiness;
- live provider readiness;
- independent security acceptance;
- human UX acceptance;
- migration/restore readiness.

Those remain governed by `VS3-H01` through `VS3-H07`.

## Decision

The generated dirty path coverage guard slice is ready to keep as part of the VS3 local/dev gate.

Recommended next step: continue with the next narrow VS3 evidence-integrity guard or pause for human review before widening beyond local/dev evidence.
