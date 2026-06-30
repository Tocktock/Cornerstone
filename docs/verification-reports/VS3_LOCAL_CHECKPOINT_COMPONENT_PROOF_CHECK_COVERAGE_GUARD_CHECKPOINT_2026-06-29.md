# VS3 Local Checkpoint Component Proof Check Coverage Guard Checkpoint

**Date:** 2026-06-29 KST
**Scope:** `cornerstone security vs3-local-checkpoint --json` component proof check-name coverage.
**Status:** Local deterministic verifier-hardening slice verified.
**Verdict:** AI-verifiable slice PASS. `VS3-L` remains a local/dev assurance claim only. `VS3-P`, production/on-prem readiness, real IdP readiness, live-provider readiness, migration/restore readiness, security acceptance, and human acceptance remain unclaimed.

## Slice Contract

Goal:

- Prevent `reports/human-gates/vs3/vs3-local-checkpoint.json` from passing when a per-scenario component proof silently drops an expected check while all remaining reported checks are `true`.

In scope:

- Define expected check names for every per-scenario VS3 component proof family.
- Require exact component proof check coverage in both the aggregate embedded proof body and the current component proof file.
- Preserve the `evidence_reconciliation` exception because it is not a per-scenario proof report with check coverage.
- Add explicit failed conditions and negative-evidence counters for component proof check coverage failures.
- Add a focused aligned-tamper regression test that removes `vs3_ctx_002_forged_authority_denied` from `request_context_proof.checks`, embeds the same body into the aggregate report, regenerates dependent human-gate hashes, and expects the local checkpoint to fail.
- Regenerate the VS3 local checkpoint report after the code change.

Out of scope:

- Changing any VS3 matrix row status.
- Replacing component proof behavior for RequestContext, RLS, OPA, egress, ConnectorHub, Tool SDK/registry, observability, or final regression.
- Accepting human evidence or promoting `VS3-H01` through `VS3-H07`.
- Claiming VS3-P, production/on-prem readiness, real IdP readiness, real network readiness, live-provider readiness, migration/restore readiness, security acceptance, or human acceptance.

Done criteria:

- Local checkpoint still passes when all component proof files match the aggregate report and every per-scenario component proof includes exactly its expected check names.
- Local checkpoint fails with exit code 4 when a component proof omits an expected check, even if the aggregate report embeds the same omitted body and all dependent human-gate hashes are regenerated.
- Failure explains the missing check name and keeps false check values separate from check coverage failures.
- Failure keeps all production, VS3-P, security-acceptance, and human-acceptance claims `NOT_CLAIMED`.

## Full Scenario Mapping

The frozen VS3 matrix remains unchanged.

| Type | Count | Slice classification |
|---|---:|---|
| MUST_PASS | 42 | Full set mapped. This slice directly hardens `VS3-GATE-004`; the component proof families remain existing-evidence guarded by the local checkpoint. |
| REGRESSION | 8 | `VS3-REG-004` and `VS3-REG-005` are directly hardened because component proof checks can no longer silently drop while preserving a local/dev claim. |
| HUMAN_REQUIRED | 7 | `VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED` and block VS3-P. |
| Total | 57 | No rows added, removed, or reclassified. |

| Scenario IDs | Type | Slice classification | Required proof surface |
|---|---|---|---|
| `VS3-GATE-001` | MUST_PASS | later_slice / existing evidence guarded | Reconciliation remains current input; this slice does not change the reconciliation result. |
| `VS3-GATE-002` | MUST_PASS | later_slice / existing evidence guarded | Matrix remains 57 rows with 42 MUST_PASS, 8 REGRESSION, 7 HUMAN_REQUIRED, and zero duplicate IDs. |
| `VS3-GATE-003` | MUST_PASS | later_slice / existing evidence guarded | Overclaim boundaries remain enforced by current reports and the local checkpoint. |
| `VS3-GATE-004` | MUST_PASS | in_this_slice | Native `cornerstone security vs3-local-checkpoint --json` now emits and enforces component check-name coverage. |
| `VS3-CTX-001`, `VS3-CTX-002`, `VS3-CTX-003`, `VS3-CTX-004`, `VS3-CTX-005` | MUST_PASS | existing evidence guarded | `request_context_proof` must include exactly the expected RequestContext check names in aggregate and file proof bodies. |
| `VS3-RLS-001`, `VS3-RLS-002`, `VS3-RLS-003`, `VS3-RLS-004`, `VS3-RLS-005`, `VS3-RLS-006` | MUST_PASS | existing evidence guarded | `postgres_rls_proof` must include exactly the expected RLS check names in aggregate and file proof bodies. |
| `VS3-OPA-001`, `VS3-OPA-002`, `VS3-OPA-003`, `VS3-OPA-004`, `VS3-OPA-005` | MUST_PASS | existing evidence guarded | `opa_policy_proof` must include exactly the expected OPA check names in aggregate and file proof bodies. |
| `VS3-EGR-001`, `VS3-EGR-002`, `VS3-EGR-003`, `VS3-EGR-004`, `VS3-EGR-005`, `VS3-EGR-006` | MUST_PASS | existing evidence guarded | `egress_sandbox_proof` must include exactly the expected egress/sandbox check names in aggregate and file proof bodies. |
| `VS3-CON-001`, `VS3-CON-002`, `VS3-CON-003`, `VS3-CON-004`, `VS3-CON-005`, `VS3-CON-006` | MUST_PASS | existing evidence guarded | `connectorhub_source_proof` must include exactly the expected ConnectorHub check names in aggregate and file proof bodies. Live-provider and physical-device evidence remains human-gated where applicable. |
| `VS3-TOOL-001`, `VS3-TOOL-002`, `VS3-TOOL-003`, `VS3-TOOL-004`, `VS3-TOOL-005`, `VS3-TOOL-006`, `VS3-TOOL-007` | MUST_PASS | existing evidence guarded | `tool_registry_proof` must include exactly the expected tool/registry check names in aggregate and file proof bodies. |
| `VS3-OBS-001`, `VS3-OBS-002`, `VS3-OBS-003` | MUST_PASS | existing evidence guarded | `observability_proof` must include exactly the expected observability check names in aggregate and file proof bodies. |
| `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-003`, `VS3-REG-004`, `VS3-REG-005`, `VS3-REG-006`, `VS3-REG-007`, `VS3-REG-008` | REGRESSION | existing evidence guarded | `final_regression_proof` must include exactly the expected final-regression check names in aggregate and file proof bodies. |
| `VS3-REG-004` | REGRESSION | in_this_slice | Component proof coverage cannot silently pass when an expected check is missing. |
| `VS3-REG-005` | REGRESSION | in_this_slice | Missing component proof check coverage cannot produce a local/dev, VS3-P, production, security-accepted, or human-accepted overclaim. |
| `VS3-H01`, `VS3-H02`, `VS3-H03`, `VS3-H04`, `VS3-H05`, `VS3-H06`, `VS3-H07` | HUMAN_REQUIRED | human_required | Signed human/on-prem evidence remains required before any VS3-P, security acceptance, production/on-prem, live-provider, real-IdP, real-network, migration/restore, or human UX acceptance claim. |

## Before Evidence

Controlled aligned check-drop tamper before the fix:

1. Removed `vs3_ctx_002_forged_authority_denied` from `reports/security/vs3-request-context-proof.json` under `checks`.
2. Embedded the same body into `reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json`.
3. Regenerated dependent human-gate hashes with:
   - `cornerstone human-gate evidence-status ...`
   - `cornerstone human-gate review-kit ...`
   - `cornerstone human-gate vs3-p-gate ...`
4. Ran `cornerstone security vs3-local-checkpoint --json`.

Observed result before the fix:

```text
dep_rc [0, 0, 4]
checkpoint_rc 0
checkpoint_status success
checkpoint_final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
failed_conditions []
request_identity_subset {"checks_success": true, "expected_check_names": null, "file_check_names": null, "file_missing_check_names": null, "matches_embedded_current_file": true, "scenario_status_coverage_success": true, "scenario_status_success": true, "semantic_error_codes": []}
negative_subset {"component_proof_report_check_coverage_failures": null, "component_proof_report_check_failures": 0, "component_proof_report_mismatches": 0, "component_proof_report_scenario_coverage_failures": 0, "component_proof_report_scenario_failures": 0, "component_proof_report_semantic_failures": 0, "human_acceptance_claimed_by_checkpoint": 0, "production_readiness_claimed_by_checkpoint": 0, "security_acceptance_claimed_by_checkpoint": 0, "vs3_p_claimed_by_checkpoint": 0}
```

Root cause:

- The previous component proof guard required non-empty `checks` and all present values to be `true`.
- It did not require exact expected check-name coverage.
- A component proof could therefore drop a safety check and still pass if all remaining checks were `true`.

## Implementation Summary

Updated `packages/cornerstone_cli/main.py`:

- Added expected check names per component proof:
  - `request_context_proof`: five RequestContext check names.
  - `postgres_rls_proof`: six Postgres/RLS check names.
  - `opa_policy_proof`: five OPA check names.
  - `egress_sandbox_proof`: eight egress/sandbox check names.
  - `connectorhub_source_proof`: seven ConnectorHub check names.
  - `tool_registry_proof`: eight tool/registry check names.
  - `observability_proof`: three observability check names.
  - `final_regression_proof`: eight final-regression check names.
- Added per-component fields:
  - `expected_check_names`
  - `embedded_check_names`
  - `file_check_names`
  - `embedded_missing_check_names`
  - `file_missing_check_names`
  - `embedded_unexpected_check_names`
  - `file_unexpected_check_names`
  - `checks_coverage_success`
- Added semantic error code:
  - `CS_VS3_COMPONENT_PROOF_CHECKS_COVERAGE_MISMATCH`
- Added checkpoint condition:
  - `component_proof_<key>_checks_coverage`
- Added summary and negative-evidence counter:
  - `component_proof_report_check_coverage_failures`

Updated `tests/scenario/test_scaffold_cli.py`:

- Positive local checkpoint test now asserts exact expected check names for every component proof family.
- Existing stale/status/scenario-failure tests now assert the check-coverage counter remains zero when check coverage is not the failing layer.
- Added `test_vs3_local_checkpoint_rejects_component_proof_missing_check_even_when_identity_matches`.

## Verification Evidence

Compile:

```text
python3 -m compileall packages/cornerstone_cli
Listing 'packages/cornerstone_cli'...
Compiling 'packages/cornerstone_cli/main.py'...
```

Focused tests:

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_manifest_is_hash_backed_and_boundary_safe \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_stale_component_proof_file \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_status_failure_even_when_identity_matches \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_inner_status_failure_even_when_identity_matches \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_missing_scenario_row_even_when_identity_matches \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_missing_check_even_when_identity_matches
```

Result:

```text
Ran 6 tests in 171.122s
OK
```

Regenerated checkpoint:

```text
PATH="$PWD:$PATH" cornerstone security vs3-local-checkpoint --json --output reports/human-gates/vs3/vs3-local-checkpoint.json
```

Result:

```text
status success
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
summary_subset {'scenario_count': 57, 'pass': 50, 'human_required': 7, 'component_proof_report_count': 9, 'component_proof_report_mismatches': 0, 'component_proof_report_semantic_failures': 0, 'component_proof_report_scenario_failures': 0, 'component_proof_report_scenario_coverage_failures': 0, 'component_proof_report_check_failures': 0, 'component_proof_report_check_coverage_failures': 0, 'vs3_l_claim': 'LOCAL_DEV_ASSURANCE_VERIFIED', 'vs3_p_claim': 'NOT_CLAIMED'}
negative_subset {'component_proof_report_mismatches': 0, 'component_proof_report_semantic_failures': 0, 'component_proof_report_scenario_failures': 0, 'component_proof_report_scenario_coverage_failures': 0, 'component_proof_report_check_failures': 0, 'component_proof_report_check_coverage_failures': 0, 'vs3_p_claimed_by_checkpoint': 0, 'production_readiness_claimed_by_checkpoint': 0, 'security_acceptance_claimed_by_checkpoint': 0, 'human_acceptance_claimed_by_checkpoint': 0}
request_context_checks {'expected_check_names': ['vs3_ctx_001_surface_context_consistent', 'vs3_ctx_002_forged_authority_denied', 'vs3_ctx_003_revocation_fail_closed', 'vs3_ctx_004_context_faults_fail_closed', 'vs3_ctx_005_mission_workspace_policy_enforced'], 'file_check_names': ['vs3_ctx_001_surface_context_consistent', 'vs3_ctx_002_forged_authority_denied', 'vs3_ctx_003_revocation_fail_closed', 'vs3_ctx_004_context_faults_fail_closed', 'vs3_ctx_005_mission_workspace_policy_enforced'], 'checks_coverage_success': True, 'checks_success': True, 'semantics_success': True, 'semantic_error_codes': []}
```

Controlled aligned check-drop tamper after the fix:

```text
dep_rc [0, 0, 4]
checkpoint_rc 4
checkpoint_status failed
checkpoint_final_verdict VS3_L_LOCAL_DEV_ASSURANCE_NOT_VERIFIED
failed_conditions ['component_proof_request_context_proof_semantics_success', 'component_proof_request_context_proof_checks_pass', 'component_proof_request_context_proof_checks_coverage']
request_identity_subset {"checks_coverage_success": false, "checks_success": false, "expected_check_names": ["vs3_ctx_001_surface_context_consistent", "vs3_ctx_002_forged_authority_denied", "vs3_ctx_003_revocation_fail_closed", "vs3_ctx_004_context_faults_fail_closed", "vs3_ctx_005_mission_workspace_policy_enforced"], "file_check_names": ["vs3_ctx_001_surface_context_consistent", "vs3_ctx_003_revocation_fail_closed", "vs3_ctx_004_context_faults_fail_closed", "vs3_ctx_005_mission_workspace_policy_enforced"], "file_missing_check_names": ["vs3_ctx_002_forged_authority_denied"], "matches_embedded_current_file": true, "scenario_status_coverage_success": true, "scenario_status_success": true, "semantic_error_codes": ["CS_VS3_COMPONENT_PROOF_CHECKS_COVERAGE_MISMATCH"]}
negative_subset {"component_proof_report_check_coverage_failures": 1, "component_proof_report_check_failures": 1, "component_proof_report_mismatches": 0, "component_proof_report_scenario_coverage_failures": 0, "component_proof_report_scenario_failures": 0, "component_proof_report_semantic_failures": 1, "human_acceptance_claimed_by_checkpoint": 0, "production_readiness_claimed_by_checkpoint": 0, "security_acceptance_claimed_by_checkpoint": 0, "vs3_p_claimed_by_checkpoint": 0}
```

## Pass / Fail Criteria

PASS:

- `cornerstone security vs3-local-checkpoint --json` exits 0 only when every per-scenario component proof contains exactly its expected check names in both aggregate and file proof bodies.
- `component_proof_report_check_coverage_failures == 0`.
- Controlled check-drop tamper exits 4 with `component_proof_request_context_proof_checks_coverage` failed.
- All VS3-P, production/on-prem, security acceptance, and human acceptance claims remain `NOT_CLAIMED`.

FAIL:

- A component proof missing an expected check can still produce a successful local checkpoint.
- The checkpoint only detects false check values but not omitted checks.
- The failure path claims VS3-P, production/on-prem readiness, security acceptance, or human acceptance.

## Remaining Human Gates

Still `HUMAN_REQUIRED`:

- `VS3-H01` owner architecture/dependency/migration/security approval.
- `VS3-H02` independent security review and retest.
- `VS3-H03` real IdP mapping and revocation evidence.
- `VS3-H04` real topology and network-policy evidence.
- `VS3-H05` approved live-provider rehearsal.
- `VS3-H06` human operator UX/trust acceptance or rejection.
- `VS3-H07` signed migration/backup/restore drill.

This checkpoint does not satisfy or replace any of those human gates.
