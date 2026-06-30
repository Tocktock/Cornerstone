# VS3 Local Checkpoint Component Proof Negative Evidence Guard Checkpoint

**Date:** 2026-06-29 KST
**Scope:** `cornerstone security vs3-local-checkpoint --json` component proof negative-evidence semantics.
**Status:** Local deterministic verifier-hardening slice verified.
**Verdict:** AI-verifiable slice PASS. `VS3-L` remains a local/dev assurance claim only. `VS3-P`, production/on-prem readiness, real IdP readiness, live-provider readiness, migration/restore readiness, security acceptance, and human acceptance remain unclaimed.

## Slice Contract

Goal:

- Prevent `reports/human-gates/vs3/vs3-local-checkpoint.json` from passing when a component proof has nonzero `negative_evidence`, even if the component proof file and the aggregate embedded proof body match exactly.

In scope:

- Require every VS3 component proof identity to expose `negative_evidence`.
- Require embedded and current file `negative_evidence` values to be all-zero.
- Add explicit semantic error codes, failed checkpoint conditions, summary counters, and negative-evidence counters for nonzero component proof negative evidence.
- Add a focused aligned-tamper regression test that sets `request_context_proof.negative_evidence.forged_authority_paths_allowed = 1`, embeds the same body into the aggregate scenario report, regenerates dependent human-gate hashes, and expects the local checkpoint to fail.
- Regenerate the VS3 local checkpoint report after the code change.

Out of scope:

- Changing any VS3 matrix row status.
- Replacing the underlying RequestContext, RLS, OPA, egress, ConnectorHub, Tool SDK/registry, observability, or final regression proof implementations.
- Accepting human evidence or promoting `VS3-H01` through `VS3-H07`.
- Claiming VS3-P, production/on-prem readiness, real IdP readiness, real network readiness, live-provider readiness, migration/restore readiness, security acceptance, or human acceptance.

Done criteria:

- Local checkpoint still passes when all component proof files match the aggregate report and every component proof `negative_evidence` value is zero.
- Local checkpoint fails with exit code 4 when a component proof has nonzero `negative_evidence`, even if the aggregate report embeds the same body and all dependent human-gate hashes are regenerated.
- Failure is attributed to component proof negative evidence and semantics, not to stale hashes, missing scenario status, or missing checks.
- Failure keeps all VS3-P, production, security-acceptance, and human-acceptance claims `NOT_CLAIMED`.

## Full Scenario Mapping

The frozen VS3 matrix remains unchanged.

| Type | Count | Slice classification |
|---|---:|---|
| MUST_PASS | 42 | Full set mapped. This slice directly hardens `VS3-GATE-004`; component proof families remain existing-evidence guarded by the local checkpoint. |
| REGRESSION | 8 | `VS3-REG-004` and `VS3-REG-005` are directly hardened because nonzero component proof negative evidence can no longer pass while preserving a local/dev claim. |
| HUMAN_REQUIRED | 7 | `VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED` and block VS3-P. |
| Total | 57 | No rows added, removed, or reclassified. |

| Scenario IDs | Type | Slice classification | Required proof surface |
|---|---|---|---|
| `VS3-GATE-001` | MUST_PASS | later_slice / existing evidence guarded | Reconciliation remains current input; this slice does not change the reconciliation result. |
| `VS3-GATE-002` | MUST_PASS | later_slice / existing evidence guarded | Matrix remains 57 rows with 42 MUST_PASS, 8 REGRESSION, 7 HUMAN_REQUIRED, and zero duplicate IDs. |
| `VS3-GATE-003` | MUST_PASS | later_slice / existing evidence guarded | Overclaim boundaries remain enforced by current reports and the local checkpoint. |
| `VS3-GATE-004` | MUST_PASS | in_this_slice | Native `cornerstone security vs3-local-checkpoint --json` now emits and enforces component proof negative-evidence success. |
| `VS3-CTX-001`, `VS3-CTX-002`, `VS3-CTX-003`, `VS3-CTX-004`, `VS3-CTX-005` | MUST_PASS | existing evidence guarded | `request_context_proof` must have all-zero `negative_evidence` in aggregate and file proof bodies. |
| `VS3-RLS-001`, `VS3-RLS-002`, `VS3-RLS-003`, `VS3-RLS-004`, `VS3-RLS-005`, `VS3-RLS-006` | MUST_PASS | existing evidence guarded | `postgres_rls_proof` must have all-zero `negative_evidence` in aggregate and file proof bodies. |
| `VS3-OPA-001`, `VS3-OPA-002`, `VS3-OPA-003`, `VS3-OPA-004`, `VS3-OPA-005` | MUST_PASS | existing evidence guarded | `opa_policy_proof` must have all-zero `negative_evidence` in aggregate and file proof bodies. |
| `VS3-EGR-001`, `VS3-EGR-002`, `VS3-EGR-003`, `VS3-EGR-004`, `VS3-EGR-005`, `VS3-EGR-006` | MUST_PASS | existing evidence guarded | `egress_sandbox_proof` must have all-zero `negative_evidence` in aggregate and file proof bodies. |
| `VS3-CON-001`, `VS3-CON-002`, `VS3-CON-003`, `VS3-CON-004`, `VS3-CON-005`, `VS3-CON-006` | MUST_PASS | existing evidence guarded | `connectorhub_source_proof` must have all-zero `negative_evidence` in aggregate and file proof bodies. Live-provider and physical-device evidence remains human-gated where applicable. |
| `VS3-TOOL-001`, `VS3-TOOL-002`, `VS3-TOOL-003`, `VS3-TOOL-004`, `VS3-TOOL-005`, `VS3-TOOL-006`, `VS3-TOOL-007` | MUST_PASS | existing evidence guarded | `tool_registry_proof` must have all-zero `negative_evidence` in aggregate and file proof bodies. |
| `VS3-OBS-001`, `VS3-OBS-002`, `VS3-OBS-003` | MUST_PASS | existing evidence guarded | `observability_proof` must have all-zero `negative_evidence` in aggregate and file proof bodies. |
| `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-003`, `VS3-REG-004`, `VS3-REG-005`, `VS3-REG-006`, `VS3-REG-007`, `VS3-REG-008` | REGRESSION | existing evidence guarded | `final_regression_proof` must have all-zero `negative_evidence` in aggregate and file proof bodies. |
| `VS3-REG-004` | REGRESSION | in_this_slice | Component proof coverage cannot silently pass when a proof report contains nonzero negative evidence. |
| `VS3-REG-005` | REGRESSION | in_this_slice | Nonzero component proof negative evidence cannot produce a local/dev, VS3-P, production, security-accepted, or human-accepted overclaim. |
| `VS3-H01`, `VS3-H02`, `VS3-H03`, `VS3-H04`, `VS3-H05`, `VS3-H06`, `VS3-H07` | HUMAN_REQUIRED | human_required | Signed human/on-prem evidence remains required before any VS3-P, security acceptance, production/on-prem, live-provider, real-IdP, real-network, migration/restore, or human UX acceptance claim. |

## Before Evidence

Controlled aligned negative-evidence tamper before the fix:

1. Set `reports/security/vs3-request-context-proof.json` field `negative_evidence.forged_authority_paths_allowed = 1`.
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
request_identity_subset {"checks_success": true, "file_negative_evidence_nonzero": null, "matches_embedded_current_file": true, "negative_evidence_success": null, "scenario_status_success": true, "semantic_error_codes": []}
negative_subset {"component_proof_report_mismatches": 0, "component_proof_report_negative_evidence_failures": null, "component_proof_report_semantic_failures": 0, "human_acceptance_claimed_by_checkpoint": 0, "production_readiness_claimed_by_checkpoint": 0, "security_acceptance_claimed_by_checkpoint": 0, "vs3_p_claimed_by_checkpoint": 0}
```

Root cause:

- The previous component proof guard required matching aggregate/file proof bodies, expected schema, `status=success`, passing scenario status, and passing checks.
- It did not require `negative_evidence` to exist or to contain only zero values.
- A component proof could therefore report nonzero safety failures and still pass if the aggregate embedded body matched the current file.

## Implementation Summary

Updated `packages/cornerstone_cli/main.py`:

- Added per-component fields:
  - `embedded_negative_evidence_present`
  - `file_negative_evidence_present`
  - `embedded_negative_evidence_nonzero`
  - `file_negative_evidence_nonzero`
  - `negative_evidence_success`
- Added semantic error codes:
  - `CS_VS3_COMPONENT_PROOF_NEGATIVE_EVIDENCE_MISSING`
  - `CS_VS3_COMPONENT_PROOF_NEGATIVE_EVIDENCE_NONZERO`
- Included `negative_evidence_success` in component proof `semantics_success`.
- Added checkpoint condition:
  - `component_proof_<key>_negative_evidence_zero`
- Added summary and negative-evidence counter:
  - `component_proof_report_negative_evidence_failures`

Updated `tests/scenario/test_scaffold_cli.py`:

- Positive local checkpoint test now asserts every component proof has present all-zero negative evidence.
- Existing stale/status/scenario/check failure tests now assert the negative-evidence counter remains zero when that layer is not failing.
- Added `test_vs3_local_checkpoint_rejects_component_proof_nonzero_negative_evidence_even_when_identity_matches`.

Regenerated:

- `reports/human-gates/vs3/vs3-local-checkpoint.json`

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
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_missing_check_even_when_identity_matches \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_nonzero_negative_evidence_even_when_identity_matches
```

Result:

```text
Ran 7 tests in 187.670s
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
claim_boundary {'checkpoint_is_local_dev_only': True, 'human_acceptance': 'NOT_CLAIMED', 'live_provider': 'NOT_CLAIMED', 'migration_restore': 'NOT_CLAIMED', 'production': 'NOT_CLAIMED', 'production_onprem': 'NOT_CLAIMED', 'real_idp': 'NOT_CLAIMED', 'real_network': 'NOT_CLAIMED', 'security_acceptance': 'NOT_CLAIMED', 'structural_validation_is_not_acceptance': True, 'vs3_l': 'LOCAL_DEV_ASSURANCE_VERIFIED', 'vs3_p': 'NOT_CLAIMED'}
summary_subset {'scenario_count': 57, 'pass': 50, 'human_required': 7, 'blocking': 0, 'component_proof_report_mismatches': 0, 'component_proof_report_semantic_failures': 0, 'component_proof_report_scenario_failures': 0, 'component_proof_report_scenario_coverage_failures': 0, 'component_proof_report_check_failures': 0, 'component_proof_report_check_coverage_failures': 0, 'component_proof_report_negative_evidence_failures': 0, 'vs3_l_claim': 'LOCAL_DEV_ASSURANCE_VERIFIED', 'vs3_p_claim': 'NOT_CLAIMED'}
negative_subset {'component_proof_report_mismatches': 0, 'component_proof_report_semantic_failures': 0, 'component_proof_report_scenario_failures': 0, 'component_proof_report_scenario_coverage_failures': 0, 'component_proof_report_check_failures': 0, 'component_proof_report_check_coverage_failures': 0, 'component_proof_report_negative_evidence_failures': 0, 'component_proof_report_missing_or_invalid': 0, 'vs3_p_claimed_by_checkpoint': 0, 'production_readiness_claimed_by_checkpoint': 0, 'security_acceptance_claimed_by_checkpoint': 0, 'human_acceptance_claimed_by_checkpoint': 0}
component_proof_identity:
evidence_reconciliation True {} {} []
request_context_proof True {} {} []
postgres_rls_proof True {} {} []
opa_policy_proof True {} {} []
egress_sandbox_proof True {} {} []
connectorhub_source_proof True {} {} []
tool_registry_proof True {} {} []
observability_proof True {} {} []
final_regression_proof True {} {} []
```

Controlled aligned negative-evidence tamper after the fix:

```text
dep_rc [0, 0, 4]
checkpoint_rc 4
checkpoint_status failed
checkpoint_final_verdict VS3_L_LOCAL_DEV_ASSURANCE_NOT_VERIFIED
failed_conditions ['component_proof_request_context_proof_semantics_success', 'component_proof_request_context_proof_negative_evidence_zero']
request_identity_subset {"checks_success": true, "file_negative_evidence_nonzero": {"forged_authority_paths_allowed": 1}, "matches_embedded_current_file": true, "negative_evidence_success": false, "scenario_status_success": true, "semantic_error_codes": ["CS_VS3_COMPONENT_PROOF_NEGATIVE_EVIDENCE_NONZERO"]}
negative_subset {"component_proof_report_check_failures": 0, "component_proof_report_mismatches": 0, "component_proof_report_negative_evidence_failures": 1, "component_proof_report_scenario_failures": 0, "component_proof_report_semantic_failures": 1, "human_acceptance_claimed_by_checkpoint": 0, "production_readiness_claimed_by_checkpoint": 0, "security_acceptance_claimed_by_checkpoint": 0, "vs3_p_claimed_by_checkpoint": 0}
```

## Pass / Fail Criteria

PASS:

- `cornerstone security vs3-local-checkpoint --json` exits 0 only when every component proof has present all-zero `negative_evidence` in both aggregate and file proof bodies.
- `component_proof_report_negative_evidence_failures == 0`.
- Controlled nonzero negative-evidence tamper exits 4 with `component_proof_request_context_proof_negative_evidence_zero` failed.
- The tamper failure preserves matching aggregate/file identity and passing scenario/check status, proving the guard is not a stale-hash or omitted-check detector.
- All VS3-P, production/on-prem, security acceptance, and human acceptance claims remain `NOT_CLAIMED`.

FAIL:

- A component proof with nonzero `negative_evidence` can still produce a successful local checkpoint.
- The checkpoint only detects stale hashes, failed proof status, or omitted checks but not nonzero negative evidence.
- The failure path claims VS3-P, production/on-prem readiness, security acceptance, or human acceptance.
- The fix requires human/external evidence or live credentials for local/CI verification.

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
