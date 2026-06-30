# VS3 Local Checkpoint Component Proof Inner Status Guard Checkpoint

**Date:** 2026-06-29 KST
**Scope:** `cornerstone security vs3-local-checkpoint --json` component proof inner `scenario_status` and `checks` semantics.
**Status:** Local deterministic verifier-hardening slice verified.
**Verdict:** AI-verifiable slice PASS. `VS3-L` remains a local/dev assurance claim only. `VS3-P`, production/on-prem readiness, real IdP readiness, live-provider readiness, migration/restore readiness, security acceptance, and human acceptance remain unclaimed.

## Slice Contract

Goal:

- Prevent `reports/human-gates/vs3/vs3-local-checkpoint.json` from passing when a component proof file and the aggregate VS3 scenario report embed the same proof body, but that body contains non-PASS per-scenario status or failed internal checks.

In scope:

- Treat component proof `scenario_status` values as part of local checkpoint semantics for per-scenario component proof reports.
- Treat component proof `checks` values as part of local checkpoint semantics for per-scenario component proof reports.
- Preserve the existing `evidence_reconciliation` exception because it is a reconciliation report, not a per-scenario component proof with `scenario_status` and `checks`.
- Add explicit failed conditions and negative-evidence counters for scenario-status and check failures.
- Add a focused aligned-tamper regression test that keeps component file identity and aggregate embedded body aligned while changing `VS3-CTX-002` to `FAIL` and `vs3_ctx_002_forged_authority_denied` to `false`.
- Regenerate the VS3 local checkpoint report after the code change.

Out of scope:

- Changing any VS3 matrix row status.
- Replacing component proof implementations for RequestContext, RLS, OPA, egress, ConnectorHub, Tool SDK/registry, observability, or final regression.
- Accepting human evidence or promoting `VS3-H01` through `VS3-H07`.
- Claiming VS3-P, production/on-prem readiness, real IdP readiness, real network readiness, live-provider readiness, migration/restore readiness, security acceptance, or human acceptance.

Done criteria:

- Local checkpoint still passes when all component proof files match the aggregate report and all required inner statuses/checks are passing.
- Local checkpoint fails with exit code 4 when a component proof has a non-PASS scenario status and false check, even if the aggregate report embeds the same body and all dependent human-gate hashes are regenerated.
- Failure explains the exact component proof semantic layer that failed.
- Failure keeps all production, VS3-P, security-acceptance, and human-acceptance claims `NOT_CLAIMED`.

## Full Scenario Mapping

The frozen VS3 matrix remains unchanged.

| Type | Count | Slice classification |
|---|---:|---|
| MUST_PASS | 42 | Full set mapped. This slice directly hardens `VS3-GATE-004`; the component proof families remain existing-evidence guarded by the local checkpoint. |
| REGRESSION | 8 | `VS3-REG-004` and `VS3-REG-005` are directly hardened because coverage/check failures and claim-boundary drift can no longer hide behind matching component hashes. |
| HUMAN_REQUIRED | 7 | `VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED` and block VS3-P. |
| Total | 57 | No rows added, removed, or reclassified. |

| Scenario IDs | Type | Slice classification | Required proof surface |
|---|---|---|---|
| `VS3-GATE-001` | MUST_PASS | later_slice / existing evidence guarded | Reconciliation remains current input; this slice does not change the reconciliation result. |
| `VS3-GATE-002` | MUST_PASS | later_slice / existing evidence guarded | Matrix remains 57 rows with 42 MUST_PASS, 8 REGRESSION, 7 HUMAN_REQUIRED, and zero duplicate IDs. |
| `VS3-GATE-003` | MUST_PASS | later_slice / existing evidence guarded | Overclaim boundaries remain enforced by current reports and the local checkpoint. |
| `VS3-GATE-004` | MUST_PASS | in_this_slice | Native `cornerstone security vs3-local-checkpoint --json` now emits and enforces component inner scenario/check semantics. |
| `VS3-CTX-001`, `VS3-CTX-002`, `VS3-CTX-003`, `VS3-CTX-004`, `VS3-CTX-005` | MUST_PASS | existing evidence guarded | `request_context_proof` must match `reports/security/vs3-request-context-proof.json`, use schema `cs.vs3_request_context_proof.v0`, have `status=success`, and have all `scenario_status` values `PASS` with all checks `true`. |
| `VS3-RLS-001`, `VS3-RLS-002`, `VS3-RLS-003`, `VS3-RLS-004`, `VS3-RLS-005`, `VS3-RLS-006` | MUST_PASS | existing evidence guarded | `postgres_rls_proof` must match `reports/db/vs3-postgres-rls-proof.json`, use schema `cs.vs3_postgres_rls_proof.v0`, have `status=success`, and have all inner statuses/checks passing. |
| `VS3-OPA-001`, `VS3-OPA-002`, `VS3-OPA-003`, `VS3-OPA-004`, `VS3-OPA-005` | MUST_PASS | existing evidence guarded | `opa_policy_proof` must match `reports/policy/vs3-opa-policy-proof.json`, use schema `cs.vs3_opa_policy_proof.v0`, have `status=success`, and have all inner statuses/checks passing. |
| `VS3-EGR-001`, `VS3-EGR-002`, `VS3-EGR-003`, `VS3-EGR-004`, `VS3-EGR-005`, `VS3-EGR-006` | MUST_PASS | existing evidence guarded | `egress_sandbox_proof` must match `reports/security/vs3-egress-sandbox-proof.json`, use schema `cs.vs3_egress_sandbox_proof.v0`, have `status=success`, and have all inner statuses/checks passing. |
| `VS3-CON-001`, `VS3-CON-002`, `VS3-CON-003`, `VS3-CON-004`, `VS3-CON-005`, `VS3-CON-006` | MUST_PASS | existing evidence guarded | `connectorhub_source_proof` must match `reports/security/vs3-connectorhub-source-proof.json`, use schema `cs.vs3_connectorhub_source_proof.v0`, have `status=success`, and have all inner statuses/checks passing. Live-provider and physical-device evidence remains human-gated where applicable. |
| `VS3-TOOL-001`, `VS3-TOOL-002`, `VS3-TOOL-003`, `VS3-TOOL-004`, `VS3-TOOL-005`, `VS3-TOOL-006`, `VS3-TOOL-007` | MUST_PASS | existing evidence guarded | `tool_registry_proof` must match `reports/security/vs3-tool-registry-proof.json`, use schema `cs.vs3_tool_registry_proof.v0`, have `status=success`, and have all inner statuses/checks passing. |
| `VS3-OBS-001`, `VS3-OBS-002`, `VS3-OBS-003` | MUST_PASS | existing evidence guarded | `observability_proof` must match `reports/observability/vs3-observability-proof.json`, use schema `cs.vs3_observability_proof.v0`, have `status=success`, and have all inner statuses/checks passing. |
| `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-003`, `VS3-REG-006`, `VS3-REG-007`, `VS3-REG-008` | REGRESSION | existing evidence guarded | `final_regression_proof` must match `reports/security/vs3-final-regression-proof.json`, use schema `cs.vs3_final_regression_proof.v0`, have `status=success`, and have all inner statuses/checks passing. |
| `VS3-REG-004` | REGRESSION | in_this_slice | Component proof coverage cannot silently pass when per-scenario status/check evidence is failing. |
| `VS3-REG-005` | REGRESSION | in_this_slice | Failing component proof internals cannot produce a local/dev, VS3-P, production, security-accepted, or human-accepted overclaim. |
| `VS3-H01`, `VS3-H02`, `VS3-H03`, `VS3-H04`, `VS3-H05`, `VS3-H06`, `VS3-H07` | HUMAN_REQUIRED | human_required | Signed human/on-prem evidence remains required before any VS3-P, security acceptance, production/on-prem, live-provider, real-IdP, real-network, migration/restore, or human UX acceptance claim. |

## Before Evidence

Controlled aligned tamper before the fix:

1. Set `reports/security/vs3-request-context-proof.json` so `scenario_status["VS3-CTX-002"] = "FAIL"`.
2. Set `checks["vs3_ctx_002_forged_authority_denied"] = false`.
3. Embedded the same tampered body into `reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json`.
4. Regenerated dependent human-gate hashes with:
   - `cornerstone human-gate evidence-status ...`
   - `cornerstone human-gate review-kit ...`
   - `cornerstone human-gate vs3-p-gate ...`
5. Ran `cornerstone security vs3-local-checkpoint --json`.

Observed result before the fix:

```text
dep_rc [0, 0, 4]
checkpoint_rc 0
checkpoint_status success
checkpoint_final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
```

Root cause:

- The previous component proof semantics guard required matching aggregate/file identity, expected schema, and top-level `status=success`.
- It did not inspect inner per-scenario `scenario_status` or `checks`.
- A component proof could therefore be internally false while still producing a successful local checkpoint if the component file and aggregate embedded body drifted together.

## Implementation Summary

Updated `packages/cornerstone_cli/main.py`:

- Added per-component `expects_scenario_checks`.
- Added per-component `scenario_status_success` and `checks_success`.
- Added per-component fields:
  - `embedded_scenario_status_present`
  - `file_scenario_status_present`
  - `embedded_scenario_status_non_pass`
  - `file_scenario_status_non_pass`
  - `embedded_checks_present`
  - `file_checks_present`
  - `embedded_check_failures`
  - `file_check_failures`
- Added semantic error codes:
  - `CS_VS3_COMPONENT_PROOF_SCENARIO_STATUS_MISSING`
  - `CS_VS3_COMPONENT_PROOF_SCENARIO_STATUS_NOT_PASS`
  - `CS_VS3_COMPONENT_PROOF_CHECKS_MISSING`
  - `CS_VS3_COMPONENT_PROOF_CHECKS_NOT_TRUE`
- Added checkpoint conditions:
  - `component_proof_<key>_scenario_status_pass`
  - `component_proof_<key>_checks_pass`
- Added summary and negative-evidence counters:
  - `component_proof_report_scenario_failures`
  - `component_proof_report_check_failures`

Updated `tests/scenario/test_scaffold_cli.py`:

- Positive local checkpoint test now asserts expected scenario/check semantics for all per-scenario component proofs.
- Existing stale-file and failed-status tests now assert scenario/check counters remain zero when the failure is not an inner scenario/check failure.
- Added `test_vs3_local_checkpoint_rejects_component_proof_inner_status_failure_even_when_identity_matches`.

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
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_inner_status_failure_even_when_identity_matches
```

Result:

```text
Ran 4 tests in 120.403s
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
summary_subset {'scenario_count': 57, 'pass': 50, 'human_required': 7, 'component_proof_report_count': 9, 'component_proof_report_mismatches': 0, 'component_proof_report_semantic_failures': 0, 'component_proof_report_scenario_failures': 0, 'component_proof_report_check_failures': 0, 'vs3_l_claim': 'LOCAL_DEV_ASSURANCE_VERIFIED', 'vs3_p_claim': 'NOT_CLAIMED'}
negative_subset {'component_proof_report_mismatches': 0, 'component_proof_report_semantic_failures': 0, 'component_proof_report_scenario_failures': 0, 'component_proof_report_check_failures': 0, 'vs3_p_claimed_by_checkpoint': 0, 'production_readiness_claimed_by_checkpoint': 0, 'security_acceptance_claimed_by_checkpoint': 0, 'human_acceptance_claimed_by_checkpoint': 0}
```

Controlled aligned tamper after the fix:

```text
dep_rc [0, 0, 4]
checkpoint_rc 4
checkpoint_status failed
checkpoint_final_verdict VS3_L_LOCAL_DEV_ASSURANCE_NOT_VERIFIED
failed_conditions ['component_proof_request_context_proof_semantics_success', 'component_proof_request_context_proof_scenario_status_pass', 'component_proof_request_context_proof_checks_pass']
request_identity_subset {"checks_success": false, "embedded_status": "success", "file_check_failures": ["vs3_ctx_002_forged_authority_denied"], "file_scenario_status_non_pass": {"VS3-CTX-002": "FAIL"}, "file_status": "success", "matches_embedded_current_file": true, "scenario_status_success": false, "semantic_error_codes": ["CS_VS3_COMPONENT_PROOF_SCENARIO_STATUS_NOT_PASS", "CS_VS3_COMPONENT_PROOF_CHECKS_NOT_TRUE"], "semantics_success": false}
negative_subset {"component_proof_report_check_failures": 1, "component_proof_report_mismatches": 0, "component_proof_report_scenario_failures": 1, "component_proof_report_semantic_failures": 1, "human_acceptance_claimed_by_checkpoint": 0, "production_readiness_claimed_by_checkpoint": 0, "security_acceptance_claimed_by_checkpoint": 0, "vs3_p_claimed_by_checkpoint": 0}
```

## Pass / Fail Criteria

PASS:

- `cornerstone security vs3-local-checkpoint --json` exits 0 only when all nine component proofs match their embedded aggregate bodies, use expected schema/status semantics, and every per-scenario component proof has all inner statuses/checks passing.
- `component_proof_report_scenario_failures == 0`.
- `component_proof_report_check_failures == 0`.
- Controlled aligned tamper exits 4 with `component_proof_request_context_proof_scenario_status_pass` and `component_proof_request_context_proof_checks_pass` failed.
- All VS3-P, production/on-prem, security acceptance, and human acceptance claims remain `NOT_CLAIMED`.

FAIL:

- A component proof with a non-PASS `scenario_status` can still produce a successful local checkpoint.
- A component proof with any false check can still produce a successful local checkpoint.
- The checkpoint only detects hash mismatch but not matching failed internals.
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
