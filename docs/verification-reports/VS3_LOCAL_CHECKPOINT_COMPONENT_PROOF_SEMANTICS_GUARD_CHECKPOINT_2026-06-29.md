# VS3 Local Checkpoint Component Proof Semantics Guard Checkpoint

**Date:** 2026-06-29 KST
**Scope:** `cornerstone security vs3-local-checkpoint --json` component proof semantic guard.
**Status:** Local deterministic verifier-hardening slice verified.
**Verdict:** AI-verifiable slice PASS. `VS3-L` remains a local/dev assurance claim only. `VS3-P`, production/on-prem readiness, real IdP readiness, live-provider readiness, migration/restore readiness, security acceptance, and human acceptance remain unclaimed.

## Slice Contract

Goal:

- Prevent `reports/human-gates/vs3/vs3-local-checkpoint.json` from passing when the aggregate VS3 scenario report embeds a component proof body whose current file identity matches but whose proof semantics are failed or schema-incompatible.

In scope:

- Add expected `schema_version` and `status=success` semantic checks for every component proof bound into `reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json`.
- Expose per-component semantic fields and error codes in `component_proof_identity`.
- Add checkpoint conditions named `component_proof_<key>_semantics_success`.
- Add `component_proof_report_semantic_failures` to summary and negative evidence.
- Add a focused aligned-tamper regression test that keeps component file and aggregate embedded body identical while changing both to `status=failed`.
- Regenerate the VS3 local checkpoint report after the code change.

Out of scope:

- Changing any VS3 scenario row status.
- Accepting human evidence or promoting `VS3-H01` through `VS3-H07`.
- Claiming VS3-P, production/on-prem readiness, real IdP readiness, real network readiness, live-provider readiness, migration/restore readiness, security acceptance, or human acceptance.
- Reworking the underlying RequestContext, RLS, OPA, egress, ConnectorHub, Tool SDK, registry, or operator UX proof implementations.

Done criteria:

- Local checkpoint still passes when all nine component proof files match the aggregate report and have expected schema/status semantics.
- Local checkpoint fails with exit code 4 when a component proof has `status=failed`, even if the aggregate report embeds the same failed body and all human-gate report hashes are regenerated.
- Failure keeps all production, VS3-P, security-acceptance, and human-acceptance claims `NOT_CLAIMED`.

## Full Scenario Mapping

The frozen VS3 matrix remains unchanged.

| Type | Count | Slice classification |
|---|---:|---|
| MUST_PASS | 42 | All rows remain mapped. This slice directly hardens the native checkpoint proof surface for `VS3-GATE-004`; component rows are existing-evidence guarded through their owning proof reports. |
| REGRESSION | 8 | `VS3-REG-004` and `VS3-REG-005` are directly hardened because coverage/evidence and claim-boundary drift can no longer hide behind matching component hashes. |
| HUMAN_REQUIRED | 7 | `VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED` and block VS3-P. |
| Total | 57 | No rows added, removed, or reclassified. |

| Scenario IDs | Type | Slice classification | Required proof surface |
|---|---|---|---|
| `VS3-GATE-001` | MUST_PASS | existing_evidence_semantics_guarded | `evidence_reconciliation` embedded body must match `reports/security/vs3-evidence-reconciliation.json`, use schema `cs.vs3_evidence_reconciliation.v0`, and have `status=success`. |
| `VS3-GATE-002` | MUST_PASS | existing_evidence_semantics_guarded | Contract, matrix, goal prompt, and report artifacts remain hash-backed by the local checkpoint manifest. |
| `VS3-GATE-003` | MUST_PASS | existing_evidence_semantics_guarded | Overclaim guard remains active; failed component proof semantics now also block the local checkpoint claim. |
| `VS3-GATE-004` | MUST_PASS | in_this_slice | Native `cornerstone security vs3-local-checkpoint --json` now emits component proof identity plus semantic status, failed condition, and negative-evidence counters. |
| `VS3-CTX-001`, `VS3-CTX-002`, `VS3-CTX-003`, `VS3-CTX-004`, `VS3-CTX-005` | MUST_PASS | existing_evidence_semantics_guarded | `request_context_proof` must match `reports/security/vs3-request-context-proof.json`, use schema `cs.vs3_request_context_proof.v0`, and have `status=success`. |
| `VS3-RLS-001`, `VS3-RLS-002`, `VS3-RLS-003`, `VS3-RLS-004`, `VS3-RLS-005`, `VS3-RLS-006` | MUST_PASS | existing_evidence_semantics_guarded | `postgres_rls_proof` must match `reports/db/vs3-postgres-rls-proof.json`, use schema `cs.vs3_postgres_rls_proof.v0`, and have `status=success`. |
| `VS3-OPA-001`, `VS3-OPA-002`, `VS3-OPA-003`, `VS3-OPA-004`, `VS3-OPA-005` | MUST_PASS | existing_evidence_semantics_guarded | `opa_policy_proof` must match `reports/policy/vs3-opa-policy-proof.json`, use schema `cs.vs3_opa_policy_proof.v0`, and have `status=success`. |
| `VS3-EGR-001`, `VS3-EGR-002`, `VS3-EGR-003`, `VS3-EGR-004`, `VS3-EGR-005`, `VS3-EGR-006` | MUST_PASS | existing_evidence_semantics_guarded | `egress_sandbox_proof` must match `reports/security/vs3-egress-sandbox-proof.json`, use schema `cs.vs3_egress_sandbox_proof.v0`, and have `status=success`. |
| `VS3-CON-001`, `VS3-CON-002`, `VS3-CON-003`, `VS3-CON-004`, `VS3-CON-005`, `VS3-CON-006` | MUST_PASS | existing_evidence_semantics_guarded | `connectorhub_source_proof` must match `reports/security/vs3-connectorhub-source-proof.json`, use schema `cs.vs3_connectorhub_source_proof.v0`, and have `status=success`; live-provider/capture claims remain human-gated where applicable. |
| `VS3-TOOL-001`, `VS3-TOOL-002`, `VS3-TOOL-003`, `VS3-TOOL-004`, `VS3-TOOL-005`, `VS3-TOOL-006`, `VS3-TOOL-007` | MUST_PASS | existing_evidence_semantics_guarded | `tool_registry_proof` must match `reports/security/vs3-tool-registry-proof.json`, use schema `cs.vs3_tool_registry_proof.v0`, and have `status=success`. |
| `VS3-OBS-001`, `VS3-OBS-002`, `VS3-OBS-003` | MUST_PASS | existing_evidence_semantics_guarded | `observability_proof` must match `reports/observability/vs3-observability-proof.json`, use schema `cs.vs3_observability_proof.v0`, and have `status=success`. |
| `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-003`, `VS3-REG-006`, `VS3-REG-007`, `VS3-REG-008` | REGRESSION | existing_evidence_semantics_guarded | `final_regression_proof` must match `reports/security/vs3-final-regression-proof.json`, use schema `cs.vs3_final_regression_proof.v0`, and have `status=success`. |
| `VS3-REG-004` | REGRESSION | in_this_slice | Component proof coverage cannot silently pass when a proof report body is semantically failed. |
| `VS3-REG-005` | REGRESSION | in_this_slice | Local/dev proof cannot become production, VS3-P, security-accepted, or human-accepted when component proof semantics fail. |
| `VS3-H01` | HUMAN_REQUIRED | human_required | Owner architecture/dependency/migration/security approval remains required before VS3-P. |
| `VS3-H02` | HUMAN_REQUIRED | human_required | Independent security review and retest remains required before VS3-P. |
| `VS3-H03` | HUMAN_REQUIRED | human_required | Real IdP mapping and revocation evidence remains required before real identity readiness. |
| `VS3-H04` | HUMAN_REQUIRED | human_required | Real topology and network-policy evidence remains required before VS3-P. |
| `VS3-H05` | HUMAN_REQUIRED | human_required | Approved live-provider rehearsal remains required before live readiness. |
| `VS3-H06` | HUMAN_REQUIRED | human_required | Human operator UX/trust acceptance or rejection remains required before human UX acceptance. |
| `VS3-H07` | HUMAN_REQUIRED | human_required | Signed migration/backup/restore drill remains required before migration/restore readiness. |

## Before Evidence

Controlled aligned tamper before the fix:

1. Set `reports/security/vs3-request-context-proof.json` to `status=failed`.
2. Embedded the same failed body into `reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json`.
3. Regenerated dependent human-gate hashes with:
   - `cornerstone human-gate evidence-status ...`
   - `cornerstone human-gate review-kit ...`
   - `cornerstone human-gate vs3-p-gate ...`
4. Ran `cornerstone security vs3-local-checkpoint --json`.

Observed result before the fix:

```text
checkpoint_rc 0
checkpoint_status success
checkpoint_final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
failed_conditions []
request_context_identity {'matches_embedded_current_file': True, 'embedded_status': None, 'file_status': None, 'status_success': None, 'semantic_error_codes': None, 'error_code': None}
negative_component_mismatches 0
negative_component_semantic_failures None
```

Root cause:

- `_vs3_local_checkpoint_component_proof_identity` only compared canonical JSON hashes for the embedded proof body and current component proof file.
- It did not independently require the expected component proof schema or `status=success`.
- If the component file and aggregate embedded body drifted together into a failed proof state while scenario counts stayed optimistic, the local checkpoint could still pass.

## Implementation Summary

Updated `packages/cornerstone_cli/main.py`:

- Added expected schema versions for the nine component proof reports.
- Added per-component fields:
  - `expected_schema_version`
  - `embedded_schema_version`
  - `file_schema_version`
  - `embedded_status`
  - `file_status`
  - `embedded_schema_matches_expected`
  - `file_schema_matches_expected`
  - `embedded_status_success`
  - `file_status_success`
  - `semantics_success`
  - `semantic_error_codes`
- Added semantic error codes:
  - `CS_VS3_COMPONENT_PROOF_SCHEMA_MISMATCH`
  - `CS_VS3_COMPONENT_PROOF_STATUS_NOT_SUCCESS`
- Added checkpoint conditions:
  - `component_proof_<key>_matches_current_file`
  - `component_proof_<key>_semantics_success`
- Added summary and negative-evidence counter:
  - `component_proof_report_semantic_failures`

Updated `tests/scenario/test_scaffold_cli.py`:

- Positive local checkpoint test now asserts every component proof has expected schema/status semantics.
- Stale component identity test now expects a semantic failure as well as an identity mismatch.
- Added `test_vs3_local_checkpoint_rejects_component_proof_status_failure_even_when_identity_matches`.

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
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_status_failure_even_when_identity_matches
```

Result:

```text
Ran 3 tests in 89.044s
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
checkpoint_id vs3l_manifest_dfd1b340c07e926e
component_semantic_failures 0
negative_component_semantic_failures 0
failed_conditions []
connectorhub_source_proof cs.vs3_connectorhub_source_proof.v0 success True []
egress_sandbox_proof cs.vs3_egress_sandbox_proof.v0 success True []
evidence_reconciliation cs.vs3_evidence_reconciliation.v0 success True []
final_regression_proof cs.vs3_final_regression_proof.v0 success True []
observability_proof cs.vs3_observability_proof.v0 success True []
opa_policy_proof cs.vs3_opa_policy_proof.v0 success True []
postgres_rls_proof cs.vs3_postgres_rls_proof.v0 success True []
request_context_proof cs.vs3_request_context_proof.v0 success True []
tool_registry_proof cs.vs3_tool_registry_proof.v0 success True []
```

Controlled aligned tamper after the fix:

```text
dep_rc human-gate evidence-status 0
dep_rc human-gate review-kit 0
dep_rc human-gate vs3-p-gate 4
checkpoint_rc 4
checkpoint_status failed
checkpoint_final_verdict VS3_L_LOCAL_DEV_ASSURANCE_NOT_VERIFIED
failed_conditions ['component_proof_request_context_proof_semantics_success']
request_context_semantics {'matches_embedded_current_file': True, 'embedded_status': 'failed', 'file_status': 'failed', 'semantics_success': False, 'semantic_error_codes': ['CS_VS3_COMPONENT_PROOF_STATUS_NOT_SUCCESS']}
negative_selected {'component_proof_report_mismatches': 0, 'component_proof_report_semantic_failures': 1, 'vs3_p_claimed_by_checkpoint': 0, 'production_readiness_claimed_by_checkpoint': 0, 'security_acceptance_claimed_by_checkpoint': 0, 'human_acceptance_claimed_by_checkpoint': 0}
```

## Pass / Fail Criteria

PASS:

- `cornerstone security vs3-local-checkpoint --json` exits 0 only when all nine component proofs match their embedded aggregate bodies and have the expected schema plus `status=success`.
- `component_proof_report_semantic_failures == 0`.
- `component_proof_<key>_semantics_success` is true for all nine component proof keys.
- Controlled aligned tamper exits 4 with `component_proof_request_context_proof_semantics_success` as a failed condition.
- All VS3-P, production/on-prem, security acceptance, and human acceptance claims remain `NOT_CLAIMED`.

FAIL:

- A component proof with `status=failed` can still produce a successful local checkpoint.
- The checkpoint only detects hash mismatch but not matching failed semantics.
- The failure path claims VS3-P, production/on-prem readiness, security acceptance, or human acceptance.
- The fix requires human/external evidence or live credentials for local/CI verification.

## Remaining Human Gates

Still `HUMAN_REQUIRED`:

- `VS3-H01` owner architecture/dependency/migration/security approval.
- `VS3-H02` independent security review.
- `VS3-H03` real IdP mapping/revocation proof.
- `VS3-H04` real topology/network-policy proof.
- `VS3-H05` approved live-provider rehearsal.
- `VS3-H06` human operator UX/trust acceptance.
- `VS3-H07` migration/backup/restore drill.

This checkpoint does not unlock or claim VS3-P.
