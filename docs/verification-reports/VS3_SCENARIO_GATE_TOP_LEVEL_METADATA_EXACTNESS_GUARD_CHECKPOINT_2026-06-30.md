# VS3 Scenario Gate Top-Level Metadata Exactness Guard Checkpoint

**Date:** 2026-06-30 KST
**Scope:** Local deterministic VS3 scenario gate hardening
**Status:** AI-verifiable slice passed; VS3-P remains HUMAN_REQUIRED

## Slice Contract

Goal:

- Prevent a VS3 local-dev assurance report from reshaping top-level safety metadata while still reporting validator-specific PASS.

In this slice:

- `VS3-GATE-003`: local/dev evidence must not overclaim or hide negative evidence semantics.
- `VS3-GATE-004`: native `cornerstone scenario gate ... --json` must expose deterministic validation status and errors.
- `VS3-OBS-002`: audit/evidence metadata must be exact enough to remain tamper-evident and queryable.
- `VS3-REG-004`: coverage and audit metadata omissions or drift must fail before release claims.
- `VS3-REG-005`: local/dev claims must stay bounded to the evidence actually present.

Later slices:

- Functional VS3 `CTX`, `RLS`, `OPA`, `EGR`, `CON`, `TOOL`, and broader `OBS` rows.
- Final regression breadth: `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-003`, `VS3-REG-006`, `VS3-REG-007`, and `VS3-REG-008`.

Human-required:

- `VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`.

## Baseline Gap

Local probes showed two validator-specific gaps:

- Adding a zero-valued unexpected top-level `negative_evidence` counter produced `negative_evidence_validation.status=passed`; the gate still failed only through unrelated traceability metadata.
- Duplicating top-level `evidence_refs`, `audit_refs`, or `policy_decision_refs` produced `aggregate_ref_validation.status=passed`; the gate still failed only through broader traceability/source-transcript checks.

Those paths made the report fail, but not for the direct metadata-exactness reason. A reviewer could miss that the dedicated validator was still accepting the malformed metadata.

## Change

`packages/cornerstone_cli/main.py` now treats top-level metadata drift as a first-class gate failure:

- unexpected top-level `negative_evidence` keys fail `negative_evidence_validation`;
- duplicate top-level `evidence_refs` fail `aggregate_ref_validation`;
- duplicate top-level `audit_refs` fail `aggregate_ref_validation`;
- duplicate top-level `policy_decision_refs` fail `aggregate_ref_validation`;
- failure payloads expose the duplicate or unexpected keys directly.

## Regression Tests

Added:

- `tests/scenario/test_scaffold_cli.py::ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_duplicate_aggregate_refs`
- `tests/scenario/test_scaffold_cli.py::ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_unexpected_negative_evidence_key`

The tests keep the generated report path stable so traceability remains valid. The aggregate-ref test also syncs transcript ref fields so `source_transcript_validation` and `self_command_transcript_validation` remain valid while `aggregate_ref_validation` fails directly.

## Verification

Focused compile:

```bash
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
```

Result: exit code `0`.

Focused new tests:

```bash
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_duplicate_aggregate_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_unexpected_negative_evidence_key
```

Result:

```text
Ran 2 tests in 46.629s
OK
```

Adjacent aggregate and negative-evidence gate suite:

```bash
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_duplicate_aggregate_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_without_aggregate_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_aggregate_evidence_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_missing_aggregate_evidence_path \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_row_source_report_ref_missing_from_aggregate_evidence_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_row_audit_ref_missing_from_aggregate_audit_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_row_policy_decision_ref_missing_from_aggregate_policy_decision_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_aggregate_audit_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_aggregate_policy_decision_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_without_negative_evidence \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_incomplete_negative_evidence \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_unexpected_negative_evidence_key \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_nonzero_negative_evidence \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_boolean_negative_evidence
```

Result:

```text
Ran 14 tests in 323.684s
OK
```

Component proof refresh:

```text
security vs3-request-context exit 0 status success output /Users/jiyong/playground/Cornerstone/reports/security/vs3-request-context-proof.json
security vs3-postgres-rls exit 0 status success output reports/db/vs3-postgres-rls-proof.json
security vs3-opa-policy exit 0 status success output reports/policy/vs3-opa-policy-proof.json
security vs3-egress-sandbox exit 0 status success output reports/security/vs3-egress-sandbox-proof.json
security vs3-connectorhub-source exit 0 status success output reports/security/vs3-connectorhub-source-proof.json
security vs3-tool-registry exit 0 status success output reports/security/vs3-tool-registry-proof.json
security vs3-observability exit 0 status success output reports/observability/vs3-observability-proof.json
```

Canonical VS3 scenario verify and gate:

```text
verify exit 0
verify status success
verify final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
verify summary {'blocking': 0, 'fail': 0, 'human_required': 7, 'not_run': 0, 'not_verified': 0, 'pass': 50, 'product_feature_claims': 'VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_GATES_REQUIRED', 'scenario_count': 57}
gate exit 0
gate status success
gate final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
gate summary {'blocking': 0, 'fail': 0, 'human_required': 7, 'not_run': 0, 'not_verified': 0, 'pass': 50, 'product_feature_claims': 'VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_GATES_REQUIRED', 'scenario_count': 57}
gate errors []
gate negative_evidence_validation passed
gate aggregate_ref_validation passed
```

## Proof Boundary

This checkpoint proves only local deterministic scenario-gate hardening.

It does not prove VS3-P, production/on-prem readiness, real IdP readiness, live-provider readiness, independent security acceptance, migration/restore readiness, or human UX acceptance.

`VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`.
