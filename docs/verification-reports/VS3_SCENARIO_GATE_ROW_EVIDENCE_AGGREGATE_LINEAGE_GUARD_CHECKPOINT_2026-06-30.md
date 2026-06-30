# VS3 Scenario Gate Row Evidence Aggregate Lineage Guard Checkpoint

**Date:** 2026-06-30 KST
**Scope:** Local deterministic VS3 scenario gate hardening
**Status:** AI-verifiable slice passed; VS3-P remains HUMAN_REQUIRED

## Slice Contract

Goal:

- Ensure a VS3 local-dev assurance report cannot keep row-level evidence while dropping that same evidence from the top-level aggregate `evidence_refs` ledger.

In this slice:

- `VS3-GATE-004`: native `cornerstone scenario gate ... --json` must reject row-to-aggregate evidence lineage loss.
- `VS3-OBS-002`: evidence/audit metadata must remain queryable and tamper-evident across row and aggregate report layers.
- `VS3-REG-004`: coverage and audit/evidence metadata cannot silently drop before release claims.
- `VS3-REG-005`: local/dev claims must stay bounded to the evidence actually present in the aggregate report.

Later slices:

- Functional VS3 `CTX`, `RLS`, `OPA`, `EGR`, `CON`, `TOOL`, and broader `OBS` rows.
- Final regression breadth: `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-003`, `VS3-REG-006`, `VS3-REG-007`, and `VS3-REG-008`.

Human-required:

- `VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`.

## Baseline Gap

A stable-path tamper probe removed the non-source-report row evidence ref `cornerstone principal context resolve --json` from top-level `evidence_refs`, while keeping it on row `VS3-CTX-001` and syncing command transcript aggregate refs.

Before this slice, the gate returned:

```text
gate_exit 0 status success
traceability_validation passed
row_ref_validation passed
aggregate_ref_validation passed
source_transcript_validation passed
self_command_transcript_validation passed
errors []
```

The existing aggregate validation checked row `source_report_refs`, row `audit_refs`, and row `policy_decision_refs`, but not ordinary row `evidence_refs`.

## Change

`packages/cornerstone_cli/main.py` now records and fails on:

- `aggregate_ref_validation.row_evidence_ref_missing_from_evidence_refs`

For VS3 local-dev assurance claims, any row-level `evidence_refs` value missing from top-level aggregate `evidence_refs` now returns `CS_VS3_AGGREGATE_EVIDENCE_METADATA_MISSING`.

## Regression Test

Added:

- `tests/scenario/test_scaffold_cli.py::ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_row_evidence_ref_missing_from_aggregate_evidence_refs`

The test removes only `cornerstone principal context resolve --json` from top-level `evidence_refs`. It keeps the generated report path stable and syncs transcript aggregate evidence refs so traceability and transcript validation remain passing while `aggregate_ref_validation` fails directly.

The valid-path gate test also asserts the new field is empty when canonical evidence is intact.

## Verification

Focused compile:

```bash
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
```

Result: exit code `0`.

Focused new test:

```bash
python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_row_evidence_ref_missing_from_aggregate_evidence_refs
```

Result:

```text
Ran 1 test in 23.361s
OK
```

Adjacent aggregate-ref gate suite:

```bash
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_row_evidence_ref_missing_from_aggregate_evidence_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_duplicate_aggregate_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_without_aggregate_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_aggregate_evidence_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_missing_aggregate_evidence_path \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_row_source_report_ref_missing_from_aggregate_evidence_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_row_audit_ref_missing_from_aggregate_audit_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_row_policy_decision_ref_missing_from_aggregate_policy_decision_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_aggregate_audit_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_aggregate_policy_decision_refs
```

Result:

```text
Ran 10 tests in 233.063s
OK
```

Canonical VS3 scenario verify and gate before checkpoint-file write:

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
gate aggregate_ref_validation passed
```

Canonical VS3 scenario verify and gate after checkpoint-file write:

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
gate coverage_validation passed
gate human_required_validation passed
gate claim_boundary_validation passed
gate completion_claim_validation passed
gate gate_metadata_validation passed
gate component_proof_validation passed
gate negative_evidence_validation passed
gate source_tree_current_validation passed
gate generated_dirty_snapshot_validation passed
gate traceability_validation passed
gate source_transcript_validation passed
gate row_ref_validation passed
gate aggregate_ref_validation passed
gate report_integrity_validation passed
gate self_command_transcript_validation passed
```

## Proof Boundary

This checkpoint proves only local deterministic scenario-gate hardening.

It does not prove VS3-P, production/on-prem readiness, real IdP readiness, live-provider readiness, independent security acceptance, migration/restore readiness, or human UX acceptance.

`VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`.
