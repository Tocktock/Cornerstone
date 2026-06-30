# VS3 Scenario Gate Row Evidence Path Field Guard Checkpoint - 2026-06-30

Status: Local deterministic checkpoint.

## Slice Contract

Goal: make `cornerstone scenario gate ... --json` fail closed when a VS3 local-dev assurance row has invalid local evidence in `evidence` or `evidence_paths`.

In this slice:

- `VS3-GATE-003`
- `VS3-GATE-004`
- `VS3-REG-004`
- `VS3-REG-005`

Later slices:

- `VS3-CTX-001` through `VS3-CTX-005`
- `VS3-RLS-001` through `VS3-RLS-006`
- `VS3-OPA-001` through `VS3-OPA-005`
- `VS3-EGR-001` through `VS3-EGR-006`
- `VS3-CON-001` through `VS3-CON-006`
- `VS3-TOOL-001` through `VS3-TOOL-007`
- `VS3-OBS-001` through `VS3-OBS-003`
- `VS3-REG-001` through `VS3-REG-003`
- `VS3-REG-006` through `VS3-REG-008`

Human-required:

- `VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`.

## Baseline Gap

The VS3 scenario gate rejected missing local paths in row `evidence_refs`, but accepted equivalent missing paths in row `evidence` or `evidence_paths`.

Baseline probes that returned `status=success` before this slice:

- `scenario_results[0].evidence = ["reports/security/DOES_NOT_EXIST_VS3_ROW_EVIDENCE.json"]`
- `scenario_results[0].evidence_paths = ["reports/security/DOES_NOT_EXIST_VS3_ROW_EVIDENCE.json"]`

## Change

`packages/cornerstone_cli/main.py` now validates row `evidence` and `evidence_paths` with the existing VS3 evidence-reference taxonomy.

The row reference validation now records:

- `missing_evidence_path_rows`
- `malformed_evidence_path_rows`

When the report claims VS3 local-dev assurance and either list is non-empty, the gate returns `CS_VS3_ROW_EVIDENCE_METADATA_MISSING`.

## Regression Test

Added:

`tests/scenario/test_scaffold_cli.py::ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_missing_row_evidence_or_evidence_paths`

The test keeps the generated report path stable so traceability remains valid, then verifies both invalid fields return:

- exit code `4`;
- `status=failed`;
- error code `CS_VS3_ROW_EVIDENCE_METADATA_MISSING`;
- `row_ref_validation.status=failed`;
- one `malformed_evidence_path_rows` entry with `issue=missing`;
- passing coverage, traceability, source transcript, human-required, claim-boundary, and aggregate-ref validations.

## Verification

Commands run:

```bash
python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_missing_row_evidence_or_evidence_paths
```

Result:

```text
Ran 1 test in 26.298s
OK
```

```bash
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
```

Result: exit code `0`.

```bash
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_without_row_evidence_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_row_evidence_ref \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_missing_row_evidence_path \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_missing_row_evidence_or_evidence_paths \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_unresolved_row_source_report_ref \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_source_report_ref_missing_from_evidence_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_unexpected_row_source_report_ref \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_row_audit_ref \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_row_policy_decision_ref
```

Result:

```text
Ran 9 tests in 229.614s
OK
```

## Proof Boundary

This checkpoint proves only local deterministic VS3 gate hardening.

It does not prove VS3-P, production/on-prem readiness, real IdP readiness, live provider readiness, independent security acceptance, migration/restore readiness, or human UX acceptance.

`VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`.
