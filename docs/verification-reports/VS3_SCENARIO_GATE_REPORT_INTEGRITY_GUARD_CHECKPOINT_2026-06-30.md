# VS3 Scenario Gate Report Integrity Guard Checkpoint - 2026-06-30

Status: Local deterministic checkpoint.

## Slice Contract

Goal: make `cornerstone scenario gate ... --json` fail closed when a VS3 local-dev assurance report contains contradictory report-integrity state.

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

The VS3 scenario gate already rejected top-level `status=failed` and top-level `errors`, but accepted these contradictory local-dev assurance report states:

- an AI scenario row with `status=PASS` plus a non-empty row-level `errors` list;
- a nested `source_report` object with `status=failed`;
- a nested `source_report` object with non-empty `errors`.

Those cases allowed the gate to return `status=success` even though the report carried failure metadata.

## Change

`packages/cornerstone_cli/main.py` now emits `report_integrity_validation` during VS3 scenario gate evaluation.

The validation fails when:

- any scenario row contains non-empty `errors`;
- a nested `source_report` is present but is not an object;
- a nested `source_report.status` is present and not `success`;
- a nested `source_report.errors` value is non-empty.

When the report claims VS3 local-dev assurance and this validation fails, the gate returns `CS_VS3_REPORT_INTEGRITY_INVALID` with structured row/source-report details.

## Regression Test

Added:

`tests/scenario/test_scaffold_cli.py::ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_row_or_source_report_errors`

The test uses a path-stable generated VS3 report and verifies that all three baseline failure modes return:

- exit code `4`;
- `status=failed`;
- error code `CS_VS3_REPORT_INTEGRITY_INVALID`;
- `report_integrity_validation.status=failed`;
- preserved passing coverage, human-required, and claim-boundary validations.

## Verification

Commands run:

```bash
python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_row_or_source_report_errors
```

Result:

```text
Ran 1 test in 25.569s
OK
```

```bash
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
```

Result: exit code `0`.

```bash
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_row_or_source_report_errors \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_missing_ai_scenario_row \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_duplicate_or_unexpected_rows \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_row_id_scenario_id_mismatch \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_row_classification_mismatch \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_row_contract_content_mismatch \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_non_object_scenario_row \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_promoted_to_vs3_p_readiness
```

Result:

```text
Ran 8 tests in 200.013s
OK
```

## Proof Boundary

This checkpoint proves only local deterministic VS3 gate hardening.

It does not prove VS3-P, production/on-prem readiness, real IdP readiness, live provider readiness, independent security acceptance, migration/restore readiness, or human UX acceptance.

`VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`.
