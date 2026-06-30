# VS3 Scenario Gate Duplicate Row Reference Guard Checkpoint - 2026-06-30

Status: Local deterministic checkpoint.

## Slice Contract

Goal: make `cornerstone scenario gate ... --json` fail closed when a VS3 local-dev assurance row duplicates row-level proof references.

In this slice:

- `VS3-GATE-003`
- `VS3-GATE-004`
- `VS3-OBS-002`
- `VS3-REG-004`
- `VS3-REG-005`

Later slices:

- `VS3-CTX-001` through `VS3-CTX-005`
- `VS3-RLS-001` through `VS3-RLS-006`
- `VS3-OPA-001` through `VS3-OPA-005`
- `VS3-EGR-001` through `VS3-EGR-006`
- `VS3-CON-001` through `VS3-CON-006`
- `VS3-TOOL-001` through `VS3-TOOL-007`
- `VS3-OBS-001`
- `VS3-OBS-003`
- remaining `VS3-REG-*`

Human-required:

- `VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`.

## Baseline Gap

The VS3 scenario gate validated row reference presence and formats, but still accepted duplicated row references in:

- `evidence`
- `evidence_paths`
- `evidence_refs`
- `source_report_refs`
- `audit_refs`
- `policy_decision_refs`

Baseline probes that duplicated a `VS3-CTX-001` row reference returned `status=success` before this slice.

Duplicate row references can make a local-dev assurance report look more evidence-rich than it is and can mask coverage or lineage mistakes.

## Change

`packages/cornerstone_cli/main.py` now records duplicate row-reference fields in `row_ref_validation`:

- `duplicate_evidence_path_rows`
- `duplicate_evidence_ref_rows`
- `duplicate_source_report_ref_rows`
- `duplicate_audit_ref_rows`
- `duplicate_policy_decision_ref_rows`

When the report claims VS3 local-dev assurance and any duplicate list is non-empty, the gate returns `CS_VS3_ROW_EVIDENCE_METADATA_MISSING`.

## Regression Test

Added:

`tests/scenario/test_scaffold_cli.py::ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_duplicate_row_refs`

The test keeps the generated report path stable, duplicates one `VS3-CTX-001` reference in each target field, and verifies:

- exit code `4`;
- `status=failed`;
- error code `CS_VS3_ROW_EVIDENCE_METADATA_MISSING`;
- `row_ref_validation.status=failed`;
- only the expected duplicate-reference list is populated for each subcase;
- source transcript, traceability, coverage, human-required, claim-boundary, and aggregate-ref validations remain passing.

## Verification

Commands run:

```bash
python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_duplicate_row_refs
```

Result:

```text
Ran 1 test in 28.682s
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
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_duplicate_row_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_unresolved_row_source_report_ref \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_source_report_ref_missing_from_evidence_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_unexpected_row_source_report_ref \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_row_audit_ref \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_row_policy_decision_ref
```

Result:

```text
Ran 10 tests in 249.050s
OK
```

## Proof Boundary

This checkpoint proves only local deterministic VS3 gate hardening.

It does not prove VS3-P, production/on-prem readiness, real IdP readiness, live provider readiness, independent security acceptance, migration/restore readiness, or human UX acceptance.

`VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`.
