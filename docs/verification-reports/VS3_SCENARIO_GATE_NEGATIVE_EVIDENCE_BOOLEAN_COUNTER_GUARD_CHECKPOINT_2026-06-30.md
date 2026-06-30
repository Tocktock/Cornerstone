# VS3 Scenario Gate Negative Evidence Boolean Counter Guard Checkpoint - 2026-06-30

Status: Local deterministic checkpoint.

## Slice Contract

Goal: make `cornerstone scenario gate ... --json` fail closed when a VS3 local-dev assurance report encodes a negative-evidence counter as a boolean.

In this slice:

- `VS3-GATE-003`
- `VS3-GATE-004`
- `VS3-EGR-001`
- `VS3-EGR-004`
- `VS3-EGR-006`
- `VS3-CON-003`
- `VS3-TOOL-005`
- `VS3-REG-003`
- `VS3-REG-004`
- `VS3-REG-005`

Later slices:

- `VS3-CTX-001` through `VS3-CTX-005`
- `VS3-RLS-001` through `VS3-RLS-006`
- `VS3-OPA-001` through `VS3-OPA-005`
- remaining `VS3-EGR-*`
- remaining `VS3-CON-*`
- remaining `VS3-TOOL-*`
- `VS3-OBS-001` through `VS3-OBS-003`
- remaining `VS3-REG-*`

Human-required:

- `VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`.

## Baseline Gap

The VS3 scenario gate rejected string and negative-number values in `negative_evidence`, but accepted boolean `false` as numeric zero because Python booleans are integer-like.

Baseline probe that returned `status=success` before this slice:

```json
{
  "negative_evidence": {
    "forged_authority_acceptances": false
  }
}
```

The report still carried the VS3 local-dev assurance claim, so boolean placeholders could make safety counters look measured when they were not explicit counters.

## Change

`packages/cornerstone_cli/main.py` now treats boolean negative-evidence values as malformed counter input.

The gate records:

```json
{
  "issue": "value_bool_not_counter"
}
```

and returns `CS_VS3_NEGATIVE_EVIDENCE_INVALID` when the report claims VS3 local-dev assurance.

## Regression Test

Added:

`tests/scenario/test_scaffold_cli.py::ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_boolean_negative_evidence`

The test verifies:

- exit code `4`;
- `status=failed`;
- error code `CS_VS3_NEGATIVE_EVIDENCE_INVALID`;
- `negative_evidence_validation.status=failed`;
- `malformed_entries.forged_authority_acceptances.issue=value_bool_not_counter`;
- coverage, human-required, claim-boundary, row-ref, aggregate-ref, and source-transcript validations remain passing.

## Verification

Commands run:

```bash
python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_boolean_negative_evidence
```

Result:

```text
Ran 1 test in 23.892s
OK
```

```bash
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
```

Result: exit code `0`.

```bash
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_without_negative_evidence \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_incomplete_negative_evidence \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_nonzero_negative_evidence \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_boolean_negative_evidence \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_without_row_evidence_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_row_or_source_report_errors
```

Result:

```text
Ran 6 tests in 151.460s
OK
```

## Proof Boundary

This checkpoint proves only local deterministic VS3 gate hardening.

It does not prove VS3-P, production/on-prem readiness, real IdP readiness, live provider readiness, independent security acceptance, migration/restore readiness, or human UX acceptance.

`VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`.
