# VS3 Scenario Gate Scenario-Results Shape Guard Checkpoint - 2026-06-30

**Status:** Local verifier hardening checkpoint.
**Scope:** VS3-GATE-003, VS3-GATE-004, VS3-REG-004, and VS3-REG-005.
**Proof boundary:** Local deterministic CLI/test evidence only.
**Non-claims:** This checkpoint does not claim VS3-P, production/on-prem readiness, live-provider readiness, real IdP readiness, migration/restore readiness, security acceptance, or human UX acceptance.

## Slice Contract

Goal:
- Prevent malformed `scenario_results` data from crashing `cornerstone scenario gate` or bypassing structured VS3 scenario coverage validation.

In scope:
- Deterministic type/shape validation for the `scenario_results` report field.
- Structured VS3 coverage failure when the scenario row ledger is not a list.
- Regression coverage for a tampered local/dev report whose `scenario_results` field is replaced with approval-like text.

Out of scope:
- New VS3 runtime feature behavior.
- New ConnectorHub, RLS, OPA, egress, tool registry, or operator UX implementation.
- Any real production, on-prem, live-provider, real-network, real-IdP, migration/restore, security-acceptance, or human-acceptance claim.

Full VS3 mapping:
- `in_this_slice`: `VS3-GATE-003`, `VS3-GATE-004`, `VS3-REG-004`, `VS3-REG-005`.
- `later_slice`: `VS3-GATE-001`, `VS3-GATE-002`, `VS3-CTX-001` through `VS3-CTX-005`, `VS3-RLS-001` through `VS3-RLS-006`, `VS3-OPA-001` through `VS3-OPA-005`, `VS3-EGR-001` through `VS3-EGR-006`, `VS3-CON-001` through `VS3-CON-006`, `VS3-TOOL-001` through `VS3-TOOL-007`, `VS3-OBS-001` through `VS3-OBS-003`, `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-003`, `VS3-REG-006`, `VS3-REG-007`, `VS3-REG-008`.
- `HUMAN_REQUIRED`: `VS3-H01` through `VS3-H07`.

## Baseline Gap

Before this slice, a tampered VS3 local/dev report could replace `scenario_results` with a string and crash the native scenario gate before it produced JSON.

Tamper applied:

```text
scenario_results = APPROVED: all scenario rows passed by local verifier
traceability.transcript_paths = [checked tampered report path]
```

Observed baseline:

```text
seed_exit 0
gate_exit 1
stdout_prefix
stderr_tail Traceback (most recent call last):
  File "/Users/jiyong/playground/Cornerstone/./cornerstone", line 13, in <module>
    raise SystemExit(main())
  File "/Users/jiyong/playground/Cornerstone/packages/cornerstone_cli/main.py", line 23701, in main
    return args.func(args)
  File "/Users/jiyong/playground/Cornerstone/packages/cornerstone_cli/main.py", line 18704, in command_scenario_gate
    if row.get("owner", "AI") != "Human"
AttributeError: 'str' object has no attribute 'get'
stdout_json_error JSONDecodeError Expecting value: line 1 column 1 (char 0)
```

Interpretation:
- The CLI failed with an unhandled traceback instead of stable JSON.
- The failure was not attributable to VS3 scenario coverage validation in the gate payload.
- A malformed row ledger could obscure whether the gate had actually checked the frozen 57-row contract.

## Change

Changed files:

- `packages/cornerstone_cli/main.py`
- `tests/scenario/test_scaffold_cli.py`

Gate behavior added:

- `cornerstone scenario gate <report> --json` now normalizes `scenario_results` before scanning rows.
- Non-list `scenario_results` is recorded as `scenario_results_not_list`.
- Blocking-row detection now ignores non-object rows safely.
- VS3 coverage validation includes `coverage_validation.scenario_results_not_list`.
- VS3-L local/dev reports with malformed `scenario_results` fail with `CS_VS3_SCENARIO_COVERAGE_INVALID`.

## Regression Tests

New negative test:

```text
tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_scenario_results
```

Expected failure for tampered report:

```text
exit 4
status failed
error CS_VS3_SCENARIO_COVERAGE_INVALID
coverage_validation.status failed
coverage_validation.scenario_results_not_list ['scenario_results']
coverage_validation.missing_scenario_ids count 57
coverage_validation.reported_scenario_count 0
```

## Verification

Syntax:

```text
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
exit 0
```

Focused new test:

```text
python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_scenario_results

Ran 1 test in 25.766s
OK
```

Adjacent gate and human-boundary tests:

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_scenario_results \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_human_required_blockers \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_human_required_blocker_overclaim \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_human_required_entry_overclaim \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_p_gate_blocks_on_human_evidence_not_ai_rows \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_final_regression_gate_rejects_human_gate_overclaim

Ran 6 tests in 211.563s
OK
```

Manual tamper after patch:

```text
seed_exit 0
gate_exit 4
status failed
codes ['CS_VS3_SCENARIO_COVERAGE_INVALID', 'CS_VS3_HUMAN_REQUIRED_BOUNDARY_INVALID']
coverage_validation failed
scenario_results_not_list ['scenario_results']
missing_scenario_count 57
reported_scenario_count 0
```

The source-tree-inclusive component proof and canonical VS3 scenario report must be refreshed after this checkpoint file exists so the final local source snapshot includes the guard and this decision trail.

## Human-Required Boundary

These rows remain `HUMAN_REQUIRED`:

- `VS3-H01`: human architecture/security approval.
- `VS3-H02`: independent security review and retest.
- `VS3-H03`: real IdP mapping and revocation evidence.
- `VS3-H04`: real on-prem network controls evidence.
- `VS3-H05`: approved live ConnectorHub/provider rehearsal.
- `VS3-H06`: human operator UX/trust review.
- `VS3-H07`: supervised migration/backup/restore drill.

This checkpoint only hardens the local VS3-L scenario gate. It does not unlock VS3-P or replace signed human/on-prem evidence.
