# VS3 Scenario Gate Scenario-Row Shape Guard Checkpoint - 2026-06-30

**Status:** Local verifier hardening checkpoint.
**Scope:** VS3-GATE-003, VS3-GATE-004, VS3-REG-004, and VS3-REG-005.
**Proof boundary:** Local deterministic CLI/test evidence only.
**Non-claims:** This checkpoint does not claim VS3-P, production/on-prem readiness, live-provider readiness, real IdP readiness, migration/restore readiness, security acceptance, or human UX acceptance.

## Slice Contract

Goal:
- Prevent a non-object row inside list-shaped `scenario_results` from crashing `cornerstone scenario gate` before VS3 scenario coverage validation can report the malformed ledger.

In scope:
- Deterministic row-object validation for entries inside `scenario_results`.
- Structured VS3 coverage failure when a scenario row is not an object.
- Regression coverage for a tampered local/dev report whose first scenario row is replaced with approval-like text.

Out of scope:
- New VS3 runtime feature behavior.
- New ConnectorHub, RLS, OPA, egress, tool registry, or operator UX implementation.
- Any real production, on-prem, live-provider, real-network, real-IdP, migration/restore, security-acceptance, or human-acceptance claim.

Full VS3 mapping:
- `in_this_slice`: `VS3-GATE-003`, `VS3-GATE-004`, `VS3-REG-004`, `VS3-REG-005`.
- `later_slice`: `VS3-GATE-001`, `VS3-GATE-002`, `VS3-CTX-001` through `VS3-CTX-005`, `VS3-RLS-001` through `VS3-RLS-006`, `VS3-OPA-001` through `VS3-OPA-005`, `VS3-EGR-001` through `VS3-EGR-006`, `VS3-CON-001` through `VS3-CON-006`, `VS3-TOOL-001` through `VS3-TOOL-007`, `VS3-OBS-001` through `VS3-OBS-003`, `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-003`, `VS3-REG-006`, `VS3-REG-007`, `VS3-REG-008`.
- `HUMAN_REQUIRED`: `VS3-H01` through `VS3-H07`.

## Baseline Gap

Before this slice, a tampered VS3 local/dev report could keep `scenario_results` as a list but replace one row with a string and crash the native scenario gate before it produced JSON.

Tamper applied:

```text
scenario_results[0] = APPROVED: VS3-GATE-001 passed by local verifier
traceability.transcript_paths = [checked tampered report path]
all object scenario row transcript_paths = [checked tampered report path]
```

Observed baseline:

```text
CASE scenario_results_row_string exit 1
no stdout
stderr_tail Traceback (most recent call last):
  File "/Users/jiyong/playground/Cornerstone/./cornerstone", line 13, in <module>
    raise SystemExit(main())
  File "/Users/jiyong/playground/Cornerstone/packages/cornerstone_cli/main.py", line 23708, in main
    return args.func(args)
  File "/Users/jiyong/playground/Cornerstone/packages/cornerstone_cli/main.py", line 18843, in command_scenario_gate
    if row.get("scenario_id") or row.get("id")
AttributeError: 'str' object has no attribute 'get'
```

After the first narrow fix, a second row-object assumption surfaced in row evidence validation:

```text
File "/Users/jiyong/playground/Cornerstone/packages/cornerstone_cli/main.py", line 20115, in command_scenario_gate
  row_id = str(row.get("scenario_id") or row.get("id") or "<unknown>")
AttributeError: 'str' object has no attribute 'get'
```

Interpretation:
- The CLI failed with unhandled tracebacks instead of stable JSON.
- The existing `row_not_object` coverage issue could not become the authoritative failure until all row-scanning paths ignored non-object rows safely.

## Change

Changed files:

- `packages/cornerstone_cli/main.py`
- `tests/scenario/test_scaffold_cli.py`

Gate behavior added:

- `cornerstone scenario gate <report> --json` now filters non-object rows when building scenario ID maps.
- Row evidence validation skips non-object rows after coverage validation records them as `row_not_object`.
- Aggregate source-report, audit, and policy-decision lineage checks also skip non-object rows.
- VS3-L local/dev reports with non-object scenario rows fail with `CS_VS3_SCENARIO_COVERAGE_INVALID`.

## Regression Tests

New negative test:

```text
tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_non_object_scenario_row
```

Expected failure for tampered report:

```text
exit 4
status failed
error CS_VS3_SCENARIO_COVERAGE_INVALID
coverage_validation.status failed
coverage_validation.scenario_results_not_list []
coverage_validation.missing_scenario_ids includes VS3-GATE-001
coverage_validation.reported_scenario_count 57
coverage_validation.row_identity_issues[0].issue row_not_object
```

## Verification

Syntax:

```text
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
exit 0
```

Focused new test:

```text
python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_non_object_scenario_row

Ran 1 test in 25.317s
OK
```

Adjacent gate and human-boundary tests:

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_non_object_scenario_row \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_scenario_results \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_human_required_blockers \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_human_required_blocker_overclaim \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_human_required_entry_overclaim \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_p_gate_blocks_on_human_evidence_not_ai_rows \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_final_regression_gate_rejects_human_gate_overclaim

Ran 7 tests in 219.081s
OK
```

Manual tamper after patch:

```text
seed_exit 0
gate_exit 4
status failed
codes ['CS_VS3_SCENARIO_COVERAGE_INVALID', 'CS_VS3_TRACEABILITY_METADATA_MISSING']
coverage_validation failed
scenario_results_not_list []
missing_contains_gate001 True
reported_scenario_count 57
row_identity_issue_0 {'id': None, 'issue': 'row_not_object', 'row_index': 0, 'scenario_id': None}
row_ref_validation passed
aggregate_ref_validation passed
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
