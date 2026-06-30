# VS3 Scenario Gate Human-Required Blocker Shape Guard Checkpoint - 2026-06-30

**Status:** Local verifier hardening checkpoint.
**Scope:** VS3-GATE-003, VS3-GATE-004, VS3-OBS-003, VS3-REG-004, and VS3-REG-005.
**Proof boundary:** Local deterministic CLI/test evidence only.
**Non-claims:** This checkpoint does not claim VS3-P, production/on-prem readiness, live-provider readiness, real IdP readiness, migration/restore readiness, security acceptance, or human UX acceptance.

## Slice Contract

Goal:
- Prevent malformed `human_required_blockers` data from crashing `cornerstone scenario gate` or bypassing structured VS3 human-gate validation.

In scope:
- Deterministic type/shape validation for the `human_required_blockers` report field.
- Structured gate errors for missing, malformed, unexpected, or duplicate human blocker rows.
- Regression coverage for a tampered local/dev report whose traceability remains valid while `human_required_blockers` is not a list.

Out of scope:
- New VS3 runtime feature behavior.
- New ConnectorHub, RLS, OPA, egress, tool registry, or operator UX implementation.
- Any real production, on-prem, live-provider, real-network, real-IdP, migration/restore, security-acceptance, or human-acceptance claim.

Full VS3 mapping:
- `in_this_slice`: `VS3-GATE-003`, `VS3-GATE-004`, `VS3-OBS-003`, `VS3-REG-004`, `VS3-REG-005`.
- `later_slice`: `VS3-GATE-001`, `VS3-GATE-002`, `VS3-CTX-001` through `VS3-CTX-005`, `VS3-RLS-001` through `VS3-RLS-006`, `VS3-OPA-001` through `VS3-OPA-005`, `VS3-EGR-001` through `VS3-EGR-006`, `VS3-CON-001` through `VS3-CON-006`, `VS3-TOOL-001` through `VS3-TOOL-007`, `VS3-OBS-001`, `VS3-OBS-002`, `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-003`, `VS3-REG-006`, `VS3-REG-007`, `VS3-REG-008`.
- `HUMAN_REQUIRED`: `VS3-H01` through `VS3-H07`.

## Baseline Gap

Before this slice, a tampered VS3 local/dev report could replace `human_required_blockers` with a string and crash the native scenario gate before it produced JSON.

Tamper applied:

```text
human_required_blockers = APPROVED: all human gates accepted by local verifier
traceability.transcript_paths = [checked tampered report path]
every row transcript_paths = [checked tampered report path]
```

Observed baseline:

```text
seed_exit 0
gate_exit 1
stdout_prefix
stderr_tail Traceback (most recent call last):
  File "/Users/jiyong/playground/Cornerstone/./cornerstone", line 13, in <module>
    raise SystemExit(main())
  File "/Users/jiyong/playground/Cornerstone/packages/cornerstone_cli/main.py", line 23659, in main
    return args.func(args)
  File "/Users/jiyong/playground/Cornerstone/packages/cornerstone_cli/main.py", line 18809, in command_scenario_gate
    row.get("scenario_id") or row.get("id")
AttributeError: 'str' object has no attribute 'get'
stdout_json_error JSONDecodeError Expecting value: line 1 column 1 (char 0)
```

Interpretation:
- The CLI failed with an unhandled traceback instead of stable JSON.
- The failure was not attributable to the VS3 human-required proof boundary in the gate payload.

## Change

Changed files:

- `packages/cornerstone_cli/main.py`
- `tests/scenario/test_scaffold_cli.py`

Gate behavior added:

- `cornerstone scenario gate <report> --json` now normalizes `human_required_blockers` before reading row fields.
- Non-list `human_required_blockers` is reported as `human_required_validation.human_required_blockers_not_list`.
- Missing, unexpected, and duplicate blocker IDs are included in `human_required_validation`.
- VS3-L local/dev reports with malformed blocker shape fail with `CS_VS3_HUMAN_REQUIRED_BLOCKERS_INVALID`.
- The gate still keeps `VS3-H01` through `VS3-H07` as `HUMAN_REQUIRED`; this is not human approval evidence.

## Regression Tests

New negative test:

```text
tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_human_required_blockers
```

Expected failure for tampered report:

```text
exit 4
status failed
error CS_VS3_HUMAN_REQUIRED_BLOCKERS_INVALID
human_required_validation.status failed
human_required_blockers_not_list ['human_required_blockers']
missing_human_required_blockers ['VS3-H01', 'VS3-H02', 'VS3-H03', 'VS3-H04', 'VS3-H05', 'VS3-H06', 'VS3-H07']
traceability_validation.status passed
claim_boundary_validation.status passed
completion_claim_validation.status passed
gate_metadata_validation.status passed
```

## Verification

Syntax:

```text
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
exit 0
```

Focused new test:

```text
python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_human_required_blockers

Ran 1 test in 28.198s
OK
```

Adjacent human-required and human-gate tests:

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_human_required_blockers \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_human_required_blocker_overclaim \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_human_required_entry_overclaim \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_p_gate_blocks_on_human_evidence_not_ai_rows \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_final_regression_gate_rejects_human_gate_overclaim

Ran 5 tests in 183.940s
OK
```

Manual tamper after patch:

```text
seed_exit 0
gate_exit 4
status failed
codes ['CS_VS3_HUMAN_REQUIRED_BOUNDARY_INVALID', 'CS_VS3_HUMAN_REQUIRED_BLOCKERS_INVALID']
human_required_validation failed
human_required_blockers_not_list ['human_required_blockers']
missing_human_required_blockers ['VS3-H01', 'VS3-H02', 'VS3-H03', 'VS3-H04', 'VS3-H05', 'VS3-H06', 'VS3-H07']
traceability_validation passed
claim_boundary_validation passed
completion_claim_validation passed
gate_metadata_validation passed
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
