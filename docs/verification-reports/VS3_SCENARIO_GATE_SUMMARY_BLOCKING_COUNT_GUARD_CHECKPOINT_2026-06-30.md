# VS3 Scenario Gate Summary Blocking Count Guard Checkpoint - 2026-06-30

**Status:** Local deterministic checkpoint for one VS3 verifier hardening slice.
**Scope:** Native VS3 scenario gate `summary.blocking` consistency.
**Related rows:** `VS3-GATE-004`, `VS3-REG-004`, proof-boundary support for `VS3-REG-005`.
**Proof boundary:** Local CLI/test evidence only. This is not VS3-L completion, VS3-P readiness, production/on-prem readiness, security acceptance, live-provider readiness, migration/restore readiness, or human acceptance.

## Full VS3 Mapping

Full VS3 inventory remains:

- `42` `MUST_PASS`
- `8` `REGRESSION`
- `7` `HUMAN_REQUIRED`
- `57` total rows

Execution classification for this slice:

- In this slice: `VS3-GATE-004`, `VS3-REG-004`, proof-boundary support for `VS3-REG-005`.
- Later slice: all other AI-owned `VS3-*` `MUST_PASS` and `REGRESSION` rows.
- Human-required: `VS3-H01` through `VS3-H07`.

## Slice Contract

Goal:

- Make `cornerstone scenario gate <report> --json` reject a VS3 local-dev assurance report when `summary.blocking` contradicts the scenario rows and gate-computed `blocking_count`.

In scope:

- `coverage_validation.expected_summary_counts.blocking`
- `coverage_validation.actual_summary_counts.blocking`
- `coverage_validation.reported_summary_counts.blocking`
- `coverage_validation.mismatched_summary_fields`
- Native JSON gate output and focused regression coverage

Out of scope:

- RequestContext, RLS, OPA, egress, ConnectorHub live/provider flows, Tool SDK, signed registry, Agent Pack activation, and all VS3 human evidence gates.

Expected behavior:

- A clean VS3 local-dev assurance report has `summary.blocking=0`.
- A report with zero AI-owned blocking rows but nonzero `summary.blocking` fails the gate with `CS_VS3_SCENARIO_COVERAGE_INVALID`.
- The failure must be attributed to `coverage_validation.mismatched_summary_fields == ["blocking"]` while unrelated identity, runtime metadata, human-required, claim-boundary, row-ref, aggregate-ref, and source-transcript validators remain passable.

## Before Evidence

Before this patch, `summary.blocking` was omitted from the scenario gate summary-count contract. A report with `summary.blocking=999` passed:

```text
seed_exit 0
original_summary_blocking 0
gate_exit 0
gate_status success
error_codes []
coverage_status passed
mismatched_summary_fields []
reported_summary_counts {'fail': 0, 'human_required': 7, 'not_run': 0, 'not_verified': 0, 'pass': 50, 'scenario_count': 57}
expected_summary_counts {'fail': 0, 'human_required': 7, 'not_run': 0, 'not_verified': 0, 'pass': 50, 'scenario_count': 57}
blocking_count 0
```

Interpretation:

- The gate detected zero actual blocking rows.
- The gate did not compare reported `summary.blocking` to the row-derived blocking count.
- A misleading summary claim could pass the native VS3 scenario gate.

## Change Summary

Changed:

- `packages/cornerstone_cli/main.py`
  - Adds `blocking` to `actual_summary_counts` using the gate-derived `blocking` rows.
  - Adds `blocking: 0` to `expected_summary_counts`.
  - Exposes reported `summary.blocking` through the existing `reported_summary_counts` projection.
- `tests/scenario/test_scaffold_cli.py`
  - Adds `test_vs3_scenario_gate_rejects_local_dev_claim_with_summary_blocking_mismatch`.
  - Mutates a generated report in place so report identity and runtime metadata remain valid.
  - Asserts the only summary mismatch is `blocking`.

## Verification Evidence

Compile:

```text
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
exit 0
```

Focused regression:

```text
PYTHONPATH=packages python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_summary_blocking_mismatch
.
----------------------------------------------------------------------
Ran 1 test in 23.671s

OK
```

Post-patch direct tamper probe:

```text
seed_exit 0
original_summary_blocking 0
gate_exit 4
gate_status failed
error_codes ['CS_VS3_SCENARIO_COVERAGE_INVALID']
coverage_status failed
mismatched_summary_fields ['blocking']
reported_summary_counts {'blocking': 999, 'fail': 0, 'human_required': 7, 'not_run': 0, 'not_verified': 0, 'pass': 50, 'scenario_count': 57}
expected_summary_counts {'blocking': 0, 'fail': 0, 'human_required': 7, 'not_run': 0, 'not_verified': 0, 'pass': 50, 'scenario_count': 57}
actual_summary_counts {'blocking': 0, 'fail': 0, 'human_required': 7, 'not_run': 0, 'not_verified': 0, 'pass': 50, 'scenario_count': 57}
blocking_count 0
```

## Decision

This slice passes locally for the native VS3 scenario gate summary blocking-count guard.

Remaining:

- `VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`.
- Full VS3-L and VS3-P remain unclaimed by this checkpoint.
