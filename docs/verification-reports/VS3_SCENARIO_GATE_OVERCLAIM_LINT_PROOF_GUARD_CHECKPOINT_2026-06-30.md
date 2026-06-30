# VS3 Scenario Gate Overclaim Lint Proof Guard Checkpoint - 2026-06-30

**Status:** Local deterministic checkpoint for one VS3 verifier hardening slice.
**Scope:** Native VS3 scenario gate top-level `overclaim_lint` proof exactness.
**Related rows:** `VS3-GATE-003`, `VS3-REG-005`.
**Proof boundary:** Local CLI/test evidence only. This is not VS3-L completion, VS3-P readiness, production/on-prem readiness, security acceptance, live-provider readiness, migration/restore readiness, or human acceptance.

## Slice Contract

Goal:
- Make `cornerstone scenario gate <report> --json` reject a VS3 local-dev assurance report when top-level `overclaim_lint` no longer proves the no-overclaim and no-production-claim boundary.

In scope:
- `overclaim_lint.schema_version`
- `overclaim_lint.status`
- `overclaim_lint.checked_paths`
- `overclaim_lint.claim_boundary`
- `overclaim_lint.claim_boundary_overclaim_fields`
- `overclaim_lint.findings`
- `overclaim_lint.reviewed_allowlist`
- `overclaim_lint.negative_evidence`
- native JSON gate output and focused regression coverage

Out of scope:
- RequestContext, RLS, OPA, egress, ConnectorHub live/provider flows, Tool SDK, signed registry, Agent Pack activation, and all VS3 human evidence gates.

Scenario mapping:
- `VS3-GATE-003`: in this slice. The gate now revalidates the top-level overclaim lint proof instead of trusting the report.
- `VS3-REG-005`: in this slice. The gate now catches local/dev proof being supported by a failed, missing, or overclaiming lint result.
- Remaining VS3 AI-owned rows: later slices.
- `VS3-H01` through `VS3-H07`: remain `HUMAN_REQUIRED`.

Full VS3 inventory remains:
- `42` `MUST_PASS`
- `8` `REGRESSION`
- `7` `HUMAN_REQUIRED`
- `57` total rows
- `0` duplicate scenario IDs

## Before Evidence

Observed current-state gap before the patch:

```text
seed_exit 0
case_count 26
evidence_reconciliation.status exit 4 status failed errors ['CS_VS3_COMPONENT_PROOF_INVALID']
overclaim_lint.status exit 0 status success errors []
  component_proof_validation passed []
  coverage_validation passed []
  traceability_validation passed []
  source_transcript_validation passed []
  matrix_cli_coverage_validation passed []
request_context_proof.status/checks/scenario_status and other embedded component proof mutations exit 4 with CS_VS3_COMPONENT_PROOF_INVALID
```

Interpretation:
- Embedded component proofs were already guarded.
- The top-level `overclaim_lint` proof object was not guarded, so a failed overclaim lint status could still pass the VS3 scenario gate.

## Change Summary

Changed:
- `packages/cornerstone_cli/main.py`
  - Adds `overclaim_lint_validation`.
  - Requires exact `cs.vs3_overclaim_lint.v0` schema, `status=passed`, empty findings, empty allowlist, empty overclaim fields, exact checked paths, exact no-claim boundary, and zero-valued overclaim negative-evidence counters.
  - Fails local-dev assurance with `CS_VS3_OVERCLAIM_LINT_INVALID` when the proof is missing or mismatched.
- `tests/scenario/test_scaffold_cli.py`
  - Adds `test_vs3_scenario_gate_rejects_local_dev_claim_with_overclaim_lint_mismatch`.
  - Mutates `overclaim_lint.status`, `findings`, `claim_boundary`, `negative_evidence`, and missing-object cases while keeping unrelated validations green.

## Verification Evidence

Compile:

```text
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
exit 0
```

Focused regression:

```text
PYTHONPATH=packages python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_overclaim_lint_mismatch
.
----------------------------------------------------------------------
Ran 1 test in 26.584s

OK
```

Post-patch direct tamper probe:

```text
seed_exit 0
gate_exit 4
status failed
error_codes ['CS_VS3_OVERCLAIM_LINT_INVALID']
overclaim_validation invalid_fields ['overclaim_lint.status']
```

Adjacent gate checks:

```text
PYTHONPATH=packages python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_matrix_or_cli_coverage_mismatch \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_overclaim_lint_mismatch \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_gate_metadata_overclaim \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_preserves_source_proof_boundary_and_transcript
....
----------------------------------------------------------------------
Ran 4 tests in 104.100s

OK
```

Clean native verify and gate:

```text
verify_exit 0
gate_exit 0
status success
overclaim_lint_validation status passed
overclaim_lint_validation invalid_fields []
error_count 0
```

## Decision

This slice passes locally for the native VS3 scenario gate overclaim-lint proof exactness guard.

Remaining proof surfaces:
- `VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`.
- This checkpoint does not prove production/on-prem readiness, real IdP readiness, live-provider readiness, security acceptance, migration/restore readiness, or human UX acceptance.
