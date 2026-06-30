# VS3 Scenario Gate Matrix and CLI Coverage Guard Checkpoint - 2026-06-30

**Status:** Local deterministic checkpoint for one VS3 verifier hardening slice.
**Scope:** Native VS3 scenario gate top-level matrix structural proof and CLI coverage exactness.
**Related rows:** `VS3-GATE-002`, `VS3-GATE-004`, `VS3-REG-004`.
**Proof boundary:** Local CLI/test evidence only. This is not VS3-L completion, VS3-P readiness, production/on-prem readiness, security acceptance, live-provider readiness, migration/restore readiness, or human acceptance.

## Slice Contract

Goal:
- Make `cornerstone scenario gate <report> --json` reject a VS3 local-dev assurance report when top-level `matrix_check` or `cli_coverage` is forged, stale, or inconsistent with the frozen VS3 matrix and native CLI contract.

In scope:
- `matrix_check.status`
- `matrix_check.row_count`
- `matrix_check.duplicates`
- matrix structural counts, paths, required checks, empty issue lists, and initial-status policy
- `cli_coverage.status`
- `cli_coverage.native_command`
- `cli_coverage.emits_per_row_evidence`
- `cli_coverage.emits_human_rows`
- `cli_coverage.emits_gate_metadata`
- `cli_coverage.command_transcript_count`
- native JSON gate output and focused regression coverage

Out of scope:
- RequestContext, RLS, OPA, egress, ConnectorHub live/provider flows, Tool SDK, signed registry, Agent Pack activation, and all VS3 human evidence gates.

Scenario mapping:
- `VS3-GATE-002`: in this slice. The gate now revalidates the frozen matrix structural proof object instead of trusting the report.
- `VS3-GATE-004`: in this slice. The gate now revalidates native CLI coverage proof for the VS3 verifier.
- `VS3-REG-004`: in this slice. The gate now catches top-level matrix/CLI coverage drift before a release claim can rely on the report.
- Remaining VS3 AI-owned rows: later slices.
- `VS3-H01` through `VS3-H07`: remain `HUMAN_REQUIRED`.

## Before Evidence

Command:

```text
PYTHONPATH=packages python3 - <<'PY'
...
PY
```

Observed current-state gap before the patch:

```text
seed_exit 0
cli_coverage.status exit 0 status success errors []
  coverage_validation passed []
  report_identity_validation passed []
  report_runtime_metadata_validation passed []
  traceability_validation passed []
  source_transcript_validation passed []
  self_command_transcript_validation passed []
cli_coverage.native_command exit 0 status success errors []
  coverage_validation passed []
  report_identity_validation passed []
  report_runtime_metadata_validation passed []
  traceability_validation passed []
  source_transcript_validation passed []
  self_command_transcript_validation passed []
cli_coverage.emits_per_row_evidence exit 0 status success errors []
  coverage_validation passed []
  report_identity_validation passed []
  report_runtime_metadata_validation passed []
  traceability_validation passed []
  source_transcript_validation passed []
  self_command_transcript_validation passed []
cli_coverage.emits_human_rows exit 0 status success errors []
  coverage_validation passed []
  report_identity_validation passed []
  report_runtime_metadata_validation passed []
  traceability_validation passed []
  source_transcript_validation passed []
  self_command_transcript_validation passed []
cli_coverage.emits_gate_metadata exit 0 status success errors []
  coverage_validation passed []
  report_identity_validation passed []
  report_runtime_metadata_validation passed []
  traceability_validation passed []
  source_transcript_validation passed []
  self_command_transcript_validation passed []
cli_coverage.command_transcript_count exit 0 status success errors []
  coverage_validation passed []
  report_identity_validation passed []
  report_runtime_metadata_validation passed []
  traceability_validation passed []
  source_transcript_validation passed []
  self_command_transcript_validation passed []
matrix_check.status exit 0 status success errors []
  coverage_validation passed []
  report_identity_validation passed []
  report_runtime_metadata_validation passed []
  traceability_validation passed []
  source_transcript_validation passed []
  self_command_transcript_validation passed []
matrix_check.row_count exit 0 status success errors []
  coverage_validation passed []
  report_identity_validation passed []
  report_runtime_metadata_validation passed []
  traceability_validation passed []
  source_transcript_validation passed []
  self_command_transcript_validation passed []
matrix_check.duplicates exit 0 status success errors []
  coverage_validation passed []
  report_identity_validation passed []
  report_runtime_metadata_validation passed []
  traceability_validation passed []
  source_transcript_validation passed []
  self_command_transcript_validation passed []
```

## Change Summary

Changed:
- `packages/cornerstone_cli/main.py`
  - Adds `matrix_cli_coverage_validation`.
  - Recomputes expected `matrix_check` from the frozen VS3 scenario set and contract paths.
  - Recomputes expected `cli_coverage` from the native VS3 verifier contract.
  - Fails local-dev assurance with `CS_VS3_MATRIX_CLI_COVERAGE_INVALID` when top-level matrix or CLI coverage proof drifts.
- `tests/scenario/test_scaffold_cli.py`
  - Adds `test_vs3_scenario_gate_rejects_local_dev_claim_with_matrix_or_cli_coverage_mismatch`.
  - Mutates `cli_coverage` and `matrix_check` while keeping identity, runtime metadata, row coverage, traceability, refs, and transcript evidence intact.

## Verification Evidence

Compile:

```text
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
exit 0
```

Focused regression:

```text
PYTHONPATH=packages python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_matrix_or_cli_coverage_mismatch
.
----------------------------------------------------------------------
Ran 1 test in 27.288s

OK
```

Post-patch tamper probe:

```text
seed_exit 0
cli_coverage.status exit 4 status failed errors ['CS_VS3_MATRIX_CLI_COVERAGE_INVALID'] matrix_cli failed ['cli_coverage.status']
  identity passed runtime passed coverage passed traceability passed source_transcript passed
cli_coverage.native_command exit 4 status failed errors ['CS_VS3_MATRIX_CLI_COVERAGE_INVALID'] matrix_cli failed ['cli_coverage.native_command']
  identity passed runtime passed coverage passed traceability passed source_transcript passed
cli_coverage.emits_per_row_evidence exit 4 status failed errors ['CS_VS3_MATRIX_CLI_COVERAGE_INVALID'] matrix_cli failed ['cli_coverage.emits_per_row_evidence']
  identity passed runtime passed coverage passed traceability passed source_transcript passed
cli_coverage.emits_human_rows exit 4 status failed errors ['CS_VS3_MATRIX_CLI_COVERAGE_INVALID'] matrix_cli failed ['cli_coverage.emits_human_rows']
  identity passed runtime passed coverage passed traceability passed source_transcript passed
cli_coverage.emits_gate_metadata exit 4 status failed errors ['CS_VS3_MATRIX_CLI_COVERAGE_INVALID'] matrix_cli failed ['cli_coverage.emits_gate_metadata']
  identity passed runtime passed coverage passed traceability passed source_transcript passed
cli_coverage.command_transcript_count exit 4 status failed errors ['CS_VS3_MATRIX_CLI_COVERAGE_INVALID'] matrix_cli failed ['cli_coverage.command_transcript_count']
  identity passed runtime passed coverage passed traceability passed source_transcript passed
matrix_check.status exit 4 status failed errors ['CS_VS3_MATRIX_CLI_COVERAGE_INVALID'] matrix_cli failed ['matrix_check.status']
  identity passed runtime passed coverage passed traceability passed source_transcript passed
matrix_check.row_count exit 4 status failed errors ['CS_VS3_MATRIX_CLI_COVERAGE_INVALID'] matrix_cli failed ['matrix_check.row_count']
  identity passed runtime passed coverage passed traceability passed source_transcript passed
matrix_check.duplicates exit 4 status failed errors ['CS_VS3_MATRIX_CLI_COVERAGE_INVALID'] matrix_cli failed ['matrix_check.duplicates']
  identity passed runtime passed coverage passed traceability passed source_transcript passed
```

Adjacent regression checks:

```text
PYTHONPATH=packages python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_report_identity_mismatch \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_runtime_metadata_mismatch \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_top_level_scope_mismatch \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_preserves_source_proof_boundary_and_transcript
....
----------------------------------------------------------------------
Ran 4 tests in 103.618s

OK
```

Clean native verify and gate:

```text
verify_exit 0
gate_exit 0
gate_status success
scenario_count 57
matrix_cli_coverage_validation passed []
report_runtime_metadata_validation passed []
report_identity_validation passed []
coverage_validation passed
traceability_validation passed
source_transcript_validation passed
```

## Decision

This slice passes locally for the native VS3 scenario gate matrix and CLI coverage proof exactness guard.

Remaining proof surfaces:
- `VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`.
- This checkpoint does not prove production/on-prem readiness, real IdP readiness, live-provider readiness, security acceptance, migration/restore readiness, or human UX acceptance.
