# VS3 Scenario Gate Report Identity Exactness Guard Checkpoint - 2026-06-30

**Status:** Local deterministic checkpoint for one VS3 verifier hardening slice.
**Scope:** Native VS3 scenario gate report identity exactness.
**Related rows:** `VS3-GATE-004`, `VS3-REG-004`.
**Proof boundary:** Local CLI/test evidence only. This is not VS3-L completion, VS3-P readiness, production/on-prem readiness, security acceptance, live-provider readiness, migration/restore readiness, or human acceptance.

## Slice Contract

Goal:
- Make `cornerstone scenario gate <report> --json` reject a VS3 local-dev assurance report when top-level report identity fields drift from the frozen VS3 verifier identity.

In scope:
- `schema_version`
- `scenario_set`
- `cli_schema_version`
- `command`
- `version`
- Native JSON gate output and focused regression coverage.

Out of scope:
- RequestContext, RLS, OPA, egress, ConnectorHub live/provider flows, Tool SDK, signed registry, Agent Pack activation, and all VS3 human evidence gates.

Scenario mapping:
- `VS3-GATE-004`: in this slice. The native VS3 scenario gate now validates exact source report identity.
- `VS3-REG-004`: in this slice. The gate now catches report-identity omission/drift before a release claim can rely on the report.
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
scenario_set exit 0 status success errors []
  coverage_validation passed []
  traceability_validation passed []
  source_transcript_validation passed []
  self_command_transcript_validation passed []
cli_schema_version exit 0 status success errors []
  coverage_validation passed []
  traceability_validation passed []
  source_transcript_validation passed []
  self_command_transcript_validation passed []
command exit 0 status success errors []
  coverage_validation passed []
  traceability_validation passed []
  source_transcript_validation passed []
  self_command_transcript_validation passed []
version exit 0 status success errors []
  coverage_validation passed []
  traceability_validation passed []
  source_transcript_validation passed []
  self_command_transcript_validation passed []
```

## Change Summary

Changed:
- `packages/cornerstone_cli/main.py`
  - Routes reports that still carry VS3 local-dev assurance markers into the VS3 gate even if top-level identity labels are tampered.
  - Adds `report_identity_validation` with exact expected and actual identity fields.
  - Fails local-dev assurance with `CS_VS3_REPORT_IDENTITY_INVALID` when identity fields drift.
- `tests/scenario/test_scaffold_cli.py`
  - Adds `test_vs3_scenario_gate_rejects_local_dev_claim_with_report_identity_mismatch`.
  - Mutates each identity field while keeping the rest of the report evidence intact.

## Verification Evidence

Compile:

```text
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
exit 0
```

Focused regression:

```text
PYTHONPATH=packages python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_report_identity_mismatch
.
----------------------------------------------------------------------
Ran 1 test in 26.680s

OK
```

Post-patch tamper probe:

```text
seed_exit 0
scenario_set exit 4 status failed errors ['CS_VS3_REPORT_IDENTITY_INVALID'] identity failed ['scenario_set']
  coverage passed traceability passed source_transcript passed
cli_schema_version exit 4 status failed errors ['CS_VS3_REPORT_IDENTITY_INVALID'] identity failed ['cli_schema_version']
  coverage passed traceability passed source_transcript passed
command exit 4 status failed errors ['CS_VS3_REPORT_IDENTITY_INVALID'] identity failed ['command']
  coverage passed traceability passed source_transcript passed
version exit 4 status failed errors ['CS_VS3_REPORT_IDENTITY_INVALID'] identity failed ['version']
  coverage passed traceability passed source_transcript passed
```

Adjacent regression checks:

```text
PYTHONPATH=packages python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_preserves_source_proof_boundary_and_transcript \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_top_level_scope_mismatch
..
----------------------------------------------------------------------
Ran 2 tests in 50.268s

OK
```

Clean native verify and gate:

```text
verify_exit 0
gate_exit 0
gate_status success
scenario_count 57
report_identity_validation passed []
coverage_validation passed
traceability_validation passed
source_transcript_validation passed
```

## Decision

This slice passes locally for the native VS3 scenario gate report identity exactness guard.

Remaining proof surfaces:
- `VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`.
- This checkpoint does not prove production/on-prem readiness, real IdP readiness, live-provider readiness, security acceptance, migration/restore readiness, or human UX acceptance.
