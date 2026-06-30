# VS3 Scenario Gate Runtime Metadata Exactness Guard Checkpoint - 2026-06-30

**Status:** Local deterministic checkpoint for one VS3 verifier hardening slice.
**Scope:** Native VS3 scenario gate top-level runtime metadata exactness.
**Related rows:** `VS3-GATE-004`, `VS3-REG-004`.
**Proof boundary:** Local CLI/test evidence only. This is not VS3-L completion, VS3-P readiness, production/on-prem readiness, security acceptance, live-provider readiness, migration/restore readiness, or human acceptance.

## Slice Contract

Goal:
- Make `cornerstone scenario gate <report> --json` reject a VS3 local-dev assurance report when top-level runtime metadata is forged or stale.

In scope:
- `product`
- `mode`
- `ids.git_commit`
- `output_path`
- Native JSON gate output and focused regression coverage.

Out of scope:
- RequestContext, RLS, OPA, egress, ConnectorHub live/provider flows, Tool SDK, signed registry, Agent Pack activation, and all VS3 human evidence gates.

Scenario mapping:
- `VS3-GATE-004`: in this slice. The native VS3 scenario gate now validates exact CornerStone local runtime metadata.
- `VS3-REG-004`: in this slice. The gate now catches report runtime metadata drift before a release claim can rely on the report.
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
base {'product': 'CornerStone', 'mode': 'local_scaffold', 'tenant_id': 'local-dev', 'owner_id': 'local-user', 'namespace_id': 'personal', 'workspace_id': 'default', 'output_path': '/Users/jiyong/playground/Cornerstone/tmp/vs3-runtime-identity-preprobe-source.json'}
base_ids {'git_commit': 'd145c8d'}
product exit 0 status success errors []
  report_identity_validation passed []
  coverage_validation passed []
  traceability_validation passed []
  source_transcript_validation passed []
  self_command_transcript_validation passed []
mode exit 0 status success errors []
  report_identity_validation passed []
  coverage_validation passed []
  traceability_validation passed []
  source_transcript_validation passed []
  self_command_transcript_validation passed []
ids exit 0 status success errors []
  report_identity_validation passed []
  coverage_validation passed []
  traceability_validation passed []
  source_transcript_validation passed []
  self_command_transcript_validation passed []
output_path exit 0 status success errors []
  report_identity_validation passed []
  coverage_validation passed []
  traceability_validation passed []
  source_transcript_validation passed []
  self_command_transcript_validation passed []
```

## Change Summary

Changed:
- `packages/cornerstone_cli/main.py`
  - Adds `report_runtime_metadata_validation` with exact expected and actual `product`, `mode`, `ids.git_commit`, and `output_path`.
  - Fails local-dev assurance with `CS_VS3_REPORT_RUNTIME_METADATA_INVALID` when runtime metadata fields drift.
- `tests/scenario/test_scaffold_cli.py`
  - Adds `test_vs3_scenario_gate_rejects_local_dev_claim_with_runtime_metadata_mismatch`.
  - Mutates each runtime metadata field while keeping report identity, coverage, traceability, row refs, aggregate refs, and source transcript evidence intact.

## Verification Evidence

Compile:

```text
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
exit 0
```

Focused regression:

```text
PYTHONPATH=packages python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_runtime_metadata_mismatch
.
----------------------------------------------------------------------
Ran 1 test in 24.150s

OK
```

Post-patch tamper probe:

```text
seed_exit 0
product exit 4 status failed errors ['CS_VS3_REPORT_RUNTIME_METADATA_INVALID'] runtime failed ['product']
  identity passed coverage passed traceability passed source_transcript passed
mode exit 4 status failed errors ['CS_VS3_REPORT_RUNTIME_METADATA_INVALID'] runtime failed ['mode']
  identity passed coverage passed traceability passed source_transcript passed
ids.git_commit exit 4 status failed errors ['CS_VS3_REPORT_RUNTIME_METADATA_INVALID'] runtime failed ['ids.git_commit']
  identity passed coverage passed traceability passed source_transcript passed
output_path exit 4 status failed errors ['CS_VS3_REPORT_RUNTIME_METADATA_INVALID'] runtime failed ['output_path']
  identity passed coverage passed traceability passed source_transcript passed
```

Adjacent regression checks:

```text
PYTHONPATH=packages python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_report_identity_mismatch \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_top_level_scope_mismatch \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_preserves_source_proof_boundary_and_transcript
...
----------------------------------------------------------------------
Ran 3 tests in 71.575s

OK
```

Clean native verify and gate:

```text
verify_exit 0
gate_exit 0
gate_status success
scenario_count 57
report_runtime_metadata_validation passed []
report_identity_validation passed []
coverage_validation passed
traceability_validation passed
source_transcript_validation passed
```

## Decision

This slice passes locally for the native VS3 scenario gate runtime metadata exactness guard.

Remaining proof surfaces:
- `VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`.
- This checkpoint does not prove production/on-prem readiness, real IdP readiness, live-provider readiness, security acceptance, migration/restore readiness, or human UX acceptance.
