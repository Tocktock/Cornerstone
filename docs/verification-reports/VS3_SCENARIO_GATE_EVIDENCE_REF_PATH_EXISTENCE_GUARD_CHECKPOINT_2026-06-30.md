# VS3 Scenario Gate Evidence Ref Path Existence Guard Checkpoint

**Date:** 2026-06-30 KST
**Status:** Local deterministic checkpoint for one VS3 scenario-gate guard slice.
**Slice:** Evidence-ref path existence validation in `cornerstone scenario gate`.

## Scenario Mapping

Full VS3 mapping remains the frozen matrix:

- `MUST_PASS`: 42 rows.
- `REGRESSION`: 8 rows.
- `HUMAN_REQUIRED`: 7 rows.
- Duplicate scenario IDs: none.
- Missing required matrix fields: none.

Current delivery slice:

- `in_this_slice`: `VS3-GATE-004`, `VS3-REG-004`.
- `HUMAN_REQUIRED`: `VS3-H01`, `VS3-H02`, `VS3-H03`, `VS3-H04`, `VS3-H05`, `VS3-H06`, `VS3-H07`.
- `later_slice`: `VS3-GATE-001`, `VS3-GATE-002`, `VS3-GATE-003`, all `VS3-CTX-*`, `VS3-RLS-*`, `VS3-OPA-*`, `VS3-EGR-*`, `VS3-CON-*`, `VS3-TOOL-*`, `VS3-OBS-*`, `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-003`, `VS3-REG-005`, `VS3-REG-006`, `VS3-REG-007`, `VS3-REG-008`.

## Goal

Make `cornerstone scenario gate` reject repo-relative `evidence_refs` that use local path prefixes but do not resolve to an existing path under the repository.

Path prefixes covered:

- `docs/`
- `reports/`
- `scripts/`
- `config/`
- `policies/`
- `fixtures/`

Directories remain valid evidence refs because current VS3 reports use directory evidence refs for package/state directories.

## Non-Scope

This slice does not implement new RequestContext, RLS, OPA, egress, ConnectorHub, Tool SDK, operator UI, production/on-prem, live-provider, real-IdP, migration/restore, independent security review, or human UX acceptance behavior.

This slice does not convert any human-required row to `PASS`.

## Pre-Fix Evidence

The scenario gate accepted missing path-shaped evidence refs before this slice:

```text
PYTHONPATH=packages python3 - <<'PY'
# row-level probe:
# verify VS3 report, mutate VS3-GATE-001 evidence_refs to:
# ["reports/security/DOES_NOT_EXIST_VS3_EVIDENCE.json"]
# then run cornerstone scenario gate on the tampered report
PY

verify_exit 0
gate_exit 0
status success
row_ref_validation.status passed
row_ref_validation.malformed_evidence_ref_rows []
aggregate_ref_validation.status passed
errors []
```

```text
PYTHONPATH=packages python3 - <<'PY'
# aggregate probe:
# verify VS3 report, mutate aggregate evidence_refs to:
# ["reports/security/DOES_NOT_EXIST_VS3_AGGREGATE_EVIDENCE.json"]
# sync command transcript refs, then run cornerstone scenario gate
PY

verify_exit 0
gate_exit 0
status success
aggregate_ref_validation.status passed
aggregate_ref_validation.malformed_evidence_refs []
row_ref_validation.status passed
errors []
```

Current generated VS3 path refs were checked before the patch:

```text
verify_exit 0
path_ref_entries 322
unique_path_refs 38
missing_count 0
outside_count 0
```

## Implementation

Changed `packages/cornerstone_cli/main.py` so VS3 evidence refs now return a concrete validation issue:

- newline or carriage return: `unsupported_ref_format`;
- absolute path or `..` path segment: `path_outside_repo`;
- local path prefix that does not exist under repo root: `missing`;
- unknown non-path/non-command/non-symbolic ref: `unsupported_ref_format`.

Row-level validation records the issue in `row_ref_validation.malformed_evidence_ref_rows`.

Aggregate validation continues to expose invalid aggregate evidence refs in `aggregate_ref_validation.malformed_evidence_refs` to preserve the existing JSON shape.

## Focused Verification

Syntax:

```text
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
exit 0
```

Focused unit tests:

```text
PYTHONPATH=packages python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_preserves_source_proof_boundary_and_transcript \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_row_evidence_ref \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_missing_row_evidence_path \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_aggregate_evidence_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_missing_aggregate_evidence_path

.....
Ran 5 tests in 2.863s
OK
```

Post-fix row probe:

```text
verify_exit 0
gate_exit 4
status failed
error_codes ["CS_VS3_ROW_EVIDENCE_METADATA_MISSING"]
row_status failed
malformed_evidence_ref_rows [
  {
    "scenario_id": "VS3-GATE-001",
    "evidence_ref": "reports/security/DOES_NOT_EXIST_VS3_EVIDENCE.json",
    "issue": "missing"
  }
]
aggregate_status passed
```

Post-fix aggregate probe:

```text
verify_exit 0
gate_exit 4
status failed
error_codes ["CS_VS3_AGGREGATE_EVIDENCE_METADATA_MISSING"]
row_status passed
aggregate_status failed
malformed_evidence_refs ["reports/security/DOES_NOT_EXIST_VS3_AGGREGATE_EVIDENCE.json"]
```

Current VS3 report still gates successfully after the guard:

```text
PYTHONPATH=packages python3 -m cornerstone_cli.main scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
exit 0

PYTHONPATH=packages python3 -m cornerstone_cli.main scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json
exit 0
status success
row_ref_validation.status passed
aggregate_ref_validation.status passed
errors []
proof_boundary.vs3_p NOT_CLAIMED
proof_boundary.production NOT_CLAIMED
proof_boundary.real_idp NOT_CLAIMED
proof_boundary.live_provider NOT_CLAIMED
proof_boundary.human_acceptance_gate HUMAN_REQUIRED
proof_boundary.security_acceptance_gate HUMAN_REQUIRED
```

## Native VS3 Refresh Evidence

The following local deterministic refresh ran after the patch:

```text
PYTHONPATH=packages python3 -m cornerstone_cli.main security vs3-evidence-reconcile --json
exit 0 status success

PYTHONPATH=packages python3 -m cornerstone_cli.main security vs3-overclaim-lint --json
exit 0 status passed

PYTHONPATH=packages python3 -m cornerstone_cli.main security vs3-request-context --json
exit 0 status success

PYTHONPATH=packages python3 -m cornerstone_cli.main security vs3-postgres-rls --reuse-vs2-local-range-report reports/security/vs2-local-range.json --json
exit 0 status success

PYTHONPATH=packages python3 -m cornerstone_cli.main security vs3-opa-policy --reuse-vs2-local-range-report reports/security/vs2-local-range.json --json
exit 0 status success

PYTHONPATH=packages python3 -m cornerstone_cli.main security vs3-egress-sandbox --reuse-vs2-local-range-report reports/security/vs2-local-range.json --json
exit 0 status success

PYTHONPATH=packages python3 -m cornerstone_cli.main security vs3-connectorhub-source --json
exit 0 status success

PYTHONPATH=packages python3 -m cornerstone_cli.main security vs3-tool-registry --json
exit 0 status success

PYTHONPATH=packages python3 -m cornerstone_cli.main security vs3-observability --json
exit 0 status success

PYTHONPATH=packages python3 -m cornerstone_cli.main security vs3-regression-gate --json
exit 0 status success

PYTHONPATH=packages python3 -m cornerstone_cli.main scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
exit 0 status success

PYTHONPATH=packages python3 -m cornerstone_cli.main scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json
exit 0 status success

PYTHONPATH=packages python3 -m cornerstone_cli.main human-gate evidence-status --scope vs3 --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --use-existing --json --output reports/human-gates/vs3/evidence-status.json
exit 0 status success

PYTHONPATH=packages python3 -m cornerstone_cli.main human-gate review-kit --scope vs3 --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --use-existing --json --output reports/human-gates/vs3/review-kit.json
exit 0 status success

PYTHONPATH=packages python3 -m cornerstone_cli.main human-gate vs3-p-gate --scope vs3 --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json --output reports/human-gates/vs3/vs3-p-gate.json
exit 4 status blocked
error CS_VS3_P_GATE_HUMAN_EVIDENCE_REQUIRED

PYTHONPATH=packages python3 -m cornerstone_cli.main security vs3-local-checkpoint --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json --output reports/security/vs3-local-checkpoint.json
exit 0 status success
```

## Proof Boundary

This checkpoint proves only a local deterministic scenario-gate guard.

It does not claim:

- VS3-P;
- production/on-prem readiness;
- real IdP readiness;
- live-provider readiness;
- real-network readiness;
- migration/restore readiness;
- independent security acceptance;
- human operator acceptance.

`human-gate vs3-p-gate` correctly remains blocked until `VS3-H01` through `VS3-H07` have dated signed human/on-prem evidence and a separate owner promotion decision.

## Decision

The current slice can be treated as locally verified for `VS3-GATE-004` and `VS3-REG-004` guard coverage:

- missing row-level path-shaped `evidence_refs` now fail the gate;
- missing aggregate path-shaped `evidence_refs` now fail the gate;
- unsupported symbolic/string refs still fail as before;
- current generated VS3 evidence refs still pass after path existence validation;
- VS3-P and all human/on-prem claims remain unclaimed or blocked.
