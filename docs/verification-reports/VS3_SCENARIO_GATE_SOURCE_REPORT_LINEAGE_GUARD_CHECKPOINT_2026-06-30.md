# VS3 Scenario Gate Source Report Lineage Guard Checkpoint

**Date:** 2026-06-30 KST
**Status:** Local deterministic checkpoint for one VS3 scenario-gate guard slice.
**Slice:** Row-level `source_report_refs` lineage validation in `cornerstone scenario gate`.

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

Make `cornerstone scenario gate` reject AI PASS rows whose `source_report_refs` point to existing files but do not match the expected VS3 source artifact for that scenario ID.

This closes the gap where source proof could be replaced by an unrelated existing file while still passing path-existence validation.

## Expected Source Mapping

The scenario gate now validates AI PASS row lineage against this local deterministic mapping:

| Row family | Expected source report ref |
|---|---|
| `VS3-GATE-001` | `reports/scenario/vs2-policy-tenancy-egress-2026-06-19.json` |
| `VS3-GATE-002` | `docs/scenario-contracts/VS3_ONPREM_SECURITY_AND_TRUSTED_EXTENSION_CONTRACT.md` |
| `VS3-GATE-003`, `VS3-GATE-004`, `VS3-REG-*` | `reports/security/vs3-final-regression-proof.json` |
| `VS3-CTX-*` | `reports/security/vs3-request-context-proof.json` |
| `VS3-RLS-*` | `reports/db/vs3-postgres-rls-proof.json` |
| `VS3-OPA-*` | `reports/policy/vs3-opa-policy-proof.json` |
| `VS3-EGR-*` | `reports/security/vs3-egress-sandbox-proof.json` |
| `VS3-CON-*` | `reports/security/vs3-connectorhub-source-proof.json` |
| `VS3-TOOL-*` | `reports/security/vs3-tool-registry-proof.json` |
| `VS3-OBS-*` | `reports/observability/vs3-observability-proof.json` |

Human rows remain `HUMAN_REQUIRED`; this guard does not convert human-required package templates into acceptance evidence.

## Non-Scope

This slice does not implement new RequestContext, RLS, OPA, egress, ConnectorHub, Tool SDK, operator UI, production/on-prem, live-provider, real-IdP, migration/restore, independent security review, or human UX acceptance behavior.

This slice does not claim VS3-P.

## Pre-Fix Evidence

The source-report path guard already rejected missing files, but accepted an existing unrelated file as source proof.

```text
PYTHONPATH=packages python3 - <<'PY'
# verify VS3 report, mutate VS3-CTX-001:
# source_report_refs = ["docs/sot/README.md"]
# append the same ref to evidence_refs so the evidence-ref guard is not the failing layer
# then run cornerstone scenario gate on the tampered report
PY

verify_exit 0
gate_exit 0
status success
row_ref_validation.status passed
row_ref_validation.unresolved_source_report_ref_rows []
errors []
```

Current generated VS3 report shape before the patch:

```text
source_refs_not_subset_count 0
VS3-CTX-001 source_report_refs ["reports/security/vs3-request-context-proof.json"]
VS3-GATE-002 source_report_refs ["docs/scenario-contracts/VS3_ONPREM_SECURITY_AND_TRUSTED_EXTENSION_CONTRACT.md"]
```

## Implementation

Changed `packages/cornerstone_cli/main.py` so VS3 AI PASS rows now validate source-report lineage:

- existing source refs that are not expected for the row are reported in `row_ref_validation.unexpected_source_report_ref_rows`;
- expected source refs missing from the row are reported in `row_ref_validation.missing_expected_source_report_ref_rows`;
- path existence and file checks remain in `row_ref_validation.unresolved_source_report_ref_rows`;
- aggregate ref JSON shape is unchanged.

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
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_unresolved_row_source_report_ref \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_unexpected_row_source_report_ref \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_row_evidence_ref \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_missing_row_evidence_path \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_aggregate_evidence_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_missing_aggregate_evidence_path

.......
Ran 7 tests in 3.916s
OK
```

Post-fix lineage probe:

```text
verify_exit 0
gate_exit 4
status failed
error_codes ["CS_VS3_ROW_EVIDENCE_METADATA_MISSING"]
row_status failed
unexpected_source_report_ref_rows [
  {
    "scenario_id": "VS3-CTX-001",
    "source_report_ref": "docs/sot/README.md",
    "expected_source_report_refs": ["reports/security/vs3-request-context-proof.json"],
    "issue": "unexpected_source_report_ref"
  }
]
missing_expected_source_report_ref_rows [
  {
    "scenario_id": "VS3-CTX-001",
    "missing_source_report_refs": ["reports/security/vs3-request-context-proof.json"],
    "issue": "missing_expected_source_report_ref"
  }
]
aggregate_status passed
```

Current VS3 report still gates successfully with the lineage guard:

```text
PYTHONPATH=packages python3 -m cornerstone_cli.main scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
exit 0

PYTHONPATH=packages python3 -m cornerstone_cli.main scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json
exit 0
status success
row_ref_validation.status passed
unexpected_source_report_ref_rows []
missing_expected_source_report_ref_rows []
aggregate_ref_validation.status passed
proof_boundary.vs3_l LOCAL_DEV_ASSURANCE_VERIFIED
proof_boundary.vs3_p NOT_CLAIMED
proof_boundary.production NOT_CLAIMED
```

## Proof Boundary

This checkpoint proves only a local deterministic scenario-gate lineage guard.

It does not claim:

- VS3-P;
- production/on-prem readiness;
- real IdP readiness;
- live-provider readiness;
- real-network readiness;
- migration/restore readiness;
- independent security acceptance;
- human operator acceptance.

## Decision

The current slice can be treated as locally verified for `VS3-GATE-004` and `VS3-REG-004` guard coverage:

- AI PASS rows can no longer swap `source_report_refs` to unrelated existing files;
- missing expected source refs fail the gate;
- the current generated VS3 report still passes;
- VS3-P and all human/on-prem claims remain unclaimed or blocked.
