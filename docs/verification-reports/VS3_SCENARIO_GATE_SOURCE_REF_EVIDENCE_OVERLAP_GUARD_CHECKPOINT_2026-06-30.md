# VS3 Scenario Gate Source Report Evidence Overlap Guard Checkpoint

**Date:** 2026-06-30 KST
**Scope:** VS3 native scenario gate hardening slice.
**Status:** Local deterministic verifier guard implemented and checked.

## Slice Contract

Goal:

- Harden `cornerstone scenario gate <vs3-report> --json` so a local/dev VS3 assurance report cannot claim a row-level `source_report_ref` unless the same proof report is also present in that row's `evidence_refs`.

In scope:

- Native VS3 scenario gate row-reference validation.
- Regression test for the source-report/evidence overlap mutation.
- Local deterministic CLI verification and human-gate boundary check.

Out of scope:

- New VS3 feature substrate implementation.
- Production/on-prem readiness.
- Real IdP, real network, live provider, independent security review, migration/restore, or human UX acceptance.
- Conversion of `VS3-H01` through `VS3-H07` to `PASS`.

## Full Scenario Mapping

Current slice:

| Scenario | Type | Classification | Required proof surface |
|---|---|---|---|
| `VS3-GATE-004` | MUST_PASS | `in_this_slice` | Native `cornerstone scenario verify ... --json` and `cornerstone scenario gate ... --json` emit row evidence, source reports, gate metadata, and stable failure for invalid evidence. |
| `VS3-REG-004` | REGRESSION | `in_this_slice` | Coverage/evidence mutation must fail before a release or local-dev assurance claim can pass. |

Mapped later slices:

| Range | Classification | Reason |
|---|---|---|
| `VS3-GATE-001` through `VS3-GATE-003` | `later_slice` | Already mapped; not changed by this row-overlap guard slice. |
| `VS3-CTX-001` through `VS3-CTX-005` | `later_slice` | RequestContext behavior remains represented by source report evidence; this slice only hardens the gate around that evidence. |
| `VS3-RLS-001` through `VS3-RLS-006` | `later_slice` | Postgres/RLS behavior is not changed by this verifier guard. |
| `VS3-OPA-001` through `VS3-OPA-005` | `later_slice` | OPA/Rego behavior is not changed by this verifier guard. |
| `VS3-EGR-001` through `VS3-EGR-006` | `later_slice` | Egress/sandbox behavior is not changed by this verifier guard. |
| `VS3-CON-001` through `VS3-CON-006` | `later_slice` | ConnectorHub/source behavior is not changed by this verifier guard. |
| `VS3-TOOL-001` through `VS3-TOOL-007` | `later_slice` | Tool SDK/registry behavior is not changed by this verifier guard. |
| `VS3-OBS-001` through `VS3-OBS-003` | `later_slice` | Operator/audit/human-gate package behavior is not changed by this verifier guard. |
| `VS3-REG-001` through `VS3-REG-003`, `VS3-REG-005` through `VS3-REG-008` | `later_slice` | Regression rows remain mapped but outside this one-slice verifier guard. |
| `VS3-H01` through `VS3-H07` | `HUMAN_REQUIRED` | Require dated, redacted, signed human/external/on-prem evidence before VS3-P or corresponding readiness claims. |

## Baseline Gap

Before the patch, this mutation incorrectly passed:

1. Generate `cornerstone scenario verify vs3-onprem-trusted-extension --json`.
2. In row `VS3-CTX-001`, remove `reports/security/vs3-request-context-proof.json` from `evidence_refs`.
3. Leave `source_report_refs=["reports/security/vs3-request-context-proof.json"]`.
4. Run `cornerstone scenario gate <tampered-report> --json`.

Observed pre-fix result:

```text
gate_exit=0
gate_status=success
row_ref_validation=passed
```

This allowed a row to claim a source proof report without carrying that proof in the row evidence list.

## Implementation

Changed:

- `packages/cornerstone_cli/main.py`
  - Added `source_report_ref_not_in_evidence_rows`.
  - Fails row reference validation when any AI-owned `PASS` row lists a `source_report_ref` that is absent from the same row's `evidence_refs`.
  - Includes the new field in `row_ref_validation`, `source_report.row_ref_validation`, and the `CS_VS3_ROW_EVIDENCE_METADATA_MISSING` error payload.

- `tests/scenario/test_scaffold_cli.py`
  - Added `test_vs3_scenario_gate_rejects_local_dev_claim_with_source_report_ref_missing_from_evidence_refs`.

## Verification

Focused checks:

```text
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
PASS

PYTHONPATH=packages python3 -m unittest \
 tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_source_report_ref_missing_from_evidence_refs \
 tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_unexpected_row_source_report_ref \
 tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_unresolved_row_source_report_ref \
 tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_missing_row_evidence_path \
 tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_row_evidence_ref
Ran 5 tests in 2.892s
OK
```

Post-fix mutation probe:

```text
removed_ref=reports/security/vs3-request-context-proof.json
remaining_evidence_refs=["cornerstone principal context resolve --json"]
gate_exit=4
gate_status=failed
errors=["CS_VS3_ROW_EVIDENCE_METADATA_MISSING"]
row_ref_validation=failed
source_report_ref_not_in_evidence_rows=[{"scenario_id":"VS3-CTX-001","source_report_ref":"reports/security/vs3-request-context-proof.json","issue":"source_report_ref_not_in_evidence_refs"}]
```

Native VS3 refresh:

```text
cornerstone security vs3-evidence-reconcile --json
exit=0 status=success

cornerstone security vs3-overclaim-lint --json
exit=0 status=passed

cornerstone security vs3-request-context --json
exit=0 status=success

cornerstone security vs3-postgres-rls --reuse-vs2-local-range-report reports/security/vs2-local-range.json --json
exit=0 status=success

cornerstone security vs3-opa-policy --reuse-vs2-local-range-report reports/security/vs2-local-range.json --json
exit=0 status=success

cornerstone security vs3-egress-sandbox --reuse-vs2-local-range-report reports/security/vs2-local-range.json --json
exit=0 status=success

cornerstone security vs3-connectorhub-source --json
exit=0 status=success

cornerstone security vs3-tool-registry --json
exit=0 status=success

cornerstone security vs3-observability --json
exit=0 status=success

cornerstone security vs3-regression-gate --json
exit=0 status=success

cornerstone scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
exit=0 status=success pass=50 human_required=7 blocking=0 vs3_l=LOCAL_DEV_ASSURANCE_VERIFIED vs3_p=NOT_CLAIMED

cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json
exit=0 status=success row_ref_validation=passed aggregate_ref_validation=passed source_tree_current_validation=passed

cornerstone human-gate evidence-status --scope vs3 --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --use-existing --json --output reports/human-gates/vs3/evidence-status.json
exit=0 status=success final_verdict=HUMAN_REQUIRED human_required=7

cornerstone human-gate review-kit --scope vs3 --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --use-existing --json --output reports/human-gates/vs3/review-kit.json
exit=0 status=success final_verdict=HUMAN_REQUIRED human_required=7

cornerstone human-gate vs3-p-gate --scope vs3 --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json --output reports/human-gates/vs3/vs3-p-gate.json
exit=4 status=blocked error=CS_VS3_P_GATE_HUMAN_EVIDENCE_REQUIRED human_required=7 vs3_p_ready=false

cornerstone security vs3-local-checkpoint --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json --output reports/security/vs3-local-checkpoint.json
exit=0 status=success ai_blocking_rows=0 human_required=7 vs3_l=LOCAL_DEV_ASSURANCE_VERIFIED vs3_p=NOT_CLAIMED
```

After this checkpoint file was added, an immediate local-checkpoint run detected stale component proof reports from the prior source tree:

```text
cornerstone security vs3-local-checkpoint --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json --output reports/security/vs3-local-checkpoint.json
exit=4 status=failed component_proof_report_source_tree_current_failures=7 component_proof_report_semantic_failures=7
```

The full VS3 component proof refresh was rerun after the checkpoint document existed. The final local checkpoint returned:

```text
cornerstone security vs3-local-checkpoint --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json --output reports/security/vs3-local-checkpoint.json
exit=0 status=success component_proof_report_source_tree_current_failures=0 component_proof_report_semantic_failures=0 ai_blocking_rows=0 human_required=7 vs3_l=LOCAL_DEV_ASSURANCE_VERIFIED vs3_p=NOT_CLAIMED
```

## Proof Boundary

This checkpoint supports only the local deterministic VS3 verifier guard claim for `VS3-GATE-004` and `VS3-REG-004`.

It does not claim:

- `VS3-P`;
- production/on-prem readiness;
- real IdP readiness;
- real network readiness;
- live provider readiness;
- independent security acceptance;
- migration/restore readiness;
- human UX acceptance.

`VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`.

## Decision

Continue VS3 in small verified slices. The next slice should keep strengthening native gate/evidence semantics or move to the next highest-risk local/dev substrate gap without widening into production or human-gated claims.
