# VS3 Scenario Gate Aggregate Source Report Evidence Overlap Guard Checkpoint

**Date:** 2026-06-30 KST
**Scope:** VS3 native scenario gate hardening slice.
**Status:** Local deterministic verifier guard implemented and checked.

## Slice Contract

Goal:

- Harden `cornerstone scenario gate <vs3-report> --json` so a local/dev VS3 assurance report cannot omit any row-level `source_report_ref` from aggregate `evidence_refs`, even if row evidence and command transcript metadata are internally consistent.

In scope:

- Native VS3 scenario gate aggregate-reference validation.
- Regression test for a coherent aggregate/source-report lineage mutation.
- Local deterministic CLI verification and proof-boundary check.

Out of scope:

- New VS3 feature substrate implementation.
- Production/on-prem readiness.
- Real IdP, real network, live provider, independent security review, migration/restore, or human UX acceptance.
- Conversion of `VS3-H01` through `VS3-H07` to `PASS`.

## Full Scenario Mapping

Current slice:

| Scenario | Type | Classification | Required proof surface |
|---|---|---|---|
| `VS3-GATE-004` | MUST_PASS | `in_this_slice` | Native `cornerstone scenario verify ... --json` and `cornerstone scenario gate ... --json` emit aggregate evidence refs, row source-report refs, gate metadata, and stable failure for invalid lineage. |
| `VS3-REG-004` | REGRESSION | `in_this_slice` | Evidence lineage mutation must fail before a release or local/dev assurance claim can pass. |

Mapped later slices:

| Range | Classification | Reason |
|---|---|---|
| `VS3-GATE-001` through `VS3-GATE-003` | `later_slice` | Already mapped; this slice only hardens aggregate source-report lineage in the native gate. |
| `VS3-CTX-001` through `VS3-CTX-005` | `later_slice` | RequestContext behavior remains represented by source report evidence; this slice hardens aggregate evidence linkage to that proof. |
| `VS3-RLS-001` through `VS3-RLS-006` | `later_slice` | Postgres/RLS behavior is not changed by this verifier guard. |
| `VS3-OPA-001` through `VS3-OPA-005` | `later_slice` | OPA/Rego behavior is not changed by this verifier guard. |
| `VS3-EGR-001` through `VS3-EGR-006` | `later_slice` | Egress/sandbox behavior is not changed by this verifier guard. |
| `VS3-CON-001` through `VS3-CON-006` | `later_slice` | ConnectorHub/source behavior is not changed by this verifier guard. |
| `VS3-TOOL-001` through `VS3-TOOL-007` | `later_slice` | Tool SDK/registry behavior is not changed by this verifier guard. |
| `VS3-OBS-001` through `VS3-OBS-003` | `later_slice` | Operator/audit/human-gate package behavior is not changed by this verifier guard. |
| `VS3-REG-001` through `VS3-REG-003`, `VS3-REG-005` through `VS3-REG-008` | `later_slice` | Regression rows remain mapped but outside this one-slice aggregate-lineage guard. |
| `VS3-H01` through `VS3-H07` | `HUMAN_REQUIRED` | Require dated, redacted, signed human/external/on-prem evidence before VS3-P or corresponding readiness claims. |

## Baseline Gap

Before the patch, this coherent mutation incorrectly passed:

1. Generate `cornerstone scenario verify vs3-onprem-trusted-extension --json`.
2. Remove `reports/security/vs3-request-context-proof.json` from top-level `evidence_refs`.
3. Update `command_transcripts[*].evidence_refs`, `self_command_transcript.evidence_refs`, `stdout_json.evidence_refs`, and `ref_summary.evidence_refs_count` to match the new aggregate list.
4. Leave `VS3-CTX-001` through `VS3-CTX-005` row-level `source_report_refs=["reports/security/vs3-request-context-proof.json"]`.
5. Run `cornerstone scenario gate <tampered-report> --json`.

Observed pre-fix result:

```text
row_source_missing_from_top_count=5
gate_exit=0
gate_status=success
aggregate_ref_validation.status=passed
source_transcript_validation.status=passed
row_ref_validation.status=passed
errors=[]
```

This allowed aggregate report evidence to omit a proof report that five row-level PASS claims still depended on.

## Implementation

Changed:

- `packages/cornerstone_cli/main.py`
  - Added `aggregate_source_report_ref_missing_from_evidence_refs`.
  - Fails aggregate reference validation when any row-level `source_report_ref` is absent from top-level `evidence_refs`.
  - Includes `source_report_ref_missing_from_evidence_refs` in `aggregate_ref_validation`, `source_report.aggregate_ref_validation`, and the `CS_VS3_AGGREGATE_EVIDENCE_METADATA_MISSING` error payload.

- `tests/scenario/test_scaffold_cli.py`
  - Added `test_vs3_scenario_gate_rejects_local_dev_claim_with_row_source_report_ref_missing_from_aggregate_evidence_refs`.
  - Extended the successful VS3 gate test to assert the new aggregate lineage list is empty.

## Verification

Focused checks:

```text
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
PASS

PYTHONPATH=packages python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_row_source_report_ref_missing_from_aggregate_evidence_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_source_report_ref_missing_from_evidence_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_aggregate_evidence_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_missing_aggregate_evidence_path
Ran 4 tests in 2.267s
OK
```

Tooling note:

```text
pytest
exit=127 command not found

python3 -m pytest
exit=1 No module named pytest
```

Post-fix mutation probe:

```text
seed_exit=0
row_source_missing_from_top_count=5
sample=[
  ("VS3-CTX-001", "reports/security/vs3-request-context-proof.json"),
  ("VS3-CTX-002", "reports/security/vs3-request-context-proof.json"),
  ("VS3-CTX-003", "reports/security/vs3-request-context-proof.json"),
  ("VS3-CTX-004", "reports/security/vs3-request-context-proof.json"),
  ("VS3-CTX-005", "reports/security/vs3-request-context-proof.json")
]
gate_exit=4
gate_status=failed
aggregate_ref_validation.status=failed
source_transcript_validation.status=passed
row_ref_validation.status=passed
errors=["CS_VS3_AGGREGATE_EVIDENCE_METADATA_MISSING"]
```

The failure payload included:

```json
[
  {
    "scenario_id": "VS3-CTX-001",
    "source_report_ref": "reports/security/vs3-request-context-proof.json",
    "issue": "source_report_ref_missing_from_aggregate_evidence_refs"
  },
  {
    "scenario_id": "VS3-CTX-002",
    "source_report_ref": "reports/security/vs3-request-context-proof.json",
    "issue": "source_report_ref_missing_from_aggregate_evidence_refs"
  },
  {
    "scenario_id": "VS3-CTX-003",
    "source_report_ref": "reports/security/vs3-request-context-proof.json",
    "issue": "source_report_ref_missing_from_aggregate_evidence_refs"
  },
  {
    "scenario_id": "VS3-CTX-004",
    "source_report_ref": "reports/security/vs3-request-context-proof.json",
    "issue": "source_report_ref_missing_from_aggregate_evidence_refs"
  },
  {
    "scenario_id": "VS3-CTX-005",
    "source_report_ref": "reports/security/vs3-request-context-proof.json",
    "issue": "source_report_ref_missing_from_aggregate_evidence_refs"
  }
]
```

Post-patch clean baseline:

```text
cornerstone scenario verify vs3-onprem-trusted-extension --json --output tmp/vs3-post-patch-baseline.json
exit=0

cornerstone scenario gate tmp/vs3-post-patch-baseline.json --json
exit=0
status=success
aggregate_ref_validation.status=passed
aggregate_ref_validation.source_report_ref_missing_from_evidence_refs=[]
source_tree_current_validation.status=passed
summary.pass=50
summary.human_required=7
claim_boundaries.vs3_l=LOCAL_DEV_ASSURANCE_VERIFIED
claim_boundaries.vs3_p=NOT_CLAIMED
```

Native VS3 refresh after this checkpoint file existed:

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
exit=0 status=success pass=50 human_required=7 blocking=0 scenario_count=57 vs3_l=LOCAL_DEV_ASSURANCE_VERIFIED vs3_p=NOT_CLAIMED

cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json
exit=0 status=success pass=50 human_required=7 blocking=0 scenario_count=57 aggregate_ref_validation=passed missing_source_refs=0 source_tree_current_validation=passed vs3_l=LOCAL_DEV_ASSURANCE_VERIFIED vs3_p=NOT_CLAIMED

cornerstone human-gate evidence-status --scope vs3 --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --use-existing --json --output reports/human-gates/vs3/evidence-status.json
exit=0 status=success final_verdict=HUMAN_REQUIRED pass=50 human_required=7

cornerstone human-gate review-kit --scope vs3 --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --use-existing --json --output reports/human-gates/vs3/review-kit.json
exit=0 status=success final_verdict=HUMAN_REQUIRED pass=50 human_required=7

cornerstone human-gate vs3-p-gate --scope vs3 --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json --output reports/human-gates/vs3/vs3-p-gate.json
exit=4 status=blocked final_verdict=HUMAN_REQUIRED error=CS_VS3_P_GATE_HUMAN_EVIDENCE_REQUIRED pass=50 human_required=7

cornerstone security vs3-local-checkpoint --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json --output reports/security/vs3-local-checkpoint.json
exit=0 status=success final_verdict=VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED pass=50 human_required=7 blocking=0 scenario_count=57
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
