# VS3 Human Gate Evidence Intake Checkpoint

**Date:** 2026-06-29
**Status:** Local deterministic human-gate evidence intake hardening verified.
**Verdict:** `HUMAN_REQUIRED` for human gates; `VS3-P` remains `NOT_CLAIMED`.

This checkpoint covers the VS3 human evidence intake path only. It proves that local record validation and evidence-status reporting can inspect proposed review records without collecting approval, recording a human decision, unlocking dependencies, marking any `VS3-H*` row `PASS`, or claiming VS3-P.

## Slice Contract

Goal:
- Strengthen the VS3 human-gate evidence intake path so overclaiming proposed records are visible at the directory-status layer, not only in single-record and batch validation.

Scope:
- `cornerstone human-gate evidence-status --scope vs3 --json`
- `cornerstone human-gate validate-record --scope vs3 --json`
- `cornerstone human-gate validate-records --scope vs3 --json`
- VS3-H01 through VS3-H07 record-status classification.
- Negative evidence for overclaim markers, sensitive markers, VS3-P unlock attempts, record-body persistence, and record-path persistence.

Non-scope:
- Accepting any human review record.
- Promoting any `VS3-H*` row from `HUMAN_REQUIRED` to `PASS`.
- VS3-P approval.
- Production/on-prem, real IdP, real network, live provider, migration/restore, security acceptance, or human UX acceptance.

Done criteria:
- Evidence-status reports `overclaim_marker_findings` in its row-level, summary, and negative-evidence outputs.
- An overclaiming proposed `VS3-H03` record is classified as `structurally_invalid`.
- The same overclaiming record leaves `final_verdict=HUMAN_REQUIRED`, `vs3_p_unlock_allowed=false`, and `vs3_p_claim=NOT_CLAIMED`.
- Single-record and batch validators still pass their no-acceptance tests.
- Aggregate VS3 scenario verification and VS3 local checkpoint remain green for VS3-L only.

## Full Scenario Mapping

All 57 frozen VS3 rows remain carried by the aggregate report. This slice directly hardens `VS3-OBS-003` and the human-gate preparation surface, while preserving the full local/dev checkpoint.

| Scenario IDs | Type | Classification | Proof Surface |
| --- | --- | --- | --- |
| `VS3-GATE-001` through `VS3-OBS-003` | MUST_PASS | in_this_slice through aggregate verifier | `cornerstone scenario verify vs3-onprem-trusted-extension --json`, local proof reports, and checkpoint manifest. |
| `VS3-REG-001` through `VS3-REG-008` | REGRESSION | in_this_slice through aggregate verifier | Fresh local regression reports and overclaim/default-deny evidence in the aggregate report. |
| `VS3-H01` through `VS3-H07` | HUMAN_REQUIRED | in_this_slice as preparation only | Record scaffold, evidence-status, validate-record, validate-records, VS3-P gate, and local checkpoint. Human acceptance remains out of scope. |

## Implementation Change

- `packages/cornerstone_cli/main.py` now aggregates `overclaim_marker_findings` in `human-gate evidence-status`, exposes it in each record status row, includes it in top-level `summary`, and includes it in `negative_evidence`.
- `tests/scenario/test_scaffold_cli.py` now mutates a proposed `VS3-H03` record with `final_verdict=PASS`, `real_idp_readiness_claim_allowed=true`, and `vs3_p=READY`; evidence-status must report three overclaim markers while keeping VS3-P locked.

## Evidence

Targeted automated checks:

```text
python3 -m compileall packages/cornerstone_cli
exit_code=0

python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_human_gate_evidence_status_reports_records_without_acceptance
exit_code=0

python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_human_gate_validate_record_is_structural_and_redacted \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_human_gate_validate_records_batches_without_acceptance
exit_code=0
```

Native CLI evidence-status check:

```text
PATH="$PWD:$PATH" cornerstone human-gate evidence-status \
  --scope vs3 \
  --record-dir reports/human-gates/vs3/record-templates \
  --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json \
  --output reports/human-gates/vs3/evidence-status.json \
  --use-existing \
  --json
exit_code=0
status=success
final_verdict=HUMAN_REQUIRED
summary.scenario_count=57
summary.pass=50
summary.human_required=7
summary.overclaim_marker_findings=0
summary.vs3_l_claim=LOCAL_DEV_ASSURANCE_VERIFIED
summary.vs3_p_claim=NOT_CLAIMED
negative_evidence.overclaim_marker_findings=0
negative_evidence.sensitive_marker_findings=0
negative_evidence.vs3_p_unlocked_by_evidence_status=0
negative_evidence.structural_validation_treated_as_acceptance=0
```

Aggregate and checkpoint checks after the patch:

```text
PATH="$PWD:$PATH" cornerstone scenario verify vs3-onprem-trusted-extension \
  --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json \
  --json
exit_code=0
status=success
final_verdict=VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
summary.pass=50
summary.human_required=7
summary.blocking=0

PATH="$PWD:$PATH" cornerstone human-gate vs3-p-gate \
  --scope vs3 \
  --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json \
  --output reports/human-gates/vs3/vs3-p-gate.json \
  --json
exit_code=4
status=blocked
final_verdict=HUMAN_REQUIRED
summary.unresolved_human_required_rows=7
summary.vs3_p_claim=NOT_CLAIMED

PATH="$PWD:$PATH" cornerstone security vs3-local-checkpoint \
  --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json \
  --output reports/human-gates/vs3/vs3-local-checkpoint.json \
  --json
exit_code=0
status=success
final_verdict=VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
negative_evidence.vs3_p_gate_scenario_report_hash_mismatches=0
negative_evidence.human_gate_evidence_status_unlocked_vs3_p=0
negative_evidence.overclaim_boundary_violations=0
negative_evidence.vs3_p_claimed_by_checkpoint=0
```

## Remaining Human Gates

| Scenario | Current Status | Required Proof |
| --- | --- | --- |
| `VS3-H01` | HUMAN_REQUIRED | Owner architecture/security approval. |
| `VS3-H02` | HUMAN_REQUIRED | Independent security review and retest. |
| `VS3-H03` | HUMAN_REQUIRED | Real IdP mapping and revocation evidence. |
| `VS3-H04` | HUMAN_REQUIRED | Real on-prem network/security evidence. |
| `VS3-H05` | HUMAN_REQUIRED | Approved live ConnectorHub/provider rehearsal. |
| `VS3-H06` | HUMAN_REQUIRED | Human operator UX/trust review. |
| `VS3-H07` | HUMAN_REQUIRED | Human-supervised migration/backup/restore/rollback drill. |

## Decision

The local human-gate intake surface is stronger: directory-level evidence-status now reports overclaim markers explicitly and still refuses to unlock VS3-P. This is preparation for human review only, not human approval.
