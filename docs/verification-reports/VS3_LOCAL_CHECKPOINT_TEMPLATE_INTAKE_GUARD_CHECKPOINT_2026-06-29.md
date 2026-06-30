# VS3 Local Checkpoint Template Intake Guard Checkpoint

**Date:** 2026-06-29
**Status:** Local deterministic checkpoint guard verified.
**Verdict:** `VS3-L` local/dev checkpoint remains verified; `VS3-P` remains `HUMAN_REQUIRED` and `NOT_CLAIMED`.

This checkpoint covers the local checkpoint consumer for VS3 human-gate evidence-status data. It makes `template_intake_summary` an enforced checkpoint invariant so a local checkpoint cannot pass if the human-gate evidence-status report loses the blank-template, filled-record, acceptance-candidate, or no-acceptance boundary semantics.

## Slice Contract

Goal:
- Make `cornerstone security vs3-local-checkpoint --json` reject missing or weakened human-gate template-intake semantics from `reports/human-gates/vs3/evidence-status.json`.

Scope:
- `cornerstone security vs3-local-checkpoint --json`
- `reports/human-gates/vs3/evidence-status.json`
- Human-gate template-intake counters and acceptance-boundary checks.
- Targeted scenario tests for the local checkpoint.

Non-scope:
- Accepting human evidence.
- Promoting `VS3-H01` through `VS3-H07` from `HUMAN_REQUIRED` to `PASS`.
- Unlocking or claiming `VS3-P`.
- Production/on-prem, real IdP, real network, live-provider, migration/restore, independent security, or human UX acceptance claims.
- Starting VS3-1 RequestContext/runtime implementation.

Done criteria:
- Local checkpoint requires `template_intake_summary` to exist.
- Local checkpoint requires `template_readiness_status=awaiting_human_completion`.
- Local checkpoint requires `blank_template_pending_count=7`, `filled_record_count=0`, `invalid_filled_record_count=0`, and `evidence_acceptance_candidate_count=0` across summary, negative evidence, and nested intake summary.
- Local checkpoint requires the nested acceptance boundary to state that structural validity does not approve evidence, unlock dependencies, or move VS3-H rows out of `HUMAN_REQUIRED`.
- A tampered template-intake summary causes `cornerstone security vs3-local-checkpoint --json` to exit 4 and keep `VS3-L` unclaimed.

## Full Scenario Mapping

The full frozen VS3 matrix remains mapped: 57 rows total, with 42 `MUST_PASS`, 8 `REGRESSION`, and 7 `HUMAN_REQUIRED` rows.

| Scenario IDs | Type | Classification | Proof Surface |
| --- | --- | --- | --- |
| `VS3-GATE-001` through `VS3-OBS-002` | MUST_PASS | later/current aggregate context | Existing aggregate verifier and local proof reports remain the carried VS3-L evidence surface. |
| `VS3-OBS-003` | MUST_PASS | in_this_slice | Human-gate package/status generation plus local checkpoint template-intake guard. |
| `VS3-REG-001` through `VS3-REG-008` | REGRESSION | carried through aggregate verifier | Existing local regression proof remains part of the aggregate VS3 report; no regression scope was widened in this slice. |
| `VS3-H01` through `VS3-H07` | HUMAN_REQUIRED | in_this_slice as guard/preparation only | Evidence-status, review-kit, VS3-P gate, and local checkpoint prove preparation only; signed human evidence remains required. |

## Implementation Change

- `packages/cornerstone_cli/main.py` now enforces seven `human_gate_evidence_status_template_*` checkpoint conditions.
- `packages/cornerstone_cli/main.py` now emits three new negative-evidence counters:
  - `human_gate_evidence_status_template_intake_missing_or_invalid`
  - `human_gate_evidence_status_template_intake_count_mismatches`
  - `human_gate_evidence_status_template_acceptance_boundary_mismatches`
- `packages/cornerstone_cli/main.py` now copies `template_intake_summary` into `human_gate_preparation.evidence_status_report` inside the local checkpoint payload.
- `tests/scenario/test_scaffold_cli.py` now verifies the happy-path checkpoint fields and rejects a tampered template-intake summary.

## Evidence

Source/matrix inspection:

```text
docs/scenario-contracts/VS3_ONPREM_SECURITY_AND_TRUSTED_EXTENSION_MATRIX.csv
total=57
MUST_PASS=42
REGRESSION=8
HUMAN_REQUIRED=7
```

Automated checks:

```text
python3 -m compileall packages/cornerstone_cli
exit_code=0

python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_manifest_is_hash_backed_and_boundary_safe \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_tampered_evidence_status_template_intake_summary
exit_code=0
Ran 2 tests in 59.940s

python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_human_gate_evidence_status_reports_records_without_acceptance \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_tampered_evidence_status_human_rows
exit_code=0
Ran 2 tests in 50.044s
```

Native CLI regenerated evidence:

```text
PATH="$PWD:$PATH" cornerstone human-gate evidence-status \
  --scope vs3 \
  --record-dir reports/human-gates/vs3/record-templates \
  --use-existing \
  --json \
  --output reports/human-gates/vs3/evidence-status.json
exit_code=0
status=success
final_verdict=HUMAN_REQUIRED
summary.scenario_count=57
summary.pass=50
summary.human_required=7
summary.blocking=0
summary.blank_template_pending_count=7
summary.filled_record_count=0
summary.invalid_filled_record_count=0
summary.evidence_acceptance_candidate_count=0
summary.vs3_l_claim=LOCAL_DEV_ASSURANCE_VERIFIED
summary.vs3_p_claim=NOT_CLAIMED
sha256=ea5e97636b5890a8465298c3af3043fc25604b49f6698f26347be58f4dac3729

PATH="$PWD:$PATH" cornerstone human-gate vs3-p-gate \
  --scope vs3 \
  --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json \
  --output reports/human-gates/vs3/vs3-p-gate.json \
  --json
exit_code=4
status=blocked
final_verdict=HUMAN_REQUIRED
summary.unresolved_human_required_rows=7
summary.vs3_p_ready=false
summary.vs3_p_claim=NOT_CLAIMED
sha256=a1430ba9cf45feeedb9721fd3d83279d137bb71153755d5406df5cca36c50e70

PATH="$PWD:$PATH" cornerstone security vs3-local-checkpoint \
  --json \
  --output reports/human-gates/vs3/vs3-local-checkpoint.json
exit_code=0
status=success
final_verdict=VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
summary.scenario_count=57
summary.pass=50
summary.human_required=7
summary.blocking=0
summary.missing_required_artifacts=0
summary.ai_blocking_rows=0
summary.unresolved_human_required_rows=7
summary.vs3_l_claim=LOCAL_DEV_ASSURANCE_VERIFIED
summary.vs3_p_claim=NOT_CLAIMED
checkpoint_conditions.human_gate_evidence_status_template_intake_summary_present=true
checkpoint_conditions.human_gate_evidence_status_template_readiness_awaiting_completion=true
checkpoint_conditions.human_gate_evidence_status_blank_template_pending_count=true
checkpoint_conditions.human_gate_evidence_status_no_filled_records=true
checkpoint_conditions.human_gate_evidence_status_no_invalid_filled_records=true
checkpoint_conditions.human_gate_evidence_status_no_evidence_acceptance_candidates=true
checkpoint_conditions.human_gate_evidence_status_template_acceptance_boundary_explicit=true
negative_evidence.human_gate_evidence_status_template_intake_missing_or_invalid=0
negative_evidence.human_gate_evidence_status_template_intake_count_mismatches=0
negative_evidence.human_gate_evidence_status_template_acceptance_boundary_mismatches=0
sha256=744f972b6613c6f4e13430f7772b16eadbe3e6a3bf9fd2e9590c846b3bd2396b
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

The local checkpoint now fails closed if evidence-status no longer proves that all seven current records are blank templates awaiting human completion, that zero filled or invalid filled records exist, that zero records are evidence-acceptance candidates, or that structural validation is not acceptance. This is a local VS3-L checkpoint guard only; it does not provide VS3-P readiness or human acceptance.
