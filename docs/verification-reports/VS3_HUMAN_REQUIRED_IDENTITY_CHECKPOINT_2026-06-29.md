# VS3 Human Required Identity Checkpoint

**Date:** 2026-06-29
**Status:** Local deterministic report-shape hardening verified.
**Verdict:** `VS3-L` remains local/dev verified; `VS3-P` remains `NOT_CLAIMED`.

This checkpoint records a small compatibility hardening slice for the aggregate VS3 report. The top-level `human_required` list now exposes `scenario_id` alongside the existing `id` field, matching `scenario_results`, `human_required_blockers`, VS3-P gate output, and local checkpoint consumers.

## Slice Contract

Goal:
- Make human-required row identity explicit and machine-consistent in the aggregate VS3 report.

Scope:
- `reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json`
- `cornerstone scenario verify vs3-onprem-trusted-extension --json`
- Top-level `human_required` rows for `VS3-H01` through `VS3-H07`
- Existing VS3-P and local checkpoint gates that consume human row identity.

Non-scope:
- Changing scenario statuses.
- Accepting human evidence.
- Unlocking VS3-P.
- Claiming production/on-prem readiness, security acceptance, real IdP readiness, real network readiness, live-provider readiness, migration/restore readiness, or human UX acceptance.

Done criteria:
- Every top-level `human_required` row includes both `id` and `scenario_id`.
- The two fields match for `VS3-H01` through `VS3-H07`.
- `human_required_blockers` still reports the same seven `scenario_id` values.
- VS3-P gate reports zero human-required row identity mismatches.
- VS3 local checkpoint remains green with H rows still `HUMAN_REQUIRED`.

## Full Scenario Mapping

All 57 VS3 rows remain carried by the aggregate verifier.

| Scenario IDs | Type | Classification | Proof Surface |
| --- | --- | --- | --- |
| `VS3-GATE-001` through `VS3-OBS-003` | MUST_PASS | in_this_slice through aggregate verifier | Native VS3 scenario verifier and local proof artifacts. |
| `VS3-REG-001` through `VS3-REG-008` | REGRESSION | in_this_slice through aggregate verifier | Final regression proof, overclaim lint, dependency/default-deny checks. |
| `VS3-H01` through `VS3-H07` | HUMAN_REQUIRED | in_this_slice for identity/report-shape only | Aggregate report, VS3-P gate, and local checkpoint. Human acceptance remains out of scope. |

## Evidence

Targeted tests:

```text
python3 -m compileall packages/cornerstone_cli
exit_code=0

python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_onprem_trusted_extension_verify_closes_local_dev_ai_rows
exit_code=0
```

Aggregate report regeneration:

```text
PATH="$PWD:$PATH" cornerstone scenario verify vs3-onprem-trusted-extension \
  --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json \
  --json
exit_code=0
status=success
final_verdict=VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
summary.scenario_count=57
summary.pass=50
summary.human_required=7
human_required[*].id=VS3-H01..VS3-H07
human_required[*].scenario_id=VS3-H01..VS3-H07
claim_boundaries.vs3_p=NOT_CLAIMED
```

Dependent gates:

```text
PATH="$PWD:$PATH" cornerstone human-gate vs3-p-gate \
  --scope vs3 \
  --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json \
  --output reports/human-gates/vs3/vs3-p-gate.json \
  --json
exit_code=4
status=blocked
final_verdict=HUMAN_REQUIRED
negative_evidence.human_required_row_identity_mismatches=0
summary.vs3_p_claim=NOT_CLAIMED

PATH="$PWD:$PATH" cornerstone security vs3-local-checkpoint \
  --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json \
  --output reports/human-gates/vs3/vs3-local-checkpoint.json \
  --json
exit_code=0
status=success
final_verdict=VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
negative_evidence.human_required_rows_marked_pass=0
negative_evidence.human_required_rows_not_human_required=0
negative_evidence.vs3_p_claimed_by_checkpoint=0
```

## Decision

The aggregate VS3 report is now easier for machine consumers to validate without fallback logic. This is a report-shape hardening only; it does not alter the VS3 claim boundary.
