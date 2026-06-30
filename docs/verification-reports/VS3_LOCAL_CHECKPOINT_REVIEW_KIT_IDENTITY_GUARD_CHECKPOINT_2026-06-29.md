# VS3 Local Checkpoint Review-Kit Identity Guard Checkpoint

**Date:** 2026-06-29 KST
**Scope:** Local deterministic VS3 checkpoint hardening only.
**Status:** Local checkpoint slice verified.
**Verdict:** `VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED` remains the local checkpoint verdict. `VS3-P`, production/on-prem readiness, real IdP readiness, live-provider readiness, migration/restore readiness, security acceptance, and human acceptance remain unclaimed.

## Slice Contract

Goal:
- `cornerstone security vs3-local-checkpoint --json` must reject a stale or mismatched `reports/human-gates/vs3/review-kit.json` when its embedded `vs3_human_gate_review_kit.scenario_report` content hash or path hash does not match the aggregate VS3 scenario report currently being checkpointed.

In scope:
- Validate `vs3_human_gate_review_kit.scenario_report.sha256` against the checkpointed aggregate VS3 scenario report hash.
- Validate `vs3_human_gate_review_kit.scenario_report.path_sha256` against the checkpointed aggregate VS3 scenario report path hash.
- Emit checkpoint condition booleans and negative-evidence mismatch counters.
- Include review-kit scenario-report identity details and match booleans in the local checkpoint JSON.
- Add positive and tamper tests for stale hash and mismatched path-hash cases.
- Regenerate the VS3 scenario report, human-gate scaffold/status/review-kit reports, VS3-P gate report, and local checkpoint report so the persisted evidence set is internally consistent.

Out of scope:
- Accepting human evidence.
- Promoting `VS3-H01` through `VS3-H07` from `HUMAN_REQUIRED` to `PASS`.
- Unlocking or claiming `VS3-P`.
- Production/on-prem, real IdP, real network, live-provider, migration/restore, independent security, or human UX acceptance claims.
- Starting VS3-1 RequestContext/runtime implementation.

## Full Scenario Mapping

The frozen VS3 matrix remains unchanged:

| Type | Count | Slice classification |
|---|---:|---|
| MUST_PASS | 42 | `VS3-OBS-003` directly hardened through review-kit identity validation. `VS3-GATE-003` and `VS3-GATE-004` are guard rails for no-overclaim and native CLI checkpoint evidence. All other AI rows remain carried by the current aggregate VS3 report or later-slice scope. |
| REGRESSION | 8 | `VS3-REG-005` directly protects claim-boundary language; remaining regression rows stay final-gate coverage. |
| HUMAN_REQUIRED | 7 | `VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED` and block VS3-P. |
| Total | 57 | No scenario rows were added, removed, or reclassified by this slice. |

Directly affected scenario:

| ID | Type | Expected behavior | Verification | Result |
|---|---|---|---|---|
| VS3-OBS-003 | MUST_PASS | Human-gate packages and review kit prepare evidence without treating generated artifacts or structural checks as human evidence. | Generate and validate review kit, VS3-P gate, and local checkpoint. | Local deterministic guard strengthened; human rows remain `HUMAN_REQUIRED`. |

Regression guard:

| ID | Type | Expected behavior | Verification | Result |
|---|---|---|---|---|
| VS3-REG-005 | REGRESSION | Local reports must not describe VS3 as production/on-prem, real IdP, live provider, security-accepted, human-accepted, or migration-ready. | Checkpoint negative counters and VS3-P gate block evidence. | No VS3-P, production, security acceptance, or human acceptance claim emitted by this slice. |

## Implementation Summary

Updated `packages/cornerstone_cli/main.py`:

- Reads `vs3_human_gate_review_kit.scenario_report` from `reports/human-gates/vs3/review-kit.json`.
- Adds checkpoint conditions:
  - `human_gate_review_kit_scenario_report_hash_matches`
  - `human_gate_review_kit_scenario_report_path_hash_matches`
- Adds negative counters:
  - `human_gate_review_kit_scenario_report_hash_mismatches`
  - `human_gate_review_kit_scenario_report_path_hash_mismatches`
- Adds `human_gate_preparation.review_kit_report.scenario_report` with:
  - report path;
  - report path hash;
  - report content hash;
  - schema/status;
  - `matches_checkpoint_scenario_report_sha256`;
  - `matches_checkpoint_scenario_report_path_sha256`.

Updated `tests/scenario/test_scaffold_cli.py`:

- Positive local-checkpoint test asserts review-kit scenario-report hash and path-hash match.
- Added stale review-kit scenario-report hash rejection test.
- Added mismatched review-kit scenario-report path-hash rejection test.
- Existing review-kit human-row tamper guard remains in the focused test set.

## Current Behavior Reverse Engineering

Before this guard, a local checkpoint could validate that `review-kit.json` looked structurally ready without proving that the review kit had been generated from the same aggregate VS3 scenario report currently being checkpointed.

The failure mode is concrete:

- A review kit generated from a stale aggregate report can still contain seven human rows and preparation-only claim boundaries.
- Without comparing both the aggregate report content hash and path hash, the checkpoint can combine a current scenario report with stale human-gate review preparation.
- That would weaken `VS3-OBS-003` because the human-gate package would no longer be tied to the exact scenario evidence bundle under checkpoint.

The local checkpoint now fails closed if either identity check is false.

## Verification Evidence

Matrix count check:

```text
total=57
HUMAN_REQUIRED=7
MUST_PASS=42
REGRESSION=8
duplicate_ids=0
```

Code compile:

```text
python3 -m compileall packages/cornerstone_cli
exit 0
Listing 'packages/cornerstone_cli'...
```

Focused regression tests:

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_manifest_is_hash_backed_and_boundary_safe \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_stale_review_kit_scenario_report_hash \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_review_kit_from_different_scenario_report_path \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_tampered_review_kit_human_rows
```

Result:

```text
exit 0
Ran 4 tests in 109.297s
OK
```

Native CLI regeneration:

```text
PATH="$PWD:$PATH" cornerstone scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
exit 0
```

```text
PATH="$PWD:$PATH" cornerstone human-gate record-scaffold --scope vs3 --output-dir reports/human-gates/vs3/record-templates --force --use-existing --json --output reports/human-gates/vs3/record-scaffold.json
exit 0
final_verdict=HUMAN_REQUIRED
template_count=7
vs3_p_claim=NOT_CLAIMED
```

```text
PATH="$PWD:$PATH" cornerstone human-gate evidence-status --scope vs3 --record-dir reports/human-gates/vs3/record-templates --use-existing --json --output reports/human-gates/vs3/evidence-status.json
exit 0
final_verdict=HUMAN_REQUIRED
blank_template_pending_count=7
filled_record_count=0
evidence_acceptance_candidate_count=0
vs3_p_claim=NOT_CLAIMED
```

```text
PATH="$PWD:$PATH" cornerstone human-gate review-kit --scope vs3 --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --use-existing --json --output reports/human-gates/vs3/review-kit.json
exit 0
status=success
final_verdict=HUMAN_REQUIRED
review_queue_count=7
package_count=7
template_count=7
vs3_p_claim=NOT_CLAIMED
scenario_report.sha256=1e8d59dcea0af00c48d51706e21d4b2ff67e25a0df4d5420887a2be45b5f477c
scenario_report.path_sha256=3a3f9a00cfe857b62ba2d1a30276c585bcde1e7ebf13a9f21f58c478542ee84c
```

```text
PATH="$PWD:$PATH" cornerstone human-gate vs3-p-gate --scope vs3 --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --output reports/human-gates/vs3/vs3-p-gate.json --json
exit 4
status=blocked
final_verdict=HUMAN_REQUIRED
unresolved_human_required_rows=7
vs3_p_ready=false
vs3_p_claim=NOT_CLAIMED
```

```text
PATH="$PWD:$PATH" cornerstone security vs3-local-checkpoint --json --output reports/human-gates/vs3/vs3-local-checkpoint.json
exit 0
status=success
final_verdict=VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
scenario_count=57
pass=50
human_required=7
blocking=0
unresolved_human_required_rows=7
vs3_l_claim=LOCAL_DEV_ASSURANCE_VERIFIED
vs3_p_claim=NOT_CLAIMED
production_readiness_claim_allowed=false
security_acceptance_claim_allowed=false
human_acceptance_claim_allowed=false
human_gate_review_kit_scenario_report_hash_matches=true
human_gate_review_kit_scenario_report_path_hash_matches=true
human_gate_review_kit_scenario_report_hash_mismatches=0
human_gate_review_kit_scenario_report_path_hash_mismatches=0
review_kit_report.scenario_report.matches_checkpoint_scenario_report_sha256=true
review_kit_report.scenario_report.matches_checkpoint_scenario_report_path_sha256=true
```

## Pass / Fail Criteria

PASS for this slice:

- The local checkpoint succeeds only when review-kit scenario-report content hash and path hash match the checkpointed aggregate VS3 scenario report.
- A stale review-kit content hash fails with exit code `4`.
- A mismatched review-kit scenario-report path hash fails with exit code `4`.
- Failure output keeps `vs3_p_claimed_by_checkpoint=0`, `production_readiness_claimed_by_checkpoint=0`, and `human_acceptance_claimed_by_checkpoint=0`.
- All seven VS3 human gates remain `HUMAN_REQUIRED`.

FAIL for this slice:

- The checkpoint accepts a review kit generated from a different scenario report hash or path identity.
- The checkpoint treats review-kit structural validity as human evidence acceptance.
- Any generated output claims VS3-P, production/on-prem, security acceptance, or human acceptance.

## Failure Reverse Engineering

During verification, the focused tests regenerated report artifacts and left the persisted local checkpoint stale relative to the regenerated review kit. A compact JSON check showed:

- `reports/human-gates/vs3/vs3-local-checkpoint.json` still carried the prior review-kit scenario-report hash.
- `reports/human-gates/vs3/review-kit.json` had been regenerated against the current scenario report.

Fix:

- Reran the native sequence from aggregate scenario verification through human-gate scaffold, evidence-status, review-kit, VS3-P gate, and local checkpoint.
- Rechecked compact JSON summaries after regeneration.

Re-verification:

- The persisted checkpoint now records review-kit hash/path-hash matches as `true`.
- Review-kit and checkpoint both record scenario-report hash `1e8d59dcea0af00c48d51706e21d4b2ff67e25a0df4d5420887a2be45b5f477c`.

## Remaining Human Required Gates

The following remain unresolved and block VS3-P:

- `VS3-H01`: owner architecture/security/dependency/migration approval.
- `VS3-H02`: independent security review and retest.
- `VS3-H03`: real IdP mapping and revocation evidence.
- `VS3-H04`: real on-prem network/firewall/proxy/service-mesh evidence.
- `VS3-H05`: approved live-provider rehearsal.
- `VS3-H06`: human operator UX/trust review.
- `VS3-H07`: supervised migration/backup/restore drill.

## Decision

Continue VS3 in the next small verified slice. This checkpoint strengthens the local evidence chain for human-gate review-kit preparation, but it does not complete VS3-P or any human/on-prem gate.
