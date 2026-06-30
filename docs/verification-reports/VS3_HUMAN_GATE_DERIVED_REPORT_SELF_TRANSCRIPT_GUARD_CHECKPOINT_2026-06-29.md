# VS3 Human-Gate Derived Report Self-Transcript Guard Checkpoint

**Date:** 2026-06-29 KST
**Owner:** JiYong / Tars
**Status:** Local deterministic checkpoint for VS3 CLI transcript evidence hardening.
**Scope:** VS3 human-gate derived report CLI self-command transcript shape, stdout JSON, and local checkpoint validation.

## Goal

Ensure VS3 human-gate derived reports carry native `cornerstone ... --json` self-command transcript evidence that the aggregate VS3 local checkpoint validates before preserving a VS3-L local/dev assurance claim.

This slice does not approve human evidence and does not move any VS3-H row out of `HUMAN_REQUIRED`.

## Full Scenario Mapping

| Classification | Scenario IDs | Reason |
|---|---|---|
| `in_this_slice` | VS3-GATE-004, VS3-OBS-003, VS3-REG-004, VS3-REG-005, VS3-REG-007 | The slice strengthens native CLI transcript evidence for human-gate derived reports and makes the local checkpoint reject missing transcript proof. |
| `later_slice` | VS3-GATE-001, VS3-GATE-002, VS3-GATE-003, VS3-CTX-001, VS3-CTX-002, VS3-CTX-003, VS3-CTX-004, VS3-CTX-005, VS3-RLS-001, VS3-RLS-002, VS3-RLS-003, VS3-RLS-004, VS3-RLS-005, VS3-RLS-006, VS3-OPA-001, VS3-OPA-002, VS3-OPA-003, VS3-OPA-004, VS3-OPA-005, VS3-EGR-001, VS3-EGR-002, VS3-EGR-003, VS3-EGR-004, VS3-EGR-005, VS3-EGR-006, VS3-CON-001, VS3-CON-002, VS3-CON-003, VS3-CON-004, VS3-CON-005, VS3-CON-006, VS3-TOOL-001, VS3-TOOL-002, VS3-TOOL-003, VS3-TOOL-004, VS3-TOOL-005, VS3-TOOL-006, VS3-TOOL-007, VS3-OBS-001, VS3-OBS-002, VS3-REG-001, VS3-REG-002, VS3-REG-003, VS3-REG-006, VS3-REG-008 | These rows require their own security, policy, RLS, egress, ConnectorHub, tool-registry, UI, or broader regression proof slices. |
| `HUMAN_REQUIRED` | VS3-H01, VS3-H02, VS3-H03, VS3-H04, VS3-H05, VS3-H06, VS3-H07 | These rows require owner, independent security, real IdP, real network, live provider, operator UX, or migration/restore evidence. |

## Slice Contract

| ID | Type | Expected Result | Verification Method | Evidence Required | Status |
|---|---|---|---|---|---|
| VS3-GATE-004 | MUST_PASS | Human-gate derived report commands expose native CLI transcript evidence with `stdout_json`, scope, source tree, evidence refs, audit refs, and proof boundary. | Focused unit tests and native CLI report regeneration. | CLI output, generated JSON reports, self transcript validation objects. | PASS |
| VS3-OBS-003 | MUST_PASS | Human-gate evidence preparation remains package/template/status/review-kit only and cannot mark human rows PASS. | Existing human-gate tests plus local checkpoint. | `reports/human-gates/vs3/*.json`, negative evidence counters, checkpoint conditions. | PASS |
| VS3-REG-004 | REGRESSION | Local checkpoint detects missing human-gate derived report transcript evidence before release claim. | Tamper test removes `self_command_transcript` from each derived report and expects checkpoint failure. | `test_vs3_local_checkpoint_rejects_human_gate_reports_missing_self_transcript`. | PASS |
| VS3-REG-005 | REGRESSION | Derived reports and checkpoint do not overclaim VS3-P, production/on-prem readiness, security acceptance, or human acceptance. | Focused tests and `security vs3-local-checkpoint`. | Claim boundary fields, negative evidence counters, final verdict. | PASS |
| VS3-REG-007 | REGRESSION | No new production dependency, signing tool, sandbox runtime, or lockfile churn is introduced. | Git diff and generated report review. | No dependency or lockfile changes in this slice. | PASS |

## Implementation Notes

- Added a shared VS3 CLI stdout proof-boundary helper for human-gate report transcripts.
- Added structured `self_command_transcript` and `self_command_transcript_validation` to:
  - `cornerstone human-gate evidence-status --scope vs3 --json`
  - `cornerstone human-gate review-kit --scope vs3 --json`
  - `cornerstone human-gate vs3-p-gate --scope vs3 --json`
- Added deterministic local audit refs for the non-mutating report-generation events.
- Extended `cornerstone security vs3-local-checkpoint --json` to revalidate each derived report transcript from the persisted report files.
- Added a negative regression test that removes the transcript from each derived report and confirms the checkpoint rejects it.

## Verification Evidence

| Check | Result | Evidence |
|---|---|---|
| `python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py` | PASS | Exit code 0. |
| Focused human-gate/checkpoint tests | PASS | `Ran 5 tests in 100.095s OK`. |
| `cornerstone scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json` | PASS | Output status `success`; final verdict `VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED`. |
| `cornerstone human-gate evidence-status --scope vs3 --record-dir reports/human-gates/vs3/record-templates --use-existing --json --output reports/human-gates/vs3/evidence-status.json` | PASS | Output status `success`; `self_command_transcript_validation=passed`; `stdout_json` present. |
| `cornerstone human-gate review-kit --scope vs3 --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --use-existing --json --output reports/human-gates/vs3/review-kit.json` | PASS | Output status `success`; `self_command_transcript_validation=passed`; `stdout_json` present. |
| `cornerstone human-gate vs3-p-gate --scope vs3 --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --output reports/human-gates/vs3/vs3-p-gate.json --json` | Expected block | Exit code 4; payload status `blocked`; final verdict `HUMAN_REQUIRED`; `self_command_transcript_validation=passed`; `stdout_json` present. |
| `cornerstone security vs3-local-checkpoint --json` | PASS | Status `success`; final verdict `VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED`; all three derived transcript conditions true; transcript failure counters all 0. |

## Proof Boundary

| Surface | Claim |
|---|---|
| Local deterministic tests | Covered for this transcript guard slice. |
| Native CLI reports | Covered for the three target derived reports and aggregate local checkpoint. |
| VS3-L | Still local/dev assurance only. |
| VS3-P | Not claimed. Remains blocked by VS3-H01 through VS3-H07. |
| Production/on-prem readiness | Not claimed. |
| Real IdP, real network, live provider, migration/restore, security acceptance, human UX acceptance | Not claimed. |

## Remaining Human Required

VS3-H01 through VS3-H07 remain `HUMAN_REQUIRED`. Generated templates, review kits, structural validation, and local checkpoint evidence do not count as human approval or production/on-prem evidence.

## Decision

Continue to the next VS3 slice. This slice improves the VS3 evidence gate by preventing human-gate derived report CLI transcript evidence from silently disappearing while preserving all human/on-prem proof boundaries.
