# VS3 Local Checkpoint Component Proof CLI Command Timeout Guard Checkpoint

**Date:** 2026-06-29 KST
**Owner:** JiYong / Tars
**Status:** Local deterministic timeout-state guard slice verified.
**Verdict:** `VS3-L` local/dev assurance remains verified; `VS3-P` remains `NOT_CLAIMED` and blocked by human evidence.

This checkpoint covers the VS3 component proof CLI command transcript timeout field. A component proof transcript with `timed_out=true` is incomplete evidence and must make `cornerstone security vs3-local-checkpoint --json` fail closed instead of preserving a VS3-L claim.

## Slice Contract

Goal:
- Prove that local checkpoint rejects component proof CLI transcript evidence when the recorded command timed out.

Scope:
- `cornerstone security vs3-local-checkpoint --json`
- Component proof report `command_transcripts[*].timed_out`
- Local deterministic tamper regression over `reports/security/vs3-request-context-proof.json`

Non-scope:
- Adding new production runtime behavior.
- Accepting human evidence.
- Moving any `VS3-H*` row from `HUMAN_REQUIRED` to `PASS`.
- Unlocking VS3-P.
- Claiming production/on-prem readiness, real IdP readiness, live-provider readiness, real-network readiness, migration/restore readiness, security acceptance, or human UX acceptance.

Done criteria:
- Timed-out component proof transcript entries are rejected with `timed_out_true`.
- The checkpoint returns exit code 4 and `VS3_L_LOCAL_DEV_ASSURANCE_NOT_VERIFIED` for the tampered report.
- Missing command evidence and malformed command shape remain separate counters.
- The successful regenerated checkpoint still reports all current component proof transcript shape checks as passing.
- VS3-P gate remains blocked on seven human rows.

## Full Scenario Mapping

All 57 frozen VS3 rows remain mapped before this slice is narrowed.

| Classification | Scenario IDs | Reason |
|---|---|---|
| `in_this_slice` | `VS3-GATE-004`, `VS3-REG-004`, `VS3-REG-005`, `VS3-REG-008` | This slice hardens native CLI transcript timeout evidence, coverage failure detection, overclaim boundaries, and conservative release defaults for timed-out local proof. |
| `later_slice` | `VS3-GATE-001`, `VS3-GATE-002`, `VS3-GATE-003`, `VS3-CTX-001`, `VS3-CTX-002`, `VS3-CTX-003`, `VS3-CTX-004`, `VS3-CTX-005`, `VS3-RLS-001`, `VS3-RLS-002`, `VS3-RLS-003`, `VS3-RLS-004`, `VS3-RLS-005`, `VS3-RLS-006`, `VS3-OPA-001`, `VS3-OPA-002`, `VS3-OPA-003`, `VS3-OPA-004`, `VS3-OPA-005`, `VS3-EGR-001`, `VS3-EGR-002`, `VS3-EGR-003`, `VS3-EGR-004`, `VS3-EGR-005`, `VS3-EGR-006`, `VS3-CON-001`, `VS3-CON-002`, `VS3-CON-003`, `VS3-CON-004`, `VS3-CON-005`, `VS3-CON-006`, `VS3-TOOL-001`, `VS3-TOOL-002`, `VS3-TOOL-003`, `VS3-TOOL-004`, `VS3-TOOL-005`, `VS3-TOOL-006`, `VS3-TOOL-007`, `VS3-OBS-001`, `VS3-OBS-002`, `VS3-OBS-003`, `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-003`, `VS3-REG-006`, `VS3-REG-007` | These rows require their own evidence reconciliation, RequestContext, RLS, OPA, egress, ConnectorHub, tool-registry, observability, human-gate, UI, supply-chain, or broader regression slices. |
| `HUMAN_REQUIRED` | `VS3-H01`, `VS3-H02`, `VS3-H03`, `VS3-H04`, `VS3-H05`, `VS3-H06`, `VS3-H07` | These rows require owner/security approval, independent security review, real IdP, real network, live provider, human UX, or migration/restore evidence. |

## Implementation Notes

- `tests/scenario/test_scaffold_cli.py` now includes `test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_timeout`.
- The production validation path already rejected timeout evidence through `_vs3_cli_command_transcript_errors`; this slice locks that behavior with a scenario-specific tamper regression.
- The test mutates a current component proof transcript to `timed_out=true`, updates the embedded aggregate scenario report copy, refreshes derived human-gate reports, and verifies local checkpoint rejection.

## Evidence

Focused checks:

```text
python3 -m py_compile tests/scenario/test_scaffold_cli.py
exit_code=0

python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_invalid_timestamp \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_reversed_timestamp \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_timeout \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_exit_code_outside_contract
exit_code=0
Ran 4 tests in 110.007s
OK
```

Regenerated native CLI artifacts:

Hash note:
- The generated VS3 reports include this checkpoint Markdown path in their dirty-source snapshot. The hashes below are from the post-checkpoint regeneration immediately before this evidence-text refresh; rerunning after editing this section will intentionally produce a new source snapshot hash.

```text
PATH="$PWD:$PATH" cornerstone scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
exit_code=0
status=success
final_verdict=VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
summary.scenario_count=57
summary.pass=50
summary.human_required=7
summary.blocking=0
pre_evidence_refresh_sha256=07162550b8b55036214437baa9edf4507ae0a511a2548105102ae541df86acd8

PATH="$PWD:$PATH" cornerstone human-gate record-scaffold --scope vs3 --output-dir reports/human-gates/vs3/record-templates --output reports/human-gates/vs3/record-scaffold.json --force --json
exit_code=0
status=success
final_verdict=HUMAN_REQUIRED
summary.template_count=7
summary.vs3_p_claim=NOT_CLAIMED
pre_evidence_refresh_sha256=06adf1f7ccd3acfffce54eb958421003b9403fd32021940b0c777065d7c48a4f

PATH="$PWD:$PATH" cornerstone human-gate evidence-status --scope vs3 --record-dir reports/human-gates/vs3/record-templates --use-existing --json --output reports/human-gates/vs3/evidence-status.json
exit_code=0
status=success
final_verdict=HUMAN_REQUIRED
summary.vs3_p_claim=NOT_CLAIMED
pre_evidence_refresh_sha256=0dfcadc616c1a748c0f12cad98f31c201c66b751f0cc1485a38809a003ce72f0

PATH="$PWD:$PATH" cornerstone human-gate review-kit --scope vs3 --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --use-existing --json --output reports/human-gates/vs3/review-kit.json
exit_code=0
status=success
final_verdict=HUMAN_REQUIRED
summary.template_count=7
summary.vs3_p_claim=NOT_CLAIMED
pre_evidence_refresh_sha256=9a29617ed126e1964d26d1f5895aeaf0d475942881d0f06302b10bc3e6827716

PATH="$PWD:$PATH" cornerstone human-gate vs3-p-gate --scope vs3 --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --output reports/human-gates/vs3/vs3-p-gate.json --json
exit_code=4
status=blocked
final_verdict=HUMAN_REQUIRED
summary.status=blocked_on_human_required_evidence
summary.vs3_p_ready=false
summary.vs3_p_claim=NOT_CLAIMED
pre_evidence_refresh_sha256=a4f926664ed5aa1ff17b7bdfb8c34a2c191722db42dfb8ec9e45af0d7b97a0cb

PATH="$PWD:$PATH" cornerstone security vs3-local-checkpoint --json --output reports/human-gates/vs3/vs3-local-checkpoint.json
exit_code=0
status=success
final_verdict=VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
summary.component_proof_report_cli_command_evidence_shape_failures=0
summary.vs3_p_claim=NOT_CLAIMED
pre_evidence_refresh_sha256=1ff66d8fe90d635566e7d74d9fcd5f6e4f49db01fba70b9ad0a7dc73a6d29b71

PATH="$PWD:$PATH" cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json
exit_code=0
status=success
final_verdict=VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
summary.scenario_count=57
summary.pass=50
summary.human_required=7
summary.blocking=0
```

## Remaining Human Gates

| Scenario | Current Status | Required Proof |
|---|---|---|
| `VS3-H01` | HUMAN_REQUIRED | Owner architecture/security approval. |
| `VS3-H02` | HUMAN_REQUIRED | Independent security review and retest. |
| `VS3-H03` | HUMAN_REQUIRED | Real IdP mapping and revocation evidence. |
| `VS3-H04` | HUMAN_REQUIRED | Real on-prem network/security evidence. |
| `VS3-H05` | HUMAN_REQUIRED | Approved live ConnectorHub/provider rehearsal. |
| `VS3-H06` | HUMAN_REQUIRED | Human operator UX/trust review. |
| `VS3-H07` | HUMAN_REQUIRED | Human-supervised migration/backup/restore/rollback drill. |

## Decision

The timeout-state guard is verified for the current local/dev slice. Timed-out native CLI transcript evidence cannot support a VS3-L local/dev claim, and the successful checkpoint still preserves all VS3-P, production/on-prem, live-provider, real-IdP, real-network, migration/restore, security-acceptance, and human-acceptance boundaries.
