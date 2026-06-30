# VS3 Local Checkpoint Component Proof CLI Command Elapsed Duration Guard Checkpoint

**Date:** 2026-06-30 KST
**Owner:** JiYong / Tars
**Status:** Local deterministic elapsed-duration evidence guard slice verified.
**Verdict:** `VS3-L` local/dev assurance remains verified; `VS3-P` remains `NOT_CLAIMED` and blocked by human evidence.

This checkpoint covers the VS3 component proof CLI command transcript elapsed-duration field. Component proof transcripts must carry a finite, non-negative `elapsed_seconds` value; missing, non-numeric, infinite, or negative elapsed-duration evidence is incomplete and must make `cornerstone security vs3-local-checkpoint --json` fail closed instead of preserving a VS3-L claim.

## Slice Contract

Goal:
- Prove that local checkpoint rejects component proof CLI transcript evidence when `elapsed_seconds` is missing or invalid.
- Ensure regenerated VS3 component proof reports include deterministic elapsed-duration evidence in both transcript and `stdout_json` envelopes.

Scope:
- `cornerstone security vs3-local-checkpoint --json`
- Component proof report `command_transcripts[*].elapsed_seconds`
- Shared VS3 component proof transcript enrichment
- Local deterministic tamper regression over `reports/security/vs3-request-context-proof.json`

Non-scope:
- Adding production runtime behavior.
- Proving real command wall-clock performance.
- Accepting human evidence.
- Moving any `VS3-H*` row from `HUMAN_REQUIRED` to `PASS`.
- Unlocking VS3-P.
- Claiming production/on-prem readiness, real IdP readiness, live-provider readiness, real-network readiness, migration/restore readiness, security acceptance, or human UX acceptance.

Done criteria:
- Component proof transcript enrichment emits `elapsed_seconds`.
- Checkpoint validation rejects invalid elapsed-duration values with a stable error code.
- The focused tamper test returns exit code 4 and `VS3_L_LOCAL_DEV_ASSURANCE_NOT_VERIFIED`.
- The successful regenerated checkpoint reports zero component proof CLI command evidence shape failures.
- VS3-P gate remains blocked on seven human rows.

## Full Scenario Mapping

All 57 frozen VS3 rows remain mapped before this slice is narrowed.

| Classification | Scenario IDs | Reason |
|---|---|---|
| `in_this_slice` | `VS3-GATE-004`, `VS3-REG-004`, `VS3-REG-005`, `VS3-REG-008` | This slice hardens native CLI transcript evidence shape, coverage failure detection, overclaim boundaries, and conservative local checkpoint defaults for stale or tampered elapsed-duration evidence. |
| `later_slice` | `VS3-GATE-001`, `VS3-GATE-002`, `VS3-GATE-003`, `VS3-CTX-001`, `VS3-CTX-002`, `VS3-CTX-003`, `VS3-CTX-004`, `VS3-CTX-005`, `VS3-RLS-001`, `VS3-RLS-002`, `VS3-RLS-003`, `VS3-RLS-004`, `VS3-RLS-005`, `VS3-RLS-006`, `VS3-OPA-001`, `VS3-OPA-002`, `VS3-OPA-003`, `VS3-OPA-004`, `VS3-OPA-005`, `VS3-EGR-001`, `VS3-EGR-002`, `VS3-EGR-003`, `VS3-EGR-004`, `VS3-EGR-005`, `VS3-EGR-006`, `VS3-CON-001`, `VS3-CON-002`, `VS3-CON-003`, `VS3-CON-004`, `VS3-CON-005`, `VS3-CON-006`, `VS3-TOOL-001`, `VS3-TOOL-002`, `VS3-TOOL-003`, `VS3-TOOL-004`, `VS3-TOOL-005`, `VS3-TOOL-006`, `VS3-TOOL-007`, `VS3-OBS-001`, `VS3-OBS-002`, `VS3-OBS-003`, `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-003`, `VS3-REG-006`, `VS3-REG-007` | These rows require their own evidence reconciliation, RequestContext, RLS, OPA, egress, ConnectorHub, tool-registry, observability, human-gate, UI, supply-chain, or broader regression slices. |
| `HUMAN_REQUIRED` | `VS3-H01`, `VS3-H02`, `VS3-H03`, `VS3-H04`, `VS3-H05`, `VS3-H06`, `VS3-H07` | These rows require owner/security approval, independent security review, real IdP, real network, live provider, human UX, or migration/restore evidence. |

## Implementation Notes

- `packages/cornerstone_cli/scenarios.py` now enriches VS3 component proof command transcripts with `elapsed_seconds=0.0` when no measured duration is available and mirrors that value into generated `stdout_json`.
- `packages/cornerstone_cli/main.py` now treats missing, non-numeric, infinite, or negative `elapsed_seconds` as invalid component proof CLI evidence.
- `tests/scenario/test_scaffold_cli.py` now includes `test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_elapsed_seconds_invalid`.
- The metadata regression expectation was tightened so a malformed transcript without secrets is counted as a CLI evidence shape failure, not a secret-scan failure.

## Evidence

Focused checks:

```text
python3 -m py_compile packages/cornerstone_cli/main.py packages/cornerstone_cli/scenarios.py tests/scenario/test_scaffold_cli.py
exit_code=0

python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_missing_metadata \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_elapsed_seconds_invalid \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_timeout \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_exit_code_outside_contract
exit_code=0
Ran 4 tests in 110.345s
OK
```

Regenerated native CLI artifacts:

```text
PATH="$PWD:$PATH" cornerstone scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
exit_code=0
status=success
final_verdict=VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
summary.scenario_count=57
summary.pass=50
summary.human_required=7
summary.blocking=0

PATH="$PWD:$PATH" cornerstone human-gate record-scaffold --scope vs3 --output-dir reports/human-gates/vs3/record-templates --force --use-existing --json --output reports/human-gates/vs3/record-scaffold.json
exit_code=0
status=success
final_verdict=HUMAN_REQUIRED
summary.template_count=7
summary.vs3_p_claim=NOT_CLAIMED

PATH="$PWD:$PATH" cornerstone human-gate evidence-status --scope vs3 --record-dir reports/human-gates/vs3/record-templates --use-existing --json --output reports/human-gates/vs3/evidence-status.json
exit_code=0
status=success
final_verdict=HUMAN_REQUIRED
summary.blank_template_count=7
summary.vs3_p_claim=NOT_CLAIMED

PATH="$PWD:$PATH" cornerstone human-gate review-kit --scope vs3 --use-existing --json --output reports/human-gates/vs3/review-kit.json
exit_code=0
status=success
final_verdict=HUMAN_REQUIRED
summary.template_count=7
summary.vs3_p_claim=NOT_CLAIMED

PATH="$PWD:$PATH" cornerstone human-gate vs3-p-gate --scope vs3 --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json --output reports/human-gates/vs3/vs3-p-gate.json
exit_code=4
status=blocked
final_verdict=HUMAN_REQUIRED
summary.status=blocked_on_human_required_evidence
summary.vs3_p_ready=false
summary.vs3_p_claim=NOT_CLAIMED

PATH="$PWD:$PATH" cornerstone security vs3-local-checkpoint --json --output reports/security/vs3-local-checkpoint.json
exit_code=0
status=success
final_verdict=VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
summary.component_proof_report_cli_command_evidence_shape_failures=0
summary.vs3_p_claim=NOT_CLAIMED

PATH="$PWD:$PATH" cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json
exit_code=0
status=success
final_verdict=VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
summary.scenario_count=57
summary.pass=50
summary.human_required=7
summary.blocking=0
```

Component proof transcript scan:

```text
reports/security/vs3-request-context-proof.json count=4 elapsed_invalid=[] stdout_elapsed_invalid=[]
reports/db/vs3-postgres-rls-proof.json count=6 elapsed_invalid=[] stdout_elapsed_invalid=[]
reports/policy/vs3-opa-policy-proof.json count=3 elapsed_invalid=[] stdout_elapsed_invalid=[]
reports/security/vs3-egress-sandbox-proof.json count=3 elapsed_invalid=[] stdout_elapsed_invalid=[]
reports/security/vs3-connectorhub-source-proof.json count=5 elapsed_invalid=[] stdout_elapsed_invalid=[]
reports/security/vs3-tool-registry-proof.json count=5 elapsed_invalid=[] stdout_elapsed_invalid=[]
reports/observability/vs3-observability-proof.json count=6 elapsed_invalid=[] stdout_elapsed_invalid=[]
reports/security/vs3-final-regression-proof.json count=4 elapsed_invalid=[] stdout_elapsed_invalid=[]
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

The elapsed-duration guard is verified for the current local/dev slice. Missing or invalid elapsed-duration evidence can no longer support a VS3-L local/dev claim, regenerated component proofs carry valid duration fields, and VS3-P, production/on-prem, live-provider, real-IdP, real-network, migration/restore, security-acceptance, and human-acceptance boundaries remain unclaimed.
