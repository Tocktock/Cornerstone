# VS3 Scenario Gate Component Proof Guard Checkpoint

**Date:** 2026-06-30 KST
**Status:** Local verifier hardening checkpoint.
**Scope:** VS3 local/dev scenario gate and report assembly only.
**Non-claim:** This is not VS3-L completion, VS3-P readiness, production/on-prem readiness, real IdP readiness, live-provider readiness, migration/restore readiness, security acceptance, or human UX acceptance.

## Slice Contract

Goal:

- A VS3 local/dev report must not pass `cornerstone scenario gate ... --json` when embedded component proofs contradict the report's local/dev assurance claim.
- The VS3 report assembler must embed component proofs refreshed for the current source snapshot instead of stale proof files from earlier slices.

Full VS3 scenario mapping:

- `in_this_slice`: `VS3-GATE-003`, `VS3-GATE-004`, `VS3-CTX-001`, `VS3-CTX-002`, `VS3-CTX-003`, `VS3-CTX-004`, `VS3-CTX-005`, `VS3-REG-003`, `VS3-REG-005`, `VS3-REG-008`.
- `later_slice`: `VS3-GATE-001`, `VS3-GATE-002`, `VS3-RLS-001` through `VS3-RLS-006`, `VS3-OPA-001` through `VS3-OPA-005`, `VS3-EGR-001` through `VS3-EGR-006`, `VS3-CON-001` through `VS3-CON-006`, `VS3-TOOL-001` through `VS3-TOOL-007`, `VS3-OBS-001` through `VS3-OBS-003`, `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-004`, `VS3-REG-006`, `VS3-REG-007`.
- `HUMAN_REQUIRED`: `VS3-H01` through `VS3-H07`.
- `out_of_scope`: production/on-prem deployment, real IdP, real network, live provider credentials, migration/restore drill, independent security review, human UX acceptance.

Proof needed:

- Native CLI success report: `cornerstone scenario verify vs3-onprem-trusted-extension --json`.
- Native CLI gate success over the fresh report: `cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json`.
- Negative/tamper proof: a local/dev report with failed embedded `request_context_proof` must exit `4` with `CS_VS3_COMPONENT_PROOF_INVALID`.
- Regression proof: focused unittest and full `test_vs3_scenario_gate*` subset.

## Current Behavior Found

Pre-fix probe observed in this work session:

- A freshly generated VS3 report was tampered by setting embedded `request_context_proof.status=failed`, `scenario_status["VS3-CTX-001"]="FAIL"`, and `checks["vs3_ctx_001_surface_context_consistent"]=false`.
- The scenario gate accepted that contradictory report with `exit=0`, `status=success`, and `errors=[]`.
- This meant the gate validated report-level coverage and negative evidence but did not validate embedded component proof truth before accepting the local/dev claim.

Additional failure found during the fix:

- After adding component proof validation, the happy path initially failed with `CS_VS3_COMPONENT_PROOF_SOURCE_TREE_STALE` for `request_context_proof`.
- Root cause: `verify_vs3_onprem_trusted_extension` reran `run_vs3_final_regression_proof(root)` but then reloaded `reports/security/vs3-request-context-proof.json`; the final-regression path refreshed RequestContext only in memory, so the top-level report embedded an older RequestContext proof file.

## Changes

- `packages/cornerstone_cli/main.py` now runs `component_proof_validation` inside `command_scenario_gate` for VS3 local/dev success claims.
- `packages/cornerstone_cli/scenarios.py` now refreshes VS3 component proofs before assembling the top-level report.
- `packages/cornerstone_cli/scenarios.py` now generates RequestContext after other component proof builders and persists the fresh RequestContext proof to `reports/security/vs3-request-context-proof.json` before the top-level verifier reloads it.
- `tests/scenario/test_scaffold_cli.py` adds a regression test that tampers the embedded RequestContext proof and expects the scenario gate to reject it.

## Evidence

Focused unittest:

```text
python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_failed_embedded_component_proof
.
Ran 1 test in 23.169s
OK
```

VS3 scenario-gate subset:

```text
python3 - <<'PY' ... test_vs3_scenario_gate* ...
Ran 39 tests in 883.199s
OK
selected 39
```

Canonical native verify and gate:

```text
./cornerstone scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
status success
errors []

./cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json
status success
errors []
component_proof_validation.status passed
source_tree_current_validation.status passed
```

Tamper probe after fix:

```text
source: reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
tamper: request_context_proof.status=failed
tamper: request_context_proof.scenario_status["VS3-CTX-001"]="FAIL"
tamper: request_context_proof.checks["vs3_ctx_001_surface_context_consistent"]=false

./cornerstone scenario gate /tmp/cs-vs3-component-proof-failed-tamper-final.json --json
exit 4
status failed
errors ['CS_VS3_COMPONENT_PROOF_INVALID']
component_proof_validation.status failed
mismatch_keys ['request_context_proof']
semantic_failure_keys ['request_context_proof']
scenario_failure_keys ['request_context_proof']
check_failure_keys ['request_context_proof']
source_tree_current_failure_keys []
semantic_error_codes ['CS_VS3_COMPONENT_PROOF_STATUS_NOT_SUCCESS', 'CS_VS3_COMPONENT_PROOF_SCENARIO_STATUS_NOT_PASS', 'CS_VS3_COMPONENT_PROOF_CHECKS_NOT_TRUE']
```

## Decision

The current slice is accepted as local verifier hardening only:

- The gate now rejects failed embedded component proofs before accepting local/dev VS3 success reports.
- The fresh canonical report gates successfully after component proof refresh and persistence.
- The human/on-prem rows remain `HUMAN_REQUIRED`.
- No production, live-provider, real-IdP, migration/restore, security-acceptance, or human-acceptance claim is made.

## Remaining Gates

- `VS3-H01` architecture/security/dependency approval remains `HUMAN_REQUIRED`.
- `VS3-H02` independent security review remains `HUMAN_REQUIRED`.
- `VS3-H03` real IdP mapping and revocation remains `HUMAN_REQUIRED`.
- `VS3-H04` real on-prem network/sandbox evidence remains `HUMAN_REQUIRED`.
- `VS3-H05` live ConnectorHub provider rehearsal remains `HUMAN_REQUIRED`.
- `VS3-H06` human operator UX/trust review remains `HUMAN_REQUIRED`.
- `VS3-H07` human-supervised migration/backup/restore drill remains `HUMAN_REQUIRED`.

## Recommended Next Slice

Continue with the next narrow VS3 gate-hardening gap only after this checkpoint:

- Prefer a missing-proof or stale-report mutation that the gate still accepts, if any.
- Do not widen into VS3-P, live providers, real IdP, migration, or human/security acceptance without the required human evidence.
