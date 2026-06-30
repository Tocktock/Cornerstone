# VS3 Local Checkpoint Component Proof Scope Key Taxonomy Guard Checkpoint

**Date:** 2026-06-30 KST
**Owner:** JiYong / Tars
**Status:** AI-verifiable verifier hardening slice passed locally.
**Scope:** VS3 local checkpoint component-proof scope schema hardening only.

## Slice Contract

Goal:
- Make `cornerstone security vs3-local-checkpoint --json` reject a scenario-backed component proof report when every embedded `scope` copy contains an unknown authority-looking key, even if the component proof file, embedded scenario-report copy, and nested command transcript copies still match each other.

In-slice scenarios:
- `VS3-GATE-003` (`MUST_PASS`): VS3 local/dev report metadata must not carry proof claims or authority fields stronger than the evidence boundary.
- `VS3-GATE-004` (`MUST_PASS`): native VS3 verifier/gate metadata must reject unsafe component-proof evidence through the CLI path.
- `VS3-REG-005` (`REGRESSION`): VS3 reports and release metadata must not describe local/dev proof as production, real IdP, live provider, security-accepted, human-accepted, or migration-ready.

Out of scope:
- VS3-P readiness.
- Production/on-prem readiness.
- Real IdP, real network, live provider, independent security review, migration/restore, or human UX acceptance.
- New RequestContext, RLS, OPA, egress, ConnectorHub, Tool SDK, or operator UI product behavior.

Human-required boundaries:
- `VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`.
- This slice adds no human approval evidence.

## Current Behavior Reverse-Engineered

Before this slice, component proof traceability checked required `scope` fields:
- `tenant_id`
- `owner_id`
- `namespace_id`
- `workspace_id`
- `scope_source`

It did not reject unknown `scope` keys. A component proof could therefore add a forged-authority-looking field such as `scope.role=admin` to every scope copy. Because all copies matched and the required fields remained valid, the local checkpoint still returned local-dev success.

## Change

Updated `packages/cornerstone_cli/main.py`:
- Added `VS3_SCOPE_ALLOWED_KEYS`.
- Added `_vs3_scope_key_errors(...)`.
- Added unexpected scope-key checks to component proof traceability state.
- Added unexpected scope-key checks to CLI command transcript and stdout JSON scope validation.
- Kept missing traceability, traceability mismatch, and invalid traceability as distinct semantic failures.

Updated `tests/scenario/test_scaffold_cli.py`:
- Added `test_vs3_local_checkpoint_rejects_component_proof_scope_extra_key_even_when_all_copies_match`.
- The test tampers every nested `scope` copy in `reports/security/vs3-request-context-proof.json` with `role=admin`.
- It refreshes derived human-gate artifacts so identity still matches, then verifies the checkpoint fails at the traceability/scope key taxonomy layer.

## Verification Evidence

Pre-patch internally consistent tamper probe:

```text
returncode 0
status success
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
component_proof_report_semantic_failures 0
component_proof_report_traceability_failures 0
component_proof_report_cli_command_evidence_shape_failures 0
embedded_traceability_invalid_fields []
file_traceability_invalid_fields []
traceability_success True
cli_command_evidence_shape_success True
semantic_error_codes []
semantics_success True
```

Post-patch internally consistent tamper probe:

```text
returncode 4
status failed
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_NOT_VERIFIED
failed_conditions ['component_proof_request_context_proof_semantics_success', 'component_proof_request_context_proof_traceability_success', 'component_proof_request_context_proof_cli_command_evidence_shape_valid']
embedded_traceability_invalid_fields ['scope.role']
file_traceability_invalid_fields ['scope.role']
traceability_success False
cli_command_evidence_shape_success False
semantic_error_codes ['CS_VS3_COMPONENT_PROOF_TRACEABILITY_INVALID', 'CS_VS3_COMPONENT_PROOF_CLI_COMMAND_EVIDENCE_INVALID']
component_proof_report_semantic_failures 1
component_proof_report_traceability_failures 1
component_proof_report_cli_command_evidence_shape_failures 1
```

Commands run:

```bash
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
```

Result:

```text
exit 0
```

Focused regression:

```bash
PYTHONPATH=packages:. python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_scope_extra_key_even_when_all_copies_match
```

Result:

```text
Ran 1 test in 54.304s
OK
```

Adjacent traceability/source-tree coverage:

```bash
PYTHONPATH=packages:. python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_missing_traceability_even_when_identity_matches \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_scope_extra_key_even_when_all_copies_match \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_transcript_missing_traceability_even_when_identity_matches \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_source_tree_extra_key_even_when_all_copies_match
```

Result:

```text
Ran 4 tests in 215.710s
OK
```

Clean checkpoint command:

```bash
PYTHONPATH=packages:. python3 -m cornerstone_cli.main security vs3-local-checkpoint --json
```

Observed summary:

```text
status success
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
pass 50
human_required 7
blocking 0
component_proof_report_semantic_failures 0
component_proof_report_traceability_failures 0
component_proof_report_cli_command_evidence_shape_failures 0
vs3_p NOT_CLAIMED
production_onprem NOT_CLAIMED
security_acceptance NOT_CLAIMED
human_acceptance NOT_CLAIMED
```

## Pass / Fail Criteria

PASS:
- Unknown component proof `scope` keys fail the local checkpoint.
- File, embedded report, and nested command transcript scope copies can still match, proving this is not only a hash-mismatch guard.
- Existing missing-traceability and transcript-traceability failures remain distinct.
- Existing clean component proof scope schemas still pass local checkpoint coverage.
- The final verdict remains no stronger than the evidence.

FAIL:
- A forged-authority-looking `scope` key can appear in every nested copy without failing the checkpoint.
- The verifier only fails because file and embedded report bytes diverge.
- Missing traceability, mismatched traceability, and invalid scope keys collapse into the same error.
- Existing local component proof reports fail only because their current legitimate keys were not included in the taxonomy.

## Verdict

This slice is locally verified for `VS3-GATE-003` / `VS3-GATE-004` / `VS3-REG-005` verifier hardening only.

It does not claim full VS3-L completion, VS3-P readiness, production/on-prem readiness, security acceptance, real IdP readiness, live-provider readiness, migration/restore readiness, or human UX acceptance.
