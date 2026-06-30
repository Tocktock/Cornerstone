# VS3 Local Checkpoint Component Proof Source Tree Key Taxonomy Guard Checkpoint

**Date:** 2026-06-30 KST
**Owner:** JiYong / Tars
**Status:** AI-verifiable verifier hardening slice passed locally.
**Scope:** VS3 local checkpoint component-proof source-tree schema hardening only.

## Slice Contract

Goal:
- Make `cornerstone security vs3-local-checkpoint --json` reject a scenario-backed component proof report when every embedded `source_tree` copy contains an unknown key, even if the component proof file, embedded scenario-report copy, and nested command transcript copies still match each other.

In-slice scenarios:
- `VS3-GATE-003` (`MUST_PASS`): VS3 local/dev report metadata must not carry proof claims stronger than the evidence boundary.
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

Before this slice, component proof `source_tree` validation checked:
- file and embedded report equality;
- current source-tree fingerprint fields;
- command transcript source-tree equality.

It did not reject unknown `source_tree` keys. A component proof could therefore add a claim-looking field such as `source_tree.onprem_security_acceptance=CLAIMED` to every source-tree copy. Because all copies matched and the current-source fingerprint ignores unknown fields, the checkpoint still returned local-dev success.

## Change

Updated `packages/cornerstone_cli/main.py`:
- Added `VS3_SOURCE_TREE_ALLOWED_KEYS` for the current 9-key, 11-key, and 13-key source-tree shapes.
- Added `_vs3_source_tree_key_errors(...)`.
- Added `source_tree_key_success`, embedded/file key error details, and `CS_VS3_COMPONENT_PROOF_SOURCE_TREE_UNSAFE`.
- Added source-tree key checks to CLI command transcript and stdout JSON source-tree validation.
- Kept missing source tree and stale source tree as distinct failures.

Updated `tests/scenario/test_scaffold_cli.py`:
- Added `test_vs3_local_checkpoint_rejects_component_proof_source_tree_extra_key_even_when_all_copies_match`.
- The test tampers every nested `source_tree` copy in `reports/security/vs3-request-context-proof.json` with `onprem_security_acceptance=CLAIMED`.
- It refreshes derived human-gate artifacts so identity still matches, then verifies the checkpoint fails at the source-tree key taxonomy layer.

## Verification Evidence

Pre-patch internally consistent tamper probe:

```text
returncode 0
status success
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
source_tree_identity_success True
source_tree_current_success True
cli_command_evidence_shape_success True
semantic_error_codes []
semantics_success True
```

Post-patch internally consistent tamper probe:

```text
returncode 4
status failed
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_NOT_VERIFIED
failed_conditions ['component_proof_request_context_proof_semantics_success', 'component_proof_request_context_proof_source_tree_keys_safe', 'component_proof_request_context_proof_cli_command_evidence_shape_valid']
source_tree_identity_success True
source_tree_key_success False
source_tree_current_success True
embedded_source_tree_key_errors ['source_tree_onprem_security_acceptance_unexpected']
file_source_tree_key_errors ['source_tree_onprem_security_acceptance_unexpected']
semantic_error_codes ['CS_VS3_COMPONENT_PROOF_SOURCE_TREE_UNSAFE', 'CS_VS3_COMPONENT_PROOF_CLI_COMMAND_EVIDENCE_INVALID']
component_proof_report_source_tree_failures 1
component_proof_report_source_tree_current_failures 0
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
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_source_tree_extra_key_even_when_all_copies_match
```

Result:

```text
Ran 1 test in 51.987s
OK
```

Adjacent source-tree and proof-boundary coverage:

```bash
PYTHONPATH=packages:. python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_missing_source_tree_even_when_identity_matches \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_stale_source_tree_even_when_identity_matches \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_source_tree_extra_key_even_when_all_copies_match \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_report_proof_boundary_extra_key_even_when_identity_matches
```

Result:

```text
Ran 4 tests in 226.905s
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
component_proof_report_source_tree_failures 0
component_proof_report_source_tree_current_failures 0
component_proof_report_cli_command_evidence_shape_failures 0
vs3_p NOT_CLAIMED
production_onprem NOT_CLAIMED
security_acceptance NOT_CLAIMED
human_acceptance NOT_CLAIMED
```

## Pass / Fail Criteria

PASS:
- Unknown component proof `source_tree` keys fail the local checkpoint.
- File, embedded report, and nested command transcript source-tree copies can still match, proving this is not only a hash-mismatch guard.
- Existing missing-source-tree and stale-source-tree failures remain distinct.
- Existing clean component proof source-tree schemas still pass local checkpoint coverage.
- The final verdict remains no stronger than the evidence.

FAIL:
- A claim-looking `source_tree` key can appear in every nested copy without failing the checkpoint.
- The verifier only fails because file and embedded report bytes diverge.
- Missing source tree, stale source tree, and unsafe source-tree keys collapse into the same error.
- Existing local component proof reports fail only because their current legitimate keys were not included in the taxonomy.

## Verdict

This slice is locally verified for `VS3-GATE-003` / `VS3-GATE-004` / `VS3-REG-005` verifier hardening only.

It does not claim full VS3-L completion, VS3-P readiness, production/on-prem readiness, security acceptance, real IdP readiness, live-provider readiness, migration/restore readiness, or human UX acceptance.
