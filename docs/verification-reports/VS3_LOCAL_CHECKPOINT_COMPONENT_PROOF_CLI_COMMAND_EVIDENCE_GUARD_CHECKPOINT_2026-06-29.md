# VS3 Local Checkpoint Component Proof CLI Command Evidence Guard Checkpoint

**Date:** 2026-06-29 KST
**Status:** AI-verifiable slice complete
**Scope:** `VS3-GATE-004`, `VS3-REG-004`, `VS3-REG-005`
**Claim boundary:** Local/dev checkpoint guard only. This report does not claim VS3-P, production/on-prem readiness, real IdP readiness, live-provider readiness, security acceptance, migration/restore readiness, or human acceptance.

## Slice Contract

Goal:

- Make `cornerstone security vs3-local-checkpoint --json` reject any scenario-bearing VS3 component proof that lacks CLI command evidence in both the embedded scenario report proof and the current proof file.

In this slice:

- `VS3-GATE-004`: the native VS3 verifier/checkpoint must enforce CLI command evidence as part of the JSON proof surface.
- `VS3-REG-004`: component proof coverage cannot silently drop command evidence while preserving matching hashes.
- `VS3-REG-005`: failure paths must keep `VS3-L` and `VS3-P` unclaimed.

Later slice:

- `VS3-GATE-001`, `VS3-GATE-002`, `VS3-GATE-003`
- `VS3-CTX-001` through `VS3-CTX-005`
- `VS3-RLS-001` through `VS3-RLS-006`
- `VS3-OPA-001` through `VS3-OPA-005`
- `VS3-EGR-001` through `VS3-EGR-006`
- `VS3-CON-001` through `VS3-CON-006`
- `VS3-TOOL-001` through `VS3-TOOL-007`
- `VS3-OBS-001` through `VS3-OBS-003`
- `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-003`, `VS3-REG-006`, `VS3-REG-007`, `VS3-REG-008`

Human-required:

- `VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`.

Non-scope:

- No RequestContext, RLS, OPA, egress, ConnectorHub, Tool SDK, registry, Agent Pack, live provider, production topology, migration, or human-review implementation.
- No broad VS3-L completion claim beyond the current generated local checkpoint state.
- No VS3-P or security/human acceptance claim.

## Implementation Summary

Changed `packages/cornerstone_cli/main.py` so each scenario-bearing component proof identity now records and enforces:

- `embedded_command_transcript_count`
- `file_command_transcript_count`
- `embedded_native_cli_command_count`
- `file_native_cli_command_count`
- `embedded_cli_command_evidence_present`
- `file_cli_command_evidence_present`
- `cli_command_evidence_success`

The verifier accepts either `command_transcripts` or `native_cli_commands` as CLI command evidence. `evidence_reconciliation` remains exempt because it is not a per-scenario component proof and has `expects_scenario_checks=false`.

New semantic failure code:

- `CS_VS3_COMPONENT_PROOF_CLI_COMMAND_EVIDENCE_MISSING`

New aggregate counters:

- `component_proof_report_cli_command_evidence_failures`
- `component_proof_<proof_key>_cli_command_evidence_present`

Updated `tests/scenario/test_scaffold_cli.py` to require the new fields in the positive checkpoint test and to add an aligned tamper regression where `request_context_proof.command_transcripts` and `request_context_proof.native_cli_commands` are emptied in both the component proof file and embedded scenario report proof.

## Verification Evidence

Syntax checks:

```text
python3 -m compileall packages/cornerstone_cli
Listing 'packages/cornerstone_cli'...
Compiling 'packages/cornerstone_cli/main.py'...

python3 -m py_compile tests/scenario/test_scaffold_cli.py
exit code: 0
```

Focused regression tests:

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_manifest_is_hash_backed_and_boundary_safe \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_stale_component_proof_file \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_status_failure_even_when_identity_matches \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_inner_status_failure_even_when_identity_matches \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_missing_scenario_row_even_when_identity_matches \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_missing_check_even_when_identity_matches \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_nonzero_negative_evidence_even_when_identity_matches \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_missing_refs_even_when_identity_matches \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_missing_cli_command_evidence_even_when_identity_matches

Ran 9 tests in 239.977s
OK
```

Regenerated local checkpoint:

```text
PATH="$PWD:$PATH" cornerstone security vs3-local-checkpoint --json --output reports/human-gates/vs3/vs3-local-checkpoint.json
exit code: 0
```

Checkpoint inspection:

```text
status success
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
errors_count 0
component_proof_report_cli_command_evidence_failures 0
vs3_l_claim LOCAL_DEV_ASSURANCE_VERIFIED
vs3_p_claim NOT_CLAIMED
```

Current component proof CLI evidence counts:

| Proof | `expects_scenario_checks` | CLI evidence success | `command_transcripts` embedded/file | `native_cli_commands` embedded/file |
|---|---:|---:|---:|---:|
| `evidence_reconciliation` | false | true | 0 / 0 | 0 / 0 |
| `request_context_proof` | true | true | 4 / 4 | 0 / 0 |
| `postgres_rls_proof` | true | true | 0 / 0 | 6 / 6 |
| `opa_policy_proof` | true | true | 0 / 0 | 3 / 3 |
| `egress_sandbox_proof` | true | true | 3 / 3 | 3 / 3 |
| `connectorhub_source_proof` | true | true | 5 / 5 | 5 / 5 |
| `tool_registry_proof` | true | true | 5 / 5 | 9 / 9 |
| `observability_proof` | true | true | 6 / 6 | 0 / 0 |
| `final_regression_proof` | true | true | 4 / 4 | 0 / 0 |

Controlled aligned CLI command evidence tamper after the fix:

```text
dep_rc [0, 0, 4]
checkpoint_rc 4
checkpoint_status failed
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_NOT_VERIFIED
failed_conditions ['component_proof_request_context_proof_semantics_success', 'component_proof_request_context_proof_cli_command_evidence_present']
identity_subset {"checks_success": true, "cli_command_evidence_success": false, "embedded_cli_command_evidence_present": false, "embedded_command_transcript_count": 0, "embedded_native_cli_command_count": 0, "file_cli_command_evidence_present": false, "file_command_transcript_count": 0, "file_native_cli_command_count": 0, "matches_embedded_current_file": true, "negative_evidence_success": true, "references_success": true, "scenario_status_success": true, "semantic_error_codes": ["CS_VS3_COMPONENT_PROOF_CLI_COMMAND_EVIDENCE_MISSING"], "semantics_success": false}
summary_subset {"component_proof_report_check_failures": 0, "component_proof_report_cli_command_evidence_failures": 1, "component_proof_report_mismatches": 0, "component_proof_report_negative_evidence_failures": 0, "component_proof_report_reference_failures": 0, "component_proof_report_scenario_failures": 0, "component_proof_report_semantic_failures": 1, "vs3_l_claim": "NOT_CLAIMED", "vs3_p_claim": "NOT_CLAIMED"}
negative_subset {"component_proof_report_check_failures": 0, "component_proof_report_cli_command_evidence_failures": 1, "component_proof_report_mismatches": 0, "component_proof_report_missing_or_invalid": 0, "component_proof_report_negative_evidence_failures": 0, "component_proof_report_reference_failures": 0, "component_proof_report_scenario_failures": 0, "component_proof_report_semantic_failures": 1, "human_acceptance_claimed_by_checkpoint": 0, "production_readiness_claimed_by_checkpoint": 0, "security_acceptance_claimed_by_checkpoint": 0, "vs3_p_claimed_by_checkpoint": 0}
```

## Pass / Fail Criteria

PASS for this slice requires:

- normal local checkpoint succeeds with `component_proof_report_cli_command_evidence_failures=0`;
- every scenario-bearing component proof has either `command_transcripts` or `native_cli_commands` in both embedded and current file proof;
- `evidence_reconciliation` remains exempt and does not need synthetic CLI command evidence;
- aligned removal of command evidence fails the checkpoint with exit code 4;
- failure is attributed to CLI command evidence, not stale file identity, scenario status, check coverage, negative evidence, or evidence/audit refs;
- failure keeps `VS3-L` and `VS3-P` unclaimed.

Observed result: PASS for this slice.

## Remaining Human Gates

Unchanged:

- `VS3-H01`: architecture/security/dependency/migration owner approval.
- `VS3-H02`: independent security review and retest.
- `VS3-H03`: real IdP mapping and revocation evidence.
- `VS3-H04`: real topology egress/network evidence.
- `VS3-H05`: approved live-provider rehearsal.
- `VS3-H06`: human operator UX/trust review.
- `VS3-H07`: human-supervised migration, backup, restore, rollback, quarantine, RLS, policy, and audit drill.

These remain `HUMAN_REQUIRED` and continue to block VS3-P.
