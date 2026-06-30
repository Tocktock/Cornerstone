# VS3 Local Checkpoint Component Proof CLI Command Evidence Shape Guard Checkpoint

**Date:** 2026-06-29 KST
**Evidence timestamp:** 2026-06-29T10:04:17Z
**Status:** AI-verifiable slice complete
**Scope:** `VS3-GATE-004`, `VS3-REG-004`, `VS3-REG-005`
**Claim boundary:** Local/dev checkpoint guard only. This report does not claim VS3-P, production/on-prem readiness, real IdP readiness, live-provider readiness, security acceptance, migration/restore readiness, or human acceptance.

## Full VS3 Mapping

Current slice:

- `VS3-GATE-004`: native VS3 verifier/checkpoint must require structured `cornerstone ... --json` command transcript evidence for scenario-bearing proof reports.
- `VS3-REG-004`: component proof coverage cannot silently drop or weaken command transcript shape while preserving matching component proof identity.
- `VS3-REG-005`: local checkpoint failures must keep VS3-L, VS3-P, production/on-prem, security acceptance, and human acceptance claims unclaimed.

Later slices:

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

## Slice Contract

Goal:

- Make `cornerstone security vs3-local-checkpoint --json` reject any scenario-bearing VS3 component proof whose CLI evidence is only a malformed or non-replayable transcript shape.

Expected behavior:

- Every scenario-bearing component proof needs at least one valid `command_transcripts` entry in both the embedded scenario report proof and the current proof file.
- A valid entry is an object with list-form `command`, includes `cornerstone`, includes `--json`, has integer `exit_code`, has schema metadata through `json_schema`, `schema_version`, or `cli_schema_version`, and is not marked `timed_out=true`.
- Existing `native_cli_commands` inventories may remain, but they are no longer sufficient as the only command proof for a scenario-bearing component report.
- Missing command evidence and malformed command evidence must be separate machine-readable failures.

Non-scope:

- No RequestContext, RLS, OPA, egress, ConnectorHub, Tool SDK, registry, Agent Pack, live provider, production topology, migration, or human-review implementation.
- No VS3-P, production/on-prem, real IdP, live-provider, migration/restore, security-acceptance, or human-acceptance claim.

## Implementation Summary

Changed `packages/cornerstone_cli/main.py` so component proof identity now records and enforces:

- `embedded_valid_command_transcript_count`
- `file_valid_command_transcript_count`
- `embedded_invalid_command_transcript_entries`
- `file_invalid_command_transcript_entries`
- `cli_command_evidence_shape_success`

New semantic failure code:

- `CS_VS3_COMPONENT_PROOF_CLI_COMMAND_EVIDENCE_INVALID`

New aggregate counter and checkpoint condition family:

- `component_proof_report_cli_command_evidence_shape_failures`
- `component_proof_<proof_key>_cli_command_evidence_shape_valid`

Changed `packages/cornerstone_cli/scenarios.py` so these proof reports now emit structured command transcripts:

- `reports/db/vs3-postgres-rls-proof.json`: 6 command transcripts.
- `reports/policy/vs3-opa-policy-proof.json`: 3 command transcripts.

Updated `tests/scenario/test_scaffold_cli.py` so:

- the positive VS3 local checkpoint test requires valid transcript shape for every scenario-bearing component proof;
- the existing missing-command-evidence test stays scoped to missing evidence;
- a new malformed-command-evidence tamper test proves a bare string transcript fails with `CS_VS3_COMPONENT_PROOF_CLI_COMMAND_EVIDENCE_INVALID`.

## Verification Evidence

Syntax checks:

```text
python3 -m compileall packages/cornerstone_cli
Listing 'packages/cornerstone_cli'...
Compiling 'packages/cornerstone_cli/main.py'...
Compiling 'packages/cornerstone_cli/scenarios.py'...

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
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_missing_cli_command_evidence_even_when_identity_matches \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_malformed_cli_command_evidence_even_when_identity_matches

Ran 10 tests in 279.133s
OK
```

Regenerated reports:

```text
PATH="$PWD:$PATH" cornerstone security vs3-postgres-rls --reuse-vs2-local-range-report reports/security/vs2-local-range.json --json
exit code: 0

PATH="$PWD:$PATH" cornerstone security vs3-opa-policy --reuse-vs2-local-range-report reports/security/vs2-local-range.json --json
exit code: 0

PATH="$PWD:$PATH" cornerstone scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
exit code: 0

PATH="$PWD:$PATH" cornerstone human-gate evidence-status --scope vs3 --record-dir reports/human-gates/vs3/record-templates --use-existing --json --output reports/human-gates/vs3/evidence-status.json
exit code: 0

PATH="$PWD:$PATH" cornerstone human-gate review-kit --scope vs3 --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --use-existing --json --output reports/human-gates/vs3/review-kit.json
exit code: 0

PATH="$PWD:$PATH" cornerstone human-gate vs3-p-gate --scope vs3 --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --output reports/human-gates/vs3/vs3-p-gate.json --json
exit code: 4

PATH="$PWD:$PATH" cornerstone security vs3-local-checkpoint --json --output reports/human-gates/vs3/vs3-local-checkpoint.json
exit code: 0
```

Normal local checkpoint inspection:

```json
{
  "component_proof_report_cli_command_evidence_failures": 0,
  "component_proof_report_cli_command_evidence_shape_failures": 0,
  "component_proof_report_semantic_failures": 0,
  "final_verdict": "VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED",
  "human_acceptance": "NOT_CLAIMED",
  "production_onprem": "NOT_CLAIMED",
  "security_acceptance": "NOT_CLAIMED",
  "status": "success",
  "vs3_l": "LOCAL_DEV_ASSURANCE_VERIFIED",
  "vs3_p": "NOT_CLAIMED"
}
```

Component proof transcript shape counts:

| Proof | Embedded valid transcripts | File valid transcripts | Shape success |
|---|---:|---:|---:|
| `request_context_proof` | 4 | 4 | true |
| `postgres_rls_proof` | 6 | 6 | true |
| `opa_policy_proof` | 3 | 3 | true |
| `egress_sandbox_proof` | 3 | 3 | true |
| `connectorhub_source_proof` | 5 | 5 | true |
| `tool_registry_proof` | 5 | 5 | true |
| `observability_proof` | 6 | 6 | true |
| `final_regression_proof` | 4 | 4 | true |

RLS/OPA report shape inspection:

```text
reports/db/vs3-postgres-rls-proof.json
schema cs.vs3_postgres_rls_proof.v0 status success
command_transcripts 6
native_cli_commands 6

reports/policy/vs3-opa-policy-proof.json
schema cs.vs3_opa_policy_proof.v0 status success
command_transcripts 3
native_cli_commands 3
```

Controlled malformed CLI command evidence tamper:

```json
{
  "checkpoint_rc": 4,
  "command_rcs": {
    "evidence_status": 0,
    "review_kit": 0,
    "vs3_p_gate": 4
  },
  "failed_conditions": [
    "component_proof_request_context_proof_semantics_success",
    "component_proof_request_context_proof_cli_command_evidence_shape_valid"
  ],
  "final_verdict": "VS3_L_LOCAL_DEV_ASSURANCE_NOT_VERIFIED",
  "invalid_entries": [
    {
      "entry_id": "0",
      "errors": [
        "not_object"
      ]
    }
  ],
  "missing_evidence_failures": 0,
  "semantic_error_codes": [
    "CS_VS3_COMPONENT_PROOF_CLI_COMMAND_EVIDENCE_INVALID"
  ],
  "shape_failures": 1,
  "status": "failed"
}
```

Tamper output path:

- `/tmp/vs3-local-checkpoint-malformed-cli-shape.json`

## Pass / Fail Criteria

PASS for this slice requires:

- normal local checkpoint succeeds with `component_proof_report_cli_command_evidence_shape_failures=0`;
- every scenario-bearing component proof has at least one valid structured command transcript in both embedded proof and current file proof;
- RLS and OPA reports no longer rely only on bare command inventories;
- malformed command transcript evidence fails the checkpoint with exit code 4;
- malformed evidence fails the shape condition without being misclassified as missing evidence;
- failure keeps VS3-L, VS3-P, production/on-prem readiness, security acceptance, and human acceptance unclaimed.

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
