# VS3 Local Checkpoint Component Proof CLI Command Semantics Guard Checkpoint

**Date:** 2026-06-29 KST
**Evidence timestamp:** 2026-06-29T10:49:11Z
**Status:** AI-verifiable slice complete
**Scope:** `VS3-GATE-004`, `VS3-REG-004`, `VS3-REG-005`
**Claim boundary:** Local/dev checkpoint guard only. This report does not claim VS3-P, production/on-prem readiness, real IdP readiness, live-provider readiness, security acceptance, migration/restore readiness, or human acceptance.

## Full VS3 Mapping

Current slice:

- `VS3-GATE-004`: native VS3 verifier/checkpoint must require scenario-bearing component proof command transcripts to prove a native `cornerstone ... --json` command path.
- `VS3-REG-004`: component proof coverage cannot silently accept a wrapper command or mismatched arguments while preserving matching component proof identity.
- `VS3-REG-005`: failure paths must keep VS3-L, VS3-P, production/on-prem, security acceptance, and human acceptance claims unclaimed.

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

- Make `cornerstone security vs3-local-checkpoint --json` reject scenario-bearing VS3 component proof transcripts that are well-shaped but do not prove a native `cornerstone ... --json` command.

Expected behavior:

- Every scenario-bearing component proof transcript `command` must be a string list whose first executable is exactly `cornerstone`.
- Every scenario-bearing component proof transcript must include `--json` in the command list.
- The transcript `arguments` list must exactly equal the command tail after `cornerstone`.
- Runtime environment overrides may be recorded separately as metadata, but must not turn the recorded command into a wrapper invocation.
- Wrapper command evidence and mismatched arguments remain separate machine-readable shape failures under the component proof CLI evidence guard.

Non-scope:

- No production shell, CI, staging, or live-provider command replay.
- No claim that real IdP, real tenant membership, or production workspace mapping has been verified.
- No production/on-prem, live-provider, migration/restore, security-acceptance, or human-acceptance claim.

## Implementation Summary

Changed `packages/cornerstone_cli/main.py`:

- `_vs3_cli_command_transcript_errors` now rejects wrapper command transcripts with `command_not_native_cornerstone`.
- It also rejects transcript argument drift with `arguments_mismatch_command_tail`.

Changed `packages/cornerstone_cli/scenarios.py`:

- `_run_command` accepts safe `env_overrides` metadata and applies those overrides to the subprocess environment.
- VS3 final-regression proof now records native `cornerstone scenario verify ... --json` commands for `vs0_evux` and `vs1_ontology_suggest_promote`.
- The `CORNERSTONE_SKIP_VS2_REGRESSION_TESTS=1` recursion guard is preserved as `env_overrides`, not as an `env ... cornerstone ...` wrapper command.

Updated `tests/scenario/test_scaffold_cli.py`:

- Added `test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_wrapped_command`.
- Added `test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_argument_mismatch`.

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
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_missing_metadata \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_missing_scope \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_wrapped_command \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_argument_mismatch

Ran 5 tests in 133.708s
OK
```

Regenerated component proof reports:

```text
PATH="$PWD:$PATH" cornerstone security vs3-request-context --json
exit code: 0

PATH="$PWD:$PATH" cornerstone security vs3-postgres-rls --reuse-vs2-local-range-report reports/security/vs2-local-range.json --json
exit code: 0

PATH="$PWD:$PATH" cornerstone security vs3-opa-policy --reuse-vs2-local-range-report reports/security/vs2-local-range.json --json
exit code: 0

PATH="$PWD:$PATH" cornerstone security vs3-egress-sandbox --reuse-vs2-local-range-report reports/security/vs2-local-range.json --json
exit code: 0

PATH="$PWD:$PATH" cornerstone security vs3-connectorhub-source --json
exit code: 0

PATH="$PWD:$PATH" cornerstone security vs3-tool-registry --json
exit code: 0

PATH="$PWD:$PATH" cornerstone security vs3-observability --json
exit code: 0

PATH="$PWD:$PATH" cornerstone security vs3-regression-gate --json
exit code: 0
```

Regenerated scenario/human-gate reports:

```text
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
  "claim_boundary": {
    "human_acceptance": "NOT_CLAIMED",
    "production_onprem": "NOT_CLAIMED",
    "security_acceptance": "NOT_CLAIMED",
    "vs3_l": "LOCAL_DEV_ASSURANCE_VERIFIED",
    "vs3_p": "NOT_CLAIMED"
  },
  "final_verdict": "VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED",
  "missing_failures": 0,
  "semantic_failures": 0,
  "shape_failures": 0,
  "status": "success"
}
```

Regenerated transcript command inspection:

```text
reports/security/vs3-request-context-proof.json rows 4 bad_native_entries [] env_overrides []
reports/db/vs3-postgres-rls-proof.json rows 6 bad_native_entries [] env_overrides []
reports/policy/vs3-opa-policy-proof.json rows 3 bad_native_entries [] env_overrides []
reports/security/vs3-egress-sandbox-proof.json rows 3 bad_native_entries [] env_overrides []
reports/security/vs3-connectorhub-source-proof.json rows 5 bad_native_entries [] env_overrides []
reports/security/vs3-tool-registry-proof.json rows 5 bad_native_entries [] env_overrides []
reports/observability/vs3-observability-proof.json rows 6 bad_native_entries [] env_overrides []
reports/security/vs3-final-regression-proof.json rows 4 bad_native_entries [] env_overrides [{'entry': 'vs0_evux', 'env_overrides': {'CORNERSTONE_SKIP_VS2_REGRESSION_TESTS': '1'}}, {'entry': 'vs1_ontology_suggest_promote', 'env_overrides': {'CORNERSTONE_SKIP_VS2_REGRESSION_TESTS': '1'}}]
```

Controlled wrapped-command tamper:

```json
{
  "checkpoint_rc": 4,
  "failed_conditions": [
    "component_proof_request_context_proof_semantics_success",
    "component_proof_request_context_proof_cli_command_evidence_shape_valid"
  ],
  "final_verdict": "VS3_L_LOCAL_DEV_ASSURANCE_NOT_VERIFIED",
  "invalid_entries": [
    {
      "entry_id": "0",
      "errors": [
        "command_not_native_cornerstone"
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

Controlled argument-mismatch tamper:

```json
{
  "checkpoint_rc": 4,
  "failed_conditions": [
    "component_proof_request_context_proof_semantics_success",
    "component_proof_request_context_proof_cli_command_evidence_shape_valid"
  ],
  "final_verdict": "VS3_L_LOCAL_DEV_ASSURANCE_NOT_VERIFIED",
  "invalid_entries": [
    {
      "entry_id": "0",
      "errors": [
        "arguments_mismatch_command_tail"
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

- `/tmp/vs3-command-semantics-tamper.json`

## Pass / Fail Criteria

PASS for this slice requires:

- normal local checkpoint succeeds with `component_proof_report_cli_command_evidence_shape_failures=0`;
- every scenario-bearing component proof transcript records a native `cornerstone ... --json` command;
- transcript `arguments` exactly match the command tail after `cornerstone`;
- wrapper command evidence fails the checkpoint with exit code 4 and `command_not_native_cornerstone`;
- argument drift fails the checkpoint with exit code 4 and `arguments_mismatch_command_tail`;
- failure keeps VS3-L, VS3-P, production/on-prem readiness, security acceptance, and human acceptance unclaimed.

Observed result: PASS for this slice.

## Remaining Human Gates

- `VS3-H01`: architecture/security approval.
- `VS3-H02`: independent security review.
- `VS3-H03`: real IdP mapping proof.
- `VS3-H04`: real on-prem network control proof.
- `VS3-H05`: live ConnectorHub/provider rehearsal.
- `VS3-H06`: human operator UX/trust review.
- `VS3-H07`: migration/backup/restore drill.
