# VS3 Local Checkpoint Component Proof CLI Command Exit-Code Guard Checkpoint

**Date:** 2026-06-29 KST
**Evidence timestamp:** 2026-06-29T11:08:05Z
**Status:** AI-verifiable slice complete
**Scope:** `VS3-GATE-004`, `VS3-REG-004`, `VS3-REG-005`
**Claim boundary:** Local/dev checkpoint guard only. This report does not claim VS3-P, production/on-prem readiness, real IdP readiness, live-provider readiness, security acceptance, migration/restore readiness, or human acceptance.

## Full VS3 Mapping

Current slice:

- `VS3-GATE-004`: native VS3 verifier/checkpoint must require scenario-bearing component proof command transcripts to use the documented CornerStone CLI exit-code baseline.
- `VS3-REG-004`: component proof coverage cannot silently accept non-contract exit codes while preserving matching component proof identity.
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

- Make `cornerstone security vs3-local-checkpoint --json` reject scenario-bearing VS3 component proof transcripts that carry exit codes outside the documented CornerStone CLI baseline.

Expected behavior:

- Transcript `exit_code` must be an integer, not a boolean.
- Transcript `exit_code` must be one of the documented CLI baseline values `0` through `8`.
- Intentional non-zero local proof exits remain valid when they are within the contract, including policy/permission denial and tamper-detection failure evidence.
- Non-contract exit codes remain a machine-readable component proof CLI evidence shape failure.

Non-scope:

- No claim that every transcript must exit `0`.
- No per-command semantic replay beyond local component proof validation.
- No production/on-prem, live-provider, migration/restore, security-acceptance, or human-acceptance claim.

## Implementation Summary

Changed `packages/cornerstone_cli/main.py`:

- Added `VS3_CLI_TRANSCRIPT_EXIT_CODES = set(range(0, 9))`.
- `_vs3_cli_command_transcript_errors` now rejects boolean exit codes as `exit_code_not_int`.
- `_vs3_cli_command_transcript_errors` now rejects non-contract integer exit codes as `exit_code_outside_contract`.

Updated `tests/scenario/test_scaffold_cli.py`:

- Added `test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_exit_code_outside_contract`.

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
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_wrapped_command \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_argument_mismatch \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_invalid_timestamp \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_reversed_timestamp \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_exit_code_outside_contract

Ran 6 tests in 160.741s
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

Regenerated transcript exit-code inspection:

```text
reports/security/vs3-request-context-proof.json exit_codes [0, 0, 2, 0] bad_exit_codes []
reports/db/vs3-postgres-rls-proof.json exit_codes [0, 0, 0, 0, 0, 0] bad_exit_codes []
reports/policy/vs3-opa-policy-proof.json exit_codes [0, 0, 0] bad_exit_codes []
reports/security/vs3-egress-sandbox-proof.json exit_codes [0, 0, 0] bad_exit_codes []
reports/security/vs3-connectorhub-source-proof.json exit_codes [0, 0, 0, 0, 0] bad_exit_codes []
reports/security/vs3-tool-registry-proof.json exit_codes [0, 0, 0, 0, 0] bad_exit_codes []
reports/observability/vs3-observability-proof.json exit_codes [0, 0, 0, 5, 0, 0] bad_exit_codes []
reports/security/vs3-final-regression-proof.json exit_codes [0, 0, 0, 0] bad_exit_codes []
```

Controlled non-contract exit-code tamper:

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
        "exit_code_outside_contract"
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

- `/tmp/vs3-exit-code-tamper.json`

## Pass / Fail Criteria

PASS for this slice requires:

- normal local checkpoint succeeds with `component_proof_report_cli_command_evidence_shape_failures=0`;
- regenerated scenario-bearing component proof transcripts use only CLI baseline exit codes `0` through `8`;
- intentional non-zero proof exits within the documented contract remain valid;
- non-contract exit-code evidence fails the checkpoint with exit code 4 and `exit_code_outside_contract`;
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
