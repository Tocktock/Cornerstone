# VS3 Local Checkpoint Component Proof CLI Command Timestamp Guard Checkpoint

**Date:** 2026-06-29 KST
**Evidence timestamp:** 2026-06-29T10:58:42Z
**Status:** AI-verifiable slice complete
**Scope:** `VS3-GATE-004`, `VS3-REG-004`, `VS3-REG-005`
**Claim boundary:** Local/dev checkpoint guard only. This report does not claim VS3-P, production/on-prem readiness, real IdP readiness, live-provider readiness, security acceptance, migration/restore readiness, or human acceptance.

## Full VS3 Mapping

Current slice:

- `VS3-GATE-004`: native VS3 verifier/checkpoint must require scenario-bearing component proof command transcripts to include valid UTC command chronology.
- `VS3-REG-004`: component proof coverage cannot silently accept malformed or reversed transcript timestamps while preserving matching component proof identity.
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

- Make `cornerstone security vs3-local-checkpoint --json` reject scenario-bearing VS3 component proof transcripts that are otherwise well-shaped but have invalid or time-reversed command chronology.

Expected behavior:

- `started_at` must be a parseable UTC timestamp ending in `Z`.
- `ended_at` must be a parseable UTC timestamp ending in `Z`.
- `ended_at` must be greater than or equal to `started_at`.
- Missing timestamps remain classified as `started_at_missing` or `ended_at_missing`.
- Malformed timestamps and reversed timestamps remain separate machine-readable shape failures under the component proof CLI evidence guard.

Non-scope:

- No wall-clock attestation beyond local report chronology validation.
- No production shell, CI, staging, live-provider, or external timestamp authority.
- No production/on-prem, live-provider, migration/restore, security-acceptance, or human-acceptance claim.

## Implementation Summary

Changed `packages/cornerstone_cli/main.py`:

- Added `_vs3_parse_utc_z_timestamp`.
- `_vs3_cli_command_transcript_errors` now rejects malformed `started_at` with `started_at_invalid`.
- `_vs3_cli_command_transcript_errors` now rejects malformed `ended_at` with `ended_at_invalid`.
- `_vs3_cli_command_transcript_errors` now rejects reversed chronology with `ended_at_before_started_at`.

Updated `tests/scenario/test_scaffold_cli.py`:

- Added `test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_invalid_timestamp`.
- Added `test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_reversed_timestamp`.

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
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_reversed_timestamp

Ran 5 tests in 133.447s
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

Regenerated transcript timestamp inspection:

```text
reports/security/vs3-request-context-proof.json rows 4 bad_timestamps []
reports/db/vs3-postgres-rls-proof.json rows 6 bad_timestamps []
reports/policy/vs3-opa-policy-proof.json rows 3 bad_timestamps []
reports/security/vs3-egress-sandbox-proof.json rows 3 bad_timestamps []
reports/security/vs3-connectorhub-source-proof.json rows 5 bad_timestamps []
reports/security/vs3-tool-registry-proof.json rows 5 bad_timestamps []
reports/observability/vs3-observability-proof.json rows 6 bad_timestamps []
reports/security/vs3-final-regression-proof.json rows 4 bad_timestamps []
```

Controlled invalid timestamp tamper:

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
        "started_at_invalid"
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

Controlled reversed timestamp tamper:

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
        "ended_at_before_started_at"
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

- `/tmp/vs3-timestamp-tamper.json`

## Pass / Fail Criteria

PASS for this slice requires:

- normal local checkpoint succeeds with `component_proof_report_cli_command_evidence_shape_failures=0`;
- every scenario-bearing component proof transcript has parseable `started_at` and `ended_at` UTC `Z` timestamps;
- no regenerated component transcript has `ended_at < started_at`;
- malformed timestamp evidence fails the checkpoint with exit code 4 and `started_at_invalid`;
- reversed timestamp evidence fails the checkpoint with exit code 4 and `ended_at_before_started_at`;
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
