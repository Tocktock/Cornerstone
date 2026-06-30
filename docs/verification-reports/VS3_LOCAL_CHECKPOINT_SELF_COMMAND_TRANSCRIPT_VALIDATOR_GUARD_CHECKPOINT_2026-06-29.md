# VS3 Local Checkpoint Self Command Transcript Validator Guard Checkpoint - 2026-06-29

## Purpose

Strengthen the VS3 local checkpoint so its own native CLI transcript is not only present in `command_transcripts`, but is deterministically validated against the checkpoint payload.

This checkpoint is a local/dev assurance guard only. It does not claim VS3-P, production/on-prem readiness, live provider readiness, real IdP/network readiness, migration restore readiness, security acceptance, or human acceptance.

## Slice Contract

- Goal: make `cornerstone security vs3-local-checkpoint --json` validate its own self command transcript and expose the validation result as checkpoint condition, summary, and negative evidence.
- In scope: deterministic validation for the checkpoint self transcript command, arguments, exit code, schema versions, scope, evidence refs, audit refs, policy decision refs, ref summary, stdout JSON verdict, stdout JSON scope, and source-tree shape.
- Out of scope: new VS3 production/on-prem proof, live provider proof, real IdP proof, real network proof, migration restore proof, and human approval evidence.
- Claim boundary: `VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED` remains the maximum successful local checkpoint verdict.

## Implementation Evidence

- Added `_vs3_local_checkpoint_self_transcript_validation(...)` in `packages/cornerstone_cli/main.py`.
- Wired the validation into `command_security_vs3_local_checkpoint(...)` as:
  - `payload["self_command_transcript_validation"]`
  - `checkpoint_conditions["self_command_transcript_shape_valid"]`
  - `summary.self_command_transcript_shape_failures`
  - `negative_evidence.self_command_transcript_shape_failures`
- Added scenario tests for the happy path and two tamper cases:
  - missing self transcript `source_tree`
  - mismatched self transcript `ref_summary.evidence_refs_count`

## Before Evidence

Command:

```bash
cornerstone security vs3-local-checkpoint --json
```

Captured in `/tmp/vs3-local-checkpoint-self-transcript-validator-before.json`.

Observed output:

```text
returncode 0
status success
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
has_self_validation False
has_condition False
summary_key None
negative_key None
saved /tmp/vs3-local-checkpoint-self-transcript-validator-before.json
```

Pass/fail implication: the prior checkpoint could assert transcript shape through tests, but the checkpoint JSON did not carry a first-class self validation result. This slice was required.

## After Evidence

Command:

```bash
cornerstone security vs3-local-checkpoint --json
```

Captured in `/tmp/vs3-local-checkpoint-self-transcript-validator-after.json`.

Observed output:

```text
returncode 0
status success
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
validation_status passed
validation_valid True
validation_errors []
condition True
summary_failures 0
negative_failures 0
saved /tmp/vs3-local-checkpoint-self-transcript-validator-after.json
```

Pass/fail criteria:

- PASS if the checkpoint exposes `self_command_transcript_validation.status == "passed"`.
- PASS if `checkpoint_conditions.self_command_transcript_shape_valid == true`.
- PASS if summary and negative evidence report zero self transcript shape failures.
- FAIL if the validator detects command, scope, ref, stdout JSON, source tree, or schema mismatches and the checkpoint still claims VS3-L success.

## Verification Commands

```bash
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
```

Result: exit code 0.

```bash
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_manifest_is_hash_backed_and_boundary_safe \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_self_transcript_validator_rejects_tampered_transcript
```

Result:

```text
Ran 2 tests in 52.618s

OK
```

```bash
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_secret_leak \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_report_secret_leak
```

Result:

```text
Ran 2 tests in 53.802s

OK
```

## MUST PASS Scenario Mapping

The full VS3 matrix has 57 rows: 42 `MUST_PASS`, 8 `REGRESSION`, and 7 `HUMAN_REQUIRED`. This slice changes only the local checkpoint self transcript guard. All other rows remain mapped, not silently promoted.

| Scenario | Priority | Slice Status | Verification in This Slice |
|---|---|---|---|
| VS3-GATE-001 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-GATE-002 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-GATE-003 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-GATE-004 | MUST_PASS | IN_THIS_SLICE | self command transcript validator and checkpoint output |
| VS3-CTX-001 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-CTX-002 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-CTX-003 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-CTX-004 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-CTX-005 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-RLS-001 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-RLS-002 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-RLS-003 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-RLS-004 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-RLS-005 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-RLS-006 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-OPA-001 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-OPA-002 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-OPA-003 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-OPA-004 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-OPA-005 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-EGR-001 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-EGR-002 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-EGR-003 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-EGR-004 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-EGR-005 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-EGR-006 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-CON-001 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-CON-002 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-CON-003 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-CON-004 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-CON-005 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-CON-006 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-TOOL-001 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-TOOL-002 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-TOOL-003 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-TOOL-004 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-TOOL-005 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-TOOL-006 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-TOOL-007 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-OBS-001 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-OBS-002 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-OBS-003 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-REG-001 | REGRESSION | LATER_SLICE | unchanged by this slice |
| VS3-REG-002 | REGRESSION | LATER_SLICE | unchanged by this slice |
| VS3-REG-003 | REGRESSION | LATER_SLICE | unchanged by this slice |
| VS3-REG-004 | REGRESSION | IN_THIS_SLICE | self command transcript validator and checkpoint output |
| VS3-REG-005 | REGRESSION | IN_THIS_SLICE | self command transcript validator and checkpoint output |
| VS3-REG-006 | REGRESSION | LATER_SLICE | unchanged by this slice |
| VS3-REG-007 | REGRESSION | IN_THIS_SLICE | self command transcript validator and checkpoint output |
| VS3-REG-008 | REGRESSION | LATER_SLICE | unchanged by this slice |
| VS3-H01 | HUMAN_REQUIRED | HUMAN_REQUIRED | human gate evidence remains unpromoted |
| VS3-H02 | HUMAN_REQUIRED | HUMAN_REQUIRED | human gate evidence remains unpromoted |
| VS3-H03 | HUMAN_REQUIRED | HUMAN_REQUIRED | human gate evidence remains unpromoted |
| VS3-H04 | HUMAN_REQUIRED | HUMAN_REQUIRED | human gate evidence remains unpromoted |
| VS3-H05 | HUMAN_REQUIRED | HUMAN_REQUIRED | human gate evidence remains unpromoted |
| VS3-H06 | HUMAN_REQUIRED | HUMAN_REQUIRED | human gate evidence remains unpromoted |
| VS3-H07 | HUMAN_REQUIRED | HUMAN_REQUIRED | human gate evidence remains unpromoted |

## Proof Boundary

- `VS3-GATE-004`, `VS3-REG-004`, `VS3-REG-005`, and `VS3-REG-007` receive additional local checkpoint evidence from this slice.
- No human-required scenario is marked `PASS`.
- No production, live provider, real IdP, real network, migration restore, security acceptance, or human acceptance claim is made.
- The worktree was dirty before this slice; unrelated pre-existing VS3 files and reports remain outside this checkpoint's implementation claim.
