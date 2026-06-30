# VS3 Local Checkpoint Self-Command Transcript Claim-Boundary Guard Checkpoint

**Date:** 2026-06-30 KST
**Status:** Local deterministic checkpoint for one VS3 evidence-layout slice.
**Scope:** `cornerstone security vs3-local-checkpoint --json` self-command transcript claim-boundary evidence.
**Proof boundary:** Local/dev verifier evidence only. This checkpoint does not claim VS3-P, production/on-prem readiness, live-provider readiness, real IdP readiness, security acceptance, migration/restore readiness, or human UX acceptance.

## Slice Contract

Goal: make the VS3 local checkpoint self-command transcript preserve the same claim-boundary summary as the checkpoint payload and source scenario report so transcript-only evidence consumers can verify the VS3-L / VS3-P boundary.

In scope:

- `VS3-GATE-003`: local/dev reports must not overclaim production/on-prem, live provider, real IdP, migration/restore, security acceptance, or human acceptance.
- `VS3-GATE-004`: the native `cornerstone security vs3-local-checkpoint --json` transcript must carry machine-readable JSON evidence.
- `VS3-REG-005`: VS3 report/metadata wording and evidence must remain no stronger than the proof surface.

Out of scope:

- VS3-P readiness.
- Real on-prem, real IdP, live provider, independent security review, migration/restore, and human UX acceptance.
- New Tool SDK, signed registry, ConnectorHub live-provider, RLS, OPA, egress, or operator UI feature behavior.

## Current Behavior Found

Before this slice, `reports/security/vs3-local-checkpoint.json` had top-level `scenario_report.claim_boundary`, `scenario_report.claim_boundaries`, and `scenario_report.claim_boundary_validation`, but the self-command transcript `stdout_json` omitted the corresponding claim-boundary fields.

That made top-level checkpoint evidence stronger than transcript-only evidence.

## Change

- `packages/cornerstone_cli/main.py` now emits `claim_boundary`, `claim_boundaries`, `claim_boundary_from_scenario_report`, `claim_boundaries_from_scenario_report`, and nested `scenario_report` claim-boundary validation fields in the local checkpoint transcript `stdout_json`.
- `_vs3_local_checkpoint_self_transcript_validation` now rejects missing or mismatched transcript claim-boundary fields.
- `tests/scenario/test_scaffold_cli.py` now asserts the happy-path transcript fields and a tampered nested `scenario_report.claim_boundary` failure.

## Evidence

Commands run:

```bash
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
```

Result: exit 0.

```bash
python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_manifest_is_hash_backed_and_boundary_safe tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_self_transcript_validator_rejects_tampered_transcript
```

Result: `Ran 2 tests in 117.411s` and `OK`.

```bash
PYTHONPATH=packages python3 -m cornerstone_cli.main security vs3-local-checkpoint --json --output reports/security/vs3-local-checkpoint.json
```

Result: exit 0, refreshed `reports/security/vs3-local-checkpoint.json`.

Bounded JSON probe after refresh:

```text
status success
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
summary_self_command_transcript_shape_failures 0
negative_self_command_transcript_shape_failures 0
validation_status passed
validation_errors []
stdout_claim_boundary_matches True
stdout_claim_boundaries_matches True
stdout_source_claim_boundary_matches True
stdout_source_claim_boundaries_matches True
stdout_scenario_claim_boundary_matches True
stdout_scenario_claim_boundaries_matches True
stdout_scenario_claim_validation_matches True
stdout_proof_boundary_vs3_l LOCAL_COMPONENT_PROOF_ONLY
stdout_proof_boundary_vs3_p NOT_CLAIMED
scenario_report_claim_boundary_validation passed
```

## Human Required

`VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`. This slice prepares and validates local transcript evidence only; it does not provide human/on-prem proof.

## Verdict

AI-verifiable slice: PASS for the bounded transcript claim-boundary guard.

VS3 milestone: still in progress. Full VS3-L requires all AI-owned VS3 `MUST_PASS` and `REGRESSION` rows to pass on the same source tree with concrete evidence. VS3-P remains blocked on signed human/on-prem evidence.
