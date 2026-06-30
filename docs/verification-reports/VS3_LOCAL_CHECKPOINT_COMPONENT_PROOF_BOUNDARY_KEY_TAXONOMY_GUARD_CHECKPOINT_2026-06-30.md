# VS3 Local Checkpoint Component Proof Boundary Key Taxonomy Guard Checkpoint

**Date:** 2026-06-30 KST
**Owner:** JiYong / Tars
**Status:** AI-verifiable verifier hardening slice passed locally.
**Scope:** VS3 local checkpoint component-proof boundary schema hardening only.

## Slice Contract

Goal:
- Make `cornerstone security vs3-local-checkpoint --json` reject a scenario-backed component proof report when the report-level `proof_boundary` contains an unknown key, even if the component proof file and the embedded scenario-report copy still match exactly.

In-slice scenarios:
- `VS3-GATE-004` (`MUST_PASS`): native VS3 verifier/gate metadata must reject unsafe component-proof evidence through the CLI path.
- `VS3-REG-005` (`REGRESSION`): VS3 reports and release metadata must not allow wording or metadata stronger than the actual proof surface.

Out of scope:
- VS3-P readiness.
- Production/on-prem readiness.
- Real IdP, real network, live provider, independent security review, migration/restore, or human UX acceptance.
- New RequestContext, RLS, OPA, egress, ConnectorHub, Tool SDK, or operator UI product behavior.

Human-required boundaries:
- `VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`.
- This slice adds no human approval evidence.

## Current Behavior Reverse-Engineered

Before this slice, `_vs3_component_proof_boundary_errors` rejected missing component proof boundaries, invalid `vs3_l`, `vs3_p` overclaims, known production/live/security/human overclaim keys, and invalid known gate keys.

It did not reject unknown `proof_boundary` keys. A component proof could therefore add a new claim-looking field such as `onprem_security_acceptance=CLAIMED`; as long as the file copy and embedded scenario-report copy matched, the local checkpoint still accepted the proof boundary.

## Change

Updated `packages/cornerstone_cli/main.py`:
- Added an explicit allowed-key taxonomy for scenario-backed component proof `proof_boundary`.
- Preserved existing legitimate component proof keys, including local fixture notes and current human/external gate fields.
- Added `proof_boundary_<key>_unexpected` errors for every key outside the known taxonomy.

Updated `tests/scenario/test_scaffold_cli.py`:
- Added `test_vs3_local_checkpoint_rejects_component_proof_report_proof_boundary_extra_key_even_when_identity_matches`.
- The test tampers `reports/security/vs3-request-context-proof.json` and the embedded `request_context_proof` in `reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json` with `proof_boundary.onprem_security_acceptance=CLAIMED`.
- It refreshes derived human-gate artifacts so identity still matches, then verifies the checkpoint fails at the proof-boundary layer.

## Verification Evidence

Pre-patch tamper probe:

```text
returncode 0
status success
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
request_context_proof_boundary_success True
embedded_errors []
file_errors []
proof_boundary_failures 0
```

Post-patch tamper probe:

```text
returncode 4
status failed
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_NOT_VERIFIED
matches_embedded_current_file True
source_tree_identity_success True
proof_boundary_matches_embedded_file True
request_context_proof_boundary_success False
embedded_errors ['proof_boundary_onprem_security_acceptance_unexpected']
file_errors ['proof_boundary_onprem_security_acceptance_unexpected']
semantic_error_codes ['CS_VS3_COMPONENT_PROOF_BOUNDARY_UNSAFE']
proof_boundary_failures 1
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
PYTHONPATH=packages python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_report_proof_boundary_extra_key_even_when_identity_matches
```

Result:

```text
Ran 1 test in 53.547s
OK
```

Adjacent proof-boundary and clean checkpoint coverage:

```bash
PYTHONPATH=packages python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_manifest_is_hash_backed_and_boundary_safe \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_report_proof_boundary_overclaim_even_when_identity_matches \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_report_proof_boundary_extra_key_even_when_identity_matches
```

Result:

```text
Ran 3 tests in 156.915s
OK
```

## Pass / Fail Criteria

PASS:
- Unknown component proof boundary keys fail the local checkpoint.
- File and embedded component proof identity can still be true, proving this is not only a hash-mismatch guard.
- Existing clean component proof boundary schemas still pass adjacent local-checkpoint coverage.
- The final verdict remains no stronger than the evidence.

FAIL:
- A new claim-looking `proof_boundary` key can appear without failing the checkpoint.
- The verifier only fails because file and embedded report bytes diverge.
- Existing local component proof reports fail only because their current legitimate keys were not included in the taxonomy.

## Verdict

This slice is locally verified for `VS3-GATE-004` / `VS3-REG-005` verifier hardening only.

It does not claim full VS3-L completion, VS3-P readiness, production/on-prem readiness, security acceptance, real IdP readiness, live-provider readiness, migration/restore readiness, or human UX acceptance.
