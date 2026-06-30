# VS3 Scenario Gate Boundary Key Taxonomy Guard Checkpoint - 2026-06-30

**Status:** Local deterministic checkpoint for one VS3 verifier hardening slice.
**Scope:** Native VS3 scenario gate proof-boundary and claim-boundary key taxonomy.
**Related rows:** `VS3-GATE-003`, `VS3-GATE-004`, proof-boundary support for `VS3-REG-005`.
**Proof boundary:** Local CLI/test evidence only. This is not VS3-L completion, VS3-P readiness, production/on-prem readiness, security acceptance, live-provider readiness, migration/restore readiness, or human acceptance.

## Full VS3 Mapping

Full VS3 inventory remains:

- `42` `MUST_PASS`
- `8` `REGRESSION`
- `7` `HUMAN_REQUIRED`
- `57` total rows

Execution classification for this slice:

- In this slice: `VS3-GATE-003`, `VS3-GATE-004`, proof-boundary support for `VS3-REG-005`.
- Later slice: all other AI-owned `VS3-*` `MUST_PASS` and `REGRESSION` rows.
- Human-required: `VS3-H01` through `VS3-H07`.

## Slice Contract

Goal:

- Make `cornerstone scenario gate <report> --json` reject a VS3 local-dev assurance report when top-level `proof_boundary` or `claim_boundaries` contains an unrecognized readiness/acceptance key, even if all known boundary fields remain safe.

In scope:

- Exact allowed-key taxonomy for top-level `proof_boundary`.
- Exact allowed-key taxonomy for top-level `claim_boundaries`.
- Native JSON gate failure under `claim_boundary_validation`.
- Focused regression coverage and direct tamper probe.

Out of scope:

- RequestContext, RLS, OPA, egress, ConnectorHub live/provider flows, Tool SDK, signed registry, Agent Pack activation, and all VS3 human evidence gates.

Expected behavior:

- Clean VS3 boundary objects contain only known local-dev proof-boundary keys.
- Adding `proof_boundary.onprem_security_acceptance = CLAIMED` fails the gate with `CS_VS3_PROOF_BOUNDARY_OVERCLAIM_INVALID`.
- Adding `claim_boundaries.onprem_security_acceptance = CLAIMED` fails the gate with `CS_VS3_PROOF_BOUNDARY_OVERCLAIM_INVALID`.
- The failure is attributed to `claim_boundary_validation.unexpected_fields` while coverage, traceability, human-required, row-ref, aggregate-ref, and source-transcript validation remain passable.

## Before Evidence

Before this patch, the same-path tamper probe passed for unrecognized readiness keys:

```text
claim_extra exit 0 status success errors []
claim_boundary_validation ... status passed
proof_extra exit 0 status success errors []
claim_boundary_validation ... status passed
both_extra exit 0 status success errors []
claim_boundary_validation ... status passed
```

Interpretation:

- The gate checked the known forbidden proof/claim boundary fields.
- The gate did not reject new local-dev report fields that could carry additional readiness or acceptance claims.

## Change Summary

Changed:

- `packages/cornerstone_cli/main.py`
  - Adds exact allowed-key sets for `proof_boundary` and `claim_boundaries`.
  - Adds `claim_boundary_validation.allowed_proof_boundary_keys`.
  - Adds `claim_boundary_validation.allowed_claim_boundary_keys`.
  - Adds `claim_boundary_validation.unexpected_fields`.
  - Fails `claim_boundary_validation` when either boundary object contains an unrecognized key.
- `tests/scenario/test_scaffold_cli.py`
  - Adds `test_vs3_scenario_gate_rejects_local_dev_claim_with_unexpected_boundary_key`.
  - Verifies both `proof_boundary.onprem_security_acceptance` and `claim_boundaries.onprem_security_acceptance` fail the gate.
  - Asserts unrelated human-required, coverage, traceability, row-ref, aggregate-ref, and source-transcript validators remain green.

## Verification Evidence

Compile:

```text
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
exit 0
```

Focused regression:

```text
PYTHONPATH=packages python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_unexpected_boundary_key
.
----------------------------------------------------------------------
Ran 1 test in 25.678s

OK
```

Adjacent claim-boundary regression suite:

```text
PYTHONPATH=packages python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_promoted_to_vs3_p_readiness \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_boundary_surface_or_vs2_status_drift \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_unexpected_boundary_key \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_success_report_with_unrecognized_local_dev_claim
....
----------------------------------------------------------------------
Ran 4 tests in 111.012s

OK
```

Fresh direct tamper probe after refreshing `reports/security/vs3-final-regression-proof.json`:

```text
verify_exit 0
verify_status success
verify_summary {'blocking': 0, 'fail': 0, 'human_required': 7, 'not_run': 0, 'not_verified': 0, 'pass': 50, 'product_feature_claims': 'VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_GATES_REQUIRED', 'scenario_count': 57}
proof_boundary gate_exit 4 gate_status failed error_codes ['CS_VS3_PROOF_BOUNDARY_OVERCLAIM_INVALID']
claim_boundary_validation ... 'invalid_fields': ['proof_boundary.onprem_security_acceptance'] ... 'unexpected_fields': ['proof_boundary.onprem_security_acceptance']
claim_boundaries gate_exit 4 gate_status failed error_codes ['CS_VS3_PROOF_BOUNDARY_OVERCLAIM_INVALID']
claim_boundary_validation ... 'invalid_fields': ['claim_boundaries.onprem_security_acceptance'] ... 'unexpected_fields': ['claim_boundaries.onprem_security_acceptance']
```

## Decision

This slice passes locally for native VS3 scenario gate boundary-key taxonomy.

Remaining:

- `VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`.
- Full VS3-L and VS3-P remain unclaimed by this checkpoint.
