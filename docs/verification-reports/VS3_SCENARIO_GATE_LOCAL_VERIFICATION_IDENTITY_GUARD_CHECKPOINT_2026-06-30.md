# VS3 Scenario Gate Local Verification Identity Guard Checkpoint - 2026-06-30

**Status:** Local deterministic checkpoint for one VS3 verifier hardening slice.
**Scope:** Native VS3 scenario gate local verification corpus/model identity exactness.
**Related rows:** `VS3-GATE-004`, `VS3-REG-005`.
**Proof boundary:** Local CLI/test evidence only. This is not VS3-L completion, VS3-P readiness, production/on-prem readiness, security acceptance, live-provider readiness, migration/restore readiness, or human acceptance.

## Slice Contract

Goal:
- Make `cornerstone scenario gate <report> --json` reject a VS3 local-dev assurance report when `corpus_pack_id`, `model_provider`, or `model_name` are consistently changed away from the deterministic local verification baseline.

In scope:
- Top-level `corpus_pack_id`
- Top-level `model_provider`
- Top-level `model_name`
- `traceability.corpus_pack_id`
- `traceability.model_provider`
- `traceability.model_name`
- Per-row `corpus_pack_id`
- Per-row `model_provider`
- Per-row `model_name`
- Native JSON gate output and focused regression coverage

Out of scope:
- RequestContext, RLS, OPA, egress, ConnectorHub live/provider flows, Tool SDK, signed registry, Agent Pack activation, and all VS3 human evidence gates.

Scenario mapping:
- `VS3-GATE-004`: in this slice. The native VS3 verifier/gate now requires the deterministic local corpus/model identity rather than only internally consistent metadata.
- `VS3-REG-005`: in this slice. A local-dev report can no longer silently replace `local_test` with an external provider identity while keeping the rest of the report internally consistent.
- Remaining VS3 AI-owned rows: later slices.
- `VS3-H01` through `VS3-H07`: remain `HUMAN_REQUIRED`.

Full VS3 inventory remains:
- `42` `MUST_PASS`
- `8` `REGRESSION`
- `7` `HUMAN_REQUIRED`
- `57` total rows
- `0` duplicate scenario IDs

## Before Evidence

Before this patch, a consistent tamper of local verification identity across top-level metadata, traceability metadata, and every scenario row passed the VS3 scenario gate:

```text
seed_exit 0
model_provider exit 0 status success codes [] failed_validations []
 traceability passed [] []
model_name exit 0 status success codes [] failed_validations []
 traceability passed [] []
corpus_pack_id exit 0 status success codes [] failed_validations []
 traceability passed [] []
```

Interpretation:
- The gate checked local verification metadata was present and internally consistent.
- It did not require the exact deterministic baseline:
  - `corpus_pack_id=fixtures/vs3/local-dev`
  - `model_provider=local_test`
  - `model_name=deterministic-local-test`

## Change Summary

Changed:
- `packages/cornerstone_cli/main.py`
  - Extends `traceability_validation` with expected local verification identity.
  - Rejects top-level, traceability, and per-row local verification identity drift.
  - Preserves existing `CS_VS3_TRACEABILITY_METADATA_MISSING` failure surface for traceability/local-verification metadata issues.
- `tests/scenario/test_scaffold_cli.py`
  - Adds `test_vs3_scenario_gate_rejects_local_dev_claim_with_local_verification_identity_mismatch`.
  - Mutates `corpus_pack_id`, `model_provider`, and `model_name` consistently across top-level, traceability, and every row.
  - Asserts unrelated identity, reconciliation, coverage, claim-boundary, component-proof, ref, and transcript validators remain green.

## Verification Evidence

Compile:

```text
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
exit 0
```

Focused regression:

```text
PYTHONPATH=packages python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_local_verification_identity_mismatch
.
----------------------------------------------------------------------
Ran 1 test in 26.880s

OK
```

Post-patch direct tamper probe:

```text
seed_exit 0
model_provider exit 4 status failed codes ['CS_VS3_TRACEABILITY_METADATA_MISSING']
 invalid_fields ['model_provider', 'traceability.model_provider']
 row_invalid_sample ['VS3-CON-001.model_provider', 'VS3-CON-002.model_provider', 'VS3-CON-003.model_provider']
model_name exit 4 status failed codes ['CS_VS3_TRACEABILITY_METADATA_MISSING']
 invalid_fields ['model_name', 'traceability.model_name']
 row_invalid_sample ['VS3-CON-001.model_name', 'VS3-CON-002.model_name', 'VS3-CON-003.model_name']
corpus_pack_id exit 4 status failed codes ['CS_VS3_TRACEABILITY_METADATA_MISSING']
 invalid_fields ['corpus_pack_id', 'traceability.corpus_pack_id']
 row_invalid_sample ['VS3-CON-001.corpus_pack_id', 'VS3-CON-002.corpus_pack_id', 'VS3-CON-003.corpus_pack_id']
```

Clean native verify and gate:

```text
verify_exit 0
gate_exit 0
status success
traceability_validation passed [] []
expected_local_verification_identity {'corpus_pack_id': 'fixtures/vs3/local-dev', 'model_name': 'deterministic-local-test', 'model_provider': 'local_test'}
actual_local_verification_identity {'corpus_pack_id': 'fixtures/vs3/local-dev', 'model_name': 'deterministic-local-test', 'model_provider': 'local_test'}
error_count 0
```

Adjacent traceability/gate checks:

```text
PYTHONPATH=packages python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_local_verification_identity_mismatch \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_without_traceability_metadata \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_traceability_count_or_source_ref_mismatch \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_top_level_scope_mismatch \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_preserves_source_proof_boundary_and_transcript
.....
----------------------------------------------------------------------
Ran 5 tests in 130.850s

OK
```

## Decision

This slice passes locally for the native VS3 scenario gate local verification identity guard.

Remaining proof surfaces:
- `VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`.
- This checkpoint does not prove production/on-prem readiness, real IdP readiness, live-provider readiness, security acceptance, migration/restore readiness, or human UX acceptance.
