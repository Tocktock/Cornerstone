# VS3 Scenario Gate Evidence Reconciliation Semantics Guard Checkpoint - 2026-06-30

**Status:** Local deterministic checkpoint for one VS3 verifier hardening slice.
**Scope:** Native VS3 scenario gate `evidence_reconciliation` semantic exactness.
**Related rows:** `VS3-GATE-001`, `VS3-REG-004`.
**Proof boundary:** Local CLI/test evidence only. This is not VS3-L completion, VS3-P readiness, production/on-prem readiness, security acceptance, live-provider readiness, migration/restore readiness, or human acceptance.

## Slice Contract

Goal:
- Make `cornerstone scenario gate <report> --json` reject a VS3 local-dev assurance report when `evidence_reconciliation` no longer matches the deterministic current VS2/VS3 evidence reconciliation result.

In scope:
- `evidence_reconciliation.schema_version`
- `evidence_reconciliation.status`
- `evidence_reconciliation.canonical_status`
- `evidence_reconciliation.canonical_status_artifact`
- `evidence_reconciliation.canonical_status_sha256`
- `evidence_reconciliation.final_product_claim_string`
- `evidence_reconciliation.conflicting_reports_classified`
- `evidence_reconciliation.evidence_refs`
- `evidence_reconciliation.audit_refs`
- `evidence_reconciliation.policy_decision_refs`
- `evidence_reconciliation.artifacts`
- `evidence_reconciliation.errors`
- `evidence_reconciliation.negative_evidence`
- `evidence_reconciliation.claim_boundary`
- native JSON gate output and focused regression coverage

Out of scope:
- RequestContext, RLS, OPA, egress, ConnectorHub live/provider flows, Tool SDK, signed registry, Agent Pack activation, and all VS3 human evidence gates.

Scenario mapping:
- `VS3-GATE-001`: in this slice. The gate now recomputes the canonical evidence reconciliation object and rejects semantic drift.
- `VS3-REG-004`: in this slice. The gate now catches reconciliation proof drift even when the embedded proof and referenced proof file match each other.
- Remaining VS3 AI-owned rows: later slices.
- `VS3-H01` through `VS3-H07`: remain `HUMAN_REQUIRED`.

Full VS3 inventory remains:
- `42` `MUST_PASS`
- `8` `REGRESSION`
- `7` `HUMAN_REQUIRED`
- `57` total rows
- `0` duplicate scenario IDs

## Before Evidence

Before this patch, a dual tamper that changed both the scenario report's embedded `evidence_reconciliation` object and the referenced `reports/security/vs3-evidence-reconciliation.json` proof file failed only through the generic generated-dirty snapshot guard:

```text
seed_exit 0
canonical_status_dual exit 4 status failed codes ['CS_VS3_SOURCE_TREE_GENERATED_DIRTY_SNAPSHOT_INVALID']
  component passed {}
canonical_artifact_dual exit 4 status failed codes ['CS_VS3_SOURCE_TREE_GENERATED_DIRTY_SNAPSHOT_INVALID']
  component passed {}
final_claim_dual exit 4 status failed codes ['CS_VS3_SOURCE_TREE_GENERATED_DIRTY_SNAPSHOT_INVALID']
  component passed {}
```

Interpretation:
- The gate failed closed, but not with a direct reconciliation-semantics error.
- `component_proof_validation` passed because the embedded proof and referenced proof file matched.
- `VS3-GATE-001` needed an explicit canonical evidence reconciliation validation.

## Change Summary

Changed:
- `packages/cornerstone_cli/main.py`
  - Adds `evidence_reconciliation_validation`.
  - Recomputes expected reconciliation with `reconcile_vs3_evidence(root)`.
  - Fails local-dev assurance with `CS_VS3_EVIDENCE_RECONCILIATION_INVALID` when embedded reconciliation semantics drift from current deterministic evidence.
- `tests/scenario/test_scaffold_cli.py`
  - Adds `test_vs3_scenario_gate_rejects_local_dev_claim_with_evidence_reconciliation_mismatch`.
  - Mutates canonical status, canonical artifact, final claim string, claim boundary, and artifact classification in both the scenario report and referenced proof file.
  - Asserts `component_proof_validation` remains passed while `evidence_reconciliation_validation` fails.

## Verification Evidence

Compile:

```text
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
exit 0
```

Focused regression:

```text
PYTHONPATH=packages python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_evidence_reconciliation_mismatch
.
----------------------------------------------------------------------
Ran 1 test in 28.282s

OK
```

Post-patch direct dual-tamper probe:

```text
seed_exit 0
gate_exit 4
status failed
error_codes ['CS_VS3_EVIDENCE_RECONCILIATION_INVALID', 'CS_VS3_SOURCE_TREE_GENERATED_DIRTY_SNAPSHOT_INVALID']
evidence_reconciliation_validation_status failed
evidence_reconciliation_validation_invalid_fields ['evidence_reconciliation.canonical_status']
component_proof_validation passed
```

Clean native verify and gate:

```text
verify_exit 0
gate_exit 0
status success
evidence_reconciliation_validation passed []
overclaim_lint_validation passed []
error_count 0
```

Adjacent gate checks:

```text
PYTHONPATH=packages python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_evidence_reconciliation_mismatch \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_overclaim_lint_mismatch \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_matrix_or_cli_coverage_mismatch \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_preserves_source_proof_boundary_and_transcript
....
----------------------------------------------------------------------
Ran 4 tests in 109.623s

OK
```

## Decision

This slice passes locally for the native VS3 scenario gate evidence reconciliation semantic exactness guard.

Remaining proof surfaces:
- `VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`.
- This checkpoint does not prove production/on-prem readiness, real IdP readiness, live-provider readiness, security acceptance, migration/restore readiness, or human UX acceptance.
