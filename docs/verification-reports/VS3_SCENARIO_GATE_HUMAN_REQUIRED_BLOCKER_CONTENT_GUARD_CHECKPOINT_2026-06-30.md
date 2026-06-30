# VS3 Scenario Gate Human-Required Blocker Content Guard Checkpoint - 2026-06-30

**Status:** Local verifier hardening checkpoint.
**Scope:** VS3-GATE-003, VS3-GATE-004, VS3-OBS-003, VS3-REG-004, and VS3-REG-005.
**Proof boundary:** Local deterministic CLI/test evidence only.
**Non-claims:** This checkpoint does not claim VS3-P, production/on-prem readiness, live-provider readiness, real IdP readiness, migration/restore readiness, security acceptance, or human UX acceptance.

## Slice Contract

Goal:
- Prevent a VS3 local/dev assurance report from passing `cornerstone scenario gate` when `human_required_blockers` mutate VS3-H01 through VS3-H07 blocker metadata into approval-like or VS3-P-unblocking content.

In scope:
- Deterministic validation of the `human_required_blockers` objects against the frozen VS3 human rows.
- A specific gate error for blocker content mismatches.
- Focused regression coverage for a tampered local/dev report whose traceability and other claim-boundary checks remain valid.
- Direct CLI tamper evidence over individual blocker fields.

Out of scope:
- New VS3 runtime feature behavior.
- New ConnectorHub, RLS, OPA, egress, tool registry, or operator UX implementation.
- Any real production, on-prem, live-provider, real-network, real-IdP, migration/restore, security-acceptance, or human-acceptance claim.

Full VS3 mapping:
- `in_this_slice`: `VS3-GATE-003`, `VS3-GATE-004`, `VS3-OBS-003`, `VS3-REG-004`, `VS3-REG-005`.
- `later_slice`: `VS3-GATE-001`, `VS3-GATE-002`, `VS3-CTX-001` through `VS3-CTX-005`, `VS3-RLS-001` through `VS3-RLS-006`, `VS3-OPA-001` through `VS3-OPA-005`, `VS3-EGR-001` through `VS3-EGR-006`, `VS3-CON-001` through `VS3-CON-006`, `VS3-TOOL-001` through `VS3-TOOL-007`, `VS3-OBS-001`, `VS3-OBS-002`, `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-003`, `VS3-REG-006`, `VS3-REG-007`, `VS3-REG-008`.
- `HUMAN_REQUIRED`: `VS3-H01` through `VS3-H07`.

## Baseline Gap

Before this slice, a tampered VS3 local/dev report could keep the expected human blocker IDs while changing blocker content into a stronger approval claim.

Tamper examples:

```text
human_required_blockers[0].status = PASS
human_required_blockers[0].expected_evidence = APPROVED signed acceptance collected; VS3-P unlocked.
human_required_blockers[0].release_impact = VS3-P unblocked; human/security accepted.
human_required_blockers[0].required_human_action = Done and approved.
```

Observed baseline:

```text
blocker_status_pass exit 4 status failed codes ['CS_VS3_SOURCE_TREE_GENERATED_DIRTY_SNAPSHOT_INVALID']
  human_required_validation passed
  claim_boundary_validation passed
  completion_claim_validation passed
  gate_metadata_validation passed
  traceability_validation passed
blocker_evidence_approved exit 4 status failed codes ['CS_VS3_SOURCE_TREE_GENERATED_DIRTY_SNAPSHOT_INVALID']
  human_required_validation passed
  claim_boundary_validation passed
  completion_claim_validation passed
  gate_metadata_validation passed
  traceability_validation passed
blocker_release_impact_unlocked exit 4 status failed codes ['CS_VS3_SOURCE_TREE_GENERATED_DIRTY_SNAPSHOT_INVALID']
  human_required_validation passed
  claim_boundary_validation passed
  completion_claim_validation passed
  gate_metadata_validation passed
  traceability_validation passed
blocker_required_action_done exit 4 status failed codes ['CS_VS3_SOURCE_TREE_GENERATED_DIRTY_SNAPSHOT_INVALID']
  human_required_validation passed
  claim_boundary_validation passed
  completion_claim_validation passed
  gate_metadata_validation passed
  traceability_validation passed
```

Interpretation:
- The source-tree/generated-snapshot guard caught that the checked report had been changed, but the human-required semantic validation still treated the blocker content as valid.
- A specific human blocker content guard was missing.

## Change

Changed files:

- `packages/cornerstone_cli/main.py`
- `tests/scenario/test_scaffold_cli.py`

Gate behavior added:

- `cornerstone scenario gate <report> --json` now builds expected blocker metadata for `VS3-H01` through `VS3-H07` from `docs/scenario-contracts/VS3_ONPREM_SECURITY_AND_TRUSTED_EXTENSION_MATRIX.csv`.
- For every recognized blocker, the gate validates:
  - `scenario_id`;
  - `status == HUMAN_REQUIRED`;
  - `required_human_action == verification_method`;
  - `expected_evidence == required_evidence`;
  - `release_impact == pass_fail_criteria`.
- Invalid blocker content is recorded in `human_required_validation.human_required_blocker_content_issues`.
- VS3-L local/dev reports with blocker content issues fail with `CS_VS3_HUMAN_REQUIRED_BLOCKERS_INVALID`.

## Regression Tests

New negative test:

```text
tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_human_required_blocker_overclaim
```

Expected failure for tampered report:

```text
exit 4
status failed
error CS_VS3_HUMAN_REQUIRED_BLOCKERS_INVALID
human_required_validation.status failed
human_required_blocker_content_issues fields:
  status
  required_human_action
  expected_evidence
  release_impact
traceability_validation.status passed
claim_boundary_validation.status passed
completion_claim_validation.status passed
gate_metadata_validation.status passed
```

## Verification

Syntax:

```text
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
exit 0
```

Focused new test:

```text
python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_human_required_blocker_overclaim

Ran 1 test in 27.745s
OK
```

Adjacent VS3 gate tests:

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_human_required_blocker_overclaim \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_gate_metadata_overclaim \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_preserves_source_proof_boundary_and_transcript

Ran 3 tests in 79.312s
OK
```

Human-gate regression tests:

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_p_gate_blocks_on_human_evidence_not_ai_rows \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_final_regression_gate_rejects_human_gate_overclaim

Ran 2 tests in 104.287s
OK
```

Manual tamper after patch:

```text
blocker_status_pass exit 4 status failed codes ['CS_VS3_HUMAN_REQUIRED_BOUNDARY_INVALID', 'CS_VS3_HUMAN_REQUIRED_BLOCKERS_INVALID', ...]
  human_required_validation failed
  fields ['status']
  claim_boundary_validation passed
  completion_claim_validation passed
  gate_metadata_validation passed
  traceability_validation passed
blocker_evidence_approved exit 4 status failed codes ['CS_VS3_HUMAN_REQUIRED_BOUNDARY_INVALID', 'CS_VS3_HUMAN_REQUIRED_BLOCKERS_INVALID', ...]
  human_required_validation failed
  fields ['expected_evidence']
  claim_boundary_validation passed
  completion_claim_validation passed
  gate_metadata_validation passed
  traceability_validation passed
blocker_release_impact_unlocked exit 4 status failed codes ['CS_VS3_HUMAN_REQUIRED_BOUNDARY_INVALID', 'CS_VS3_HUMAN_REQUIRED_BLOCKERS_INVALID', ...]
  human_required_validation failed
  fields ['release_impact']
  claim_boundary_validation passed
  completion_claim_validation passed
  gate_metadata_validation passed
  traceability_validation passed
blocker_required_action_done exit 4 status failed codes ['CS_VS3_HUMAN_REQUIRED_BOUNDARY_INVALID', 'CS_VS3_HUMAN_REQUIRED_BLOCKERS_INVALID', ...]
  human_required_validation failed
  fields ['required_human_action']
  claim_boundary_validation passed
  completion_claim_validation passed
  gate_metadata_validation passed
  traceability_validation passed
```

The extra source-tree/component proof errors in the manual tamper output are expected before refreshing source-tree-bound generated proof files after source edits. They are not the semantic assertion for this slice.

Component proof refresh after source changes:

```text
./cornerstone security vs3-request-context --json
./cornerstone security vs3-postgres-rls --json
./cornerstone security vs3-opa-policy --json
./cornerstone security vs3-egress-sandbox --json
./cornerstone security vs3-connectorhub-source --json
./cornerstone security vs3-tool-registry --json
./cornerstone security vs3-observability --json

all exit 0
all status success
```

The source-tree-inclusive canonical VS3 scenario verify/gate must be rerun after this checkpoint file is present so the report and gate validate the final local source snapshot for this slice.

## Human-Required Boundary

These rows remain `HUMAN_REQUIRED`:

- `VS3-H01`: human architecture/security approval.
- `VS3-H02`: independent security review and retest.
- `VS3-H03`: real IdP mapping and revocation evidence.
- `VS3-H04`: real on-prem network controls evidence.
- `VS3-H05`: approved live ConnectorHub/provider rehearsal.
- `VS3-H06`: human operator UX/trust review.
- `VS3-H07`: human-supervised migration/backup/restore drill.

This slice only prevents generated local/dev reports from weakening those blockers.
