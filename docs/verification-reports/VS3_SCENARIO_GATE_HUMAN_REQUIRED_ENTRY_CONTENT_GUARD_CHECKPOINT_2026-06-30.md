# VS3 Scenario Gate Human-Required Entry Content Guard Checkpoint - 2026-06-30

**Status:** Local verifier hardening checkpoint.
**Scope:** VS3-GATE-003, VS3-GATE-004, VS3-OBS-003, VS3-REG-004, and VS3-REG-005.
**Proof boundary:** Local deterministic CLI/test evidence only.
**Non-claims:** This checkpoint does not claim VS3-P, production/on-prem readiness, live-provider readiness, real IdP readiness, migration/restore readiness, security acceptance, or human UX acceptance.

## Slice Contract

Goal:
- Prevent a VS3 local/dev assurance report from passing `cornerstone scenario gate` when top-level `human_required` entries mutate VS3-H01 through VS3-H07 metadata into approval-like or VS3-P-unblocking content.

In scope:
- Deterministic validation of top-level `human_required` objects against the frozen VS3 human rows.
- A specific gate error for top-level human-required entry content mismatches.
- Focused regression coverage for a tampered local/dev report whose traceability and other claim-boundary checks remain valid.
- Direct baseline evidence showing the pre-change semantic gap.

Out of scope:
- New VS3 runtime feature behavior.
- New ConnectorHub, RLS, OPA, egress, tool registry, or operator UX implementation.
- Any real production, on-prem, live-provider, real-network, real-IdP, migration/restore, security-acceptance, or human-acceptance claim.

Full VS3 mapping:
- `in_this_slice`: `VS3-GATE-003`, `VS3-GATE-004`, `VS3-OBS-003`, `VS3-REG-004`, `VS3-REG-005`.
- `later_slice`: `VS3-GATE-001`, `VS3-GATE-002`, `VS3-CTX-001` through `VS3-CTX-005`, `VS3-RLS-001` through `VS3-RLS-006`, `VS3-OPA-001` through `VS3-OPA-005`, `VS3-EGR-001` through `VS3-EGR-006`, `VS3-CON-001` through `VS3-CON-006`, `VS3-TOOL-001` through `VS3-TOOL-007`, `VS3-OBS-001`, `VS3-OBS-002`, `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-003`, `VS3-REG-006`, `VS3-REG-007`, `VS3-REG-008`.
- `HUMAN_REQUIRED`: `VS3-H01` through `VS3-H07`.

## Baseline Gap

Before this slice, a tampered VS3 local/dev report could keep the expected human blocker rows intact while changing top-level `human_required` content into a stronger approval claim.

Tamper applied:

```text
human_required[0].status = PASS
human_required[0].type = MUST_PASS
human_required[0].why_ai_cannot_verify = Human approval already accepted by local verifier.
human_required[0].required_human_action = Done and approved.
human_required[0].expected_evidence = APPROVED signed acceptance collected; VS3-P unlocked.
human_required[0].release_impact = VS3-P unblocked; human/security accepted.
traceability.transcript_paths = [checked tampered report path]
every row transcript_paths = [checked tampered report path]
```

Observed baseline:

```text
seed_exit 0
baseline_entry_id VS3-H01
gate_exit 0
status success
codes []
human_required_validation passed
human_required_entry_content_issues None
```

Interpretation:
- Existing human-required validation covered scenario rows, proof-boundary fields, and `human_required_blockers`.
- It did not validate the top-level `human_required` list that appears in the scenario report and human-gate package surface.
- The gate accepted a local/dev report whose human-required metadata could be read as human approval.

## Change

Changed files:

- `packages/cornerstone_cli/main.py`
- `tests/scenario/test_scaffold_cli.py`

Gate behavior added:

- `cornerstone scenario gate <report> --json` now validates top-level `human_required` entries for `VS3-H01` through `VS3-H07`.
- The gate validates exact expected IDs, missing entries, unexpected entries, duplicate entries, and object shape.
- For every recognized entry, the gate validates:
  - `id`;
  - `scenario_id`;
  - `type == HUMAN_REQUIRED`;
  - `status == HUMAN_REQUIRED`;
  - `why_ai_cannot_verify == expected_result`;
  - `required_human_action == verification_method`;
  - `expected_evidence == required_evidence`;
  - `release_impact == pass_fail_criteria`.
- Invalid top-level entry content is recorded in `human_required_validation.human_required_entry_content_issues`.
- VS3-L local/dev reports with top-level human-required entry issues fail with `CS_VS3_HUMAN_REQUIRED_ENTRIES_INVALID`.

## Regression Tests

New negative test:

```text
tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_human_required_entry_overclaim
```

Expected failure for tampered report:

```text
exit 4
status failed
error CS_VS3_HUMAN_REQUIRED_ENTRIES_INVALID
human_required_validation.status failed
human_required_entry_content_issues fields:
  type
  status
  why_ai_cannot_verify
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
python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_human_required_entry_overclaim

Ran 1 test in 26.936s
OK
```

Adjacent human-required and human-gate tests:

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_human_required_entry_overclaim \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_human_required_blocker_overclaim \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_p_gate_blocks_on_human_evidence_not_ai_rows \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_final_regression_gate_rejects_human_gate_overclaim

Ran 4 tests in 154.381s
OK
```

The source-tree-inclusive component proof and canonical VS3 scenario report must be refreshed after this checkpoint file exists so the final local source snapshot includes the guard and this decision trail.

## Human-Required Boundary

These rows remain `HUMAN_REQUIRED`:

- `VS3-H01`: human architecture/security approval.
- `VS3-H02`: independent security review and retest.
- `VS3-H03`: real IdP mapping and revocation evidence.
- `VS3-H04`: real on-prem network controls evidence.
- `VS3-H05`: approved live ConnectorHub/provider rehearsal.
- `VS3-H06`: human operator UX/trust review.
- `VS3-H07`: supervised migration/backup/restore drill.

This checkpoint does not unlock VS3-P. It only strengthens the local VS3-L scenario gate so generated reports cannot treat human-required metadata as human approval.
