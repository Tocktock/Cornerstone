# VS3 Local Checkpoint Scenario Report Source-Tree Current Guard Checkpoint - 2026-06-30

Status: CHECKPOINT PASS FOR THIS SLICE
Owner: Codex
Scope: VS3 local checkpoint aggregate scenario-report current-source guard

## Slice Contract

Goal:
- `cornerstone security vs3-local-checkpoint --json` must reject an aggregate VS3 scenario report whose top-level `source_tree` no longer matches the current source tree.
- The rejection must still occur when the scenario report, human-gate derivative reports, and VS3-P gate report hashes are internally consistent.
- The failure must remain local-dev proof only and must not claim VS3-P, production/on-prem readiness, security acceptance, migration/restore readiness, live-provider readiness, real-IdP readiness, real-network readiness, or human acceptance.

Selected scenarios:
- `VS3-GATE-004`: native VS3 verifier/checkpoint emits status, counts, per-row evidence, human rows, and gate metadata.
- `VS3-REG-004`: coverage and audit gates cannot silently drop or accept stale evidence before a release claim.
- `VS3-REG-005`: reports and metadata cannot overclaim beyond the evidence boundary.

Full VS3 scenario mapping:
- In this slice: `VS3-GATE-004`, `VS3-REG-004`, `VS3-REG-005`.
- Later slice: `VS3-GATE-001`, `VS3-GATE-002`, `VS3-GATE-003`, `VS3-CTX-001`, `VS3-CTX-002`, `VS3-CTX-003`, `VS3-CTX-004`, `VS3-CTX-005`, `VS3-RLS-001`, `VS3-RLS-002`, `VS3-RLS-003`, `VS3-RLS-004`, `VS3-RLS-005`, `VS3-RLS-006`, `VS3-OPA-001`, `VS3-OPA-002`, `VS3-OPA-003`, `VS3-OPA-004`, `VS3-OPA-005`, `VS3-EGR-001`, `VS3-EGR-002`, `VS3-EGR-003`, `VS3-EGR-004`, `VS3-EGR-005`, `VS3-EGR-006`, `VS3-CON-001`, `VS3-CON-002`, `VS3-CON-003`, `VS3-CON-004`, `VS3-CON-005`, `VS3-CON-006`, `VS3-TOOL-001`, `VS3-TOOL-002`, `VS3-TOOL-003`, `VS3-TOOL-004`, `VS3-TOOL-005`, `VS3-TOOL-006`, `VS3-TOOL-007`, `VS3-OBS-001`, `VS3-OBS-002`, `VS3-OBS-003`, `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-003`, `VS3-REG-006`, `VS3-REG-007`, `VS3-REG-008`.
- HUMAN_REQUIRED: `VS3-H01`, `VS3-H02`, `VS3-H03`, `VS3-H04`, `VS3-H05`, `VS3-H06`, `VS3-H07`.
- Out of scope: component proof source-tree freshness, accepting human evidence, unlocking VS3-P, production/on-prem deployment, real external provider proof, real IdP proof, real network proof, migration/restore drill, independent security review, and human UX acceptance.

## Implementation Evidence

Changed behavior:
- The local checkpoint compares the aggregate scenario report top-level `source_tree` fingerprint against the current source tree.
- The checkpoint now emits:
  - `checkpoint_conditions.scenario_report_source_tree_current`
  - `summary.scenario_report_source_tree_current_status`
  - `negative_evidence.scenario_report_source_tree_current_failures`
  - `scenario_report.source_tree_current_validation`
  - `scenario_report_source_tree_current_validation`
- A stale aggregate scenario report now fails with `VS3_L_LOCAL_DEV_ASSURANCE_NOT_VERIFIED` even if human-gate derivative report hashes match the stale aggregate report.

## Verification Evidence

Syntax:

```text
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
exit: 0
```

Focused negative test:

```text
python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_stale_scenario_report_source_tree

Ran 1 test in 27.057s
OK
```

Focused checkpoint regression:

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_manifest_is_hash_backed_and_boundary_safe \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_scenario_report_missing_traceability \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_stale_scenario_report_source_tree \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_stale_source_tree_even_when_identity_matches

Ran 4 tests in 107.274s
OK
```

Native report refresh after this document is present:
- Refresh the VS3 proof reports, scenario report, human-gate derivative reports, VS3-P gate, and local checkpoint.
- Required final local checkpoint evidence:
  - `status=success`
  - `final_verdict=VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED`
  - `summary.pass=50`
  - `summary.human_required=7`
  - `summary.blocking=0`
  - `summary.scenario_report_source_tree_current_status=passed`
  - `negative_evidence.scenario_report_source_tree_current_failures=0`
  - `claim_boundary.vs3_l=LOCAL_DEV_ASSURANCE_VERIFIED`
  - `claim_boundary.vs3_p=NOT_CLAIMED`

## Proof Boundary

This checkpoint proves only that the local VS3 aggregate scenario report used by the local checkpoint is tied to the current source-tree fingerprint.

It does not prove:
- owner approval,
- independent security acceptance,
- real IdP integration readiness,
- real on-prem network readiness,
- live provider rehearsal readiness,
- human operator acceptance,
- migration, backup, or restore readiness,
- VS3-P production/on-prem readiness.

The seven VS3 human gates remain HUMAN_REQUIRED.
