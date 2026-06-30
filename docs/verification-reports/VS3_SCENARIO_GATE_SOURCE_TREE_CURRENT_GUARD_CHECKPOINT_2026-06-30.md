# VS3 Scenario Gate Source-Tree Current Guard Checkpoint - 2026-06-30

Status: CHECKPOINT PASS FOR THIS SLICE
Owner: Codex
Scope: standalone VS3 scenario gate current-source guard

## Slice Contract

Goal:
- `cornerstone scenario gate <vs3-report> --json` must reject a VS3 local-dev assurance report whose top-level `source_tree` no longer matches the current source tree.
- The rejection must occur even when the report's command transcripts are internally consistent with the stale `source_tree`.
- The gate must continue to preserve proof boundaries: no VS3-P, production/on-prem readiness, live-provider readiness, real-IdP readiness, real-network readiness, migration/restore readiness, security acceptance, or human acceptance may be claimed.

Selected scenarios:
- `VS3-GATE-004`: the native VS3 scenario gate emits status, metadata, evidence refs, and gate failure details.
- `VS3-REG-004`: scenario coverage and release gates cannot silently accept stale report metadata.
- `VS3-REG-005`: reports and release metadata cannot describe local/dev proof as broader readiness.

Full VS3 scenario mapping:
- In this slice: `VS3-GATE-004`, `VS3-REG-004`, `VS3-REG-005`.
- Later slice: `VS3-GATE-001`, `VS3-GATE-002`, `VS3-GATE-003`, `VS3-CTX-001`, `VS3-CTX-002`, `VS3-CTX-003`, `VS3-CTX-004`, `VS3-CTX-005`, `VS3-RLS-001`, `VS3-RLS-002`, `VS3-RLS-003`, `VS3-RLS-004`, `VS3-RLS-005`, `VS3-RLS-006`, `VS3-OPA-001`, `VS3-OPA-002`, `VS3-OPA-003`, `VS3-OPA-004`, `VS3-OPA-005`, `VS3-EGR-001`, `VS3-EGR-002`, `VS3-EGR-003`, `VS3-EGR-004`, `VS3-EGR-005`, `VS3-EGR-006`, `VS3-CON-001`, `VS3-CON-002`, `VS3-CON-003`, `VS3-CON-004`, `VS3-CON-005`, `VS3-CON-006`, `VS3-TOOL-001`, `VS3-TOOL-002`, `VS3-TOOL-003`, `VS3-TOOL-004`, `VS3-TOOL-005`, `VS3-TOOL-006`, `VS3-TOOL-007`, `VS3-OBS-001`, `VS3-OBS-002`, `VS3-OBS-003`, `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-003`, `VS3-REG-006`, `VS3-REG-007`, `VS3-REG-008`.
- HUMAN_REQUIRED: `VS3-H01`, `VS3-H02`, `VS3-H03`, `VS3-H04`, `VS3-H05`, `VS3-H06`, `VS3-H07`.
- Out of scope: VS3 local checkpoint behavior, component proof source-tree freshness, accepting human evidence, unlocking VS3-P, production/on-prem deployment, real external provider proof, real IdP proof, real network proof, migration/restore drill, independent security review, and human UX acceptance.

## Implementation Evidence

Changed behavior:
- The standalone VS3 scenario gate computes current-source validation for the report's top-level `source_tree`.
- The gate now emits:
  - `source_tree_current_validation`
  - `source_report.source_tree_current_validation`
  - `scenario_gate_conditions.source_tree_current`
  - `scenario_gate_summary.source_tree_current_failures`
  - `scenario_gate_negative_evidence.source_tree_current_failures`
- A stale local-dev assurance report now fails with `CS_VS3_SOURCE_TREE_METADATA_STALE`.

## Verification Evidence

Pre-fix probe:

```text
cornerstone scenario gate tmp/vs3-scenario-gate-stale-source-tree-probe.json --json
exit: 0
status: success
source_tree_hash: 0000000000000000000000000000000000000000000000000000000000000000
```

Syntax:

```text
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
exit: 0
```

Focused regression:

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_preserves_source_proof_boundary_and_transcript \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_without_source_tree \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_stale_source_tree

Ran 3 tests in 1.640s
OK
```

Native refresh requirement after this document is present:
- Run the VS3 proof report refresh, aggregate scenario verify, standalone scenario gate, human-gate derivative refresh, VS3-P gate, and local checkpoint.
- Required final evidence:
  - `cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json` exits 0 for the current report.
  - `source_tree_current_validation.status=passed`.
  - `scenario_gate_negative_evidence.source_tree_current_failures=0`.
  - `proof_boundary.vs3_p=NOT_CLAIMED`.
  - VS3-P gate still exits 4 with `HUMAN_REQUIRED`.

## Proof Boundary

This checkpoint proves only that the standalone local VS3 scenario gate rejects stale top-level source-tree metadata.

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
