# VS3 Local Checkpoint Component Proof Source-Tree Current Guard Checkpoint - 2026-06-30

Status: CHECKPOINT PASS FOR THIS SLICE
Owner: Codex
Scope: VS3 local checkpoint component proof current-source guard

## Slice Contract

Goal:
- Scenario-backed VS3 component proof reports must carry source-tree provenance that matches the current `cornerstone security vs3-local-checkpoint --json` source fingerprint.
- The checkpoint must reject internally consistent but stale component proof source metadata.
- Stale component proof source metadata must remain a separate failure from embedded/file proof identity, missing source-tree metadata, and command transcript source-tree mismatch.

Full scenario mapping:
- In this slice: current-source guard for AI evidence reports backing `VS3-CTX-001` through `VS3-CTX-005`, `VS3-RLS-001` through `VS3-RLS-006`, `VS3-OPA-001` through `VS3-OPA-005`, `VS3-EGR-001` through `VS3-EGR-006`, `VS3-CON-001` through `VS3-CON-006`, `VS3-TOOL-001` through `VS3-TOOL-007`, `VS3-OBS-001` through `VS3-OBS-003`, and `VS3-REG-001` through `VS3-REG-008`.
- Supporting gate rows: `VS3-GATE-004`, `VS3-REG-004`, and `VS3-REG-005`.
- Human required: `VS3-H01`, `VS3-H02`, `VS3-H03`, `VS3-H04`, `VS3-H05`, `VS3-H06`, `VS3-H07`.
- Out of scope: accepting human evidence, unlocking VS3-P, production/on-prem readiness, real IdP readiness, real network readiness, live provider readiness, migration/restore readiness, security acceptance, or human UX acceptance.

## Implementation Evidence

Changed behavior:
- Component proof identity now records:
  - `current_source_tree_fingerprint`
  - `embedded_source_tree_fingerprint`
  - `file_source_tree_fingerprint`
  - `embedded_source_tree_current_mismatches`
  - `file_source_tree_current_mismatches`
  - `source_tree_current_success`
- The local checkpoint now emits one per-component condition:
  - `component_proof_<key>_source_tree_current`
- The local checkpoint now emits summary and negative-evidence counters:
  - `component_proof_report_source_tree_current_failures`
- New semantic error code:
  - `CS_VS3_COMPONENT_PROOF_SOURCE_TREE_STALE`

## Verification Evidence

Syntax:

```text
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
exit: 0
```

Focused regression:

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_manifest_is_hash_backed_and_boundary_safe \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_missing_source_tree_even_when_identity_matches \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_stale_source_tree_even_when_identity_matches \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_source_tree_mismatch

Ran 4 tests in 107.483s
OK
```

Native CLI refresh and checkpoint:

```text
cornerstone security vs3-evidence-reconcile --json
exit: 0
status: success

cornerstone security vs3-overclaim-lint --json
exit: 0
status: passed

cornerstone security vs3-request-context --json
exit: 0
status: success

cornerstone security vs3-postgres-rls --reuse-vs2-local-range-report reports/security/vs2-local-range.json --json
exit: 0
status: success

cornerstone security vs3-opa-policy --reuse-vs2-local-range-report reports/security/vs2-local-range.json --json
exit: 0
status: success

cornerstone security vs3-egress-sandbox --reuse-vs2-local-range-report reports/security/vs2-local-range.json --json
exit: 0
status: success

cornerstone security vs3-connectorhub-source --json
exit: 0
status: success

cornerstone security vs3-tool-registry --json
exit: 0
status: success

cornerstone security vs3-observability --json
exit: 0
status: success

cornerstone security vs3-regression-gate --json
exit: 0
status: success

cornerstone scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
exit: 0
status: success
final_verdict: VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
summary: pass=50, human_required=7, blocking=0

cornerstone human-gate vs3-p-gate --scope vs3 --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --output reports/human-gates/vs3/vs3-p-gate.json --json
exit: 4
status: blocked
final_verdict: HUMAN_REQUIRED

cornerstone security vs3-local-checkpoint --json --output reports/security/vs3-local-checkpoint.json
exit: 0
status: success
final_verdict: VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
summary: pass=50, human_required=7, blocking=0, component_proof_report_source_tree_current_failures=0, component_proof_report_semantic_failures=0
```

Generated current-source evidence:

```text
reports/security/vs3-local-checkpoint.json
summary.component_proof_report_count = 9
summary.component_proof_report_source_tree_failures = 0
summary.component_proof_report_source_tree_current_failures = 0
summary.component_proof_report_semantic_failures = 0
claim_boundary.vs3_l = LOCAL_DEV_ASSURANCE_VERIFIED
claim_boundary.vs3_p = NOT_CLAIMED
component_proof_identity.request_context_proof.source_tree_current_success = true
component_proof_identity.request_context_proof.embedded_source_tree_current_mismatches = []
component_proof_identity.request_context_proof.file_source_tree_current_mismatches = []
```

## Proof Boundary

This checkpoint proves only local deterministic current-source freshness for scenario-backed VS3 component proof reports.

It does not prove:
- owner approval,
- independent security acceptance,
- real IdP integration readiness,
- real on-prem network readiness,
- live provider rehearsal readiness,
- human operator acceptance,
- migration, backup, or restore readiness,
- VS3-P production readiness.

The seven VS3 human gates remain HUMAN_REQUIRED.
