# VS3 Local Checkpoint Component Proof Source Tree Guard Checkpoint - 2026-06-30

Status: CHECKPOINT PASS FOR THIS SLICE
Owner: Codex
Scope: VS3 local checkpoint component proof source-tree provenance guard

## Slice Contract

Goal:
- The VS3 local checkpoint must reject scenario-backed component proof reports that do not carry top-level `source_tree` provenance.
- The checkpoint must keep component proof file identity, scenario status, checks, negative evidence, refs, CLI transcript evidence, and source-tree provenance as separate failure counters.
- The normal VS3 local path must still preserve the proof boundary: VS3-L local/dev assurance may be claimed, while VS3-P and human/external readiness remain NOT_CLAIMED.

Full scenario mapping:
- In this slice: source-tree provenance guard for AI evidence reports backing `VS3-GATE-001` through `VS3-GATE-004`, `VS3-CTX-001` through `VS3-CTX-005`, `VS3-RLS-001` through `VS3-RLS-006`, `VS3-OPA-001` through `VS3-OPA-005`, `VS3-EGR-001` through `VS3-EGR-006`, `VS3-CON-001` through `VS3-CON-006`, `VS3-TOOL-001` through `VS3-TOOL-007`, `VS3-OBS-001` through `VS3-OBS-003`, and `VS3-REG-001` through `VS3-REG-008`.
- Human required: `VS3-H01`, `VS3-H02`, `VS3-H03`, `VS3-H04`, `VS3-H05`, `VS3-H06`, `VS3-H07`.
- Out of scope: accepting human evidence, unlocking VS3-P, production/on-prem readiness, real IdP readiness, real network readiness, live provider readiness, migration/restore readiness, security acceptance, or human UX acceptance.

## Implementation Evidence

Changed behavior:
- Scenario-backed component proof identity rows now include:
  - `embedded_source_tree_present`
  - `file_source_tree_present`
  - `source_tree_matches_embedded_file`
  - `source_tree_identity_success`
  - `embedded_command_source_tree_mismatches`
  - `file_command_source_tree_mismatches`
- The local checkpoint now emits one per-component condition:
  - `component_proof_<key>_source_tree_identity`
- The local checkpoint now emits summary and negative-evidence counters:
  - `component_proof_report_source_tree_failures`
- New semantic error codes:
  - `CS_VS3_COMPONENT_PROOF_SOURCE_TREE_MISSING`
  - `CS_VS3_COMPONENT_PROOF_SOURCE_TREE_MISMATCH`
  - `CS_VS3_COMPONENT_PROOF_SOURCE_TREE_TRANSCRIPT_MISMATCH`

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
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_missing_cli_command_evidence_even_when_identity_matches \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_source_tree_mismatch

Ran 4 tests in 106.955s
OK
```

Native CLI refresh:

```json
{"name": "scenario_verify", "exit": 0, "expected_exit": 0, "status": "success", "final_verdict": "VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED", "scenario_count": 57, "pass": 50, "human_required": 7, "blocking": 0, "trace_status": "cs.scenario_traceability.v0", "vs3_l_claim": "LOCAL_DEV_ASSURANCE_VERIFIED", "vs3_p_claim": "NOT_CLAIMED"}
{"name": "record_scaffold", "exit": 0, "expected_exit": 0, "status": "success", "final_verdict": "HUMAN_REQUIRED", "scenario_count": 57, "pass": 50, "human_required": 7, "blocking": 0, "vs3_l_claim": "LOCAL_DEV_ASSURANCE_VERIFIED", "vs3_p_claim": "NOT_CLAIMED"}
{"name": "evidence_status", "exit": 0, "expected_exit": 0, "status": "success", "final_verdict": "HUMAN_REQUIRED", "body_status": "success", "trace_status": "passed", "scenario_count": 57, "pass": 50, "human_required": 7, "blocking": 0, "vs3_l_claim": "LOCAL_DEV_ASSURANCE_VERIFIED", "vs3_p_claim": "NOT_CLAIMED"}
{"name": "review_kit", "exit": 0, "expected_exit": 0, "status": "success", "final_verdict": "HUMAN_REQUIRED", "body_status": "ready_for_human_review", "trace_status": "passed", "scenario_count": 57, "pass": 50, "human_required": 7, "blocking": 0, "vs3_l_claim": "LOCAL_DEV_ASSURANCE_VERIFIED", "vs3_p_claim": "NOT_CLAIMED"}
{"name": "vs3_p_gate", "exit": 4, "expected_exit": 4, "status": "blocked", "final_verdict": "HUMAN_REQUIRED", "body_status": "blocked_on_human_required_evidence", "trace_status": "passed", "scenario_count": 57, "pass": 50, "human_required": 7, "vs3_l_claim": "LOCAL_DEV_ASSURANCE_VERIFIED", "vs3_p_claim": "NOT_CLAIMED"}
{"name": "local_checkpoint", "exit": 0, "expected_exit": 0, "status": "success", "final_verdict": "VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED", "trace_status": "passed", "component_source_tree_failures": 0, "scenario_count": 57, "pass": 50, "human_required": 7, "blocking": 0, "vs3_l_claim": "LOCAL_DEV_ASSURANCE_VERIFIED", "vs3_p_claim": "NOT_CLAIMED"}
{"name": "scenario_gate", "exit": 0, "expected_exit": 0, "status": "success", "final_verdict": "VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED", "scenario_count": 57, "pass": 50, "human_required": 7, "blocking": 0, "vs3_l_claim": "LOCAL_DEV_ASSURANCE_VERIFIED", "vs3_p_claim": "NOT_CLAIMED"}
```

Refreshed artifact hashes:

```text
63b763b9eb1f1728f2a808f7386ca62e1756ce0451425efd37fef7b374427c88  reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
63be382138e321baf9b560d4cc56cb253c1202017c7bfbf2d1df2e16a14328f6  reports/human-gates/vs3/evidence-status.json
ea73d6b41962a6ddb80d291cb06960b72bfc430cdb3730ae4efa1e08c0fb2236  reports/human-gates/vs3/review-kit.json
bbb459a72a15d42544fa357e040afb22aa89c95feca301394d77d83c36087213  reports/human-gates/vs3/vs3-p-gate.json
4832c9f3ce5463e12feae072d305ac11e16c44f5f41fc86a19287f0314155200  reports/security/vs3-local-checkpoint.json
```

Current local checkpoint source-tree summary:

```json
{"component_count": 9, "source_tree_failures": 0, "semantic_failures": 0, "status": "success", "final_verdict": "VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED", "vs3_l": "LOCAL_DEV_ASSURANCE_VERIFIED", "vs3_p": "NOT_CLAIMED", "human_required": 7}
```

## Proof Boundary

This checkpoint proves only local deterministic provenance guard behavior for VS3 component proof reports.

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
