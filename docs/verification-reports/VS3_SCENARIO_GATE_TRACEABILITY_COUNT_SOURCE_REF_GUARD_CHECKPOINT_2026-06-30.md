# VS3 Scenario Gate Traceability Count and Source Ref Guard Checkpoint

**Date:** 2026-06-30 KST
**Status:** Local deterministic checkpoint for one VS3 verifier hardening slice.
**Scope:** `VS3-GATE-004` and `VS3-REG-004`.

## Slice Contract

Goal:

- Strengthen the native VS3 scenario gate so a local/dev assurance report cannot pass while its traceability metadata misstates the number of AI-verifiable rows or the source-tree lineage used for verification.

Full VS3 mapping:

- In this slice: `VS3-GATE-004`, `VS3-REG-004`.
- Later AI-verifiable slices: `VS3-GATE-001`, `VS3-GATE-002`, `VS3-GATE-003`, `VS3-CTX-001` through `VS3-CTX-005`, `VS3-RLS-001` through `VS3-RLS-006`, `VS3-OPA-001` through `VS3-OPA-005`, `VS3-EGR-001` through `VS3-EGR-006`, `VS3-CON-001` through `VS3-CON-006`, `VS3-TOOL-001` through `VS3-TOOL-007`, `VS3-OBS-001` through `VS3-OBS-003`, `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-003`, `VS3-REG-005`, `VS3-REG-006`, `VS3-REG-007`, and `VS3-REG-008`.
- Human-required: `VS3-H01` through `VS3-H07`.

Non-scope:

- No VS3-P, production/on-prem readiness, real IdP, live provider, migration/restore, security acceptance, or human UX acceptance claim.
- No broad generated report cleanup or unrelated VS3 source-tree snapshot refresh.

## Gap Found

Before this slice, the gate accepted reports where:

- `traceability.ai_verifiable_rows` was changed from `50` to `49`.
- `traceability.source_tree_ref` was changed to a stale/bogus value.

Both probes returned `status=success` from `cornerstone scenario gate ... --json`, leaving traceability count and source lineage weaker than `VS3-GATE-004` and `VS3-REG-004` require.

## Change

- `packages/cornerstone_cli/main.py` now validates:
  - `traceability.ai_verifiable_rows` equals the frozen non-human VS3 row count.
  - `traceability.source_tree_ref` equals `source_tree.verified_source_worktree_hash`.
- `tests/scenario/test_scaffold_cli.py` now covers:
  - the positive gate path exposing expected/actual AI row count and source-tree ref.
  - negative tamper cases for row-count mismatch and source-tree ref mismatch.

## Evidence

Commands run:

```text
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
```

Result: exit `0`.

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_preserves_source_proof_boundary_and_transcript \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_traceability_count_or_source_ref_mismatch \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_wrong_traceability_transcript_path \
  -v
```

Result: `Ran 3 tests in 70.293s`, `OK`.

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_scenario_report_missing_traceability \
  -v
```

Result: `Ran 1 test in 51.121s`, `OK`.

```text
PYTHONPATH=packages python3 packages/cornerstone_cli/main.py scenario verify vs3-onprem-trusted-extension --json --output tmp/vs3-traceability-guard-canonical.json
PYTHONPATH=packages python3 packages/cornerstone_cli/main.py scenario gate tmp/vs3-traceability-guard-canonical.json --json
```

Result:

- verify status: `success`
- gate status: `success`
- final verdict: `VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED`
- traceability status: `passed`
- actual AI-verifiable rows: `50`
- expected AI-verifiable rows: `50`
- source-tree ref match: `true`

## Remaining Gates

- `VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`.
- VS3-P remains `NOT_CLAIMED`.
- Production/on-prem readiness, real IdP readiness, live-provider readiness, migration/restore readiness, security acceptance, and human UX acceptance remain `NOT_CLAIMED`.

## Decision

This slice can be treated as a local deterministic verifier hardening checkpoint. Continue to the next VS3 slice only after this checkpoint and its doc verification pass are recorded.
