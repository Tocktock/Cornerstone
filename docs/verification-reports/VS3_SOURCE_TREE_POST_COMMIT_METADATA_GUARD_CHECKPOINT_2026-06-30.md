# VS3 Source Tree Post-Commit Metadata Guard Checkpoint

**Date:** 2026-06-30 KST
**Owner:** JiYong / Tars
**Status:** Local/dev source-tree post-commit metadata guard slice verified.
**Scope:** VS3 local/dev scenario-gate hardening only.

## Slice Contract

Goal:

- Make the VS3 local/dev scenario gate reject reports whose source-tree post-commit relationship metadata is stale or tampered.
- Guarded fields: `final_commit`, `final_commit_pending_reason`, and `report_generated_before_commit`.

In this slice:

- `VS3-GATE-003` - local/dev reports must not overclaim unsupported readiness.
- `VS3-GATE-004` - native `cornerstone scenario verify/gate ... --json` must expose reliable gate metadata.
- `VS3-REG-004` - missing or stale source/scenario coverage must fail before release claims.
- `VS3-REG-005` - report wording must stay no stronger than evidence.

Full VS3 mapping:

- In this slice: `VS3-GATE-003`, `VS3-GATE-004`, `VS3-REG-004`, `VS3-REG-005`.
- Later AI slices: `VS3-GATE-001`, `VS3-GATE-002`, `VS3-CTX-001`, `VS3-CTX-002`, `VS3-CTX-003`, `VS3-CTX-004`, `VS3-CTX-005`, `VS3-RLS-001`, `VS3-RLS-002`, `VS3-RLS-003`, `VS3-RLS-004`, `VS3-RLS-005`, `VS3-RLS-006`, `VS3-OPA-001`, `VS3-OPA-002`, `VS3-OPA-003`, `VS3-OPA-004`, `VS3-OPA-005`, `VS3-EGR-001`, `VS3-EGR-002`, `VS3-EGR-003`, `VS3-EGR-004`, `VS3-EGR-005`, `VS3-EGR-006`, `VS3-CON-001`, `VS3-CON-002`, `VS3-CON-003`, `VS3-CON-004`, `VS3-CON-005`, `VS3-CON-006`, `VS3-TOOL-001`, `VS3-TOOL-002`, `VS3-TOOL-003`, `VS3-TOOL-004`, `VS3-TOOL-005`, `VS3-TOOL-006`, `VS3-TOOL-007`, `VS3-OBS-001`, `VS3-OBS-002`, `VS3-OBS-003`, `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-003`, `VS3-REG-006`, `VS3-REG-007`, `VS3-REG-008`.
- Human-required rows: `VS3-H01`, `VS3-H02`, `VS3-H03`, `VS3-H04`, `VS3-H05`, `VS3-H06`, `VS3-H07`.

Non-scope:

- No VS3-P production/on-prem readiness claim.
- No real IdP, real network, live provider, independent security review, human UX acceptance, production migration, or restore readiness claim.
- No product feature expansion beyond current source-tree proof metadata verification.

## Baseline Gap

Before this slice, the VS3 gate accepted tampered post-commit source-tree metadata:

```text
final_commit return 0 status success errors [] source_current passed [] dirty passed generated passed
report_generated_before_commit return 0 status success errors [] source_current passed [] dirty passed generated passed
final_commit_pending_reason return 0 status success errors [] source_current passed [] dirty passed generated passed
```

This meant a local/dev report could misstate whether it was generated before a clean final commit without tripping the scenario gate.

## Implementation

Changed source-tree fingerprint coverage:

- `packages/cornerstone_cli/main.py` now includes `final_commit`, `final_commit_pending_reason`, and `report_generated_before_commit` in `VS3_SOURCE_TREE_FINGERPRINT_FIELDS`.
- `_vs3_source_tree_current_mismatches` now distinguishes a missing field from a present `null` value, because `final_commit` is intentionally `null` while the worktree is dirty.
- `_vs3_current_source_tree` now carries the three post-commit relationship fields from `git_verification_metadata`.

Changed component proof report generation:

- `packages/cornerstone_cli/scenarios.py` now writes the three post-commit relationship fields into the `source_tree` object for regenerated VS3 component proof reports:
  - request context
  - Postgres/RLS
  - OPA policy
  - egress sandbox
  - ConnectorHub source
  - tool registry
  - observability

Changed regression coverage:

- `tests/scenario/test_scaffold_cli.py` adds `test_vs3_scenario_gate_rejects_local_dev_claim_with_stale_post_commit_source_tree`.

## Verification Evidence

Syntax:

```text
python3 -m py_compile packages/cornerstone_cli/main.py packages/cornerstone_cli/scenarios.py tests/scenario/test_scaffold_cli.py
exit: 0
```

Component proof refresh:

```text
./cornerstone security vs3-request-context --json > /tmp/cs-vs3-post-commit-refresh-request-context.json
./cornerstone security vs3-postgres-rls --json > /tmp/cs-vs3-post-commit-refresh-postgres-rls.json
./cornerstone security vs3-opa-policy --json > /tmp/cs-vs3-post-commit-refresh-opa-policy.json
./cornerstone security vs3-egress-sandbox --json > /tmp/cs-vs3-post-commit-refresh-egress-sandbox.json
./cornerstone security vs3-connectorhub-source --json > /tmp/cs-vs3-post-commit-refresh-connectorhub-source.json
./cornerstone security vs3-tool-registry --json > /tmp/cs-vs3-post-commit-refresh-tool-registry.json
./cornerstone security vs3-observability --json > /tmp/cs-vs3-post-commit-refresh-observability.json
exit: 0
```

Regenerated component report post-commit metadata check:

```text
reports/security/vs3-request-context-proof.json success True None True worktree_dirty_at_verification True True
reports/db/vs3-postgres-rls-proof.json success True None True worktree_dirty_at_verification True True
reports/policy/vs3-opa-policy-proof.json success True None True worktree_dirty_at_verification True True
reports/security/vs3-egress-sandbox-proof.json success True None True worktree_dirty_at_verification True True
reports/security/vs3-connectorhub-source-proof.json success True None True worktree_dirty_at_verification True True
reports/security/vs3-tool-registry-proof.json success True None True worktree_dirty_at_verification True True
reports/observability/vs3-observability-proof.json success True None True worktree_dirty_at_verification True True
```

Focused unit tests:

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_preserves_source_proof_boundary_and_transcript \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_stale_post_commit_source_tree

Ran 2 tests in 45.695s
OK
```

Neighboring source-tree guard tests:

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_stale_source_tree \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_stale_base_tree_hash \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_tampered_source_snapshot_status \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_missing_source_snapshot_path

Ran 4 tests in 90.666s
OK
```

Canonical VS3 local/dev report and gate:

```text
./cornerstone scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json > /tmp/cs-vs3-post-commit-verify.json
./cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json > /tmp/cs-vs3-post-commit-gate.json
exit: 0
```

Gate summary:

```text
status success
scenario_count 57
summary {'blocking': 0, 'fail': 0, 'human_required': 7, 'not_run': 0, 'not_verified': 0, 'pass': 50, 'product_feature_claims': 'VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_GATES_REQUIRED', 'scenario_count': 57}
component_status passed
source_tree_current passed []
fingerprint {'final_commit': None, 'final_commit_pending_reason': 'worktree_dirty_at_verification', 'report_generated_before_commit': True, 'verified_base_commit': 'd145c8d', 'verified_base_commit_full': 'd145c8d4a604623a4d955a0aba205d41efedb148', 'verified_base_tree_hash': '52dbaeced44736b754b11cbaec66439aa42528d2', 'verified_source_worktree_hash': 'b390b65025b1d9a5e86eb0cdfcaf402d987feeaf39b9b81c96c938ace0e05ee4', 'worktree_dirty_at_verification': True}
errors []
```

Tamper rehearsal:

```text
./cornerstone scenario gate /tmp/cs-vs3-post-commit-tampered-report.json --json > /tmp/cs-vs3-post-commit-tampered-gate.json
exit: 4
```

Tamper result:

```text
status failed
errors[0].code CS_VS3_SOURCE_TREE_METADATA_STALE
source_tree_current failed ['final_commit_mismatch', 'final_commit_pending_reason_mismatch', 'report_generated_before_commit_mismatch']
gate_conditions {'self_command_transcript_shape_valid': True, 'source_tree_current': False, 'source_tree_snapshot_coverage': True, 'source_tree_snapshot_entries': True}
gate_summary {'self_command_transcript_shape_failures': 0, 'source_tree_current_failures': 1, 'source_tree_snapshot_coverage_failures': 0, 'source_tree_snapshot_entry_failures': 0}
component_status passed
```

Post-checkpoint gate:

```text
./cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json > /tmp/cs-vs3-post-commit-post-checkpoint-gate.json
exit: 0

status success
scenario_count 57
summary {'blocking': 0, 'fail': 0, 'human_required': 7, 'not_run': 0, 'not_verified': 0, 'pass': 50, 'product_feature_claims': 'VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_GATES_REQUIRED', 'scenario_count': 57}
component_status passed
source_tree_current passed []
source_tree_snapshot_coverage passed
source_tree_snapshot_entries passed
errors []
```

Documentation and diff checks:

```text
scripts/verify_sot_docs.sh
PASS: CornerStone SoT docs verified (206 full scenarios, design system, VS-0 scaffold readiness, VS-0 scaffold gate, 58 VS-0 scenarios, CLI native-first gate, local verification plane).

git diff --check
exit: 0
```

## Proof Boundary

This checkpoint proves only that the local/dev VS3 scenario gate rejects stale post-commit source-tree relationship metadata and that regenerated local component proof reports carry those fields.

It does not prove:

- production/on-prem readiness;
- real IdP readiness;
- live provider readiness;
- independent security acceptance;
- human UX acceptance;
- migration/restore readiness.

Those remain governed by `VS3-H01` through `VS3-H07`.

## Decision

The post-commit metadata guard slice is ready to keep as part of the VS3 local/dev gate.

Recommended next step: continue with the next narrow VS3 evidence-integrity guard or pause for human review before widening beyond local/dev evidence.
