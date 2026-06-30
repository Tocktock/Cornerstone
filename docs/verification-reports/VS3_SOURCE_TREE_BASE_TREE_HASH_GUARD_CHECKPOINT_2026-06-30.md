# VS3 Source Tree Base Tree Hash Guard Checkpoint

**Date:** 2026-06-30 KST
**Owner:** JiYong / Tars
**Status:** Local/dev source-tree metadata guard slice verified.
**Scope:** VS3 local/dev scenario-gate hardening only.

## Slice Contract

Goal:

- Make the VS3 local/dev scenario gate reject reports whose source-tree proof has a stale `verified_base_tree_hash`, even when the base commit and worktree hash still match.

In this slice:

- `VS3-GATE-003` - local/dev report must not overclaim unsupported readiness.
- `VS3-GATE-004` - native `cornerstone scenario verify/gate ... --json` must expose reliable gate metadata.
- `VS3-REG-004` - missing or stale source/audit/scenario coverage must fail before release claims.
- `VS3-REG-005` - report wording must stay no stronger than evidence.

Full VS3 mapping:

- In this slice: `VS3-GATE-003`, `VS3-GATE-004`, `VS3-REG-004`, `VS3-REG-005`.
- Later AI slices: `VS3-GATE-001`, `VS3-GATE-002`, `VS3-CTX-001`, `VS3-CTX-002`, `VS3-CTX-003`, `VS3-CTX-004`, `VS3-CTX-005`, `VS3-RLS-001`, `VS3-RLS-002`, `VS3-RLS-003`, `VS3-RLS-004`, `VS3-RLS-005`, `VS3-RLS-006`, `VS3-OPA-001`, `VS3-OPA-002`, `VS3-OPA-003`, `VS3-OPA-004`, `VS3-OPA-005`, `VS3-EGR-001`, `VS3-EGR-002`, `VS3-EGR-003`, `VS3-EGR-004`, `VS3-EGR-005`, `VS3-EGR-006`, `VS3-CON-001`, `VS3-CON-002`, `VS3-CON-003`, `VS3-CON-004`, `VS3-CON-005`, `VS3-CON-006`, `VS3-TOOL-001`, `VS3-TOOL-002`, `VS3-TOOL-003`, `VS3-TOOL-004`, `VS3-TOOL-005`, `VS3-TOOL-006`, `VS3-TOOL-007`, `VS3-OBS-001`, `VS3-OBS-002`, `VS3-OBS-003`, `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-003`, `VS3-REG-006`, `VS3-REG-007`, `VS3-REG-008`.
- Human-required rows: `VS3-H01`, `VS3-H02`, `VS3-H03`, `VS3-H04`, `VS3-H05`, `VS3-H06`, `VS3-H07`.

Non-scope:

- No VS3-P production/on-prem readiness claim.
- No real IdP, real network, live provider, independent security review, human UX acceptance, production migration, or restore readiness claim.
- No broad refactor outside current source-tree proof metadata and report regeneration.

## Implementation

Changed source-tree fingerprint coverage:

- `packages/cornerstone_cli/main.py` now includes `verified_base_tree_hash` in `VS3_SOURCE_TREE_FINGERPRINT_FIELDS`.
- `packages/cornerstone_cli/main.py` now includes `verified_base_tree_hash` in `_vs3_current_source_tree`.

Changed component proof report generation:

- `packages/cornerstone_cli/scenarios.py` now writes `verified_base_tree_hash` into the `source_tree` object for all regenerated VS3 component proof reports:
  - request context
  - Postgres/RLS
  - OPA policy
  - egress sandbox
  - ConnectorHub source
  - tool registry
  - observability

Changed regression coverage:

- `tests/scenario/test_scaffold_cli.py` adds `test_vs3_scenario_gate_rejects_local_dev_claim_with_stale_base_tree_hash`.

## Verification Evidence

Syntax:

```text
python3 -m py_compile packages/cornerstone_cli/main.py packages/cornerstone_cli/scenarios.py tests/scenario/test_scaffold_cli.py
exit: 0
```

Component proof refresh:

```text
./cornerstone security vs3-request-context --json > /tmp/cs-vs3-refresh-request-context.json
./cornerstone security vs3-postgres-rls --json > /tmp/cs-vs3-refresh-postgres-rls.json
./cornerstone security vs3-opa-policy --json > /tmp/cs-vs3-refresh-opa-policy.json
./cornerstone security vs3-egress-sandbox --json > /tmp/cs-vs3-refresh-egress-sandbox.json
./cornerstone security vs3-connectorhub-source --json > /tmp/cs-vs3-refresh-connectorhub-source.json
./cornerstone security vs3-tool-registry --json > /tmp/cs-vs3-refresh-tool-registry.json
./cornerstone security vs3-observability --json > /tmp/cs-vs3-refresh-observability.json
exit: 0
```

Regenerated component report source-tree check:

```text
reports/security/vs3-request-context-proof.json success True 52dbaeced44736b754b11cbaec66439aa42528d2
reports/db/vs3-postgres-rls-proof.json success True 52dbaeced44736b754b11cbaec66439aa42528d2
reports/policy/vs3-opa-policy-proof.json success True 52dbaeced44736b754b11cbaec66439aa42528d2
reports/security/vs3-egress-sandbox-proof.json success True 52dbaeced44736b754b11cbaec66439aa42528d2
reports/security/vs3-connectorhub-source-proof.json success True 52dbaeced44736b754b11cbaec66439aa42528d2
reports/security/vs3-tool-registry-proof.json success True 52dbaeced44736b754b11cbaec66439aa42528d2
reports/observability/vs3-observability-proof.json success True 52dbaeced44736b754b11cbaec66439aa42528d2
```

Focused unit tests:

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_preserves_source_proof_boundary_and_transcript \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_stale_base_tree_hash

Ran 2 tests in 45.671s
OK
```

Neighboring source-tree guard tests:

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_stale_source_tree \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_tampered_source_snapshot_status \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_missing_source_snapshot_path

Ran 3 tests in 67.903s
OK
```

Canonical VS3 local/dev report and gate:

```text
./cornerstone scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json > /tmp/cs-vs3-base-tree-verify.json
./cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json > /tmp/cs-vs3-base-tree-gate.json
exit: 0
```

Gate summary:

```text
status success
scenario_count 57
summary {'blocking': 0, 'fail': 0, 'human_required': 7, 'not_run': 0, 'not_verified': 0, 'pass': 50, 'product_feature_claims': 'VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_GATES_REQUIRED', 'scenario_count': 57}
component_status passed
component_count 9
source_tree_current passed []
source_tree_snapshot_coverage passed
source_tree_snapshot_entries passed
gate_conditions {'self_command_transcript_shape_valid': True, 'source_tree_current': True, 'source_tree_snapshot_coverage': True, 'source_tree_snapshot_entries': True}
gate_summary {'self_command_transcript_shape_failures': 0, 'source_tree_current_failures': 0, 'source_tree_snapshot_coverage_failures': 0, 'source_tree_snapshot_entry_failures': 0}
```

Tamper rehearsal:

```text
./cornerstone scenario gate /tmp/cs-vs3-base-tree-tampered-report.json --json > /tmp/cs-vs3-base-tree-tampered-gate.json
exit: 4
```

Tamper result:

```text
status failed
errors[0].code CS_VS3_SOURCE_TREE_METADATA_STALE
source_tree_current failed ['verified_base_tree_hash_mismatch']
gate_conditions {'self_command_transcript_shape_valid': True, 'source_tree_current': False, 'source_tree_snapshot_coverage': True, 'source_tree_snapshot_entries': True}
gate_summary {'self_command_transcript_shape_failures': 0, 'source_tree_current_failures': 1, 'source_tree_snapshot_coverage_failures': 0, 'source_tree_snapshot_entry_failures': 0}
component_status passed
```

Post-checkpoint gate:

```text
./cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json > /tmp/cs-vs3-base-tree-post-checkpoint-gate.json
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

This checkpoint proves only the local/dev VS3 scenario gate rejects stale `verified_base_tree_hash` source-tree metadata and that regenerated local component proof reports carry that metadata.

It does not prove:

- production/on-prem readiness;
- real IdP readiness;
- live provider readiness;
- independent security acceptance;
- human UX acceptance;
- migration/restore readiness.

Those remain governed by `VS3-H01` through `VS3-H07`.

## Decision

The base-tree hash guard slice is ready to keep as part of the VS3 local/dev gate.

Recommended next step: continue with the next narrow VS3 source/evidence integrity guard or pause for human review before widening beyond local/dev evidence.
