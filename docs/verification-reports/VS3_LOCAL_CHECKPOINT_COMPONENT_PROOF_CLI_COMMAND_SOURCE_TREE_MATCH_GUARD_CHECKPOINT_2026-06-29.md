# VS3 Local Checkpoint Component Proof CLI Command Source Tree Match Guard Checkpoint

**Date:** 2026-06-29 KST
**Evidence timestamp:** 2026-06-29T12:02:52Z
**Status:** AI-verifiable slice complete
**Owner:** JiYong / Tars
**Scope:** VS3 local/dev checkpoint evidence hardening only

## Claim Boundary

This checkpoint verifies one local/dev evidence guard:

- every scenario-bearing component proof CLI command transcript must carry `source_tree` metadata matching the enclosing component proof report's `source_tree`;
- a mismatch is rejected as invalid command evidence shape with `source_tree_mismatch`;
- the rejection must not be counted as missing command evidence;
- the checkpoint must keep `VS3-P`, production/on-prem readiness, real IdP, real network, live-provider, migration/restore, security acceptance, and human acceptance as `NOT_CLAIMED`.

This report does not claim VS3 production readiness, independent security acceptance, real on-prem deployment readiness, live-provider readiness, migration readiness, or human acceptance.

## Full VS3 Mapping

The current VS3 matrix contains 57 rows:

| Type | Count |
|---|---:|
| MUST_PASS | 42 |
| REGRESSION | 8 |
| HUMAN_REQUIRED | 7 |
| Total | 57 |

Phase counts:

| Phase | Count |
|---|---:|
| VS3-0 | 4 |
| VS3-1 | 5 |
| VS3-2 | 6 |
| VS3-3 | 5 |
| VS3-4 | 6 |
| VS3-5 | 6 |
| VS3-6 | 7 |
| VS3-7 | 3 |
| Final gate | 8 |
| Human gate | 7 |

Execution classification for this slice:

| Classification | Scenario IDs | Reason |
|---|---|---|
| `in_this_slice` | `VS3-GATE-004`, `VS3-REG-004`, `VS3-REG-005` | Native VS3 local checkpoint evidence shape guard, coverage omission detection, and overclaim boundary preservation. |
| `HUMAN_REQUIRED` | `VS3-H01`, `VS3-H02`, `VS3-H03`, `VS3-H04`, `VS3-H05`, `VS3-H06`, `VS3-H07` | Require owner/security/operator/provider/topology/migration evidence that local AI verification cannot create. |
| `later_slice` | `VS3-GATE-001`, `VS3-GATE-002`, `VS3-GATE-003`, `VS3-CTX-001` through `VS3-CTX-005`, `VS3-RLS-001` through `VS3-RLS-006`, `VS3-OPA-001` through `VS3-OPA-005`, `VS3-EGR-001` through `VS3-EGR-006`, `VS3-CON-001` through `VS3-CON-006`, `VS3-TOOL-001` through `VS3-TOOL-007`, `VS3-OBS-001` through `VS3-OBS-003`, `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-003`, `VS3-REG-006`, `VS3-REG-007`, `VS3-REG-008` | Behavior, substrate, integration, UI, supply-chain, and regression gates beyond this source-tree consistency evidence slice. |

## Slice Contract

Goal:

- strengthen `cornerstone security vs3-local-checkpoint --json` so component proof CLI command transcript evidence cannot be reused from a different source tree than the proof report it is attached to.

Expected behavior:

- `source_tree` must exist on each command transcript;
- required source-tree fields must include `verified_base_commit`, `verified_base_commit_full`, `verified_source_worktree_hash`, and `worktree_dirty_at_verification`;
- command transcript `source_tree` must exactly equal the enclosing proof report `source_tree`;
- mismatch emits `source_tree_mismatch`;
- mismatch fails `component_proof_*_cli_command_evidence_shape_valid`;
- mismatch does not fail `component_proof_*_cli_command_evidence_present`;
- failed local checkpoint exits with code 4 and keeps all broader claims `NOT_CLAIMED`.

Non-scope:

- no Tool SDK, registry, live ConnectorHub, real IdP, real network, production, migration/restore, or human-gate completion work;
- no claim that VS3-L is fully complete;
- no claim that any `HUMAN_REQUIRED` row is satisfied.

## Implementation Evidence

Changed source:

- `packages/cornerstone_cli/main.py`
  - `_vs3_cli_command_transcript_errors` accepts `expected_source_tree` and emits `source_tree_mismatch` when command evidence differs from the enclosing proof report.
  - `_vs3_cli_command_transcript_shape` passes the expected source tree into per-entry validation.
  - `_vs3_local_checkpoint_component_proof_identity` derives embedded and file proof source trees and validates both transcript sets against their own proof report metadata.

Changed test:

- `tests/scenario/test_scaffold_cli.py`
  - added `test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_source_tree_mismatch`.
  - the test tampers only `verified_source_worktree_hash` in a component proof command transcript.
  - expected result: checkpoint fails with `source_tree_mismatch`, command evidence remains present, shape failure count is 1, missing evidence failure count is 0, and broader VS3-P/production/security/human claims remain `NOT_CLAIMED`.

## Verification Evidence

Compile checks:

```text
$ python3 -m compileall packages/cornerstone_cli
Listing 'packages/cornerstone_cli'...
Compiling 'packages/cornerstone_cli/main.py'...

$ python3 -m py_compile tests/scenario/test_scaffold_cli.py
<no output; command exited 0>
```

Normal local checkpoint:

```text
$ PATH="$PWD:$PATH" cornerstone security vs3-local-checkpoint --json --output reports/human-gates/vs3/vs3-local-checkpoint.json > /tmp/vs3-local-checkpoint-output.json
```

Observed JSON summary:

```json
{
  "status": "success",
  "final_verdict": "VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED",
  "shape_failures": 0,
  "semantic_failures": 0,
  "claim_boundary": {
    "checkpoint_is_local_dev_only": true,
    "human_acceptance": "NOT_CLAIMED",
    "live_provider": "NOT_CLAIMED",
    "migration_restore": "NOT_CLAIMED",
    "production": "NOT_CLAIMED",
    "production_onprem": "NOT_CLAIMED",
    "real_idp": "NOT_CLAIMED",
    "real_network": "NOT_CLAIMED",
    "security_acceptance": "NOT_CLAIMED",
    "structural_validation_is_not_acceptance": true,
    "vs3_l": "LOCAL_DEV_ASSURANCE_VERIFIED",
    "vs3_p": "NOT_CLAIMED"
  }
}
```

Generated component proof reports were inspected for source-tree equality:

```text
reports/security/vs3-request-context-proof.json rows 4 source_tree_mismatches []
reports/db/vs3-postgres-rls-proof.json rows 6 source_tree_mismatches []
reports/policy/vs3-opa-policy-proof.json rows 3 source_tree_mismatches []
reports/security/vs3-egress-sandbox-proof.json rows 3 source_tree_mismatches []
reports/security/vs3-connectorhub-source-proof.json rows 5 source_tree_mismatches []
reports/security/vs3-tool-registry-proof.json rows 5 source_tree_mismatches []
reports/observability/vs3-observability-proof.json rows 6 source_tree_mismatches []
reports/security/vs3-final-regression-proof.json rows 4 source_tree_mismatches []
```

Focused unittest set:

```text
$ python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_manifest_is_hash_backed_and_boundary_safe \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_missing_scope \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_missing_source_tree \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_source_tree_mismatch \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_untrusted_scope_source \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_wrapped_command \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_argument_mismatch \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_invalid_timestamp \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_reversed_timestamp \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_exit_code_outside_contract

Ran 10 tests in 304.214s
OK
```

Manual tamper probe:

```text
Output captured at /tmp/vs3-component-transcript-source-tree-mismatch.json
```

Observed tamper result:

```json
{
  "claim_boundary": {
    "checkpoint_is_local_dev_only": true,
    "human_acceptance": "NOT_CLAIMED",
    "live_provider": "NOT_CLAIMED",
    "migration_restore": "NOT_CLAIMED",
    "production": "NOT_CLAIMED",
    "production_onprem": "NOT_CLAIMED",
    "real_idp": "NOT_CLAIMED",
    "real_network": "NOT_CLAIMED",
    "security_acceptance": "NOT_CLAIMED",
    "structural_validation_is_not_acceptance": true,
    "vs3_l": "NOT_CLAIMED",
    "vs3_p": "NOT_CLAIMED"
  },
  "command_rcs": {
    "checkpoint": 4,
    "evidence_status": 0,
    "review_kit": 0,
    "vs3_p_gate": 4
  },
  "failed_conditions": [
    "component_proof_request_context_proof_semantics_success",
    "component_proof_request_context_proof_cli_command_evidence_shape_valid"
  ],
  "final_verdict": "VS3_L_LOCAL_DEV_ASSURANCE_NOT_VERIFIED",
  "invalid_entries": [
    {
      "entry_id": "0",
      "errors": [
        "source_tree_mismatch"
      ]
    }
  ],
  "missing_evidence_failures": 0,
  "semantic_error_codes": [
    "CS_VS3_COMPONENT_PROOF_CLI_COMMAND_EVIDENCE_INVALID"
  ],
  "shape_failures": 1,
  "status": "failed"
}
```

## Pass/Fail Criteria

PASS for this slice requires all of the following:

- valid component proof transcripts pass normal local checkpoint validation;
- tampered command transcript source-tree metadata fails with `source_tree_mismatch`;
- the failed case preserves the distinction between command evidence presence and evidence shape validity;
- failed checkpoint output keeps VS3-L, VS3-P, production/on-prem, security acceptance, and human acceptance as `NOT_CLAIMED`;
- compile checks and focused regression tests pass;
- documentation verifier accepts this report.

FAIL if any of the following occurs:

- transcript source-tree metadata is missing but the checkpoint still passes;
- transcript source-tree metadata differs from the proof report but the checkpoint still passes;
- mismatch is reported only as missing evidence;
- a local mismatch probe claims VS3-P, production readiness, security acceptance, or human acceptance;
- the report omits the full VS3 scenario mapping boundary.

## Remaining Human Gates

The following rows remain `HUMAN_REQUIRED` and are not advanced by this checkpoint:

- `VS3-H01` architecture/security/dependency approval;
- `VS3-H02` independent security review and retest;
- `VS3-H03` real IdP mapping;
- `VS3-H04` real on-prem network controls;
- `VS3-H05` live ConnectorHub/provider rehearsal;
- `VS3-H06` operator UX/trust review;
- `VS3-H07` migration/backup/restore drill.

## Decision

The source-tree match guard is accepted as a local VS3 checkpoint evidence-hardening slice.

Recommendation: continue with the next smallest VS3 slice, preferably another verifier/evidence-boundary guard before widening into behavior substrate implementation.
