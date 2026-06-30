# VS3 Local Checkpoint Component Proof CLI Command Source Tree Guard Checkpoint

**Date:** 2026-06-29 KST
**Evidence timestamp:** 2026-06-29T11:50:28Z
**Status:** AI-verifiable slice complete
**Scope:** `VS3-GATE-004`, `VS3-REG-004`, `VS3-REG-005`
**Claim boundary:** Local/dev checkpoint guard only. This report does not claim VS3-P, production/on-prem readiness, real IdP readiness, live-provider readiness, security acceptance, migration/restore readiness, or human acceptance.

## Full VS3 Mapping

Current slice:

- `VS3-GATE-004`: native VS3 verifier/checkpoint must require source-tree metadata in scenario-bearing component proof CLI transcript evidence.
- `VS3-REG-004`: component proof coverage cannot silently preserve command evidence while dropping the source-tree boundary for the command that produced it.
- `VS3-REG-005`: failure paths must keep VS3-L, VS3-P, production/on-prem, security acceptance, and human acceptance claims unclaimed.

Later slices:

- `VS3-GATE-001`, `VS3-GATE-002`, `VS3-GATE-003`
- `VS3-CTX-001` through `VS3-CTX-005`
- `VS3-RLS-001` through `VS3-RLS-006`
- `VS3-OPA-001` through `VS3-OPA-005`
- `VS3-EGR-001` through `VS3-EGR-006`
- `VS3-CON-001` through `VS3-CON-006`
- `VS3-TOOL-001` through `VS3-TOOL-007`
- `VS3-OBS-001` through `VS3-OBS-003`
- `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-003`, `VS3-REG-006`, `VS3-REG-007`, `VS3-REG-008`

Human-required:

- `VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`.

## Slice Contract

Goal:

- Make `cornerstone security vs3-local-checkpoint --json` reject scenario-bearing VS3 component proof command transcripts that are not tied to source-tree metadata.

Expected behavior:

- Every scenario-bearing component proof transcript must include a `source_tree` object.
- The object must include non-empty `verified_base_commit`, `verified_base_commit_full`, and `verified_source_worktree_hash`.
- The object must include boolean `worktree_dirty_at_verification`.
- Missing source-tree metadata fails as `source_tree_missing`.
- Missing required source-tree fields fail as field-specific `source_tree_*_missing` errors.
- The failure is classified under component proof CLI command evidence shape, not as missing command evidence.

Non-scope:

- No claim that the dirty worktree is ready for release.
- No PR, staging, production, real IdP, real network, live provider, migration/restore, security-acceptance, or human-acceptance claim.
- No new product behavior beyond local checkpoint evidence hardening.

## Implementation Summary

Changed `packages/cornerstone_cli/scenarios.py`:

- `_vs3_enrich_component_command_transcripts` now accepts report `source_tree`.
- `_vs3_enrich_report_command_transcripts` passes the report-level source-tree object into component command transcript enrichment.
- Scenario-bearing component proof transcripts now inherit the same source-tree metadata as their proof report.

Changed `packages/cornerstone_cli/main.py`:

- `_vs3_cli_command_transcript_errors` now rejects missing source-tree metadata as `source_tree_missing`.
- It validates required source-tree fields:
  - `verified_base_commit`
  - `verified_base_commit_full`
  - `verified_source_worktree_hash`
  - `worktree_dirty_at_verification`

Changed `tests/scenario/test_scaffold_cli.py`:

- Added `test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_missing_source_tree`.
- Updated the missing-scope test so it still isolates `scope_missing`.
- Updated the missing-metadata test so missing source-tree metadata is part of the expected invalid evidence shape.

## Verification Evidence

Reference and matrix read:

```text
Read user-listed VS3 governing references and parsed the frozen VS3 matrix.
VS3 matrix parse: rows=57, MUST_PASS=42, REGRESSION=8, HUMAN_REQUIRED=7.
Phase counts: VS3-0=4, VS3-1=5, VS3-2=6, VS3-3=5, VS3-4=6, VS3-5=6, VS3-6=7, VS3-7=3, Final gate=8, Human gate=7.
```

Syntax checks:

```text
python3 -m compileall packages/cornerstone_cli
Listing 'packages/cornerstone_cli'...
Compiling 'packages/cornerstone_cli/main.py'...
Compiling 'packages/cornerstone_cli/scenarios.py'...

python3 -m py_compile tests/scenario/test_scaffold_cli.py
exit code: 0
```

Regenerated VS3 proof reports:

```text
PATH="$PWD:$PATH" cornerstone security vs3-request-context --json
exit code: 0

PATH="$PWD:$PATH" cornerstone security vs3-postgres-rls --reuse-vs2-local-range-report reports/security/vs2-local-range.json --json
exit code: 0

PATH="$PWD:$PATH" cornerstone security vs3-opa-policy --reuse-vs2-local-range-report reports/security/vs2-local-range.json --json
exit code: 0

PATH="$PWD:$PATH" cornerstone security vs3-egress-sandbox --reuse-vs2-local-range-report reports/security/vs2-local-range.json --json
exit code: 0

PATH="$PWD:$PATH" cornerstone security vs3-connectorhub-source --json
exit code: 0

PATH="$PWD:$PATH" cornerstone security vs3-tool-registry --json
exit code: 0

PATH="$PWD:$PATH" cornerstone security vs3-observability --json
exit code: 0

PATH="$PWD:$PATH" cornerstone security vs3-regression-gate --json
exit code: 0

PATH="$PWD:$PATH" cornerstone scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
exit code: 0

PATH="$PWD:$PATH" cornerstone human-gate evidence-status --scope vs3 --record-dir reports/human-gates/vs3/record-templates --use-existing --json --output reports/human-gates/vs3/evidence-status.json
exit code: 0

PATH="$PWD:$PATH" cornerstone human-gate review-kit --scope vs3 --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --use-existing --json --output reports/human-gates/vs3/review-kit.json
exit code: 0

PATH="$PWD:$PATH" cornerstone human-gate vs3-p-gate --scope vs3 --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --output reports/human-gates/vs3/vs3-p-gate.json --json
exit code: 4

PATH="$PWD:$PATH" cornerstone security vs3-local-checkpoint --json --output reports/human-gates/vs3/vs3-local-checkpoint.json
exit code: 0
```

Normal local checkpoint inspection:

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
    "vs3_l": "LOCAL_DEV_ASSURANCE_VERIFIED",
    "vs3_p": "NOT_CLAIMED"
  },
  "component_proof_report_cli_command_evidence_shape_failures": 0,
  "final_verdict": "VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED",
  "status": "success"
}
```

Generated transcript source-tree inspection:

```text
reports/security/vs3-request-context-proof.json rows 4 missing_source_tree [] invalid_source_tree []
reports/db/vs3-postgres-rls-proof.json rows 6 missing_source_tree [] invalid_source_tree []
reports/policy/vs3-opa-policy-proof.json rows 3 missing_source_tree [] invalid_source_tree []
reports/security/vs3-egress-sandbox-proof.json rows 3 missing_source_tree [] invalid_source_tree []
reports/security/vs3-connectorhub-source-proof.json rows 5 missing_source_tree [] invalid_source_tree []
reports/security/vs3-tool-registry-proof.json rows 5 missing_source_tree [] invalid_source_tree []
reports/observability/vs3-observability-proof.json rows 6 missing_source_tree [] invalid_source_tree []
reports/security/vs3-final-regression-proof.json rows 4 missing_source_tree [] invalid_source_tree []
```

Focused regression tests:

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_manifest_is_hash_backed_and_boundary_safe \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_missing_scope \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_missing_source_tree \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_untrusted_scope_source \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_wrapped_command \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_argument_mismatch \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_invalid_timestamp \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_reversed_timestamp \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_exit_code_outside_contract

Ran 9 tests in 240.964s
OK
```

Controlled missing source-tree tamper:

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
        "source_tree_missing"
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

Tamper output path:

- `/tmp/vs3-component-transcript-source-tree-tamper.json`

## Pass / Fail Criteria

PASS for this slice requires:

- normal local checkpoint succeeds with `component_proof_report_cli_command_evidence_shape_failures=0`;
- every generated scenario-bearing component proof transcript includes valid source-tree metadata;
- missing source-tree metadata fails the checkpoint with exit code 4;
- the failure is classified as invalid command evidence shape, not missing command evidence;
- failure keeps VS3-L, VS3-P, production/on-prem readiness, security acceptance, and human acceptance unclaimed.

Observed result: PASS for this slice.

## Remaining Human Gates

Unchanged:

- `VS3-H01`: architecture/security/dependency/migration owner approval.
- `VS3-H02`: independent security review and retest.
- `VS3-H03`: real IdP mapping and revocation evidence.
- `VS3-H04`: real topology egress/network evidence.
- `VS3-H05`: approved live-provider rehearsal.
- `VS3-H06`: human operator UX/trust review.
- `VS3-H07`: human-supervised migration, backup, restore, rollback, quarantine, RLS, policy, and audit drill.

These remain `HUMAN_REQUIRED` and continue to block VS3-P.
