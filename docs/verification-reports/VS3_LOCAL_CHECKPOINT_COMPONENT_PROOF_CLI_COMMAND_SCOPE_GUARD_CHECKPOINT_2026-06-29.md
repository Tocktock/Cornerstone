# VS3 Local Checkpoint Component Proof CLI Command Scope Guard Checkpoint

**Date:** 2026-06-29 KST
**Evidence timestamp:** 2026-06-29T10:34:10Z
**Status:** AI-verifiable slice complete
**Scope:** `VS3-GATE-004`, `VS3-REG-004`, `VS3-REG-005`
**Claim boundary:** Local/dev checkpoint guard only. This report does not claim VS3-P, production/on-prem readiness, real IdP readiness, live-provider readiness, security acceptance, migration/restore readiness, or human acceptance.

## Full VS3 Mapping

Current slice:

- `VS3-GATE-004`: native VS3 verifier/checkpoint must require tenant/owner/namespace/workspace scope in scenario-bearing component CLI transcript evidence.
- `VS3-REG-004`: component proof coverage cannot silently drop transcript scope while preserving matching component proof identity.
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

- Make `cornerstone security vs3-local-checkpoint --json` reject scenario-bearing VS3 component proof transcripts that omit tenant/owner/namespace/workspace scope.

Expected behavior:

- Every scenario-bearing component proof transcript must include a `scope` object with:
  - `tenant_id`
  - `owner_id`
  - `namespace_id`
  - `workspace_id`
- RequestContext proof transcripts should derive scope from `trusted_request_context`.
- Other local component proof transcripts should declare the local VS3 fixture scope explicitly.
- Missing command evidence, incomplete command metadata, and missing command scope remain separate machine-readable failures under the component proof CLI evidence shape guard.

Non-scope:

- No claim that real IdP, real tenant membership, or production workspace mapping has been verified.
- No per-command authorization replay beyond transcript evidence validation.
- No production/on-prem, live-provider, migration/restore, security-acceptance, or human-acceptance claim.

## Implementation Summary

Changed `packages/cornerstone_cli/scenarios.py`:

- Added `VS3_LOCAL_DEV_TRANSCRIPT_SCOPE`.
- Added `_vs3_report_transcript_scope`.
- Updated `_vs3_enrich_component_command_transcripts` so scenario-bearing component transcripts receive a `scope` object.
- RequestContext proof transcripts use `scope_source=trusted_request_context`.
- Other component proof transcripts use `scope_source=local_vs3_fixture`.

Changed `packages/cornerstone_cli/main.py`:

- `_vs3_cli_command_transcript_errors` now rejects missing scope with `scope_missing`.
- It also rejects incomplete scope fields with:
  - `scope_tenant_id_missing`
  - `scope_owner_id_missing`
  - `scope_namespace_id_missing`
  - `scope_workspace_id_missing`

Updated `tests/scenario/test_scaffold_cli.py`:

- Added `test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_missing_scope`.
- Updated the metadata-missing test so missing metadata now also reports `scope_missing`.

## Verification Evidence

Syntax checks:

```text
python3 -m compileall packages/cornerstone_cli
Listing 'packages/cornerstone_cli'...
Compiling 'packages/cornerstone_cli/main.py'...
Compiling 'packages/cornerstone_cli/scenarios.py'...

python3 -m py_compile tests/scenario/test_scaffold_cli.py
exit code: 0
```

Focused regression tests:

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_manifest_is_hash_backed_and_boundary_safe \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_stale_component_proof_file \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_status_failure_even_when_identity_matches \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_inner_status_failure_even_when_identity_matches \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_missing_scenario_row_even_when_identity_matches \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_missing_check_even_when_identity_matches \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_nonzero_negative_evidence_even_when_identity_matches \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_missing_refs_even_when_identity_matches \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_missing_cli_command_evidence_even_when_identity_matches \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_malformed_cli_command_evidence_even_when_identity_matches \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_missing_metadata \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_missing_scope

Ran 12 tests in 320.325s
OK
```

Regenerated component proof reports:

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
```

Regenerated scenario/human-gate reports:

```text
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
    "human_acceptance": "NOT_CLAIMED",
    "production_onprem": "NOT_CLAIMED",
    "security_acceptance": "NOT_CLAIMED",
    "vs3_l": "LOCAL_DEV_ASSURANCE_VERIFIED",
    "vs3_p": "NOT_CLAIMED"
  },
  "final_verdict": "VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED",
  "missing_failures": 0,
  "semantic_failures": 0,
  "shape_failures": 0,
  "status": "success"
}
```

Regenerated transcript scope inspection:

```text
reports/security/vs3-request-context-proof.json rows 4 missing_scope_entries [] sample_scope {'namespace_id': 'personal', 'owner_id': 'owner_alice', 'scope_source': 'trusted_request_context', 'tenant_id': 'tenant_alpha', 'workspace_id': 'workspace_alpha'}
reports/db/vs3-postgres-rls-proof.json rows 6 missing_scope_entries [] sample_scope {'namespace_id': 'personal', 'owner_id': 'owner_alice', 'scope_source': 'local_vs3_fixture', 'tenant_id': 'tenant_alpha', 'workspace_id': 'workspace_alpha'}
reports/policy/vs3-opa-policy-proof.json rows 3 missing_scope_entries [] sample_scope {'namespace_id': 'personal', 'owner_id': 'owner_alice', 'scope_source': 'local_vs3_fixture', 'tenant_id': 'tenant_alpha', 'workspace_id': 'workspace_alpha'}
reports/security/vs3-egress-sandbox-proof.json rows 3 missing_scope_entries [] sample_scope {'namespace_id': 'personal', 'owner_id': 'owner_alice', 'scope_source': 'local_vs3_fixture', 'tenant_id': 'tenant_alpha', 'workspace_id': 'workspace_alpha'}
reports/security/vs3-connectorhub-source-proof.json rows 5 missing_scope_entries [] sample_scope {'namespace_id': 'personal', 'owner_id': 'owner_alice', 'scope_source': 'local_vs3_fixture', 'tenant_id': 'tenant_alpha', 'workspace_id': 'workspace_alpha'}
reports/security/vs3-tool-registry-proof.json rows 5 missing_scope_entries [] sample_scope {'namespace_id': 'personal', 'owner_id': 'owner_alice', 'scope_source': 'local_vs3_fixture', 'tenant_id': 'tenant_alpha', 'workspace_id': 'workspace_alpha'}
reports/observability/vs3-observability-proof.json rows 6 missing_scope_entries [] sample_scope {'namespace_id': 'personal', 'owner_id': 'owner_alice', 'scope_source': 'local_vs3_fixture', 'tenant_id': 'tenant_alpha', 'workspace_id': 'workspace_alpha'}
reports/security/vs3-final-regression-proof.json rows 4 missing_scope_entries [] sample_scope {'namespace_id': 'personal', 'owner_id': 'owner_alice', 'scope_source': 'local_vs3_fixture', 'tenant_id': 'tenant_alpha', 'workspace_id': 'workspace_alpha'}
```

Controlled missing CLI scope tamper:

```json
{
  "checkpoint_rc": 4,
  "command_rcs": {
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
        "scope_missing"
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

- `/tmp/vs3-local-checkpoint-missing-cli-scope.json`

## Pass / Fail Criteria

PASS for this slice requires:

- normal local checkpoint succeeds with `component_proof_report_cli_command_evidence_shape_failures=0`;
- every scenario-bearing component proof transcript includes `scope.tenant_id`, `scope.owner_id`, `scope.namespace_id`, and `scope.workspace_id`;
- missing scope fails the checkpoint with exit code 4;
- missing scope fails the shape condition without being misclassified as missing command evidence;
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
