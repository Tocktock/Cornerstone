# VS3 Local Checkpoint Component Proof CLI Command Scope Source Guard Checkpoint

**Date:** 2026-06-29 KST
**Evidence timestamp:** 2026-06-29T11:22:29Z
**Status:** AI-verifiable slice complete
**Scope:** `VS3-GATE-004`, `VS3-CTX-001`, `VS3-REG-004`, `VS3-REG-005`
**Claim boundary:** Local/dev checkpoint guard only. This report does not claim VS3-P, production/on-prem readiness, real IdP readiness, live-provider readiness, security acceptance, migration/restore readiness, or human acceptance.

## Full VS3 Mapping

Current slice:

- `VS3-GATE-004`: native VS3 verifier/checkpoint must reject scenario-bearing component proof CLI transcripts whose scope provenance is caller-supplied or otherwise outside the local VS3 contract.
- `VS3-CTX-001`: RequestContext proof transcript evidence must preserve that scope is derived from `trusted_request_context`, not caller-provided scope.
- `VS3-REG-004`: component proof coverage cannot silently preserve command evidence shape while weakening scope provenance.
- `VS3-REG-005`: failure paths must keep VS3-L, VS3-P, production/on-prem, security acceptance, and human acceptance claims unclaimed.

Later slices:

- `VS3-GATE-001`, `VS3-GATE-002`, `VS3-GATE-003`
- `VS3-CTX-002` through `VS3-CTX-005`
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

- Make `cornerstone security vs3-local-checkpoint --json` reject scenario-bearing VS3 component proof command transcripts that carry caller-supplied or arbitrary scope provenance.

Expected behavior:

- Accepted component proof transcript `scope.scope_source` values are only:
  - `trusted_request_context`
  - `local_vs3_fixture`
- Missing `scope_source` fails as `scope_source_missing`.
- Any other value, including `caller_supplied`, fails as `scope_source_untrusted`.
- The failure is classified under component proof CLI command evidence shape, not as missing command evidence.

Non-scope:

- No real IdP, real tenant membership, production workspace mapping, or live provider claim.
- No replay of authorization decisions from transcript evidence alone.
- No VS3-P, production/on-prem, migration/restore, security-acceptance, or human-acceptance claim.

## Implementation Summary

Changed `packages/cornerstone_cli/main.py`:

- Added `VS3_CLI_TRANSCRIPT_SCOPE_SOURCES`.
- `_vs3_cli_command_transcript_errors` now rejects missing scope provenance as `scope_source_missing`.
- It rejects non-contract scope provenance as `scope_source_untrusted`.

Changed `tests/scenario/test_scaffold_cli.py`:

- Added `test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_untrusted_scope_source`.
- The test tampers a matching RequestContext component proof transcript to `scope_source=caller_supplied` and verifies the checkpoint fails closed.

## Verification Evidence

Reference and matrix read:

```text
Primary reference digest command read all user-listed VS3 governing references.
VS3 matrix parse: rows=57, MUST_PASS=42, REGRESSION=8, HUMAN_REQUIRED=7.
Phase counts: VS3-0=4, VS3-1=5, VS3-2=6, VS3-3=5, VS3-4=6, VS3-5=6, VS3-6=7, VS3-7=3, Final gate=8, Human gate=7.
```

Docs and hygiene checks:

```text
scripts/verify_sot_docs.sh
PASS: CornerStone CLI native-first docs verified (39 feature-family rows; all CLI-required and release-blocking).
PASS: CornerStone local verification plane docs verified (20 numbered sections; deterministic PASS gate documented).
PASS: design tokens verified (11 state tokens, 8 color groups).
PASS: CornerStone design system docs verified.
PASS: CornerStone VS-0 scaffold readiness docs verified.
PASS: CornerStone SoT docs verified (206 full scenarios, design system, VS-0 scaffold readiness, VS-0 scaffold gate, 58 VS-0 scenarios, CLI native-first gate, local verification plane).

git diff --check
exit code: 0

git status --short tmp
exit code: 0
```

Syntax checks:

```text
python3 -m compileall packages/cornerstone_cli
Listing 'packages/cornerstone_cli'...
Compiling 'packages/cornerstone_cli/main.py'...

python3 -m py_compile tests/scenario/test_scaffold_cli.py
exit code: 0
```

Focused regression tests:

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_manifest_is_hash_backed_and_boundary_safe \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_missing_scope \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_untrusted_scope_source \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_wrapped_command \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_argument_mismatch \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_invalid_timestamp \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_reversed_timestamp \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_exit_code_outside_contract

Ran 8 tests in 212.430s
OK
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

Generated transcript scope source inspection:

```text
reports/security/vs3-request-context-proof.json rows 4 scope_sources ['trusted_request_context'] bad_scope_sources []
reports/db/vs3-postgres-rls-proof.json rows 6 scope_sources ['local_vs3_fixture'] bad_scope_sources []
reports/policy/vs3-opa-policy-proof.json rows 3 scope_sources ['local_vs3_fixture'] bad_scope_sources []
reports/security/vs3-egress-sandbox-proof.json rows 3 scope_sources ['local_vs3_fixture'] bad_scope_sources []
reports/security/vs3-connectorhub-source-proof.json rows 5 scope_sources ['local_vs3_fixture'] bad_scope_sources []
reports/security/vs3-tool-registry-proof.json rows 5 scope_sources ['local_vs3_fixture'] bad_scope_sources []
reports/observability/vs3-observability-proof.json rows 6 scope_sources ['local_vs3_fixture'] bad_scope_sources []
reports/security/vs3-final-regression-proof.json rows 4 scope_sources ['local_vs3_fixture'] bad_scope_sources []
```

Controlled caller-supplied scope-source tamper:

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
        "scope_source_untrusted"
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

- `/tmp/vs3-scope-source-tamper.json`

## Pass / Fail Criteria

PASS for this slice requires:

- normal local checkpoint succeeds with `component_proof_report_cli_command_evidence_shape_failures=0`;
- all generated component proof command transcripts use only allowed scope sources;
- `caller_supplied` scope provenance fails the checkpoint with exit code 4;
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
