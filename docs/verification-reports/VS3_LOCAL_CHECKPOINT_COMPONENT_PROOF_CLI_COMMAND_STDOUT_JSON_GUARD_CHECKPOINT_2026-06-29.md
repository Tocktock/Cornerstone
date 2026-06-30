# VS3 Local Checkpoint Component Proof CLI Command Stdout JSON Guard Checkpoint

**Date:** 2026-06-29 KST
**Evidence timestamp:** 2026-06-29T12:23:33Z
**Status:** AI-verifiable slice complete
**Owner:** JiYong / Tars
**Scope:** VS3 local/dev checkpoint evidence hardening only

## Claim Boundary

This checkpoint verifies one local/dev evidence guard:

- every scenario-bearing component proof CLI command transcript must include structured `stdout_json`;
- `stdout_json` must carry `schema_version`, `status`, `evidence_refs`, `audit_refs`, `policy_decision_refs`, `scope`, and `source_tree`;
- `stdout_json` evidence, audit, policy-decision, scope, and source-tree fields must match the enclosing transcript metadata;
- missing or mismatched structured stdout fails component proof CLI command evidence shape validation;
- this failure must not be counted as missing CLI command evidence when the transcript itself exists;
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
| `later_slice` | `VS3-GATE-001`, `VS3-GATE-002`, `VS3-GATE-003`, `VS3-CTX-001` through `VS3-CTX-005`, `VS3-RLS-001` through `VS3-RLS-006`, `VS3-OPA-001` through `VS3-OPA-005`, `VS3-EGR-001` through `VS3-EGR-006`, `VS3-CON-001` through `VS3-CON-006`, `VS3-TOOL-001` through `VS3-TOOL-007`, `VS3-OBS-001` through `VS3-OBS-003`, `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-003`, `VS3-REG-006`, `VS3-REG-007`, `VS3-REG-008` | Behavior, substrate, integration, UI, supply-chain, and regression gates beyond this stdout-json transcript evidence slice. |

## Slice Contract

Goal:

- strengthen `cornerstone security vs3-local-checkpoint --json` so component proof CLI command transcript evidence contains a replayable structured stdout summary, not only metadata around an opaque command run.

Expected behavior:

- generated component proof transcripts include `stdout_json`;
- `stdout_json.schema_version` and `stdout_json.status` are non-empty strings;
- `stdout_json.evidence_refs` and `stdout_json.audit_refs` are non-empty string lists;
- `stdout_json.policy_decision_refs` is a string list;
- `stdout_json.scope` exactly matches transcript `scope`;
- `stdout_json.source_tree` exactly matches transcript `source_tree`;
- stdout ref mismatches emit the matching `stdout_json_*_mismatch` errors;
- missing stdout emits `stdout_json_missing`;
- failed stdout shape validation fails `component_proof_*_cli_command_evidence_shape_valid`;
- failed stdout shape validation does not fail `component_proof_*_cli_command_evidence_present`;
- failed local checkpoint exits with code 4 and keeps all broader claims `NOT_CLAIMED`.

Non-scope:

- no Tool SDK, registry, live ConnectorHub, real IdP, real network, production, migration/restore, or human-gate completion work;
- no claim that VS3-L is fully complete outside this local checkpoint state;
- no claim that any `HUMAN_REQUIRED` row is satisfied.

## Implementation Evidence

Changed source:

- `packages/cornerstone_cli/scenarios.py`
  - `_vs3_transcript_stdout_json` builds the structured stdout summary for VS3 component proof transcripts.
  - `_vs3_enrich_component_command_transcripts` attaches `stdout_json` during component proof evidence enrichment.
- `packages/cornerstone_cli/main.py`
  - `_vs3_cli_command_transcript_errors` validates stdout schema, status, evidence refs, audit refs, policy-decision refs, scope, and source tree.
  - stdout shape failures are classified as command evidence shape failures, not missing command evidence.

Changed tests:

- `tests/scenario/test_scaffold_cli.py`
  - added `test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_missing_stdout_json`.
  - updated existing missing-scope, source-tree mismatch, and untrusted-scope tests so they isolate the intended failure without accidental stdout mismatches.

## Verification Evidence

Compile checks:

```text
$ python3 -m compileall packages/cornerstone_cli
Listing 'packages/cornerstone_cli'...
Compiling 'packages/cornerstone_cli/main.py'...
Compiling 'packages/cornerstone_cli/scenarios.py'...

$ python3 -m py_compile tests/scenario/test_scaffold_cli.py
<no output; command exited 0>
```

Focused unittest set:

```text
$ python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_manifest_is_hash_backed_and_boundary_safe \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_missing_metadata \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_missing_scope \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_missing_source_tree \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_source_tree_mismatch \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_missing_stdout_json \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_untrusted_scope_source

Ran 7 tests in 213.909s
OK
```

Normal local checkpoint:

```text
$ PATH="$PWD:$PATH" cornerstone security vs3-local-checkpoint --json --output reports/human-gates/vs3/vs3-local-checkpoint.json > /tmp/vs3-local-checkpoint-stdout-json-output.json
```

Observed JSON summary:

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
  "final_verdict": "VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED",
  "missing_evidence_failures": 0,
  "shape_failures": 0,
  "status": "success"
}
```

Generated component proof reports were inspected for structured stdout:

```text
reports/security/vs3-request-context-proof.json entries 4 missing_stdout_json []
reports/db/vs3-postgres-rls-proof.json entries 6 missing_stdout_json []
reports/policy/vs3-opa-policy-proof.json entries 3 missing_stdout_json []
reports/security/vs3-egress-sandbox-proof.json entries 3 missing_stdout_json []
reports/security/vs3-connectorhub-source-proof.json entries 5 missing_stdout_json []
reports/security/vs3-tool-registry-proof.json entries 5 missing_stdout_json []
reports/observability/vs3-observability-proof.json entries 6 missing_stdout_json []
reports/security/vs3-final-regression-proof.json entries 4 missing_stdout_json []
```

Manual tamper probe:

```text
Output captured at /tmp/vs3-component-transcript-missing-stdout-json.json
```

Tamper steps:

1. Removed `stdout_json` from the first `request_context_proof` command transcript.
2. Cleared `native_cli_commands`.
3. Embedded the tampered proof into `reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json`.
4. Regenerated VS3 human-gate derived artifacts with existing human-gate records.
5. Ran `cornerstone security vs3-local-checkpoint --json`.
6. Restored all modified JSON artifacts.

Observed tamper result:

```json
{
  "checkpoint_returncode": 4,
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
  "failed_conditions_subset": [
    "component_proof_request_context_proof_semantics_success",
    "component_proof_request_context_proof_cli_command_evidence_shape_valid"
  ],
  "final_verdict": "VS3_L_LOCAL_DEV_ASSURANCE_NOT_VERIFIED",
  "negative_evidence": {
    "component_proof_report_cli_command_evidence_failures": 0,
    "component_proof_report_cli_command_evidence_shape_failures": 1,
    "human_acceptance_claimed_by_checkpoint": 0,
    "production_readiness_claimed_by_checkpoint": 0,
    "security_acceptance_claimed_by_checkpoint": 0,
    "vs3_p_claimed_by_checkpoint": 0
  },
  "request_context_identity": {
    "cli_command_evidence_shape_success": false,
    "cli_command_evidence_success": true,
    "embedded_invalid_command_transcript_entries": [
      {
        "entry_id": "0",
        "errors": [
          "stdout_json_missing"
        ]
      }
    ],
    "embedded_valid_command_transcript_count": 0,
    "file_invalid_command_transcript_entries": [
      {
        "entry_id": "0",
        "errors": [
          "stdout_json_missing"
        ]
      }
    ],
    "file_valid_command_transcript_count": 0,
    "matches_embedded_current_file": true,
    "semantic_error_codes": [
      "CS_VS3_COMPONENT_PROOF_CLI_COMMAND_EVIDENCE_INVALID"
    ]
  },
  "status": "failed",
  "summary": {
    "component_proof_report_cli_command_evidence_failures": 0,
    "component_proof_report_cli_command_evidence_shape_failures": 1
  }
}
```

## Pass/Fail Criteria

PASS for this slice requires all of the following:

- generated component proof transcripts include structured `stdout_json`;
- normal `vs3-local-checkpoint` returns status `success` with zero stdout-related shape failures;
- removing `stdout_json` from a scenario-bearing component proof transcript fails local checkpoint validation;
- the failed case reports `stdout_json_missing`;
- the failed case preserves the distinction between command evidence presence and evidence shape validity;
- the failed checkpoint output keeps VS3-L, VS3-P, production/on-prem, security acceptance, and human acceptance as `NOT_CLAIMED`;
- compile checks, focused regression tests, and documentation verification pass.

FAIL if any of the following occurs:

- a component proof transcript lacks `stdout_json` but the checkpoint still passes;
- stdout evidence/audit/policy/scope/source-tree fields differ from transcript metadata but the checkpoint still passes;
- stdout shape errors are reported only as missing command evidence;
- a stdout-shape failure claims VS3-P, production readiness, security acceptance, or human acceptance;
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

The structured stdout JSON guard is accepted as a local VS3 checkpoint evidence-hardening slice.

Recommendation: continue with the next smallest VS3 slice, preferably another verifier/evidence-boundary guard before widening into behavior substrate implementation.
