# VS3 RequestContext Checkpoint

## Summary

- Verdict: PASS for the VS3-1 local deterministic RequestContext slice only.
- Scope: VS3-CTX-001 through VS3-CTX-005.
- Date: 2026-06-29 KST.
- Owner: AI local verification.
- Report: `reports/security/vs3-request-context-proof.json`.
- Aggregate scenario report: `reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json`.

This checkpoint does not claim VS3-P, production/on-prem readiness, real IdP readiness, real network readiness, live-provider readiness, independent security acceptance, migration/restore readiness, or human UX acceptance.

## Goal

Prove the VS3 RequestContext and mission/workspace authority baseline in a local deterministic fixture:

- RequestContext is derived from trusted identity and membership fixture state, not caller-provided authority.
- CLI, API, UI, worker, and tool-runtime surfaces produce matching context digests and policy outcomes.
- Forged tenant, owner, namespace, role, classification, egress, connector, and tool grants are denied before protected side effects.
- Revocation, malformed context, and mission/workspace authority checks fail closed.
- Native CLI evidence uses the expected JSON schemas, evidence refs, policy decision refs, audit refs, and policy-denial exit code.

## Full Scenario Mapping Gate

The frozen VS3 matrix currently contains:

| Type | Count | Classification for this slice |
|---|---:|---|
| MUST_PASS | 42 | VS3-CTX-001 through VS3-CTX-005 are in this slice. VS3-GATE-001 through VS3-GATE-004 have separate checkpoint evidence. Remaining MUST_PASS rows stay mapped to later slices or existing local proof reports. |
| REGRESSION | 8 | No REGRESSION row is in this slice; all eight remain final-gate coverage. |
| HUMAN_REQUIRED | 7 | VS3-H01 through VS3-H07 remain HUMAN_REQUIRED and are not promoted by this checkpoint. |
| Total | 57 | Full 57-row inventory remains the release coverage basis. |

## Scenario Verification

| ID | Type | Expected Result | Verification Method | Evidence | Status |
|---|---|---|---|---|---|
| VS3-CTX-001 | MUST_PASS | CLI, API, UI, worker, and tool runtime derive the same RequestContext from trusted identity and membership. | Run `cornerstone security vs3-request-context --json` and inspect surface transcripts, context digest, policy decision refs, and audit refs. | `reports/security/vs3-request-context-proof.json`; `/tmp/vs3_request_context_after.json`; aggregate scenario report. | PASS for local deterministic fixture. |
| VS3-CTX-002 | MUST_PASS | Caller-forged authority fields are denied before protected DB, egress, connector, or tool side effects. | Inspect forgery matrix and negative evidence counters in the RequestContext proof. | Zero forged-authority allows; zero protected DB rows, egress, connector calls, and tool executions from forgery. | PASS for local deterministic fixture. |
| VS3-CTX-003 | MUST_PASS | Revoked membership or activation grants deny cached sessions, workers, and tool runtimes with zero post-revocation side effects. | Inspect allow -> revoke -> retry matrix and cache invalidation evidence. | Revocation matrix, policy decision refs, audit refs, zero post-revocation side effects. | PASS for local deterministic fixture. |
| VS3-CTX-004 | MUST_PASS | Missing, malformed, expired, conflicted, or unresolvable RequestContext fails closed with redacted errors and no downstream access. | Inspect fault matrix over gateway, service, worker, CLI, and tool runtime. | Context fault matrix, stable status/exit codes, redacted errors, zero downstream counters. | PASS for local deterministic fixture. |
| VS3-CTX-005 | MUST_PASS | Tenant membership alone is insufficient; mission/workspace policy controls memory, connector, model, tool, and action use. | Run native `cornerstone access check` allowed and denied commands; inspect policy/audit refs and proof transcript. | `/tmp/vs3_access_memory_read_after.json`; `/tmp/vs3_access_tool_execute_after.json`; proof command transcript exit codes. | PASS for local deterministic fixture. |

## CLI Parity Summary

| Feature / Scenario | CLI Command(s) | JSON Schema | Exit-Code Tests | Evidence/Audit Refs | Same Backend Path | Status |
|---|---|---|---|---|---|---|
| RequestContext proof | `cornerstone security vs3-request-context --json` | `cs.cli.v0 + cs.vs3_request_context_proof.v0` | Exit 0 on local deterministic proof success. | `report:reports/security/vs3-request-context-proof.json`; policy and audit refs in proof payload. | Native CLI calls `run_vs3_request_context_proof`. | PASS for this slice. |
| Principal context resolve | `cornerstone principal context resolve --json` | `cs.cli.v0` plus `cs.vs3_request_context.v0` | Exit 0 on trusted context resolution. | Policy ref `policy:policy_vs3_ctx_*`; audit ref `audit:vs3_ctx_*`. | Native CLI calls `run_vs3_request_context_proof`. | PASS for this slice. |
| Mission/workspace allowed access | `cornerstone access check --operation memory_read --json` | `cs.vs3_access_check.v0` | Exit 0 for allowed memory read. | `policy:policy_vs3_mission_memory_read`; `audit:vs3_mission_memory_read`. | Native CLI reads the proof mission policy matrix. | PASS for this slice. |
| Tenant-membership-only denial | `cornerstone access check --operation tool_execute --json` | `cs.vs3_access_check.v0` | Exit 2 for policy/permission denial per CLI-native-first baseline. | `policy:policy_vs3_mission_tool_denied`; `audit:vs3_mission_tool_denied`; `CS_VS3_ACCESS_POLICY_DENIED`. | Native CLI reads the proof mission policy matrix. | PASS for this slice. |

## Command Evidence

### RequestContext proof

```text
PATH="$PWD:$PATH" cornerstone security vs3-request-context --json
proof_exit:0
proof_status success
scenario_status {'VS3-CTX-001': 'PASS', 'VS3-CTX-002': 'PASS', 'VS3-CTX-003': 'PASS', 'VS3-CTX-004': 'PASS', 'VS3-CTX-005': 'PASS'}
checks_all True
tool_transcript_exit_code 2
proof_boundary {'human_security_acceptance': 'HUMAN_REQUIRED', 'production_onprem': 'HUMAN_REQUIRED', 'real_idp': 'HUMAN_REQUIRED', 'surface': 'local_deterministic_fixture', 'vs3_l': 'NOT_CLAIMED', 'vs3_p': 'NOT_CLAIMED'}
```

### Negative evidence

```text
negative_evidence {
  'caller_controlled_scope_accepted': 0,
  'surface_context_digest_mismatches': 0,
  'surface_policy_outcome_mismatches': 0,
  'forged_authority_paths_allowed': 0,
  'protected_db_rows_touched_by_forgery': 0,
  'egress_calls_from_forgery': 0,
  'connector_calls_from_forgery': 0,
  'tool_executions_from_forgery': 0,
  'post_revocation_side_effects': 0,
  'revocation_stale_allows': 0,
  'downstream_access_on_context_faults': 0,
  'context_faults_fell_open': 0,
  'tenant_membership_only_privileged_allows': 0,
  'implicit_cross_context_use': 0,
  'production_or_real_idp_claimed': 0,
  'human_acceptance_claimed': 0
}
```

### Principal and access CLI probes

```text
PATH="$PWD:$PATH" cornerstone principal context resolve --json
principal_exit:0
principal_status success
principal_context_schema cs.vs3_request_context.v0
principal_caller_supplied_authority_used False
principal_policy_refs ['policy:policy_vs3_ctx_bcd16e59568cb13a']
principal_audit_refs ['audit:vs3_ctx_bcd16e59568c']
```

```text
PATH="$PWD:$PATH" cornerstone access check --operation memory_read --json
memory_exit:0
memory_status allowed
memory_schema cs.vs3_access_check.v0
memory_decision allow
memory_policy_refs ['policy:policy_vs3_mission_memory_read']
memory_audit_refs ['audit:vs3_mission_memory_read']
```

```text
PATH="$PWD:$PATH" cornerstone access check --operation tool_execute --json
tool_exit:2
access_status denied
access_error_codes ['CS_VS3_ACCESS_POLICY_DENIED']
access_policy_refs ['policy:policy_vs3_mission_tool_denied']
access_audit_refs ['audit:vs3_mission_tool_denied']
```

### Aggregate scenario report and gate

```text
PATH="$PWD:$PATH" cornerstone scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
status success
request_context_status success
request_context_checks {'vs3_ctx_001_surface_context_consistent': True, 'vs3_ctx_002_forged_authority_denied': True, 'vs3_ctx_003_revocation_fail_closed': True, 'vs3_ctx_004_context_faults_fail_closed': True, 'vs3_ctx_005_mission_workspace_policy_enforced': True}
ctx_row_statuses {'VS3-CTX-001': 'PASS', 'VS3-CTX-002': 'PASS', 'VS3-CTX-003': 'PASS', 'VS3-CTX-004': 'PASS', 'VS3-CTX-005': 'PASS'}
ctx_evidence_refs {'VS3-CTX-001': ['reports/security/vs3-request-context-proof.json', 'cornerstone principal context resolve --json'], 'VS3-CTX-002': ['reports/security/vs3-request-context-proof.json', 'cornerstone security vs3-request-context --json'], 'VS3-CTX-003': ['reports/security/vs3-request-context-proof.json', 'cornerstone security vs3-request-context --json'], 'VS3-CTX-004': ['reports/security/vs3-request-context-proof.json', 'cornerstone security vs3-request-context --json'], 'VS3-CTX-005': ['reports/security/vs3-request-context-proof.json', 'cornerstone access check --operation memory_read --json', 'cornerstone access check --operation tool_execute --json']}
proof_boundary_vs3_p NOT_CLAIMED
```

```text
PATH="$PWD:$PATH" cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json
status success
scenario_count 57
coverage_validation_status passed
human_required_validation_status passed
claim_boundary_validation_status passed
row_ref_validation_status passed
error_codes []
```

### Automated checks

```text
python3 -m compileall packages/cornerstone_cli/main.py packages/cornerstone_cli/scenarios.py tests/scenario/test_scaffold_cli.py
exit code: 0
```

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_request_context_proof_is_local_and_negative_evidence_backed \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_principal_and_access_cli_paths_are_native \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_onprem_trusted_extension_verify_closes_local_dev_ai_rows
Ran 3 tests in 29.183s
OK
```

## Implementation Evidence

- `packages/cornerstone_cli/main.py` now defines `EXIT_PERMISSION_DENIED = 2` and uses it for the VS3 `cornerstone access check` policy-denial path.
- `packages/cornerstone_cli/scenarios.py` records the denied `cornerstone access check --operation tool_execute --json` proof transcript with exit code 2.
- `tests/scenario/test_scaffold_cli.py` asserts the actual denied access-check exit code and the embedded RequestContext proof transcript exit code are both 2.

## Human Required

| ID | Why AI Cannot Verify | Required Human Action | Expected Evidence | Release Impact |
|---|---|---|---|---|
| VS3-H01 through VS3-H07 | These require human/external/on-prem/security/operator evidence. | Complete the human review or external rehearsal named by each row. | Dated, redacted, signed approval/review/topology/provider/UX/migration evidence. | Blocks VS3-P, production/on-prem readiness, live readiness, security acceptance, UX acceptance, and migration/restore readiness. |

## Deliberately Not Done

- Did not claim VS3-P or production/on-prem readiness.
- Did not convert human rows to PASS.
- Did not claim real IdP, live provider, real network, or independent security acceptance.
- Did not rewrite global legacy policy-denial exit codes outside the VS3 `access check` path.
- Did not begin VS3-2 Postgres/RLS work in this slice.
- Did not run full repository tests.

## Risks

- This proof uses local deterministic fixtures, not real identity provider, real on-prem topology, production network, or external provider evidence.
- Broader CLI exit-code taxonomy still contains older policy-denial uses of exit code 8; this slice only hardens the VS3 `access check` command required by VS3-CTX-005.
- The aggregate VS3 report records a dirty worktree and remains local/dev evidence only.

## Verdict

- AI-verifiable scope: done for VS3-CTX-001 through VS3-CTX-005 local deterministic RequestContext proof.
- Human/release gate: needs-human-verification for VS3-H01 through VS3-H07.
- Recommendation: continue to the next VS3 slice only after accepting this checkpoint boundary.
