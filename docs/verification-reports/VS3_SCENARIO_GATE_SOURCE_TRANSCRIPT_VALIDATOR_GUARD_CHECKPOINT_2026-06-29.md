# VS3 Scenario Gate Source Transcript Validator Guard Checkpoint - 2026-06-29

## Purpose

Strengthen the VS3 scenario verifier and scenario gate transcript proof so local/dev assurance cannot be accepted from a source report whose native verifier transcript is missing trusted scope provenance or structured stdout evidence.

This is a local/dev verification guard only. It does not claim VS3-P, production/on-prem readiness, live-provider readiness, real IdP readiness, real-network readiness, migration/restore readiness, security acceptance, or human acceptance.

## Slice Contract

- Goal: make `cornerstone scenario verify vs3-onprem-trusted-extension --json` emit a replayable self transcript and make `cornerstone scenario gate <report> --json` reject source reports whose verifier transcripts lack that shape.
- In scope: VS3 source verifier transcript `scope.scope_source`, `stdout_json`, proof-boundary stdout fields, generic transcript-shape validation in the scenario gate, scenario-gate self transcript validation, and a controlled missing-stdout tamper test.
- Out of scope: new VS3-P evidence, real IdP/provider/network evidence, human acceptance evidence, migration/restore drill evidence, and production/on-prem deployment proof.
- Claim boundary: successful local output remains `VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED`.

## Implementation Evidence

- Added `_vs3_scenario_gate_self_transcript_validation(...)` in `packages/cornerstone_cli/main.py`.
- Extended VS3 `scenario verify` self transcript with:
  - `scope.scope_source = "local_vs3_fixture"`
  - `stdout_json`
  - `stdout_json.proof_boundary.vs3_l = "LOCAL_COMPONENT_PROOF_ONLY"`
  - explicit `NOT_CLAIMED` values for VS3-P, production, live provider, real IdP/network, migration, security acceptance, and human acceptance.
- Tightened VS3 `scenario gate` source transcript validation through `_vs3_cli_command_transcript_errors(...)`.
- Added VS3 scenario-gate self transcript validation fields:
  - `self_command_transcript_validation`
  - `scenario_gate_conditions.self_command_transcript_shape_valid`
  - `scenario_gate_summary.self_command_transcript_shape_failures`
  - `scenario_gate_negative_evidence.self_command_transcript_shape_failures`
- Added a regression test that removes `stdout_json` from both source verifier transcripts and expects the gate to fail.

## Before Evidence

Pre-change inspection of `cornerstone scenario verify vs3-onprem-trusted-extension --json` showed the VS3 source self transcript did not include structured stdout proof:

```text
self_command_transcript keys:
arguments, audit_refs, cli_schema_version, command, elapsed_seconds, ended_at,
evidence_refs, exit_code, json_schema, name, output_mode, policy_decision_refs,
ref_summary, required, schema_version, scope, source, source_tree, started_at,
stderr_tail, stdout_tail, timed_out
```

The transcript also did not include `scope.scope_source`. Existing scenario-gate validation accepted source transcripts through a narrower local check that did not call the shared VS3 command-transcript shape validator.

Pass/fail implication: a local/dev assurance report could carry a native command transcript but still lack replayable structured stdout proof. This slice was required before using scenario-gate output as a stronger release-facing proof surface.

## After Evidence

Canonical source verifier command:

```bash
PATH="$PWD:$PATH" cornerstone scenario verify vs3-onprem-trusted-extension --json
```

Captured in `/tmp/vs3-scenario-verify-canonical-after.json`.

Observed output:

```text
status success
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
output_path /Users/jiyong/playground/Cornerstone/reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
self_scope_source local_vs3_fixture
self_has_stdout_json True
command_transcript_count 1
```

Canonical scenario gate command:

```bash
PATH="$PWD:$PATH" cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json
```

Captured in `/tmp/vs3-scenario-gate-canonical-after.json`.

Observed output:

```text
status success
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
source_transcript_validation {'invalid_fields': [], 'mismatched_ref_fields': [], 'missing_fields': [], 'status': 'passed'}
self_validation {'error_codes': [], 'expected_command': ['cornerstone', 'scenario', 'gate', 'reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json', '--json'], 'expected_scope': {'namespace_id': 'personal', 'owner_id': 'local-user', 'scope_source': 'local_vs3_fixture', 'tenant_id': 'local-dev', 'workspace_id': 'default'}, 'expected_source_tree_present': True, 'status': 'passed', 'valid': True}
gate_failures 0
```

Controlled tamper command:

```bash
cornerstone scenario gate tmp/vs3-gate-tamper-source-transcript-missing-stdout-after.json --json
```

Captured in `/tmp/vs3-scenario-gate-source-transcript-missing-stdout-after.json`.

Observed output:

```text
tamper_returncode 4
tamper_status failed
error_codes ['CS_VS3_SOURCE_TRANSCRIPT_METADATA_MISSING']
source_transcript_validation {'invalid_fields': ['command_transcripts.scenario_verify_vs3.stdout_json_missing', 'self_command_transcript.stdout_json_missing'], 'mismatched_ref_fields': [], 'missing_fields': [], 'status': 'failed'}
gate_transcript_exit_code 4
```

After regenerating the canonical scenario report, the dependent human-gate status artifacts initially became stale by scenario-report hash. Refreshing `evidence-status`, `review-kit`, and `vs3-p-gate` through their native CLI paths restored the local checkpoint.

Local checkpoint after refresh:

```bash
PATH="$PWD:$PATH" cornerstone security vs3-local-checkpoint --json
```

Captured in `/tmp/vs3-local-checkpoint-after-scenario-gate-transcript.json`.

Observed output:

```text
status success
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
conditions_false []
summary_self_failures 0
negative_nonzero {}
```

## Verification Commands

```bash
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
```

Result: exit code 0.

```bash
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_preserves_source_proof_boundary_and_transcript \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_source_transcript_missing_stdout_json
```

Result:

```text
Ran 2 tests in 0.879s

OK
```

```bash
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_without_source_tree \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_without_row_evidence_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_without_aggregate_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_without_source_verify_transcript \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_human_rows_marked_pass \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_promoted_to_vs3_p_readiness
```

Result:

```text
Ran 6 tests in 2.619s

OK
```

```bash
python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_manifest_is_hash_backed_and_boundary_safe
```

Result:

```text
Ran 1 test in 26.301s

OK
```

## MUST PASS Scenario Mapping

The full VS3 matrix has 57 rows: 42 `MUST_PASS`, 8 `REGRESSION`, and 7 `HUMAN_REQUIRED`. This slice only strengthens scenario verifier/gate transcript proof. All other rows remain mapped, not silently promoted.

| Scenario | Priority | Slice Status | Verification in This Slice |
|---|---|---|---|
| VS3-GATE-001 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-GATE-002 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-GATE-003 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-GATE-004 | MUST_PASS | IN_THIS_SLICE | source verifier and scenario-gate transcript validation |
| VS3-CTX-001 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-CTX-002 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-CTX-003 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-CTX-004 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-CTX-005 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-RLS-001 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-RLS-002 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-RLS-003 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-RLS-004 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-RLS-005 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-RLS-006 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-OPA-001 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-OPA-002 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-OPA-003 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-OPA-004 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-OPA-005 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-EGR-001 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-EGR-002 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-EGR-003 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-EGR-004 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-EGR-005 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-EGR-006 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-CON-001 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-CON-002 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-CON-003 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-CON-004 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-CON-005 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-CON-006 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-TOOL-001 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-TOOL-002 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-TOOL-003 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-TOOL-004 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-TOOL-005 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-TOOL-006 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-TOOL-007 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-OBS-001 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-OBS-002 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-OBS-003 | MUST_PASS | LATER_SLICE | unchanged by this slice |
| VS3-REG-001 | REGRESSION | LATER_SLICE | unchanged by this slice |
| VS3-REG-002 | REGRESSION | LATER_SLICE | unchanged by this slice |
| VS3-REG-003 | REGRESSION | LATER_SLICE | unchanged by this slice |
| VS3-REG-004 | REGRESSION | IN_THIS_SLICE | missing source transcript stdout fails the gate |
| VS3-REG-005 | REGRESSION | IN_THIS_SLICE | stdout proof boundary keeps VS3-P and production claims unclaimed |
| VS3-REG-006 | REGRESSION | LATER_SLICE | unchanged by this slice |
| VS3-REG-007 | REGRESSION | IN_THIS_SLICE | no dependency or supply-chain change |
| VS3-REG-008 | REGRESSION | LATER_SLICE | unchanged by this slice |
| VS3-H01 | HUMAN_REQUIRED | HUMAN_REQUIRED | human gate evidence remains unpromoted |
| VS3-H02 | HUMAN_REQUIRED | HUMAN_REQUIRED | human gate evidence remains unpromoted |
| VS3-H03 | HUMAN_REQUIRED | HUMAN_REQUIRED | human gate evidence remains unpromoted |
| VS3-H04 | HUMAN_REQUIRED | HUMAN_REQUIRED | human gate evidence remains unpromoted |
| VS3-H05 | HUMAN_REQUIRED | HUMAN_REQUIRED | human gate evidence remains unpromoted |
| VS3-H06 | HUMAN_REQUIRED | HUMAN_REQUIRED | human gate evidence remains unpromoted |
| VS3-H07 | HUMAN_REQUIRED | HUMAN_REQUIRED | human gate evidence remains unpromoted |

## Proof Boundary

- `VS3-GATE-004`, `VS3-REG-004`, `VS3-REG-005`, and `VS3-REG-007` receive additional local checkpoint evidence from this slice.
- `VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`.
- Regenerating the scenario report required refreshing dependent human-gate derived artifacts by hash; this is expected and preserves the hash-bound evidence model.
- No production, live provider, real IdP, real network, migration restore, security acceptance, or human acceptance claim is made.
- The worktree was dirty before this slice; unrelated pre-existing VS3 files and reports remain outside this checkpoint's implementation claim.
