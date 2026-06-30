# VS3 Scenario Gate Coverage Checkpoint

## Summary

- Verdict: PASS for the VS3 scenario-gate coverage guard slice only.
- Scope: VS3-GATE-004 and VS3-REG-004 coverage integrity for `cornerstone scenario gate`.
- Date: 2026-06-29 KST.
- Owner: AI local verification.
- Commit: `d145c8d4a604623a4d955a0aba205d41efedb148` base tree with dirty worktree evidence recorded by the generated VS3 report.

This checkpoint does not claim VS3-P, production/on-prem readiness, live-provider readiness, real IdP readiness, independent security acceptance, migration/restore readiness, or human UX acceptance.

## Goal

Ensure a VS3 local/dev scenario report cannot silently drop, duplicate, or add scenario rows before `cornerstone scenario gate` accepts the report.

## Full Scenario Mapping Gate

The frozen VS3 matrix currently contains:

| Type | Count | Classification for this slice |
|---|---:|---|
| MUST_PASS | 42 | VS3-GATE-004 in this slice; remaining MUST_PASS rows stay mapped to existing VS3 slice evidence or later verification checkpoints. |
| REGRESSION | 8 | VS3-REG-004 in this slice; remaining REGRESSION rows stay mapped to final gate verification. |
| HUMAN_REQUIRED | 7 | VS3-H01 through VS3-H07 remain HUMAN_REQUIRED and are not promoted by this checkpoint. |
| Total | 57 | Gate coverage requires every row exactly once. |

## Scenario Verification

| ID | Type | Expected Result | Verification Method | Evidence | Status |
|---|---|---|---|---|---|
| VS3-GATE-004 | MUST_PASS | `cornerstone scenario verify vs3-onprem-trusted-extension --json` emits status, counts, per-row evidence, human rows, and gate metadata. | Regenerated the VS3 report, then gated it with `cornerstone scenario gate`. | `reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json`; `/tmp/vs3_verify_current.json`; `/tmp/vs3_gate_current.json`. | PASS for coverage guard. |
| VS3-REG-004 | REGRESSION | Scenario coverage cannot silently drop before a release/local-dev claim. | Missing-row, duplicate-row, and unexpected-row tamper probes must fail the gate. | Direct tamper probes returned exit code 4 with `CS_VS3_SCENARIO_COVERAGE_INVALID`; focused unit tests passed. | PASS for coverage guard. |

## CLI Parity Summary

| Feature / Scenario | CLI Command(s) | JSON Schema | Exit-Code Tests | Evidence/Audit Refs | Same Backend Path | Status |
|---|---|---|---|---|---|---|
| VS3 report generation | `cornerstone scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json` | `cs.vs3_onprem_trusted_extension.v0` | Exit 0 on canonical current report generation. | Report output and source tree metadata in `reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json`. | Native CLI verifier. | PASS for this slice. |
| VS3 coverage gate | `cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json` | `coverage_validation` object in gate payload. | Exit 0 for exact 57-row report; exit 4 for missing, duplicate, or unexpected rows. | Gate output in `/tmp/vs3_gate_current.json`; tamper outputs in `/tmp/vs3_missing_direct_gate.json` and `/tmp/vs3_duplicate_unexpected_direct_gate.json`. | Native CLI gate. | PASS for this slice. |

## Command Evidence

### Canonical positive path

```text
PATH="$PWD:$PATH" cornerstone scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
verify_status success
scenario_set vs3-onprem-trusted-extension
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
summary {'blocking': 0, 'fail': 0, 'human_required': 7, 'not_run': 0, 'not_verified': 0, 'pass': 50, 'product_feature_claims': 'VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_GATES_REQUIRED', 'scenario_count': 57}
row_count 57
proof_vs3_p NOT_CLAIMED
human_rows ['VS3-H01', 'VS3-H02', 'VS3-H03', 'VS3-H04', 'VS3-H05', 'VS3-H06', 'VS3-H07']
```

```text
PATH="$PWD:$PATH" cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json
gate_status success
scenario_count 57
coverage_validation_status passed
expected_scenario_count 57
reported_scenario_count 57
missing_scenario_ids []
unexpected_scenario_ids []
duplicate_scenario_ids []
mismatched_summary_fields []
human_required_validation_status passed
claim_boundary_validation_status passed
error_codes []
```

### Negative coverage probes

```text
missing-row tamper:
gate_status failed
error_codes ['CS_VS3_SCENARIO_COVERAGE_INVALID']
coverage_validation_status failed
missing_scenario_ids ['VS3-CTX-001']
mismatched_summary_fields ['scenario_count', 'pass']
human_required_validation_status passed
claim_boundary_validation_status passed
transcript_exit_code 4
shell_exit_code 4
```

```text
duplicate/unexpected-row tamper:
gate_status failed
error_codes ['CS_VS3_SCENARIO_COVERAGE_INVALID']
coverage_validation_status failed
duplicate_scenario_ids ['VS3-CTX-001']
unexpected_scenario_ids ['VS3-UNKNOWN-999']
mismatched_summary_fields ['scenario_count', 'pass']
human_required_validation_status passed
claim_boundary_validation_status passed
transcript_exit_code 4
shell_exit_code 4
```

### Automated checks

```text
python3 -m compileall packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
exit code: 0
```

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_preserves_source_proof_boundary_and_transcript \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_missing_ai_scenario_row \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_duplicate_or_unexpected_rows
Ran 3 tests in 1.101s
OK
```

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_onprem_trusted_extension_verify_closes_local_dev_ai_rows \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_scenario_gate_rejects_report_level_errors \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_preserves_source_proof_boundary_and_transcript \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_without_source_tree \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_without_row_evidence_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_without_aggregate_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_without_source_verify_transcript \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_human_rows_marked_pass \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_promoted_to_vs3_p_readiness \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_missing_ai_scenario_row \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_duplicate_or_unexpected_rows
Ran 11 tests in 29.080s
OK
```

```text
git diff --check
exit code: 0
```

## Implementation Evidence

- `packages/cornerstone_cli/main.py` computes `coverage_validation` from the frozen `vs3-onprem-trusted-extension` scenario registry, rejects missing, unexpected, duplicate, or mismatched summary counts, and emits `CS_VS3_SCENARIO_COVERAGE_INVALID`.
- `tests/scenario/test_scaffold_cli.py` asserts the positive 57-row gate path and both negative coverage-tamper paths.

## Human Required

| ID | Why AI Cannot Verify | Required Human Action | Expected Evidence | Release Impact |
|---|---|---|---|---|
| VS3-H01 through VS3-H07 | These require human/external/on-prem/security/operator evidence. | Complete the human review or external rehearsal named by each row. | Dated, redacted, signed approval/review/topology/provider/UX/migration evidence. | Blocks VS3-P, production/on-prem readiness, live readiness, security acceptance, UX acceptance, and migration/restore readiness. |

## Deliberately Not Done

- Did not claim VS3-P or production/on-prem readiness.
- Did not convert human rows to PASS.
- Did not run a full repository test suite.
- Did not resolve the broader dirty worktree or unrelated generated report churn.
- Did not begin the next VS3 delivery slice after this checkpoint.

## Risks

- The current generated VS3 report records `worktree_dirty_at_verification=true`; this is acceptable local evidence but not a release-clean state.
- The checkpoint proves the scenario-gate coverage guard, not all VS3 security, tenancy, connector, registry, operator, or migration behavior.
- The direct tamper outputs are in `/tmp`; rerun the commands above if durable local files are required.

## Verdict

- AI-verifiable scope: done for VS3-GATE-004 and VS3-REG-004 coverage guard.
- Human/release gate: needs-human-verification for VS3-H01 through VS3-H07.
- Recommendation: continue to the next VS3 slice only after accepting this checkpoint boundary.
