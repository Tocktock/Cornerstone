# VS3 Scenario Gate Row Contract Content Guard Checkpoint - 2026-06-30

**Status:** Local verifier hardening checkpoint.
**Scope:** VS3-GATE-004 and VS3-REG-004.
**Proof boundary:** Local deterministic CLI/test evidence only.
**Non-claims:** This checkpoint does not claim VS3-P, production/on-prem readiness, live-provider readiness, real IdP readiness, migration/restore readiness, security acceptance, or human UX acceptance.

## Slice Contract

Goal:
- Prevent a VS3 local/dev scenario report from passing `cornerstone scenario gate` after frozen scenario contract fields are weakened or rewritten.

In scope:
- Row-level contract content validation for VS3 scenario reports.
- Focused regression test for tampered `verification_method`, `required_evidence`, and `pass_fail_criteria`.
- Direct before/after CLI probe evidence.

Out of scope:
- New VS3 runtime features.
- New ConnectorHub, RLS, OPA, egress, tool registry, or operator UX behavior.
- Any production, live-provider, real-network, real-IdP, migration, security-acceptance, or human-acceptance claim.

Full VS3 mapping:
- `in_this_slice`: `VS3-GATE-004`, `VS3-REG-004`.
- `later_slice`: `VS3-GATE-001`, `VS3-GATE-002`, `VS3-GATE-003`, `VS3-CTX-001` through `VS3-CTX-005`, `VS3-RLS-001` through `VS3-RLS-006`, `VS3-OPA-001` through `VS3-OPA-005`, `VS3-EGR-001` through `VS3-EGR-006`, `VS3-CON-001` through `VS3-CON-006`, `VS3-TOOL-001` through `VS3-TOOL-007`, `VS3-OBS-001` through `VS3-OBS-003`, `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-003`, `VS3-REG-005`, `VS3-REG-006`, `VS3-REG-007`, `VS3-REG-008`.
- `HUMAN_REQUIRED`: `VS3-H01` through `VS3-H07`.

## Before Evidence

Command:

```bash
./cornerstone scenario verify vs3-onprem-trusted-extension --json --output "$base"
python3 - "$base" "$tampered"  # rewrote VS3-GATE-001 verification_method, required_evidence, pass_fail_criteria
./cornerstone scenario gate "$tampered" --json
```

Observed result before this checkpoint:

```text
gate_exit 0
gate_status success
row_identity_issues []
row_classification_issues []
row_contract_issues None
```

Interpretation:
- The gate detected row identity and classification integrity, but did not detect frozen scenario contract content drift.
- This was a false PASS for `VS3-GATE-004` / `VS3-REG-004` because coverage could silently preserve IDs while weakening verification and pass/fail criteria.

## Change

The VS3 scenario gate now compares each reported row against the frozen matrix for these contract fields:

- `given`
- `when`
- `then`
- `implementation_area`
- `verification_method`
- `required_evidence`
- `pass_fail_criteria`

Mismatches are emitted as `coverage_validation.row_contract_issues` and fail the gate with `CS_VS3_SCENARIO_COVERAGE_INVALID`.

## After Evidence

Focused unittest:

```bash
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_preserves_source_proof_boundary_and_transcript \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_row_contract_content_mismatch \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_row_classification_mismatch \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_row_id_scenario_id_mismatch
```

Result:

```text
Ran 4 tests in 2.312s
OK
```

Broader VS3 scenario-gate unittest subset:

```bash
python3 -m unittest $(cat /tmp/cs-vs3-gate-tests.args)
```

Result:

```text
Ran 34 tests in 20.175s
OK
```

Direct tampered-report CLI probe after the change:

```text
gate_exit 4
gate_status failed
error_codes ['CS_VS3_SCENARIO_COVERAGE_INVALID']
coverage_status failed
row_identity_issues []
row_classification_issues []
row_contract_issues [
  verification_method mismatch for VS3-GATE-001,
  required_evidence mismatch for VS3-GATE-001,
  pass_fail_criteria mismatch for VS3-GATE-001
]
```

Untampered baseline CLI probe after the change:

```text
verify_status success
summary {'blocking': 0, 'fail': 0, 'human_required': 7, 'not_run': 0, 'not_verified': 0, 'pass': 50, 'product_feature_claims': 'VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_GATES_REQUIRED', 'scenario_count': 57}
gate_status success
scenario_count 57
blocking_count 0
coverage_status passed
row_contract_issues []
human_required_status passed
claim_boundary_status passed
```

Canonical report refresh after the change:

```bash
./cornerstone scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
./cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json
```

Result:

```text
verify_status success
summary {'blocking': 0, 'fail': 0, 'human_required': 7, 'not_run': 0, 'not_verified': 0, 'pass': 50, 'product_feature_claims': 'VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_GATES_REQUIRED', 'scenario_count': 57}
gate_status success
scenario_count 57
blocking_count 0
coverage_status passed
row_contract_issues []
human_required_status passed
claim_boundary_status passed
errors []
```

Syntax check:

```bash
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
```

Result:

```text
exit 0
```

Documentation and whitespace checks:

```bash
scripts/verify_sot_docs.sh
git diff --check
```

Result:

```text
PASS: CornerStone SoT docs verified
git diff --check exit 0
```

Unavailable check:

```text
python3 -m pytest --version
/opt/homebrew/opt/python@3.14/bin/python3.14: No module named pytest
```

## Remaining Proof Surfaces

- VS3-H01 through VS3-H07 remain `HUMAN_REQUIRED`.
- This checkpoint proves only local gate hardening for row contract content drift.
- It does not replace later scenario-specific runtime proof for RequestContext, Postgres/RLS, OPA, egress sandboxing, ConnectorHub source policy, trusted tool registry, operator status, or VS0/VS1 final regression gates.

## Decision

Continue to the next small VS3 verifier or runtime substrate slice. Do not widen from this checkpoint into VS3-P or human/on-prem claims.
