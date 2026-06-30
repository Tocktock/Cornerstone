# VS3 Scenario Gate Negative Evidence Guard Checkpoint - 2026-06-30

**Status:** Local verifier hardening checkpoint.
**Scope:** VS3-GATE-003, VS3-GATE-004, VS3-REG-003, VS3-REG-005, and VS3-REG-008.
**Proof boundary:** Local deterministic CLI/test evidence only.
**Non-claims:** This checkpoint does not claim VS3-P, production/on-prem readiness, live-provider readiness, real IdP readiness, migration/restore readiness, security acceptance, or human UX acceptance.

## Slice Contract

Goal:
- Prevent a successful VS3 local/dev scenario report from passing `cornerstone scenario gate` when top-level safety `negative_evidence` is missing, malformed, or nonzero.

In scope:
- Top-level `negative_evidence_validation` for VS3 scenario reports.
- Stable gate failure code `CS_VS3_NEGATIVE_EVIDENCE_INVALID`.
- Focused regression tests for missing top-level `negative_evidence` and one nonzero safety counter.
- Direct before/after CLI tamper probes.

Out of scope:
- New VS3 runtime features.
- New ConnectorHub, RLS, OPA, egress, tool registry, or operator UX behavior.
- Any production, live-provider, real-network, real-IdP, migration, security-acceptance, or human-acceptance claim.

Full VS3 mapping:
- `in_this_slice`: `VS3-GATE-003`, `VS3-GATE-004`, `VS3-REG-003`, `VS3-REG-005`, `VS3-REG-008`.
- `later_slice`: `VS3-GATE-001`, `VS3-GATE-002`, `VS3-CTX-001` through `VS3-CTX-005`, `VS3-RLS-001` through `VS3-RLS-006`, `VS3-OPA-001` through `VS3-OPA-005`, `VS3-EGR-001` through `VS3-EGR-006`, `VS3-CON-001` through `VS3-CON-006`, `VS3-TOOL-001` through `VS3-TOOL-007`, `VS3-OBS-001` through `VS3-OBS-003`, `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-004`, `VS3-REG-006`, `VS3-REG-007`.
- `HUMAN_REQUIRED`: `VS3-H01` through `VS3-H07`.

## Before Evidence

After refreshing the canonical report:

```bash
./cornerstone scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
./cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json
```

Observed canonical result:

```text
verify_exit=0
status success
summary {'blocking': 0, 'fail': 0, 'human_required': 7, 'not_run': 0, 'not_verified': 0, 'pass': 50, 'product_feature_claims': 'VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_GATES_REQUIRED', 'scenario_count': 57}
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
canonical_gate_exit=0
canonical_status success
canonical_errors []
```

Tamper probe before this checkpoint:

```text
missing-negative exit 0 status success errors [] negative_validation None
nonzero-negative exit 0 status success errors [] negative_validation None
```

Interpretation:
- The gate accepted a successful VS3 local/dev report after top-level `negative_evidence` was removed.
- The gate also accepted the same report after `negative_evidence.untrusted_content_egress_calls` was set to `1`.
- This contradicted the Local Verification Plane requirement that safety scenarios carry negative evidence such as zero tool calls, zero egress, zero unauthorized action cards, and zero secret leaks.

## Change

The VS3 scenario gate now computes `negative_evidence_validation` for successful local/dev or AI-complete VS3 reports.

The gate fails with `CS_VS3_NEGATIVE_EVIDENCE_INVALID` when:

- top-level `negative_evidence` is missing or empty;
- any key is not a non-empty string;
- any safety counter is nonzero;
- any value is not numeric zero or boolean false.

The validation records:

- whether negative evidence was required;
- whether it was present;
- counter count;
- nonzero entries;
- malformed entries.

## After Evidence

Syntax check:

```bash
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
```

Result:

```text
exit 0
```

Focused unittest:

```bash
python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_without_negative_evidence tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_nonzero_negative_evidence
```

Result:

```text
Ran 2 tests in 1.116s
OK
```

Broader VS3 scenario-gate unittest subset:

```bash
python3 -m unittest $(cat /tmp/cs-vs3-scenario-gate-tests.args)
```

Result:

```text
vs3_scenario_gate_test_count 37
Ran 37 tests in 20.812s
OK
```

Untampered canonical CLI probe after the change:

```text
verify success {'blocking': 0, 'fail': 0, 'human_required': 7, 'not_run': 0, 'not_verified': 0, 'pass': 50, 'product_feature_claims': 'VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_GATES_REQUIRED', 'scenario_count': 57} VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
gate success 0 [] {'count': 190, 'malformed_entries': {}, 'nonzero_entries': {}, 'present': True, 'required': True, 'status': 'passed'}
```

Direct tampered-report CLI probes after the change:

```text
missing-negative exit 4 status failed errors ['CS_VS3_NEGATIVE_EVIDENCE_INVALID'] negative_validation {'count': 0, 'malformed_entries': {}, 'nonzero_entries': {}, 'present': False, 'required': True, 'status': 'failed'}
nonzero-negative exit 4 status failed errors ['CS_VS3_NEGATIVE_EVIDENCE_INVALID'] negative_validation {'count': 190, 'malformed_entries': {}, 'nonzero_entries': {'untrusted_content_egress_calls': 1}, 'present': True, 'required': True, 'status': 'failed'}
```

Pytest availability check:

```bash
python3 -m pytest --version
```

Result:

```text
/opt/homebrew/opt/python@3.14/bin/python3.14: No module named pytest
```

## Remaining Proof Surfaces

- VS3-H01 through VS3-H07 remain `HUMAN_REQUIRED`.
- This checkpoint proves only local scenario-gate hardening for top-level VS3 negative evidence.
- It does not replace later scenario-specific runtime proof for RequestContext, Postgres/RLS, OPA, egress sandboxing, ConnectorHub source policy, trusted tool registry, operator status, or final VS0/VS1 regression gates.

## Decision

Continue to the next small VS3 verifier or runtime substrate slice. Do not widen from this checkpoint into VS3-P, production/on-prem readiness, live-provider readiness, real IdP readiness, migration/restore readiness, security acceptance, or human acceptance.
