# VS3 Scenario Gate Row Audit and Policy Reference Guard Checkpoint

**Date:** 2026-06-30 KST
**Status:** Local verifier hardening checkpoint
**Scope:** `cornerstone scenario gate <vs3-report> --json`

## Slice Contract

Goal:

- Make standalone VS3 scenario gate reject local-dev assurance reports where an AI `PASS` row has missing or malformed row-level audit or policy decision references.

In this slice:

- `VS3-GATE-004`
- `VS3-REG-004`

Full scenario mapping:

- `in_this_slice`: `VS3-GATE-004`, `VS3-REG-004`
- `HUMAN_REQUIRED`: `VS3-H01`, `VS3-H02`, `VS3-H03`, `VS3-H04`, `VS3-H05`, `VS3-H06`, `VS3-H07`
- `later_slice`: `VS3-GATE-001`, `VS3-GATE-002`, `VS3-GATE-003`, all `VS3-CTX-*`, `VS3-RLS-*`, `VS3-OPA-*`, `VS3-EGR-*`, `VS3-CON-*`, `VS3-TOOL-*`, `VS3-OBS-*`, `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-003`, `VS3-REG-005`, `VS3-REG-006`, `VS3-REG-007`, `VS3-REG-008`

Out of scope:

- Generic `evidence_refs` taxonomy. Current evidence refs intentionally mix repo paths, native commands, and symbolic fixture labels; this needs a separate taxonomy slice.
- Production/on-prem readiness.
- Live provider, real IdP, real network, independent security review, migration/restore, or human UX acceptance.
- Converting human-required rows to PASS.

## Gap Found

Fresh-tree pre-fix probes used a temporary current source-tree report:

```text
scenario verify vs3-onprem-trusted-extension --json --output tmp/vs3-current-for-ref-probe.json
Result: exit 0
```

Probe results:

```text
Mutation: VS3-CTX-001 audit_refs = ["not-audit-ref"]
Command: scenario gate tmp/vs3-scenario-gate-bogus-row-audit_refs.json --json
Observed: exit 0, status success, errors [], row_ref_validation status passed

Mutation: VS3-CTX-001 policy_decision_refs = ["not-policy-ref"]
Command: scenario gate tmp/vs3-scenario-gate-bogus-row-policy_decision_refs.json --json
Observed: exit 0, status success, errors [], row_ref_validation status passed
```

This allowed an AI `PASS` row to carry nonempty but semantically invalid audit/policy references.

## Implementation

Changed:

- `packages/cornerstone_cli/main.py`
- `tests/scenario/test_scaffold_cli.py`

Behavior:

- For VS3 local-dev assurance reports, AI `PASS` rows must have nonempty row-level `audit_refs`.
- Each row-level audit ref must start with `audit:` or `audit_`.
- For VS3 local-dev assurance reports, AI `PASS` rows must have nonempty row-level `policy_decision_refs`.
- Each row-level policy decision ref must start with `policy:` or `policy_`.
- Human rows remain `HUMAN_REQUIRED`; this guard does not collect or accept human evidence.
- Proof boundaries remain unchanged: `VS3-P`, production/on-prem readiness, live provider readiness, real IdP readiness, migration/restore readiness, security acceptance, and human acceptance remain `NOT_CLAIMED`.

New validation output:

```json
{
  "row_ref_validation": {
    "status": "failed",
    "malformed_audit_ref_rows": [
      {
        "scenario_id": "VS3-CTX-001",
        "audit_ref": "not-audit-ref",
        "issue": "unsupported_ref_format"
      }
    ],
    "malformed_policy_decision_ref_rows": [
      {
        "scenario_id": "VS3-CTX-001",
        "policy_decision_ref": "not-policy-ref",
        "issue": "unsupported_ref_format"
      }
    ]
  }
}
```

## Verification

Focused checks run:

```text
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
Result: exit 0
```

```text
PYTHONPATH=packages python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_preserves_source_proof_boundary_and_transcript \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_without_row_evidence_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_unresolved_row_source_report_ref \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_row_audit_ref \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_row_policy_decision_ref \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_without_row_policy_decision_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_vs3_0_pass_row_without_audit_refs
Result: Ran 7 tests in 3.968s, OK
```

Native VS3 refresh:

```text
security vs3-evidence-reconcile --json
Result: exit 0, status success

security vs3-overclaim-lint --json
Result: exit 0, status passed

security vs3-request-context --json
Result: exit 0, status success

security vs3-postgres-rls --reuse-vs2-local-range-report reports/security/vs2-local-range.json --json
Result: exit 0, status success

security vs3-opa-policy --reuse-vs2-local-range-report reports/security/vs2-local-range.json --json
Result: exit 0, status success

security vs3-egress-sandbox --reuse-vs2-local-range-report reports/security/vs2-local-range.json --json
Result: exit 0, status success

security vs3-connectorhub-source --json
Result: exit 0, status success

security vs3-tool-registry --json
Result: exit 0, status success

security vs3-observability --json
Result: exit 0, status success

security vs3-regression-gate --json
Result: exit 0, status success
```

Aggregate refresh:

```text
scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
Result: exit 0, status success, 57 scenarios, 50 PASS, 7 HUMAN_REQUIRED

scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json
Result: exit 0, status success
row_ref_validation: status passed; missing/malformed audit and policy row refs []
source_tree_current_validation: status passed, mismatches []
```

Human-gate derived reports:

```text
human-gate evidence-status --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --output reports/human-gates/vs3/evidence-status.json --json
Result: exit 0, status success, final_verdict HUMAN_REQUIRED

human-gate review-kit --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --output reports/human-gates/vs3/review-kit.json --json
Result: exit 0, status success, final_verdict HUMAN_REQUIRED

human-gate vs3-p-gate --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --output reports/human-gates/vs3/vs3-p-gate.json --json
Result: exit 4, status blocked, final_verdict HUMAN_REQUIRED, code CS_VS3_P_GATE_HUMAN_EVIDENCE_REQUIRED
```

Local checkpoint:

```text
security vs3-local-checkpoint --json --output reports/security/vs3-local-checkpoint.json
Result: exit 0, status success
final_verdict: VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
summary: 57 scenarios, 50 PASS, 7 HUMAN_REQUIRED, VS3-L LOCAL_DEV_ASSURANCE_VERIFIED, VS3-P NOT_CLAIMED
negative_evidence: scenario_report_source_tree_current_failures 0, component_proof_report_source_tree_current_failures 0, vs3_p_claimed_by_checkpoint 0, production_readiness_claimed_by_checkpoint 0, security_acceptance_claimed_by_checkpoint 0, human_acceptance_claimed_by_checkpoint 0
```

## Decision

This slice is a local verifier hardening improvement for `VS3-GATE-004` and `VS3-REG-004`.

It does not claim `VS3-P`.
