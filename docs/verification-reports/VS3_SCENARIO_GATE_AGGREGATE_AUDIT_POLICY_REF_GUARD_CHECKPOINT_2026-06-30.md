# VS3 Scenario Gate Aggregate Audit and Policy Reference Guard Checkpoint

**Date:** 2026-06-30 KST
**Status:** Local verifier hardening checkpoint
**Scope:** `cornerstone scenario gate <vs3-report> --json`
**Claim boundary:** VS3-L local/dev evidence guard improved. VS3-P, production/on-prem readiness, live-provider readiness, real IdP readiness, real-network readiness, migration/restore readiness, security acceptance, and human acceptance remain unclaimed.

## Slice Contract

Goal:

- Make standalone VS3 scenario gate reject local-dev assurance reports where aggregate `audit_refs` or `policy_decision_refs` are nonempty but semantically malformed.

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
python3 -m cornerstone_cli.main scenario verify vs3-onprem-trusted-extension --json --output tmp/vs3-current-for-aggregate-ref-probe.json
Result: exit 0
```

Probe results:

```text
Mutation: aggregate audit_refs = ["not-audit-ref"]
Command: scenario gate tmp/vs3-scenario-gate-bogus-aggregate-audit_refs.json --json
Observed: exit 0, status success, errors [], aggregate_ref_validation status passed
Observed supporting validators: source_transcript_validation passed, self_command_transcript_validation passed
```

```text
Mutation: aggregate policy_decision_refs = ["not-policy-ref"]
Command: scenario gate tmp/vs3-scenario-gate-bogus-aggregate-policy_decision_refs.json --json
Observed: exit 0, status success, errors [], aggregate_ref_validation status passed
Observed supporting validators: source_transcript_validation passed, self_command_transcript_validation passed
```

This allowed a local-dev assurance report to carry internally consistent but semantically invalid aggregate audit/policy references.

## Implementation

Changed:

- `packages/cornerstone_cli/main.py`
- `tests/scenario/test_scaffold_cli.py`

Behavior:

- For VS3 local-dev assurance reports, aggregate `audit_refs` must be nonempty and each nonblank ref must use the supported audit reference shape.
- Supported aggregate audit refs start with `audit:` or `audit_`.
- For VS3 local-dev assurance reports, aggregate `policy_decision_refs` must be nonempty and each nonblank ref must use the supported policy decision reference shape.
- Supported aggregate policy decision refs start with `policy:` or `policy_`.
- `aggregate_ref_validation` now exposes `malformed_audit_refs` and `malformed_policy_decision_refs`.
- The existing `CS_VS3_AGGREGATE_EVIDENCE_METADATA_MISSING` gate failure now covers both missing aggregate ref fields and malformed aggregate audit/policy refs.
- Human rows remain `HUMAN_REQUIRED`; this guard does not collect or accept human evidence.
- Proof boundaries remain unchanged: `VS3-P`, production/on-prem readiness, live provider readiness, real IdP readiness, migration/restore readiness, security acceptance, and human acceptance remain `NOT_CLAIMED`.

New validation output shape:

```json
{
  "aggregate_ref_validation": {
    "status": "failed",
    "missing_ref_fields": [],
    "malformed_audit_refs": ["not-audit-ref"],
    "malformed_policy_decision_refs": ["not-policy-ref"]
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
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_without_aggregate_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_aggregate_audit_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_aggregate_policy_decision_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_row_audit_ref \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_row_policy_decision_ref
Result: Ran 6 tests in 3.332s, OK
```

Required final refresh after this checkpoint file is written:

```text
cornerstone security vs3-evidence-reconcile --json
cornerstone security vs3-overclaim-lint --json
cornerstone security vs3-request-context --json
cornerstone security vs3-postgres-rls --reuse-vs2-local-range-report reports/security/vs2-local-range.json --json
cornerstone security vs3-opa-policy --reuse-vs2-local-range-report reports/security/vs2-local-range.json --json
cornerstone security vs3-egress-sandbox --reuse-vs2-local-range-report reports/security/vs2-local-range.json --json
cornerstone security vs3-connectorhub-source --json
cornerstone security vs3-tool-registry --json
cornerstone security vs3-observability --json
cornerstone security vs3-regression-gate --json
cornerstone scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json
cornerstone human-gate evidence-status --scope vs3 --use-existing --json --output reports/human-gates/vs3/evidence-status.json
cornerstone human-gate review-kit --scope vs3 --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --use-existing --json --output reports/human-gates/vs3/review-kit.json
cornerstone human-gate vs3-p-gate --scope vs3 --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --output reports/human-gates/vs3/vs3-p-gate.json --json
cornerstone security vs3-local-checkpoint --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --output reports/security/vs3-local-checkpoint.json --json
```

## Decision

This slice is a local verifier hardening improvement for `VS3-GATE-004` and `VS3-REG-004`.

It does not claim `VS3-P`.
