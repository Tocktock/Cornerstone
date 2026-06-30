# VS3 Scenario Gate Row Evidence Ref Taxonomy Guard Checkpoint

**Date:** 2026-06-30 KST
**Status:** Local verifier hardening checkpoint
**Scope:** `cornerstone scenario gate <vs3-report> --json`
**Claim boundary:** VS3-L local/dev evidence guard improved. VS3-P, production/on-prem readiness, live-provider readiness, real IdP readiness, real-network readiness, migration/restore readiness, security acceptance, and human acceptance remain unclaimed.

## Slice Contract

Goal:

- Make standalone VS3 scenario gate reject local-dev assurance reports where row-level `evidence_refs` are nonempty but do not match the accepted VS3 evidence-ref taxonomy.

In this slice:

- `VS3-GATE-004`
- `VS3-REG-004`

Full scenario mapping:

- `in_this_slice`: `VS3-GATE-004`, `VS3-REG-004`
- `HUMAN_REQUIRED`: `VS3-H01`, `VS3-H02`, `VS3-H03`, `VS3-H04`, `VS3-H05`, `VS3-H06`, `VS3-H07`
- `later_slice`: `VS3-GATE-001`, `VS3-GATE-002`, `VS3-GATE-003`, all `VS3-CTX-*`, `VS3-RLS-*`, `VS3-OPA-*`, `VS3-EGR-*`, `VS3-CON-*`, `VS3-TOOL-*`, `VS3-OBS-*`, `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-003`, `VS3-REG-005`, `VS3-REG-006`, `VS3-REG-007`, `VS3-REG-008`

Out of scope:

- New VS3 feature behavior for RequestContext, RLS, OPA, egress, ConnectorHub, Tool SDK, registry, Agent Pack activation, operator UI, or human-gate acceptance.
- Production/on-prem readiness.
- Live provider, real IdP, real network, independent security review, migration/restore, or human UX acceptance.
- Converting human-required rows to PASS.

## Gap Found

Fresh-tree pre-fix probe used a temporary current source-tree report:

```text
python3 -m cornerstone_cli.main scenario verify vs3-onprem-trusted-extension --json --output tmp/vs3-current-for-row-evidence-ref-probe.json
Result: exit 0
```

Probe result:

```text
Mutation: VS3-GATE-001 evidence_refs = ["not-evidence-ref"]
Command: scenario gate tmp/vs3-scenario-gate-bogus-row-evidence_refs.json --json
Observed: exit 0, status success, errors [], row_ref_validation status passed
Observed supporting validators: aggregate_ref_validation passed
```

This allowed an AI `PASS` row to carry an internally present but semantically meaningless evidence reference.

## Implementation

Changed:

- `packages/cornerstone_cli/main.py`
- `tests/scenario/test_scaffold_cli.py`

Behavior:

- For VS3 local-dev assurance reports, every row-level `evidence_refs` list must be nonempty and every nonblank ref must match one accepted evidence-ref category.
- Accepted repo-relative evidence paths start with `docs/`, `reports/`, `scripts/`, `config/`, `policies/`, or `fixtures/`.
- Accepted native command transcript refs start with `cornerstone ` or `git status `.
- Accepted symbolic proof refs are limited to the VS3 fixture/proof labels emitted by the current verifier, such as controlled sinks, sandbox matrices, ConnectorHub quarantine/ack labels, trusted registry negative tests, operator status snapshots, and overclaim lint.
- Absolute paths, parent-directory traversal segments, newline-bearing refs, and unsupported free-form labels are rejected.
- `row_ref_validation` now exposes `malformed_evidence_ref_rows`.
- The existing `CS_VS3_ROW_EVIDENCE_METADATA_MISSING` gate failure now covers missing row evidence refs and malformed row evidence refs.
- Human rows remain `HUMAN_REQUIRED`; this guard validates metadata shape but does not collect or accept human evidence.
- Proof boundaries remain unchanged: `VS3-P`, production/on-prem readiness, live provider readiness, real IdP readiness, migration/restore readiness, security acceptance, and human acceptance remain `NOT_CLAIMED`.

New validation output shape:

```json
{
  "row_ref_validation": {
    "status": "failed",
    "missing_evidence_ref_rows": [],
    "malformed_evidence_ref_rows": [
      {
        "scenario_id": "VS3-GATE-001",
        "evidence_ref": "not-evidence-ref",
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
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_row_evidence_ref \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_unresolved_row_source_report_ref \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_row_audit_ref \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_row_policy_decision_ref
Result: Ran 6 tests in 3.421s, OK
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
