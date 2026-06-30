# VS3 Scenario Gate Aggregate Row Audit/Policy Lineage Guard Checkpoint

**Date:** 2026-06-30 KST
**Status:** Local verifier hardening checkpoint
**Scope:** `cornerstone scenario gate <vs3-report> --json`
**Claim boundary:** VS3-L local/dev evidence guard improved. VS3-P, production/on-prem readiness, live-provider readiness, real IdP readiness, real-network readiness, migration/restore readiness, security acceptance, and human acceptance remain unclaimed.

## Slice Contract

Goal:

- Make the standalone VS3 scenario gate reject local-dev assurance reports where row-level `audit_refs` or `policy_decision_refs` remain present but the corresponding aggregate report refs are silently omitted.

In this slice:

- `VS3-GATE-004`
- `VS3-REG-004`

Full scenario mapping:

- `in_this_slice`: `VS3-GATE-004`, `VS3-REG-004`
- `HUMAN_REQUIRED`: `VS3-H01`, `VS3-H02`, `VS3-H03`, `VS3-H04`, `VS3-H05`, `VS3-H06`, `VS3-H07`
- `later_slice`: `VS3-GATE-001`, `VS3-GATE-002`, `VS3-GATE-003`, all `VS3-CTX-*`, `VS3-RLS-*`, `VS3-OPA-*`, `VS3-EGR-*`, `VS3-CON-*`, `VS3-TOOL-*`, `VS3-OBS-*`, `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-003`, `VS3-REG-005`, `VS3-REG-006`, `VS3-REG-007`, `VS3-REG-008`

Out of scope:

- Production/on-prem readiness.
- Live provider, real IdP, real network, independent security review, migration/restore, or human UX acceptance.
- Converting human-required rows to PASS.
- New VS3 feature-family implementation beyond scenario-gate validation.

## Gap Found

Fresh current-tree pre-fix probes generated a temporary VS3 report, then removed one row-level ref from the matching aggregate list while synchronizing `command_transcripts`, `self_command_transcript`, `stdout_json`, and `ref_summary`.

Seed command:

```text
PYTHONPATH=packages python3 -m cornerstone_cli.main scenario verify vs3-onprem-trusted-extension --json --output tmp/vs3-row-aggregate-lineage-source.json
Result: exit 0, status success, pass 50, human_required 7
```

Audit-ref omission probe:

```text
Mutation: remove audit:vs3_evidence_reconciliation:vs2_conflict_classified from aggregate audit_refs only.
Row left intact: VS3-GATE-001 audit_refs still included audit:vs3_evidence_reconciliation:vs2_conflict_classified.
Observed before fix:
gate_exit 0
status success
error_codes []
row_ref_status passed
aggregate_status passed
source_transcript_status passed
self_transcript_status passed
```

Policy-decision-ref omission probe:

```text
Mutation: remove policy:vs3_evidence_reconciliation:conservative_vs2_boundary from aggregate policy_decision_refs only.
Row left intact: VS3-GATE-001 policy_decision_refs still included policy:vs3_evidence_reconciliation:conservative_vs2_boundary.
Observed before fix:
gate_exit 0
status success
error_codes []
row_ref_status passed
aggregate_status passed
source_transcript_status passed
self_transcript_status passed
```

This meant a local/dev assurance report could keep row-level audit or policy lineage while silently dropping that lineage from the aggregate report refs used by release/report consumers.

## Implementation

Changed:

- `packages/cornerstone_cli/main.py`
- `tests/scenario/test_scaffold_cli.py`

Behavior:

- `aggregate_ref_validation` now verifies that every nonblank row `audit_refs` value appears in top-level aggregate `audit_refs`.
- `aggregate_ref_validation` now verifies that every nonblank row `policy_decision_refs` value appears in top-level aggregate `policy_decision_refs`.
- Failures are reported as:
  - `row_audit_ref_missing_from_audit_refs`
  - `row_policy_decision_ref_missing_from_policy_decision_refs`
- The existing `CS_VS3_AGGREGATE_EVIDENCE_METADATA_MISSING` gate failure now covers row-to-aggregate audit and policy lineage omissions.
- Human rows remain `HUMAN_REQUIRED`; this guard does not collect or accept human evidence.

New validation output shape:

```json
{
  "aggregate_ref_validation": {
    "status": "failed",
    "row_audit_ref_missing_from_audit_refs": [
      {
        "scenario_id": "VS3-GATE-001",
        "audit_ref": "audit:vs3_evidence_reconciliation:vs2_conflict_classified",
        "issue": "row_audit_ref_missing_from_aggregate_audit_refs"
      }
    ],
    "row_policy_decision_ref_missing_from_policy_decision_refs": [
      {
        "scenario_id": "VS3-GATE-001",
        "policy_decision_ref": "policy:vs3_evidence_reconciliation:conservative_vs2_boundary",
        "issue": "row_policy_decision_ref_missing_from_aggregate_policy_decision_refs"
      }
    ]
  }
}
```

## Verification

Syntax:

```text
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
Result: exit 0
```

Focused tests:

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_row_audit_ref_missing_from_aggregate_audit_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_row_policy_decision_ref_missing_from_aggregate_policy_decision_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_row_source_report_ref_missing_from_aggregate_evidence_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_aggregate_audit_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_aggregate_policy_decision_refs
Result: Ran 5 tests in 2.926s, OK
```

Post-fix audit-ref omission probe:

```text
variant audit_refs
target_row VS3-GATE-001
removed_ref audit:vs3_evidence_reconciliation:vs2_conflict_classified
gate_exit 4
status failed
error_codes ['CS_VS3_AGGREGATE_EVIDENCE_METADATA_MISSING']
row_ref_status passed
aggregate_status failed
row_audit_missing [{'audit_ref': 'audit:vs3_evidence_reconciliation:vs2_conflict_classified', 'issue': 'row_audit_ref_missing_from_aggregate_audit_refs', 'scenario_id': 'VS3-GATE-001'}]
row_policy_missing []
source_transcript_status passed
self_transcript_status passed
```

Post-fix policy-decision-ref omission probe:

```text
variant policy_decision_refs
target_row VS3-GATE-001
removed_ref policy:vs3_evidence_reconciliation:conservative_vs2_boundary
gate_exit 4
status failed
error_codes ['CS_VS3_AGGREGATE_EVIDENCE_METADATA_MISSING']
row_ref_status passed
aggregate_status failed
row_audit_missing []
row_policy_missing [{'issue': 'row_policy_decision_ref_missing_from_aggregate_policy_decision_refs', 'policy_decision_ref': 'policy:vs3_evidence_reconciliation:conservative_vs2_boundary', 'scenario_id': 'VS3-GATE-001'}]
source_transcript_status passed
self_transcript_status passed
```

## Required Final Refresh

After this checkpoint file is written, rerun the local VS3 verifier/report stack before using the updated source tree as a local/dev assurance signal:

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

It does not claim `VS3-P`, production/on-prem readiness, live-provider readiness, real IdP readiness, real-network readiness, migration/restore readiness, security acceptance, or human acceptance.
