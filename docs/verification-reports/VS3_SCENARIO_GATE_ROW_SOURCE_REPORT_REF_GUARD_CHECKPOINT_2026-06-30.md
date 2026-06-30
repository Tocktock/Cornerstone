# VS3 Scenario Gate Row Source-Report Reference Guard Checkpoint

**Date:** 2026-06-30 KST
**Status:** Local verifier hardening checkpoint
**Scope:** `cornerstone scenario gate <vs3-report> --json`

## Slice Contract

Goal:

- Make standalone VS3 scenario gate reject local-dev assurance reports where an AI `PASS` row has nonempty but unresolved `source_report_refs`.

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
- Changing VS3 component proof semantics beyond row source-report reference validation.

## Gap Found

Pre-fix probe:

```text
Input: tmp/vs3-scenario-gate-bogus-row-source-report-ref-probe.json
Mutation: VS3-CTX-001 source_report_refs = ["reports/security/DOES_NOT_EXIST_VS3_PROOF.json"]
Command: PYTHONPATH=packages python3 -m cornerstone_cli.main scenario gate tmp/vs3-scenario-gate-bogus-row-source-report-ref-probe.json --json
Observed: exit 0, status success, errors []
row_ref_validation: status passed
source_transcript_validation: status passed
```

This allowed an AI `PASS` row to cite a missing source proof file while the standalone scenario gate still passed.

## Implementation

Changed:

- `packages/cornerstone_cli/main.py`
- `tests/scenario/test_scaffold_cli.py`

Behavior:

- For VS3 local-dev assurance reports, AI `PASS` rows must have `source_report_refs` that resolve to repo-local existing files.
- Absolute paths, paths outside the repo, missing paths, and directories fail row reference validation.
- Human rows remain `HUMAN_REQUIRED`; this guard does not convert human evidence into `PASS`.
- Proof boundaries remain unchanged: `VS3-P`, production/on-prem readiness, live provider readiness, real IdP readiness, migration/restore readiness, security acceptance, and human acceptance remain `NOT_CLAIMED`.

New validation output:

```json
{
  "row_ref_validation": {
    "status": "failed",
    "unresolved_source_report_ref_rows": [
      {
        "scenario_id": "VS3-CTX-001",
        "source_report_ref": "reports/security/DOES_NOT_EXIST_VS3_PROOF.json",
        "issue": "missing"
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
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_vs3_0_pass_row_without_audit_refs
Result: Ran 4 tests in 2.209s, OK
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
row_ref_validation: status passed, unresolved_source_report_ref_rows []
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

Docs and whitespace:

```text
scripts/verify_sot_docs.sh
Result: PASS

git diff --check
Result: exit 0
```

## Decision

This slice is a local verifier hardening improvement for `VS3-GATE-004` and `VS3-REG-004`.

It does not claim `VS3-P`.
