# VS3 Scenario Gate Row Identity Consistency Guard Checkpoint

**Date:** 2026-06-30 KST
**Status:** Local verifier hardening checkpoint
**Scope:** `cornerstone scenario gate <vs3-report> --json`
**Claim boundary:** VS3-L local/dev scenario-gate evidence guard improved. VS3-P, production/on-prem readiness, live-provider readiness, real IdP readiness, real-network readiness, migration/restore readiness, security acceptance, and human acceptance remain unclaimed.

## Slice Contract

Goal:

- Make the standalone VS3 scenario gate reject local-dev assurance reports where a row's public `id` and canonical `scenario_id` disagree.

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

Fresh current-tree pre-fix probe generated a temporary VS3 report, then changed only `scenario_results[0].id` while leaving `scenario_id`, summary counts, traceability, source transcripts, row refs, aggregate refs, human-required boundaries, and proof boundaries intact.

Seed command:

```text
PYTHONPATH=packages python3 -m cornerstone_cli.main scenario verify vs3-onprem-trusted-extension --json --output tmp/vs3-row-identity-source.json
Result: exit 0, status success
```

Identity mutation:

```text
Before: id VS3-GATE-001, scenario_id VS3-GATE-001
After:  id VS3-GATE-999, scenario_id VS3-GATE-001
```

Observed before fix:

```text
gate_exit 0
status success
error_codes []
coverage_validation passed
traceability_validation passed
row_ref_validation passed
aggregate_ref_validation passed
source_transcript_validation passed
human_required_validation passed
claim_boundary_validation passed
coverage_missing []
coverage_unexpected []
coverage_duplicates []
```

This allowed one report row to carry two competing identities. That is unsafe because report readers and tests already consume both fields.

## Implementation

Changed:

- `packages/cornerstone_cli/main.py`
- `tests/scenario/test_scaffold_cli.py`

Behavior:

- `coverage_validation` now includes `row_identity_issues`.
- Every VS3 row must be a JSON object with nonblank `id` and nonblank `scenario_id`.
- For every VS3 row, `id` and `scenario_id` must match exactly.
- Mismatches fail `coverage_validation` and emit the existing `CS_VS3_SCENARIO_COVERAGE_INVALID` gate error.
- Human rows remain `HUMAN_REQUIRED`; this guard does not collect or accept human evidence.

New validation output shape:

```json
{
  "coverage_validation": {
    "status": "failed",
    "row_identity_issues": [
      {
        "row_index": 0,
        "id": "VS3-GATE-999",
        "scenario_id": "VS3-GATE-001",
        "issue": "id_scenario_id_mismatch"
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
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_preserves_source_proof_boundary_and_transcript \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_row_id_scenario_id_mismatch \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_missing_ai_scenario_row \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_duplicate_or_unexpected_rows
Result: Ran 4 tests in 2.316s, OK
```

Post-fix identity mutation probe:

```text
seed_exit 0
before_row_id VS3-GATE-001 scenario_id VS3-GATE-001
after_row_id VS3-GATE-999 scenario_id VS3-GATE-001
gate_exit 4
status failed
error_codes ['CS_VS3_SCENARIO_COVERAGE_INVALID']
coverage_validation failed
traceability_validation passed
row_ref_validation passed
aggregate_ref_validation passed
source_transcript_validation passed
human_required_validation passed
claim_boundary_validation passed
row_identity_issues [{'id': 'VS3-GATE-999', 'issue': 'id_scenario_id_mismatch', 'row_index': 0, 'scenario_id': 'VS3-GATE-001'}]
coverage_missing []
coverage_unexpected []
coverage_duplicates []
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
