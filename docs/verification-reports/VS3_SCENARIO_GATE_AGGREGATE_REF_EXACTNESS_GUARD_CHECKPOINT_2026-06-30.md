# VS3 Scenario Gate Aggregate Ref Exactness Guard Checkpoint - 2026-06-30

**Status:** Local deterministic checkpoint for one VS3 verifier hardening slice.
**Scope:** Native VS3 scenario gate aggregate evidence/audit/policy ref exactness.
**Related rows:** `VS3-GATE-004`, `VS3-REG-004`, proof-boundary support for `VS3-REG-005`.
**Proof boundary:** Local CLI/test evidence only. This is not VS3-L completion, VS3-P readiness, production/on-prem readiness, security acceptance, live-provider readiness, migration/restore readiness, or human acceptance.

## Full VS3 Mapping

Full VS3 inventory remains:

- `42` `MUST_PASS`
- `8` `REGRESSION`
- `7` `HUMAN_REQUIRED`
- `57` total rows

Execution classification for this slice:

- In this slice: `VS3-GATE-004`, `VS3-REG-004`, proof-boundary support for `VS3-REG-005`.
- Later slice: all other AI-owned `VS3-*` `MUST_PASS` and `REGRESSION` rows.
- Human-required: `VS3-H01` through `VS3-H07`.

## Slice Contract

Goal:

- Make `cornerstone scenario gate <report> --json` reject a VS3 local-dev assurance report when top-level aggregate `evidence_refs`, `audit_refs`, or `policy_decision_refs` contain valid-looking refs that are not owned by any scenario row.

In scope:

- Aggregate `evidence_refs` exactness against row `evidence_refs` plus `source_report_refs`
- Aggregate `audit_refs` exactness against row `audit_refs`
- Aggregate `policy_decision_refs` exactness against row `policy_decision_refs`
- Native JSON gate output and focused regression coverage

Out of scope:

- RequestContext, RLS, OPA, egress, ConnectorHub live/provider flows, Tool SDK, signed registry, Agent Pack activation, and all VS3 human evidence gates.

Expected behavior:

- Clean VS3 aggregate refs equal the union of row-owned refs.
- Adding a syntactically valid but row-unowned aggregate ref fails the gate with `CS_VS3_AGGREGATE_EVIDENCE_METADATA_MISSING`.
- The failure must be attributed to `aggregate_ref_validation.unexpected_*_refs` while row-level refs, source transcripts, claim boundary, and coverage remain passable.

## Before Evidence

Before this patch, the source report's aggregate refs were cleanly exact:

```text
evidence_refs aggregate_count 94 row_union_count 94 extra_aggregate_count 0 missing_aggregate_count 0
audit_refs aggregate_count 199 row_union_count 199 extra_aggregate_count 0 missing_aggregate_count 0
policy_decision_refs aggregate_count 124 row_union_count 124 extra_aggregate_count 0 missing_aggregate_count 0
```

But a report with three unused aggregate refs passed when the refs were also mirrored into the source transcripts:

```text
seed_exit 0
gate_exit 0
gate_status success
error_codes []
aggregate_ref_validation passed
  malformed_evidence_refs []
  malformed_audit_refs []
  malformed_policy_decision_refs []
source_transcript_validation passed
  mismatched_ref_fields []
row_ref_validation passed
claim_boundary_validation passed
coverage_validation passed
aggregate counts 95 200 125
```

Interpretation:

- The gate verified row refs were included in aggregate refs.
- The gate did not reject aggregate refs that no scenario row owned.
- A report could overstate evidence/audit/policy surfaces without changing row evidence.

## Change Summary

Changed:

- `packages/cornerstone_cli/main.py`
  - Computes expected aggregate `evidence_refs` from row `evidence_refs` plus row `source_report_refs`.
  - Computes expected aggregate `audit_refs` from row `audit_refs`.
  - Computes expected aggregate `policy_decision_refs` from row `policy_decision_refs`.
  - Adds `unexpected_evidence_refs`, `unexpected_audit_refs`, and `unexpected_policy_decision_refs` to `aggregate_ref_validation`.
  - Fails `aggregate_ref_validation` when any row-unowned aggregate ref is present after row-level ref validation has passed.
- `tests/scenario/test_scaffold_cli.py`
  - Adds `test_vs3_scenario_gate_rejects_local_dev_claim_with_unowned_aggregate_refs`.
  - Adds valid-looking unused aggregate refs to the source report and mirrors them into source command transcripts.
  - Asserts unrelated row-ref, source-transcript, traceability, coverage, human-required, and claim-boundary validators remain green.

## Verification Evidence

Compile:

```text
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
exit 0
```

Focused regression:

```text
PYTHONPATH=packages python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_unowned_aggregate_refs
.
----------------------------------------------------------------------
Ran 1 test in 23.367s

OK
```

Post-patch direct tamper probe:

```text
seed_exit 0
extra_preexisting evidence_refs False
extra_preexisting audit_refs False
extra_preexisting policy_decision_refs False
gate_exit 4
gate_status failed
error_codes ['CS_VS3_AGGREGATE_EVIDENCE_METADATA_MISSING']
aggregate_ref_validation failed
unexpected_evidence_refs ['docs/sot/README.md']
unexpected_audit_refs ['audit:unused_extra_aggregate_ref']
unexpected_policy_decision_refs ['policy:unused_extra_aggregate_ref']
source_transcript_validation passed []
row_ref_validation passed
claim_boundary_validation passed
coverage_validation passed
```

Adjacent aggregate/row-ref regression suite:

```text
PYTHONPATH=packages python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_unowned_aggregate_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_row_evidence_ref_missing_from_aggregate_evidence_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_duplicate_aggregate_refs \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_malformed_row_evidence_ref \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_missing_row_evidence_path \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_missing_row_evidence_or_evidence_paths \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_duplicate_row_refs
.......
----------------------------------------------------------------------
Ran 7 tests in 167.387s

OK
```

## Decision

This slice passes locally for native VS3 scenario gate aggregate ref exactness.

Remaining:

- `VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`.
- Full VS3-L and VS3-P remain unclaimed by this checkpoint.
