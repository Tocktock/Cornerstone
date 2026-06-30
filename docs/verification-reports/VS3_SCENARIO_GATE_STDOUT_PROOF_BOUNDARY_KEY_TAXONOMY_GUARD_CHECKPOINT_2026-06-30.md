# VS3 Scenario Gate Stdout Proof-Boundary Key Taxonomy Guard Checkpoint - 2026-06-30

**Status:** Local deterministic checkpoint for one VS3 verifier hardening slice.
**Scope:** Native VS3 command-transcript `stdout_json.proof_boundary` key taxonomy.
**Related rows:** `VS3-GATE-004`, proof-boundary support for `VS3-REG-005`.
**Proof boundary:** Local CLI/test evidence only. This is not VS3-L completion, VS3-P readiness, production/on-prem readiness, security acceptance, live-provider readiness, migration/restore readiness, or human acceptance.

## Full VS3 Mapping

Full VS3 inventory remains:

- `42` `MUST_PASS`
- `8` `REGRESSION`
- `7` `HUMAN_REQUIRED`
- `57` total rows

Execution classification for this slice:

- In this slice: `VS3-GATE-004`, proof-boundary support for `VS3-REG-005`.
- Later slice: all other AI-owned `VS3-*` `MUST_PASS` and `REGRESSION` rows.
- Human-required: `VS3-H01` through `VS3-H07`.

## Slice Contract

Goal:

- Make `cornerstone scenario gate <report> --json` reject a VS3 local-dev assurance report when native CLI command transcripts contain unrecognized `stdout_json.proof_boundary` keys, even if top-level proof boundaries remain safe.

In scope:

- Exact allowed-key taxonomy for `command_transcripts[*].stdout_json.proof_boundary`.
- Exact allowed-key taxonomy for `self_command_transcript.stdout_json.proof_boundary`.
- Native JSON gate failure under `source_transcript_validation`.
- Focused regression coverage and direct tamper probe.

Out of scope:

- RequestContext, RLS, OPA, egress, ConnectorHub live/provider flows, Tool SDK, signed registry, Agent Pack activation, and all VS3 human evidence gates.

Expected behavior:

- Clean VS3 source command transcripts include only known local-dev stdout proof-boundary keys.
- Adding `stdout_json.proof_boundary.onprem_security_acceptance = CLAIMED` to the source verify transcript fails the gate.
- Adding `stdout_json.proof_boundary.onprem_security_acceptance = CLAIMED` to the source report's self transcript fails the gate.
- The failure is attributed to `source_transcript_validation.invalid_fields` while row-ref, aggregate-ref, claim-boundary, coverage, and human-required validation remain passable.

## Before Evidence

Before this patch, the same-path tamper probe passed when `stdout_json.proof_boundary.onprem_security_acceptance = CLAIMED` was added to source and self command transcripts:

```text
verify_exit 0
verify_status success
verify_summary {'blocking': 0, 'fail': 0, 'human_required': 7, 'not_run': 0, 'not_verified': 0, 'pass': 50, 'product_feature_claims': 'VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_GATES_REQUIRED', 'scenario_count': 57}
gate_exit 0
gate_status success
error_codes []
source_transcript_validation {'invalid_fields': [], 'mismatched_ref_fields': [], 'missing_fields': [], 'status': 'passed'}
self_command_transcript_validation {'error_codes': [], ... 'status': 'passed', 'valid': True}
```

Interpretation:

- The gate checked known stdout proof-boundary fields.
- The gate did not reject extra stdout proof-boundary keys that could carry additional readiness or acceptance claims inside native CLI evidence.

## Change Summary

Changed:

- `packages/cornerstone_cli/main.py`
  - Adds exact allowed-key validation inside `_vs3_cli_command_transcript_errors`.
  - Rejects unrecognized `stdout_json.proof_boundary` keys with `stdout_json_proof_boundary_<key>_unexpected`.
- `tests/scenario/test_scaffold_cli.py`
  - Adds `test_vs3_scenario_gate_rejects_local_dev_claim_with_source_transcript_stdout_proof_boundary_extra_key`.
  - Verifies the source verify transcript and source self transcript both reject `onprem_security_acceptance`.
  - Asserts unrelated row-ref, aggregate-ref, claim-boundary, coverage, and human-required validators remain green.

## Verification Evidence

Compile:

```text
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
exit 0
```

Focused regression:

```text
PYTHONPATH=packages python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_source_transcript_stdout_proof_boundary_extra_key
.
----------------------------------------------------------------------
Ran 1 test in 24.004s

OK
```

Adjacent source-transcript regression suite after refreshing `reports/security/vs3-final-regression-proof.json`:

```text
PYTHONPATH=packages python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_without_source_verify_transcript \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_source_transcript_missing_stdout_json \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_source_transcript_stdout_proof_boundary_extra_key \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_source_transcript_stdout_elapsed_mismatch \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_source_transcript_stdout_json_schema_mismatch \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_source_transcript_stdout_tail_mismatch
......
----------------------------------------------------------------------
Ran 6 tests in 143.667s

OK
```

Fresh direct tamper probe:

```text
verify_exit 0
verify_status success
verify_summary {'blocking': 0, 'fail': 0, 'human_required': 7, 'not_run': 0, 'not_verified': 0, 'pass': 50, 'product_feature_claims': 'VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_GATES_REQUIRED', 'scenario_count': 57}
gate_exit 4
gate_status failed
error_codes ['CS_VS3_SOURCE_TRANSCRIPT_METADATA_MISSING']
source_transcript_validation {'invalid_fields': ['command_transcripts.scenario_verify_vs3.stdout_json_proof_boundary_onprem_security_acceptance_unexpected', 'self_command_transcript.stdout_json_proof_boundary_onprem_security_acceptance_unexpected'], 'mismatched_ref_fields': [], 'missing_fields': [], 'status': 'failed'}
self_command_transcript_validation {'error_codes': [], ... 'status': 'passed', 'valid': True}
```

## Decision

This slice passes locally for native VS3 command-transcript stdout proof-boundary key taxonomy.

Remaining:

- `VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`.
- Full VS3-L and VS3-P remain unclaimed by this checkpoint.
