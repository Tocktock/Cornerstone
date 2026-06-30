# VS3 Human-Gate Validate-Record Load Failure Output Guard Checkpoint

**Date:** 2026-06-30 KST
**Status:** Local deterministic checkpoint for one VS3 human-gate evidence-preparation guard.
**Scope:** `cornerstone human-gate validate-record --scope vs3 --json --output ...`

## Slice Contract

Goal:
- Ensure early `validate-record` load failures still produce a durable, redacted JSON envelope when `--output` is provided.

In this slice:
- `VS3-GATE-004`: native CLI JSON failure envelope for a VS3 verifier/support gate.
- `VS3-OBS-003`: human-gate evidence-preparation output remains preparation only.
- `VS3-REG-005`: failure output does not overclaim VS3-P, production readiness, security acceptance, migration readiness, or human acceptance.

Out of scope:
- Any VS3-H row promotion.
- VS3-P readiness.
- Production/on-prem, live-provider, real-IdP, real-network, security-acceptance, migration/restore, or human-UX acceptance evidence.

## Change Summary

`human-gate validate-record` now attaches the VS3 no-claim boundary and writes the redacted JSON envelope before returning for these early failure modes:

- record file not found;
- record file is invalid JSON;
- record file is valid JSON but not a JSON object.

The failure envelope keeps:

- `final_verdict=HUMAN_REQUIRED`;
- `weakest_applicable_scenario_result=HUMAN_REQUIRED`;
- `record_file_path_recorded=false`;
- only `record_file_path_sha256` for the input path;
- zero counters for product claim, PASS claim, VS3-P unlock, production readiness, security acceptance, migration readiness, human acceptance, record-body persistence, and record-path persistence.

## Verification Evidence

Syntax check:

```text
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
exit=0
```

Focused tests:

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_human_gate_validate_record_load_failures_write_redacted_output \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_human_gate_validate_record_is_structural_and_redacted

Ran 2 tests in 1.044s
OK
```

Manual CLI smoke:

```text
./cornerstone human-gate validate-record \
  --scope vs3 \
  --scenario VS3-H01 \
  --record-file tmp/vs3-human-gate-cli-smoke/missing.json \
  --output tmp/vs3-human-gate-cli-smoke/missing-output.json \
  --json

exit=1
output_exists=True
status=failed
final_verdict=HUMAN_REQUIRED
error_code=CS_VS3_HUMAN_GATE_RECORD_NOT_FOUND
record_file_path_recorded=False
vs3_p_unlock_allowed=False
stdout_has_raw_record_path=False
```

## Decision

The slice improves local VS3 human-gate evidence preparation only. It does not change the status of `VS3-H01` through `VS3-H07`; all remain `HUMAN_REQUIRED` until signed, dated, redacted human evidence exists and a separate owner-approved promotion decision is made.
