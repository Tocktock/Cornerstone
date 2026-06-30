# VS3 Scenario Traceability Report Path Binding Guard Checkpoint - 2026-06-30

## Scope

This checkpoint covers a narrow VS3 local/dev proof-boundary hardening slice.

Applicable frozen VS3 rows:

- `VS3-GATE-003`: VS3 local/dev report must not overclaim beyond local/dev evidence.
- `VS3-GATE-004`: VS3 native scenario verifier/gate must emit concrete gate metadata and transcript evidence.
- `VS3-REG-005`: VS3 reports, help, and release metadata must not describe stronger readiness than the evidence proves.

Out of scope:

- VS3-P production/on-prem readiness.
- Real IdP, live provider, real network, migration/restore, independent security acceptance, or human UX acceptance.
- Human rows `VS3-H01` through `VS3-H07`, which remain `HUMAN_REQUIRED`.

## Baseline Gap

Before this slice, the scenario gate accepted a tampered VS3 local/dev report whose aggregate `traceability.transcript_paths` and every row-level `transcript_paths` value pointed at a different report path:

```text
seed_exit 0
gate_exit 0
gate_status success
errors []
traceability_validation {'invalid_fields': [], 'missing_fields': [], 'row_invalid_fields': [], 'row_missing_fields': [], 'status': 'passed'}
```

The tampered path was:

```text
reports/scenario/not-the-checked-vs3-report.json
```

This meant traceability metadata was present, but it was not bound to the exact report being gated.

## Implementation

Changed files:

- `packages/cornerstone_cli/main.py`
- `tests/scenario/test_scaffold_cli.py`

Guard behavior added:

- `_vs3_scenario_report_traceability_validation(...)` now accepts `expected_report_path`.
- Local checkpoint, human review kit, human evidence status, and VS3-P gate pass their resolved scenario-report path into that validator.
- `cornerstone scenario gate <report> --json` now requires:
  - `traceability.transcript_paths == [<checked report path>]`;
  - every scenario row `transcript_paths == [<checked report path>]`.
- Gate payload now records:
  - `traceability_validation.expected_transcript_paths`;
  - `traceability_validation.actual_transcript_paths`.

## Regression Test

Added test:

```text
tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_wrong_traceability_transcript_path
```

Expected failure for tampered report:

```text
exit 4
status failed
error CS_VS3_TRACEABILITY_METADATA_MISSING
traceability_validation.status failed
expected_transcript_paths ['tmp/vs3-local-dev-wrong-trace-path.json']
actual_transcript_paths ['reports/scenario/not-the-checked-vs3-report.json']
```

## Verification

Syntax:

```text
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
exit 0
```

Focused gate tests:

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_preserves_source_proof_boundary_and_transcript \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_wrong_traceability_transcript_path

Ran 2 tests in 53.040s
OK
```

Neighboring checkpoint/human-gate traceability tests:

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_manifest_is_hash_backed_and_boundary_safe \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_scenario_report_missing_traceability \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_human_gate_derivatives_reject_scenario_report_missing_traceability \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_p_gate_blocks_on_human_evidence_not_ai_rows

Ran 4 tests in 193.496s
OK
```

Manual tamper after patch:

```text
seed_exit 0
gate_exit 4
gate_status failed
errors ['CS_VS3_TRACEABILITY_METADATA_MISSING']
traceability_status failed
expected ['tmp/vs3-trace-path-after-tampered.json']
actual ['reports/scenario/not-the-checked-vs3-report.json']
```

Component proof refresh after source changes:

```text
./cornerstone security vs3-request-context --json
./cornerstone security vs3-postgres-rls --json
./cornerstone security vs3-opa-policy --json
./cornerstone security vs3-egress-sandbox --json
./cornerstone security vs3-connectorhub-source --json
./cornerstone security vs3-tool-registry --json
./cornerstone security vs3-observability --json

all exit 0
all status success
refreshed source hash 2657033006b57d3a1358201445e61a0472c6619b0135d948cef0d2d69a4c1906
```

Canonical VS3 verify/gate before this checkpoint file was added:

```text
./cornerstone scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
exit 0
status success
final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
summary pass 50 human_required 7 blocking 0
source_hash 2657033006b57d3a1358201445e61a0472c6619b0135d948cef0d2d69a4c1906

./cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json
exit 0
status success
traceability passed ['reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json']
generated_dirty_snapshot passed recorded 135 current 135 hash_mismatch False
```

## Proof Boundary

This checkpoint proves only local/dev VS3 scenario traceability path binding. It does not prove VS3-P, production/on-prem readiness, real IdP readiness, real network readiness, live-provider readiness, migration/restore readiness, independent security acceptance, or human UX acceptance.
