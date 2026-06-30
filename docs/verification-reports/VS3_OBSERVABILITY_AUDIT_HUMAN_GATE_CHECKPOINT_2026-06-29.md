# VS3 Observability, Audit, and Human-Gate Checkpoint - 2026-06-29

**Status:** VS3-7 local observability / audit / human-gate package slice PASS.
**Scope:** `VS3-OBS-001` through `VS3-OBS-003` only.
**Proof boundary:** Local deterministic observability fixture, local audit hash-chain state, controlled audit tamper fixture, generated VS3 human-gate packages, native CLI output, and a deterministic local operator-status DOM snapshot.

This checkpoint does not claim VS3-P, production/on-prem readiness, real topology readiness, live-provider readiness, migration/restore readiness, independent security review, human UX acceptance, or human security acceptance.

## Slice Contract

Goal:

- Verify the local VS3 operator status surface through native `cornerstone ... --json` commands.
- Prove the status truth is consistent across local CLI/API/UI projections, including a DOM snapshot for the UI projection.
- Verify the audit ledger is append-only and tamper-evident for the VS3 event inventory.
- Generate seven human-gate packages as review inputs only, without marking any `VS3-H` row `PASS`.

Selected scenarios:

| Scenario | Status in this checkpoint | Required proof surface |
|---|---|---|
| `VS3-OBS-001` | PASS | `observe status` JSON, CLI/API/UI status comparison, UI/DOM snapshot, component fault results, audit refs. |
| `VS3-OBS-002` | PASS | Required audit event inventory, clean hash-chain verification, controlled tamper failure, scoped event records. |
| `VS3-OBS-003` | PASS | Seven human-gate package files with scope, why AI cannot verify, required human action, expected evidence, redaction rules, release impact, and blank approval/rejection record. |

Full VS3 mapping remains the frozen 57-row inventory: 42 `MUST_PASS`, 8 `REGRESSION`, and 7 `HUMAN_REQUIRED`. Human/on-prem rows remain `HUMAN_REQUIRED`.

## Implementation Delta

- `packages/cornerstone_cli/scenarios.py` now writes `reports/observability/vs3-operator-status.dom.html` during the VS3 observability proof.
- `reports/observability/vs3-observability-proof.json` now includes `status_surface_comparison` with local `cli`, `api`, and `ui` projections tied to one status truth digest.
- `VS3-OBS-001` can no longer pass if the CLI/API/UI comparison, DOM snapshot, DOM component inventory, or surface audit refs are missing.
- `cornerstone observe status --scope vs3 --json` now exposes the status-surface comparison and DOM snapshot path.
- `tests/scenario/test_scaffold_cli.py` now asserts the DOM snapshot, comparison checks, OBS negative counters, CLI payload fields, and human-gate package required fields.

## Command Evidence

Focused compile:

```text
python3 -m compileall packages/cornerstone_cli/scenarios.py packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
Compiling 'packages/cornerstone_cli/scenarios.py'...
exit=0
```

Focused OBS and aggregate VS3 tests:

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_observability_proof_is_local_and_human_gate_backed \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_observability_cli_paths_are_native \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_onprem_trusted_extension_verify_closes_local_dev_ai_rows

Ran 3 tests in 30.220s
OK
```

Direct native CLI probes:

```text
cornerstone security vs3-observability --json
exit=0
schema_version=cs.vs3_observability_proof.v0
status=success

cornerstone observe status --scope vs3 --json
exit=0
schema_version=cs.cli.v0
observe_status_schema_version=cs.vs3_observe_status.v0
component_count=10
ui_dom_snapshot_path=reports/observability/vs3-operator-status.dom.html
comparison_checks:
  audit_refs_visible_all_surfaces=true
  cli_api_ui_component_keys_match=true
  cli_api_ui_truth_digest_match=true
  fault_rows_visible_all_surfaces=true
  human_gates_human_required_all_surfaces=true
  ui_dom_snapshot_contains_component_statuses=true
  ui_dom_snapshot_exists=true

cornerstone human-gate package --scope vs3 --json
exit=0
schema_version=cs.cli.v0
human_gate_package_schema_version=cs.vs3_human_gate_package_set.v0
package_count=7
final_verdict=HUMAN_REQUIRED

cornerstone human-gate report --scope vs3 --use-existing --json
exit=0
schema_version=cs.cli.v0
human_gate_readiness_report_schema_version=cs.vs3_human_gate_readiness_report.v0
next_scenario_id=VS3-H01
final_verdict=HUMAN_REQUIRED

cornerstone human-gate next --scope vs3 --use-existing --json
exit=0
schema_version=cs.cli.v0
human_gate_readiness_report_schema_version=cs.vs3_human_gate_readiness_report.v0
next_scenario_id=VS3-H01
final_verdict=HUMAN_REQUIRED
```

Audit integrity probes:

```text
cornerstone audit verify --state-dir reports/runtime/vs3-observability-state --json
exit=0
schema_version=cs.cli.v0
status=success
audit_integrity.status=success
audit_integrity.event_count=14

cornerstone audit verify --state-dir reports/runtime/vs3-observability-tamper-state --json
exit=5
schema_version=cs.cli.v0
status=failed
audit_integrity.status=failed
audit_integrity.event_count=14
audit_error_codes=AUDIT_EVENT_HASH_MISMATCH
```

Aggregate scenario verification:

```text
cornerstone scenario verify vs3-onprem-trusted-extension --json \
  --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
exit=0
schema_version=cs.vs3_onprem_trusted_extension.v0
status=success
summary: PASS=50, HUMAN_REQUIRED=7, blocking=0, fail=0, scenario_count=57
VS3-OBS-001=PASS
VS3-OBS-002=PASS
VS3-OBS-003=PASS
claim_boundaries.vs3_l=LOCAL_DEV_ASSURANCE_VERIFIED
claim_boundaries.vs3_p=NOT_CLAIMED
claim_boundaries.production_onprem=NOT_CLAIMED
claim_boundaries.human_acceptance=NOT_CLAIMED
```

Aggregate scenario gate:

```text
cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json
exit=0
schema_version=cs.cli.v0
status=success
failed_conditions=null
```

## Scenario Evidence Mapping

| Scenario | Evidence |
|---|---|
| `VS3-OBS-001` | `reports/observability/vs3-observability-proof.json`; `reports/observability/vs3-operator-status.dom.html`; `cornerstone observe status --scope vs3 --json`; `status_surface_comparison`; `operator_status_snapshot`. |
| `VS3-OBS-002` | `reports/observability/vs3-observability-proof.json`; `reports/runtime/vs3-observability-state`; `reports/runtime/vs3-observability-tamper-state`; clean audit verify exit 0; tamper audit verify exit 5 with `AUDIT_EVENT_HASH_MISMATCH`. |
| `VS3-OBS-003` | `reports/human-gates/vs3`; `cornerstone human-gate package --scope vs3 --json`; `cornerstone human-gate report --scope vs3 --use-existing --json`; `cornerstone human-gate next --scope vs3 --use-existing --json`. |

## Negative Evidence

```text
status_cli_api_ui_mismatches=0
status_ui_dom_snapshot_missing=0
status_ui_dom_snapshot_missing_components=0
status_surfaces_without_audit_refs=0
operator_components_missing=0
misleading_green_fault_statuses=0
missing_required_audit_event_families=0
audit_tamper_accepted=0
human_gate_packages_missing=0
human_gate_packages_marked_pass=0
human_gate_approvals_collected_by_package_generator=0
human_gate_pass_claims_allowed_by_package_generator=0
human_gate_product_claims_allowed_by_package_generator=0
raw_secret_values_in_human_gate_packages=0
vs3_l_claimed=0
vs3_p_claimed=0
production_onprem_claimed=0
human_acceptance_claimed=0
```

## Human Required

The following remain outside this checkpoint and cannot be marked `PASS` by this local proof:

- `VS3-H01`: owner architecture/security approval.
- `VS3-H02`: real IdP / directory integration evidence.
- `VS3-H03`: on-prem network/proxy/DNS/firewall evidence.
- `VS3-H04`: customer-controlled key, backup, retention, and restore approval.
- `VS3-H05`: migration rehearsal approval for real customer data shape.
- `VS3-H06`: human operator UX/trust acceptance.
- `VS3-H07`: independent security review / penetration-test evidence.

## Deliberately Not Claimed

- No VS3-P release candidate claim.
- No production/on-prem deployment claim.
- No real topology, real IdP, real network, live provider, or real migration/restore claim.
- No human acceptance or independent security acceptance claim.
