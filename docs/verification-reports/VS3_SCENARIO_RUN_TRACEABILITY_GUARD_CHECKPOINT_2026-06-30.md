# VS3 Scenario Run Traceability Guard Checkpoint

**Date:** 2026-06-30 KST
**Status:** Local deterministic checkpoint complete.
**Scope:** VS3 scenario-run traceability metadata and scenario-gate validation.
**Claim boundary:** VS3-L local/dev evidence traceability improved. VS3-P, production/on-prem readiness, live-provider readiness, real IdP readiness, real-network readiness, migration/restore readiness, security acceptance, and human acceptance remain unclaimed.

## Slice Contract

Goal:

- Add deterministic scenario-run traceability metadata to the VS3 local/dev scenario report.
- Add row-level traceability fields for every frozen VS3 scenario row.
- Make `cornerstone scenario gate` fail closed when a local-dev VS3 assurance report is missing traceability metadata.

In scope:

- `cornerstone scenario verify vs3-onprem-trusted-extension --json` report metadata.
- Per-row `scenario_run_id`, `trace_id`, local corpus/model/scope, and transcript path metadata.
- `cornerstone scenario gate <report> --json` traceability validation.
- Focused unittest coverage and regenerated local report artifacts.

Out of scope:

- New RequestContext, RLS, OPA, egress, ConnectorHub, Tool SDK, registry, Agent Pack, or operator UI behavior.
- Production/on-prem, live-provider, real IdP, real-network, migration/restore, independent security review, or human UX acceptance proof.
- Converting any `HUMAN_REQUIRED` row to `PASS`.

## Full Scenario Mapping

The authoritative full row details remain in `docs/scenario-contracts/VS3_ONPREM_SECURITY_AND_TRUSTED_EXTENSION_MATRIX.csv`.

Matrix check:

- Total rows: 57
- `MUST_PASS`: 42
- `REGRESSION`: 8
- `HUMAN_REQUIRED`: 7
- Duplicate IDs: 0

Current slice classification:

| Classification | Scenario IDs | Reason |
|---|---|---|
| `in_this_slice` | `VS3-GATE-004`, `VS3-REG-004` | This slice hardens native verifier/gate traceability and makes missing run metadata fail before local-dev assurance can pass. |
| `HUMAN_REQUIRED` | `VS3-H01`, `VS3-H02`, `VS3-H03`, `VS3-H04`, `VS3-H05`, `VS3-H06`, `VS3-H07` | Requires signed human/external evidence and cannot be converted to AI PASS by local proof. |
| `later_slice_or_existing_component_proof` | `VS3-GATE-001`, `VS3-GATE-002`, `VS3-GATE-003`, `VS3-CTX-001` through `VS3-CTX-005`, `VS3-RLS-001` through `VS3-RLS-006`, `VS3-OPA-001` through `VS3-OPA-005`, `VS3-EGR-001` through `VS3-EGR-006`, `VS3-CON-001` through `VS3-CON-006`, `VS3-TOOL-001` through `VS3-TOOL-007`, `VS3-OBS-001` through `VS3-OBS-003`, `VS3-REG-001` through `VS3-REG-003`, `VS3-REG-005` through `VS3-REG-008` | Outside this narrow traceability slice; their behavior and proof expectations remain exactly as defined in the matrix and existing component proofs. |

Selected scenario criteria:

| Scenario | Expected behavior | Required evidence | Pass/fail criteria |
|---|---|---|---|
| `VS3-GATE-004` | Native VS3 verifier emits traceable JSON report metadata for the scenario run and every row. | Scenario report, command transcript, traceability object, row trace fields. | PASS only if report and rows include non-empty `scenario_run_id`, `trace_id`, corpus/model/scope, and transcript path metadata; FAIL if local-dev assurance can pass without them. |
| `VS3-REG-004` | Scenario coverage and audit/trace coverage cannot silently disappear. | Tampered-report negative test and scenario-gate failure output. | PASS only if missing traceability is detected before local-dev assurance can pass; FAIL on silent omission. |

## Implementation Decision

Before this slice, the VS3 scenario report had CLI command transcript metadata, source-tree metadata, row-level evidence refs, and row-level audit refs, but it did not expose the Local Verification Plane traceability fields `scenario_run_id`, `trace_id`, corpus/model identity, local scope, or transcript paths at the report and row levels.

The guard now:

- Emits deterministic `scenario_run_id` and `trace_id` derived from the verified source-worktree hash.
- Emits `corpus_pack_id=fixtures/vs3/local-dev`, `model_provider=local_test`, and `model_name=deterministic-local-test`.
- Emits the validated local scenario-verifier scope: `local-dev` / `local-user` / `personal` / `default` / `local_vs3_fixture`.
- Emits row-level traceability metadata for all 57 frozen VS3 rows.
- Adds `traceability_validation` to `cornerstone scenario gate`.
- Fails the gate with `CS_VS3_TRACEABILITY_METADATA_MISSING` when a local-dev assurance report omits traceability metadata.

## Verification Evidence

Syntax:

- `python3 -m py_compile packages/cornerstone_cli/main.py packages/cornerstone_cli/scenarios.py tests/scenario/test_scaffold_cli.py`
- Result: exit 0.

Focused tests:

- Command: `python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_onprem_trusted_extension_verify_closes_local_dev_ai_rows tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_preserves_source_proof_boundary_and_transcript tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_without_traceability_metadata`
- Result: `Ran 3 tests in 26.588s`, `OK`.

Native CLI refresh:

- `./cornerstone scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json`: exit 0, `status=success`, `final_verdict=VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED`.
- Scenario traceability: `scenario-run:vs3-onprem-trusted-extension:8274fec33f99db4d`, `trace:vs3-onprem-trusted-extension:8274fec33f99db4d`, `transcript_paths=["reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json"]`.
- `./cornerstone human-gate record-scaffold --scope vs3 ...`: exit 0, `final_verdict=HUMAN_REQUIRED`.
- `./cornerstone human-gate evidence-status --scope vs3 ...`: exit 0, `final_verdict=HUMAN_REQUIRED`.
- `./cornerstone human-gate review-kit --scope vs3 ...`: exit 0, `final_verdict=HUMAN_REQUIRED`.
- `./cornerstone human-gate vs3-p-gate --scope vs3 ...`: expected exit 4, `status=blocked`, `final_verdict=HUMAN_REQUIRED`.
- `./cornerstone security vs3-local-checkpoint --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --output reports/security/vs3-local-checkpoint.json --json`: exit 0, `status=success`, `final_verdict=VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED`.
- `./cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json`: exit 0, `status=success`, `final_verdict=VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED`.
- Scenario gate `traceability_validation`: `status=passed`, `missing_fields=[]`, `invalid_fields=[]`, `row_missing_fields=[]`, `row_invalid_fields=[]`.

Artifact hashes after refresh:

| Artifact | SHA-256 | Status / verdict |
|---|---|---|
| `reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json` | `60365bb5663614e159e1b372221d54b6c0eff36b944b60b3a305559dc6bb3598` | `success`, `VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED` |
| `reports/human-gates/vs3/evidence-status.json` | `5eee929c63fce1fb9ccc0c2be8942e3471d34818a52e256826dfbec808a5ea4d` | `success`, `HUMAN_REQUIRED` |
| `reports/human-gates/vs3/review-kit.json` | `91290e33ad85835d6893618531ca4376feceb7e02caaf2fd52738f18494e2b92` | `success`, `HUMAN_REQUIRED` |
| `reports/human-gates/vs3/vs3-p-gate.json` | `822c4c19ea743872925de6fbfdebc10cea2439d3b92b5e6b16ab0998048609b9` | `blocked`, `HUMAN_REQUIRED` |
| `reports/security/vs3-local-checkpoint.json` | `150a31e963df5ba8283db4775445bc440cff2d62582a83b32fd4588d9f2b64d7` | `success`, `VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED` |

## Remaining Human Gates

Still `HUMAN_REQUIRED`:

- `VS3-H01`: owner architecture/security approval.
- `VS3-H02`: independent security review and retest.
- `VS3-H03`: real IdP mapping and revocation evidence.
- `VS3-H04`: real on-prem network/security evidence.
- `VS3-H05`: approved live ConnectorHub/provider rehearsal.
- `VS3-H06`: human operator UX/trust review.
- `VS3-H07`: human-supervised migration/backup/restore/rollback drill.

This checkpoint does not satisfy any VS3-P gate.
