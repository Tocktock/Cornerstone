# VS3 Local Checkpoint Self Command Transcript Stdout Elapsed Duration Guard Checkpoint

**Date:** 2026-06-30 KST
**Status:** Local deterministic checkpoint for one VS3 verification-surface hardening slice.
**Verdict:** PASS for this slice only. VS3-P, production/on-prem readiness, live-provider readiness, real-IdP readiness, real-network readiness, migration/restore readiness, security acceptance, and human acceptance remain NOT_CLAIMED or HUMAN_REQUIRED.

## Source Inputs

- `docs/agent/PRIMARY_GOAL_INSTRUCTION_FOLLOWING.md`
- `docs/sot/README.md`
- `docs/sot/01_PRODUCT_GOAL_AND_DIRECTION.md`
- `docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md`
- `docs/sot/03_TECHNICAL_ARCHITECTURE_DEFAULTS.md`
- `docs/scenario-contracts/SCENARIO_MATRIX_FULL.md`
- `docs/scenario-contracts/CLI_NATIVE_FIRST_CONTRACT.md`
- `docs/scenario-contracts/LOCAL_VERIFICATION_PLANE_V0.md`
- `docs/scenario-contracts/VS3_ONPREM_SECURITY_AND_TRUSTED_EXTENSION_CONTRACT.md`
- `docs/scenario-contracts/VS3_ONPREM_SECURITY_AND_TRUSTED_EXTENSION_MATRIX.csv`
- `docs/agent/SCENARIO_FIRST_AGENT_INSTRUCTION.md`
- `docs/agent/PROJECT_OPERATING_CONSTITUTION.md`

## Slice Contract

Goal:
- Harden the VS3 command transcript evidence guard so self-command transcripts cannot carry a top-level `elapsed_seconds` value that is absent from or contradicted by the embedded `stdout_json` evidence.

Scope:
- Shared VS3 transcript validator.
- VS3 human-gate self transcript builder.
- VS3 local checkpoint self transcript builder.
- VS3 scenario verify self transcript builder.
- VS3 scenario gate self transcript builder.
- Focused regression coverage for scenario-gate source transcript tampering.

Non-scope:
- No new VS3 product capability.
- No broad report model redesign.
- No production, live-provider, real-IdP, real-network, migration/restore, security-acceptance, or human-acceptance claim.
- No human gate promotion.

Done criteria:
- `stdout_json.elapsed_seconds` is required, finite, non-negative, and equal to top-level transcript `elapsed_seconds`.
- All VS3 self transcript builders emit the mirrored elapsed value.
- A tampered VS3 source scenario report with mismatched stdout elapsed evidence is rejected by `cornerstone scenario gate`.
- Refreshed VS3 scenario, local checkpoint, and human-gate artifacts preserve proof boundaries.

## Full Scenario Mapping

The full VS3 matrix remains 57 rows: 42 `MUST_PASS`, 8 `REGRESSION`, and 7 `HUMAN_REQUIRED`.

| Scenario IDs | Type | Classification for this slice | Required proof surface | Reason |
|---|---|---|---|---|
| VS3-GATE-004 | MUST_PASS | in_this_slice | Native CLI transcript validator, scenario report, scenario gate | This slice hardens required `cornerstone scenario verify ... --json` transcript metadata. |
| VS3-REG-004 | REGRESSION | in_this_slice | Coverage/audit omission gate and transcript tamper test | The stricter transcript shape prevents incomplete verification evidence from passing silently. |
| VS3-REG-005 | REGRESSION | in_this_slice | Overclaim-safe local reports and gate output | The refreshed reports keep local/dev proof separate from VS3-P and production/on-prem claims. |
| VS3-REG-008 | REGRESSION | in_this_slice | Local checkpoint and conservative default report fields | The guard preserves fail-closed verification metadata for the local checkpoint path. |
| VS3-GATE-001, VS3-GATE-002, VS3-GATE-003 | MUST_PASS | later_slice | Reconciliation report, docs verifier, overclaim lint | Already covered by existing VS3 gates; this slice did not alter their product semantics. |
| VS3-CTX-001, VS3-CTX-002, VS3-CTX-003, VS3-CTX-004, VS3-CTX-005 | MUST_PASS | later_slice | RequestContext proof reports and CLI/API/UI/worker/tool fixtures | This slice only validates the transcript evidence envelope around existing proof reports. |
| VS3-RLS-001, VS3-RLS-002, VS3-RLS-003, VS3-RLS-004, VS3-RLS-005, VS3-RLS-006 | MUST_PASS | later_slice | Postgres/RLS inventory, migration, backup, restore, and isolation proof | No RLS behavior was changed. |
| VS3-OPA-001, VS3-OPA-002, VS3-OPA-003, VS3-OPA-004, VS3-OPA-005 | MUST_PASS | later_slice | OPA/Rego local proof and policy CLI transcripts | No policy behavior was changed. |
| VS3-EGR-001, VS3-EGR-002, VS3-EGR-003, VS3-EGR-004, VS3-EGR-005, VS3-EGR-006 | MUST_PASS | later_slice | Egress/sandbox controlled sink and negative evidence proof | No egress behavior was changed. |
| VS3-CON-001, VS3-CON-002, VS3-CON-003, VS3-CON-004, VS3-CON-005, VS3-CON-006 | MUST_PASS | later_slice | ConnectorHub/source proof reports and fixture transcripts | No ConnectorHub behavior was changed. |
| VS3-TOOL-001, VS3-TOOL-002, VS3-TOOL-003, VS3-TOOL-004, VS3-TOOL-005, VS3-TOOL-006, VS3-TOOL-007 | MUST_PASS | later_slice | Tool registry package, signature, SBOM, activation, sandbox, update, rollback proof | No tool registry behavior was changed. |
| VS3-OBS-001, VS3-OBS-002, VS3-OBS-003 | MUST_PASS | later_slice | Operator status, audit, and human-gate package reports | This slice only hardens self transcript metadata consumed by the checkpoint. |
| VS3-REG-001, VS3-REG-002, VS3-REG-003, VS3-REG-006, VS3-REG-007 | REGRESSION | later_slice | Final regression proof reports and UI/supply-chain evidence | Not touched by this slice except through aggregate report refresh. |
| VS3-H01, VS3-H02, VS3-H03, VS3-H04, VS3-H05, VS3-H06, VS3-H07 | HUMAN_REQUIRED | HUMAN_REQUIRED | Dated human/on-prem evidence | Local templates and reports remain preparation only; these rows still block VS3-P. |

## Implementation Notes

- `packages/cornerstone_cli/main.py` now rejects VS3 command transcripts whose `stdout_json.elapsed_seconds` is missing, non-numeric, non-finite, negative, or different from the top-level transcript elapsed duration.
- `packages/cornerstone_cli/main.py` now mirrors rounded elapsed duration into stdout JSON for VS3 human-gate, local-checkpoint, scenario-verify, and scenario-gate self transcripts.
- `tests/scenario/test_scaffold_cli.py` now covers a tampered VS3 scenario report whose source `command_transcripts.scenario_verify_vs3` and `self_command_transcript` have mismatched `stdout_json.elapsed_seconds`.

## Verification Evidence

Commands run:

```text
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
```

Result: exit 0.

```text
python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_elapsed_seconds_invalid tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_source_transcript_stdout_elapsed_mismatch
```

Result: `Ran 2 tests in 27.797s OK`.

```text
python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_manifest_is_hash_backed_and_boundary_safe tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_preserves_source_proof_boundary_and_transcript tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_source_transcript_stdout_elapsed_mismatch tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_elapsed_seconds_invalid tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_human_gate_reports_missing_self_transcript
```

Result: `Ran 5 tests in 81.089s OK`.

```text
PATH="$PWD:$PATH" cornerstone scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
```

Result: exit 0.

Evidence:
- `reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json`
- sha256 `4893fbd79304a74e6a216ac8d19036bf7b737a4bdbc32842679261662713d5b2`
- status `success`
- final verdict `VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED`
- summary: `scenario_count=57`, `pass=50`, `human_required=7`, `blocking=0`, `fail=0`, `not_run=0`, `not_verified=0`
- claim boundaries: `vs3_l=LOCAL_DEV_ASSURANCE_VERIFIED`; `vs3_p`, `production_onprem`, `live_provider`, `real_idp`, `real_network`, `migration_restore`, `security_acceptance`, and `human_acceptance` are `NOT_CLAIMED`
- transcript elapsed mirror: top-level `0.264`, stdout JSON `0.264`

Human-gate derived artifacts:
- `reports/human-gates/vs3/record-scaffold.json`, exit 0, sha256 `4a32d0d0ec0365395b950500da01e03d739ea9aceed2840449bd434fcd258d71`, final verdict `HUMAN_REQUIRED`, template count 7
- `reports/human-gates/vs3/evidence-status.json`, exit 0, sha256 `21c8dc0f8ee4b18215e6e4aeb377128965f50280e904b2e1858bc09de0b3d6b3`, final verdict `HUMAN_REQUIRED`, structurally valid count 0, structurally invalid count 7, self transcript validation `passed`
- `reports/human-gates/vs3/review-kit.json`, exit 0, sha256 `493215b29269ad1818b66e94e76024c6b81a1e78613b2f17e3647e0867383f33`, final verdict `HUMAN_REQUIRED`, review queue count 7, self transcript validation `passed`
- `reports/human-gates/vs3/vs3-p-gate.json`, expected exit 4, sha256 `e3c9cca8f30119b7ebd4d8d1cf4d08205690f474422f28495b20f350297eba73`, status `blocked`, final verdict `HUMAN_REQUIRED`, unresolved human rows 7, `vs3_p_ready=false`, self transcript validation `passed`

```text
PATH="$PWD:$PATH" cornerstone security vs3-local-checkpoint --json --output reports/security/vs3-local-checkpoint.json
```

Result: exit 0.

Evidence:
- `reports/security/vs3-local-checkpoint.json`
- sha256 `13e64fee185f88e60b12034924aa150ae0593b0c721860a9088d155c26f7fd87`
- status `success`
- final verdict `VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED`
- summary: `scenario_count=57`, `pass=50`, `human_required=7`, `blocking=0`, `component_proof_report_cli_command_evidence_shape_failures=0`, `self_command_transcript_shape_failures=0`, `unresolved_human_required_rows=7`
- transcript elapsed mirror: top-level `0.044`, stdout JSON `0.044`, self transcript validation `passed`

```text
PATH="$PWD:$PATH" cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json
```

Result: exit 0.

Evidence:
- stdout sha256 `d13d047ef479bfe7aa40c3fe38f3e50e670e07e496d20202de4bf06191a64bd5`
- status `success`
- final verdict `VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED`
- scenario gate summary: `self_command_transcript_shape_failures=0`
- transcript elapsed mirror: top-level `0.018`, stdout JSON `0.018`, self transcript validation `passed`

Additional transcript scan:
- `reports/security/vs3-request-context-proof.json`: `bad_elapsed_mirrors=[]`
- `reports/db/vs3-postgres-rls-proof.json`: `bad_elapsed_mirrors=[]`
- `reports/policy/vs3-opa-policy-proof.json`: `bad_elapsed_mirrors=[]`
- `reports/security/vs3-egress-sandbox-proof.json`: `bad_elapsed_mirrors=[]`
- `reports/security/vs3-connectorhub-source-proof.json`: `bad_elapsed_mirrors=[]`
- `reports/security/vs3-tool-registry-proof.json`: `bad_elapsed_mirrors=[]`
- `reports/observability/vs3-observability-proof.json`: `bad_elapsed_mirrors=[]`
- `reports/security/vs3-final-regression-proof.json`: `bad_elapsed_mirrors=[]`

## Human Required Gates

The following remain unresolved and cannot be converted into AI PASS:

- `VS3-H01`: architecture/security/dependency/migration approval
- `VS3-H02`: independent security review and retest
- `VS3-H03`: real IdP mapping and revocation evidence
- `VS3-H04`: real on-prem network evidence
- `VS3-H05`: approved live provider rehearsal
- `VS3-H06`: human operator UX/trust review
- `VS3-H07`: supervised migration/backup/restore drill

## Decision

This slice is locally verified. It improves the VS3-L evidence envelope by rejecting stale or inconsistent elapsed-duration metadata in command transcript stdout JSON. It does not complete VS3 as a whole and does not unlock VS3-P or any production/on-prem claim.
