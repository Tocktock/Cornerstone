# VS3 Scenario Gate Stdout JSON Schema Guard Checkpoint

**Date:** 2026-06-30 KST
**Status:** Local deterministic checkpoint complete.
**Scope:** VS3 scenario verifier/gate transcript shape hardening only.
**Claim boundary:** VS3-L local/dev evidence guard improved. VS3-P, production/on-prem readiness, live-provider readiness, real IdP readiness, real-network readiness, migration/restore readiness, security acceptance, and human acceptance remain unclaimed.

## Slice Contract

Goal:

- Make VS3 command transcript `stdout_json` prove both the CLI envelope schema and the payload schema it claims to represent.
- Reject VS3 local/dev scenario reports when a source scenario-verify transcript has a mismatched `stdout_json.json_schema`.

In scope:

- Shared VS3 command transcript validator.
- VS3 scenario verify self-command transcript.
- VS3 scenario gate self-command transcript.
- VS3 local checkpoint self-command transcript.
- VS3 human-gate derived report self-command transcripts.
- Focused unittest coverage and regenerated local report artifacts.

Out of scope:

- New RequestContext, RLS, OPA, egress, ConnectorHub, Tool SDK, registry, or Agent Pack behavior.
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
| `in_this_slice` | `VS3-GATE-004`, `VS3-REG-005` | Native VS3 scenario verify/gate JSON transcript evidence must be replayable, schema-bound, and no stronger than local/dev evidence. |
| `HUMAN_REQUIRED` | `VS3-H01`, `VS3-H02`, `VS3-H03`, `VS3-H04`, `VS3-H05`, `VS3-H06`, `VS3-H07` | Requires signed human/external evidence and cannot be converted to AI PASS by local proof. |
| `later_slice` | `VS3-GATE-001`, `VS3-GATE-002`, `VS3-GATE-003`, `VS3-CTX-001` through `VS3-CTX-005`, `VS3-RLS-001` through `VS3-RLS-006`, `VS3-OPA-001` through `VS3-OPA-005`, `VS3-EGR-001` through `VS3-EGR-006`, `VS3-CON-001` through `VS3-CON-006`, `VS3-TOOL-001` through `VS3-TOOL-007`, `VS3-OBS-001` through `VS3-OBS-003`, `VS3-REG-001` through `VS3-REG-004`, `VS3-REG-006` through `VS3-REG-008` | Outside this narrow verifier/evidence-layout slice; their proof expectations remain exactly as defined in the matrix. |

Selected scenario criteria:

| Scenario | Expected behavior | Required evidence | Pass/fail criteria |
|---|---|---|---|
| `VS3-GATE-004` | `cornerstone scenario verify vs3-onprem-trusted-extension --json` emits status, counts, per-row evidence, human rows, and gate metadata. | CLI transcript, JSON schema validation, coverage matrix. | PASS only if CLI is native and status-neutral before implementation; FAIL if raw scripts replace CLI parity. |
| `VS3-REG-005` | VS3 wording does not turn local/dev proof into production, real IdP, live provider, penetration-tested, human-accepted, or migration-ready claims. | Static overclaim lint and evidence manifest review. | PASS only if wording is no stronger than evidence; FAIL on unqualified readiness claim. |

## Implementation Decision

Before this slice, VS3 self-command transcripts stored the payload schema in the transcript entry, but their captured `stdout_json` often only showed `schema_version=cs.cli.v0`.

The guard now allows `stdout_json.schema_version` to remain the CLI envelope schema while requiring the payload schema to be visible and matching through `stdout_json.json_schema`.

Validator behavior:

- If `stdout_json.schema_version` equals the transcript payload `json_schema`, it may stand alone.
- If `stdout_json.schema_version` is the CLI envelope schema, `stdout_json.json_schema` must equal the transcript payload `json_schema`.
- If `stdout_json.json_schema` is present but mismatched, validation fails with `stdout_json_json_schema_mismatch`.
- If `stdout_json.schema_version` is neither the CLI schema nor the payload schema, validation fails with `stdout_json_schema_version_unexpected`.

## Verification Evidence

Syntax:

- `python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py`
- Result: exit 0.

Focused scenario-gate tests:

- Command: `python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_preserves_source_proof_boundary_and_transcript tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_source_transcript_missing_stdout_json tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_source_transcript_stdout_elapsed_mismatch tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_source_transcript_stdout_json_schema_mismatch`
- Result: `Ran 4 tests in 1.855s`, `OK`.

Focused local-checkpoint tests:

- Command: `python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_manifest_is_hash_backed_and_boundary_safe tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_self_transcript_validator_rejects_tampered_transcript tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_evidence_missing_stdout_json tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_component_proof_cli_command_stdout_json_execution_mismatch`
- Result: `Ran 4 tests in 105.871s`, `OK`.

Focused post-refresh tests:

- Command: `python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_preserves_source_proof_boundary_and_transcript tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_source_transcript_stdout_json_schema_mismatch tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_manifest_is_hash_backed_and_boundary_safe`
- Result: `Ran 3 tests in 27.451s`, `OK`.

Native CLI refresh:

- `cornerstone security vs3-evidence-reconcile --json`: exit 0, `status=success`.
- `cornerstone security vs3-overclaim-lint --json`: exit 0, `status=passed`.
- `cornerstone security vs3-request-context --json`: exit 0, `status=success`.
- `cornerstone security vs3-postgres-rls --reuse-vs2-local-range-report reports/security/vs2-local-range.json --json`: exit 0, `status=success`.
- `cornerstone security vs3-opa-policy --reuse-vs2-local-range-report reports/security/vs2-local-range.json --json`: exit 0, `status=success`.
- `cornerstone security vs3-egress-sandbox --reuse-vs2-local-range-report reports/security/vs2-local-range.json --json`: exit 0, `status=success`.
- `cornerstone security vs3-connectorhub-source --json`: exit 0, `status=success`.
- `cornerstone security vs3-tool-registry --json`: exit 0, `status=success`.
- `cornerstone security vs3-observability --json`: exit 0, `status=success`.
- `cornerstone security vs3-regression-gate --json`: exit 0, `status=success`.
- `cornerstone scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json`: exit 0, `status=success`, `scenario_count=57`, `pass=50`, `human_required=7`, `blocking=0`, `final_verdict=VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED`.
- `cornerstone human-gate record-scaffold --scope vs3 --output-dir reports/human-gates/vs3/record-templates --force --use-existing --json --output reports/human-gates/vs3/record-scaffold.json`: exit 0, `final_verdict=HUMAN_REQUIRED`.
- `cornerstone human-gate evidence-status --scope vs3 --record-dir reports/human-gates/vs3/record-templates --use-existing --json --output reports/human-gates/vs3/evidence-status.json`: exit 0, `final_verdict=HUMAN_REQUIRED`, self transcript validation passed.
- `cornerstone human-gate review-kit --scope vs3 --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --use-existing --json --output reports/human-gates/vs3/review-kit.json`: exit 0, `final_verdict=HUMAN_REQUIRED`, self transcript validation passed.
- `cornerstone human-gate vs3-p-gate --scope vs3 --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --output reports/human-gates/vs3/vs3-p-gate.json --json`: expected exit 4, `status=blocked`, `final_verdict=HUMAN_REQUIRED`.
- `cornerstone security vs3-local-checkpoint --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --output reports/security/vs3-local-checkpoint.json --json`: exit 0, `status=success`, self transcript validation passed.
- `cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json`: exit 0, `status=success`, source transcript validation passed, self transcript validation passed.

Stdout schema mirror scan:

- Checked VS3 scenario report, local checkpoint, evidence-status, review-kit, VS3-P gate, and the scenario gate stdout artifact.
- Result: every self/command transcript satisfied `stdout_json.schema_version == transcript.json_schema` or `stdout_json.json_schema == transcript.json_schema`.
- Issues: `[]`.

Artifact hashes after refresh:

| Artifact | SHA-256 | Status | Final verdict |
|---|---|---|---|
| `reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json` | `b93bd8efe59714c43300a227ce0e11b667ec32a8d16f7f28103e75808c92b02f` | `success` | `VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED` |
| `reports/security/vs3-local-checkpoint.json` | `b1ce545e80564ae209b477b5aa3d189a925b9f2c55bc61a6e68425089900dd23` | `success` | `VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED` |
| `reports/human-gates/vs3/record-scaffold.json` | `e4a49f882d658f55f68bb5ab1dce165eca535ad971afa4dd18f2b3afbd576199` | `success` | `HUMAN_REQUIRED` |
| `reports/human-gates/vs3/evidence-status.json` | `7c07264f50c485371cc92691b150d34b371c6335aa8b0206ca55b8598eb1bca3` | `success` | `HUMAN_REQUIRED` |
| `reports/human-gates/vs3/review-kit.json` | `08d30e584fc9308ad7f7d83921d28e3bab64eb87a344cf7888480ceda5e82154` | `success` | `HUMAN_REQUIRED` |
| `reports/human-gates/vs3/vs3-p-gate.json` | `bc04e02473765c8402a9b12ecff0fd4dd3ad22d8e227dd34290573f76f0c8b92` | `blocked` | `HUMAN_REQUIRED` |
| `/tmp/vs3-scenario-gate.json` | `465f657ff641ccbc52f010550719fc588dbde1ac484e5dfdb475197759933b6a` | `success` | `VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED` |

Whitespace:

- `git diff --check`
- Result: exit 0.

## Remaining Human Gates

Still `HUMAN_REQUIRED`:

- `VS3-H01`: owner architecture/security approval.
- `VS3-H02`: independent security review and retest.
- `VS3-H03`: real IdP mapping and revocation evidence.
- `VS3-H04`: real on-prem network control evidence.
- `VS3-H05`: approved live-provider rehearsal.
- `VS3-H06`: human operator UX/trust review.
- `VS3-H07`: human-supervised migration/backup/restore drill.

This checkpoint does not satisfy any VS3-P gate.
