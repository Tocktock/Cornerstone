# VS3 Scenario Gate Stdout Tail Consistency Guard Checkpoint

**Date:** 2026-06-30 KST
**Status:** Local deterministic checkpoint complete.
**Scope:** VS3 scenario verifier/gate transcript `stdout_tail` consistency hardening only.
**Claim boundary:** VS3-L local/dev evidence guard improved. VS3-P, production/on-prem readiness, live-provider readiness, real IdP readiness, real-network readiness, migration/restore readiness, security acceptance, and human acceptance remain unclaimed.

## Slice Contract

Goal:

- Make VS3 self/source command transcripts fail when the captured JSON `stdout_tail` contradicts the structured `stdout_json`.
- Preserve ordinary text stdout tails in component proof transcripts as evidence tails, not JSON-tail mirrors.
- Keep all proof boundaries local/dev unless human/external VS3-P evidence is present.

In scope:

- Shared VS3 command transcript validator.
- VS3 source scenario verify transcript validation.
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
| `VS3-GATE-004` | `cornerstone scenario verify vs3-onprem-trusted-extension --json` and `cornerstone scenario gate <report> --json` include native command transcripts whose JSON tail mirrors the structured JSON payload. | CLI transcript, `stdout_json`, parseable JSON `stdout_tail`, source transcript validation, gate transcript validation. | PASS only if required self/source JSON tails match `schema_version`, `json_schema`, `status`, and contextual fields such as `final_verdict`, `scenario_set`, or `checked_report`; FAIL if the tail overclaims or contradicts structured JSON. |
| `VS3-REG-005` | VS3 wording and transcript evidence do not turn local/dev proof into production, real IdP, live provider, penetration-tested, human-accepted, or migration-ready claims. | Static overclaim lint, evidence manifest review, transcript tail mismatch negative test. | PASS only if tail and JSON evidence remain no stronger than local/dev proof; FAIL on unqualified readiness claim or `VS3_P_READY` tail overclaim. |

## Implementation Decision

Before this slice, VS3 self/source transcripts had structured `stdout_json`, but `stdout_tail` could be empty or could omit the same schema/verdict fields. That left room for a transcript tail to imply a stronger or different result than the structured payload.

Validator behavior now:

- Required VS3 self/source transcript validations require `stdout_tail` to be a non-empty list whose last entry parses as a JSON object.
- Required JSON tails must include `schema_version`, `json_schema`, and `status`.
- If present, tail fields must match `stdout_json` for `schema_version`, `json_schema`, `status`, `final_verdict`, `scenario_set`, and `checked_report`.
- Component proof transcripts may retain ordinary text stdout tails. Optional text tails are type-checked but are not treated as JSON mirrors unless the last tail entry is itself a parseable JSON object.

## Verification Evidence

Syntax:

- `python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py`
- Result: exit 0.

Focused scenario-gate tests:

- Command: `python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_preserves_source_proof_boundary_and_transcript tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_source_transcript_stdout_tail_mismatch tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_source_transcript_stdout_json_schema_mismatch`
- Result: `Ran 3 tests in 1.429s`, `OK`.

Focused local-checkpoint tests:

- Command: `python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_manifest_is_hash_backed_and_boundary_safe tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_self_transcript_validator_rejects_tampered_transcript`
- Result: `Ran 2 tests in 53.068s`, `OK`.

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
- `cornerstone scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json`: exit 0, `status=success`, `final_verdict=VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED`.
- `cornerstone human-gate record-scaffold --scope vs3 --output-dir reports/human-gates/vs3/record-templates --force --use-existing --json --output reports/human-gates/vs3/record-scaffold.json`: exit 0, `final_verdict=HUMAN_REQUIRED`.
- `cornerstone human-gate evidence-status --scope vs3 --record-dir reports/human-gates/vs3/record-templates --use-existing --json --output reports/human-gates/vs3/evidence-status.json`: exit 0, `final_verdict=HUMAN_REQUIRED`.
- `cornerstone human-gate review-kit --scope vs3 --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --use-existing --json --output reports/human-gates/vs3/review-kit.json`: exit 0, `final_verdict=HUMAN_REQUIRED`.
- `cornerstone human-gate vs3-p-gate --scope vs3 --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --output reports/human-gates/vs3/vs3-p-gate.json --json`: expected exit 4, `status=blocked`, `final_verdict=HUMAN_REQUIRED`.
- `cornerstone security vs3-local-checkpoint --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --output reports/security/vs3-local-checkpoint.json --json`: exit 0, `status=success`, `final_verdict=VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED`.
- `cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json`: exit 0, `status=success`, `final_verdict=VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED`.

Stdout tail scan:

- Checked artifacts: VS3 scenario report, local checkpoint, evidence-status, review-kit, VS3-P gate, and scenario gate stdout artifact.
- Result: 10 JSON-tail transcripts checked.
- Issues: `[]`.
- Pass marker: `TAIL_SCAN_PASS`.

Artifact hashes after refresh:

| Artifact | SHA-256 | Status / verdict |
|---|---|---|
| `reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json` | `14ab1c83e01373036ec6deefce5bc1fac85dc936f670eade89c9be29c9659c5e` | `success`, `VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED` |
| `reports/security/vs3-local-checkpoint.json` | `1a87f04d8585b8d8449ea7355c6ded0d49c04a050ae81eea8e4ae7656d070d8f` | `success`, `VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED` |
| `reports/human-gates/vs3/evidence-status.json` | `b9567f0ac6db147b45f4d36110814012aaff67f8ef332a4ea6c49e50ea311418` | `success`, `HUMAN_REQUIRED` |
| `reports/human-gates/vs3/review-kit.json` | `0af5c0c37975d53650b715dd2eeb4eaaee9216fd9b20baf46afca2b364163a60` | `success`, `HUMAN_REQUIRED` |
| `reports/human-gates/vs3/vs3-p-gate.json` | `958eb54bb920747467a8d32265f8a5b2179daa0e38abe0c0750f9181e5a4c97d` | `blocked`, `HUMAN_REQUIRED` |
| `/tmp/vs3-scenario-gate.json` | `73e2e0c54ef90972706baa8b00dbdb86ba4434140e4e6248ec21050feb674f83` | `success`, `VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED` |

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
