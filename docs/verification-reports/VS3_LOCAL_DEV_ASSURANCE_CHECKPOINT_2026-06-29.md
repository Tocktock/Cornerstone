# VS3 Local Dev Assurance Checkpoint

**Date:** 2026-06-29
**Latest freeze evidence:** 2026-06-30
**Status:** Local checkpoint verified after refreshing stale VS3-P human-gate evidence and repairing human-gate summary claim-boundary fields.
**Verdict:** `VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED`

This checkpoint records the VS3-L local/dev proof boundary for the frozen VS3 on-prem security and trusted extension contract. It does not claim VS3-P, production/on-prem readiness, security acceptance, real IdP readiness, real network readiness, live-provider readiness, migration/restore readiness, or human UX acceptance.

## Slice Contract

Goal:
- Prove the current VS3-L local/dev checkpoint end to end against the aggregate VS3 scenario report and the VS3-P human-gate block.

Scope:
- Aggregate VS3 scenario report.
- VS3-P human-gate block.
- Human-gate review kit, evidence-status, and record scaffold artifacts.
- Local checkpoint manifest and claim-boundary validation.
- Native `cornerstone ... --json` verification paths.

Non-scope:
- VS3-P approval.
- Production or on-prem deployment readiness.
- Real OIDC/SSO or enterprise IdP verification.
- Real network, DNS, proxy, firewall, service mesh, or sandbox verification.
- Live ConnectorHub/provider rehearsal.
- Human security, architecture, UX, or migration acceptance.

Done criteria:
- `cornerstone scenario verify vs3-onprem-trusted-extension --json` exits 0 and reports 50 AI-verifiable rows as `PASS`, 7 rows as `HUMAN_REQUIRED`, and 0 blocking rows.
- `cornerstone human-gate vs3-p-gate --json` exits 4 with `status=blocked`, `final_verdict=HUMAN_REQUIRED`, `unresolved_human_required_rows=7`, and `vs3_p_claim=NOT_CLAIMED`.
- `cornerstone security vs3-local-checkpoint --json` exits 0 with `final_verdict=VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED`.
- Negative evidence confirms zero VS3-P, production, security-acceptance, human-acceptance, real-IdP, real-network, live-provider, and migration/restore claims.
- Scenario, local checkpoint, evidence-status, review-kit, and VS3-P gate summaries expose all no-readiness-claim fields as explicit `false` values.
- Focused checkpoint tests pass.

## Full Scenario Mapping

All 57 VS3 rows are carried in this checkpoint. This mapping is a coverage guard, not a production readiness claim.

| Scenario IDs | Type | Classification | Proof Surface |
| --- | --- | --- | --- |
| `VS3-GATE-001`, `VS3-GATE-002`, `VS3-GATE-003`, `VS3-GATE-004` | MUST_PASS | in_this_slice | Aggregate report, matrix/contract checks, overclaim lint, native CLI verifier output. |
| `VS3-CTX-001`, `VS3-CTX-002`, `VS3-CTX-003`, `VS3-CTX-004`, `VS3-CTX-005` | MUST_PASS | in_this_slice | Local RequestContext proof report, policy/audit refs, negative forged-authority counters. |
| `VS3-RLS-001`, `VS3-RLS-002`, `VS3-RLS-003`, `VS3-RLS-004`, `VS3-RLS-005`, `VS3-RLS-006` | MUST_PASS | in_this_slice | Local Postgres/RLS proof report, migration/backup restore rehearsal artifacts, tenant-boundary counters. |
| `VS3-OPA-001`, `VS3-OPA-002`, `VS3-OPA-003`, `VS3-OPA-004`, `VS3-OPA-005` | MUST_PASS | in_this_slice | Local OPA/policy proof report, policy-decision fixtures, bundle lifecycle and redaction evidence. |
| `VS3-EGR-001`, `VS3-EGR-002`, `VS3-EGR-003`, `VS3-EGR-004`, `VS3-EGR-005`, `VS3-EGR-006` | MUST_PASS | in_this_slice | Local egress/sandbox proof report, sink counters, bypass denial matrix, prompt-injection authority counters. |
| `VS3-CON-001`, `VS3-CON-002`, `VS3-CON-003`, `VS3-CON-004`, `VS3-CON-005`, `VS3-CON-006` | MUST_PASS | in_this_slice | Local ConnectorHub/source proof report, credential custody checks, source-policy revoke tests, retry/quarantine evidence. |
| `VS3-TOOL-001`, `VS3-TOOL-002`, `VS3-TOOL-003`, `VS3-TOOL-004`, `VS3-TOOL-005`, `VS3-TOOL-006`, `VS3-TOOL-007` | MUST_PASS | in_this_slice | Local Tool SDK/registry proof report, signed package fixtures, inactive-install and activation denial counters. |
| `VS3-OBS-001`, `VS3-OBS-002`, `VS3-OBS-003` | MUST_PASS | in_this_slice | Local observability/audit proof, status JSON, DOM comparison, human-gate package validation. |
| `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-003`, `VS3-REG-004`, `VS3-REG-005`, `VS3-REG-006`, `VS3-REG-007`, `VS3-REG-008` | REGRESSION | in_this_slice | Fresh local regression reports, overclaim lint, UI/status checks, dependency and default-deny evidence. |
| `VS3-H01`, `VS3-H02`, `VS3-H03`, `VS3-H04`, `VS3-H05`, `VS3-H06`, `VS3-H07` | HUMAN_REQUIRED | HUMAN_REQUIRED | Human-owned approval, independent security review, real IdP, real network, live provider, operator UX, and migration/restore evidence. |

## Evidence

Initial stale-gate probe:

```text
PATH="$PWD:$PATH" cornerstone security vs3-local-checkpoint --json
exit_code=4
status=failed
final_verdict=VS3_L_LOCAL_DEV_ASSURANCE_NOT_VERIFIED
failed_conditions=["vs3_p_gate_scenario_report_hash_matches"]
negative_evidence.vs3_p_gate_scenario_report_hash_mismatches=1
```

Refresh VS3-P human gate from the current aggregate report:

```text
PATH="$PWD:$PATH" cornerstone human-gate vs3-p-gate \
  --scope vs3 \
  --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json \
  --output reports/human-gates/vs3/vs3-p-gate.json \
  --json
exit_code=4
status=blocked
final_verdict=HUMAN_REQUIRED
summary.scenario_count=57
summary.pass=50
summary.human_required=7
summary.unresolved_human_required_rows=7
summary.vs3_l_claim=LOCAL_DEV_ASSURANCE_VERIFIED
summary.vs3_p_claim=NOT_CLAIMED
negative_evidence.overclaim_boundary_violations=0
negative_evidence.vs3_p_unlocked_by_gate=0
```

Local checkpoint after refresh:

```text
PATH="$PWD:$PATH" cornerstone security vs3-local-checkpoint \
  --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json \
  --output reports/security/vs3-local-checkpoint.json \
  --json
exit_code=0
status=success
final_verdict=VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
summary.scenario_count=57
summary.pass=50
summary.human_required=7
summary.blocking=0
summary.ai_blocking_rows=0
summary.unresolved_human_required_rows=7
summary.vs3_l_claim=LOCAL_DEV_ASSURANCE_VERIFIED
summary.vs3_p_claim=NOT_CLAIMED
negative_evidence.vs3_p_gate_scenario_report_hash_mismatches=0
negative_evidence.overclaim_boundary_violations=0
negative_evidence.vs3_p_claimed_by_checkpoint=0
```

2026-06-30 freeze verification after human-gate summary claim-boundary repair:

```text
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
exit_code=0

python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_human_gate_review_kit_is_hash_backed_and_preparation_only \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_human_gate_evidence_status_reports_records_without_acceptance \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_p_gate_blocks_on_human_evidence_not_ai_rows
exit_code=0
result="Ran 3 tests in 129.304s / OK"

./cornerstone scenario verify vs3-onprem-trusted-extension \
  --json \
  --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
exit_code=0
status=success
final_verdict=VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
summary.pass=50
summary.human_required=7
summary.vs3_l_claim=LOCAL_DEV_ASSURANCE_VERIFIED
summary.vs3_p_claim=NOT_CLAIMED

./cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json
exit_code=0
status=success
final_verdict=VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
summary.pass=50
summary.human_required=7
summary.vs3_l_claim=LOCAL_DEV_ASSURANCE_VERIFIED
summary.vs3_p_claim=NOT_CLAIMED

./cornerstone human-gate vs3-p-gate \
  --scope vs3 \
  --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json \
  --output reports/human-gates/vs3/vs3-p-gate.json \
  --json
exit_code=4
status=blocked
final_verdict=HUMAN_REQUIRED
summary.unresolved_human_required_rows=7
summary.vs3_p_claim=NOT_CLAIMED

./cornerstone security vs3-local-checkpoint \
  --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json \
  --output reports/security/vs3-local-checkpoint.json \
  --json
exit_code=0
status=success
final_verdict=VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
summary.ai_blocking_rows=0
summary.unresolved_human_required_rows=7
summary.vs3_l_claim=LOCAL_DEV_ASSURANCE_VERIFIED
summary.vs3_p_claim=NOT_CLAIMED

scripts/verify_sot_docs.sh
exit_code=0

git diff --check
exit_code=0
```

Summary claim-boundary shape check:

```text
reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json status=success final=VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED missing=[] truthy={}
reports/security/vs3-local-checkpoint.json status=success final=VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED missing=[] truthy={}
reports/human-gates/vs3/evidence-status.json status=success final=HUMAN_REQUIRED missing=[] truthy={}
reports/human-gates/vs3/review-kit.json status=success final=HUMAN_REQUIRED missing=[] truthy={}
reports/human-gates/vs3/vs3-p-gate.json status=blocked final=HUMAN_REQUIRED missing=[] truthy={}
```

Focused tests:

```text
python3 -m compileall packages/cornerstone_cli
exit_code=0

python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_manifest_is_hash_backed_and_boundary_safe
exit_code=0

python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_stale_vs3_p_gate_scenario_report_hash \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_vs3_p_gate_from_different_scenario_report_path \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_p_gate_blocks_on_human_evidence_not_ai_rows
exit_code=0
```

Implementation hardening from this checkpoint:
- `packages/cornerstone_cli/scenarios.py` now uses idempotent cleanup for `reports/runtime/vs3-tool-registry-state`, so repeated or concurrent local verifier runs do not fail when a generated state directory is already removed.

## Evidence Artifacts

Hash-bearing generated artifacts:
- `reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json`
- `reports/security/vs3-local-checkpoint.json`
- `reports/human-gates/vs3/vs3-p-gate.json`
- `reports/human-gates/vs3/review-kit.json`
- `reports/human-gates/vs3/evidence-status.json`
- `reports/human-gates/vs3/record-scaffold.json`
- `reports/human-gates/vs3/record-templates/manifest.json`

Legacy compatible local checkpoint artifact:
- `reports/human-gates/vs3/vs3-local-checkpoint.json`

Per-gate human packages:
- `reports/human-gates/vs3/VS3-H01.json`
- `reports/human-gates/vs3/VS3-H02.json`
- `reports/human-gates/vs3/VS3-H03.json`
- `reports/human-gates/vs3/VS3-H04.json`
- `reports/human-gates/vs3/VS3-H05.json`
- `reports/human-gates/vs3/VS3-H06.json`
- `reports/human-gates/vs3/VS3-H07.json`

Artifact hashes are intentionally kept in the generated JSON manifests and checkpoint outputs instead of duplicated here, to avoid self-referential stale-hash prose.

## Human Gates

| Scenario | Required Evidence | Current Status | Release Impact |
| --- | --- | --- | --- |
| `VS3-H01` | Owner architecture/security approval. | HUMAN_REQUIRED | Blocks VS3-P and security-sensitive approval claims. |
| `VS3-H02` | Independent security review and retest. | HUMAN_REQUIRED | Blocks VS3-P. |
| `VS3-H03` | Real IdP mapping and revocation transcript. | HUMAN_REQUIRED | Blocks real IdP readiness. |
| `VS3-H04` | Real on-prem network/security control evidence. | HUMAN_REQUIRED | Blocks real on-prem network readiness. |
| `VS3-H05` | Approved live ConnectorHub/provider rehearsal. | HUMAN_REQUIRED | Blocks live-provider readiness. |
| `VS3-H06` | Operator UX/trust accept or reject record. | HUMAN_REQUIRED | Blocks human UX acceptance. |
| `VS3-H07` | Human-supervised migration/backup/restore/rollback drill. | HUMAN_REQUIRED | Blocks migration/restore readiness. |

## Source Tree Boundary

The current VS3-L reports intentionally record a dirty source tree snapshot instead of treating the checkpoint as a clean release artifact. The current default local checkpoint report is `reports/security/vs3-local-checkpoint.json`; it records `source_tree.dirty_paths` and keeps `summary.scenario_report_source_tree_current_status=passed`, meaning the generated report is bound to the dirty tree it verified. This checkpoint remains a local/dev assurance package, not PR merge state, release state, or production readiness.

## Decision

VS3-L local/dev assurance is checkpointed for the current aggregate evidence surface. VS3-P remains blocked by seven unresolved human gates. The next action should prepare human review from the existing human-gate package without marking human rows `PASS`.
