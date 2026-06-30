# VS3 Tool Registry Checkpoint - 2026-06-29

**Status:** VS3-6 local Tool SDK / signed registry / Agent Pack slice PASS.
**Scope:** `VS3-TOOL-001` through `VS3-TOOL-007` only.
**Proof boundary:** Local deterministic Agent Pack and tool-package fixtures, local trusted-registry state, signed/SBOM/provenance fixture metadata, local pack install/activation/update/rollback commands, runtime sandbox negative fixture, native CLI output.

This checkpoint does not claim VS3-P, production/on-prem readiness, production registry governance, real WASM runtime isolation, live-provider readiness, independent penetration-test completion, signing-root/key-rotation governance, or human security acceptance.

## Slice Contract

Goal:

- Verify the local Tool SDK and trusted registry baseline through native `cornerstone ... --json` commands.
- Capture negative evidence for missing package metadata, unsigned/tampered/stale/revoked/unknown packages, install-as-activation, ungranted capability use, post-revocation use, undeclared file/env/network/shell/model/connector/memory access, silent behavior updates, rollback failure, and behavior-changing emergency patches without review.
- Keep production registry governance, real runtime isolation, signing-root operations, live-provider access, independent security review, and human acceptance as `HUMAN_REQUIRED`.

Selected scenarios:

| Scenario | Status in this checkpoint | Required proof surface |
|---|---|---|
| `VS3-TOOL-001` | PASS | Local sample tool package manifest contains required metadata, runtime grants, ConnectorHub requirements, signature, SBOM, and provenance. |
| `VS3-TOOL-002` | PASS | Local trusted registry accepts signed first-party fixture and rejects unsigned, tampered, stale, revoked, or unknown-source packages. |
| `VS3-TOOL-003` | PASS | Install and install dry-run make pack available but inactive; inactive connector and capability attempts deny. |
| `VS3-TOOL-004` | PASS | Activation dry-run, activation, ungranted-capability denial, granted ConnectorHub-mediated capability, revoke, and post-revoke denial are explicit, reversible, policy-linked, and audited. |
| `VS3-TOOL-005` | PASS | Runtime sandbox negative suite denies undeclared file, env, network, shell, model, connector, and memory access with zero secret findings. |
| `VS3-TOOL-006` | PASS | Update dry-run exposes diff/evaluation/rollback path; behavior-changing update without approval is denied; approved update is explicit. |
| `VS3-TOOL-007` | PASS | Rollback returns to pinned version; non-behavior emergency patch is governed and audited; behavior-changing emergency patch remains review-required. |

Full VS3 mapping remains the frozen 57-row inventory: 42 `MUST_PASS`, 8 `REGRESSION`, and 7 `HUMAN_REQUIRED`. Non-tool rows are outside this checkpoint except where the aggregate scenario report is cited as supporting local scenario-gate context.

## Implementation Delta

- `tests/scenario/test_scaffold_cli.py` now asserts deeper native VS3 tool-registry CLI proof fields instead of only checking command success and the first scenario row.
- New assertions cover:
  - local-only proof boundary and no VS3-P / production-registry / real-WASM claim;
  - manifest required fields, runtime grant classes, ConnectorHub requirements, signature, SBOM, and provenance;
  - trusted registry certification and negative package rejection cases;
  - sandbox negative suite decisions and zero side effects;
  - silent update, rollback, and behavior-changing emergency-patch negative evidence;
  - native `pack update` dry-run, update-without-approval denial, approved update, rollback, emergency patch, and behavior-changing emergency-patch denial paths.

## Command Evidence

Focused compile:

```text
python3 -m compileall tests/scenario/test_scaffold_cli.py
Compiling 'tests/scenario/test_scaffold_cli.py'...
exit=0
```

Focused tests:

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_tool_registry_proof_is_local_and_negative_evidence_backed \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_tool_registry_cli_paths_are_native \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_onprem_trusted_extension_verify_closes_local_dev_ai_rows

Ran 3 tests in 29.102s
OK
```

Direct CLI probes:

```text
cornerstone security vs3-tool-registry --json
exit=0
status=success
schema_version=cs.vs3_tool_registry_proof.v0
evidence_refs=5
audit_refs=19
policy_decision_refs=7
negative_nonzero={}
```

```text
cornerstone tool verify --json
exit=0
status=success
tool_verify_schema_version=cs.vs3_tool_verify.v0
evidence_refs=1
audit_refs=19
policy_decision_refs=7
```

Native pack lifecycle probe against `tmp/vs3-tool-checkpoint-cli`:

```text
cornerstone pack import --manifest fixtures/vs3/tool_registry/sample_tool_pack_manifest.json --state-dir tmp/vs3-tool-checkpoint-cli --json
exit=0
agent_pack.status=available
evidence_refs=2
audit_refs=1

cornerstone pack install pack_vs3_tool_registry_sample --dry-run --state-dir tmp/vs3-tool-checkpoint-cli --json
exit=0
install.status=install_preview
install.can_act=false
evidence_refs=1
audit_refs=1

cornerstone pack install pack_vs3_tool_registry_sample --state-dir tmp/vs3-tool-checkpoint-cli --json
exit=0
install.status=installed
install.activation_status=inactive
install.can_act=false
evidence_refs=1
audit_refs=1

cornerstone pack activate pack_vs3_tool_registry_sample --dry-run --grant artifact.read --grant tool.local.evaluate --grant connector.mock.read --state-dir tmp/vs3-tool-checkpoint-cli --json
exit=0
activation.status=activation_preview
activation.grant_applied=false
evidence_refs=1
audit_refs=1
policy_decision_refs=1

cornerstone pack activate pack_vs3_tool_registry_sample --grant artifact.read --grant tool.local.evaluate --grant connector.mock.read --state-dir tmp/vs3-tool-checkpoint-cli --json
exit=0
activation.status=active
activation.grant_applied=true
evidence_refs=1
audit_refs=1
policy_decision_refs=1

cornerstone pack revoke pack_vs3_tool_registry_sample --reason checkpoint --state-dir tmp/vs3-tool-checkpoint-cli --json
exit=0
revocation.status=revoked
revocation.capabilities_active_after_revoke=[]
evidence_refs=1
audit_refs=1
policy_decision_refs=1

cornerstone pack update pack_vs3_tool_registry_sample --to-version 1.1.0 --dry-run --state-dir tmp/vs3-tool-checkpoint-cli --json
exit=0
pack_update.status=dry_run
pack_update.applied=false
pack_update.evaluation_gate.status=pass
pack_update.diff.rollback_path present
evidence_refs=1
audit_refs=1

cornerstone pack update pack_vs3_tool_registry_sample --to-version 1.1.0 --state-dir tmp/vs3-tool-checkpoint-cli --json
exit=8
status=failed
error_code=CS_PACK_APPROVAL_REQUIRED
policy=agent_pack_behavior_update_requires_owner_approval
audit_refs=1
policy_decision_refs=1

cornerstone pack update pack_vs3_tool_registry_sample --to-version 1.1.0 --approve --state-dir tmp/vs3-tool-checkpoint-cli --json
exit=0
pack_update.status=approved_applied
pack_update.applied=true
pack_update.behavior_changing_silent_apply=false
evidence_refs=1
audit_refs=1

cornerstone pack rollback pack_vs3_tool_registry_sample --to-version 1.0.0 --reason checkpoint --state-dir tmp/vs3-tool-checkpoint-cli --json
exit=0
pack_rollback.status=rolled_back
pack_rollback.to_version=1.0.0
pack_rollback.changes_recorded=true
evidence_refs=1
audit_refs=1

cornerstone pack emergency-patch pack_vs3_tool_registry_sample --patch-version 1.0.1-security --state-dir tmp/vs3-tool-checkpoint-cli --json
exit=0
security_patch.status=applied
security_patch.behavior_change=false
security_patch.compatibility_checks.authority_expands=false
evidence_refs=1
audit_refs=1

cornerstone pack emergency-patch pack_vs3_tool_registry_sample --patch-version 1.0.1-security --behavior-change --state-dir tmp/vs3-tool-checkpoint-cli --json
exit=8
status=failed
error_code=CS_PACK_APPROVAL_REQUIRED
policy=agent_pack_emergency_patch_behavior_change_review
audit_refs=1
policy_decision_refs=1
```

Tool registry proof report:

```text
reports/security/vs3-tool-registry-proof.json
status=success
schema_version=cs.vs3_tool_registry_proof.v0
scenario_status:
  VS3-TOOL-001=PASS
  VS3-TOOL-002=PASS
  VS3-TOOL-003=PASS
  VS3-TOOL-004=PASS
  VS3-TOOL-005=PASS
  VS3-TOOL-006=PASS
  VS3-TOOL-007=PASS
checks:
  no_vs3_l_vs3_p_or_production_registry_claim=true
  vs3_tool_001_manifest_package_signature_sbom=true
  vs3_tool_002_trusted_registry_rejects_bad_packages=true
  vs3_tool_003_install_not_activation=true
  vs3_tool_004_activation_grants_reversible_audited=true
  vs3_tool_005_runtime_sandbox_denies_undeclared_access=true
  vs3_tool_006_update_diff_evaluation_gate_no_silent_apply=true
  vs3_tool_007_rollback_and_emergency_patch_policy=true
proof_boundary.surface=local_tool_registry_fixture
proof_boundary.production_registry=NOT_CLAIMED
proof_boundary.real_wasm_runtime=NOT_CLAIMED
proof_boundary.live_provider=HUMAN_REQUIRED
proof_boundary.human_security_acceptance=HUMAN_REQUIRED
proof_boundary.vs3_l=NOT_CLAIMED
proof_boundary.vs3_p=NOT_CLAIMED
```

Negative evidence:

```text
missing_required_manifest_fields=0
missing_runtime_grant_classes=0
missing_signature=0
missing_sbom=0
missing_provenance=0
unsigned_packages_accepted=0
tampered_packages_accepted=0
stale_packages_accepted=0
revoked_packages_accepted=0
unknown_source_packages_accepted=0
install_as_activation_count=0
inactive_connector_requests_allowed=0
inactive_capability_attempts_allowed=0
activation_preview_applied_authority=0
ungranted_capabilities_allowed=0
post_revocation_capability_allows=0
undeclared_file_reads=0
undeclared_env_reads=0
undeclared_network_calls=0
undeclared_shell_processes=0
undeclared_model_routes=0
undeclared_connector_calls=0
undeclared_memory_writes=0
secret_scanner_findings=0
silent_behavior_updates_applied=0
rollback_failures=0
emergency_patch_authority_expansions=0
behavior_changing_emergency_patches_applied_without_review=0
vs3_l_claimed=0
vs3_p_claimed=0
production_registry_claimed=0
real_wasm_runtime_claimed=0
```

Aggregate scenario verification:

```text
cornerstone scenario verify vs3-onprem-trusted-extension --json \
  --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
exit=0
status=success
final_verdict=VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
scenario_result_count=57
status_counts: PASS=50, HUMAN_REQUIRED=7
type_counts: MUST_PASS=42, REGRESSION=8, HUMAN_REQUIRED=7
proof_boundary.vs3_p=NOT_CLAIMED
proof_boundary.production_onprem=NOT_CLAIMED
proof_boundary.live_provider=NOT_CLAIMED
proof_boundary.security_acceptance=NOT_CLAIMED
```

Aggregate Tool row evidence:

```text
VS3-TOOL-001 PASS evidence_refs=6 audit_refs=19 policy_decision_refs=7 source_report_refs=reports/security/vs3-tool-registry-proof.json
VS3-TOOL-002 PASS evidence_refs=7 audit_refs=19 policy_decision_refs=7 source_report_refs=reports/security/vs3-tool-registry-proof.json
VS3-TOOL-003 PASS evidence_refs=7 audit_refs=19 policy_decision_refs=7 source_report_refs=reports/security/vs3-tool-registry-proof.json
VS3-TOOL-004 PASS evidence_refs=7 audit_refs=19 policy_decision_refs=7 source_report_refs=reports/security/vs3-tool-registry-proof.json
VS3-TOOL-005 PASS evidence_refs=7 audit_refs=19 policy_decision_refs=7 source_report_refs=reports/security/vs3-tool-registry-proof.json
VS3-TOOL-006 PASS evidence_refs=6 audit_refs=19 policy_decision_refs=7 source_report_refs=reports/security/vs3-tool-registry-proof.json
VS3-TOOL-007 PASS evidence_refs=7 audit_refs=19 policy_decision_refs=7 source_report_refs=reports/security/vs3-tool-registry-proof.json
```

Aggregate scenario gate:

```text
cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json
exit=0
status=success
coverage_validation.status=passed
row_ref_validation.status=passed
claim_boundary_validation.status=passed
missing_scenario_ids=[]
missing_evidence_ref_rows=[]
missing_source_report_ref_rows=[]
missing_audit_ref_rows=[]
invalid_claim_boundary_fields=[]
```

## Scenario Evidence Mapping

| Scenario | Evidence refs |
|---|---|
| `VS3-TOOL-001` | `reports/security/vs3-tool-registry-proof.json`, `fixtures/vs3/tool_registry/sample_tool_pack_manifest.json`, `fixtures/vs3/tool_registry/sample_tool_pack.signature.json`, `fixtures/vs3/tool_registry/sample_tool_pack.sbom.spdx.json`, `fixtures/vs3/tool_registry/tool_vs3_local_evidence_checker.wasm.fixture`, `cornerstone tool verify --json` |
| `VS3-TOOL-002` | `reports/security/vs3-tool-registry-proof.json`, `cornerstone security vs3-tool-registry --json`, `trusted_registry_negative_tests`, `fixtures/vs3/tool_registry/sample_tool_pack_manifest.json`, `fixtures/vs3/tool_registry/sample_tool_pack.signature.json`, `fixtures/vs3/tool_registry/sample_tool_pack.sbom.spdx.json`, `reports/runtime/vs3-tool-registry-state` |
| `VS3-TOOL-003` | `reports/security/vs3-tool-registry-proof.json`, `cornerstone pack install --dry-run --json`, `cornerstone pack install --json`, `inactive_connector_denial`, `inactive_capability_denial`, `fixtures/vs3/tool_registry/sample_tool_pack_manifest.json`, `reports/runtime/vs3-tool-registry-state` |
| `VS3-TOOL-004` | `reports/security/vs3-tool-registry-proof.json`, `cornerstone pack activate --dry-run --json`, `cornerstone pack activate --json`, `cornerstone pack revoke --json`, `ungranted_capability_denial`, `granted_capability_attempt`, `reports/runtime/vs3-tool-registry-state` |
| `VS3-TOOL-005` | `reports/security/vs3-tool-registry-proof.json`, `cornerstone tool verify --json`, `sandbox_suite:vs3_tool_runtime_negative`, `secret_scan`, `fixtures/vs3/tool_registry/tool_vs3_local_evidence_checker.wasm.fixture`, `reports/runtime/vs3-tool-registry-state` |
| `VS3-TOOL-006` | `reports/security/vs3-tool-registry-proof.json`, `cornerstone pack update --dry-run --json`, `cornerstone pack update --json`, `cornerstone pack update --approve --json`, `update_diff`, `evaluation_gate` |
| `VS3-TOOL-007` | `reports/security/vs3-tool-registry-proof.json`, `cornerstone pack rollback --json`, `cornerstone pack emergency-patch --json`, `cornerstone pack emergency-patch --behavior-change --json`, `rollback_record`, `security_patch_record`, `behavior_patch_block` |

## Human Required

Still `HUMAN_REQUIRED` and not converted to PASS:

- Owner/security approval for production dependency, signing-root, key rotation, registry governance, and emergency process ownership.
- Independent security review of extension, sandbox, registry, signing, and bypass resistance.
- Real WASM/runtime isolation validation outside deterministic fixture.
- Real on-prem topology and live-provider rehearsal where tool/ConnectorHub capabilities cross real systems.
- Human operator UX/trust review for tool registry, activation, update, rollback, and emergency patch surfaces.
- VS3-P release approval.

## Deliberately Not Done

- No production registry, signing root, or key-rotation process was exercised.
- No real WASM runtime isolation was claimed.
- No live provider or external credential was used.
- No independent penetration test, production/on-prem readiness, VS3-P, or human security acceptance was claimed.
- No unreviewed behavior-changing update or emergency patch was allowed.

## Decision

`VS3-TOOL-001` through `VS3-TOOL-007` are locally AI-verifiable and PASS for the VS3-6 local/dev proof surface. Continue to the next slice only with the same proof-boundary separation.
