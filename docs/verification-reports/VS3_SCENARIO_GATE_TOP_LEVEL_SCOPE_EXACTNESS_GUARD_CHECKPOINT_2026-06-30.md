# VS3 Scenario Gate Top-Level Scope Exactness Guard Checkpoint

**Date:** 2026-06-30 KST
**Scope:** Local deterministic VS3 scenario gate hardening
**Status:** AI-verifiable slice passed; VS3-P remains HUMAN_REQUIRED

## Slice Contract

Goal:

- Prevent a VS3 local-dev assurance report from changing top-level `tenant_id`, `owner_id`, `namespace_id`, or `workspace_id` while still passing `cornerstone scenario gate ... --json`.

In this slice:

- `VS3-CTX-001`: report-level scope must match the trusted local fixture context used by traceability and rows.
- `VS3-CTX-002`: caller-controlled or forged top-level scope values must not influence the gate outcome.
- `VS3-GATE-004`: native `cornerstone scenario gate ... --json` must expose deterministic validation status and errors for scope drift.
- `VS3-REG-004`: scenario gate coverage must fail on metadata drift before release claims.

Later slices:

- `VS3-GATE-001`, `VS3-GATE-002`, `VS3-GATE-003`.
- `VS3-CTX-003`, `VS3-CTX-004`, `VS3-CTX-005`.
- All `VS3-RLS-*`, `VS3-OPA-*`, `VS3-EGR-*`, `VS3-CON-*`, `VS3-TOOL-*`, and `VS3-OBS-*` functional rows beyond this verifier guard.
- Final regression breadth in `VS3-REG-001`, `VS3-REG-002`, `VS3-REG-003`, `VS3-REG-005`, `VS3-REG-006`, `VS3-REG-007`, and `VS3-REG-008`.

Human-required:

- `VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`.

Out of scope:

- Real IdP, real network, live provider, independent security review, production/on-prem readiness, migration/restore readiness, and human UX acceptance.

## Full VS3 Scenario Mapping

| Scenario | Type | Classification | Required proof surface | Reason |
|---|---|---|---|---|
| VS3-GATE-001 | MUST_PASS | later_slice | Evidence reconciliation CLI/report lint | Not changed by top-level scope exactness. |
| VS3-GATE-002 | MUST_PASS | later_slice | Docs verifier and matrix structural checks | Matrix freeze remains covered by existing docs gates. |
| VS3-GATE-003 | MUST_PASS | later_slice | Static overclaim lint | Overclaim wording guard is not changed here. |
| VS3-GATE-004 | MUST_PASS | in_this_slice | Native scenario verify/gate JSON transcript | Gate must reject forged top-level scope metadata. |
| VS3-CTX-001 | MUST_PASS | in_this_slice | Normalized scope fixture and gate validation | Top-level scope must match trusted local fixture context. |
| VS3-CTX-002 | MUST_PASS | in_this_slice | Negative forged-scope gate test | Forged top-level scope must not be accepted. |
| VS3-CTX-003 | MUST_PASS | later_slice | Allow, revoke, retry fixture | Revocation behavior is not changed here. |
| VS3-CTX-004 | MUST_PASS | later_slice | Fault matrix over entry points | Malformed/missing RequestContext behavior is not changed here. |
| VS3-CTX-005 | MUST_PASS | later_slice | Same-tenant workspace/mission fixture | Mission/workspace authority is not changed here. |
| VS3-RLS-001 | MUST_PASS | later_slice | Schema inventory and create/read/null-insert tests | Postgres schema scope is not changed here. |
| VS3-RLS-002 | MUST_PASS | later_slice | Two-tenant DB integration matrix | RLS read isolation is not changed here. |
| VS3-RLS-003 | MUST_PASS | later_slice | Cross-tenant mutation matrix | RLS write denial is not changed here. |
| VS3-RLS-004 | MUST_PASS | later_slice | Pool/worker stress test | Transaction context reset is not changed here. |
| VS3-RLS-005 | MUST_PASS | later_slice | Migration/quarantine/rollback tests | Migration behavior is not changed here. |
| VS3-RLS-006 | MUST_PASS | later_slice | Backup/restore tenant-safe rehearsal | Backup and restore behavior is not changed here. |
| VS3-OPA-001 | MUST_PASS | later_slice | PolicyInput schema fixtures | Policy input builders are not changed here. |
| VS3-OPA-002 | MUST_PASS | later_slice | OPA/Rego and HTTP decision tests | PolicyDecision semantics are not changed here. |
| VS3-OPA-003 | MUST_PASS | later_slice | OPA service hardening tests | OPA service exposure is not changed here. |
| VS3-OPA-004 | MUST_PASS | later_slice | Bundle lifecycle tests | Policy bundle activation is not changed here. |
| VS3-OPA-005 | MUST_PASS | later_slice | Secret-canary decision log scan | Policy log masking is not changed here. |
| VS3-EGR-001 | MUST_PASS | later_slice | Controlled forbidden sink counters | Runtime egress denial is not changed here. |
| VS3-EGR-002 | MUST_PASS | later_slice | Approved local provider sink test | Allowed governed egress is not changed here. |
| VS3-EGR-003 | MUST_PASS | later_slice | URL/DNS/redirect matrix | Destination normalization is not changed here. |
| VS3-EGR-004 | MUST_PASS | later_slice | Sandbox adversarial suite | Sandbox host access behavior is not changed here. |
| VS3-EGR-005 | MUST_PASS | later_slice | Egress component outage tests | Egress fail-closed behavior is not changed here. |
| VS3-EGR-006 | MUST_PASS | later_slice | Prompt-injection fixtures | Untrusted content authority behavior is not changed here. |
| VS3-CON-001 | MUST_PASS | later_slice | Projection delivery crash/retry fixture | Artifact-before-ack behavior is not changed here. |
| VS3-CON-002 | MUST_PASS | later_slice | Connector write-denial tests | GitHub read-only connector behavior is not changed here. |
| VS3-CON-003 | MUST_PASS | later_slice | Secret canary scan | Credential custody behavior is not changed here. |
| VS3-CON-004 | MUST_PASS | later_slice | SourcePolicy update/revoke fixture | Source policy enforcement is not changed here. |
| VS3-CON-005 | MUST_PASS | later_slice | Capture consent/pause/revoke fixture | WatchAgent/browser capture behavior is not changed here. |
| VS3-CON-006 | MUST_PASS | later_slice | Retry/quarantine fault fixture | Connector retry/quarantine behavior is not changed here. |
| VS3-TOOL-001 | MUST_PASS | later_slice | Tool package manifest/SBOM verification | Tool package format is not changed here. |
| VS3-TOOL-002 | MUST_PASS | later_slice | Registry signature positive/negative tests | Trusted registry behavior is not changed here. |
| VS3-TOOL-003 | MUST_PASS | later_slice | Installed-inactive execution attempts | Install/activation boundary is not changed here. |
| VS3-TOOL-004 | MUST_PASS | later_slice | Activation grant dry-run/revoke tests | Activation grant behavior is not changed here. |
| VS3-TOOL-005 | MUST_PASS | later_slice | Runtime sandbox negative suite | Tool runtime sandbox behavior is not changed here. |
| VS3-TOOL-006 | MUST_PASS | later_slice | Update dry-run/evaluation gate tests | Pack update behavior is not changed here. |
| VS3-TOOL-007 | MUST_PASS | later_slice | Rollback/emergency patch simulation | Pack rollback behavior is not changed here. |
| VS3-OBS-001 | MUST_PASS | later_slice | Fault-injection status tests | Operator status behavior is not changed here. |
| VS3-OBS-002 | MUST_PASS | later_slice | Audit contract and tamper fixture | Audit ledger behavior is not changed here. |
| VS3-OBS-003 | MUST_PASS | later_slice | Human-gate package generation validation | Human-gate package behavior is not changed here. |
| VS3-REG-001 | REGRESSION | later_slice | Fresh VS0 gates | VS0 regression breadth is not rerun in this narrow slice. |
| VS3-REG-002 | REGRESSION | later_slice | Fresh VS1 gate and cross-scope tests | VS1 regression breadth is not rerun in this narrow slice. |
| VS3-REG-003 | REGRESSION | later_slice | Red-team authority fixture suite | Prompt/authority injection breadth is not changed here. |
| VS3-REG-004 | REGRESSION | in_this_slice | Coverage and metadata mutation test | Scope metadata drift must fail before release claims. |
| VS3-REG-005 | REGRESSION | later_slice | Static overclaim lint and evidence manifest review | Claim wording is not changed here. |
| VS3-REG-006 | REGRESSION | later_slice | UI/nav review and browser/DOM check | Product UX is not changed here. |
| VS3-REG-007 | REGRESSION | later_slice | Dependency diff and approval-gate check | No dependency change in this slice. |
| VS3-REG-008 | REGRESSION | later_slice | Fresh/reset/partial-config integration suite | Secure defaults breadth is not changed here. |
| VS3-H01 | HUMAN_REQUIRED | HUMAN_REQUIRED | Owner architecture/security approval | Requires signed owner evidence. |
| VS3-H02 | HUMAN_REQUIRED | HUMAN_REQUIRED | Independent security review and retest | Requires independent reviewer evidence. |
| VS3-H03 | HUMAN_REQUIRED | HUMAN_REQUIRED | Real IdP mapping and revocation test | Requires real IdP evidence. |
| VS3-H04 | HUMAN_REQUIRED | HUMAN_REQUIRED | Real on-prem network controls review | Requires target topology evidence. |
| VS3-H05 | HUMAN_REQUIRED | HUMAN_REQUIRED | Live ConnectorHub/provider rehearsal | Requires live credentials and redacted provider evidence. |
| VS3-H06 | HUMAN_REQUIRED | HUMAN_REQUIRED | Human operator UX/trust review | Requires human acceptance or rejection evidence. |
| VS3-H07 | HUMAN_REQUIRED | HUMAN_REQUIRED | Human-supervised migration/backup/restore drill | Requires signed drill evidence. |

## Baseline Gap

Local repro before the fix:

```text
seed_exit 0
tenant_id exit 0 status success errors [] traceability passed
owner_id exit 0 status success errors [] traceability passed
namespace_id exit 0 status success errors [] traceability passed
workspace_id exit 0 status success errors [] traceability passed
```

The gate validated `traceability.scope` and row scopes, but it did not reject independent drift in top-level report scope fields.

## Change

`packages/cornerstone_cli/main.py` now requires:

- top-level `tenant_id == local-dev`;
- top-level `owner_id == local-user`;
- top-level `namespace_id == personal`;
- top-level `workspace_id == default`;
- `traceability.scope == VS3_SCENARIO_VERIFY_SCOPE`;
- every scenario row `scope == VS3_SCENARIO_VERIFY_SCOPE`.

Failure payloads expose:

- `traceability_validation.expected_scope`;
- `traceability_validation.actual_scope`;
- `traceability_validation.actual_top_level_scope`;
- the exact invalid top-level field.

`tests/scenario/test_scaffold_cli.py` adds:

- `test_vs3_scenario_gate_rejects_local_dev_claim_with_top_level_scope_mismatch`.

## Verification

Focused compile:

```bash
python3 -m py_compile packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
```

Result: exit code `0`.

Focused new test:

```bash
python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_top_level_scope_mismatch
```

Result:

```text
Ran 1 test in 24.013s
OK
```

Direct native tamper probe after the fix:

```text
seed_exit 0
tenant_id exit 4 status failed errors ['CS_VS3_TRACEABILITY_METADATA_MISSING'] traceability failed invalid ['tenant_id']
owner_id exit 4 status failed errors ['CS_VS3_TRACEABILITY_METADATA_MISSING'] traceability failed invalid ['owner_id']
namespace_id exit 4 status failed errors ['CS_VS3_TRACEABILITY_METADATA_MISSING'] traceability failed invalid ['namespace_id']
workspace_id exit 4 status failed errors ['CS_VS3_TRACEABILITY_METADATA_MISSING'] traceability failed invalid ['workspace_id']
```

Adjacent traceability gate tests:

```bash
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_without_traceability_metadata \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_wrong_traceability_transcript_path \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_traceability_count_or_source_ref_mismatch \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_scenario_gate_rejects_local_dev_claim_with_top_level_scope_mismatch
```

Result:

```text
Ran 4 tests in 93.611s
OK
```

Clean native verify and gate:

```text
verify_exit 0
verify_status success
verify_final_verdict VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED
verify_summary {'blocking': 0, 'fail': 0, 'human_required': 7, 'not_run': 0, 'not_verified': 0, 'pass': 50, 'product_feature_claims': 'VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_GATES_REQUIRED', 'scenario_count': 57}
gate_exit 0
gate_status success
gate_errors []
traceability_status passed
traceability_invalid []
actual_top_level_scope {'namespace_id': 'personal', 'owner_id': 'local-user', 'tenant_id': 'local-dev', 'workspace_id': 'default'}
```

Shared local checkpoint traceability path:

```bash
python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_manifest_is_hash_backed_and_boundary_safe
```

Result:

```text
Ran 1 test in 49.939s
OK
```

## Proof Boundary

This checkpoint proves only local deterministic VS3 scenario-gate scope exactness.

It does not prove VS3-P, production/on-prem readiness, real IdP readiness, live-provider readiness, independent security acceptance, migration/restore readiness, or human UX acceptance.

`VS3-H01` through `VS3-H07` remain `HUMAN_REQUIRED`.
