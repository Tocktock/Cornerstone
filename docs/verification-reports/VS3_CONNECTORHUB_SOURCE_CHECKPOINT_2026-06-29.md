# VS3 ConnectorHub Source Checkpoint - 2026-06-29

**Status:** VS3-5 local ConnectorHub source/capture slice PASS.
**Scope:** `VS3-CON-001` through `VS3-CON-006` only.
**Proof boundary:** Local deterministic ConnectorHub/source fixtures, read-only GitHub connector fixture, credential-redaction fixture, SourcePolicy revoke/cross-scope fixture, WatchAgent/browser-capture fixture, retry/quarantine and prompt-injection fixture, native CLI output.

This checkpoint does not claim VS3-P, production/on-prem readiness, live GitHub/provider readiness, real credential custody, real macOS/Chrome capture acceptance, independent security review, or human security/UX acceptance.

## Slice Contract

Goal:

- Verify the local ConnectorHub boundary and read-only source confidence path through native `cornerstone ... --json` commands.
- Capture negative evidence for zero ack-before-commit, zero provider mutation, zero raw credential exposure, zero stale or cross-scope SourcePolicy delivery, zero disallowed raw capture output, and zero unauthorized side effects from failed or malicious connector delivery.
- Keep live credentials, real provider rehearsal, physical-device capture, and human operator acceptance as `HUMAN_REQUIRED`.

Selected scenarios:

| Scenario | Status in this checkpoint | Required proof surface |
|---|---|---|
| `VS3-CON-001` | PASS | Projection delivery commits immutable Artifact and evidence metadata before ack; retry cannot acknowledge uncommitted truth. |
| `VS3-CON-002` | PASS | GitHub source fixture is read-only; no write mappings, mutation commands, write calls, or external mutations. |
| `VS3-CON-003` | PASS | Connector credentials remain in ConnectorHub; Product payloads expose refs/redaction only. |
| `VS3-CON-004` | PASS | SourcePolicySnapshot is scoped, auditable, revocable, and enforced on next delivery/capture. |
| `VS3-CON-005` | PASS | Local WatchAgent/browser capture fixture is consented, bounded, pauseable, revocable, scope-visible, and summary-only. |
| `VS3-CON-006` | PASS | Failed, duplicate, stale, or prompt-injected delivery is retried or quarantined without memory, policy, action, egress, or authority side effects. |

Full VS3 mapping remains the frozen 57-row inventory: 42 `MUST_PASS`, 8 `REGRESSION`, and 7 `HUMAN_REQUIRED`. Non-ConnectorHub rows are outside this checkpoint except where the aggregate scenario report is cited as supporting local scenario-gate context.

## Implementation Delta

- `tests/scenario/test_scaffold_cli.py` now asserts the VS3 ConnectorHub native CLI proof fields instead of only checking command success.
- New assertions cover:
  - local-only proof boundary and `live_provider=HUMAN_REQUIRED`;
  - SourcePolicy allow, revoke denial, cross-scope denial, and zero stale delivery;
  - projection Artifact immutability, original preservation, ack-after-commit attempts, retry-after-commit hash reuse, and zero ack-before-commit;
  - duplicate/stale/prompt-injected delivery quarantine and zero unauthorized side effects;
  - GitHub read-only manifest, denied/quarantined write attempts, controlled egress denials, and zero provider mutation;
  - ConnectorHub credential custody, credential-ref-only output, redacted provider payloads, and zero secret scanner findings;
  - capture consent, visible scope, summary-only mode, pause/revoke controls, and zero disallowed raw output.

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
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_connectorhub_source_proof_is_local_and_negative_evidence_backed \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_connectorhub_cli_paths_are_native \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_onprem_trusted_extension_verify_closes_local_dev_ai_rows

Ran 3 tests in 27.692s
OK
```

Direct CLI probes:

```text
cornerstone security vs3-connectorhub-source --json
exit=0
status=success
schema_version=cs.vs3_connectorhub_source_proof.v0
evidence_refs=3
audit_refs=16
policy_decision_refs=4
negative_nonzero={}
```

```text
cornerstone connector source-policy show --json
exit=0
status=success
connector_source_policy_show_schema_version=cs.vs3_connector_source_policy_show.v0
evidence_refs=3
audit_refs=16
policy_decision_refs=4
negative_nonzero={}
```

```text
cornerstone connector projection verify --json
exit=0
status=success
connector_projection_verify_schema_version=cs.vs3_connector_projection_verify.v0
evidence_refs=6
audit_refs=16
policy_decision_refs=4
negative_nonzero={}
```

```text
cornerstone connector action dry-run --json
exit=0
status=success
connector_action_dry_run_schema_version=cs.vs3_connector_action_dry_run.v0
evidence_refs=4
audit_refs=16
policy_decision_refs=4
negative_nonzero={}
```

```text
cornerstone connector capture verify --profile vs3 --json
exit=0
status=success
connector_capture_verify_schema_version=cs.vs3_connector_capture_verify.v0
evidence_refs=2
audit_refs=16
policy_decision_refs=4
negative_nonzero={}
```

ConnectorHub proof report:

```text
reports/security/vs3-connectorhub-source-proof.json
status=success
schema_version=cs.vs3_connectorhub_source_proof.v0
scenario_status:
  VS3-CON-001=PASS
  VS3-CON-002=PASS
  VS3-CON-003=PASS
  VS3-CON-004=PASS
  VS3-CON-005=PASS
  VS3-CON-006=PASS
checks:
  no_live_provider_or_human_claim=true
  vs3_con_001_projection_ack_after_commit=true
  vs3_con_002_github_readonly_no_write_paths=true
  vs3_con_003_credentials_stay_connectorhub=true
  vs3_con_004_source_policy_scoped_revocable=true
  vs3_con_005_capture_fixture_consent_bounded_revocable=true
  vs3_con_006_faults_quarantine_no_side_effects=true
proof_boundary.surface=local_connectorhub_source_fixture
proof_boundary.live_provider=HUMAN_REQUIRED
proof_boundary.real_device_capture=HUMAN_REQUIRED
proof_boundary.real_provider_credentials=HUMAN_REQUIRED
proof_boundary.human_operator_acceptance=HUMAN_REQUIRED
proof_boundary.vs3_l=NOT_CLAIMED
proof_boundary.vs3_p=NOT_CLAIMED
```

Negative evidence:

```text
ack_before_commit_count=0
lost_projection_count=0
duplicate_truth_records=0
github_write_mappings=0
github_external_mutations=0
github_write_calls=0
raw_credentials_exposed=0
secret_scanner_findings=0
source_policy_cross_scope_deliveries=0
source_policy_stale_delivery_after_revoke=0
silent_capture_sessions=0
unbounded_capture_sessions=0
disallowed_raw_capture_outputs=0
capture_after_revoke=0
unauthorized_memory_writes=0
unauthorized_policy_changes=0
unauthorized_action_approvals=0
unauthorized_egress_calls=0
authority_expansions_from_connector_content=0
live_provider_claimed=0
vs3_p_claimed=0
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
proof_boundary.human_acceptance=NOT_CLAIMED
```

Aggregate ConnectorHub row evidence:

```text
VS3-CON-001 PASS evidence_refs=5 audit_refs=16 policy_decision_refs=4 source_report_refs=reports/security/vs3-connectorhub-source-proof.json
VS3-CON-002 PASS evidence_refs=5 audit_refs=16 policy_decision_refs=4 source_report_refs=reports/security/vs3-connectorhub-source-proof.json
VS3-CON-003 PASS evidence_refs=5 audit_refs=16 policy_decision_refs=4 source_report_refs=reports/security/vs3-connectorhub-source-proof.json
VS3-CON-004 PASS evidence_refs=4 audit_refs=16 policy_decision_refs=4 source_report_refs=reports/security/vs3-connectorhub-source-proof.json
VS3-CON-005 PASS evidence_refs=4 audit_refs=16 policy_decision_refs=4 source_report_refs=reports/security/vs3-connectorhub-source-proof.json
VS3-CON-006 PASS evidence_refs=5 audit_refs=16 policy_decision_refs=4 source_report_refs=reports/security/vs3-connectorhub-source-proof.json
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
| `VS3-CON-001` | `reports/security/vs3-connectorhub-source-proof.json`, `cornerstone connector projection verify --json`, `connector_ack_outbox:ack_vs3_con_readonly_commit_001`, `fixtures/connectorhub/contracts/github_readonly_contract.json`, `fixtures/connectorhub/deliveries/github_commit_projection_delivery.json` |
| `VS3-CON-002` | `reports/security/vs3-connectorhub-source-proof.json`, `cornerstone connector github-write-guard --json`, `cornerstone connector action dry-run --json`, `fixtures/connectorhub/contracts/github_readonly_contract.json`, `fixtures/connectorhub/deliveries/github_commit_projection_delivery.json` |
| `VS3-CON-003` | `reports/security/vs3-connectorhub-source-proof.json`, `cornerstone connector credential-boundary-test --json`, `cornerstone connector action dry-run --json`, `fixtures/connectorhub/contracts/github_readonly_contract.json`, `fixtures/connectorhub/deliveries/github_commit_projection_delivery.json` |
| `VS3-CON-004` | `reports/security/vs3-connectorhub-source-proof.json`, `cornerstone connector source-policy show --json`, `fixtures/connectorhub/contracts/github_readonly_contract.json`, `fixtures/connectorhub/deliveries/github_commit_projection_delivery.json` |
| `VS3-CON-005` | `reports/security/vs3-connectorhub-source-proof.json`, `cornerstone connector capture verify --profile vs3 --json`, `fixtures/connectorhub/contracts/github_readonly_contract.json`, `fixtures/connectorhub/deliveries/github_commit_projection_delivery.json` |
| `VS3-CON-006` | `reports/security/vs3-connectorhub-source-proof.json`, `cornerstone connector projection verify --json`, `connector_delivery_quarantine:cq_vs3_con_fault_prompt_injection`, `fixtures/connectorhub/contracts/github_readonly_contract.json`, `fixtures/connectorhub/deliveries/github_commit_projection_delivery.json` |

## Human Required

Still `HUMAN_REQUIRED` and not converted to PASS:

- Live provider credentials and approved live GitHub/provider rehearsal.
- Real credential custody review outside deterministic fixtures.
- Real macOS/Chrome physical-device WatchAgent capture review.
- Human operator UX/trust review for capture, denial, audit, connector, and SourcePolicy surfaces.
- Independent security review.
- VS3-P release approval.

## Deliberately Not Done

- No live GitHub or external provider account was called.
- No real provider credential was used.
- No real macOS/Chrome capture session was accepted as evidence.
- No production/on-prem, VS3-P, independent security review, or human acceptance claim was made.
- No external writeback or provider mutation path was enabled.

## Decision

`VS3-CON-001` through `VS3-CON-006` are locally AI-verifiable and PASS for the VS3-5 local/dev proof surface. Continue to the next slice only with the same proof-boundary separation.
