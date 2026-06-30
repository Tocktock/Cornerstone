# VS3 OPA/Rego Policy Checkpoint - 2026-06-29

**Status:** VS3-3 local OPA/Rego control-plane slice PASS.
**Scope:** `VS3-OPA-001` through `VS3-OPA-005` only.
**Proof boundary:** Local/dev deterministic policy schema, Rego fixture, local policy-gateway access, bundle lifecycle, cache/fail-closed behavior, decision-log masking, native CLI output.

This checkpoint does not claim VS3-P, production/on-prem readiness, real IdP readiness, live-provider readiness, migration/restore readiness, independent security review, or human security acceptance.

## Slice Contract

Goal:

- Harden and verify the local VS3 OPA/Rego control-plane proof path.
- Ensure native `cornerstone ... --json` policy commands expose schema versions, evidence refs, audit refs, policy decision refs, and proof-boundary fields on allowed and denied paths.

Selected scenarios:

| Scenario | Status in this checkpoint | Required proof surface |
|---|---|---|
| `VS3-OPA-001` | PASS | Policy input schema, valid/invalid fixtures, source-of-attribute map, input digest. |
| `VS3-OPA-002` | PASS | OPA/Rego decision fixtures, deterministic PolicyDecision fields, reason codes, audit refs. |
| `VS3-OPA-003` | PASS | Local policy-gateway allowed/denied access fixture and unauthorized API denial evidence. |
| `VS3-OPA-004` | PASS | Bundle dry-run lifecycle, invalid bundle rejection, rollback/fail-closed decision evidence. |
| `VS3-OPA-005` | PASS | Decision-log masking, audit mirror, zero raw secret canary leaks. |

Full VS3 mapping remains the frozen 57-row inventory: 42 `MUST_PASS`, 8 `REGRESSION`, and 7 `HUMAN_REQUIRED`. Non-OPA rows are outside this checkpoint except where the aggregate scenario report is cited as supporting local scenario-gate context.

## Implementation Delta

- `packages/cornerstone_cli/main.py` now makes non-dry-run `cornerstone policy bundle activate --json` denial use the VS3 OPA base payload instead of a bare CLI failure payload.
- The denied path now includes:
  - `policy_bundle_activate_schema_version: cs.vs3_policy_bundle_activate.v0`
  - `dry_run: false`
  - `bundle_activation.status: blocked_non_dry_run`
  - `production_activation: NOT_CLAIMED`
  - `human_security_acceptance: HUMAN_REQUIRED`
  - local proof boundary, evidence refs, audit refs, and policy decision refs.
- `tests/scenario/test_scaffold_cli.py` now asserts the non-dry-run denial contract and exit code `6`.

## Command Evidence

Focused compile:

```text
python3 -m compileall packages/cornerstone_cli/main.py tests/scenario/test_scaffold_cli.py
Compiling 'packages/cornerstone_cli/main.py'...
exit=0
```

Focused tests:

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_opa_policy_proof_is_local_and_negative_evidence_backed \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_policy_cli_paths_are_native \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_onprem_trusted_extension_verify_closes_local_dev_ai_rows

Ran 3 tests in 31.401s
OK
```

Direct CLI probes:

```text
cornerstone policy bundle activate --json
exit=6
status=failed
schema_version=cs.cli.v0
policy_bundle_activate_schema_version=cs.vs3_policy_bundle_activate.v0
dry_run=false
bundle_activation.status=blocked_non_dry_run
bundle_activation.production_activation=NOT_CLAIMED
bundle_activation.human_security_acceptance=HUMAN_REQUIRED
proof_boundary.vs3_p=NOT_CLAIMED
evidence_refs=1
audit_refs=21
policy_decision_refs=19
error_code=CS_VS3_POLICY_BUNDLE_DRY_RUN_REQUIRED
```

```text
cornerstone policy bundle activate --dry-run --json
exit=0
status=success
schema_version=cs.cli.v0
policy_bundle_activate_schema_version=cs.vs3_policy_bundle_activate.v0
dry_run=true
bundle_activation.status=dry_run_passed
bundle_activation.invalid_bundle_activated=false
bundle_activation.production_activation=NOT_CLAIMED
proof_boundary.vs3_p=NOT_CLAIMED
evidence_refs=1
audit_refs=22
policy_decision_refs=20
```

```text
cornerstone policy evaluate --input fixtures/vs3/policy/allow_artifact_read.json --json
exit=0
status=allowed
schema_version=cs.cli.v0
policy_evaluate_schema_version=cs.vs3_policy_evaluate.v0
policy_decision.decision=allow
proof_boundary.vs3_p=NOT_CLAIMED
evidence_refs=2
audit_refs=22
policy_decision_refs=20
```

OPA proof report:

```text
reports/policy/vs3-opa-policy-proof.json
status=success
schema_version=cs.vs3_opa_policy_proof.v0
scenario_status:
  VS3-OPA-001=PASS
  VS3-OPA-002=PASS
  VS3-OPA-003=PASS
  VS3-OPA-004=PASS
  VS3-OPA-005=PASS
checks:
  vs3_opa_001_policy_input_schema_and_source_map=true
  vs3_opa_002_policy_decision_contract=true
  vs3_opa_003_opa_http_access_hardened=true
  vs3_opa_004_bundle_lifecycle_fail_closed=true
  vs3_opa_005_decision_log_masked=true
audit_refs=21
policy_decision_refs=19
```

Negative evidence:

```text
caller_authoritative_policy_fields_accepted=0
unknown_policy_implicit_allows=0
anonymous_management_api_allows=0
invalid_bundle_activated=0
raw_secret_canary_leaks=0
production_opa_claimed=0
vs3_l_claimed=0
vs3_p_claimed=0
```

Aggregate scenario verification:

```text
cornerstone scenario verify vs3-onprem-trusted-extension --json \
  --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
exit=0
schema_version=cs.vs3_onprem_trusted_extension.v0
scenario_result_count=57
status_counts: PASS=50, HUMAN_REQUIRED=7
type_counts: MUST_PASS=42, REGRESSION=8, HUMAN_REQUIRED=7
claim_boundaries.vs3_p=NOT_CLAIMED
claim_boundaries.production_onprem=NOT_CLAIMED
claim_boundaries.security_acceptance=NOT_CLAIMED
```

Aggregate scenario gate:

```text
cornerstone scenario gate reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --json
exit=0
status=success
schema_version=cs.cli.v0
errors=[]
warnings=[]
claim_boundaries.vs3_p=NOT_CLAIMED
```

## Scenario Evidence Mapping

| Scenario | Evidence refs |
|---|---|
| `VS3-OPA-001` | `reports/policy/vs3-opa-policy-proof.json`, `config/vs3/policy_input_schema.v0.json`, `config/vs3/reason_code_catalog.v0.json`, `policies/vs3/policy.rego`, `policies/vs3/policy_test.rego`, `policies/vs3/system_log_mask.rego`, `fixtures/vs3/policy/allow_artifact_read.json`, `cornerstone security vs3-opa-policy --json` |
| `VS3-OPA-002` | `reports/policy/vs3-opa-policy-proof.json`, `policies/vs3/policy.rego`, `policies/vs3/policy_test.rego`, `cornerstone policy evaluate --input fixtures/vs3/policy/allow_artifact_read.json --json` |
| `VS3-OPA-003` | `reports/policy/vs3-opa-policy-proof.json`, local policy-gateway HTTP fixture, `cornerstone policy evaluate --input fixtures/vs3/policy/allow_artifact_read.json --json` |
| `VS3-OPA-004` | `reports/policy/vs3-opa-policy-proof.json`, `policies/vs3/policy.rego`, `policies/vs3/policy_test.rego`, `cornerstone policy bundle activate --dry-run --json`, denied non-dry-run activation transcript |
| `VS3-OPA-005` | `reports/policy/vs3-opa-policy-proof.json`, `policies/vs3/system_log_mask.rego`, decision-log masking evidence |

## Human Required

Still `HUMAN_REQUIRED` and not converted to PASS:

- Independent security review.
- Human security acceptance.
- Production/on-prem OPA deployment review.
- Real IdP/on-prem topology validation.
- VS3-P release approval.

## Deliberately Not Done

- No production OPA service was deployed.
- No real IdP, live provider, real network, or on-prem topology was exercised.
- No independent penetration test or security acceptance was claimed.
- No non-dry-run policy bundle activation was allowed.

## Decision

`VS3-OPA-001` through `VS3-OPA-005` are locally AI-verifiable and PASS for the VS3-3 local/dev proof surface. Continue to the next slice only with the same proof-boundary separation.
