# VS3 Local Checkpoint Component Proof Identity Guard Checkpoint

**Date:** 2026-06-29 KST
**Status:** Local deterministic VS3-L checkpoint hardening slice.
**Scope:** `cornerstone security vs3-local-checkpoint --json` component proof identity guard.
**Verdict:** AI-verifiable slice PASS; VS3-P and all human/on-prem claims remain `NOT_CLAIMED` / `HUMAN_REQUIRED`.

## Slice Contract

Goal:

- Prevent `reports/human-gates/vs3/vs3-local-checkpoint.json` from passing when the aggregate VS3 scenario report embeds stale component proof bodies that no longer match the current component proof-report files.

In scope:

- Add deterministic canonical JSON identity checks for component proof reports embedded in `reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json`.
- Expose match results, hashes, failed conditions, and negative evidence in the native CLI JSON output.
- Add focused positive and stale-component negative tests.
- Regenerate local VS3-L evidence artifacts after the code change.

Out of scope:

- New production, on-prem, real IdP, real network, live-provider, migration/restore, independent security review, or human UX acceptance proof.
- New behavior implementation for RequestContext, RLS, OPA, egress, ConnectorHub, Tool SDK, registry, or operator UX beyond this identity guard.
- Treating generated human-gate templates, review kits, or structural validation as signed human evidence.

Done criteria:

- Local checkpoint fails closed if any guarded component proof file differs from the embedded component proof body inside the aggregate scenario report.
- Local checkpoint still passes when all guarded component proofs match and all existing VS3-L conditions remain true.
- `VS3-L` may be reported only as local/dev assurance; `VS3-P` remains `NOT_CLAIMED`.

## Full Scenario Mapping

| Scenario ID | Type | Slice classification | Required proof surface for this slice |
|---|---|---|---|
| VS3-GATE-001 | MUST_PASS | identity_guarded_existing_evidence | `evidence_reconciliation` body must match `reports/security/vs3-evidence-reconciliation.json`. |
| VS3-GATE-002 | MUST_PASS | artifact_hash_guarded_existing_evidence | Contract/matrix/goal prompt remain hash-backed in checkpoint manifest. |
| VS3-GATE-003 | MUST_PASS | in_this_slice | Aggregate report and component proofs must stay bound without overclaiming production or human acceptance. |
| VS3-GATE-004 | MUST_PASS | in_this_slice | Native `cornerstone security vs3-local-checkpoint --json` exposes component identity and failed conditions. |
| VS3-CTX-001 | MUST_PASS | identity_guarded_existing_evidence | `request_context_proof` body must match `reports/security/vs3-request-context-proof.json`. |
| VS3-CTX-002 | MUST_PASS | identity_guarded_existing_evidence | Same RequestContext component proof identity guard. |
| VS3-CTX-003 | MUST_PASS | identity_guarded_existing_evidence | Same RequestContext component proof identity guard. |
| VS3-CTX-004 | MUST_PASS | identity_guarded_existing_evidence | Same RequestContext component proof identity guard. |
| VS3-CTX-005 | MUST_PASS | identity_guarded_existing_evidence | Same RequestContext component proof identity guard. |
| VS3-RLS-001 | MUST_PASS | identity_guarded_existing_evidence | `postgres_rls_proof` body must match `reports/db/vs3-postgres-rls-proof.json`. |
| VS3-RLS-002 | MUST_PASS | identity_guarded_existing_evidence | Same Postgres/RLS component proof identity guard. |
| VS3-RLS-003 | MUST_PASS | identity_guarded_existing_evidence | Same Postgres/RLS component proof identity guard. |
| VS3-RLS-004 | MUST_PASS | identity_guarded_existing_evidence | Same Postgres/RLS component proof identity guard. |
| VS3-RLS-005 | MUST_PASS | identity_guarded_existing_evidence | Same Postgres/RLS component proof identity guard. |
| VS3-RLS-006 | MUST_PASS | identity_guarded_existing_evidence | Same Postgres/RLS component proof identity guard. |
| VS3-OPA-001 | MUST_PASS | identity_guarded_existing_evidence | `opa_policy_proof` body must match `reports/policy/vs3-opa-policy-proof.json`. |
| VS3-OPA-002 | MUST_PASS | identity_guarded_existing_evidence | Same OPA/Rego component proof identity guard. |
| VS3-OPA-003 | MUST_PASS | identity_guarded_existing_evidence | Same OPA/Rego component proof identity guard. |
| VS3-OPA-004 | MUST_PASS | identity_guarded_existing_evidence | Same OPA/Rego component proof identity guard. |
| VS3-OPA-005 | MUST_PASS | identity_guarded_existing_evidence | Same OPA/Rego component proof identity guard. |
| VS3-EGR-001 | MUST_PASS | identity_guarded_existing_evidence | `egress_sandbox_proof` body must match `reports/security/vs3-egress-sandbox-proof.json`. |
| VS3-EGR-002 | MUST_PASS | identity_guarded_existing_evidence | Same egress/sandbox component proof identity guard. |
| VS3-EGR-003 | MUST_PASS | identity_guarded_existing_evidence | Same egress/sandbox component proof identity guard. |
| VS3-EGR-004 | MUST_PASS | identity_guarded_existing_evidence | Same egress/sandbox component proof identity guard. |
| VS3-EGR-005 | MUST_PASS | identity_guarded_existing_evidence | Same egress/sandbox component proof identity guard. |
| VS3-EGR-006 | MUST_PASS | identity_guarded_existing_evidence | Same egress/sandbox component proof identity guard. |
| VS3-CON-001 | MUST_PASS | identity_guarded_existing_evidence | `connectorhub_source_proof` body must match `reports/security/vs3-connectorhub-source-proof.json`. |
| VS3-CON-002 | MUST_PASS | identity_guarded_existing_evidence | Same ConnectorHub/source component proof identity guard. |
| VS3-CON-003 | MUST_PASS | identity_guarded_existing_evidence | Same ConnectorHub/source component proof identity guard. |
| VS3-CON-004 | MUST_PASS | identity_guarded_existing_evidence | Same ConnectorHub/source component proof identity guard. |
| VS3-CON-005 | MUST_PASS | identity_guarded_existing_evidence | Same ConnectorHub/source component proof identity guard; physical-device/live capture remains human where applicable. |
| VS3-CON-006 | MUST_PASS | identity_guarded_existing_evidence | Same ConnectorHub/source component proof identity guard. |
| VS3-TOOL-001 | MUST_PASS | identity_guarded_existing_evidence | `tool_registry_proof` body must match `reports/security/vs3-tool-registry-proof.json`. |
| VS3-TOOL-002 | MUST_PASS | identity_guarded_existing_evidence | Same tool/registry component proof identity guard. |
| VS3-TOOL-003 | MUST_PASS | identity_guarded_existing_evidence | Same tool/registry component proof identity guard. |
| VS3-TOOL-004 | MUST_PASS | identity_guarded_existing_evidence | Same tool/registry component proof identity guard. |
| VS3-TOOL-005 | MUST_PASS | identity_guarded_existing_evidence | Same tool/registry component proof identity guard. |
| VS3-TOOL-006 | MUST_PASS | identity_guarded_existing_evidence | Same tool/registry component proof identity guard. |
| VS3-TOOL-007 | MUST_PASS | identity_guarded_existing_evidence | Same tool/registry component proof identity guard. |
| VS3-OBS-001 | MUST_PASS | identity_guarded_existing_evidence | `observability_proof` body must match `reports/observability/vs3-observability-proof.json`. |
| VS3-OBS-002 | MUST_PASS | identity_guarded_existing_evidence | Same observability/audit component proof identity guard. |
| VS3-OBS-003 | MUST_PASS | identity_guarded_existing_evidence | Same observability/audit component proof identity guard. |
| VS3-REG-001 | REGRESSION | identity_guarded_existing_evidence | `final_regression_proof` body must match `reports/security/vs3-final-regression-proof.json`. |
| VS3-REG-002 | REGRESSION | identity_guarded_existing_evidence | Same final regression component proof identity guard. |
| VS3-REG-003 | REGRESSION | identity_guarded_existing_evidence | Same final regression component proof identity guard. |
| VS3-REG-004 | REGRESSION | in_this_slice | Coverage/evidence cannot silently drift between aggregate report and component proof files. |
| VS3-REG-005 | REGRESSION | in_this_slice | Report/evidence overclaim guard remains active while component proof identity is enforced. |
| VS3-REG-006 | REGRESSION | identity_guarded_existing_evidence | Same final regression component proof identity guard; no new UI/human UX claim. |
| VS3-REG-007 | REGRESSION | identity_guarded_existing_evidence | Same final regression component proof identity guard; no new dependency approval claim. |
| VS3-REG-008 | REGRESSION | identity_guarded_existing_evidence | Same final regression component proof identity guard. |
| VS3-H01 | HUMAN_REQUIRED | human_required | Dated signed architecture/security/dependency/migration approval remains required before VS3-P. |
| VS3-H02 | HUMAN_REQUIRED | human_required | Independent security review and retest remains required before VS3-P. |
| VS3-H03 | HUMAN_REQUIRED | human_required | Real IdP mapping/revocation evidence remains required before real identity readiness. |
| VS3-H04 | HUMAN_REQUIRED | human_required | Real on-prem network/firewall/proxy/service-mesh evidence remains required before VS3-P. |
| VS3-H05 | HUMAN_REQUIRED | human_required | Approved live-provider rehearsal remains required before live readiness. |
| VS3-H06 | HUMAN_REQUIRED | human_required | Human operator UX/trust acceptance or rejection remains required before human UX acceptance. |
| VS3-H07 | HUMAN_REQUIRED | human_required | Signed migration/backup/restore drill remains required before migration/restore readiness. |

## Implementation Summary

Changed:

- `packages/cornerstone_cli/main.py`
  - Added canonical JSON identity hashing for guarded component proof bodies.
  - Added `component_proof_identity` to `cornerstone security vs3-local-checkpoint --json`.
  - Added per-component checkpoint conditions named `component_proof_<key>_matches_current_file`.
  - Added `component_proof_report_mismatches` and `component_proof_report_missing_or_invalid` negative evidence counters.
- `tests/scenario/test_scaffold_cli.py`
  - Positive checkpoint test now asserts 9 guarded component proofs match.
  - Added stale component proof negative test that mutates `reports/security/vs3-request-context-proof.json` after aggregate report generation and verifies fail-closed behavior.

Guarded component report keys:

- `evidence_reconciliation`
- `request_context_proof`
- `postgres_rls_proof`
- `opa_policy_proof`
- `egress_sandbox_proof`
- `connectorhub_source_proof`
- `tool_registry_proof`
- `observability_proof`
- `final_regression_proof`

`overclaim_lint` is intentionally not part of this identity guard because the aggregate report embeds the lint result used during aggregation, while `reports/security/vs3-overclaim-lint.json` is a standalone CLI artifact that includes additional CLI-only fields.

## Verification Evidence

Commands run:

```text
python3 -m compileall packages/cornerstone_cli
```

Result:

```text
Listing 'packages/cornerstone_cli'...
Compiling 'packages/cornerstone_cli/main.py'...
```

```text
python3 -m unittest \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_manifest_is_hash_backed_and_boundary_safe \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_stale_component_proof_file \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_stale_review_kit_scenario_report_hash \
  tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs3_local_checkpoint_rejects_review_kit_from_different_scenario_report_path
```

Result:

```text
Ran 4 tests in 105.722s
OK
```

Regenerated local evidence:

```text
cornerstone scenario verify vs3-onprem-trusted-extension --json --output reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json
cornerstone human-gate record-scaffold --scope vs3 --output-dir reports/human-gates/vs3/record-templates --force --use-existing --json --output reports/human-gates/vs3/record-scaffold.json
cornerstone human-gate evidence-status --scope vs3 --record-dir reports/human-gates/vs3/record-templates --use-existing --json --output reports/human-gates/vs3/evidence-status.json
cornerstone human-gate review-kit --scope vs3 --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --use-existing --json --output reports/human-gates/vs3/review-kit.json
cornerstone human-gate vs3-p-gate --scope vs3 --scenario-report reports/scenario/vs3-onprem-trusted-extension-2026-06-29.json --output reports/human-gates/vs3/vs3-p-gate.json --json
cornerstone security vs3-local-checkpoint --json --output reports/human-gates/vs3/vs3-local-checkpoint.json
```

Expected closed VS3-P gate:

```text
cornerstone human-gate vs3-p-gate ... exited 4
status=blocked
final_verdict=HUMAN_REQUIRED
unresolved_human_required_rows=7
vs3_p_claim=NOT_CLAIMED
```

Local checkpoint summary:

```json
{
  "status": "success",
  "final_verdict": "VS3_L_LOCAL_DEV_ASSURANCE_VERIFIED_VS3_P_HUMAN_REQUIRED",
  "summary": {
    "scenario_count": 57,
    "pass": 50,
    "human_required": 7,
    "blocking": 0,
    "component_proof_report_count": 9,
    "component_proof_report_mismatches": 0,
    "vs3_l_claim": "LOCAL_DEV_ASSURANCE_VERIFIED",
    "vs3_p_claim": "NOT_CLAIMED"
  },
  "negative_component": {
    "component_proof_report_mismatches": 0,
    "component_proof_report_missing_or_invalid": 0
  },
  "failed_conditions": []
}
```

Component proof identity sample:

```json
{
  "key": "request_context_proof",
  "path": "reports/security/vs3-request-context-proof.json",
  "matches": true,
  "error": null
}
```

Whitespace check:

```text
git diff --check
```

Result: exit 0, no output.

Documentation verifier:

```text
scripts/verify_sot_docs.sh
```

Result:

```text
PASS: CornerStone CLI native-first docs verified (39 feature-family rows; all CLI-required and release-blocking).
PASS: CornerStone local verification plane docs verified (20 numbered sections; deterministic PASS gate documented).
PASS: design tokens verified (11 state tokens, 8 color groups).
PASS: CornerStone design system docs verified.
PASS: CornerStone VS-0 scaffold readiness docs verified.
PASS: CornerStone SoT docs verified (206 full scenarios, design system, VS-0 scaffold readiness, VS-0 scaffold gate, 58 VS-0 scenarios, CLI native-first gate, local verification plane).
```

## Pass / Fail Criteria

PASS for this slice:

- `component_proof_report_count == 9`.
- `component_proof_report_mismatches == 0`.
- `component_proof_report_missing_or_invalid == 0`.
- Every `component_proof_<key>_matches_current_file` checkpoint condition is `true`.
- Stale component proof mutation returns exit 4, `VS3_L_LOCAL_DEV_ASSURANCE_NOT_VERIFIED`, `vs3_l=NOT_CLAIMED`, and the exact failed condition.
- No production, VS3-P, live-provider, real-IdP, real-network, migration/restore, security-acceptance, or human-acceptance claim is made.

FAIL for this slice:

- Any guarded component proof file is missing, invalid JSON, not an object, or has a canonical JSON identity that differs from the body embedded in the aggregate scenario report.
- The checkpoint keeps `VS3-L` claimed after a component proof mismatch.
- Any human-required row is marked `PASS` by this guard.
- Any local checkpoint language claims VS3-P, production/on-prem readiness, live-provider readiness, real IdP readiness, real network readiness, migration/restore readiness, security acceptance, or human acceptance.

## Remaining Human Gates

Still `HUMAN_REQUIRED`:

- `VS3-H01`: architecture/security/dependency/migration approval.
- `VS3-H02`: independent security review and retest.
- `VS3-H03`: real IdP mapping and revocation.
- `VS3-H04`: real on-prem network controls.
- `VS3-H05`: live ConnectorHub/provider rehearsal.
- `VS3-H06`: operator UX/trust review.
- `VS3-H07`: migration/backup/restore drill.

This checkpoint prepares and validates local evidence only. It does not promote any human-required row.
