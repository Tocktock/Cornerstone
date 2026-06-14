# VS0 Operator Acceptance UI Gate Report - 2026-06-14

**Owner:** JiYong / Tars  
**Scope:** Local VS0 operator UI gate only.  
**Contract:** `docs/scenario-contracts/VS0_OPERATOR_ACCEPTANCE_UI_GATE_CONTRACT.md`  
**Scenario report:** `reports/scenario/vs0-operator-acceptance-ui-2026-06-14.json`  
**Browser proof:** `reports/browser/vs0-operator-acceptance-ui-2026-06-14/`  
**Status:** AI-verifiable rows PASS; human operator acceptance remains `HUMAN_REQUIRED`.

## Goal

Close the VS0 human-review blocker by changing the local UI from one opaque proof button into an operator-readable flow:

```text
Select/upload Artifact
-> Search
-> Review Evidence
-> Create Claim
-> Review Action Card
-> Dry-run
-> Approve
-> Execute local/mock action
-> Inspect Audit
```

This report does not claim production release, live-provider readiness, or human acceptance.

## Implementation Summary

- Added a step-by-step local VS0 operator flow to the generated runtime UI.
- Kept the existing assisted proof path, but made it drive the same visible operator steps.
- Added visible Artifact, Search, Evidence, Claim, Action Card, Dry-run, Approval, Execution, and Audit state fields.
- Added deterministic browser-proof markers for every `VS0-UI-*` AI-verifiable row.
- Added `cornerstone scenario verify vs0-operator-acceptance-ui --json` and `make verify-vs0-operator-ui`.
- Preserved local/mock action safety: `mock_connector_calls=1`, `real_external_http_calls=0`.

## Scenario Table

| ID | Type | Status | Evidence |
|---|---|---|---|
| VS0-UI-001 | MUST_PASS | PASS | Browser proof shows `step_by_step_flow=true` and visible operator controls. |
| VS0-UI-002 | MUST_PASS | PASS | Browser proof shows artifact ID, checksum, source, derived status, evidence refs, and audit refs. |
| VS0-UI-003 | MUST_PASS | PASS | Browser proof shows query, snippet, search snapshot ID, and evidence eligibility. |
| VS0-UI-004 | MUST_PASS | PASS | Browser proof shows evidence support and insufficient-evidence guidance. |
| VS0-UI-005 | MUST_PASS | PASS | Browser proof shows draft, evidence-backed, and approved Claim states. |
| VS0-UI-006 | MUST_PASS | PASS | Browser proof shows zero-evidence denial cause `CS_CLAIM_EVIDENCE_REQUIRED` and resolution guidance. |
| VS0-UI-007 | MUST_PASS | PASS | Browser proof shows Action Card diff, expected impact, evidence bundle, policy, risk, approval, mock/local boundary, and rollback note. |
| VS0-UI-008 | MUST_PASS | PASS | Browser proof shows successful mock execution with `mock_connector_calls=1` and `real_external_http_calls=0`. |
| VS0-UI-009 | MUST_PASS | PASS | Browser proof shows audit event timeline and audit verification status. |
| VS0-UI-010 | MUST_PASS | PASS | Browser proof and scenario report show local-only proof with production release false, live connector false, and no human acceptance claim. |
| VS0-UI-R01 | REGRESSION_GUARD | PASS | Scenario report embeds EVUX governance summary: 16 rows, 14 PASS, 2 HUMAN_REQUIRED, 0 blocking. |
| VS0-UI-R02 | REGRESSION_GUARD | PASS | Browser proof status requires no timeout before PASS. Timeout remains non-clean PASS. |
| VS0-UI-H01 | HUMAN_REQUIRED | HUMAN_REQUIRED | JiYong/Tars must complete the UI walkthrough and record accept/reject evidence. |

## Evidence Artifacts

| Artifact | Purpose |
|---|---|
| `reports/scenario/vs0-operator-acceptance-ui-2026-06-14.json` | Machine-readable scenario result: 13 rows, 12 PASS, 1 HUMAN_REQUIRED, 0 blocking. |
| `reports/browser/vs0-operator-acceptance-ui-2026-06-14/browser-proof.json` | Browser proof summary with 13/13 operator markers true. |
| `reports/browser/vs0-operator-acceptance-ui-2026-06-14/workflow.dom.html` | DOM snapshot after the guided operator flow completed. |
| `reports/browser/vs0-operator-acceptance-ui-2026-06-14/workflow.png` | Browser screenshot after the guided operator flow completed. |
| `reports/browser/vs0-operator-acceptance-ui-2026-06-14/workflow-trace.json` | API/UI workflow trace including artifact, search, evidence, claim, action, approval, execution, and audit payloads. |

## Command Evidence

```sh
make verify-vs0-operator-ui
```

Result summary:

```json
{
  "exit_code": 0,
  "operator_report_status": "success",
  "scenario_count": 13,
  "pass": 12,
  "human_required": 1,
  "blocking": 0,
  "browser_proof_status": "PASS",
  "operator_markers_true": "13/13"
}
```

```sh
PATH="$PWD:$PATH" cornerstone scenario verify vs0-operator-acceptance-ui --json --output reports/scenario/vs0-operator-acceptance-ui-2026-06-14.json
```

Result summary:

```json
{
  "status": "success",
  "summary": {
    "scenario_count": 13,
    "pass": 12,
    "human_required": 1,
    "blocking": 0,
    "product_feature_claims": "LOCAL_VS0_OPERATOR_UI_READY_HUMAN_REQUIRED_PRODUCTION_NOT_READY"
  },
  "browser_proof_status": "PASS",
  "operator_markers_true": "13/13"
}
```

```sh
PATH="$PWD:$PATH" cornerstone scenario gate reports/scenario/vs0-operator-acceptance-ui-2026-06-14.json --json
```

Result summary:

```json
{
  "status": "success",
  "scenario_count": 13,
  "blocking_count": 0
}
```

```sh
python3 -m py_compile packages/cornerstone_cli/product_runtime.py packages/cornerstone_cli/acceptance.py packages/cornerstone_cli/scenarios.py packages/cornerstone_cli/main.py
```

Result: exit code 0.

```sh
make verify-vs0-evux
```

Result summary:

```json
{
  "exit_code": 0,
  "evux_summary": {
    "scenario_count": 14,
    "pass": 12,
    "human_required": 2,
    "blocking": 0
  },
  "quickstart_status": "success",
  "release_evidence_status": "success",
  "real_external_http_calls": 0
}
```

```sh
PATH="$PWD:$PATH" cornerstone scenario verify vs0-evux-governance --json --output reports/scenario/vs0-evux-governance-2026-06-14.json
```

Result summary:

```json
{
  "status": "success",
  "scenario_count": 16,
  "pass": 14,
  "human_required": 2,
  "blocking": 0
}
```

```sh
scripts/verify_sot_docs.sh
python3 scripts/verify_scenario_matrix.py
git diff --check
```

Result: all exit code 0. SoT docs verified 206 full scenarios, scenario matrix verified 206 scenarios with no unevidenced PASS claims, and diff whitespace check was clean.

## In-App Browser Check

A fresh local runtime was opened at `http://127.0.0.1:8790/?scenario=vs0-evux` in the in-app browser.

Observed visible state after the guided proof completed:

```json
{
  "status": "passed",
  "currentStep": "9. Inspect Audit",
  "artifactId": "art_2735313e0cb92563",
  "searchSnapshotId": "search_fed8ef4d6ce69f5a",
  "evidenceBundleId": "evb_2aadd2495950b62c",
  "claimId": "claim_255c9ec2b44162d7",
  "actionId": "action_964dfe8af66668c1",
  "zeroEvidence": "CS_CLAIM_EVIDENCE_REQUIRED: Claim approval requires evidence.",
  "actionBoundary": "ConnectorHub mediated; mock_connector; direct_provider_access=false; credentials_exposed=false",
  "mockCalls": "mock_connector_calls=1",
  "realCalls": "real_external_http_calls=0",
  "auditVerification": "success; events=15"
}
```

## Safety And Boundary Checks

- No live provider credentials were used.
- No real external writeback occurred.
- Runtime proof keeps `real_external_http_calls=0`.
- Action execution remains local/mock only.
- Zero-evidence Claim approval remains denied before evidence is attached.
- UI text and scenario report explicitly avoid production release, live connector readiness, and human acceptance claims.

## Human Required

`VS0-UI-H01` remains unresolved until JiYong/Tars uses the local UI and records one of:

- acceptance note with screenshot or recording; or
- rejection note with issue list.

Full VS-1 main implementation remains blocked until that human evidence exists.

## Verdict

```text
AI-verifiable VS0 operator UI gate: PASS
Human operator UX acceptance: HUMAN_REQUIRED
Production release readiness: NOT CLAIMED
Live-provider readiness: NOT CLAIMED
Full VS-1 main implementation: WAIT
```
