# CornerStone VS0 Operator Acceptance UI Gate Contract

**Date:** 2026-06-14
**Owner:** JiYong / Tars
**Status:** Frozen task-scoped scenario contract. This is a documentation-only freeze; current implementation status belongs in scenario reports, verification reports, browser proof, and the human review record.

## Feature / Task

`VS0_OPERATOR_ACCEPTANCE_UI_GATE`

## Goal

Turn the existing AI-verifiable VS0 EVUX loop from a one-click scenario proof into a human-understandable operator flow before full VS-1 implementation starts.

The UI must make the local VS0 workflow understandable and controllable:

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

## Decision

Full VS-1 implementation waits until this gate is accepted by JiYong/Tars.

The current VS0 EVUX governance evidence can remain AI-verifiable PASS, but the product must not claim human operator UX acceptance while the UI exposes the workflow as one opaque "Run local VS0 loop" action.

VS-1 scenario planning and backend preparation may continue only when it does not claim VS-1 milestone progress and does not add ontology complexity to the current VS0 operator UI before this gate is accepted.

## Purpose

This is a narrow VS0 UI acceptance slice, not a feature expansion, not a production UI redesign, and not VS-1 ontology implementation.

The operator must be able to answer:

```text
Where am I?
What happened?
What evidence supports this?
What is safe or unsafe?
What will the action do?
Is this real external writeback or mock/local only?
What was recorded in audit?
What is not production-ready?
```

## Relationship To Existing Scenario Authority

This contract is subordinate to:

- the product SoT;
- the full 206-scenario MUST-PASS standard;
- the CLI-native-first contract;
- the local verification plane;
- the design-system contract;
- the VS0 EVUX clean sign-off governance contract.

It does not replace `VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP` or `VS0_EVUX_CLEAN_SIGNOFF_GOVERNANCE`. It closes the human-operator usability gap left intentionally as `HUMAN_REQUIRED`.

## Success Criteria

The future implementation may claim `VS0_OPERATOR_ACCEPTANCE_UI_GATE` only when:

- every AI-verifiable `VS0-UI-*` MUST_PASS and REGRESSION_GUARD row is `PASS` with concrete evidence;
- the UI exposes the VS0 loop as clear operator steps, not only as an automated run button;
- artifact, search, evidence, claim, action, execution, and audit states are inspectable in the UI;
- zero-evidence Claim approval is denied with cause and resolution guidance;
- action execution evidence shows `mock_connector_calls=1` and `real_external_http_calls=0`;
- existing EVUX governance remains PASS;
- browser timeout cannot be marked clean PASS;
- the UI clearly says local VS0 proof only and does not claim production release, live connector readiness, or human acceptance;
- `VS0-UI-H01` is recorded by JiYong/Tars as accepted, with screenshots, recording, or issue list.

## Constraints

### Product And UX

- Keep the change narrow.
- Do not overbuild polished final product UI.
- Do not introduce ontology UI before the VS0 Artifact/Search/Evidence/Claim/Action/Audit loop is understandable.
- Preserve CornerStone as one product experience.
- Preserve the design-system direction: calm workspace, evidence-aware, safe-action UI.

### Data And State

- Local file-backed runtime remains acceptable for this gate.
- Do not migrate to Postgres, RLS, OPA, or production storage in this task.
- Do not change tenant/security boundaries without explicit approval.
- Preserve Artifact/Evidence/Audit/Action safety semantics.

### Permission And Safety

- No live provider credentials.
- No real external writeback.
- `real_external_http_calls` must remain `0`.
- No destructive action, production deployment, release tag, or irreversible migration.
- No secrets in screenshots, logs, reports, generated docs, or committed evidence.

### Compatibility And Format

- No UI-only feature PASS: every product operation exposed through UI must still have a native `cornerstone ...` CLI/API-backed path or be explicitly classified as non-feature presentation.
- Preserve `--json` machine-readable verification output.
- Keep scenario contracts status-neutral; implementation status belongs in reports and matrices.

### Operational And Environment

- Local browser proof is required for AI-verifiable UI rows.
- If browser proof times out or cannot complete, the affected row is `PARTIAL`, `FAIL`, or `NOT_VERIFIED`, not clean `PASS`.
- No new production dependency without explicit approval.

## Assumptions

- The existing EVUX behavior and governance proof are useful and should be preserved.
- The current human review found the UI understandable only partially because it is still one-click proof instead of step-by-step operator control.
- This gate is the smallest safe bridge from automated EVUX evidence to human operator acceptance.
- VS-1 ontology concepts will make UX debt worse if added before this VS0 flow is accepted.

## Out Of Scope

- Full VS-1 implementation.
- Ontology auto-suggest, promote, object explorer, or ontology UX claims.
- Live ConnectorHub/provider execution.
- Production release readiness.
- Production tenancy, authz, retention, or storage migration.
- Broad redesign or visual polish beyond what is necessary for operator comprehension.
- Claiming human acceptance from browser automation alone.

## Allowed Parallel Work

Allowed before this gate is accepted:

- freeze VS-1 ontology scenario contract;
- design data model for ontology suggestions;
- add backend-only draft suggestion fixtures if they do not require UI acceptance or release claims;
- identify reusable UI components needed by both VS0 and VS1.

Not allowed before this gate is accepted:

- claim VS-1 started as the main product milestone;
- claim ontology UX is usable;
- add ontology complexity to the current UI before the VS0 operator flow is accepted;
- use VS-1 work to bypass this human operator acceptance gate.

## CLI Parity

- Command group: `cornerstone scenario|artifact|search|evidence|claim|mission|action|audit|release`
- Future scenario verification command: `cornerstone scenario verify vs0-operator-acceptance-ui --json --output reports/scenario/vs0-operator-acceptance-ui-YYYY-MM-DD.json`
- Future scenario gate command: `cornerstone scenario gate reports/scenario/vs0-operator-acceptance-ui-YYYY-MM-DD.json --json`
- Regression governance command: `cornerstone scenario verify vs0-evux-governance --json --output reports/scenario/vs0-evux-governance-YYYY-MM-DD.json`
- Regression make target: `make verify-vs0-evux`
- JSON schema path: scenario report, browser proof, UI walkthrough transcript, action result, audit verification, and human review evidence are implementation-owned and must be referenced from the final verification report.
- Exit codes covered: success, verification gap, browser timeout/partial proof, policy/safety denial, missing artifact, and runtime failure.
- Workspace/namespace scope: local VS0 fixture workspace only unless the implementation report explicitly documents another local namespace.
- Dry-run behavior: external or risky action remains local/mock only; real writeback remains out of scope.
- Evidence refs emitted: artifact ID, checksum, search snapshot ID, evidence bundle ID, claim ID/state, action ID/state, policy decision, execution result, audit refs, browser proof, and scenario report.
- Audit refs emitted: artifact/search/evidence/claim/action/approval/execution/audit events from the same VS0 loop.
- Same backend path evidence: future implementation must show UI, API, and CLI paths use the same Artifact, Evidence, Claim, Action, Policy, and Audit boundaries.

## Scenario Table

Total task-scoped scenarios: **13**.

| ID | Type | Expected Result | Verification Method | Evidence Required | Owner |
|---|---|---|---|---|---|
| VS0-UI-001 | MUST_PASS | UI shows a clear step-by-step VS0 flow, not one opaque run-loop button. | Browser walkthrough plus DOM/state inspection. | Browser proof showing distinct Artifact, Search, Evidence, Claim, Action, Execution, and Audit steps. | AI |
| VS0-UI-002 | MUST_PASS | Artifact step shows artifact ID, checksum, source, derived status, evidence refs, and audit refs. | Browser interaction plus artifact/API inspection. | Screenshot/DOM snapshot, artifact JSON, evidence refs, audit refs. | AI |
| VS0-UI-003 | MUST_PASS | Search step shows query, result snippet, search snapshot ID, and evidence eligibility. | Browser interaction plus search snapshot inspection. | Screenshot/DOM snapshot, search snapshot JSON, eligibility marker, audit refs. | AI |
| VS0-UI-004 | MUST_PASS | Evidence step shows what supports the Claim and what would be insufficient. | Browser interaction plus evidence bundle inspection. | Evidence bundle JSON, UI state showing included support and insufficient-evidence guidance. | AI |
| VS0-UI-005 | MUST_PASS | Claim step shows Draft, Evidence-backed, and Approved state clearly. | Browser/API claim-state inspection. | Claim JSON plus UI proof for each reachable state in the local flow. | AI |
| VS0-UI-006 | MUST_PASS | Zero-evidence Claim approval is denied with cause and resolution guide. | Browser/API negative test. | Denial response, unchanged claim state, UI cause/resolution text, audit ref. | AI |
| VS0-UI-007 | MUST_PASS | Action Card shows diff, expected impact, evidence, policy decision, risk, approval state, mock/local boundary, and rollback/compensation note. | Browser interaction plus action/policy inspection. | Action Card screenshot/DOM snapshot, action JSON, policy decision, audit refs. | AI |
| VS0-UI-008 | MUST_PASS | Execution step shows `mock_connector_calls=1` and `real_external_http_calls=0`. | Browser interaction plus execution result inspection. | Execution JSON, UI execution state, negative egress evidence, audit refs. | AI |
| VS0-UI-009 | MUST_PASS | Audit step shows artifact/search/evidence/claim/action/approval/execution events and audit verification status. | Browser interaction plus `audit verify`. | Audit timeline screenshot/DOM snapshot, audit JSON, audit verification output. | AI |
| VS0-UI-010 | MUST_PASS | UI clearly says local VS0 proof only; production release, live connector, and human acceptance are not claimed. | Browser text/state inspection plus report review. | UI proof and scenario/report fields showing production release false, live connector false, human acceptance unclaimed until H01. | AI |
| VS0-UI-R01 | REGRESSION_GUARD | Existing EVUX governance remains PASS. | Run governance verifier and/or `make verify-vs0-evux`. | Exit-code transcript and scenario summary with no AI-verifiable failures. | AI |
| VS0-UI-R02 | REGRESSION_GUARD | Browser proof still cannot mark timeout as clean PASS. | Browser proof validator or targeted timeout test. | Timeout flag, browser exit status, and proof status showing timeout is not clean PASS. | AI |
| VS0-UI-H01 | HUMAN_REQUIRED | JiYong/Tars uses the UI and records accept/reject. | Human walkthrough. | Acceptance note with screenshots/recording, or rejection note with issue list. | Human |

## Required Local Verification

Future implementation should run these commands, or report exactly why any command could not run:

```sh
PATH="$PWD:$PATH" cornerstone scenario verify vs0-operator-acceptance-ui --json --output reports/scenario/vs0-operator-acceptance-ui-YYYY-MM-DD.json
PATH="$PWD:$PATH" cornerstone scenario gate reports/scenario/vs0-operator-acceptance-ui-YYYY-MM-DD.json --json
PATH="$PWD:$PATH" cornerstone scenario verify vs0-evux-governance --json --output reports/scenario/vs0-evux-governance-YYYY-MM-DD.json
PATH="$PWD:$PATH" cornerstone scenario gate reports/scenario/vs0-evux-governance-YYYY-MM-DD.json --json
make verify-vs0-evux
```

If the implementation includes release evidence packaging for this gate, it must record the exact command used, command exit codes, browser proof, audit verification, and manifest hashes.

## Definition Of All Scenarios Passing

All scenarios pass when:

1. Every AI-owned `VS0-UI-*` MUST_PASS row is `PASS`.
2. Every AI-owned `VS0-UI-*` REGRESSION_GUARD row is `PASS`.
3. No AI-owned row is `FAIL`, `NOT_VERIFIED`, or `NOT_RUN`.
4. `VS0-UI-H01` is accepted by JiYong/Tars with explicit human evidence.
5. Existing EVUX governance still passes.
6. Evidence includes browser/UI, CLI/API, action-result, audit, and report artifacts.
7. The final verdict does not claim production release, live-provider readiness, or human acceptance beyond the recorded human walkthrough.

If any AI-owned row is not `PASS`, the future implementation agent must not say "done." It must provide root cause, failed layer, fix/blocker, and re-verification plan.

## Human Required

| ID | Why AI Cannot Verify | Required Human Action | Expected Evidence | Release Impact |
|---|---|---|---|---|
| VS0-UI-H01 | Human operator acceptance is subjective and requires JiYong/Tars to judge whether the UI is understandable and controllable. | JiYong/Tars completes the local UI walkthrough and records accept or reject. | Acceptance note with screenshots/recording, or rejection note with issue list. | Blocks full VS-1 main implementation and operator-accepted VS0 claim; does not invalidate AI-verifiable EVUX governance PASS. |

## Stop Condition

Stop this slice when JiYong/Tars can say:

```text
Yes, I can understand and control the local VS0 workflow.
It is clear what is evidence, what is a Claim, what is an Action, what is mock/local, and what is not production-ready.
```

If the answer remains "No," do not move full VS-1 onto the main implementation track.

## Done Means

This scenario contract is frozen when this document and its machine-readable matrix are discoverable from the SoT/README.

Implementation is done only when every AI-verifiable `VS0-UI-*` MUST_PASS and REGRESSION_GUARD scenario is `PASS` with concrete evidence, no AI row remains `FAIL`, `NOT_VERIFIED`, or `NOT_RUN`, and `VS0-UI-H01` is accepted by JiYong/Tars with explicit human evidence.
