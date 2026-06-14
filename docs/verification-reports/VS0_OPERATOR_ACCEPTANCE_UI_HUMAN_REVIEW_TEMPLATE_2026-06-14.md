# VS0 Operator Acceptance UI Human Review Template - 2026-06-14

**Owner:** JiYong / Tars  
**Scenario:** `VS0-UI-H01`  
**Gate:** `VS0_OPERATOR_ACCEPTANCE_UI_GATE`  
**Status:** PENDING HUMAN REVIEW

## Review Target

Open the local VS0 operator UI and decide whether the workflow is understandable and controllable enough to unblock full VS-1 main implementation.

Recommended local URL:

```text
http://127.0.0.1:8790/?scenario=vs0-evux
```

If that server is not running, start it from the repository root:

```sh
PATH="$PWD:$PATH" cornerstone runtime serve --port 8790 --state-dir tmp/manual-vs0-operator-ui
```

## Decision

```text
Decision: ACCEPT / REJECT
Reviewer:
Review date:
Evidence: screenshot / recording / issue list / notes
```

## Walkthrough Checklist

| Item | Human Result | Notes |
|---|---|---|
| UI shows the current step and workflow position. | PENDING | |
| Select/upload Artifact step is understandable. | PENDING | |
| Artifact details show ID, checksum, source, derived status, evidence refs, and audit refs. | PENDING | |
| Search step shows query, result snippet, search snapshot ID, and evidence eligibility. | PENDING | |
| Evidence step shows what supports the Claim and what would be insufficient. | PENDING | |
| Claim step makes Draft, Evidence-backed, and Approved states clear. | PENDING | |
| Zero-evidence Claim approval denial shows cause and resolution guide. | PENDING | |
| Action Card shows diff, expected impact, evidence, policy decision, risk, approval state, mock/local boundary, and rollback/compensation note. | PENDING | |
| Dry-run step is understandable before approval/execution. | PENDING | |
| Approval step makes Claim and Action approval explicit. | PENDING | |
| Execute step clearly shows local/mock execution only. | PENDING | |
| Execution proof shows `mock_connector_calls=1` and `real_external_http_calls=0`. | PENDING | |
| Audit step shows artifact/search/evidence/claim/action/approval/execution events. | PENDING | |
| Audit verification status is understandable. | PENDING | |
| UI does not imply production release readiness. | PENDING | |
| UI does not imply live connector/provider readiness. | PENDING | |
| UI does not imply human acceptance before this review is recorded. | PENDING | |

## Acceptance Statement

Use this if accepted:

```text
I accept VS0-UI-H01.
I can understand and control the local VS0 workflow.
It is clear what is evidence, what is a Claim, what is an Action, what is mock/local, and what is not production-ready.
Full VS-1 main implementation may proceed only after this acceptance is committed as evidence.
```

## Rejection Statement

Use this if rejected:

```text
I reject VS0-UI-H01.
The UI is not yet understandable or controllable enough for the local VS0 operator workflow.
Blocking issues:
1.
2.
3.
```

## Boundary

This human review can only accept local VS0 operator UX. It must not claim:

- production release readiness;
- live ConnectorHub/provider readiness;
- autonomous external writeback readiness;
- full VS-1 completion.
