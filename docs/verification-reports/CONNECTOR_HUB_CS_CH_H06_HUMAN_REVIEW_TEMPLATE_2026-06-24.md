# Connector Hub CS-CH-H06 Human Review Template - 2026-06-24

**Owner:** JiYong / Tars
**Scenario:** `CS-CH-H06`
**Gate:** Human connected-source usability trust study
**Status:** PENDING HUMAN REVIEW

## Review Target

Decide whether a representative user can complete Connected Sources and Capture Inbox first-use tasks with enough understanding and trust to adopt ConnectorHub concepts inside CornerStone.

This template prepares the human evidence shape only. It does not mark `CS-CH-H06` as `PASS`; the scenario remains `HUMAN_REQUIRED` until a dated usability/trust study exists.

## Preparation Command

```sh
PATH="$PWD:$PATH" cornerstone connector human-gate package --scenario CS-CH-H06 --state-dir tmp/manual-connector-h06 --json
```

Current usability/trust study evidence-packet workflow:

```sh
cornerstone connector human-gate evidence-packet-contract --scenario CS-CH-H06 --json
cornerstone connector human-gate evidence-packet-file-contract --scenario CS-CH-H06 --json
cornerstone connector human-gate evidence-packet-scaffold --scenario CS-CH-H06 --packet-dir <h06-usability-trust-packet-dir> --json --write
cornerstone connector human-gate evidence-packet-validate --scenario CS-CH-H06 --packet-dir <h06-usability-trust-packet-dir> --json
cornerstone connector human-gate evidence-packet-record-draft --scenario CS-CH-H06 --packet-dir <h06-usability-trust-packet-dir> --json --record-output <reviewer-record-draft.json>
cornerstone connector human-gate validate-record --scenario CS-CH-H06 --record-file <filled-reviewer-record.json> --json --output <redacted-validation-envelope.json>
```

Workflow boundary:

```text
schema_version=cs.connector_human_gate_h06_evidence_packet_workflow.v1
claim_boundary=h06_usability_trust_packet_workflow_is_operator_handoff_not_human_acceptance
acceptance_sufficient=false
product_claim_allowed=false
pass_claim_allowed=false
dependency_unlock_allowed_by_workflow=false
human_acceptance_collected_by_workflow=false
raw_packet_file_contents_recorded_by_workflow=false
packet_file_contents_persisted_by_workflow=false
```

Expected package properties:

```text
schema_version=cs.connector_human_gate_package.v1
scenario_id=CS-CH-H06
status=human_review_required
approval_status=pending
human_ux_privacy_accepted=HUMAN_REQUIRED
product_claim_allowed=false
pass_claim_allowed_without_human_record=false
live_provider_calls_executed_by_package=0
provider_mutations_executed_by_package=0
external_mutations_executed_by_package=0
```

## Current Machine-Readable Handoff Snapshot

This snapshot is derived from `reports/scenario/connectorhub-human-gate-package-cs-ch-h06-2026-06-24.json`. It is operator-preparation only: `final_verdict=HUMAN_REQUIRED`, `goal_completion_claim_blocked=true`, and `full_goal_completion_allowed=false`.

| Handoff artifact | Purpose |
|---|---|
| `reports/scenario/connectorhub-human-gate-package-cs-ch-h06-2026-06-24.json` | Pinned H06 package with required evidence, dependency refs, and blank reviewer-record template. |
| `reports/scenario/connectorhub-human-gate-readiness-2026-06-24.json` | Pinned all-human-gate readiness rollup; this is not approval evidence. |
| `reports/scenario/connectorhub-human-gate-validation-handoff-2026-06-24.json` | Pinned ordered validation handoff with the H06 row and dependency blockers. |
| `reports/scenario/connectorhub-human-gate-record-template-cs-ch-h06-2026-06-24.json` | Pinned blank H06 reviewer-record template; this is not approval evidence. |
| `reports/scenario/connectorhub-human-gate-validation-blank-cs-ch-h06-2026-06-24.json` | Pinned blocked validation envelope proving the blank template remains invalid until a human fills required evidence. |

## Reviewer Record Submission Checklist

The blank reviewer record is still invalid until a human fills the generated `record_template` with redacted evidence refs and runs the structural validator. No row below is approval evidence, product acceptance, or a `PASS` claim.

| Submission item | Template source | Required action | Validator guard |
|---|---|---|---|
| Blank reviewer template | `reports/scenario/connectorhub-human-gate-record-template-cs-ch-h06-2026-06-24.json` | Write a blank template with `cornerstone connector human-gate package --scenario CS-CH-H06 --json --record-template-output <reviewer-record-template.json>`, then fill it from the usability/trust study packet. | The blank template is preparation data only; `blank_template_requires_human_evidence` keeps H06 `HUMAN_REQUIRED`. |
| Redacted validation envelope | `proposed_record_template.validation_output_command` | Submit the filled JSON with `cornerstone connector human-gate validate-record --scenario CS-CH-H06 --record-file <filled-json> --json --output <redacted-validation-envelope.json>`. | Redacted structural validation only; no raw record body, raw record path, decision value, senior-review finding text, or evidence-packet manifest values are persisted. |
| Required fields | `proposed_record_template.required_fields` | Fill `reviewer`, `review_timestamp`, `participant_profile_redacted`, `task_script_ref`, `fixture_workspace_ref`, `timed_task_notes`, `screenshots_or_recording_ref`, `scoring_rubric`, `acceptance_decision`, and `issues_or_exceptions`. | Missing fields keep the validator envelope blocked and keep final verdict `HUMAN_REQUIRED`. |
| Evidence packet manifest | `proposed_record_template.required_evidence_packet_manifest` | Attach one distinct redacted evidence ref for `Task script and fixture workspace used for the study.`, `Timed task notes and recording or screenshots.`, `Scoring rubric for comprehension, trust, completion, and risk.`, and `Acceptance decision with issue list and release impact.` | Each row must keep the matching required-evidence label and use only `redacted`, `public_safe`, or `no_sensitive_material` as `redaction_status`. |
| Senior review perspectives | `proposed_record_template.required_senior_review_perspectives` | Fill `product_value`, `domain_architecture`, `data_contract`, `reliability_observability`, `security_privacy`, and `testability_migration` findings. | Finding text is not persisted by the validator; missing perspectives remain structural issues. |
| Dependency refs | `proposed_record_template.dependency_human_gates` | Attach structurally valid ACCEPT `connector_human_gate_record_validation:<id>` refs for `CS-CH-H01`, `CS-CH-H02`, `CS-CH-H03`, `CS-CH-H04`, `CS-CH-H05`, and `CS-CH-H07`. | REJECT records do not unlock dependents; missing dependency refs keep H06 blocked. |

## Scenario-First Study Runbook

1. Freeze the study scope before participant contact: record participant profile class, fixture workspace, task script version, product build, reviewed surfaces, facilitator, observer, evidence directory, and redaction plan. Reject the study if the participant must know repository names, ConnectorHub internals, Provider Pack mechanics, or implementation architecture to complete normal-user tasks.
2. Start with a neutral first-use briefing: ask the participant to complete tasks as one CornerStone product, with no explanation of expected answers. Record baseline expectations and concerns about connected sources, capture, evidence, privacy, and actions.
3. Run the Connected Sources setup task: participant identifies purpose, selected resources, readiness, Source Policy, permissions, freshness, last sync, setup gaps, and safe next steps. Reject if the participant cannot tell what is connected, what is not connected, what data is allowed, or what setup gap blocks value.
4. Run the Capture Inbox and Watch Result review task: participant opens a captured item, identifies Observation, Inference, Evidence/Caveats, Proposed next step, source, freshness, save/dismiss/feedback controls, and why unsupported inference remains reviewable rather than fact. Reject if observation and inference are conflated or if evidence/caveats are invisible.
5. Run the privacy-control task: participant finds source toggles, domain/source-pack controls, pause/resume, retention, export, delete/disable, and permission diagnostics. Reject if pause/revoke/delete expectations are misleading, if the user thinks collection is always on, or if surveillance concern cannot be resolved from the UI.
6. Run the action-boundary task without executing a live action: participant reviews an ActionCard or proposed action and explains evidence basis, policy/risk label, approval need, expected consequence, ConnectorHub preflight/provider feasibility, and why GitHub writes remain unavailable. Reject if the participant treats connector permission as Product approval or cannot explain what would happen before execution.
7. Run a progressive-disclosure check: participant locates admin/operator details only when needed, such as provider mapping, Source Policy, retries, quarantine, audit, and raw-access controls. Reject if normal-user navigation is connector-admin-first or if internal package/repo names are required for first value.
8. Complete a teach-back and scoring rubric: participant explains source scope, evidence basis, risk labels, capture controls, action consequences, and trust concerns in their own words. The reviewer records task timing, assistance count, comprehension score, trust score, unacceptable-risk findings, issue severity, and accept/reject decision.

## Acceptance Evidence Packet

| Artifact | Required content | Human result |
|---|---|---|
| `study-scope.md` | Participant profile class, fixture workspace, task script version, product build, surfaces reviewed, facilitator/observer, evidence directory, redaction plan. | PENDING |
| `participant-consent-redacted.md` | Consent to participate and record, redacted participant identifier, privacy handling note. | PENDING |
| `fixture-workspace-seed.json` | Scenario fixtures, connected-source states, Capture Inbox items, Watch Results, ActionCard/proposed-action state, no private/live provider data. | PENDING |
| `task-script.md` | Connected Sources, Capture Inbox, Watch Result, privacy controls, action-boundary, and progressive-disclosure tasks. | PENDING |
| `timed-task-notes.md` | Start/end times, completion status, assistance count, errors, hesitation points, facilitator-neutral prompts. | PENDING |
| `screenshots-or-recording-manifest.md` | Recording/screenshot refs with redaction notes and timestamps for task-critical moments. | PENDING |
| `connected-sources-comprehension.json` | Participant explanation of purpose, selected resources, readiness, policy, permissions, freshness, setup gaps, and safe next steps. | PENDING |
| `capture-inbox-watch-result-comprehension.json` | Participant explanation of Observation, Inference, Evidence/Caveats, Proposed next step, source/freshness, save/dismiss/feedback. | PENDING |
| `privacy-control-comprehension.json` | Participant explanation of pause/resume, revoke, retention, export, delete/disable, diagnostics, and surveillance concern handling. | PENDING |
| `action-boundary-comprehension.json` | Participant explanation of evidence basis, risk label, approval need, expected consequence, provider feasibility, and no GitHub writes. | PENDING |
| `rubric-scores.json` | Task completion, comprehension, trust, risk clarity, assistance, unacceptable-risk flags, reviewer threshold decision. | PENDING |
| `issue-log.md` | Issue severity, scenario reference, observed evidence, recommended fix, release impact. | PENDING |
| `audit-redaction-report.json` | Evidence manifest or checksum, redaction scan, zero private participant/provider payload exposure. | PENDING |
| `review-decision.md` | Reviewer decision, timestamp, accept/reject rationale, exceptions, follow-ups. | PENDING |

## Scoring And Reject Rules

- Minimum acceptable outcome: all core tasks completed, no critical safety misunderstanding, source scope and evidence basis explained without facilitator rescue, and action consequences/approval boundaries explained correctly.
- Reject if the participant needs repository, package, Provider Pack, Setup Result, Projection, Delivery, or ConnectorHub implementation knowledge to complete normal-user value tasks.
- Reject if the participant cannot distinguish connected-source readiness from production readiness, observation from inference, provider feasibility from Product approval, or pause/revoke from deletion.
- Reject if any task evidence exposes participant private data, provider secrets, raw provider payloads, unrelated browser/app content, or unredacted screenshots/recordings.
- Record non-blocking usability issues separately from acceptance blockers; do not convert a study with unresolved critical trust or safety issues into `PASS`.

## Senior Review Perspectives

| Perspective | H06 acceptance question |
|---|---|
| Product value | Can users complete Connected Sources and Capture Inbox tasks as one CornerStone product? |
| Domain architecture | Are setup, evidence, policy, and audit details progressively disclosed instead of connector-admin-first? |
| Data contract | Are task script, participant profile, timed notes, issue list, and acceptance decision refs attached? |
| Reliability and observability | Is there enough trace material to reproduce confusion, trust gaps, and recovery paths? |
| Security and privacy | Do users correctly understand capture scope, approval boundaries, raw access, and external action risk? |
| Testability and migration | Does the study show whether ConnectorHub concepts are adoptable in CornerStone UI language? |

## Decision Record

```text
Decision: ACCEPT / REJECT
Reviewer:
Review timestamp:
Participant profile, redacted:
Task script reference:
Evidence location:
Issues or exceptions:
```

## Evidence Checklist

| Required evidence | Human result | Notes |
|---|---|---|
| Task script and fixture workspace are attached. | PENDING | |
| Participant completes Connected Sources setup and evidence review tasks. | PENDING | |
| Participant completes Capture Inbox review tasks. | PENDING | |
| Timed notes and screenshots or recording are attached. | PENDING | |
| Scoring rubric covers comprehension, trust, task completion, and unacceptable risk. | PENDING | |
| User does not need internal repository or connector architecture knowledge. | PENDING | |
| Acceptance decision and issue list are attached. | PENDING | |

## Acceptance Statement

Accept only if the user can complete the workflow as one CornerStone product with clear evidence, policy, capture, and approval boundaries.

## Boundary

This review blocks human UX and trust acceptance. It does not prove live provider, production security, or recovery readiness.
