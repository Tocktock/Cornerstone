# Connector Hub CS-CH-H02 Human Review Template - 2026-06-24

**Owner:** JiYong / Tars
**Scenario:** `CS-CH-H02`
**Gate:** Human macOS permission walkthrough
**Status:** PENDING HUMAN REVIEW

## Review Target

Decide whether a supported physical Mac proves that capture starts only after visible consent and platform permission, then stops on pause or revoke.

This template prepares the human evidence shape only. It does not mark `CS-CH-H02` as `PASS`; the scenario remains `HUMAN_REQUIRED` until a dated human record with physical-device evidence exists.

## Preparation Command

```sh
PATH="$PWD:$PATH" cornerstone connector human-gate package --scenario CS-CH-H02 --state-dir tmp/manual-connector-h02 --json
```

Expected package properties:

```text
schema_version=cs.connector_human_gate_package.v1
scenario_id=CS-CH-H02
status=human_review_required
approval_status=pending
local_physical_device_behavior_verified=HUMAN_REQUIRED
product_claim_allowed=false
pass_claim_allowed_without_human_record=false
live_provider_calls_executed_by_package=0
provider_mutations_executed_by_package=0
external_mutations_executed_by_package=0
```

## Current Machine-Readable Handoff Snapshot

This snapshot is derived from `reports/scenario/connectorhub-human-gate-package-cs-ch-h02-2026-06-24.json`. It is operator-preparation only: `final_verdict=HUMAN_REQUIRED`, `goal_completion_claim_blocked=true`, and `full_goal_completion_allowed=false`.

| Handoff artifact | Purpose |
|---|---|
| `reports/scenario/connectorhub-human-gate-package-cs-ch-h02-2026-06-24.json` | Pinned H02 package with required evidence, dependency refs, and blank reviewer-record template. |
| `reports/scenario/connectorhub-human-gate-readiness-2026-06-24.json` | Pinned all-human-gate readiness rollup; this is not approval evidence. |
| `reports/scenario/connectorhub-human-gate-validation-handoff-2026-06-24.json` | Pinned ordered validation handoff with the H02 row and dependency blockers. |
| `reports/scenario/connectorhub-human-gate-record-template-cs-ch-h02-2026-06-24.json` | Pinned blank H02 reviewer-record template; this is not approval evidence. |
| `reports/scenario/connectorhub-human-gate-validation-blank-cs-ch-h02-2026-06-24.json` | Pinned blocked validation envelope proving the blank template remains invalid until a human fills required evidence. |

Current macOS permission evidence-packet workflow:

This workflow is operator-preparation only. It records packet file hashes and redacted refs, but it does not record raw screenshots, raw captured content, physical-device evidence, human acceptance, dependency unlock, or `CS-CH-H02` `PASS`.

| Step | Command | Output role |
|---:|---|---|
| 1 | `cornerstone connector human-gate evidence-packet-contract --scenario CS-CH-H02 --json` | Required evidence manifest and redaction contract. |
| 2 | `cornerstone connector human-gate evidence-packet-file-contract --scenario CS-CH-H02 --json` | Required macOS permission packet file list and content expectations. |
| 3 | `cornerstone connector human-gate evidence-packet-scaffold --scenario CS-CH-H02 --packet-dir <h02-macos-permission-packet-dir> --json --write` | Blank local packet templates without overwriting existing evidence. |
| 4 | `cornerstone connector human-gate evidence-packet-validate --scenario CS-CH-H02 --packet-dir <h02-macos-permission-packet-dir> --json` | Hash-only packet validation envelope. |
| 5 | `cornerstone connector human-gate evidence-packet-record-draft --scenario CS-CH-H02 --packet-dir <h02-macos-permission-packet-dir> --json --record-output <reviewer-record-draft.json>` | Hash-only reviewer record draft; human decision fields remain human-owned. |
| 6 | `cornerstone connector human-gate validate-record --scenario CS-CH-H02 --record-file <filled-reviewer-record.json> --json --output <redacted-validation-envelope.json>` | Redacted structural validation envelope for the completed reviewer record. |

Boundary flags: `schema_version=cs.connector_human_gate_h02_evidence_packet_workflow.v1`, `claim_boundary=h02_macos_permission_packet_workflow_is_operator_handoff_not_human_acceptance`, `acceptance_sufficient=false`, `product_claim_allowed=false`, `pass_claim_allowed=false`, `dependency_unlock_allowed_by_workflow=false`, `human_acceptance_collected_by_workflow=false`, `raw_packet_file_contents_recorded_by_workflow=false`, and `packet_file_contents_persisted_by_workflow=false`.

## Reviewer Record Submission Checklist

The blank reviewer record is still invalid until a human fills the generated `record_template` with redacted evidence refs and runs the structural validator. No row below is approval evidence, product acceptance, or a `PASS` claim.

| Submission item | Template source | Required action | Validator guard |
|---|---|---|---|
| Blank reviewer template | `reports/scenario/connectorhub-human-gate-record-template-cs-ch-h02-2026-06-24.json` | Write a blank template with `cornerstone connector human-gate package --scenario CS-CH-H02 --json --record-template-output <reviewer-record-template.json>`, then fill it from the physical-device walkthrough packet. | The blank template is preparation data only; `blank_template_requires_human_evidence` keeps H02 `HUMAN_REQUIRED`. |
| Redacted validation envelope | `proposed_record_template.validation_output_command` | Submit the filled JSON with `cornerstone connector human-gate validate-record --scenario CS-CH-H02 --record-file <filled-json> --json --output <redacted-validation-envelope.json>`. | Redacted structural validation only; no raw record body, raw record path, decision value, senior-review finding text, or evidence-packet manifest values are persisted. |
| Required fields | `proposed_record_template.required_fields` | Fill `reviewer`, `review_timestamp`, `device_os_version_redacted`, `consent_record`, `permission_state_snapshot`, `first_sample_ref`, `pause_revoke_timestamps`, `screenshots_or_recording_ref`, `audit_refs`, and `issues_or_exceptions`. | Missing fields keep the validator envelope blocked and keep final verdict `HUMAN_REQUIRED`. |
| Evidence packet manifest | `proposed_record_template.required_evidence_packet_manifest` | Attach one distinct redacted evidence ref for `Recording or screenshots of consent, permission grant, pause, and revoke.`, `Status transcript showing permission and capture state transitions.`, `First bounded sample reference with owner/workspace scope.`, and `Proof that pause and revoke stop capture or block sample attempts.` | Each row must keep the matching required-evidence label and use only `redacted`, `public_safe`, or `no_sensitive_material` as `redaction_status`. |
| Senior review perspectives | `proposed_record_template.required_senior_review_perspectives` | Fill `product_value`, `domain_architecture`, `data_contract`, `reliability_observability`, `security_privacy`, and `testability_migration` findings. | Finding text is not persisted by the validator; missing perspectives remain structural issues. |
| Dependency refs | `proposed_record_template.dependency_human_gates` | Attach structurally valid ACCEPT `connector_human_gate_record_validation:<id>` refs for `CS-CH-H04` and `CS-CH-H07`. | REJECT records do not unlock dependents; missing dependency refs keep H02 blocked. |

## Scenario-First Execution Runbook

1. Freeze the review scope before touching the device: record the physical device label, macOS version and build, app/build identifier, source id, workspace, retention expectation, and evidence directory. Reject the review if the device is shared, the OS/app build is unsupported or unknown, or the scope cannot be tied to one workspace and source.
2. Verify capture is disabled before consent and platform permission: attach the initial status transcript and redacted screenshot. Reject if any sample, audit event, or background observation exists before consent and platform permission are granted.
3. Walk platform permission and owner consent as separate gates: record the macOS prompt or Settings state, the CornerStone consent screen, the consent timestamp, and the permission timestamp. Reject if either gate is bypassed, hidden, misleading, or impossible to revoke.
4. Capture the first permitted sample: attach redacted sample metadata with source id, timestamp, app or domain class, evidence ref, and audit ref. Do not attach raw screenshots, raw window titles, raw text, or private content unless explicitly approved and redacted.
5. Pause capture and attempt one sample while paused: attach the pause timestamp, status transcript, blocked or denied sample attempt, and audit ref. Reject if capture continues, status remains ambiguous, or the sample attempt is silently accepted while paused.
6. Resume only when needed for the walkthrough, then revoke permission or consent: attach the revoke timestamp, status transcript, blocked post-revoke sample attempt, and audit ref. Reject if capture continues after revoke or if the UI/API claims readiness after authority is removed.
7. Record lifecycle handling for retained evidence: attach export, retention, deletion, or state-review notes according to the local lifecycle contract in force for the sample. Reject if evidence retention or deletion behavior is unsupported, undocumented, or inconsistent with the operator-facing status.
8. Verify audit and redaction: attach audit ids, a manifest or checksum for the evidence packet, and a redaction scan summary. Reject if secrets, private payloads, unrelated app/window data, or unapproved raw content appear in the evidence packet.

## Acceptance Evidence Packet

| Artifact | Required content | Human result |
|---|---|---|
| `macos-review-scope.md` | Device label, macOS build, app/build id, source id, workspace, retention expectation, evidence directory. | PENDING |
| `initial-disabled-state.json` | Initial status transcript proving no capture before consent and platform permission. | PENDING |
| `permission-consent-record.md` | macOS permission evidence, CornerStone consent evidence, permission timestamp, consent timestamp. | PENDING |
| `first-sample-record.json` | Redacted first-sample metadata, evidence ref, audit ref, no unapproved raw private content. | PENDING |
| `pause-proof.json` | Pause timestamp, status transcript, blocked or denied sample attempt, audit ref. | PENDING |
| `revoke-proof.json` | Revoke timestamp, post-revoke status transcript, blocked post-revoke sample attempt, audit ref. | PENDING |
| `retention-export-delete-notes.md` | Export, retention, deletion, or state-review notes aligned to local lifecycle semantics. | PENDING |
| `audit-redaction-report.json` | Audit ids, manifest or checksum, redaction scan result, zero secret/private payload exposure. | PENDING |
| `review-decision.md` | Reviewer decision, timestamp, accept/reject rationale, exceptions, follow-ups. | PENDING |

## Redaction And Handling Rules

- Do not place screenshots, raw text, raw window titles, private app names, private URLs, or user content in this template unless the reviewer explicitly approves the redacted form.
- Use stable redacted labels for app, domain, window, source, and workspace identifiers so the sequence remains auditable without exposing private content.
- Prefer metadata, hashes, classes, timestamps, evidence refs, and audit refs over raw captured content.
- Treat any capture before consent, capture after pause, capture after revoke, hidden startup capture, or unredacted secret/private payload exposure as a reject condition.

## Senior Review Perspectives

| Perspective | H02 acceptance question |
|---|---|
| Product value | Does physical-device capture earn trust because users can see, pause, revoke, and understand observation state? |
| Domain architecture | Does platform permission remain ConnectorHub input evidence instead of hidden Product authority? |
| Data contract | Are consent, permission, source state, sample, lifecycle decision, and audit refs present for the walkthrough? |
| Reliability and observability | Are grant, first sample, pause, revoke, and blocked post-revoke sample attempts timestamped and replayable? |
| Security and privacy | Is capture absent before explicit consent and stopped after pause or revoke? |
| Testability and migration | Does physical-device behavior match the `CS-CH-021`, `CS-CH-022`, and `CS-CH-027` local fixture contracts? |

## Decision Record

```text
Decision: ACCEPT / REJECT
Reviewer:
Review timestamp:
Device and OS version, redacted:
Evidence location:
Issues or exceptions:
```

## Evidence Checklist

| Required evidence | Human result | Notes |
|---|---|---|
| Signed or locally approved build is identified. | PENDING | |
| Consent screen and platform permission grant are recorded. | PENDING | |
| First bounded activity sample is captured with source, scope, and audit refs. | PENDING | |
| Pause timestamp and no-unexpected-observation proof are attached. | PENDING | |
| Revoke timestamp and blocked post-revoke sample attempt proof are attached. | PENDING | |
| No secret, private content, or unrelated app/window data appears in evidence. | PENDING | |

## Acceptance Statement

Accept only if consent, permission, first sample, pause, and revoke behavior match the local `CS-CH-021`, `CS-CH-022`, and `CS-CH-027` fixture contracts. Reject if capture starts without consent or continues after pause/revoke.

## Boundary

This review blocks physical-device Watch and capture readiness. It does not prove live provider readiness, production topology, or human UX acceptance.
