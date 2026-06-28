# Connector Hub CS-CH-H03 Human Review Template - 2026-06-24

**Owner:** JiYong / Tars
**Scenario:** `CS-CH-H03`
**Gate:** Human Chrome privacy review
**Status:** PENDING HUMAN REVIEW

## Review Target

Decide whether the real Chrome extension and local backend make browser capture permissions, active-tab capture, allowlist auto capture, sensitive-page handling, pause, and revoke understandable and correct.

This template prepares the human evidence shape only. It does not mark `CS-CH-H03` as `PASS`; the scenario remains `HUMAN_REQUIRED` until a dated human privacy review exists.

## Preparation Command

```sh
PATH="$PWD:$PATH" cornerstone connector human-gate package --scenario CS-CH-H03 --state-dir tmp/manual-connector-h03 --json
```

Expected package properties:

```text
schema_version=cs.connector_human_gate_package.v1
scenario_id=CS-CH-H03
status=human_review_required
approval_status=pending
real_browser_privacy_accepted=HUMAN_REQUIRED
product_claim_allowed=false
pass_claim_allowed_without_human_record=false
live_provider_calls_executed_by_package=0
provider_mutations_executed_by_package=0
external_mutations_executed_by_package=0
```

## Current Machine-Readable Handoff Snapshot

This snapshot is derived from `reports/scenario/connectorhub-human-gate-package-cs-ch-h03-2026-06-24.json`. It is operator-preparation only: `final_verdict=HUMAN_REQUIRED`, `goal_completion_claim_blocked=true`, and `full_goal_completion_allowed=false`.

| Handoff artifact | Purpose |
|---|---|
| `reports/scenario/connectorhub-human-gate-package-cs-ch-h03-2026-06-24.json` | Pinned H03 package with required evidence, dependency refs, and blank reviewer-record template. |
| `reports/scenario/connectorhub-human-gate-readiness-2026-06-24.json` | Pinned all-human-gate readiness rollup; this is not approval evidence. |
| `reports/scenario/connectorhub-human-gate-validation-handoff-2026-06-24.json` | Pinned ordered validation handoff with the H03 row and dependency blockers. |
| `reports/scenario/connectorhub-human-gate-record-template-cs-ch-h03-2026-06-24.json` | Pinned blank H03 reviewer-record template; this is not approval evidence. |
| `reports/scenario/connectorhub-human-gate-validation-blank-cs-ch-h03-2026-06-24.json` | Pinned blocked validation envelope proving the blank template remains invalid until a human fills required evidence. |

Current Chrome privacy evidence-packet workflow:

1. `cornerstone connector human-gate evidence-packet-contract --scenario CS-CH-H03 --json`
2. `cornerstone connector human-gate evidence-packet-file-contract --scenario CS-CH-H03 --json`
3. `cornerstone connector human-gate evidence-packet-scaffold --scenario CS-CH-H03 --packet-dir <h03-chrome-privacy-packet-dir> --json --write`
4. `cornerstone connector human-gate evidence-packet-validate --scenario CS-CH-H03 --packet-dir <h03-chrome-privacy-packet-dir> --json`
5. `cornerstone connector human-gate evidence-packet-record-draft --scenario CS-CH-H03 --packet-dir <h03-chrome-privacy-packet-dir> --json --record-output <reviewer-record-draft.json>`
6. `cornerstone connector human-gate validate-record --scenario CS-CH-H03 --record-file <filled-reviewer-record.json> --json --output <redacted-validation-envelope.json>`

Boundary flags: `schema_version=cs.connector_human_gate_h03_evidence_packet_workflow.v1`, `claim_boundary=h03_chrome_privacy_packet_workflow_is_operator_handoff_not_human_acceptance`, `acceptance_sufficient=false`, `product_claim_allowed=false`, `pass_claim_allowed=false`, `dependency_unlock_allowed_by_workflow=false`, `human_acceptance_collected_by_workflow=false`, `raw_packet_file_contents_recorded_by_workflow=false`, and `packet_file_contents_persisted_by_workflow=false`.

## Reviewer Record Submission Checklist

The blank reviewer record is still invalid until a human fills the generated `record_template` with redacted evidence refs and runs the structural validator. No row below is approval evidence, product acceptance, or a `PASS` claim.

| Submission item | Template source | Required action | Validator guard |
|---|---|---|---|
| Blank reviewer template | `reports/scenario/connectorhub-human-gate-record-template-cs-ch-h03-2026-06-24.json` | Write a blank template with `cornerstone connector human-gate package --scenario CS-CH-H03 --json --record-template-output <reviewer-record-template.json>`, then fill it from the browser privacy evidence packet. | The blank template is preparation data only; `blank_template_requires_human_evidence` keeps H03 `HUMAN_REQUIRED`. |
| Redacted validation envelope | `proposed_record_template.validation_output_command` | Submit the filled JSON with `cornerstone connector human-gate validate-record --scenario CS-CH-H03 --record-file <filled-json> --json --output <redacted-validation-envelope.json>`. | Redacted structural validation only; no raw record body, raw record path, decision value, senior-review finding text, or evidence-packet manifest values are persisted. |
| Required fields | `proposed_record_template.required_fields` | Fill `reviewer`, `review_timestamp`, `browser_profile_redacted`, `extension_version`, `permission_pages_or_recording_ref`, `active_tab_capture_ref`, `allowlist_auto_capture_ref`, `sensitive_block_ref`, `pause_revoke_ref`, `audit_refs`, `accept_or_reject_note`, and `issues_or_exceptions`. | Missing fields keep the validator envelope blocked and keep final verdict `HUMAN_REQUIRED`. |
| Evidence packet manifest | `proposed_record_template.required_evidence_packet_manifest` | Attach one distinct redacted evidence ref for `Recording or screenshots of permission UX and capture controls.`, `Timeline covering active-tab capture, allowlist auto-capture, sensitive-page handling, pause, and revoke.`, `Policy and audit refs for allowed, blocked, and degraded browser capture decisions.`, and `Human privacy acceptance or rejection note.` | Each row must keep the matching required-evidence label and use only `redacted`, `public_safe`, or `no_sensitive_material` as `redaction_status`. |
| Senior review perspectives | `proposed_record_template.required_senior_review_perspectives` | Fill `product_value`, `domain_architecture`, `data_contract`, `reliability_observability`, `security_privacy`, and `testability_migration` findings. | Finding text is not persisted by the validator; missing perspectives remain structural issues. |
| Dependency refs | `proposed_record_template.dependency_human_gates` | Attach a structurally valid ACCEPT `connector_human_gate_record_validation:<id>` ref for `CS-CH-H02`. | REJECT records do not unlock dependents; missing dependency refs keep H03 blocked. |

## Scenario-First Execution Runbook

1. Freeze the review scope before browser use: record the Chrome profile label, Chrome version, extension version/build, local backend version, source id, workspace, approved domains/source packs, sensitive test pages, retention expectation, and evidence directory. Reject the review if the browser profile contains unrelated private state that cannot be redacted or if the extension/backend version is unknown.
2. Verify baseline permission posture: record the extension permission page, host permissions, optional permissions, and first-use copy before capture. Reject if broad all-site access is granted by default, if capture is enabled before explicit setup, or if setup language hides source scope, browser permission, retention, pause, or revoke controls.
3. Exercise explicit active-tab capture: use a deliberate user gesture on one approved active tab, record the extension preflight/confirmation, and attach backend policy and audit refs. Reject if opening the popup alone captures, if another tab/page is considered, or if raw HTML/text, cookies, storage, screenshots, form values, or browser history are collected.
4. Exercise allowlist auto capture: configure one approved domain/source pack and one unapproved domain, then record page-load, URL-change, or tab-activate behavior. Reject if unknown domains are opportunistically added, if unapproved-domain content is sent/stored, if client consent/config differs from backend consent/config, or if throttling/idempotency is missing from the timeline.
5. Exercise sensitive-page handling using synthetic sensitive pages only: cover password, payment, token/secret, compose/editable, browser-internal, private-account, unsupported-scheme, and oversized-page classes when available. Reject if a client block is downgraded by the backend, if raw sensitive content is persisted or model-sent, or if the user-facing reason is absent or misleading.
6. Exercise pause and revoke: pause the source or Watch Rule, attempt capture, resume only if needed, revoke permission/consent, then attempt capture again. Reject if new samples appear while paused or revoked, if the extension UI still claims capture readiness, or if backend ingestion accepts stale client payloads.
7. Review the capture timeline and diagnostics: attach captured, degraded, skipped, duplicate, blocked, paused, and revoked outcomes with reason codes, evidence refs, and audit refs. Reject if diagnostics expose private content or if an operator cannot distinguish allowed capture, safe skip, policy block, and permission failure.
8. Verify retention, export, delete, and audit implications: attach scoped export or state-review notes, deletion/dismissal guidance if applicable, retained-audit explanation, manifest/checksum, and redaction scan. Reject if the review implies unsupported "delete everything" behavior or if secrets/private payloads appear in evidence.

## Acceptance Evidence Packet

| Artifact | Required content | Human result |
|---|---|---|
| `chrome-review-scope.md` | Chrome profile label, Chrome version, extension/build id, backend version, source id, workspace, approved domains/source packs, sensitive fixture list, evidence directory. | PENDING |
| `permission-baseline.md` | Extension permission pages, host/optional permissions, first-use copy, explicit no-default-capture statement. | PENDING |
| `active-tab-capture-proof.json` | User gesture, active tab identifier, preflight/confirmation state, bounded payload metadata, backend policy/audit refs, raw persistence counters. | PENDING |
| `allowlist-auto-capture-proof.json` | Approved-domain capture, unapproved-domain skip, consent/config refs, trigger metadata, throttle/idempotency evidence, timeline refs. | PENDING |
| `sensitive-page-policy-proof.json` | Synthetic sensitive classes, block/degrade decisions, backend revalidation result, user-facing reason, raw persistence/model-send counters. | PENDING |
| `pause-revoke-proof.json` | Pause timestamp, paused capture attempt denial, revoke timestamp, post-revoke denial, stale-payload rejection, audit refs. | PENDING |
| `timeline-diagnostics.json` | Captured/degraded/skipped/duplicate/blocked/paused/revoked entries, reason codes, evidence refs, audit refs, no private payloads. | PENDING |
| `retention-export-delete-notes.md` | Scoped export or state-review notes, deletion/dismissal guidance, retained-audit explanation. | PENDING |
| `privacy-issue-log.md` | Reviewer issue list, severity, scenario reference, accept/reject impact, follow-up owner. | PENDING |
| `audit-redaction-report.json` | Evidence manifest or checksum, redaction scan result, zero raw secrets/private payload/browser-history exposure. | PENDING |
| `review-decision.md` | Reviewer decision, timestamp, accept/reject rationale, exceptions, follow-ups. | PENDING |

## Redaction And Handling Rules

- Use synthetic test pages for sensitive classes. Do not capture or attach real passwords, payment data, tokens, private messages, private account pages, raw HTML, raw browser text, cookies, storage values, screenshots with private content, or browser history.
- Prefer metadata, hashes, policy decisions, reason codes, bounded payload shape, evidence refs, and audit refs over raw webpage content.
- Screenshots and recordings must show permission/privacy UX clearly while redacting page content, profile identifiers, private URLs, accounts, and unrelated tabs/windows.
- Treat broad default all-site access, capture without user gesture or allowlist, unapproved-domain capture, backend downgrade of a client block, capture while paused/revoked, or unredacted sensitive/private payload exposure as reject conditions.

## Senior Review Perspectives

| Perspective | H03 acceptance question |
|---|---|
| Product value | Is the browser capture experience understandable as one CornerStone workflow without exposing internal ConnectorHub mechanics? |
| Domain architecture | Does extension input remain behind ConnectorPort and backend policy validation? |
| Data contract | Are active-tab payload, allowlist config, sensitive-page decision, lifecycle decision, and audit refs present? |
| Reliability and observability | Does the packet distinguish manual capture, auto capture, block/degrade, pause, and revoke on a clear timeline? |
| Security and privacy | Are sensitive-page leaks, outside-allowlist capture, and confusing permission states absent? |
| Testability and migration | Does the review match the `CS-CH-024`, `CS-CH-025`, `CS-CH-026`, and `CS-CH-027` fixture contracts? |

## Decision Record

```text
Decision: ACCEPT / REJECT
Reviewer:
Review timestamp:
Browser profile identifier, redacted:
Extension version:
Evidence location:
Issues or exceptions:
```

## Evidence Checklist

| Required evidence | Human result | Notes |
|---|---|---|
| Permission pages and first-use browser capture copy are recorded. | PENDING | |
| Explicit active-tab capture produces policy and audit refs. | PENDING | |
| Allowlist auto capture works only for allowed targets. | PENDING | |
| Outside-allowlist capture is blocked with understandable diagnostics. | PENDING | |
| Sensitive-page capture is blocked or degraded with safe reason guidance. | PENDING | |
| Pause and revoke prevent further capture. | PENDING | |
| Human privacy acceptance or rejection note is attached. | PENDING | |

## Acceptance Statement

Accept only if the real browser behavior matches `CS-CH-024`, `CS-CH-025`, `CS-CH-026`, and `CS-CH-027` without confusing users about scope or permission boundaries.

## Boundary

This review blocks Chrome capture and browser privacy acceptance. It does not replace backend fixture proof or production security proof.
