# Connector Hub CS-CH-H07 Human Review Template - 2026-06-24

**Owner:** JiYong / Tars
**Scenario:** `CS-CH-H07`
**Gate:** Human recovery exercise
**Status:** PENDING HUMAN REVIEW

## Review Target

Decide whether production-like backup, restore, cursor reconciliation, replay, and audit verification recover ConnectorHub-related CornerStone state without duplicate or lost logical truth.

This template prepares the human evidence shape only. It does not mark `CS-CH-H07` as `PASS`; the scenario remains `HUMAN_REQUIRED` until an operator recovery exercise exists.

## Preparation Command

```sh
PATH="$PWD:$PATH" cornerstone connector human-gate package --scenario CS-CH-H07 --state-dir tmp/manual-connector-h07 --json
```

Expected package properties:

```text
schema_version=cs.connector_human_gate_package.v1
scenario_id=CS-CH-H07
status=human_review_required
approval_status=pending
production_like_recovery_verified=HUMAN_REQUIRED
production_readiness_verified=NOT_VERIFIED
product_claim_allowed=false
pass_claim_allowed_without_human_record=false
live_provider_calls_executed_by_package=0
provider_mutations_executed_by_package=0
external_mutations_executed_by_package=0
```

## Current Machine-Readable Handoff Snapshot

This snapshot is derived from `reports/scenario/connectorhub-human-gate-package-cs-ch-h07-2026-06-24.json`. It is operator-preparation only: `final_verdict=HUMAN_REQUIRED`, `goal_completion_claim_blocked=true`, and `full_goal_completion_allowed=false`.

| Handoff artifact | Purpose |
|---|---|
| `reports/scenario/connectorhub-human-gate-package-cs-ch-h07-2026-06-24.json` | Pinned H07 package with required evidence, dependency refs, and blank reviewer-record template. |
| `reports/scenario/connectorhub-human-gate-readiness-2026-06-24.json` | Pinned all-human-gate readiness rollup; this is not approval evidence. |
| `reports/scenario/connectorhub-human-gate-validation-handoff-2026-06-24.json` | Pinned ordered validation handoff with the H07 row and dependency blockers. |
| `reports/scenario/connectorhub-human-gate-record-template-cs-ch-h07-2026-06-24.json` | Pinned blank H07 reviewer-record template; this is not approval evidence. |
| `reports/scenario/connectorhub-human-gate-validation-blank-cs-ch-h07-2026-06-24.json` | Pinned blocked validation envelope proving the blank template remains invalid until a human fills required evidence. |

Current recovery evidence-packet workflow:

This workflow is operator-preparation only. It records packet file hashes and redacted refs, but it does not record raw packet contents, collect human acceptance, unlock dependencies, or promote `CS-CH-H07` to `PASS`.

| Step | Command | Output role |
|---:|---|---|
| 1 | `cornerstone connector human-gate evidence-packet-contract --scenario CS-CH-H07 --json` | Required evidence manifest and redaction contract. |
| 2 | `cornerstone connector human-gate evidence-packet-file-contract --scenario CS-CH-H07 --json` | Required recovery packet file list and content expectations. |
| 3 | `cornerstone connector human-gate evidence-packet-scaffold --scenario CS-CH-H07 --packet-dir <h07-recovery-packet-dir> --json --write` | Blank local packet templates without overwriting existing evidence. |
| 4 | `cornerstone connector human-gate evidence-packet-validate --scenario CS-CH-H07 --packet-dir <h07-recovery-packet-dir> --json` | Hash-only packet validation envelope. |
| 5 | `cornerstone connector human-gate evidence-packet-record-draft --scenario CS-CH-H07 --packet-dir <h07-recovery-packet-dir> --json --record-output <reviewer-record-draft.json>` | Hash-only reviewer record draft; human decision fields remain human-owned. |
| 6 | `cornerstone connector human-gate validate-record --scenario CS-CH-H07 --record-file <filled-reviewer-record.json> --json --output <redacted-validation-envelope.json>` | Redacted structural validation envelope for the completed reviewer record. |

Boundary flags: `schema_version=cs.connector_human_gate_h07_evidence_packet_workflow.v1`, `claim_boundary=h07_recovery_packet_workflow_is_operator_handoff_not_human_acceptance`, `acceptance_sufficient=false`, `product_claim_allowed=false`, `pass_claim_allowed=false`, `dependency_unlock_allowed_by_workflow=false`, `human_acceptance_collected_by_workflow=false`, `raw_packet_file_contents_recorded_by_workflow=false`, and `packet_file_contents_persisted_by_workflow=false`.

## Reviewer Record Submission Checklist

The blank reviewer record is still invalid until a human fills the generated `record_template` with redacted evidence refs and runs the structural validator. No row below is approval evidence, product acceptance, or a `PASS` claim.

| Submission item | Template source | Required action | Validator guard |
|---|---|---|---|
| Blank reviewer template | `reports/scenario/connectorhub-human-gate-record-template-cs-ch-h07-2026-06-24.json` | Write a blank template with `cornerstone connector human-gate package --scenario CS-CH-H07 --json --record-template-output <reviewer-record-template.json>`, then fill it from the recovery exercise evidence packet. | The blank template is preparation data only; `blank_template_requires_human_evidence` keeps H07 `HUMAN_REQUIRED`. |
| Redacted validation envelope | `proposed_record_template.validation_output_command` | Submit the filled JSON with `cornerstone connector human-gate validate-record --scenario CS-CH-H07 --record-file <filled-json> --json --output <redacted-validation-envelope.json>`. | Redacted structural validation only; no raw record body, raw record path, decision value, senior-review finding text, or evidence-packet manifest values are persisted. |
| Required fields | `proposed_record_template.required_fields` | Fill `reviewer`, `review_timestamp`, `backup_manifest_ref`, `restore_log_ref`, `cursor_reconciliation_ref`, `replay_results_ref`, `audit_verification_ref`, `before_after_counts_hashes`, and `issues_or_exceptions`. | Missing fields keep the validator envelope blocked and keep final verdict `HUMAN_REQUIRED`. |
| Evidence packet manifest | `proposed_record_template.required_evidence_packet_manifest` | Attach one distinct redacted evidence ref for `Backup manifest and restore logs for production-like durable state.`, `Cursor reconciliation and replay results.`, `Before/after logical counts and hashes.`, and `Audit verification proving recovered state integrity.` | Each row must keep the matching required-evidence label and use only `redacted`, `public_safe`, or `no_sensitive_material` as `redaction_status`. |
| Senior review perspectives | `proposed_record_template.required_senior_review_perspectives` | Fill `product_value`, `domain_architecture`, `data_contract`, `reliability_observability`, `security_privacy`, and `testability_migration` findings. | Finding text is not persisted by the validator; missing perspectives remain structural issues. |
| Dependency refs | `proposed_record_template.dependency_human_gates` | Attach a structurally valid ACCEPT `connector_human_gate_record_validation:<id>` ref for `CS-CH-H04`. | REJECT records do not unlock dependents; missing dependency refs keep H07 blocked. |

## Scenario-First Execution Runbook

Run this gate after `CS-CH-H04` has an accepted or explicitly bounded security-topology review. This recovery exercise proves that ConnectorHub-related CornerStone state can be restored and replayed without losing immutable evidence, duplicating logical truth, broadening scope, exposing restricted content, or breaking audit continuity.

| Step | Operator action | Evidence to attach | Reject immediately if |
|---:|---|---|---|
| 1 | Freeze the recovery scope: tenant, owner, namespace, workspace, connector app ids, source refs, backup window, restore target, retention policy, and evidence directory. | Recovery scope manifest, backup policy reference, restore target reference, reviewer, timestamp. | Scope is ambiguous, restore target differs from the reviewed environment, or retention/legal constraints are unknown. |
| 2 | Capture a pre-backup baseline for artifacts, evidence bundles, delivery receipts, content versions, search/index state, connector cursors, quarantine, and audit events. | Baseline counts, selected hashes, cursor positions, quarantine ids, audit high-water mark. | Baseline cannot be reproduced, lacks scope fields, or includes cross-namespace/private payload content. |
| 3 | Execute or inspect the production-like backup covering Archive, Product state, Connector state, search/index state, quarantine, and audit/evidence manifest data. | Backup manifest, storage/object refs, DB dump or snapshot refs, search/index snapshot refs, audit/evidence manifest refs. | Any required state family is absent, backup refs are mutable/ambiguous, or secrets/raw provider payloads are exposed. |
| 4 | Restore into the controlled target and run post-restore integrity checks before replay. | Restore log, restored counts/hashes, schema/version compatibility notes, restore warnings. | Restore fails silently, drops connector/evidence/audit state, or changes retained/deleted content contrary to policy. |
| 5 | Reconcile connector cursors and pending acknowledgements after restore. | Cursor reconciliation transcript, pending ack/outbox state, source revision comparison, policy decision refs. | Cursors move ahead of durable state, regress without replay plan, or point outside the approved scope. |
| 6 | Replay pending or quarantined deliveries where applicable, preserving idempotency and original failure evidence. | Replay command transcript, replay result ids, dedupe/version lineage output, quarantine state after replay. | Replay creates duplicate active truth, erases quarantine history, mutates external providers, or acknowledges before commit. |
| 7 | Validate search/read-model and Evidence Bundle behavior against restored state. | Search/query transcript, Evidence Bundle sample, before/after logical count comparison, stale/unavailable warnings if applicable. | Search exposes cross-namespace data, loses expected evidence refs, or treats restored connector metadata as original truth by itself. |
| 8 | Verify audit continuity and tamper evidence across backup, restore, reconciliation, replay, and review. | Audit integrity report, event correlation ids, tamper check result, evidence manifest hash/checksum. | Audit gaps, mutable events, broken correlation, missing high-water mark, or unredacted secret/provider material appear. |

## Acceptance Evidence Packet

Attach a dated packet with these files or equivalent redacted artifacts:

| Artifact | Required contents |
|---|---|
| `recovery-scope.md` | Tenant/owner/namespace/workspace, connector apps, source refs, retention policy, backup window, restore target, reviewer, timestamp. |
| `pre-backup-baseline.json` | Scoped counts, selected hashes, cursor positions, quarantine ids, audit high-water mark, evidence manifest ref. |
| `backup-manifest.json` | Archive/Product/Connector/search/quarantine/audit state coverage, snapshot ids, object refs, schema versions, checksum or digest data. |
| `restore-log.txt` | Restore command or run reference, restored counts/hashes, schema compatibility, warnings, restore completion evidence. |
| `cursor-reconciliation.json` | Pre/post cursor positions, pending ack state, source revision comparison, reconciliation decision ids. |
| `replay-results.json` | Pending/quarantined delivery ids, replay outcomes, dedupe keys, version-lineage refs, retained quarantine evidence. |
| `read-model-validation.txt` | Search/query checks, Evidence Bundle sample, before/after logical count comparison, stale/unavailable warnings. |
| `audit-integrity-report.json` | Audit event ids, correlation ids, high-water mark, tamper check, evidence manifest hash/checksum. |
| `review-decision.md` | ACCEPT or REJECT, reviewer, timestamp, evidence packet path, issues/exceptions, release impact. |

## Redaction And Handling Rules

- Do not paste secrets, provider tokens, private keys, raw provider payloads, private database rows, or credential-bearing URLs into this template.
- Keep private restored content out of the template. Use redacted identifiers, hashes, counts, and scoped evidence refs that can be correlated across packet files.
- Deleted or restricted content must remain deleted/restricted after restore unless the reviewer records a policy-approved exception.
- If replay touches a live provider, reject this gate unless that provider interaction was separately authorized; this recovery gate must not create unapproved external mutations.

## Senior Review Perspectives

| Perspective | H07 acceptance question |
|---|---|
| Product value | Can CornerStone recover connected-source knowledge without hidden loss or duplicate logical truth? |
| Domain architecture | Do Archive, Connector, search, quarantine, cursor, and audit recovery boundaries remain coherent? |
| Data contract | Are backup manifest, restore log, cursor reconciliation, replay results, count/hash comparisons, and audit refs attached? |
| Reliability and observability | Do before/after counts and hashes plus audit verification prove recovery beyond operator narrative? |
| Security and privacy | Are cross-namespace restore leaks, restricted-content resurrection, and secret exposure absent? |
| Testability and migration | Are exercise gaps converted into durable runbook/tooling requirements before production operations readiness? |

## Decision Record

```text
Decision: ACCEPT / REJECT
Reviewer:
Review timestamp:
Backup manifest reference:
Restore log reference:
Evidence location:
Issues or exceptions:
```

## Evidence Checklist

| Required evidence | Human result | Notes |
|---|---|---|
| Backup manifest covers artifacts, evidence, connector state, search, quarantine, and audit. | PENDING | |
| Restore log is attached. | PENDING | |
| Connector cursor reconciliation evidence is attached. | PENDING | |
| Replay results for pending or quarantined deliveries are attached where applicable. | PENDING | |
| Before/after logical counts and hashes are attached. | PENDING | |
| Audit verification proves recovered state integrity. | PENDING | |
| No cross-namespace data, deleted restricted content, or secret exposure appears after restore. | PENDING | |

## Acceptance Statement

Accept only if recovery evidence proves no duplicate or lost logical state across artifacts, evidence, connector cursors, search, quarantine, and audit.

## Boundary

This review blocks production operations and recovery readiness. It does not alter local fixture ConnectorHub PASS claims.
