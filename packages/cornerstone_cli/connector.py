from __future__ import annotations

from copy import deepcopy
import fnmatch
import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from cornerstone_cli.runtime import LocalRuntimeStore, _write_json, action_preflight_binding_for_action, detect_unsafe_instructions


SUPPORTED_CONTRACT_SCHEMA = "cornerstone.connector_contract.v1"
SUPPORTED_DELIVERY_SCHEMA = "cornerstone.connector_projection_delivery.v1"
CONTRACT_RECORD_SCHEMA = "cs.connector_capability_contract.v1"
SETUP_RESULT_SCHEMA = "cs.connector_setup_result.v1"
SOURCE_POLICY_SCHEMA = "cs.connector_source_policy.v1"
DELIVERY_RECEIPT_SCHEMA = "cs.connector_delivery_receipt.v1"
PROJECTION_SNAPSHOT_SCHEMA = "cs.connector_projection_snapshot.v1"
EVIDENCE_LINK_SCHEMA = "cs.connector_evidence_link.v1"
ACK_OUTBOX_SCHEMA = "cs.connector_ack_outbox.v1"
ACK_RECONCILIATION_SCHEMA = "cs.connector_ack_reconciliation.v1"
DELIVERY_RETRY_STATE_SCHEMA = "cs.connector_delivery_retry_state.v1"
DELIVERY_QUARANTINE_SCHEMA = "cs.connector_delivery_quarantine.v1"
QUARANTINE_LIST_SCHEMA = "cs.connector_quarantine_list.v1"
DELIVERY_DEDUPE_SCHEMA = "cs.connector_delivery_dedupe_state.v1"
CONTENT_VERSION_SCHEMA = "cs.connector_content_version.v1"
CONTENT_CURRENT_SCHEMA = "cs.connector_content_current.v1"
CONTENT_LINEAGE_SCHEMA = "cs.connector_content_lineage.v1"
PROJECTION_POLICY_DECISION_SCHEMA = "cs.connector_projection_policy_decision.v1"
CONTENT_RESTRICTION_DECISION_SCHEMA = "cs.connector_content_restriction_decision.v1"
CONNECTOR_EVIDENCE_BUNDLE_LINK_SCHEMA = "cs.connector_evidence_bundle_link.v1"
RAW_ACCESS_REQUEST_SCHEMA = "cs.connector_raw_access_request.v1"
RAW_ACCESS_GRANT_SCHEMA = "cs.connector_raw_access_grant.v1"
RAW_ACCESS_EXPORT_SCHEMA = "cs.connector_raw_access_export.v1"
UNTRUSTED_CONTENT_REVIEW_SCHEMA = "cs.connector_untrusted_content_review.v1"
SYNC_SIGNAL_RECEIPT_SCHEMA = "cs.connector_sync_signal_receipt.v1"
SYNC_CURSOR_SCHEMA = "cs.connector_sync_cursor.v1"
SYNC_RECONCILIATION_SCHEMA = "cs.connector_sync_reconciliation.v1"
GITHUB_WRITE_GUARD_SCHEMA = "cs.connector_github_write_guard.v1"
PROVIDER_FAILURE_STATE_SCHEMA = "cs.connector_provider_failure_state.v1"
CAPTURE_PERMISSION_PROBE_SCHEMA = "cs.connector_capture_permission_probe.v1"
WATCH_SOURCE_CONSENT_SCHEMA = "cs.connector_watch_source_consent.v1"
CAPTURE_GUARD_DECISION_SCHEMA = "cs.connector_capture_guard_decision.v1"
ACTIVITY_SAMPLE_BATCH_SCHEMA = "cs.connector_activity_sample_batch.v1"
ACTIVITY_SESSIONIZATION_SCHEMA = "cs.connector_activity_sessionization.v1"
ACTIVITY_SESSION_PROJECTION_SCHEMA = "cs.activity_session_projection.v1"
CHROME_ACTIVE_TAB_INPUT_SCHEMA = "cs.chrome_active_tab_capture_payload.v1"
CHROME_ACTIVE_TAB_PERMISSION_EVENT_SCHEMA = "cs.connector_chrome_active_tab_permission_event.v1"
CHROME_ACTIVE_TAB_PAYLOAD_SCHEMA = "cs.connector_chrome_active_tab_payload.v1"
CHROME_ACTIVE_TAB_POLICY_DECISION_SCHEMA = "cs.connector_chrome_active_tab_policy_decision.v1"
CHROME_ACTIVE_TAB_CAPTURE_SUMMARY_SCHEMA = "cs.connector_chrome_active_tab_capture_summary.v1"
CHROME_AUTO_CAPTURE_CONFIG_INPUT_SCHEMA = "cs.chrome_auto_capture_config.v1"
CHROME_AUTO_CAPTURE_TRIGGER_INPUT_SCHEMA = "cs.chrome_auto_capture_trigger.v1"
CHROME_AUTO_CAPTURE_CONFIG_SCHEMA = "cs.connector_chrome_auto_capture_config.v1"
CHROME_AUTO_CAPTURE_TRIGGER_SCHEMA = "cs.connector_chrome_auto_capture_trigger.v1"
CHROME_AUTO_CAPTURE_POLICY_DECISION_SCHEMA = "cs.connector_chrome_auto_capture_policy_decision.v1"
CHROME_AUTO_CAPTURE_SUMMARY_SCHEMA = "cs.connector_chrome_auto_capture_summary.v1"
CHROME_SENSITIVE_PAGE_INPUT_SCHEMA = "cs.chrome_sensitive_page_policy_evaluation.v1"
CHROME_SENSITIVE_PAGE_POLICY_DECISION_SCHEMA = "cs.connector_chrome_sensitive_page_policy_decision.v1"
CHROME_SENSITIVE_PAGE_DEGRADED_PAYLOAD_SCHEMA = "cs.connector_chrome_sensitive_page_degraded_payload.v1"
CHROME_SENSITIVE_PAGE_HISTORY_ITEM_SCHEMA = "cs.connector_chrome_sensitive_page_history_item.v1"
CAPTURE_INBOX_ITEM_SCHEMA = "cs.capture_inbox_item.v1"
CAPTURE_LIFECYCLE_SEED_SCHEMA = "cs.capture_lifecycle_seed.v1"
CAPTURE_LIFECYCLE_SOURCE_STATE_SCHEMA = "cs.connector_capture_lifecycle_source_state.v1"
CAPTURE_LIFECYCLE_DECISION_SCHEMA = "cs.connector_capture_lifecycle_decision.v1"
CAPTURE_LIFECYCLE_EXPORT_SCHEMA = "cs.connector_capture_lifecycle_export.v1"
CAPTURE_LIFECYCLE_DELETION_RECEIPT_SCHEMA = "cs.connector_capture_lifecycle_deletion_receipt.v1"
CAPTURE_RESULT_REVIEW_SCHEMA = "cs.connector_capture_result_review.v1"
WATCH_RESULT_INPUT_SCHEMA = "cs.watch_result_fixture.v1"
WATCH_OBSERVATION_SCHEMA = "cs.watch_observation.v1"
WATCH_INFERENCE_SCHEMA = "cs.watch_inference.v1"
WATCH_RESULT_SCHEMA = "cs.watch_result.v1"
WATCH_RESULT_CORRECTION_SCHEMA = "cs.watch_result_correction.v1"
WATCH_RESULT_REVIEW_SCHEMA = "cs.watch_result_review.v1"
CONNECTOR_ACTION_PREFLIGHT_FIXTURE_SCHEMA = "cs.connector_action_preflight_fixture.v1"
CONNECTOR_ACTION_PREFLIGHT_SCHEMA = "cs.connector_action_preflight.v1"
CONNECTOR_ACTION_PREFLIGHT_REVIEW_SCHEMA = "cs.connector_action_preflight_review.v1"
WATCH_RULE_SCHEMA = "cs.watch_rule.v1"
WATCH_RULE_VERSION_SCHEMA = "cs.watch_rule_version.v1"
WATCH_RULE_POLICY_DECISION_SCHEMA = "cs.watch_rule_policy_decision.v1"
WATCH_RULE_EVALUATION_TRACE_SCHEMA = "cs.watch_rule_evaluation_trace.v1"
CONNECTOR_AUDIT_CORRELATION_SCHEMA = "cs.connector_audit_correlation.v1"
CONNECTOR_HUMAN_GATE_PACKAGE_SCHEMA = "cs.connector_human_gate_package.v1"
CONNECTOR_HUMAN_GATE_READINESS_REPORT_SCHEMA = "cs.connector_human_gate_readiness_report.v1"
CONNECTOR_HUMAN_GATE_NEXT_SCHEMA = "cs.connector_human_gate_next.v1"
CONNECTOR_HUMAN_GATE_RECORD_VALIDATION_SCHEMA = "cs.connector_human_gate_record_validation.v1"
CONNECTOR_HUMAN_GATE_RECORD_TEMPLATE_SCHEMA = "cs.connector_human_gate_record_template.v1"
CONNECTOR_HUMAN_GATE_REDACTION_GUIDANCE_SCHEMA = "cs.connector_human_gate_redaction_guidance.v1"
CONNECTOR_HUMAN_GATE_REVIEWER_CHECKLIST_SCHEMA = "cs.connector_human_gate_reviewer_checklist.v1"
CONNECTOR_HUMAN_GATE_DELIVERY_UNIT_PLAN_SCHEMA = "cs.connector_human_gate_delivery_unit_plan.v1"
CONNECTOR_HUMAN_GATE_VALIDATION_HANDOFF_SCHEMA = "cs.connector_human_gate_validation_handoff.v1"
CONNECTOR_HUMAN_GATE_FIELD_REF_CONTRACT_SCHEMA = "cs.connector_human_gate_field_ref_contract.v1"
CONNECTOR_HUMAN_GATE_FIELD_REF_CONTRACT_REPORT_SCHEMA = (
    "cs.connector_human_gate_field_ref_contract_report.v1"
)
CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_CONTRACT_SCHEMA = (
    "cs.connector_human_gate_evidence_packet_contract.v1"
)
CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_CONTRACT_REPORT_SCHEMA = (
    "cs.connector_human_gate_evidence_packet_contract_report.v1"
)
CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_FILE_CONTRACT_SCHEMA = (
    "cs.connector_human_gate_evidence_packet_file_contract.v1"
)
CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_FILE_CONTRACT_REPORT_SCHEMA = (
    "cs.connector_human_gate_evidence_packet_file_contract_report.v1"
)
CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_SCAFFOLD_SCHEMA = (
    "cs.connector_human_gate_evidence_packet_scaffold.v1"
)
CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_SCAFFOLD_REPORT_SCHEMA = (
    "cs.connector_human_gate_evidence_packet_scaffold_report.v1"
)
CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_VALIDATION_REPORT_SCHEMA = (
    "cs.connector_human_gate_evidence_packet_validation_report.v1"
)
CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_RECORD_DRAFT_REPORT_SCHEMA = (
    "cs.connector_human_gate_evidence_packet_record_draft_report.v1"
)
CONNECTOR_HUMAN_GATE_PREFLIGHT_BUNDLE_REPORT_SCHEMA = (
    "cs.connector_human_gate_preflight_bundle_report.v1"
)
CONNECTOR_HUMAN_GATE_REMAINING_EVIDENCE_SUMMARY_SCHEMA = (
    "cs.connector_human_gate_remaining_evidence_summary.v1"
)
CONNECTOR_HUMAN_GATE_H01_EVIDENCE_PACKET_WORKFLOW_SCHEMA = (
    "cs.connector_human_gate_h01_evidence_packet_workflow.v1"
)
CONNECTOR_HUMAN_GATE_H01_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY = (
    "h01_github_readonly_packet_workflow_is_operator_handoff_not_human_acceptance"
)
CONNECTOR_HUMAN_GATE_H02_EVIDENCE_PACKET_WORKFLOW_SCHEMA = (
    "cs.connector_human_gate_h02_evidence_packet_workflow.v1"
)
CONNECTOR_HUMAN_GATE_H02_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY = (
    "h02_macos_permission_packet_workflow_is_operator_handoff_not_human_acceptance"
)
CONNECTOR_HUMAN_GATE_H03_EVIDENCE_PACKET_WORKFLOW_SCHEMA = (
    "cs.connector_human_gate_h03_evidence_packet_workflow.v1"
)
CONNECTOR_HUMAN_GATE_H03_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY = (
    "h03_chrome_privacy_packet_workflow_is_operator_handoff_not_human_acceptance"
)
CONNECTOR_HUMAN_GATE_H05_EVIDENCE_PACKET_WORKFLOW_SCHEMA = (
    "cs.connector_human_gate_h05_evidence_packet_workflow.v1"
)
CONNECTOR_HUMAN_GATE_H05_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY = (
    "h05_live_action_packet_workflow_is_operator_handoff_not_human_acceptance"
)
CONNECTOR_HUMAN_GATE_H06_EVIDENCE_PACKET_WORKFLOW_SCHEMA = (
    "cs.connector_human_gate_h06_evidence_packet_workflow.v1"
)
CONNECTOR_HUMAN_GATE_H06_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY = (
    "h06_usability_trust_packet_workflow_is_operator_handoff_not_human_acceptance"
)
CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_WORKFLOW_SCHEMA = (
    "cs.connector_human_gate_h04_evidence_packet_workflow.v1"
)
CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY = (
    "h04_evidence_packet_workflow_is_operator_handoff_not_human_acceptance"
)
CONNECTOR_HUMAN_GATE_H07_EVIDENCE_PACKET_WORKFLOW_SCHEMA = (
    "cs.connector_human_gate_h07_evidence_packet_workflow.v1"
)
CONNECTOR_HUMAN_GATE_H07_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY = (
    "h07_recovery_packet_workflow_is_operator_handoff_not_human_acceptance"
)
CONNECTOR_HUMAN_GATE_H04_BASELINE_CLAIM_BOUNDARY = (
    "h04_local_baseline_snapshot_is_review_input_not_human_acceptance"
)
CONNECTOR_HUMAN_GATE_H04_BASELINE_PREFLIGHT_COMMAND_PLAN_SCHEMA = (
    "cs.connector_human_gate_local_baseline_preflight_command_plan.v1"
)
CONNECTOR_HUMAN_GATE_H04_BASELINE_PREFLIGHT_BUNDLE_SCHEMA = (
    "cs.connector_human_gate_local_baseline_preflight_bundle.v1"
)
CONNECTOR_HUMAN_GATE_H04_BASELINE_PREFLIGHT_CLAIM_BOUNDARY = (
    "h04_local_baseline_preflight_is_review_input_not_human_acceptance"
)
CONNECTOR_HUMAN_GATE_H04_BASELINE_PREFLIGHT_BUNDLE_CLAIM_BOUNDARY = (
    "h04_local_baseline_preflight_bundle_is_review_input_not_human_acceptance"
)
CONNECTOR_HUMAN_GATE_H04_FIELD_REF_ITEMS = [
    {
        "field": "environment_topology_ref",
        "accepted_container": "string",
        "accepted_ref_prefixes": ["topology:"],
    },
    {
        "field": "request_context_proof",
        "accepted_container": "string",
        "accepted_ref_prefixes": ["request_context:"],
    },
    {
        "field": "db_policy_transcripts",
        "accepted_container": "non_empty_string_list",
        "accepted_ref_prefixes": ["db_policy:"],
    },
    {
        "field": "network_egress_transcripts",
        "accepted_container": "non_empty_string_list",
        "accepted_ref_prefixes": ["egress:"],
    },
    {
        "field": "backup_restore_evidence",
        "accepted_container": "non_empty_string_list",
        "accepted_ref_prefixes": ["backup_restore:"],
    },
    {
        "field": "audit_integrity_report",
        "accepted_container": "string",
        "accepted_ref_prefixes": ["audit_integrity:"],
    },
    {
        "field": "evidence_manifest_ref",
        "accepted_container": "string_or_non_empty_string_list",
        "accepted_ref_prefixes": ["evidence_manifest:"],
    },
]
CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_ITEMS = [
    {
        "packet_file": "environment-topology.md",
        "required_contents": (
            "Scope identifiers, service versions, deployment or compose/task references, "
            "policy bundle version, DB revision, network topology, reviewer, timestamp."
        ),
    },
    {
        "packet_file": "request-context-trace.json",
        "required_contents": (
            "Trusted RequestContext fields, connector app/source ids, owner/namespace/workspace, "
            "evidence refs, audit refs."
        ),
    },
    {
        "packet_file": "postgres-rls-transcript.txt",
        "required_contents": (
            "Allowed and denied product-path cases, RLS policy refs, redacted row identifiers, "
            "zero cross-namespace read/write finding."
        ),
    },
    {
        "packet_file": "opa-policy-transcript.json",
        "required_contents": (
            "Allowed and denied decisions, bundle digest, input shape, decision ids, denial reasons."
        ),
    },
    {
        "packet_file": "egress-transcript.txt",
        "required_contents": (
            "ConnectorHub allowed path, Product/API denied path, worker denied path, "
            "tool/agent denied path, gateway/audit refs."
        ),
    },
    {
        "packet_file": "backup-restore-evidence.md",
        "required_contents": (
            "Backup id, restore command or run reference, restored checksum/sample proof, "
            "recovery limitations."
        ),
    },
    {
        "packet_file": "audit-integrity-report.json",
        "required_contents": (
            "Audit event ids, correlation ids, tamper/evidence manifest result, redacted exceptions."
        ),
    },
    {
        "packet_file": "review-decision.md",
        "required_contents": (
            "ACCEPT or REJECT, reviewer, timestamp, evidence packet path, exceptions, release impact."
        ),
    },
]
CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY = (
    "<h04-acceptance-packet-dir>"
)
CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_SCAFFOLD_COMMAND_COUNT = (
    len(CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_ITEMS) + 1
)
CONNECTOR_HUMAN_GATE_H01_EVIDENCE_PACKET_FILE_ITEMS = [
    {
        "packet_file": "github-scope.md",
        "required_contents": (
            "Tenant/owner/namespace/workspace, GitHub installation/account label, selected "
            "repositories, unselected repository count/labels, Source Policy version, reviewer, "
            "timestamp."
        ),
    },
    {
        "packet_file": "permission-snapshot.md",
        "required_contents": (
            "Redacted permission screenshot/export, granted scopes, selected-repository mode, "
            "explicit statement that write/admin scopes are absent."
        ),
    },
    {
        "packet_file": "read-call-ledger.json",
        "required_contents": (
            "Live read request ids, endpoints or operation labels, HTTP methods, selected "
            "repository refs, response metadata, zero mutation markers."
        ),
    },
    {
        "packet_file": "delivery-evidence.json",
        "required_contents": (
            "Delivery refs, Artifact/intake refs, Evidence Bundle refs where applicable, Source "
            "Policy refs, audit refs."
        ),
    },
    {
        "packet_file": "unselected-denial.txt",
        "required_contents": (
            "Denial transcript, policy decision ids, zero unselected Artifact/receipt/ack counts, "
            "no organization-wide fallback proof."
        ),
    },
    {
        "packet_file": "zero-write-proof.json",
        "required_contents": (
            "GitHub write guard output, provider-pack/contract/CLI/runtime scan result, call "
            "ledger github_write_calls=0."
        ),
    },
    {
        "packet_file": "redaction-audit-report.json",
        "required_contents": (
            "Token/header/key/raw-payload scan result, audit correlation ids, evidence manifest "
            "hash/checksum."
        ),
    },
    {
        "packet_file": "review-decision.md",
        "required_contents": (
            "ACCEPT or REJECT, reviewer, timestamp, evidence packet path, issues/exceptions, "
            "release impact."
        ),
    },
]
CONNECTOR_HUMAN_GATE_H01_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY = (
    "<h01-github-readonly-packet-dir>"
)
CONNECTOR_HUMAN_GATE_H01_EVIDENCE_PACKET_FILE_SCAFFOLD_COMMAND_COUNT = (
    len(CONNECTOR_HUMAN_GATE_H01_EVIDENCE_PACKET_FILE_ITEMS) + 1
)
CONNECTOR_HUMAN_GATE_H02_EVIDENCE_PACKET_FILE_ITEMS = [
    {
        "packet_file": "macos-review-scope.md",
        "required_contents": (
            "Device label, macOS build, app/build id, source id, workspace, retention "
            "expectation, evidence directory."
        ),
    },
    {
        "packet_file": "initial-disabled-state.json",
        "required_contents": (
            "Initial status transcript proving no capture before consent and platform permission."
        ),
    },
    {
        "packet_file": "permission-consent-record.md",
        "required_contents": (
            "macOS permission evidence, CornerStone consent evidence, permission timestamp, "
            "consent timestamp."
        ),
    },
    {
        "packet_file": "first-sample-record.json",
        "required_contents": (
            "Redacted first-sample metadata, evidence ref, audit ref, no unapproved raw "
            "private content."
        ),
    },
    {
        "packet_file": "pause-proof.json",
        "required_contents": (
            "Pause timestamp, status transcript, blocked or denied sample attempt, audit ref."
        ),
    },
    {
        "packet_file": "revoke-proof.json",
        "required_contents": (
            "Revoke timestamp, post-revoke status transcript, blocked post-revoke sample attempt, "
            "audit ref."
        ),
    },
    {
        "packet_file": "retention-export-delete-notes.md",
        "required_contents": (
            "Export, retention, deletion, or state-review notes aligned to local lifecycle semantics."
        ),
    },
    {
        "packet_file": "audit-redaction-report.json",
        "required_contents": (
            "Audit ids, manifest or checksum, redaction scan result, zero secret/private "
            "payload exposure."
        ),
    },
    {
        "packet_file": "review-decision.md",
        "required_contents": (
            "Reviewer decision, timestamp, accept/reject rationale, exceptions, follow-ups."
        ),
    },
]
CONNECTOR_HUMAN_GATE_H02_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY = (
    "<h02-macos-permission-packet-dir>"
)
CONNECTOR_HUMAN_GATE_H02_EVIDENCE_PACKET_FILE_SCAFFOLD_COMMAND_COUNT = (
    len(CONNECTOR_HUMAN_GATE_H02_EVIDENCE_PACKET_FILE_ITEMS) + 1
)
CONNECTOR_HUMAN_GATE_H03_EVIDENCE_PACKET_FILE_ITEMS = [
    {
        "packet_file": "chrome-review-scope.md",
        "required_contents": (
            "Chrome profile label, Chrome version, extension/build id, backend version, source id, "
            "workspace, approved domains/source packs, sensitive fixture list, evidence directory."
        ),
    },
    {
        "packet_file": "permission-baseline.md",
        "required_contents": (
            "Extension permission pages, host/optional permissions, first-use copy, explicit "
            "no-default-capture statement."
        ),
    },
    {
        "packet_file": "active-tab-capture-proof.json",
        "required_contents": (
            "User gesture, active tab identifier, preflight/confirmation state, bounded payload "
            "metadata, backend policy/audit refs, raw persistence counters."
        ),
    },
    {
        "packet_file": "allowlist-auto-capture-proof.json",
        "required_contents": (
            "Approved-domain capture, unapproved-domain skip, consent/config refs, trigger metadata, "
            "throttle/idempotency evidence, timeline refs."
        ),
    },
    {
        "packet_file": "sensitive-page-policy-proof.json",
        "required_contents": (
            "Synthetic sensitive classes, block/degrade decisions, backend revalidation result, "
            "user-facing reason, raw persistence/model-send counters."
        ),
    },
    {
        "packet_file": "pause-revoke-proof.json",
        "required_contents": (
            "Pause timestamp, paused capture attempt denial, revoke timestamp, post-revoke denial, "
            "stale-payload rejection, audit refs."
        ),
    },
    {
        "packet_file": "timeline-diagnostics.json",
        "required_contents": (
            "Captured/degraded/skipped/duplicate/blocked/paused/revoked entries, reason codes, "
            "evidence refs, audit refs, no private payloads."
        ),
    },
    {
        "packet_file": "retention-export-delete-notes.md",
        "required_contents": (
            "Scoped export or state-review notes, deletion/dismissal guidance, retained-audit explanation."
        ),
    },
    {
        "packet_file": "privacy-issue-log.md",
        "required_contents": (
            "Reviewer issue list, severity, scenario reference, accept/reject impact, follow-up owner."
        ),
    },
    {
        "packet_file": "audit-redaction-report.json",
        "required_contents": (
            "Evidence manifest or checksum, redaction scan result, zero raw secrets/private "
            "payload/browser-history exposure."
        ),
    },
    {
        "packet_file": "review-decision.md",
        "required_contents": (
            "Reviewer decision, timestamp, accept/reject rationale, exceptions, follow-ups."
        ),
    },
]
CONNECTOR_HUMAN_GATE_H03_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY = (
    "<h03-chrome-privacy-packet-dir>"
)
CONNECTOR_HUMAN_GATE_H03_EVIDENCE_PACKET_FILE_SCAFFOLD_COMMAND_COUNT = (
    len(CONNECTOR_HUMAN_GATE_H03_EVIDENCE_PACKET_FILE_ITEMS) + 1
)
CONNECTOR_HUMAN_GATE_H05_EVIDENCE_PACKET_FILE_ITEMS = [
    {
        "packet_file": "live-action-scope.md",
        "required_contents": (
            "Non-GitHub provider, account/workspace, reversible target, capability/action type, "
            "CornerStone workspace/namespace, reviewer/approver, rollback or compensation plan, "
            "evidence directory."
        ),
    },
    {
        "packet_file": "pre-execution-safety-envelope.json",
        "required_contents": (
            "Evidence Bundle, ActionCard dry-run, ConnectorHub preflight, policy decision, "
            "Source Policy, expected calls, predicted diff, risk, permissions, idempotency key, "
            "audit refs."
        ),
    },
    {
        "packet_file": "approval-record.md",
        "required_contents": (
            "Approver authority, timestamp, approved action digest, expected state delta, "
            "rollback/compensation acknowledgement."
        ),
    },
    {
        "packet_file": "execution-transcript.json",
        "required_contents": (
            "Governed WorkflowRun/ConnectorHub execution transcript, redacted request, Action Result, "
            "provider receipt, idempotency record, audit refs."
        ),
    },
    {
        "packet_file": "provider-state-delta.md",
        "required_contents": (
            "Before/after provider-state proof, exactly one intended external mutation, no unrelated "
            "state changes."
        ),
    },
    {
        "packet_file": "idempotency-replay-proof.json",
        "required_contents": (
            "Same-key replay result, duplicate side-effect count, conflict denial or reconciliation "
            "note, durable counts."
        ),
    },
    {
        "packet_file": "outcome-reingest-proof.json",
        "required_contents": (
            "Outcome Artifact/Evidence Bundle, connected outcome record, provider receipt linkage, "
            "audit chain, follow-up Claim/Mission state if any."
        ),
    },
    {
        "packet_file": "rollback-compensation-proof.md",
        "required_contents": (
            "Rollback/compensation execution or explicit not-needed rationale, provider receipt, "
            "final state proof, residual-risk note."
        ),
    },
    {
        "packet_file": "github-exclusion-proof.json",
        "required_contents": (
            "Provider is not GitHub, GitHub write actions/endpoints/calls are absent, release "
            "read-only invariant remains intact."
        ),
    },
    {
        "packet_file": "boundary-redaction-report.json",
        "required_contents": (
            "Direct-provider-bypass scan, credential/secret scan, redacted request/result review, "
            "evidence manifest or checksum."
        ),
    },
    {
        "packet_file": "review-decision.md",
        "required_contents": (
            "Reviewer decision, timestamp, accept/reject rationale, exceptions, follow-ups."
        ),
    },
]
CONNECTOR_HUMAN_GATE_H05_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY = (
    "<h05-live-action-packet-dir>"
)
CONNECTOR_HUMAN_GATE_H05_EVIDENCE_PACKET_FILE_SCAFFOLD_COMMAND_COUNT = (
    len(CONNECTOR_HUMAN_GATE_H05_EVIDENCE_PACKET_FILE_ITEMS) + 1
)
CONNECTOR_HUMAN_GATE_H06_EVIDENCE_PACKET_FILE_ITEMS = [
    {
        "packet_file": "study-scope.md",
        "required_contents": (
            "Participant profile class, fixture workspace, task script version, product build, "
            "surfaces reviewed, facilitator/observer, evidence directory, redaction plan."
        ),
    },
    {
        "packet_file": "participant-consent-redacted.md",
        "required_contents": (
            "Consent to participate and record, redacted participant identifier, privacy handling note."
        ),
    },
    {
        "packet_file": "fixture-workspace-seed.json",
        "required_contents": (
            "Scenario fixtures, connected-source states, Capture Inbox items, Watch Results, "
            "ActionCard/proposed-action state, no private/live provider data."
        ),
    },
    {
        "packet_file": "task-script.md",
        "required_contents": (
            "Connected Sources, Capture Inbox, Watch Result, privacy controls, action-boundary, "
            "and progressive-disclosure tasks."
        ),
    },
    {
        "packet_file": "timed-task-notes.md",
        "required_contents": (
            "Start/end times, completion status, assistance count, errors, hesitation points, "
            "facilitator-neutral prompts."
        ),
    },
    {
        "packet_file": "screenshots-or-recording-manifest.md",
        "required_contents": (
            "Recording/screenshot refs with redaction notes and timestamps for task-critical moments."
        ),
    },
    {
        "packet_file": "connected-sources-comprehension.json",
        "required_contents": (
            "Participant explanation of purpose, selected resources, readiness, policy, permissions, "
            "freshness, setup gaps, and safe next steps."
        ),
    },
    {
        "packet_file": "capture-inbox-watch-result-comprehension.json",
        "required_contents": (
            "Participant explanation of Observation, Inference, Evidence/Caveats, Proposed next step, "
            "source/freshness, save/dismiss/feedback."
        ),
    },
    {
        "packet_file": "privacy-control-comprehension.json",
        "required_contents": (
            "Participant explanation of pause/resume, revoke, retention, export, delete/disable, "
            "diagnostics, and surveillance concern handling."
        ),
    },
    {
        "packet_file": "action-boundary-comprehension.json",
        "required_contents": (
            "Participant explanation of evidence basis, risk label, approval need, expected consequence, "
            "provider feasibility, and no GitHub writes."
        ),
    },
    {
        "packet_file": "rubric-scores.json",
        "required_contents": (
            "Task completion, comprehension, trust, risk clarity, assistance, unacceptable-risk flags, "
            "reviewer threshold decision."
        ),
    },
    {
        "packet_file": "issue-log.md",
        "required_contents": (
            "Issue severity, scenario reference, observed evidence, recommended fix, release impact."
        ),
    },
    {
        "packet_file": "audit-redaction-report.json",
        "required_contents": (
            "Evidence manifest or checksum, redaction scan, zero private participant/provider payload exposure."
        ),
    },
    {
        "packet_file": "review-decision.md",
        "required_contents": (
            "Reviewer decision, timestamp, accept/reject rationale, exceptions, follow-ups."
        ),
    },
]
CONNECTOR_HUMAN_GATE_H06_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY = (
    "<h06-usability-trust-packet-dir>"
)
CONNECTOR_HUMAN_GATE_H06_EVIDENCE_PACKET_FILE_SCAFFOLD_COMMAND_COUNT = (
    len(CONNECTOR_HUMAN_GATE_H06_EVIDENCE_PACKET_FILE_ITEMS) + 1
)
CONNECTOR_HUMAN_GATE_H07_EVIDENCE_PACKET_FILE_ITEMS = [
    {
        "packet_file": "recovery-scope.md",
        "required_contents": (
            "Tenant/owner/namespace/workspace, connector apps, source refs, retention policy, "
            "backup window, restore target, reviewer, timestamp."
        ),
    },
    {
        "packet_file": "pre-backup-baseline.json",
        "required_contents": (
            "Scoped counts, selected hashes, cursor positions, quarantine ids, audit high-water mark, "
            "evidence manifest ref."
        ),
    },
    {
        "packet_file": "backup-manifest.json",
        "required_contents": (
            "Archive/Product/Connector/search/quarantine/audit state coverage, snapshot ids, "
            "object refs, schema versions, checksum or digest data."
        ),
    },
    {
        "packet_file": "restore-log.txt",
        "required_contents": (
            "Restore command or run reference, restored counts/hashes, schema compatibility, "
            "warnings, restore completion evidence."
        ),
    },
    {
        "packet_file": "cursor-reconciliation.json",
        "required_contents": (
            "Pre/post cursor positions, pending ack state, source revision comparison, "
            "reconciliation decision ids."
        ),
    },
    {
        "packet_file": "replay-results.json",
        "required_contents": (
            "Pending/quarantined delivery ids, replay outcomes, dedupe keys, version-lineage refs, "
            "retained quarantine evidence."
        ),
    },
    {
        "packet_file": "read-model-validation.txt",
        "required_contents": (
            "Search/query checks, Evidence Bundle sample, before/after logical count comparison, "
            "stale/unavailable warnings."
        ),
    },
    {
        "packet_file": "audit-integrity-report.json",
        "required_contents": (
            "Audit event ids, correlation ids, high-water mark, tamper check, evidence manifest "
            "hash/checksum."
        ),
    },
    {
        "packet_file": "review-decision.md",
        "required_contents": (
            "ACCEPT or REJECT, reviewer, timestamp, evidence packet path, issues/exceptions, "
            "release impact."
        ),
    },
]
CONNECTOR_HUMAN_GATE_H07_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY = (
    "<h07-recovery-packet-dir>"
)
CONNECTOR_HUMAN_GATE_H07_EVIDENCE_PACKET_FILE_SCAFFOLD_COMMAND_COUNT = (
    len(CONNECTOR_HUMAN_GATE_H07_EVIDENCE_PACKET_FILE_ITEMS) + 1
)
CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_FILE_ITEMS_BY_SCENARIO = {
    "CS-CH-H01": CONNECTOR_HUMAN_GATE_H01_EVIDENCE_PACKET_FILE_ITEMS,
    "CS-CH-H02": CONNECTOR_HUMAN_GATE_H02_EVIDENCE_PACKET_FILE_ITEMS,
    "CS-CH-H03": CONNECTOR_HUMAN_GATE_H03_EVIDENCE_PACKET_FILE_ITEMS,
    "CS-CH-H04": CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_ITEMS,
    "CS-CH-H05": CONNECTOR_HUMAN_GATE_H05_EVIDENCE_PACKET_FILE_ITEMS,
    "CS-CH-H06": CONNECTOR_HUMAN_GATE_H06_EVIDENCE_PACKET_FILE_ITEMS,
    "CS-CH-H07": CONNECTOR_HUMAN_GATE_H07_EVIDENCE_PACKET_FILE_ITEMS,
}
CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY_BY_SCENARIO = {
    "CS-CH-H01": CONNECTOR_HUMAN_GATE_H01_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY,
    "CS-CH-H02": CONNECTOR_HUMAN_GATE_H02_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY,
    "CS-CH-H03": CONNECTOR_HUMAN_GATE_H03_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY,
    "CS-CH-H04": CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY,
    "CS-CH-H05": CONNECTOR_HUMAN_GATE_H05_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY,
    "CS-CH-H06": CONNECTOR_HUMAN_GATE_H06_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY,
    "CS-CH-H07": CONNECTOR_HUMAN_GATE_H07_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY,
}
CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_WORKFLOW_SCHEMA_BY_SCENARIO = {
    "CS-CH-H01": CONNECTOR_HUMAN_GATE_H01_EVIDENCE_PACKET_WORKFLOW_SCHEMA,
    "CS-CH-H02": CONNECTOR_HUMAN_GATE_H02_EVIDENCE_PACKET_WORKFLOW_SCHEMA,
    "CS-CH-H03": CONNECTOR_HUMAN_GATE_H03_EVIDENCE_PACKET_WORKFLOW_SCHEMA,
    "CS-CH-H04": CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_WORKFLOW_SCHEMA,
    "CS-CH-H05": CONNECTOR_HUMAN_GATE_H05_EVIDENCE_PACKET_WORKFLOW_SCHEMA,
    "CS-CH-H06": CONNECTOR_HUMAN_GATE_H06_EVIDENCE_PACKET_WORKFLOW_SCHEMA,
    "CS-CH-H07": CONNECTOR_HUMAN_GATE_H07_EVIDENCE_PACKET_WORKFLOW_SCHEMA,
}
CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY_BY_SCENARIO = {
    "CS-CH-H01": CONNECTOR_HUMAN_GATE_H01_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY,
    "CS-CH-H02": CONNECTOR_HUMAN_GATE_H02_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY,
    "CS-CH-H03": CONNECTOR_HUMAN_GATE_H03_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY,
    "CS-CH-H04": CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY,
    "CS-CH-H05": CONNECTOR_HUMAN_GATE_H05_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY,
    "CS-CH-H06": CONNECTOR_HUMAN_GATE_H06_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY,
    "CS-CH-H07": CONNECTOR_HUMAN_GATE_H07_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY,
}
CONNECTOR_SENSITIVE_MARKER_POLICY_SCHEMA = "cs.connector_sensitive_marker_policy.v1"
CONNECTOR_SENSITIVE_MARKER_FINDING_FIELDS = ["marker_type", "fingerprint", "length"]
CONNECTOR_HUMAN_TEMPLATE_REQUIRED_TOKENS = [
    "## Review Target",
    "**Status:** PENDING HUMAN REVIEW",
    "## Current Machine-Readable Handoff Snapshot",
    "## Reviewer Record Submission Checklist",
    "## Acceptance Evidence Packet",
    "## Senior Review Perspectives",
    "Human result",
    "PENDING",
    "## Decision Record",
    "## Evidence Checklist",
    "## Boundary",
]
CONNECTOR_HUMAN_TEMPLATE_REQUIRED_PERSPECTIVE_LABELS = [
    "Product value",
    "Domain architecture",
    "Data contract",
    "Reliability and observability",
    "Security and privacy",
    "Testability and migration",
]
CONNECTOR_HUMAN_GATE_REQUIRED_PERSPECTIVE_ROLES = [
    "product_value",
    "domain_architecture",
    "data_contract",
    "reliability_observability",
    "security_privacy",
    "testability_migration",
]
CONNECTOR_HUMAN_GATE_ALLOWED_REDACTION_STATUSES = [
    "redacted",
    "public_safe",
    "no_sensitive_material",
]
CONNECTOR_SOURCE_REQUIREMENT_ORDER = [
    "ER-01",
    "ER-02",
    "ER-03",
    "ER-04",
    "ER-05",
    "ER-06",
    "ER-07",
    "ER-08",
    "ER-09",
    "IR-01",
    "IR-02",
    "IR-03",
    "IR-04",
    "IR-05",
    "IR-06",
    "IR-07",
    "IR-08",
    "IR-09",
    "IR-10",
    "IR-11",
    "IR-12",
    "IR-13",
    "IR-14",
    "IR-15",
    "IR-16",
    "IR-17",
    "IR-18",
]
CONNECTOR_HUMAN_GATE_SOURCE_REQUIREMENT_IDS = {
    "CS-CH-H01": ["ER-06", "IR-13"],
    "CS-CH-H02": ["ER-05", "IR-14"],
    "CS-CH-H03": ["ER-05", "IR-14"],
    "CS-CH-H04": ["IR-04", "IR-18"],
    "CS-CH-H05": ["IR-11", "IR-13", "IR-18"],
    "CS-CH-H06": ["IR-01", "IR-14", "IR-16"],
    "CS-CH-H07": ["IR-07", "IR-17", "IR-18"],
}
CONNECTOR_HUMAN_GATE_SOURCE_REQUIREMENT_CLAIM_BOUNDARY = (
    "human_gate_preparation_does_not_close_source_requirements"
)
CONNECTOR_HUMAN_GATE_REMAINING_EVIDENCE_CLAIM_BOUNDARY = (
    "remaining_human_evidence_summary_is_operator_input_not_acceptance"
)
CONNECTOR_HUMAN_GATE_COMPLETION_CLAIM_BOUNDARY = (
    "connectorhub_full_goal_requires_dated_accept_records_for_all_human_external_gates"
)


def connector_human_gate_completion_boundary() -> dict[str, Any]:
    return {
        "full_goal_completion_allowed": False,
        "goal_completion_claim_blocked": True,
        "completion_claim_boundary": CONNECTOR_HUMAN_GATE_COMPLETION_CLAIM_BOUNDARY,
    }


def connector_human_gate_source_requirement_ids(scenario_id: str) -> list[str]:
    return list(CONNECTOR_HUMAN_GATE_SOURCE_REQUIREMENT_IDS.get(scenario_id, []))


def connector_human_gate_unique_source_requirement_ids(rows: list[dict[str, Any]]) -> list[str]:
    requirement_ids = {
        requirement_id
        for row in rows
        for requirement_id in row.get("source_requirement_ids", [])
    }
    return sorted(
        requirement_ids,
        key=lambda requirement_id: CONNECTOR_SOURCE_REQUIREMENT_ORDER.index(requirement_id)
        if requirement_id in CONNECTOR_SOURCE_REQUIREMENT_ORDER
        else len(CONNECTOR_SOURCE_REQUIREMENT_ORDER),
    )


def connector_human_gate_remaining_evidence_summary(row: dict[str, Any]) -> dict[str, Any]:
    required_human_fields = list(row.get("required_human_fields") or [])
    required_evidence = list(row.get("required_evidence") or [])
    summary = {
        "schema_version": CONNECTOR_HUMAN_GATE_REMAINING_EVIDENCE_SUMMARY_SCHEMA,
        "scenario_id": row["scenario_id"],
        "required_human_fields": required_human_fields,
        "required_human_field_count": len(required_human_fields),
        "required_evidence": required_evidence,
        "required_evidence_count": len(required_evidence),
        "release_impact": row["release_impact"],
        "stop_or_reject_when": row["stop_or_reject_when"],
        "record_template_output_command": row["record_template_output_command"],
        "validate_record_output_command": row["record_validation_output_command"],
        "claim_boundary": CONNECTOR_HUMAN_GATE_REMAINING_EVIDENCE_CLAIM_BOUNDARY,
    }
    evidence_packet_workflow = connector_human_gate_evidence_packet_workflow_commands(
        str(row["scenario_id"])
    )
    if evidence_packet_workflow is not None:
        summary["evidence_packet_workflow"] = evidence_packet_workflow
        summary["evidence_packet_workflow_command_count"] = evidence_packet_workflow[
            "command_count"
        ]
        summary["evidence_packet_workflow_commands"] = evidence_packet_workflow["commands"]
        summary["evidence_packet_workflow_claim_boundary"] = evidence_packet_workflow[
            "claim_boundary"
        ]
        summary["dependency_unlock_allowed_by_evidence_packet_workflow"] = (
            evidence_packet_workflow["dependency_unlock_allowed_by_workflow"]
        )
        summary["human_acceptance_collected_by_evidence_packet_workflow"] = (
            evidence_packet_workflow["human_acceptance_collected_by_workflow"]
        )
        summary["raw_packet_file_contents_recorded_by_evidence_packet_workflow"] = (
            evidence_packet_workflow["raw_packet_file_contents_recorded_by_workflow"]
        )
        summary["packet_file_contents_persisted_by_evidence_packet_workflow"] = (
            evidence_packet_workflow["packet_file_contents_persisted_by_workflow"]
        )
    return summary

CONNECTOR_HUMAN_GATE_PACKAGE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "CS-CH-H01": {
        "title": "Human GitHub read-only rehearsal",
        "proof_boundary": {
            "local_fixture_rows_complete": True,
            "live_provider_read_verified": "HUMAN_REQUIRED",
            "live_provider_write_verified": "OUT_OF_SCOPE_READ_ONLY",
            "production_readiness_verified": "NOT_VERIFIED",
            "product_claim_allowed": False,
            "pass_claim_allowed_without_human_record": False,
        },
        "senior_review_perspectives": [
            {
                "role": "product_value",
                "decision": "Verify that live GitHub evidence supports one CornerStone connected-source workflow without exposing ConnectorHub as a second product.",
            },
            {
                "role": "domain_architecture",
                "decision": "Keep Product code behind ConnectorPort; GitHub permissions and provider clients stay inside the ConnectorHub boundary.",
            },
            {
                "role": "data_contract",
                "decision": "Require selected-repository source refs, Projection/Delivery refs, Artifact/Evidence refs, and audit refs for every accepted live item.",
            },
            {
                "role": "reliability_observability",
                "decision": "Capture a call ledger, delivery receipts, freshness/failure state, and replayable audit evidence before accepting live-read readiness.",
            },
            {
                "role": "security_privacy",
                "decision": "Reject the gate if any write permission, write endpoint, credential leak, unselected repository access, or raw provider payload exposure appears.",
            },
            {
                "role": "testability_migration",
                "decision": "Treat the human record as evidence that live provider semantics match the local CS-CH-015 through CS-CH-020 fixture contracts.",
            },
        ],
        "rehearsal_checklist": [
            "Confirm the GitHub installation uses read-only permissions only.",
            "Record the selected repositories and verify unselected repositories are absent from CornerStone outputs.",
            "Run one selected-repository read ingestion path and collect Delivery, Artifact/Evidence, and audit refs.",
            "Collect redacted provider/API call ledger and verify write calls equal zero.",
            "Run or attach the local zero-write guard evidence for the same source scope.",
            "Record ACCEPT or REJECT with issues, exceptions, and release impact.",
        ],
        "required_human_record": {
            "decision_values": ["ACCEPT", "REJECT"],
            "required_fields": [
                "reviewer",
                "review_timestamp",
                "github_app_installation_id_redacted",
                "selected_repositories",
                "permission_snapshot",
                "call_ledger",
                "delivery_refs",
                "audit_refs",
                "zero_write_proof",
                "issues_or_exceptions",
            ],
            "required_evidence": [
                "Redacted GitHub App or equivalent least-privilege installation permission snapshot.",
                "Selected repository list and explicit confirmation that no unselected repositories were ingested.",
                "Read-only call ledger showing zero write-capable HTTP methods or endpoints.",
                "Connector Delivery, Artifact/Evidence, and audit refs produced from selected repositories.",
                "Independent zero-write proof covering permissions, declared actions, routes, UI/CLI, and observed calls.",
            ],
        },
        "non_mutation_evidence": {"github_write_calls_by_package": 0},
        "release_impact": "Blocks live GitHub read-only readiness. Does not block local fixture ConnectorHub adoption proof.",
    },
    "CS-CH-H02": {
        "title": "Human macOS permission walkthrough",
        "proof_boundary": {
            "local_fixture_rows_complete": True,
            "local_physical_device_behavior_verified": "HUMAN_REQUIRED",
            "live_provider_read_verified": "NOT_APPLICABLE",
            "production_readiness_verified": "NOT_VERIFIED",
            "product_claim_allowed": False,
            "pass_claim_allowed_without_human_record": False,
        },
        "senior_review_perspectives": [
            {"role": "product_value", "decision": "Confirm that capture earns trust because users can see, pause, revoke, and understand observation state."},
            {"role": "domain_architecture", "decision": "Keep platform permission state as ConnectorHub input evidence, not as hidden Product authority."},
            {"role": "data_contract", "decision": "Require consent, permission, source state, sample, lifecycle decision, and audit refs for the walkthrough."},
            {"role": "reliability_observability", "decision": "Record timestamps for grant, first sample, pause, revoke, and blocked post-revoke sample attempts."},
            {"role": "security_privacy", "decision": "Reject if capture starts without explicit consent or continues after pause or revoke."},
            {"role": "testability_migration", "decision": "Compare physical-device behavior against CS-CH-021, CS-CH-022, and CS-CH-027 local fixture contracts."},
        ],
        "rehearsal_checklist": [
            "Use a signed or locally approved build on a supported Mac.",
            "Record visible consent and platform permission grant.",
            "Capture first bounded activity sample and associated audit refs.",
            "Pause capture and verify no unexpected observation continues.",
            "Revoke permission and verify later sample attempts remain blocked.",
            "Record ACCEPT or REJECT with issues, exceptions, and release impact.",
        ],
        "required_human_record": {
            "decision_values": ["ACCEPT", "REJECT"],
            "required_fields": [
                "reviewer",
                "review_timestamp",
                "device_os_version_redacted",
                "consent_record",
                "permission_state_snapshot",
                "first_sample_ref",
                "pause_revoke_timestamps",
                "screenshots_or_recording_ref",
                "audit_refs",
                "issues_or_exceptions",
            ],
            "required_evidence": [
                "Recording or screenshots of consent, permission grant, pause, and revoke.",
                "Status transcript showing permission and capture state transitions.",
                "First bounded sample reference with owner/workspace scope.",
                "Proof that pause and revoke stop capture or block sample attempts.",
            ],
        },
        "release_impact": "Blocks physical-device Watch and capture readiness. Does not block local fixture capture contract proof.",
    },
    "CS-CH-H03": {
        "title": "Human Chrome privacy review",
        "proof_boundary": {
            "local_fixture_rows_complete": True,
            "real_browser_privacy_accepted": "HUMAN_REQUIRED",
            "live_provider_read_verified": "NOT_APPLICABLE",
            "production_readiness_verified": "NOT_VERIFIED",
            "product_claim_allowed": False,
            "pass_claim_allowed_without_human_record": False,
        },
        "senior_review_perspectives": [
            {"role": "product_value", "decision": "Confirm the browser capture experience is understandable without exposing internal ConnectorHub mechanics."},
            {"role": "domain_architecture", "decision": "Keep extension input behind ConnectorPort and backend policy validation."},
            {"role": "data_contract", "decision": "Require active-tab payload, allowlist config, sensitive-page decision, lifecycle decision, and audit refs."},
            {"role": "reliability_observability", "decision": "Record a timeline that distinguishes manual capture, auto capture, block/degrade, pause, and revoke."},
            {"role": "security_privacy", "decision": "Reject if sensitive pages leak, capture happens outside allowlist, or users cannot understand permissions."},
            {"role": "testability_migration", "decision": "Compare the review against CS-CH-024, CS-CH-025, CS-CH-026, and CS-CH-027 fixture contracts."},
        ],
        "rehearsal_checklist": [
            "Install the Chrome extension and local backend in a real browser profile.",
            "Review the permission pages and first-use copy.",
            "Run explicit active-tab capture and collect policy/audit refs.",
            "Run allowlist auto-capture and verify blocked outside-allowlist behavior.",
            "Open a sensitive target and verify block or degrade behavior.",
            "Pause and revoke capture, then record ACCEPT or REJECT.",
        ],
        "required_human_record": {
            "decision_values": ["ACCEPT", "REJECT"],
            "required_fields": [
                "reviewer",
                "review_timestamp",
                "browser_profile_redacted",
                "extension_version",
                "permission_pages_or_recording_ref",
                "active_tab_capture_ref",
                "allowlist_auto_capture_ref",
                "sensitive_block_ref",
                "pause_revoke_ref",
                "audit_refs",
                "accept_or_reject_note",
                "issues_or_exceptions",
            ],
            "required_evidence": [
                "Recording or screenshots of permission UX and capture controls.",
                "Timeline covering active-tab capture, allowlist auto-capture, sensitive-page handling, pause, and revoke.",
                "Policy and audit refs for allowed, blocked, and degraded browser capture decisions.",
                "Human privacy acceptance or rejection note.",
            ],
        },
        "release_impact": "Blocks Chrome capture and browser privacy acceptance. Does not block backend fixture policy proof.",
    },
    "CS-CH-H04": {
        "title": "Human production-like VS2 integrated security proof",
        "proof_boundary": {
            "local_fixture_rows_complete": True,
            "production_like_request_context_verified": "HUMAN_REQUIRED",
            "production_tenancy_policy_egress_verified": "HUMAN_REQUIRED",
            "production_readiness_verified": "NOT_VERIFIED",
            "product_claim_allowed": False,
            "pass_claim_allowed_without_human_record": False,
        },
        "senior_review_perspectives": [
            {"role": "product_value", "decision": "Confirm connected-source release claims do not outrun tenant, policy, egress, and recovery evidence."},
            {"role": "domain_architecture", "decision": "Verify Product, Archive, and Connector boundaries hold under real RequestContext and deployment topology."},
            {"role": "data_contract", "decision": "Require scenario-specific DB, policy, network, backup, restore, and audit transcripts."},
            {"role": "reliability_observability", "decision": "Treat generated VS2 reports as inputs and require direct report, transcript, and audit integrity review."},
            {"role": "security_privacy", "decision": "Reject if RLS, OPA, egress deny, secret custody, or audit integrity is simulated where production-like proof is required."},
            {"role": "testability_migration", "decision": "Keep this gate separate from local VS2 proof and from ConnectorHub fixture PASS claims."},
        ],
        "rehearsal_checklist": [
            "Confirm integrated topology has trusted RequestContext, PostgreSQL/RLS, OPA, network controls, and backup/restore path.",
            "Run the corrected integrated security and tenancy scenario suite.",
            "Collect scenario-specific DB policy transcripts.",
            "Collect OPA and egress-control transcripts.",
            "Collect backup, restore, audit integrity, and evidence manifest reports.",
            "Record ACCEPT or REJECT with findings and release impact.",
        ],
        "required_human_record": {
            "decision_values": ["ACCEPT", "REJECT"],
            "required_fields": [
                "reviewer",
                "review_timestamp",
                "environment_topology_ref",
                "request_context_proof",
                "db_policy_transcripts",
                "network_egress_transcripts",
                "backup_restore_evidence",
                "audit_integrity_report",
                "evidence_manifest_ref",
                "findings_or_exceptions",
            ],
            "required_evidence": [
                "Production-like topology description and trusted RequestContext proof.",
                "Scenario-specific PostgreSQL/RLS and OPA transcripts.",
                "Network default-deny and governed-egress transcripts.",
                "Backup/restore, evidence manifest, and audit integrity reports.",
            ],
        },
        "release_impact": "Blocks production connected-source and security readiness. Local fixture and current local VS2 proofs remain separate.",
    },
    "CS-CH-H05": {
        "title": "Human live non-GitHub Action execution",
        "proof_boundary": {
            "local_fixture_rows_complete": True,
            "live_external_mutation_verified": "HUMAN_REQUIRED",
            "github_actions_excluded": True,
            "production_readiness_verified": "NOT_VERIFIED",
            "product_claim_allowed": False,
            "pass_claim_allowed_without_human_record": False,
        },
        "senior_review_perspectives": [
            {"role": "product_value", "decision": "Confirm a governed live Action creates useful outcome evidence without normalizing autonomous external mutation."},
            {"role": "domain_architecture", "decision": "Keep execution behind declared actions, policy, approval, idempotency, and ConnectorHub-mediated provider access."},
            {"role": "data_contract", "decision": "Require ActionCard, policy decision, approval, result receipt, outcome evidence, idempotency, and audit refs."},
            {"role": "reliability_observability", "decision": "Require provider receipt/state evidence and a rollback or compensation plan for the reversible target."},
            {"role": "security_privacy", "decision": "Reject if GitHub is used, approval is missing, the provider target is not reversible, or credentials leak."},
            {"role": "testability_migration", "decision": "Compare live execution semantics against CS-CH-029 through CS-CH-033 fixture contracts."},
        ],
        "rehearsal_checklist": [
            "Choose a separately approved non-GitHub provider and reversible test target.",
            "Record the rollback or compensation plan before execution.",
            "Capture human approval for the declared Action and exact target.",
            "Execute once and collect redacted request/result plus provider receipt/state evidence.",
            "Verify idempotency and re-ingested outcome evidence.",
            "Record ACCEPT or REJECT with issues, exceptions, and release impact.",
        ],
        "required_human_record": {
            "decision_values": ["ACCEPT", "REJECT"],
            "required_fields": [
                "reviewer",
                "review_timestamp",
                "approved_provider",
                "reversible_test_target",
                "rollback_or_compensation_plan",
                "approval_ref",
                "redacted_request_result",
                "provider_receipt",
                "idempotency_evidence",
                "audit_refs",
                "issues_or_exceptions",
            ],
            "required_evidence": [
                "Non-GitHub provider and reversible target approval.",
                "Pre-execution rollback or compensation plan.",
                "Redacted request/result and provider receipt/state evidence.",
                "Idempotency and audit refs proving exactly-once execution.",
            ],
        },
        "non_mutation_evidence": {"github_actions_executed_by_package": 0},
        "release_impact": "Blocks live Action readiness. GitHub remains excluded from this human mutation gate.",
    },
    "CS-CH-H06": {
        "title": "Human connected-source usability trust study",
        "proof_boundary": {
            "local_fixture_rows_complete": True,
            "human_ux_privacy_accepted": "HUMAN_REQUIRED",
            "live_provider_read_verified": "NOT_APPLICABLE",
            "production_readiness_verified": "NOT_VERIFIED",
            "product_claim_allowed": False,
            "pass_claim_allowed_without_human_record": False,
        },
        "senior_review_perspectives": [
            {"role": "product_value", "decision": "Confirm users can complete Connected Sources and Capture Inbox tasks as one CornerStone product."},
            {"role": "domain_architecture", "decision": "Ensure setup, evidence, policy, and audit details are progressively disclosed instead of connector-admin-first."},
            {"role": "data_contract", "decision": "Require task script, participant profile, timed notes, issue list, and acceptance decision refs."},
            {"role": "reliability_observability", "decision": "Capture enough trace material to reproduce confusion, trust gaps, and recovery paths."},
            {"role": "security_privacy", "decision": "Reject if users misunderstand capture scope, approval boundaries, raw access, or external action risk."},
            {"role": "testability_migration", "decision": "Use the study to decide whether ConnectorHub concepts are adoptable in CornerStone UI language."},
        ],
        "rehearsal_checklist": [
            "Prepare a task script and fixture workspace.",
            "Run Connected Sources setup and evidence review tasks with a representative user.",
            "Run Capture Inbox review and trust-boundary tasks.",
            "Record timed notes, screenshots or recording, and issue severity.",
            "Score comprehension, trust, task completion, and unacceptable-risk findings.",
            "Record ACCEPT or REJECT with release impact.",
        ],
        "required_human_record": {
            "decision_values": ["ACCEPT", "REJECT"],
            "required_fields": [
                "reviewer",
                "review_timestamp",
                "participant_profile_redacted",
                "task_script_ref",
                "fixture_workspace_ref",
                "timed_task_notes",
                "screenshots_or_recording_ref",
                "scoring_rubric",
                "acceptance_decision",
                "issues_or_exceptions",
            ],
            "required_evidence": [
                "Task script and fixture workspace used for the study.",
                "Timed task notes and recording or screenshots.",
                "Scoring rubric for comprehension, trust, completion, and risk.",
                "Acceptance decision with issue list and release impact.",
            ],
        },
        "release_impact": "Blocks human UX and trust acceptance. Does not block local CLI/data-contract proof.",
    },
    "CS-CH-H07": {
        "title": "Human recovery exercise",
        "proof_boundary": {
            "local_fixture_rows_complete": True,
            "production_like_recovery_verified": "HUMAN_REQUIRED",
            "production_readiness_verified": "NOT_VERIFIED",
            "product_claim_allowed": False,
            "pass_claim_allowed_without_human_record": False,
        },
        "senior_review_perspectives": [
            {"role": "product_value", "decision": "Confirm CornerStone can recover connected-source knowledge without hidden loss or duplicate logical truth."},
            {"role": "domain_architecture", "decision": "Verify Archive, Connector, search, quarantine, cursor, and audit recovery boundaries remain coherent."},
            {"role": "data_contract", "decision": "Require backup manifest, restore log, cursor reconciliation, replay results, count/hash comparisons, and audit refs."},
            {"role": "reliability_observability", "decision": "Use before/after counts and hashes plus audit verification to prove recovery, not operator narrative alone."},
            {"role": "security_privacy", "decision": "Reject if restore leaks cross-namespace data, resurrects deleted restricted content, or exposes secrets."},
            {"role": "testability_migration", "decision": "Convert exercise gaps into durable runbook/tooling requirements before production operations readiness."},
        ],
        "rehearsal_checklist": [
            "Prepare a production-like backup manifest and restore target.",
            "Restore artifacts, evidence, connector state, search indexes, quarantine, and audit state.",
            "Reconcile connector cursors and replay pending or quarantined deliveries where appropriate.",
            "Compare before/after counts and hashes for logical state.",
            "Run audit verification and capture recovery logs.",
            "Record ACCEPT or REJECT with issues, exceptions, and release impact.",
        ],
        "required_human_record": {
            "decision_values": ["ACCEPT", "REJECT"],
            "required_fields": [
                "reviewer",
                "review_timestamp",
                "backup_manifest_ref",
                "restore_log_ref",
                "cursor_reconciliation_ref",
                "replay_results_ref",
                "audit_verification_ref",
                "before_after_counts_hashes",
                "issues_or_exceptions",
            ],
            "required_evidence": [
                "Backup manifest and restore logs for production-like durable state.",
                "Cursor reconciliation and replay results.",
                "Before/after logical counts and hashes.",
                "Audit verification proving recovered state integrity.",
            ],
        },
        "release_impact": "Blocks production operations and recovery readiness. Does not block local fixture ConnectorHub adoption proof.",
    },
}

CONNECTOR_HUMAN_GATE_EXECUTION_QUEUE: list[dict[str, Any]] = [
    {
        "order": 1,
        "scenario_id": "CS-CH-H04",
        "gate_category": "production_like_security",
        "depends_on": [],
        "primary_operator_action": "Run the production-like VS2 integrated security rehearsal in a controlled namespace.",
        "required_proof_package": "RequestContext trace, PostgreSQL/RLS evidence, OPA decision logs, egress transcript, backup/restore evidence, and audit refs.",
        "stop_or_reject_when": "Any cross-namespace read, unexpected egress, missing audit ref, unredacted secret, or unverifiable restore appears.",
    },
    {
        "order": 2,
        "scenario_id": "CS-CH-H07",
        "gate_category": "production_like_recovery",
        "depends_on": ["CS-CH-H04"],
        "primary_operator_action": "Exercise backup, restore, replay, cursor reconciliation, and audit durability after the security rehearsal.",
        "required_proof_package": "Backup logs, restore logs, reconciled cursor state, replay result, and immutable audit verification.",
        "stop_or_reject_when": "Replay changes external state unexpectedly, cursors diverge, audit continuity is broken, or restore evidence is incomplete.",
    },
    {
        "order": 3,
        "scenario_id": "CS-CH-H01",
        "gate_category": "live_readonly_provider",
        "depends_on": ["CS-CH-H04", "CS-CH-H07"],
        "primary_operator_action": "Rehearse live GitHub read-only connector access against selected repositories.",
        "required_proof_package": "Redacted permission selection, selected repository list, read-only call ledger, delivery artifacts, audit refs, and zero-write proof.",
        "stop_or_reject_when": "Any write-capable scope is granted, a mutation endpoint is called, repository scope is wider than approved, or audit refs are missing.",
    },
    {
        "order": 4,
        "scenario_id": "CS-CH-H02",
        "gate_category": "physical_device_capture",
        "depends_on": ["CS-CH-H04", "CS-CH-H07"],
        "primary_operator_action": "Walk through physical macOS permission grant, first sample capture, pause, and revoke.",
        "required_proof_package": "Recording, screenshots, status transcript, first sample, pause timestamp, revoke timestamp, and operator notes.",
        "stop_or_reject_when": "Capture starts before consent, pause/revoke is ineffective, status is misleading, or sensitive data is exposed in proof artifacts.",
    },
    {
        "order": 5,
        "scenario_id": "CS-CH-H03",
        "gate_category": "browser_privacy",
        "depends_on": ["CS-CH-H02"],
        "primary_operator_action": "Review real Chrome permission and privacy behavior.",
        "required_proof_package": "Recording, screenshots, browser permission pages, event timeline, issue log, and accept/reject note.",
        "stop_or_reject_when": "Browser state is captured without clear consent, denied permission is bypassed, degraded mode is unclear, or redaction is insufficient.",
    },
    {
        "order": 6,
        "scenario_id": "CS-CH-H05",
        "gate_category": "live_non_github_action",
        "depends_on": ["CS-CH-H04", "CS-CH-H07"],
        "primary_operator_action": "Execute one approved live non-GitHub Action with an explicit human approval record.",
        "required_proof_package": "Approval record, redacted request/result, provider receipt, state delta, idempotency record, audit refs, and GitHub exclusion proof.",
        "stop_or_reject_when": "Approval is ambiguous, provider receipt is missing, idempotency cannot be demonstrated, or the action uses GitHub instead of the selected non-GitHub provider.",
    },
    {
        "order": 7,
        "scenario_id": "CS-CH-H06",
        "gate_category": "human_usability_trust",
        "depends_on": ["CS-CH-H01", "CS-CH-H02", "CS-CH-H03", "CS-CH-H04", "CS-CH-H05", "CS-CH-H07"],
        "primary_operator_action": "Run a connected-source usability and trust study after the safety gates have evidence.",
        "required_proof_package": "Timed task notes, screenshots, recording, scoring rubric, trust/risk comprehension notes, and acceptance decision.",
        "stop_or_reject_when": "Operators cannot explain source scope, evidence basis, risk labels, or action consequences without assistance.",
    },
]
CONNECTOR_HUMAN_GATE_QUEUE_BY_SCENARIO = {
    item["scenario_id"]: item for item in CONNECTOR_HUMAN_GATE_EXECUTION_QUEUE
}

CONNECTOR_AUDIT_REQUIRED_EVENT_FAMILIES: dict[str, tuple[str, ...]] = {
    "setup": ("connector.contract.validated", "connector.setup.planned"),
    "policy": (
        "connector.source_policy.confirmed",
        "connector.source_policy.enforced",
        "connector.source_policy.broadening_denied",
    ),
    "delivery": (
        "connector.delivery.archived",
        "connector.delivery.ack_outbox.created",
        "connector.delivery.acknowledged",
        "connector.delivery.deduplicated",
    ),
    "evidence": (
        "connector.evidence_bundle.assembled",
        "connector.evidence_bundle.denied",
        "connector.raw_access.denied",
        "connector.raw_access.granted",
        "connector.raw_access.read",
        "connector.raw_access.revoked",
        "connector.raw_access.metadata_exported",
    ),
    "retry": ("connector.delivery.retry_scheduled", "connector.delivery.retry_resolved"),
    "quarantine": (
        "connector.delivery.quarantined",
        "connector.content_restriction.quarantined",
        "connector.quarantine.replay_requested",
    ),
    "action": (
        "connector.action_preflight.completed",
        "connector.direct_write.denied",
        "connector.action_trace.recorded",
    ),
    "credential": (
        "connector.credential.status",
        "connector.credential.rotate",
        "connector.credential.revoke",
        "connector.credential_boundary.checked",
    ),
}

CONNECTOR_AUDIT_FORBIDDEN_DETAIL_KEYS = {
    "raw_provider_payload",
    "raw_provider_response",
    "raw_payload",
    "raw_secret",
    "raw_secret_value",
    "raw_access_handle",
    "access_token",
    "authorization",
    "auth_header",
    "private_key",
}

CONNECTOR_AUDIT_BOOLEAN_LEAK_KEYS = {
    "provider_tokens_exposed",
    "provider_clients_exposed",
    "raw_local_paths_exposed",
    "direct_api_handles_exposed",
    "raw_content_persisted",
    "raw_provider_payload_persisted",
    "raw_provider_payload_stored_in_product_state",
    "credentials_exposed_to_agent",
    "credentials_exposed_to_product_output",
    "credential_values_exposed",
}
DELIVERY_ARTIFACT_MEDIA_TYPE = "application/vnd.cornerstone.connector.projection+json"
RAW_ACCESS_ALLOWED_MODE = "temporary_scoped"
RAW_ACCESS_CLASSIFICATIONS = {"internal", "confidential", "restricted"}

CONTENT_TEXT_FIELDS = {
    "body_markdown_excerpt",
    "content_excerpt",
    "diff_excerpt",
    "markdown_excerpt",
    "summary_excerpt",
}
CONTENT_METADATA_FIELDS = {
    "content_hash",
    "mime_type",
    "path",
    "ref",
    "size_bytes",
    "source_revision",
    "source_url_hash",
    "updated_at",
}
GENERATED_PATH_PATTERNS = [
    "build/**",
    "dist/**",
    "node_modules/**",
    "vendor/**",
    "**/*.lock",
    "**/*.min.js",
    "**/generated/**",
]
TEXT_MIME_TYPES = {
    "application/json",
    "application/markdown",
    "application/vnd.github.diff",
    "text/markdown",
    "text/plain",
}
SENSITIVE_MARKER_PATTERNS = [
    (
        "github_token_like",
        re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{12,}\b"),
    ),
    (
        "aws_access_key_like",
        re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    ),
    (
        "private_key_block",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
    ),
    (
        "scenario_sensitive_canary",
        re.compile(r"\bCS_CH_018_[A-Z0-9_]*(?:SECRET|PRIVATE_KEY)[A-Z0-9_]*\b"),
    ),
]
CONNECTOR_SENSITIVE_MARKER_TYPES = [marker_type for marker_type, _ in SENSITIVE_MARKER_PATTERNS]

SCOPE_FIELDS = ["tenant_id", "owner_id", "namespace_id", "workspace_id"]
FORBIDDEN_PROVIDER_INTERNAL_KEYS = {
    "provider_token",
    "provider_client",
    "raw_local_path",
    "direct_api_handle",
    "auth_header",
    "private_key",
    "secret",
    "token",
}
GITHUB_WRITE_HTTP_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
GITHUB_WRITE_FORBIDDEN_TERMS = (
    "write",
    "mutation",
    "mutate",
    "comment",
    "label",
    "merge",
    "push",
    "issue.create",
    "issue.update",
    "issue.close",
    "pull_request.merge",
    "release.create",
    "release.publish",
    "branch.create",
    "branch.delete",
    "file.write",
    "contents.write",
    "settings.update",
    "workflow.dispatch",
    "repository.admin",
    "repository.write",
)
GITHUB_WRITE_ENDPOINT_PATTERNS = (
    "/issues",
    "/issues/",
    "/comments",
    "/labels",
    "/contents/",
    "/pulls/",
    "/merge",
    "/git/refs",
    "/releases",
    "/dispatches",
    "/hooks",
    "/collaborators",
)
GITHUB_WRITE_FORBIDDEN_CLI_COMMANDS = (
    "github-comment",
    "github-label",
    "github-merge",
    "github-push",
    "github-issue-create",
    "github-issue-update",
    "github-file-write",
    "github-settings-update",
    "github-release-create",
    "source-control-write",
)
GITHUB_WRITE_DIRECT_ATTEMPTS = [
    {
        "operation": "issue.create",
        "method": "POST",
        "endpoint": "/repos/owner/project-alpha/issues",
        "target": "github:repo:owner/project-alpha:issues",
    },
    {
        "operation": "issue.comment",
        "method": "POST",
        "endpoint": "/repos/owner/project-alpha/issues/1001/comments",
        "target": "github:repo:owner/project-alpha:issue:1001:comments",
    },
    {
        "operation": "issue.label",
        "method": "POST",
        "endpoint": "/repos/owner/project-alpha/issues/1001/labels",
        "target": "github:repo:owner/project-alpha:issue:1001:labels",
    },
    {
        "operation": "file.write",
        "method": "PUT",
        "endpoint": "/repos/owner/project-alpha/contents/README.md",
        "target": "github:repo:owner/project-alpha:file:README.md",
    },
    {
        "operation": "pull_request.merge",
        "method": "PUT",
        "endpoint": "/repos/owner/project-alpha/pulls/7/merge",
        "target": "github:repo:owner/project-alpha:pull:7:merge",
    },
    {
        "operation": "branch.create",
        "method": "POST",
        "endpoint": "/repos/owner/project-alpha/git/refs",
        "target": "github:repo:owner/project-alpha:refs",
    },
    {
        "operation": "repository.settings.update",
        "method": "PATCH",
        "endpoint": "/repos/owner/project-alpha",
        "target": "github:repo:owner/project-alpha:settings",
    },
    {
        "operation": "issue.comment.delete",
        "method": "DELETE",
        "endpoint": "/repos/owner/project-alpha/issues/comments/1",
        "target": "github:repo:owner/project-alpha:comment:1",
    },
]
GITHUB_PROVIDER_FAILURE_FIXTURES: dict[str, dict[str, Any]] = {
    "rate_limit": {
        "reason_code": "CS_CONNECTOR_GITHUB_RATE_LIMITED",
        "health_state": "degraded",
        "setup_state": "ready_delayed",
        "stream_state": "delayed",
        "freshness_state": "delayed",
        "source_availability": "available_delayed",
        "recoverability": "automatic_retry",
        "retry_after_seconds": 300,
        "owner_action_required": False,
        "requires_new_verification": False,
        "setup_gap_permanent": False,
        "future_ingestion_allowed": True,
        "current_data_claim_allowed": False,
        "message": "GitHub rate limit delayed source freshness; retry is scheduled outside a tight loop.",
    },
    "permission_revoked": {
        "reason_code": "CS_CONNECTOR_GITHUB_PERMISSION_REVOKED",
        "health_state": "blocked",
        "setup_state": "setup_gap",
        "stream_state": "suspended",
        "freshness_state": "stale",
        "source_availability": "permission_revoked",
        "recoverability": "owner_reconnect_required",
        "retry_after_seconds": None,
        "owner_action_required": True,
        "requires_new_verification": True,
        "setup_gap_permanent": True,
        "future_ingestion_allowed": False,
        "current_data_claim_allowed": False,
        "message": "GitHub permission was revoked; affected streams are suspended until owner reconnection and verification.",
    },
    "repository_removed": {
        "reason_code": "CS_CONNECTOR_GITHUB_REPOSITORY_REMOVED",
        "health_state": "unavailable",
        "setup_state": "source_unavailable",
        "stream_state": "stopped",
        "freshness_state": "unavailable",
        "source_availability": "repository_removed",
        "recoverability": "owner_reselect_required",
        "retry_after_seconds": None,
        "owner_action_required": True,
        "requires_new_verification": True,
        "setup_gap_permanent": True,
        "future_ingestion_allowed": False,
        "current_data_claim_allowed": False,
        "message": "GitHub repository is unavailable or removed; future ingestion is stopped for this source.",
    },
    "transient_transport": {
        "reason_code": "CS_CONNECTOR_GITHUB_TRANSPORT_TRANSIENT",
        "health_state": "degraded",
        "setup_state": "ready_retrying",
        "stream_state": "retrying",
        "freshness_state": "delayed",
        "source_availability": "available_retrying",
        "recoverability": "automatic_retry",
        "retry_after_seconds": 60,
        "owner_action_required": False,
        "requires_new_verification": False,
        "setup_gap_permanent": False,
        "future_ingestion_allowed": True,
        "current_data_claim_allowed": False,
        "message": "Transient GitHub transport failure scheduled a bounded retry without changing setup authority.",
    },
}
GITHUB_PROVIDER_FAILURE_MODES = tuple(GITHUB_PROVIDER_FAILURE_FIXTURES.keys())
SUPPORTED_CAPTURE_PLATFORMS = ("macos",)
CAPTURE_PLATFORM_PERMISSION_STATES = ("not_granted", "granted", "unsupported_host", "unknown")
CAPTURE_CONSENT_DECISIONS = ("granted", "denied", "revoked")
CAPTURE_DEFAULT_SOURCE_ID = "macos_activity"
CHROME_ACTIVE_TAB_DEFAULT_SOURCE_ID = "chrome_active_tab"
CHROME_AUTO_CAPTURE_DEFAULT_SOURCE_ID = "chrome_auto_capture"
CHROME_SENSITIVE_PAGE_DEFAULT_SOURCE_ID = "chrome_sensitive_page"
CAPTURE_LIFECYCLE_TARGET_KINDS = ("source", "watch_rule", "global")
CAPTURE_LIFECYCLE_ACTIONS = ("pause", "resume", "revoke", "retention")
CAPTURE_LIFECYCLE_STATUSES = ("active", "paused", "revoked", "disabled")
CAPTURE_RESULT_REVIEW_DECISIONS = ("save", "dismiss")
WATCH_RESULT_REVIEW_DECISIONS = ("save_draft_memory", "dismiss", "create_claim_draft", "open_mission_draft")
CHROME_ACTIVE_TAB_MAX_TEXT_CLIP_CHARS = 512
ACTIVITY_SESSIONIZER_ALGORITHM = "cornerstone_activity_sessionizer_v1"
ACTIVITY_DEFAULT_IDLE_GAP_THRESHOLD_SECONDS = 300
ACTIVITY_DEFAULT_SAMPLE_INTERVAL_SECONDS = 60
CAPTURE_NEGATIVE_EVIDENCE_TEMPLATE: dict[str, int] = {
    "capture_before_consent": 0,
    "capture_before_platform_permission": 0,
    "capture_samples_before_both_gates": 0,
    "hidden_startup_capture": 0,
    "cross_namespace_capture": 0,
    "screenshots_before_permission": 0,
    "window_titles_before_permission": 0,
    "raw_keystrokes_collected": 0,
    "clipboard_values_collected": 0,
    "browser_history_collected": 0,
    "external_http_calls": 0,
    "provider_mutations": 0,
}
ACTIVITY_SESSION_NEGATIVE_EVIDENCE_TEMPLATE: dict[str, int] = {
    "unsupported_intent_claims": 0,
    "inference_stored_as_observed_fact": 0,
    "raw_window_titles_stored": 0,
    "full_urls_stored": 0,
    "keystrokes_collected": 0,
    "clipboard_values_collected": 0,
    "screenshots_collected": 0,
    "cookies_collected": 0,
    "browser_history_collected": 0,
    "external_http_calls": 0,
    "provider_mutations": 0,
}
WATCH_RULE_NEGATIVE_EVIDENCE_TEMPLATE: dict[str, int] = {
    "ownerless_global_rules": 0,
    "cross_namespace_lifecycle_mutations": 0,
    "authority_expansions_from_rule_text": 0,
    "external_actions_authorized_by_rule": 0,
    "capture_broadening_without_confirmation": 0,
    "provider_mutations": 0,
    "external_http_calls": 0,
}
CHROME_ACTIVE_TAB_NEGATIVE_EVIDENCE_TEMPLATE: dict[str, int] = {
    "broad_all_urls_permission": 0,
    "captures_without_user_gesture": 0,
    "captures_without_confirmation": 0,
    "popup_open_captures": 0,
    "non_active_tab_captures": 0,
    "backend_policy_bypasses": 0,
    "blocked_page_text_clip_stored": 0,
    "raw_text_stored": 0,
    "raw_html_stored": 0,
    "cookies_collected": 0,
    "local_storage_collected": 0,
    "session_storage_collected": 0,
    "screenshots_collected": 0,
    "form_values_collected": 0,
    "browser_history_collected": 0,
    "external_http_calls": 0,
    "provider_mutations": 0,
}
CHROME_AUTO_CAPTURE_NEGATIVE_EVIDENCE_TEMPLATE: dict[str, int] = {
    "captures_without_owner_rule": 0,
    "captures_without_site_allowance": 0,
    "captures_without_source_pack_allowance": 0,
    "captures_without_browser_permission": 0,
    "consent_config_version_mismatches": 0,
    "unapproved_domain_captures": 0,
    "inactive_tab_captures": 0,
    "throttle_bypasses": 0,
    "session_limit_bypasses": 0,
    "duplicate_idempotency_captures": 0,
    "raw_text_stored": 0,
    "raw_html_stored": 0,
    "cookies_collected": 0,
    "local_storage_collected": 0,
    "session_storage_collected": 0,
    "screenshots_collected": 0,
    "form_values_collected": 0,
    "browser_history_collected": 0,
    "external_http_calls": 0,
    "provider_mutations": 0,
}
CHROME_SENSITIVE_PAGE_NEGATIVE_EVIDENCE_TEMPLATE: dict[str, int] = {
    "client_block_downgrades": 0,
    "backend_false_safe_bypasses": 0,
    "blocked_page_text_persisted": 0,
    "degraded_raw_text_persisted": 0,
    "raw_html_stored": 0,
    "cookies_collected": 0,
    "local_storage_collected": 0,
    "session_storage_collected": 0,
    "screenshots_collected": 0,
    "form_values_collected": 0,
    "browser_history_collected": 0,
    "full_urls_stored": 0,
    "full_origins_stored": 0,
    "title_text_stored": 0,
    "content_sent_to_models": 0,
    "searchable_content_artifacts_created": 0,
    "capture_inbox_items_created": 0,
    "external_http_calls": 0,
    "provider_mutations": 0,
}
CAPTURE_LIFECYCLE_NEGATIVE_EVIDENCE_TEMPLATE: dict[str, int] = {
    "samples_collected_while_paused": 0,
    "samples_collected_while_revoked": 0,
    "configuration_deleted_on_pause": 0,
    "unscoped_exports": 0,
    "raw_content_exported": 0,
    "raw_browser_payload_exported": 0,
    "credential_values_exported": 0,
    "delete_everything_misleading_claims": 0,
    "audit_records_deleted": 0,
    "unauthorized_delete_executions": 0,
    "external_http_calls": 0,
    "provider_mutations": 0,
}
WATCH_RESULT_NEGATIVE_EVIDENCE_TEMPLATE: dict[str, int] = {
    "inferred_intent_labeled_observed_fact": 0,
    "inference_stored_as_observed_fact": 0,
    "unsupported_inference_approved": 0,
    "low_confidence_memory_approved_without_review": 0,
    "observation_mutated_by_correction": 0,
    "proposal_executed_directly": 0,
    "action_card_created_directly": 0,
    "claim_created_directly": 0,
    "mission_opened_directly": 0,
    "workflow_runs_started": 0,
    "raw_content_stored": 0,
    "external_http_calls": 0,
    "provider_mutations": 0,
}
CONNECTOR_ACTION_PREFLIGHT_NEGATIVE_EVIDENCE_TEMPLATE: dict[str, int] = {
    "dry_run_executed": 0,
    "preflight_counted_as_approval": 0,
    "undeclared_actions_executed": 0,
    "execution_result_created": 0,
    "workflow_runs_started": 0,
    "provider_mutations": 0,
    "external_http_calls": 0,
    "real_provider_calls": 0,
    "direct_provider_access": 0,
    "provider_clients_exposed": 0,
    "credential_values_exposed": 0,
    "github_read_only_action_admitted": 0,
}
WATCH_SOURCE_READINESS_STATES = ("ready", "missing")
ACTIVE_GITHUB_CONTRACT_FIXTURES = (
    "fixtures/connectorhub/contracts/github_readonly_contract.json",
    "fixtures/connectorhub/contracts/github_raw_access_contract.json",
    "fixtures/connectorhub/contracts/github_selected_repositories_contract.json",
    "fixtures/connectorhub/contracts/github_required_missing_contract.json",
    "fixtures/connectorhub/contracts/github_optional_missing_contract.json",
)

FIXTURE_PROVIDER_PACKS: dict[str, dict[str, Any]] = {
    "local_source_control_readonly.v1": {
        "provider_pack_id": "local_source_control_readonly.v1",
        "display_name": "Local source control read-only fixture",
        "transport": "local_fixture",
        "capabilities": {
            "source_control.repository.read": ["source_control.repository.v1"],
            "source_control.change.read": ["source_control.commit.v1", "source_control.change.v1"],
            "source_control.issue.read": ["source_control.issue.v1"],
            "source_control.file.read": ["source_control.file_snapshot.v1"],
        },
        "declared_actions": [],
        "provider_calls_before_activation": 0,
    },
    "local_source_control_readonly_alt.v1": {
        "provider_pack_id": "local_source_control_readonly_alt.v1",
        "display_name": "Alternate local source control read-only fixture",
        "transport": "local_fixture_alt",
        "capabilities": {
            "source_control.repository.read": ["source_control.repository.v1"],
            "source_control.change.read": ["source_control.commit.v1", "source_control.change.v1"],
            "source_control.issue.read": ["source_control.issue.v1"],
            "source_control.file.read": ["source_control.file_snapshot.v1"],
        },
        "declared_actions": [],
        "provider_calls_before_activation": 0,
    },
    "local_source_control_permission_gap.v1": {
        "provider_pack_id": "local_source_control_permission_gap.v1",
        "display_name": "Local source control permission gap fixture",
        "transport": "local_fixture_permission_gap",
        "capabilities": {
            "source_control.repository.read": ["source_control.repository.v1"],
            "source_control.change.read": ["source_control.commit.v1", "source_control.change.v1"],
        },
        "declared_actions": [],
        "provider_calls_before_activation": 0,
        "setup_gap": {
            "reason_code": "CS_CONNECTOR_PERMISSION_REQUIRED",
            "cause": "A read-only selected-repository grant is required before this connector can activate.",
            "impact": "Repository and change streams stay unavailable until the owner grants the missing read permission.",
            "resolution_steps": [
                "Open Connected Sources.",
                "Review the selected repository scope.",
                "Grant the read-only permission or choose a provider pack that is already authorized.",
            ],
        },
    },
    "local_source_control_breaking_v2.v2": {
        "provider_pack_id": "local_source_control_breaking_v2.v2",
        "display_name": "Breaking local source control fixture",
        "transport": "local_fixture_breaking",
        "capabilities": {"source_control.repository.read": ["source_control.repository.v2"]},
        "declared_actions": [],
        "provider_calls_before_activation": 0,
    },
}

PRODUCT_HANDLER_CONTRACT = {
    "schema_version": "cs.connector_product_handler_contract.v1",
    "handler_family": "source_control_readonly",
    "requires_provider_sdk": False,
    "product_object_schema": "cs.connected_source.preview.v1",
}

CAPABILITY_SURFACES = {
    "source_control.repository.read": "Repository evidence",
    "source_control.change.read": "Commit and change evidence",
    "source_control.issue.read": "Issue evidence",
    "source_control.file.read": "File snapshot evidence",
    "source_control.pull_request.read": "Pull request evidence",
}

NORMAL_USER_PRODUCT_SURFACES = [
    {"id": "home", "label": "Home"},
    {"id": "search", "label": "Search"},
    {"id": "artifacts", "label": "Artifacts"},
    {"id": "claims", "label": "Claims"},
    {"id": "actions", "label": "Actions"},
    {"id": "connected_sources", "label": "Connected Sources"},
]
CONNECTED_SOURCE_PRODUCT_COPY = [
    "Connect a selected source.",
    "Review permitted content and evidence coverage.",
    "Use search, briefs, claims, and governed actions from CornerStone.",
]
ADMIN_DETAIL_SURFACES = [
    "Source Policy",
    "Setup Result",
    "Provider Pack compatibility",
    "Retry and quarantine diagnostics",
    "Connector audit",
]
FORBIDDEN_NORMAL_USER_TERMS = [
    "ConnectorHub",
    "ConnectorHubKit",
    "Connector-Hub",
    "Provider Pack",
    "Setup Result",
    "Source Policy",
    "Projection",
    "Delivery",
]
REPORT_READINESS_DIMENSIONS = {
    "contract_schema_verified": "LOCAL_FIXTURE_VERIFIED",
    "local_fixture_behavior_verified": "LOCAL_FIXTURE_VERIFIED",
    "local_physical_device_behavior_verified": "HUMAN_REQUIRED",
    "live_provider_read_verified": "NOT_VERIFIED",
    "live_provider_write_verified": "OUT_OF_SCOPE_READ_ONLY",
    "production_tenancy_policy_egress_verified": "NOT_VERIFIED",
    "human_ux_privacy_accepted": "HUMAN_REQUIRED",
    "release_publishing_approved": "NOT_VERIFIED",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def utc_after(seconds: int) -> str:
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat().replace("+00:00", "Z")


def json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _json_report_summary(root: Path, relative_path: str) -> dict[str, Any]:
    path = root / relative_path
    report: dict[str, Any] = {
        "path": relative_path,
        "present": path.exists() and path.is_file(),
    }
    if not report["present"]:
        return report
    content = path.read_bytes()
    report["sha256"] = hashlib.sha256(content).hexdigest()
    report["bytes"] = len(content)
    try:
        payload = json.loads(content)
    except ValueError:
        report["json_valid"] = False
        return report
    report["json_valid"] = True
    for key in ("schema_version", "status", "command"):
        if key in payload:
            report[key] = payload[key]
    summary = payload.get("summary")
    if isinstance(summary, dict):
        report["summary"] = {
            key: summary[key]
            for key in (
                "pass",
                "fail",
                "blocking",
                "human_required",
                "not_verified",
                "not_run",
                "scenario_count",
                "product_feature_claims",
            )
            if key in summary
        }
    proof_hash = payload.get("proof_hash")
    if isinstance(proof_hash, str):
        report["proof_hash"] = proof_hash
    scenario_results = payload.get("scenario_results")
    if isinstance(scenario_results, list):
        report["scenario_count"] = len(scenario_results)
    return report


def connector_human_gate_h04_local_baseline_preflight_command_plan() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = [
        {
            "step_order": 1,
            "operator_phase": "refresh_local_vs2_baseline_inputs",
            "command": "cornerstone security vs2-local-proof --json",
            "purpose": (
                "Refresh the current local VS2 proof inputs before H04 review without treating "
                "local proof as production-like acceptance."
            ),
            "expected_report_paths": [
                "reports/security/vs2-local-security-proof.json",
                "reports/network/vs2-egress-proof.json",
                "reports/security/vs2-local-range.json",
            ],
        },
        {
            "step_order": 2,
            "operator_phase": "refresh_vs2_scenario_report",
            "command": (
                "cornerstone scenario verify vs2-policy-tenancy-egress "
                "--reuse-vs2-local-proof-report reports/security/vs2-local-security-proof.json --json"
            ),
            "purpose": (
                "Refresh the local VS2 scenario report that H04 reviewers compare against the "
                "production-like environment transcript."
            ),
            "expected_report_paths": [
                "reports/security/vs2-local-security-proof.json",
                "reports/scenario/vs2-policy-tenancy-egress-2026-06-19.json",
            ],
        },
        {
            "step_order": 3,
            "operator_phase": "refresh_connectorhub_dependency_report",
            "command": "cornerstone scenario verify connector-contract-adapter --scenario CS-CH-036 --json",
            "purpose": (
                "Refresh the ConnectorHub CS-CH-036 dependency report that remains local fixture "
                "evidence until H04/H07 human proof exists."
            ),
            "expected_report_paths": [
                "reports/scenario/connector-contract-adapter-cs-ch-036-2026-06-23.json",
            ],
        },
    ]
    for row in rows:
        row.update(
            {
                "schema_version": CONNECTOR_HUMAN_GATE_H04_BASELINE_PREFLIGHT_COMMAND_PLAN_SCHEMA,
                "expected_report_count": len(row["expected_report_paths"]),
                "review_input_only": True,
                "acceptance_sufficient": False,
                "product_claim_allowed": False,
                "pass_claim_allowed": False,
                "claim_boundary": CONNECTOR_HUMAN_GATE_H04_BASELINE_PREFLIGHT_CLAIM_BOUNDARY,
            }
        )
    return rows


def connector_human_gate_h04_local_baseline_preflight_bundle(
    reports: list[dict[str, Any]],
    missing_reports: list[str],
    invalid_json_reports: list[str],
    preflight_command_plan: list[dict[str, Any]],
    required_human_delta: list[str],
) -> dict[str, Any]:
    report_fingerprints = []
    for report in reports:
        fingerprint = {
            "path": report.get("path"),
            "present": report.get("present") is True,
            "json_valid": report.get("json_valid") is True,
            "sha256": report.get("sha256"),
            "status": report.get("status"),
            "schema_version": report.get("schema_version"),
            "command": report.get("command"),
            "scenario_count": report.get("scenario_count"),
            "review_input_only": report.get("review_input_only") is True,
            "acceptance_sufficient": report.get("acceptance_sufficient") is True,
            "product_claim_allowed": report.get("product_claim_allowed") is True,
            "pass_claim_allowed": report.get("pass_claim_allowed") is True,
            "claim_boundary": report.get("claim_boundary"),
        }
        report_fingerprints.append({key: value for key, value in fingerprint.items() if value is not None})
    current_report_paths = [str(report.get("path")) for report in reports if report.get("path")]
    command_plan_expected_paths = sorted(
        {
            str(path)
            for row in preflight_command_plan
            for path in (row.get("expected_report_paths") or [])
        }
    )
    covered_paths = sorted(set(command_plan_expected_paths).intersection(current_report_paths))
    missing_from_current = sorted(set(command_plan_expected_paths).difference(current_report_paths))
    ready_report_paths = [
        str(report.get("path"))
        for report in reports
        if report.get("path") and report.get("present") is True and report.get("json_valid") is True
    ]
    return {
        "schema_version": CONNECTOR_HUMAN_GATE_H04_BASELINE_PREFLIGHT_BUNDLE_SCHEMA,
        "scenario_id": "CS-CH-H04",
        "status": "operator_preparation_only",
        "baseline_scope": "local_ai_verifiable_vs2_and_connectorhub_dependency_proof",
        "review_input_only": True,
        "acceptance_sufficient": False,
        "product_claim_allowed": False,
        "pass_claim_allowed": False,
        "claim_boundary": CONNECTOR_HUMAN_GATE_H04_BASELINE_PREFLIGHT_BUNDLE_CLAIM_BOUNDARY,
        "required_human_delta": list(required_human_delta),
        "required_human_delta_count": len(required_human_delta),
        "command_plan_schema_version": CONNECTOR_HUMAN_GATE_H04_BASELINE_PREFLIGHT_COMMAND_PLAN_SCHEMA,
        "command_plan_count": len(preflight_command_plan),
        "command_plan": deepcopy(preflight_command_plan),
        "recommended_preflight_command_plan_schema_version": (
            CONNECTOR_HUMAN_GATE_H04_BASELINE_PREFLIGHT_COMMAND_PLAN_SCHEMA
        ),
        "recommended_preflight_command_plan_count": len(preflight_command_plan),
        "recommended_preflight_command_plan": deepcopy(preflight_command_plan),
        "recommended_preflight_commands": [row["command"] for row in preflight_command_plan],
        "current_report_count": len(current_report_paths),
        "current_report_paths": current_report_paths,
        "current_report_fingerprints": report_fingerprints,
        "ready_report_count": len(ready_report_paths),
        "ready_report_paths": ready_report_paths,
        "missing_reports": list(missing_reports),
        "invalid_json_reports": list(invalid_json_reports),
        "all_reports_present": not missing_reports,
        "all_reports_json_valid": not invalid_json_reports,
        "command_plan_expected_report_path_count": len(command_plan_expected_paths),
        "command_plan_expected_report_paths": command_plan_expected_paths,
        "command_plan_paths_covered_by_current_reports": covered_paths,
        "command_plan_paths_missing_from_current_reports": missing_from_current,
        "commands_executed_by_bundle": 0,
        "live_provider_calls_executed_by_bundle": 0,
        "provider_mutations_executed_by_bundle": 0,
        "external_mutations_executed_by_bundle": 0,
        "human_acceptance_collected_by_bundle": False,
        "operator_next_step": (
            "Run the command plan, collect production-like H04 evidence, and validate a dated "
            "human reviewer record; this bundle is only a local comparison input."
        ),
    }


def connector_human_gate_local_baseline_review_inputs(root: Path, scenario_id: str) -> dict[str, Any] | None:
    if scenario_id != "CS-CH-H04":
        return None
    report_paths = [
        "reports/security/vs2-local-security-proof.json",
        "reports/scenario/vs2-policy-tenancy-egress-2026-06-19.json",
        "reports/network/vs2-egress-proof.json",
        "reports/security/vs2-local-range.json",
        "reports/scenario/connector-contract-adapter-cs-ch-036-2026-06-23.json",
    ]
    reports = []
    for path in report_paths:
        report = _json_report_summary(root, path)
        report.update(
            {
                "review_input_only": True,
                "acceptance_sufficient": False,
                "product_claim_allowed": False,
                "pass_claim_allowed": False,
                "claim_boundary": CONNECTOR_HUMAN_GATE_H04_BASELINE_CLAIM_BOUNDARY,
            }
        )
        reports.append(report)
    missing = [report["path"] for report in reports if not report["present"]]
    invalid_json = [report["path"] for report in reports if report["present"] and not report.get("json_valid", False)]
    preflight_command_plan = connector_human_gate_h04_local_baseline_preflight_command_plan()
    required_human_delta = [
        "Production-like topology identifier and trusted RequestContext transcript.",
        "Scenario-specific PostgreSQL/RLS and OPA transcripts from the reviewed environment.",
        "Network default-deny and governed-egress transcripts from the reviewed topology.",
        "Backup/restore evidence and audit-integrity report from the reviewed environment.",
        "Dated ACCEPT or REJECT decision with redacted evidence packet manifest.",
    ]
    preflight_bundle = connector_human_gate_h04_local_baseline_preflight_bundle(
        reports,
        missing,
        invalid_json,
        preflight_command_plan,
        required_human_delta,
    )
    return {
        "schema_version": "cs.connector_human_gate_local_baseline_review_inputs.v1",
        "scenario_id": scenario_id,
        "status": "review_input_only",
        "acceptance_sufficient": False,
        "product_claim_allowed": False,
        "pass_claim_allowed": False,
        "baseline_scope": "local_ai_verifiable_vs2_and_connectorhub_dependency_proof",
        "boundary": (
            "These local reports are comparison inputs for the H04 reviewer. They do not "
            "prove production-like RequestContext, PostgreSQL/RLS, OPA, egress, "
            "backup/restore, or audit readiness."
        ),
        "required_human_delta": required_human_delta,
        "recommended_preflight_commands": [row["command"] for row in preflight_command_plan],
        "recommended_preflight_command_plan_schema_version": (
            CONNECTOR_HUMAN_GATE_H04_BASELINE_PREFLIGHT_COMMAND_PLAN_SCHEMA
        ),
        "recommended_preflight_command_plan_count": len(preflight_command_plan),
        "recommended_preflight_command_plan": preflight_command_plan,
        "preflight_bundle": preflight_bundle,
        "reports": reports,
        "missing_reports": missing,
        "invalid_json_reports": invalid_json,
        "all_reports_present": not missing,
        "all_reports_json_valid": not invalid_json,
    }


def connector_human_template_structure(template_path: Path, scenario_id: str) -> dict[str, Any]:
    if not template_path.exists():
        return {
            "template_present": False,
            "structure_ready": False,
            "missing_required_tokens": list(CONNECTOR_HUMAN_TEMPLATE_REQUIRED_TOKENS),
            "has_scenario_id": False,
            "has_no_pass_boundary": False,
            "has_scenario_first_runbook_or_study": False,
            "has_acceptance_evidence_packet": False,
            "has_senior_review_perspectives": False,
            "missing_senior_review_perspectives": list(
                CONNECTOR_HUMAN_TEMPLATE_REQUIRED_PERSPECTIVE_LABELS
            ),
            "has_pending_human_result_rows": False,
            "has_reject_conditions": False,
        }

    template_text = template_path.read_text()
    missing_required_tokens = [
        token for token in CONNECTOR_HUMAN_TEMPLATE_REQUIRED_TOKENS if token not in template_text
    ]
    has_scenario_id = scenario_id in template_text
    has_no_pass_boundary = "does not mark" in template_text and "HUMAN_REQUIRED" in template_text
    has_scenario_first_runbook_or_study = (
        "## Scenario-First Execution Runbook" in template_text
        or "## Scenario-First Study Runbook" in template_text
    )
    has_acceptance_evidence_packet = "## Acceptance Evidence Packet" in template_text
    missing_senior_review_perspectives = [
        label
        for label in CONNECTOR_HUMAN_TEMPLATE_REQUIRED_PERSPECTIVE_LABELS
        if label not in template_text
    ]
    has_senior_review_perspectives = (
        "## Senior Review Perspectives" in template_text
        and not missing_senior_review_perspectives
    )
    has_pending_human_result_rows = "Human result" in template_text and "PENDING" in template_text
    has_reject_conditions = "Reject" in template_text or "reject" in template_text
    structure_ready = (
        not missing_required_tokens
        and has_scenario_id
        and has_no_pass_boundary
        and has_scenario_first_runbook_or_study
        and has_acceptance_evidence_packet
        and has_senior_review_perspectives
        and has_pending_human_result_rows
        and has_reject_conditions
    )
    return {
        "template_present": True,
        "structure_ready": structure_ready,
        "missing_required_tokens": missing_required_tokens,
        "has_scenario_id": has_scenario_id,
        "has_no_pass_boundary": has_no_pass_boundary,
        "has_scenario_first_runbook_or_study": has_scenario_first_runbook_or_study,
        "has_acceptance_evidence_packet": has_acceptance_evidence_packet,
        "has_senior_review_perspectives": has_senior_review_perspectives,
        "missing_senior_review_perspectives": missing_senior_review_perspectives,
        "has_pending_human_result_rows": has_pending_human_result_rows,
        "has_reject_conditions": has_reject_conditions,
    }


def connector_human_template_path(scenario_id: str) -> Path:
    return (
        Path("docs/verification-reports")
        / f"CONNECTOR_HUB_{scenario_id.replace('-', '_')}_HUMAN_REVIEW_TEMPLATE_2026-06-24.md"
    )


def connector_human_gate_field_ref_contract(scenario_id: str) -> dict[str, Any] | None:
    if scenario_id != "CS-CH-H04":
        return None
    field_items = [
        {
            **deepcopy(item),
            "required": True,
            "raw_value_persisted_by_validator": False,
        }
        for item in CONNECTOR_HUMAN_GATE_H04_FIELD_REF_ITEMS
    ]
    return {
        "schema_version": CONNECTOR_HUMAN_GATE_FIELD_REF_CONTRACT_SCHEMA,
        "scenario_id": scenario_id,
        "status": "operator_preparation_only",
        "validation_scope": "field_reference_shape_only",
        "raw_field_values_persisted_by_validator": False,
        "invalid_value_report_shape": "field_names_only",
        "required_field_ref_items": field_items,
    }


def connector_human_gate_evidence_packet_file_scaffold_plan(
    scenario_id: str,
) -> list[dict[str, Any]] | None:
    packet_file_items = CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_FILE_ITEMS_BY_SCENARIO.get(
        scenario_id
    )
    packet_dir = CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY_BY_SCENARIO.get(
        scenario_id
    )
    if packet_file_items is None or packet_dir is None:
        return None
    plan: list[dict[str, Any]] = [
        {
            "step": 1,
            "operation": "prepare_packet_directory",
            "packet_directory": packet_dir,
            "command": f"mkdir -p {packet_dir}",
            "command_executed_by_report": False,
            "review_input_only": True,
            "acceptance_sufficient": False,
            "product_claim_allowed": False,
            "pass_claim_allowed": False,
            "packet_file_contents_recorded_by_report": False,
            "packet_file_contents_persisted_by_report": False,
        }
    ]
    for index, item in enumerate(packet_file_items, start=2):
        packet_file = str(item["packet_file"])
        plan.append(
            {
                "step": index,
                "operation": "create_packet_file",
                "packet_directory": packet_dir,
                "packet_file": packet_file,
                "command": f"touch {packet_dir}/{packet_file}",
                "required": True,
                "required_contents": item["required_contents"],
                "command_executed_by_report": False,
                "review_input_only": True,
                "acceptance_sufficient": False,
                "product_claim_allowed": False,
                "pass_claim_allowed": False,
                "packet_file_contents_recorded_by_report": False,
                "packet_file_contents_persisted_by_report": False,
            }
        )
    return plan


def connector_human_gate_evidence_packet_scaffold_template_content(
    packet_file: str,
    required_contents: str,
    scenario_id: str = "CS-CH-H04",
) -> str:
    short_scenario = scenario_id.replace("CS-CH-", "")
    if packet_file.endswith(".json"):
        template = {
            "scenario_id": scenario_id,
            "status": "TEMPLATE_ONLY_NOT_HUMAN_EVIDENCE",
            "packet_file": packet_file,
            "required_contents": required_contents,
            "evidence_ref": "",
            "redaction_status": "",
            "reviewer": "",
            "review_timestamp": "",
            "notes": "",
            "operator_warning": f"Do not mark {short_scenario} PASS from this template file.",
            "acceptance_claim": "HUMAN_REQUIRED_UNTIL_OWNER_ACCEPT_RECORD",
        }
        return json.dumps(template, indent=2, sort_keys=True) + "\n"
    if packet_file.endswith(".md"):
        return (
            f"# {short_scenario} Evidence Packet Template: {packet_file}\n\n"
            "Status: TEMPLATE_ONLY_NOT_HUMAN_EVIDENCE\n"
            f"Scenario: {scenario_id}\n\n"
            "## Required Contents\n\n"
            f"{required_contents}\n\n"
            "## Evidence Ref\n\n"
            "- evidence_ref:\n"
            "- redaction_status:\n"
            "- reviewer:\n"
            "- review_timestamp:\n\n"
            "## Notes\n\n"
            "- Keep secrets and private payloads redacted.\n"
            f"- Do not mark {short_scenario} PASS from this template file.\n"
        )
    return (
        f"{short_scenario} Evidence Packet Template: {packet_file}\n"
        "Status: TEMPLATE_ONLY_NOT_HUMAN_EVIDENCE\n"
        f"Scenario: {scenario_id}\n\n"
        "Required Contents:\n"
        f"{required_contents}\n\n"
        "Evidence Ref:\n"
        "Redaction Status:\n"
        "Reviewer:\n"
        "Review Timestamp:\n\n"
        "Notes:\n"
        "- Keep secrets and private payloads redacted.\n"
        f"- Do not mark {short_scenario} PASS from this template file.\n"
    )


def connector_human_gate_evidence_packet_scaffold(
    scenario_id: str,
) -> dict[str, Any] | None:
    evidence_packet_file_contract = connector_human_gate_evidence_packet_file_contract(
        scenario_id
    )
    if evidence_packet_file_contract is None:
        return None
    templates: list[dict[str, Any]] = []
    for item in evidence_packet_file_contract["required_packet_files"]:
        packet_file = str(item["packet_file"])
        required_contents = str(item["required_contents"])
        content = connector_human_gate_evidence_packet_scaffold_template_content(
            packet_file,
            required_contents,
            scenario_id,
        )
        templates.append(
            {
                "packet_file": packet_file,
                "required": True,
                "required_contents": required_contents,
                "template_only": True,
                "template_content_sha256": hashlib.sha256(
                    content.encode("utf-8")
                ).hexdigest(),
                "template_content_line_count": len(content.splitlines()),
                "human_evidence_recorded_by_template": False,
                "acceptance_sufficient": False,
                "product_claim_allowed": False,
                "pass_claim_allowed": False,
                "packet_file_contents_read_by_scaffold": False,
            }
        )
    return {
        "schema_version": CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_SCAFFOLD_SCHEMA,
        "scenario_id": scenario_id,
        "status": "operator_preparation_only",
        "scaffold_scope": "blank_acceptance_packet_templates_only",
        "evidence_packet_file_contract_schema_version": (
            evidence_packet_file_contract["schema_version"]
        ),
        "required_packet_file_count": evidence_packet_file_contract[
            "required_packet_file_count"
        ],
        "required_packet_file_names": evidence_packet_file_contract[
            "packet_file_names"
        ],
        "scaffold_template_count": len(templates),
        "scaffold_templates": templates,
        "template_contents_included_in_report": False,
        "packet_file_contents_read_by_scaffold": False,
        "human_evidence_recorded_by_scaffold": False,
        "review_input_only": True,
        "acceptance_sufficient": False,
        "product_claim_allowed": False,
        "pass_claim_allowed": False,
    }


def connector_human_gate_packet_file_metadata(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    line_count = 0
    size_bytes = 0
    last_byte: int | None = None
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            if not chunk:
                continue
            digest.update(chunk)
            size_bytes += len(chunk)
            line_count += chunk.count(b"\n")
            last_byte = chunk[-1]
    if size_bytes and last_byte != 10:
        line_count += 1
    return {
        "sha256": digest.hexdigest(),
        "size_bytes": size_bytes,
        "line_count": line_count,
    }


def connector_human_gate_packet_slug(scenario_id: str) -> str:
    return scenario_id.lower().replace("cs-ch-", "").replace("_", "-")


def connector_human_gate_evidence_packet_product_feature_claim(
    scenario_id: str,
    feature: str,
) -> str:
    return (
        f"CONNECTOR_HUB_{scenario_id.replace('CS-CH-', '')}_EVIDENCE_PACKET_"
        f"{feature}_PREPARED_HUMAN_EVIDENCE_REQUIRED"
    )


def connector_human_gate_packet_hash_ref(
    prefix: str,
    scenario_id: str,
    packet_file: str,
    sha256: str,
) -> str:
    return f"{prefix}:{connector_human_gate_packet_slug(scenario_id)}-packet/{packet_file}#sha256={sha256}"


def connector_human_gate_h04_packet_hash_ref(prefix: str, packet_file: str, sha256: str) -> str:
    return connector_human_gate_packet_hash_ref(prefix, "CS-CH-H04", packet_file, sha256)


def connector_human_gate_packet_file_map(packet_files: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(entry.get("packet_file")): entry
        for entry in packet_files
        if isinstance(entry, dict) and entry.get("present") is True and entry.get("sha256")
    }


def connector_human_gate_h04_packet_file_map(packet_files: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return connector_human_gate_packet_file_map(packet_files)


def connector_human_gate_packet_manifest_ref(
    scenario_id: str,
    packet_file_map: dict[str, dict[str, Any]],
) -> str:
    packet_slug = connector_human_gate_packet_slug(scenario_id)
    manifest_input = {
        packet_file: {
            "sha256": entry.get("sha256"),
            "size_bytes": entry.get("size_bytes"),
            "line_count": entry.get("line_count"),
        }
        for packet_file, entry in sorted(packet_file_map.items())
    }
    return f"evidence_manifest:{packet_slug}-packet/hash-manifest#sha256={json_hash(manifest_input)}"


def connector_human_gate_h04_packet_manifest_ref(packet_file_map: dict[str, dict[str, Any]]) -> str:
    return connector_human_gate_packet_manifest_ref("CS-CH-H04", packet_file_map)


def connector_human_gate_h04_record_draft_from_packet(
    packet_validation_report: dict[str, Any],
) -> dict[str, Any] | None:
    if packet_validation_report.get("scenario_id") != "CS-CH-H04":
        return None
    if packet_validation_report.get("packet_structurally_complete") is not True:
        return None
    definition = CONNECTOR_HUMAN_GATE_PACKAGE_DEFINITIONS["CS-CH-H04"]
    execution_queue_item = CONNECTOR_HUMAN_GATE_QUEUE_BY_SCENARIO["CS-CH-H04"]
    proposed_record_template = connector_human_gate_record_template(
        "CS-CH-H04",
        definition,
        execution_queue_item,
    )
    draft = deepcopy(proposed_record_template["record_template"])
    packet_file_map = connector_human_gate_h04_packet_file_map(
        packet_validation_report.get("packet_files") or []
    )
    required_files = {
        item["packet_file"]
        for item in CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_ITEMS
    }
    if required_files.difference(packet_file_map):
        return None

    def ref(prefix: str, packet_file: str) -> str:
        return connector_human_gate_h04_packet_hash_ref(
            prefix,
            packet_file,
            str(packet_file_map[packet_file]["sha256"]),
        )

    draft["environment_topology_ref"] = ref("topology", "environment-topology.md")
    draft["request_context_proof"] = ref("request_context", "request-context-trace.json")
    draft["db_policy_transcripts"] = [
        ref("db_policy", "postgres-rls-transcript.txt"),
        ref("db_policy", "opa-policy-transcript.json"),
    ]
    draft["network_egress_transcripts"] = [ref("egress", "egress-transcript.txt")]
    draft["backup_restore_evidence"] = [ref("backup_restore", "backup-restore-evidence.md")]
    draft["audit_integrity_report"] = ref("audit_integrity", "audit-integrity-report.json")
    draft["evidence_manifest_ref"] = connector_human_gate_h04_packet_manifest_ref(packet_file_map)
    draft["evidence_packet_manifest"] = [
        {
            "required_evidence_index": 1,
            "required_evidence": "Production-like topology description and trusted RequestContext proof.",
            "evidence_ref": (
                f"evidence_manifest:h04-packet/group-1#sha256="
                f"{json_hash([packet_file_map['environment-topology.md'], packet_file_map['request-context-trace.json']])}"
            ),
            "redaction_status": "redacted",
        },
        {
            "required_evidence_index": 2,
            "required_evidence": "Scenario-specific PostgreSQL/RLS and OPA transcripts.",
            "evidence_ref": (
                f"evidence_manifest:h04-packet/group-2#sha256="
                f"{json_hash([packet_file_map['postgres-rls-transcript.txt'], packet_file_map['opa-policy-transcript.json']])}"
            ),
            "redaction_status": "redacted",
        },
        {
            "required_evidence_index": 3,
            "required_evidence": "Network default-deny and governed-egress transcripts.",
            "evidence_ref": (
                f"evidence_manifest:h04-packet/group-3#sha256="
                f"{json_hash(packet_file_map['egress-transcript.txt'])}"
            ),
            "redaction_status": "redacted",
        },
        {
            "required_evidence_index": 4,
            "required_evidence": "Backup/restore, evidence manifest, and audit integrity reports.",
            "evidence_ref": (
                f"evidence_manifest:h04-packet/group-4#sha256="
                f"{json_hash([packet_file_map['backup-restore-evidence.md'], packet_file_map['audit-integrity-report.json'], packet_file_map['review-decision.md']])}"
            ),
            "redaction_status": "redacted",
        },
    ]
    return draft


def connector_human_gate_h07_record_draft_from_packet(
    packet_validation_report: dict[str, Any],
) -> dict[str, Any] | None:
    scenario_id = "CS-CH-H07"
    if packet_validation_report.get("scenario_id") != scenario_id:
        return None
    if packet_validation_report.get("packet_structurally_complete") is not True:
        return None
    definition = CONNECTOR_HUMAN_GATE_PACKAGE_DEFINITIONS[scenario_id]
    execution_queue_item = CONNECTOR_HUMAN_GATE_QUEUE_BY_SCENARIO[scenario_id]
    proposed_record_template = connector_human_gate_record_template(
        scenario_id,
        definition,
        execution_queue_item,
    )
    draft = deepcopy(proposed_record_template["record_template"])
    packet_file_map = connector_human_gate_packet_file_map(
        packet_validation_report.get("packet_files") or []
    )
    required_files = {
        item["packet_file"]
        for item in CONNECTOR_HUMAN_GATE_H07_EVIDENCE_PACKET_FILE_ITEMS
    }
    if required_files.difference(packet_file_map):
        return None

    def ref(prefix: str, packet_file: str) -> str:
        return connector_human_gate_packet_hash_ref(
            prefix,
            scenario_id,
            packet_file,
            str(packet_file_map[packet_file]["sha256"]),
        )

    draft["backup_manifest_ref"] = ref("backup_manifest", "backup-manifest.json")
    draft["restore_log_ref"] = ref("restore_log", "restore-log.txt")
    draft["cursor_reconciliation_ref"] = ref(
        "cursor_reconciliation",
        "cursor-reconciliation.json",
    )
    draft["replay_results_ref"] = ref("replay_results", "replay-results.json")
    draft["audit_verification_ref"] = ref(
        "audit_integrity",
        "audit-integrity-report.json",
    )
    draft["before_after_counts_hashes"] = (
        "counts_hashes:h07-packet/before-after#sha256="
        f"{json_hash([packet_file_map['pre-backup-baseline.json'], packet_file_map['read-model-validation.txt']])}"
    )
    draft["evidence_packet_manifest"] = [
        {
            "required_evidence_index": 1,
            "required_evidence": "Backup manifest and restore logs for production-like durable state.",
            "evidence_ref": (
                "evidence_manifest:h07-packet/group-1#sha256="
                f"{json_hash([packet_file_map['recovery-scope.md'], packet_file_map['backup-manifest.json'], packet_file_map['restore-log.txt']])}"
            ),
            "redaction_status": "redacted",
        },
        {
            "required_evidence_index": 2,
            "required_evidence": "Cursor reconciliation and replay results.",
            "evidence_ref": (
                "evidence_manifest:h07-packet/group-2#sha256="
                f"{json_hash([packet_file_map['cursor-reconciliation.json'], packet_file_map['replay-results.json']])}"
            ),
            "redaction_status": "redacted",
        },
        {
            "required_evidence_index": 3,
            "required_evidence": "Before/after logical counts and hashes.",
            "evidence_ref": (
                "evidence_manifest:h07-packet/group-3#sha256="
                f"{json_hash([packet_file_map['pre-backup-baseline.json'], packet_file_map['read-model-validation.txt']])}"
            ),
            "redaction_status": "redacted",
        },
        {
            "required_evidence_index": 4,
            "required_evidence": "Audit verification proving recovered state integrity.",
            "evidence_ref": (
                "evidence_manifest:h07-packet/group-4#sha256="
                f"{json_hash([packet_file_map['audit-integrity-report.json'], packet_file_map['review-decision.md']])}"
            ),
            "redaction_status": "redacted",
        },
    ]
    return draft


def connector_human_gate_h01_record_draft_from_packet(
    packet_validation_report: dict[str, Any],
) -> dict[str, Any] | None:
    scenario_id = "CS-CH-H01"
    if packet_validation_report.get("scenario_id") != scenario_id:
        return None
    if packet_validation_report.get("packet_structurally_complete") is not True:
        return None
    definition = CONNECTOR_HUMAN_GATE_PACKAGE_DEFINITIONS[scenario_id]
    execution_queue_item = CONNECTOR_HUMAN_GATE_QUEUE_BY_SCENARIO[scenario_id]
    proposed_record_template = connector_human_gate_record_template(
        scenario_id,
        definition,
        execution_queue_item,
    )
    draft = deepcopy(proposed_record_template["record_template"])
    packet_file_map = connector_human_gate_packet_file_map(
        packet_validation_report.get("packet_files") or []
    )
    required_files = {
        item["packet_file"]
        for item in CONNECTOR_HUMAN_GATE_H01_EVIDENCE_PACKET_FILE_ITEMS
    }
    if required_files.difference(packet_file_map):
        return None

    def ref(prefix: str, packet_file: str) -> str:
        return connector_human_gate_packet_hash_ref(
            prefix,
            scenario_id,
            packet_file,
            str(packet_file_map[packet_file]["sha256"]),
        )

    draft["github_app_installation_id_redacted"] = ref("github_installation", "github-scope.md")
    draft["selected_repositories"] = [ref("selected_repositories", "github-scope.md")]
    draft["permission_snapshot"] = ref("github_permission", "permission-snapshot.md")
    draft["call_ledger"] = ref("github_call_ledger", "read-call-ledger.json")
    draft["delivery_refs"] = [ref("connector_delivery", "delivery-evidence.json")]
    draft["audit_refs"] = [
        ref("audit", "delivery-evidence.json"),
        ref("audit", "redaction-audit-report.json"),
    ]
    draft["zero_write_proof"] = ref("zero_write_proof", "zero-write-proof.json")
    draft["evidence_packet_manifest"] = [
        {
            "required_evidence_index": 1,
            "required_evidence": (
                "Redacted GitHub App or equivalent least-privilege installation permission snapshot."
            ),
            "evidence_ref": (
                "evidence_manifest:h01-packet/group-1#sha256="
                f"{json_hash([packet_file_map['github-scope.md'], packet_file_map['permission-snapshot.md']])}"
            ),
            "redaction_status": "redacted",
        },
        {
            "required_evidence_index": 2,
            "required_evidence": (
                "Selected repository list and explicit confirmation that no unselected repositories "
                "were ingested."
            ),
            "evidence_ref": (
                "evidence_manifest:h01-packet/group-2#sha256="
                f"{json_hash([packet_file_map['github-scope.md'], packet_file_map['unselected-denial.txt']])}"
            ),
            "redaction_status": "redacted",
        },
        {
            "required_evidence_index": 3,
            "required_evidence": (
                "Read-only call ledger showing zero write-capable HTTP methods or endpoints."
            ),
            "evidence_ref": (
                "evidence_manifest:h01-packet/group-3#sha256="
                f"{json_hash([packet_file_map['read-call-ledger.json'], packet_file_map['zero-write-proof.json']])}"
            ),
            "redaction_status": "redacted",
        },
        {
            "required_evidence_index": 4,
            "required_evidence": (
                "Connector Delivery, Artifact/Evidence, and audit refs produced from selected repositories."
            ),
            "evidence_ref": (
                "evidence_manifest:h01-packet/group-4#sha256="
                f"{json_hash([packet_file_map['delivery-evidence.json'], packet_file_map['redaction-audit-report.json']])}"
            ),
            "redaction_status": "redacted",
        },
        {
            "required_evidence_index": 5,
            "required_evidence": (
                "Independent zero-write proof covering permissions, declared actions, routes, UI/CLI, "
                "and observed calls."
            ),
            "evidence_ref": (
                "evidence_manifest:h01-packet/group-5#sha256="
                f"{json_hash([packet_file_map['permission-snapshot.md'], packet_file_map['zero-write-proof.json'], packet_file_map['review-decision.md']])}"
            ),
            "redaction_status": "redacted",
        },
    ]
    return draft


def connector_human_gate_h02_record_draft_from_packet(
    packet_validation_report: dict[str, Any],
) -> dict[str, Any] | None:
    scenario_id = "CS-CH-H02"
    if packet_validation_report.get("scenario_id") != scenario_id:
        return None
    if packet_validation_report.get("packet_structurally_complete") is not True:
        return None
    definition = CONNECTOR_HUMAN_GATE_PACKAGE_DEFINITIONS[scenario_id]
    execution_queue_item = CONNECTOR_HUMAN_GATE_QUEUE_BY_SCENARIO[scenario_id]
    proposed_record_template = connector_human_gate_record_template(
        scenario_id,
        definition,
        execution_queue_item,
    )
    draft = deepcopy(proposed_record_template["record_template"])
    packet_file_map = connector_human_gate_packet_file_map(
        packet_validation_report.get("packet_files") or []
    )
    required_files = {
        item["packet_file"]
        for item in CONNECTOR_HUMAN_GATE_H02_EVIDENCE_PACKET_FILE_ITEMS
    }
    if required_files.difference(packet_file_map):
        return None

    def ref(prefix: str, packet_file: str) -> str:
        return connector_human_gate_packet_hash_ref(
            prefix,
            scenario_id,
            packet_file,
            str(packet_file_map[packet_file]["sha256"]),
        )

    draft["device_os_version_redacted"] = ref("macos_device", "macos-review-scope.md")
    draft["consent_record"] = ref("consent_record", "permission-consent-record.md")
    draft["permission_state_snapshot"] = ref(
        "macos_permission",
        "permission-consent-record.md",
    )
    draft["first_sample_ref"] = ref("capture_sample", "first-sample-record.json")
    draft["pause_revoke_timestamps"] = [
        ref("capture_pause", "pause-proof.json"),
        ref("capture_revoke", "revoke-proof.json"),
    ]
    draft["screenshots_or_recording_ref"] = ref(
        "redacted_recording",
        "permission-consent-record.md",
    )
    draft["audit_refs"] = [
        ref("audit", "first-sample-record.json"),
        ref("audit", "pause-proof.json"),
        ref("audit", "revoke-proof.json"),
        ref("audit", "audit-redaction-report.json"),
    ]
    draft["evidence_packet_manifest"] = [
        {
            "required_evidence_index": 1,
            "required_evidence": (
                "Recording or screenshots of consent, permission grant, pause, and revoke."
            ),
            "evidence_ref": (
                "evidence_manifest:h02-packet/group-1#sha256="
                f"{json_hash([packet_file_map['permission-consent-record.md'], packet_file_map['pause-proof.json'], packet_file_map['revoke-proof.json']])}"
            ),
            "redaction_status": "redacted",
        },
        {
            "required_evidence_index": 2,
            "required_evidence": (
                "Status transcript showing permission and capture state transitions."
            ),
            "evidence_ref": (
                "evidence_manifest:h02-packet/group-2#sha256="
                f"{json_hash([packet_file_map['initial-disabled-state.json'], packet_file_map['pause-proof.json'], packet_file_map['revoke-proof.json']])}"
            ),
            "redaction_status": "redacted",
        },
        {
            "required_evidence_index": 3,
            "required_evidence": (
                "First bounded sample reference with owner/workspace scope."
            ),
            "evidence_ref": (
                "evidence_manifest:h02-packet/group-3#sha256="
                f"{json_hash([packet_file_map['macos-review-scope.md'], packet_file_map['first-sample-record.json']])}"
            ),
            "redaction_status": "redacted",
        },
        {
            "required_evidence_index": 4,
            "required_evidence": (
                "Proof that pause and revoke stop capture or block sample attempts."
            ),
            "evidence_ref": (
                "evidence_manifest:h02-packet/group-4#sha256="
                f"{json_hash([packet_file_map['pause-proof.json'], packet_file_map['revoke-proof.json'], packet_file_map['retention-export-delete-notes.md'], packet_file_map['audit-redaction-report.json'], packet_file_map['review-decision.md']])}"
            ),
            "redaction_status": "redacted",
        },
    ]
    return draft


def connector_human_gate_h03_record_draft_from_packet(
    packet_validation_report: dict[str, Any],
) -> dict[str, Any] | None:
    scenario_id = "CS-CH-H03"
    if packet_validation_report.get("scenario_id") != scenario_id:
        return None
    if packet_validation_report.get("packet_structurally_complete") is not True:
        return None
    definition = CONNECTOR_HUMAN_GATE_PACKAGE_DEFINITIONS[scenario_id]
    execution_queue_item = CONNECTOR_HUMAN_GATE_QUEUE_BY_SCENARIO[scenario_id]
    proposed_record_template = connector_human_gate_record_template(
        scenario_id,
        definition,
        execution_queue_item,
    )
    draft = deepcopy(proposed_record_template["record_template"])
    packet_file_map = connector_human_gate_packet_file_map(
        packet_validation_report.get("packet_files") or []
    )
    required_files = {
        item["packet_file"]
        for item in CONNECTOR_HUMAN_GATE_H03_EVIDENCE_PACKET_FILE_ITEMS
    }
    if required_files.difference(packet_file_map):
        return None

    def ref(prefix: str, packet_file: str) -> str:
        return connector_human_gate_packet_hash_ref(
            prefix,
            scenario_id,
            packet_file,
            str(packet_file_map[packet_file]["sha256"]),
        )

    draft["browser_profile_redacted"] = ref("chrome_profile", "chrome-review-scope.md")
    draft["extension_version"] = ref("chrome_extension", "chrome-review-scope.md")
    draft["permission_pages_or_recording_ref"] = ref(
        "chrome_permission",
        "permission-baseline.md",
    )
    draft["active_tab_capture_ref"] = ref(
        "chrome_active_tab_capture",
        "active-tab-capture-proof.json",
    )
    draft["allowlist_auto_capture_ref"] = ref(
        "chrome_allowlist_capture",
        "allowlist-auto-capture-proof.json",
    )
    draft["sensitive_block_ref"] = ref(
        "chrome_sensitive_policy",
        "sensitive-page-policy-proof.json",
    )
    draft["pause_revoke_ref"] = ref("chrome_pause_revoke", "pause-revoke-proof.json")
    draft["audit_refs"] = [
        ref("audit", "active-tab-capture-proof.json"),
        ref("audit", "allowlist-auto-capture-proof.json"),
        ref("audit", "sensitive-page-policy-proof.json"),
        ref("audit", "pause-revoke-proof.json"),
        ref("audit", "timeline-diagnostics.json"),
        ref("audit", "audit-redaction-report.json"),
    ]
    draft["evidence_packet_manifest"] = [
        {
            "required_evidence_index": 1,
            "required_evidence": (
                "Recording or screenshots of permission UX and capture controls."
            ),
            "evidence_ref": (
                "evidence_manifest:h03-packet/group-1#sha256="
                f"{json_hash([packet_file_map['permission-baseline.md'], packet_file_map['chrome-review-scope.md']])}"
            ),
            "redaction_status": "redacted",
        },
        {
            "required_evidence_index": 2,
            "required_evidence": (
                "Timeline covering active-tab capture, allowlist auto-capture, sensitive-page handling, pause, and revoke."
            ),
            "evidence_ref": (
                "evidence_manifest:h03-packet/group-2#sha256="
                f"{json_hash([packet_file_map['active-tab-capture-proof.json'], packet_file_map['allowlist-auto-capture-proof.json'], packet_file_map['sensitive-page-policy-proof.json'], packet_file_map['pause-revoke-proof.json'], packet_file_map['timeline-diagnostics.json']])}"
            ),
            "redaction_status": "redacted",
        },
        {
            "required_evidence_index": 3,
            "required_evidence": (
                "Policy and audit refs for allowed, blocked, and degraded browser capture decisions."
            ),
            "evidence_ref": (
                "evidence_manifest:h03-packet/group-3#sha256="
                f"{json_hash([packet_file_map['active-tab-capture-proof.json'], packet_file_map['allowlist-auto-capture-proof.json'], packet_file_map['sensitive-page-policy-proof.json'], packet_file_map['audit-redaction-report.json']])}"
            ),
            "redaction_status": "redacted",
        },
        {
            "required_evidence_index": 4,
            "required_evidence": "Human privacy acceptance or rejection note.",
            "evidence_ref": (
                "evidence_manifest:h03-packet/group-4#sha256="
                f"{json_hash([packet_file_map['privacy-issue-log.md'], packet_file_map['retention-export-delete-notes.md'], packet_file_map['review-decision.md']])}"
            ),
            "redaction_status": "redacted",
        },
    ]
    return draft


def connector_human_gate_h05_record_draft_from_packet(
    packet_validation_report: dict[str, Any],
) -> dict[str, Any] | None:
    scenario_id = "CS-CH-H05"
    if packet_validation_report.get("scenario_id") != scenario_id:
        return None
    if packet_validation_report.get("packet_structurally_complete") is not True:
        return None
    definition = CONNECTOR_HUMAN_GATE_PACKAGE_DEFINITIONS[scenario_id]
    execution_queue_item = CONNECTOR_HUMAN_GATE_QUEUE_BY_SCENARIO[scenario_id]
    proposed_record_template = connector_human_gate_record_template(
        scenario_id,
        definition,
        execution_queue_item,
    )
    draft = deepcopy(proposed_record_template["record_template"])
    packet_file_map = connector_human_gate_packet_file_map(
        packet_validation_report.get("packet_files") or []
    )
    required_files = {
        item["packet_file"]
        for item in CONNECTOR_HUMAN_GATE_H05_EVIDENCE_PACKET_FILE_ITEMS
    }
    if required_files.difference(packet_file_map):
        return None

    def ref(prefix: str, packet_file: str) -> str:
        return connector_human_gate_packet_hash_ref(
            prefix,
            scenario_id,
            packet_file,
            str(packet_file_map[packet_file]["sha256"]),
        )

    required_evidence = list(definition["required_human_record"]["required_evidence"])
    draft["approved_provider"] = ref("non_github_provider", "live-action-scope.md")
    draft["reversible_test_target"] = ref(
        "reversible_target",
        "live-action-scope.md",
    )
    draft["rollback_or_compensation_plan"] = ref(
        "rollback_compensation",
        "rollback-compensation-proof.md",
    )
    draft["approval_ref"] = ref("approval", "approval-record.md")
    draft["redacted_request_result"] = ref(
        "redacted_action_result",
        "execution-transcript.json",
    )
    draft["provider_receipt"] = ref("provider_receipt", "execution-transcript.json")
    draft["idempotency_evidence"] = ref(
        "idempotency",
        "idempotency-replay-proof.json",
    )
    draft["audit_refs"] = [
        ref("audit", "pre-execution-safety-envelope.json"),
        ref("audit", "approval-record.md"),
        ref("audit", "execution-transcript.json"),
        ref("audit", "outcome-reingest-proof.json"),
        ref("audit", "boundary-redaction-report.json"),
    ]
    draft["evidence_packet_manifest"] = [
        {
            "required_evidence_index": 1,
            "required_evidence": required_evidence[0],
            "evidence_ref": (
                "evidence_manifest:h05-packet/group-1#sha256="
                f"{json_hash([packet_file_map['live-action-scope.md'], packet_file_map['approval-record.md'], packet_file_map['github-exclusion-proof.json']])}"
            ),
            "redaction_status": "redacted",
        },
        {
            "required_evidence_index": 2,
            "required_evidence": required_evidence[1],
            "evidence_ref": (
                "evidence_manifest:h05-packet/group-2#sha256="
                f"{json_hash([packet_file_map['live-action-scope.md'], packet_file_map['pre-execution-safety-envelope.json'], packet_file_map['rollback-compensation-proof.md']])}"
            ),
            "redaction_status": "redacted",
        },
        {
            "required_evidence_index": 3,
            "required_evidence": required_evidence[2],
            "evidence_ref": (
                "evidence_manifest:h05-packet/group-3#sha256="
                f"{json_hash([packet_file_map['execution-transcript.json'], packet_file_map['provider-state-delta.md'], packet_file_map['boundary-redaction-report.json']])}"
            ),
            "redaction_status": "redacted",
        },
        {
            "required_evidence_index": 4,
            "required_evidence": required_evidence[3],
            "evidence_ref": (
                "evidence_manifest:h05-packet/group-4#sha256="
                f"{json_hash([packet_file_map['idempotency-replay-proof.json'], packet_file_map['outcome-reingest-proof.json'], packet_file_map['review-decision.md']])}"
            ),
            "redaction_status": "redacted",
        },
    ]
    return draft


def connector_human_gate_h06_record_draft_from_packet(
    packet_validation_report: dict[str, Any],
) -> dict[str, Any] | None:
    scenario_id = "CS-CH-H06"
    if packet_validation_report.get("scenario_id") != scenario_id:
        return None
    if packet_validation_report.get("packet_structurally_complete") is not True:
        return None
    definition = CONNECTOR_HUMAN_GATE_PACKAGE_DEFINITIONS[scenario_id]
    execution_queue_item = CONNECTOR_HUMAN_GATE_QUEUE_BY_SCENARIO[scenario_id]
    proposed_record_template = connector_human_gate_record_template(
        scenario_id,
        definition,
        execution_queue_item,
    )
    draft = deepcopy(proposed_record_template["record_template"])
    packet_file_map = connector_human_gate_packet_file_map(
        packet_validation_report.get("packet_files") or []
    )
    required_files = {
        item["packet_file"]
        for item in CONNECTOR_HUMAN_GATE_H06_EVIDENCE_PACKET_FILE_ITEMS
    }
    if required_files.difference(packet_file_map):
        return None

    def ref(prefix: str, packet_file: str) -> str:
        return connector_human_gate_packet_hash_ref(
            prefix,
            scenario_id,
            packet_file,
            str(packet_file_map[packet_file]["sha256"]),
        )

    required_evidence = list(definition["required_human_record"]["required_evidence"])
    draft["participant_profile_redacted"] = ref(
        "participant_profile",
        "participant-consent-redacted.md",
    )
    draft["task_script_ref"] = ref("task_script", "task-script.md")
    draft["fixture_workspace_ref"] = ref(
        "fixture_workspace",
        "fixture-workspace-seed.json",
    )
    draft["timed_task_notes"] = ref("timed_task_notes", "timed-task-notes.md")
    draft["screenshots_or_recording_ref"] = ref(
        "study_recording",
        "screenshots-or-recording-manifest.md",
    )
    draft["scoring_rubric"] = ref("scoring_rubric", "rubric-scores.json")
    draft["evidence_packet_manifest"] = [
        {
            "required_evidence_index": 1,
            "required_evidence": required_evidence[0],
            "evidence_ref": (
                "evidence_manifest:h06-packet/group-1#sha256="
                f"{json_hash([packet_file_map['study-scope.md'], packet_file_map['fixture-workspace-seed.json'], packet_file_map['task-script.md']])}"
            ),
            "redaction_status": "redacted",
        },
        {
            "required_evidence_index": 2,
            "required_evidence": required_evidence[1],
            "evidence_ref": (
                "evidence_manifest:h06-packet/group-2#sha256="
                f"{json_hash([packet_file_map['participant-consent-redacted.md'], packet_file_map['timed-task-notes.md'], packet_file_map['screenshots-or-recording-manifest.md']])}"
            ),
            "redaction_status": "redacted",
        },
        {
            "required_evidence_index": 3,
            "required_evidence": required_evidence[2],
            "evidence_ref": (
                "evidence_manifest:h06-packet/group-3#sha256="
                f"{json_hash([packet_file_map['connected-sources-comprehension.json'], packet_file_map['capture-inbox-watch-result-comprehension.json'], packet_file_map['privacy-control-comprehension.json'], packet_file_map['action-boundary-comprehension.json'], packet_file_map['rubric-scores.json']])}"
            ),
            "redaction_status": "redacted",
        },
        {
            "required_evidence_index": 4,
            "required_evidence": required_evidence[3],
            "evidence_ref": (
                "evidence_manifest:h06-packet/group-4#sha256="
                f"{json_hash([packet_file_map['issue-log.md'], packet_file_map['audit-redaction-report.json'], packet_file_map['review-decision.md']])}"
            ),
            "redaction_status": "redacted",
        },
    ]
    return draft


def connector_human_gate_record_draft_from_packet(
    packet_validation_report: dict[str, Any],
) -> dict[str, Any] | None:
    scenario_id = str(packet_validation_report.get("scenario_id") or "")
    if scenario_id == "CS-CH-H01":
        return connector_human_gate_h01_record_draft_from_packet(packet_validation_report)
    if scenario_id == "CS-CH-H02":
        return connector_human_gate_h02_record_draft_from_packet(packet_validation_report)
    if scenario_id == "CS-CH-H03":
        return connector_human_gate_h03_record_draft_from_packet(packet_validation_report)
    if scenario_id == "CS-CH-H04":
        return connector_human_gate_h04_record_draft_from_packet(packet_validation_report)
    if scenario_id == "CS-CH-H05":
        return connector_human_gate_h05_record_draft_from_packet(packet_validation_report)
    if scenario_id == "CS-CH-H06":
        return connector_human_gate_h06_record_draft_from_packet(packet_validation_report)
    if scenario_id == "CS-CH-H07":
        return connector_human_gate_h07_record_draft_from_packet(packet_validation_report)
    return None


def connector_human_gate_evidence_packet_file_contract(scenario_id: str) -> dict[str, Any] | None:
    packet_items = CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_FILE_ITEMS_BY_SCENARIO.get(scenario_id)
    packet_dir = CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY_BY_SCENARIO.get(
        scenario_id
    )
    if packet_items is None or packet_dir is None:
        return None
    packet_file_items = [
        {
            **deepcopy(item),
            "required": True,
            "raw_packet_file_contents_persisted_by_validator": False,
        }
        for item in packet_items
    ]
    scaffold_plan = connector_human_gate_evidence_packet_file_scaffold_plan(scenario_id) or []
    return {
        "schema_version": CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_FILE_CONTRACT_SCHEMA,
        "scenario_id": scenario_id,
        "status": "operator_preparation_only",
        "validation_scope": "acceptance_packet_file_shape_only",
        "required_packet_files": packet_file_items,
        "required_packet_file_count": len(packet_file_items),
        "packet_file_names": [str(item["packet_file"]) for item in packet_file_items],
        "raw_packet_file_contents_recorded_by_report": False,
        "packet_file_contents_persisted_by_report": False,
        "packet_file_contents_persisted_by_validator": False,
        "review_input_only": True,
        "acceptance_sufficient": False,
        "product_claim_allowed": False,
        "pass_claim_allowed": False,
        "invalid_value_report_shape": "packet_file_names_only",
        "packet_file_scaffold_plan_available": True,
        "packet_file_scaffold_directory": packet_dir,
        "packet_file_scaffold_plan": scaffold_plan,
        "packet_file_scaffold_commands": [
            str(item["command"]) for item in scaffold_plan
        ],
        "packet_file_scaffold_command_count": len(scaffold_plan),
        "packet_file_scaffold_plan_executed_by_report": False,
        "packet_file_scaffold_plan_review_input_only": True,
        "packet_file_scaffold_plan_acceptance_sufficient": False,
    }


def connector_human_gate_field_ref_value_valid(value: Any, contract_item: dict[str, Any]) -> bool:
    prefixes = [
        str(prefix)
        for prefix in contract_item.get("accepted_ref_prefixes", [])
        if isinstance(prefix, str) and prefix
    ]
    if not prefixes:
        return False

    def ref_valid(ref_value: Any) -> bool:
        return (
            isinstance(ref_value, str)
            and bool(ref_value.strip())
            and any(ref_value.strip().startswith(prefix) for prefix in prefixes)
        )

    container = str(contract_item.get("accepted_container") or "")
    if container == "string":
        return ref_valid(value)
    if container == "non_empty_string_list":
        return isinstance(value, list) and bool(value) and all(ref_valid(item) for item in value)
    if container == "string_or_non_empty_string_list":
        return ref_valid(value) or (
            isinstance(value, list) and bool(value) and all(ref_valid(item) for item in value)
        )
    return False


def connector_human_gate_invalid_field_ref_shapes(
    record: dict[str, Any],
    field_ref_contract: dict[str, Any] | None,
) -> list[str]:
    if not isinstance(field_ref_contract, dict):
        return []
    invalid_fields: list[str] = []
    for item in field_ref_contract.get("required_field_ref_items", []):
        if not isinstance(item, dict):
            continue
        field = str(item.get("field") or "")
        if not field:
            continue
        value = record.get(field)
        if value is None or value == "" or value == [] or value == {}:
            continue
        if not connector_human_gate_field_ref_value_valid(value, item):
            invalid_fields.append(field)
    return sorted(set(invalid_fields))


def connector_human_gate_redaction_guidance(
    scenario_id: str,
    definition: dict[str, Any],
    execution_queue_item: dict[str, Any],
) -> dict[str, Any]:
    required_human_record = deepcopy(definition["required_human_record"])
    required_fields = list(required_human_record.get("required_fields", []))
    required_evidence = list(required_human_record.get("required_evidence", []))
    required_perspective_roles = [
        str(perspective["role"])
        for perspective in definition.get("senior_review_perspectives", [])
        if perspective.get("role")
    ]
    field_ref_contract = connector_human_gate_field_ref_contract(scenario_id)
    guidance = {
        "schema_version": CONNECTOR_HUMAN_GATE_REDACTION_GUIDANCE_SCHEMA,
        "scenario_id": scenario_id,
        "status": "operator_guidance_only",
        "raw_secret_values_allowed": False,
        "raw_provider_payloads_allowed": False,
        "raw_evidence_values_allowed": False,
        "raw_record_body_persisted_by_validator": False,
        "raw_record_path_persisted_by_validator": False,
        "sensitive_marker_scan_required_by_validator": True,
        "allowed_redaction_statuses": list(CONNECTOR_HUMAN_GATE_ALLOWED_REDACTION_STATUSES),
        "sensitive_marker_policy": {
            "schema_version": CONNECTOR_SENSITIVE_MARKER_POLICY_SCHEMA,
            "marker_types": list(CONNECTOR_SENSITIVE_MARKER_TYPES),
            "finding_fields": list(CONNECTOR_SENSITIVE_MARKER_FINDING_FIELDS),
            "raw_match_values_returned": False,
            "raw_match_values_persisted": False,
            "fingerprints_only": True,
            "validator_structural_error": "sensitive_marker_detected",
            "operator_action": (
                "Replace any matching material with a redacted evidence reference or "
                "public-safe summary before submitting the reviewer record."
            ),
        },
        "field_guidance": {
            field: {
                "required": True,
                "accepted_value_shape": "redacted reference, artifact id, or public-safe summary",
                "raw_secret_values_allowed": False,
                "raw_provider_payloads_allowed": False,
            }
            for field in required_fields
        },
        "senior_review_perspective_findings": {
            "required_roles": required_perspective_roles,
            "accepted_value_shape": "redacted finding summary per required senior-review perspective",
            "raw_secret_values_allowed": False,
            "raw_provider_payloads_allowed": False,
            "persisted_by_validator": False,
        },
        "evidence_packet_manifest": {
            "required_evidence_count": len(required_evidence),
            "required_evidence_labels": required_evidence,
            "evidence_ref_shape": "redacted evidence reference or public-safe artifact id",
            "evidence_ref_uniqueness_required": True,
            "redaction_status_required": True,
            "allowed_redaction_statuses": list(CONNECTOR_HUMAN_GATE_ALLOWED_REDACTION_STATUSES),
            "values_persisted_by_validator": False,
        },
        "dependency_human_gate_refs": {
            "required_gates": list(execution_queue_item["depends_on"]),
            "accepted_ref_prefix": "connector_human_gate_record_validation:",
            "must_reference_structurally_valid_accept_validation": True,
        },
        "operator_warning": (
            "Do not paste raw secrets, provider payloads, local file paths, private user data, "
            "or unredacted external transcripts into the reviewer record. Use redacted refs or "
            "public-safe summaries; validate-record will fail on detected sensitive markers."
        ),
    }
    if field_ref_contract is not None:
        guidance["field_ref_contract"] = field_ref_contract
        field_ref_items = {
            str(item["field"]): item
            for item in field_ref_contract["required_field_ref_items"]
            if isinstance(item, dict) and item.get("field")
        }
        for field, item in field_ref_items.items():
            if field in guidance["field_guidance"]:
                guidance["field_guidance"][field]["accepted_value_shape"] = (
                    "typed redacted evidence reference matching field_ref_contract"
                )
                guidance["field_guidance"][field]["accepted_container"] = item.get("accepted_container")
                guidance["field_guidance"][field]["accepted_ref_prefixes"] = item.get("accepted_ref_prefixes")
                guidance["field_guidance"][field]["raw_value_persisted_by_validator"] = False
    return guidance


def connector_human_gate_record_template_output_command(
    scenario_id: str,
    *,
    output_placeholder: str = "<reviewer-record-template.json>",
) -> str:
    return (
        f"cornerstone connector human-gate package --scenario {scenario_id} "
        f"--json --record-template-output {output_placeholder}"
    )


def connector_human_gate_h04_evidence_packet_workflow_commands(
    scenario_id: str,
    *,
    packet_dir_placeholder: str = CONNECTOR_HUMAN_GATE_H04_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY,
    record_output_placeholder: str = "<reviewer-record-draft.json>",
    validation_output_placeholder: str = "<redacted-validation-envelope.json>",
) -> dict[str, Any] | None:
    return connector_human_gate_evidence_packet_workflow_commands(
        scenario_id,
        packet_dir_placeholder=packet_dir_placeholder,
        record_output_placeholder=record_output_placeholder,
        validation_output_placeholder=validation_output_placeholder,
    )


def connector_human_gate_evidence_packet_workflow_commands(
    scenario_id: str,
    *,
    packet_dir_placeholder: str | None = None,
    record_output_placeholder: str = "<reviewer-record-draft.json>",
    validation_output_placeholder: str = "<redacted-validation-envelope.json>",
) -> dict[str, Any] | None:
    schema_version = CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_WORKFLOW_SCHEMA_BY_SCENARIO.get(
        scenario_id
    )
    claim_boundary = CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_WORKFLOW_CLAIM_BOUNDARY_BY_SCENARIO.get(
        scenario_id
    )
    default_packet_dir = CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_FILE_SCAFFOLD_DIRECTORY_BY_SCENARIO.get(
        scenario_id
    )
    if schema_version is None or claim_boundary is None or default_packet_dir is None:
        return None
    packet_dir_placeholder = packet_dir_placeholder or default_packet_dir
    command_rows = [
        {
            "step_order": 1,
            "phase": "inspect_evidence_packet_contract",
            "owner": "operator",
            "command": (
                "cornerstone connector human-gate evidence-packet-contract "
                f"--scenario {scenario_id} --json"
            ),
            "output_role": "required evidence manifest and redaction contract",
        },
        {
            "step_order": 2,
            "phase": "inspect_evidence_packet_file_contract",
            "owner": "operator",
            "command": (
                "cornerstone connector human-gate evidence-packet-file-contract "
                f"--scenario {scenario_id} --json"
            ),
            "output_role": "required acceptance packet file list and content expectations",
        },
        {
            "step_order": 3,
            "phase": "scaffold_packet_templates",
            "owner": "operator",
            "command": (
                "cornerstone connector human-gate evidence-packet-scaffold "
                f"--scenario {scenario_id} --packet-dir {packet_dir_placeholder} --json --write"
            ),
            "output_role": "blank local packet templates without overwriting existing evidence",
        },
        {
            "step_order": 4,
            "phase": "validate_packet_hashes",
            "owner": "operator",
            "command": (
                "cornerstone connector human-gate evidence-packet-validate "
                f"--scenario {scenario_id} --packet-dir {packet_dir_placeholder} --json"
            ),
            "output_role": "hash-only packet validation envelope",
        },
        {
            "step_order": 5,
            "phase": "draft_record_from_packet_hashes",
            "owner": "operator",
            "command": (
                "cornerstone connector human-gate evidence-packet-record-draft "
                f"--scenario {scenario_id} --packet-dir {packet_dir_placeholder} --json "
                f"--record-output {record_output_placeholder}"
            ),
            "output_role": "hash-only reviewer record draft; human decision fields remain human-owned",
        },
        {
            "step_order": 6,
            "phase": "validate_completed_reviewer_record",
            "owner": "operator",
            "command": connector_human_gate_record_validation_command(
                scenario_id,
                record_placeholder="<filled-reviewer-record.json>",
                output_placeholder=validation_output_placeholder,
            ),
            "output_role": "redacted structural validation envelope for the completed reviewer record",
        },
    ]
    return {
        "schema_version": schema_version,
        "scenario_id": scenario_id,
        "status": "operator_preparation_only",
        "packet_dir_placeholder": packet_dir_placeholder,
        "record_output_placeholder": record_output_placeholder,
        "validation_output_placeholder": validation_output_placeholder,
        "command_sequence": command_rows,
        "commands": [row["command"] for row in command_rows],
        "command_count": len(command_rows),
        "review_input_only": True,
        "acceptance_sufficient": False,
        "product_claim_allowed": False,
        "pass_claim_allowed": False,
        "dependency_unlock_allowed_by_workflow": False,
        "human_acceptance_collected_by_workflow": False,
        "raw_packet_file_contents_recorded_by_workflow": False,
        "packet_file_contents_persisted_by_workflow": False,
        "claim_boundary": claim_boundary,
    }


def connector_human_gate_reviewer_checklist(
    scenario_id: str,
    *,
    required_fields: list[str],
    required_evidence_packet_manifest: list[dict[str, Any]],
    required_perspective_roles: list[str],
    dependencies: list[str],
    record_template_output_command: str,
    validation_output_command: str,
) -> dict[str, Any]:
    field_ref_contract = connector_human_gate_field_ref_contract(scenario_id)
    field_ref_items_by_field = {}
    if field_ref_contract is not None:
        field_ref_items_by_field = {
            str(item["field"]): item
            for item in field_ref_contract["required_field_ref_items"]
            if isinstance(item, dict) and item.get("field")
        }
    checklist = {
        "schema_version": CONNECTOR_HUMAN_GATE_REVIEWER_CHECKLIST_SCHEMA,
        "scenario_id": scenario_id,
        "status": "operator_preparation_only",
        "product_claim_allowed_by_checklist": False,
        "pass_claim_allowed_by_checklist": False,
        "reviewer_record_validation_required": True,
        "record_template_output_command": record_template_output_command,
        "validation_output_command": validation_output_command,
        "required_field_items": [
            {
                "field": field,
                "required": True,
                "accepted_value_shape": (
                    "typed redacted evidence reference matching field_ref_contract"
                    if field in field_ref_items_by_field
                    else "redacted reference, artifact id, or public-safe summary"
                ),
                **(
                    {
                        "accepted_container": field_ref_items_by_field[field].get("accepted_container"),
                        "accepted_ref_prefixes": field_ref_items_by_field[field].get("accepted_ref_prefixes"),
                        "raw_value_persisted_by_validator": False,
                    }
                    if field in field_ref_items_by_field
                    else {}
                ),
                "raw_secret_values_allowed": False,
                "raw_provider_payloads_allowed": False,
            }
            for field in required_fields
        ],
        "senior_review_perspective_items": [
            {
                "role": role,
                "required": True,
                "accepted_value_shape": "redacted finding summary per required senior-review perspective",
                "persisted_by_validator": False,
                "raw_secret_values_allowed": False,
                "raw_provider_payloads_allowed": False,
            }
            for role in required_perspective_roles
        ],
        "evidence_packet_manifest_items": [
            {
                "required_evidence_index": item["required_evidence_index"],
                "required_evidence": item["required_evidence"],
                "evidence_ref_required": True,
                "evidence_ref_uniqueness_required": True,
                "redaction_status_required": True,
                "allowed_redaction_statuses": list(CONNECTOR_HUMAN_GATE_ALLOWED_REDACTION_STATUSES),
                "values_persisted_by_validator": False,
            }
            for item in required_evidence_packet_manifest
        ],
        "dependency_human_gate_ref_items": [
            {
                "scenario_id": dependency,
                "required": True,
                "accepted_ref_prefix": "connector_human_gate_record_validation:",
                "must_reference_structurally_valid_accept_validation": True,
            }
            for dependency in dependencies
        ],
        "completion_rule": (
            "Write the blank reviewer record template, fill every checklist item with "
            "redacted human evidence, then run validation_output_command. This checklist "
            "does not collect approval or promote HUMAN_REQUIRED rows."
        ),
    }
    if field_ref_contract is not None:
        checklist["field_ref_contract"] = field_ref_contract
    evidence_packet_workflow = connector_human_gate_evidence_packet_workflow_commands(
        scenario_id
    )
    if evidence_packet_workflow is not None:
        checklist["evidence_packet_workflow"] = evidence_packet_workflow
        checklist["evidence_packet_workflow_command_count"] = evidence_packet_workflow[
            "command_count"
        ]
        checklist["evidence_packet_workflow_claim_boundary"] = evidence_packet_workflow[
            "claim_boundary"
        ]
    return checklist


def connector_human_gate_delivery_unit_plan(
    scenario_id: str,
    definition: dict[str, Any],
    execution_queue_item: dict[str, Any],
    *,
    package_command: str,
    record_template_output_command: str,
    validation_command: str,
    validation_output_command: str,
) -> dict[str, Any]:
    required_human_record = deepcopy(definition["required_human_record"])
    dependencies = list(execution_queue_item["depends_on"])
    plan = {
        "schema_version": CONNECTOR_HUMAN_GATE_DELIVERY_UNIT_PLAN_SCHEMA,
        "scenario_id": scenario_id,
        "status": "operator_preparation_only",
        "current_verdict": "HUMAN_REQUIRED",
        "scenario_first_independent_delivery_unit": True,
        "product_claim_allowed_by_plan": False,
        "pass_claim_allowed_by_plan": False,
        "approval_collected_by_plan": False,
        "dependency_unlock_allowed_by_plan": False,
        "gate_category": execution_queue_item["gate_category"],
        "review_order": execution_queue_item["order"],
        "depends_on_human_gates": dependencies,
        "stop_or_reject_when": execution_queue_item["stop_or_reject_when"],
        "senior_review_perspective_sequence": [
            {
                "role": perspective["role"],
                "decision": perspective["decision"],
                "required_before_human_acceptance": True,
            }
            for perspective in definition.get("senior_review_perspectives", [])
        ],
        "lifecycle_steps": [
            {
                "step_order": 1,
                "phase": "research_from_senior_perspectives",
                "owner": "human_reviewer",
                "action": "Review each senior-review perspective and collect missing external evidence before judging the gate.",
                "evidence_required": "Filled senior_review_perspective_findings for every required role.",
            },
            {
                "step_order": 2,
                "phase": "define_implementation_approach",
                "owner": "human_reviewer",
                "action": execution_queue_item["primary_operator_action"],
                "evidence_required": execution_queue_item["required_proof_package"],
            },
            {
                "step_order": 3,
                "phase": "execute_smallest_complete_rehearsal",
                "owner": "human_reviewer",
                "action": "Run the minimum external rehearsal needed to satisfy this scenario's required human record.",
                "evidence_required": "Every required field and evidence-packet manifest row is filled with redacted refs.",
            },
            {
                "step_order": 4,
                "phase": "refactor_or_remediate_before_acceptance",
                "owner": "human_reviewer",
                "action": "Record issues, exceptions, or remediation needs instead of accepting a partial rehearsal.",
                "evidence_required": "issues_or_exceptions plus reject/blocker notes when any stop condition appears.",
            },
            {
                "step_order": 5,
                "phase": "verify_record_structure",
                "owner": "ai_validator",
                "action": validation_command,
                "evidence_required": "Structural validation envelope remains HUMAN_REQUIRED and redacted.",
            },
            {
                "step_order": 6,
                "phase": "document_scenario_result",
                "owner": "human_reviewer",
                "action": "Store the completed reviewer record, validation envelope, and updated scenario result trail.",
                "evidence_required": "Dated ACCEPT or REJECT reviewer record plus validation ref.",
            },
            {
                "step_order": 7,
                "phase": "move_to_next_gate_only_after_dependency_rule",
                "owner": "operator",
                "action": "Use the next selector to advance only after dependency unlock evidence exists.",
                "evidence_required": "Structurally valid ACCEPT validation refs for all dependency gates.",
            },
        ],
        "command_sequence": {
            "package": package_command,
            "record_template_output": record_template_output_command,
            "validate_record": validation_command,
            "validate_record_output": validation_output_command,
            "readiness_report": "cornerstone connector human-gate report --json",
            "next_selector": "cornerstone connector human-gate next --json",
        },
        "required_human_record_summary": {
            "allowed_decision_values": list(required_human_record.get("decision_values", [])),
            "required_fields": list(required_human_record.get("required_fields", [])),
            "required_evidence": list(required_human_record.get("required_evidence", [])),
        },
        "dependency_rule": {
            "required_dependency_human_gates": dependencies,
            "accepted_ref_prefix": "connector_human_gate_record_validation:",
            "only_structurally_valid_accept_records_unlock_dependencies": True,
            "structurally_valid_reject_records_do_not_unlock_dependencies": True,
        },
        "documentation_rule": (
            "Complete one human-gate result before moving to the next dependency-gated row. "
            "This plan is preparation metadata only and cannot promote HUMAN_REQUIRED to PASS."
        ),
    }
    evidence_packet_workflow = connector_human_gate_evidence_packet_workflow_commands(
        scenario_id
    )
    if evidence_packet_workflow is not None:
        plan["evidence_packet_workflow"] = evidence_packet_workflow
        plan["evidence_packet_workflow_command_count"] = evidence_packet_workflow[
            "command_count"
        ]
        plan["evidence_packet_workflow_claim_boundary"] = evidence_packet_workflow[
            "claim_boundary"
        ]
        plan["command_sequence"]["evidence_packet_workflow"] = evidence_packet_workflow[
            "commands"
        ]
    return plan


def connector_human_gate_delivery_unit_plan_summary(plan: dict[str, Any] | None) -> dict[str, Any]:
    plan = plan if isinstance(plan, dict) else {}
    lifecycle_steps = plan.get("lifecycle_steps") if isinstance(plan.get("lifecycle_steps"), list) else []
    perspective_sequence = (
        plan.get("senior_review_perspective_sequence")
        if isinstance(plan.get("senior_review_perspective_sequence"), list)
        else []
    )
    return {
        "schema_version": "cs.connector_human_gate_delivery_unit_plan_summary.v1",
        "scenario_delivery_unit_plan_schema_version": plan.get("schema_version"),
        "scenario_delivery_unit_plan_ready": (
            plan.get("schema_version") == CONNECTOR_HUMAN_GATE_DELIVERY_UNIT_PLAN_SCHEMA
            and plan.get("scenario_first_independent_delivery_unit") is True
            and len(lifecycle_steps) == 7
            and len(perspective_sequence) >= 6
            and plan.get("product_claim_allowed_by_plan") is False
            and plan.get("pass_claim_allowed_by_plan") is False
            and plan.get("approval_collected_by_plan") is False
            and plan.get("dependency_unlock_allowed_by_plan") is False
        ),
        "scenario_delivery_unit_plan_lifecycle_step_count": len(lifecycle_steps),
        "scenario_delivery_unit_plan_senior_review_perspective_count": len(perspective_sequence),
        "scenario_delivery_unit_plan_product_claim_allowed": plan.get("product_claim_allowed_by_plan") is True,
        "scenario_delivery_unit_plan_pass_claim_allowed": plan.get("pass_claim_allowed_by_plan") is True,
        "scenario_delivery_unit_plan_approval_collected": plan.get("approval_collected_by_plan") is True,
        "scenario_delivery_unit_plan_dependency_unlock_allowed": (
            plan.get("dependency_unlock_allowed_by_plan") is True
        ),
    }


def connector_human_gate_record_template(
    scenario_id: str,
    definition: dict[str, Any],
    execution_queue_item: dict[str, Any],
) -> dict[str, Any]:
    required_human_record = deepcopy(definition["required_human_record"])
    required_fields = list(required_human_record.get("required_fields", []))
    required_evidence = list(required_human_record.get("required_evidence", []))
    required_evidence_packet_manifest = [
        {
            "required_evidence_index": index,
            "required_evidence": str(evidence),
        }
        for index, evidence in enumerate(required_evidence, start=1)
    ]
    record_template: dict[str, Any] = {
        "scenario_id": scenario_id,
        "decision": "",
        "reviewer": "",
        "review_timestamp": "",
    }
    list_hint_tokens = (
        "refs",
        "repositories",
        "ledger",
        "timestamps",
        "transcripts",
        "evidence",
        "notes",
        "logs",
    )
    for field in required_fields:
        if field in record_template:
            continue
        record_template[field] = [] if any(token in field for token in list_hint_tokens) else ""
    required_perspective_roles = [
        str(perspective["role"])
        for perspective in definition.get("senior_review_perspectives", [])
        if perspective.get("role")
    ]
    record_template["senior_review_perspective_findings"] = {
        role: ""
        for role in required_perspective_roles
    }
    record_template["evidence_packet_manifest"] = [
        {
            "required_evidence_index": item["required_evidence_index"],
            "required_evidence": item["required_evidence"],
            "evidence_ref": "",
            "redaction_status": "",
        }
        for item in required_evidence_packet_manifest
    ]
    dependencies = list(execution_queue_item["depends_on"])
    if dependencies:
        record_template["dependency_human_gate_refs"] = {dependency: "" for dependency in dependencies}
    validation_command = connector_human_gate_record_validation_command(scenario_id)
    validation_output_command = connector_human_gate_record_validation_command(
        scenario_id,
        output_placeholder="<redacted-validation-envelope.json>",
    )
    record_template_output_command = connector_human_gate_record_template_output_command(scenario_id)
    reviewer_checklist = connector_human_gate_reviewer_checklist(
        scenario_id,
        required_fields=required_fields,
        required_evidence_packet_manifest=required_evidence_packet_manifest,
        required_perspective_roles=required_perspective_roles,
        dependencies=dependencies,
        record_template_output_command=record_template_output_command,
        validation_output_command=validation_output_command,
    )
    redaction_guidance = connector_human_gate_redaction_guidance(
        scenario_id,
        definition,
        execution_queue_item,
    )
    field_ref_contract = connector_human_gate_field_ref_contract(scenario_id)
    template = {
        "schema_version": CONNECTOR_HUMAN_GATE_RECORD_TEMPLATE_SCHEMA,
        "scenario_id": scenario_id,
        "status": "blank_template_requires_human_evidence",
        "record_template": record_template,
        "required_fields": required_fields,
        "required_evidence": required_evidence,
        "required_evidence_packet_manifest": required_evidence_packet_manifest,
        "allowed_redaction_statuses": list(CONNECTOR_HUMAN_GATE_ALLOWED_REDACTION_STATUSES),
        "allowed_decision_values": list(required_human_record.get("decision_values", [])),
        "required_senior_review_perspectives": required_perspective_roles,
        "dependency_human_gates": dependencies,
        "format_rules": {
            "review_timestamp": "ISO-8601 timestamp with timezone, for example 2026-06-24T12:00:00Z",
        },
        "record_template_output_command": record_template_output_command,
        "validation_command": validation_command,
        "validation_output_command": validation_output_command,
        "reviewer_checklist": reviewer_checklist,
        "redaction_guidance": redaction_guidance,
        "template_rule": (
            "This blank template is operator-preparation data only. It is intentionally "
            "not structurally valid until a human reviewer fills every required field "
            "with redacted external evidence references, each senior review "
            "perspective finding, and every evidence packet manifest entry."
        ),
        "product_claim_allowed_by_template": False,
        "pass_claim_allowed_by_template": False,
    }
    if field_ref_contract is not None:
        template["field_ref_contract"] = field_ref_contract
    return template


def connector_human_gate_record_validation_command(
    scenario_id: str,
    *,
    record_placeholder: str = "<filled-json>",
    output_placeholder: str | None = None,
) -> str:
    command = (
        f"cornerstone connector human-gate validate-record --scenario {scenario_id} "
        f"--record-file {record_placeholder} --json"
    )
    if output_placeholder:
        command = f"{command} --output {output_placeholder}"
    return command


def connector_human_review_timestamp_valid(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    timestamp_text = value.strip()
    if timestamp_text.endswith("Z"):
        timestamp_text = f"{timestamp_text[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(timestamp_text)
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() is not None


def connector_human_gate_validation_is_structurally_valid(validation: dict[str, Any]) -> bool:
    scenario_id = str(validation.get("scenario_id") or "")
    return (
        validation.get("schema_version") == CONNECTOR_HUMAN_GATE_RECORD_VALIDATION_SCHEMA
        and scenario_id in CONNECTOR_HUMAN_GATE_QUEUE_BY_SCENARIO
        and validation.get("status") == "record_structurally_valid"
        and validation.get("validation_scope") == "structure_and_safety_only"
        and validation.get("matrix_status_after_validation") == "HUMAN_REQUIRED"
        and validation.get("product_claim_allowed") is False
        and validation.get("pass_claim_allowed_by_validator") is False
        and validation.get("decision_recorded_by_validator") is False
        and validation.get("evidence_packet_manifest_complete") is True
        and validation.get("evidence_packet_manifest_recorded_by_validator") is False
        and (validation.get("non_mutation_evidence") or {}).get("record_body_persisted_by_validator") is False
        and (validation.get("non_mutation_evidence") or {}).get("record_path_persisted_by_validator") is False
        and (validation.get("non_mutation_evidence") or {}).get("senior_review_perspective_findings_persisted_by_validator") is False
        and (validation.get("non_mutation_evidence") or {}).get("evidence_packet_manifest_values_persisted_by_validator") is False
        and (validation.get("negative_evidence") or {}).get("record_body_persisted_by_validator", 0) == 0
        and (validation.get("negative_evidence") or {}).get("record_path_persisted_by_validator", 0) == 0
        and (validation.get("negative_evidence") or {}).get("human_decision_value_persisted_by_validator", 0) == 0
        and (validation.get("negative_evidence") or {}).get("senior_review_perspective_findings_persisted_by_validator", 0) == 0
        and (validation.get("negative_evidence") or {}).get("evidence_packet_manifest_values_persisted_by_validator", 0) == 0
    )


def connector_human_gate_validation_allows_dependency_unlock(validation: dict[str, Any]) -> bool:
    return (
        connector_human_gate_validation_is_structurally_valid(validation)
        and validation.get("dependency_unlock_allowed_by_validator") is True
    )


def connector_human_gate_validation_ref_id(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = value.strip()
    prefix = "connector_human_gate_record_validation:"
    if text.startswith(prefix):
        return text[len(prefix):]
    if text.startswith("cshval_"):
        return text
    return ""


def connector_human_gate_validation_issue_summary(validation: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(validation, dict):
        return None

    def safe_list(key: str) -> list[Any]:
        value = validation.get(key)
        return list(value) if isinstance(value, list) else []

    issue_list_fields = [
        "missing_required_fields",
        "empty_required_fields",
        "invalid_field_formats",
        "invalid_required_field_ref_shapes",
        "missing_senior_review_perspectives",
        "empty_senior_review_perspectives",
        "invalid_senior_review_perspective_roles",
        "missing_evidence_packet_manifest_items",
        "empty_evidence_packet_manifest_items",
        "invalid_evidence_packet_manifest_items",
        "duplicate_evidence_packet_manifest_items",
        "missing_dependency_human_gates",
        "invalid_dependency_human_gate_refs",
    ]
    issue_details = {field: safe_list(field) for field in issue_list_fields}
    issue_details["duplicate_evidence_packet_manifest_refs"] = safe_list(
        "duplicate_evidence_packet_manifest_ref_fingerprints"
    )
    issue_counts = {field: len(values) for field, values in issue_details.items()}
    structural_errors = safe_list("structural_errors")
    sensitive_marker_findings = validation.get("sensitive_marker_findings", 0)
    if not isinstance(sensitive_marker_findings, int):
        sensitive_marker_findings = 0
    issue_counts["sensitive_marker_findings"] = sensitive_marker_findings
    issue_categories = [
        field
        for field, count in issue_counts.items()
        if count
    ]
    return {
        "schema_version": "cs.connector_human_gate_validation_issue_summary.v1",
        "status": validation.get("status"),
        "validation_scope": validation.get("validation_scope"),
        "matrix_status_after_validation": validation.get("matrix_status_after_validation"),
        "structural_correction_required": validation.get("status") != "record_structurally_valid",
        "structural_error_count": len(structural_errors),
        "structural_errors": structural_errors,
        "issue_categories": issue_categories,
        "issue_counts": issue_counts,
        "issue_details": issue_details,
        "sensitive_marker_finding_count": sensitive_marker_findings,
        "dependency_unlock_ready": validation.get("dependency_unlock_allowed_by_validator") is True,
        "dependency_unlock_blocked_reason": validation.get("dependency_unlock_blocked_reason"),
        "operator_next_step": (
            "fix_structural_errors_and_rerun_validate_record"
            if validation.get("status") != "record_structurally_valid"
            else "use_validation_ref_for_dependent_human_gates"
            if validation.get("dependency_unlock_allowed_by_validator") is True
            else "record_is_structurally_valid_but_does_not_unlock_dependencies"
        ),
        "product_claim_allowed": False,
        "pass_claim_allowed": False,
        "structural_validation_is_human_acceptance": False,
        "human_acceptance_requires_owner_promotion": True,
        "completion_claim_boundary": CONNECTOR_HUMAN_GATE_COMPLETION_CLAIM_BOUNDARY,
        "raw_record_body_included": False,
        "raw_record_path_included": False,
        "human_decision_value_included": False,
        "senior_review_finding_text_included": False,
        "evidence_packet_manifest_values_included": False,
    }


def read_contract_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_delivery_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def scope_complete(scope: Any) -> bool:
    if not isinstance(scope, dict):
        return False
    for field in SCOPE_FIELDS:
        value = scope.get(field)
        if not isinstance(value, str) or not value.strip():
            return False
    return True


def _scope_matches(contract_scope: dict[str, Any], requested_scope: dict[str, str]) -> bool:
    return all(contract_scope.get(field) == requested_scope.get(field) for field in SCOPE_FIELDS)


def _contract_version_id(contract: dict[str, Any]) -> str:
    return f"cconver_{json_hash(contract)[:16]}"


def _setup_result_id(contract_version_id: str, source_policy: dict[str, Any], mappings: list[dict[str, Any]]) -> str:
    return f"csetup_{json_hash({'contract_version_id': contract_version_id, 'source_policy': source_policy, 'mappings': mappings})[:16]}"


def _source_policy_id(contract_version_id: str, policy_request: dict[str, Any]) -> str:
    return f"cspol_{json_hash({'contract_version_id': contract_version_id, 'policy_request': policy_request})[:16]}"


def _stream_id(contract_version_id: str, capability: str) -> str:
    return f"cstream_{json_hash({'contract_version_id': contract_version_id, 'capability': capability})[:16]}"


def _delivery_receipt_id(logical_delivery_id: str, artifact_id: str) -> str:
    return f"cdelrec_{json_hash({'logical_delivery_id': logical_delivery_id, 'artifact_id': artifact_id})[:16]}"


def _projection_snapshot_id(projection_id: str, checksum: str) -> str:
    return f"cproj_{json_hash({'projection_id': projection_id, 'checksum_sha256': checksum})[:16]}"


def _evidence_link_id(receipt_id: str, artifact_id: str) -> str:
    return f"celink_{json_hash({'delivery_receipt_id': receipt_id, 'artifact_id': artifact_id})[:16]}"


def _projection_policy_decision_id(scope: dict[str, str], contract_version_id: str, delivery: dict[str, Any]) -> str:
    return f"cpdec_{json_hash({'scope': scope, 'contract_version_id': contract_version_id, 'delivery_id': delivery.get('delivery_id'), 'projection_id': delivery.get('projection_id'), 'provider_event_id': delivery.get('provider_event_id')})[:16]}"


def _content_restriction_decision_id(scope: dict[str, str], contract_version_id: str, delivery: dict[str, Any]) -> str:
    return f"ccont_{json_hash({'scope': scope, 'contract_version_id': contract_version_id, 'delivery_id': delivery.get('delivery_id'), 'projection_id': delivery.get('projection_id'), 'provider_event_id': delivery.get('provider_event_id')})[:16]}"


def _source_external_id(delivery: dict[str, Any]) -> str:
    source_summary = delivery.get("source_summary") if isinstance(delivery.get("source_summary"), dict) else {}
    return str(
        source_summary.get("object_ref")
        or source_summary.get("source_ref")
        or delivery.get("projection_id")
        or delivery.get("provider_event_id")
        or "unknown_source"
    )


def _source_revision(delivery: dict[str, Any]) -> str:
    payload = delivery.get("payload") if isinstance(delivery.get("payload"), dict) else {}
    evidence_ref = delivery.get("evidence_ref") if isinstance(delivery.get("evidence_ref"), dict) else {}
    return str(payload.get("source_revision") or payload.get("updated_at") or evidence_ref.get("source_revision") or delivery.get("provider_event_id") or "")


def _source_content_hash(delivery: dict[str, Any]) -> str:
    payload = delivery.get("payload") if isinstance(delivery.get("payload"), dict) else {}
    stable_payload = {key: value for key, value in payload.items() if key not in {"updated_at", "source_revision", "revision"}}
    return json_hash(
        {
            "source_external_id": _source_external_id(delivery),
            "projection_type": delivery.get("projection_type"),
            "payload": stable_payload,
        }
    )


def _delivery_idempotency_key(scope: dict[str, str], contract_version_id: str, delivery: dict[str, Any]) -> str:
    return f"cdedup_{json_hash({'scope': scope, 'contract_version_id': contract_version_id, 'source_external_id': _source_external_id(delivery), 'projection_type': delivery.get('projection_type'), 'source_content_hash': _source_content_hash(delivery)})[:16]}"


def _content_source_key(scope: dict[str, str], contract_version_id: str, source_external_id: str) -> str:
    return f"csrc_{json_hash({'scope': scope, 'contract_version_id': contract_version_id, 'source_external_id': source_external_id})[:16]}"


def _content_version_id(scope: dict[str, str], contract_version_id: str, source_external_id: str, source_content_hash: str) -> str:
    return f"cver_{json_hash({'scope': scope, 'contract_version_id': contract_version_id, 'source_external_id': source_external_id, 'source_content_hash': source_content_hash})[:16]}"


def _ack_outbox_id(receipt_id: str) -> str:
    return f"cack_{json_hash({'delivery_receipt_id': receipt_id})[:16]}"


def _sync_metadata(delivery: dict[str, Any]) -> dict[str, Any]:
    sync_metadata = delivery.get("sync")
    return sync_metadata if isinstance(sync_metadata, dict) else {}


def _sync_provider_event_key_parts(delivery: dict[str, Any]) -> dict[str, str]:
    sync_metadata = _sync_metadata(delivery)
    source_summary = delivery.get("source_summary") if isinstance(delivery.get("source_summary"), dict) else {}
    provider_event_id = str(delivery.get("provider_event_id") or "")
    provider_event_parts = provider_event_id.split(":")
    inferred_action = provider_event_parts[-2] if len(provider_event_parts) >= 2 else ""
    return {
        "provider_installation_id": str(sync_metadata.get("provider_installation_id") or delivery.get("provider_pack_id") or ""),
        "repository_ref": str(sync_metadata.get("repository_ref") or source_summary.get("source_ref") or ""),
        "object_ref": str(sync_metadata.get("object_ref") or source_summary.get("object_ref") or _source_external_id(delivery)),
        "action": str(sync_metadata.get("action") or inferred_action or "event"),
        "source_revision": str(sync_metadata.get("source_revision") or _source_revision(delivery)),
    }


def _sync_provider_event_key(delivery: dict[str, Any]) -> str:
    sync_metadata = _sync_metadata(delivery)
    explicit = sync_metadata.get("provider_event_key")
    if explicit:
        return str(explicit)
    return f"csyncevt_{json_hash(_sync_provider_event_key_parts(delivery))[:16]}"


def _sync_cursor_storage_id(scope: dict[str, str], contract_id: str, cursor_id: str) -> str:
    return f"ccursor_{json_hash({'scope': scope, 'contract_id': contract_id, 'cursor_id': cursor_id})[:16]}"


def _sync_signal_receipt_id(scope: dict[str, str], contract_id: str, delivery: dict[str, Any], signal: str) -> str:
    return f"csig_{json_hash({'scope': scope, 'contract_id': contract_id, 'signal': signal, 'delivery_id': delivery.get('delivery_id'), 'provider_event_key': _sync_provider_event_key(delivery)})[:16]}"


def _sync_reconciliation_id(scope: dict[str, str], cursor_id: str | None = None) -> str:
    return f"csyncrec_{json_hash({'scope': scope, 'cursor_id': cursor_id or 'all'})[:16]}"


def _provider_failure_state_id(scope: dict[str, str], contract_id: str, source_ref: str, failure_mode: str) -> str:
    return f"cghfail_{json_hash({'scope': scope, 'contract_id': contract_id, 'source_ref': source_ref, 'failure_mode': failure_mode})[:16]}"


def _capture_permission_probe_id(scope: dict[str, str], source_id: str, platform: str, permission_state: str) -> str:
    return f"ccpp_{json_hash({'scope': scope, 'source_id': source_id, 'platform': platform, 'permission_state': permission_state})[:16]}"


def _watch_source_consent_id(scope: dict[str, str], source_id: str, decision: str, purpose: str) -> str:
    return f"cwsc_{json_hash({'scope': scope, 'source_id': source_id, 'decision': decision, 'purpose': purpose})[:16]}"


def _capture_guard_decision_id(
    scope: dict[str, str],
    source_id: str,
    platform: str,
    permission_state: str,
    consent_record_id: str | None,
) -> str:
    return f"ccgd_{json_hash({'scope': scope, 'source_id': source_id, 'platform': platform, 'permission_state': permission_state, 'consent_record_id': consent_record_id})[:16]}"


def _chrome_active_tab_permission_event_id(scope: dict[str, str], payload: dict[str, Any]) -> str:
    invocation = payload.get("invocation", {}) if isinstance(payload.get("invocation"), dict) else {}
    active_tab = payload.get("active_tab", {}) if isinstance(payload.get("active_tab"), dict) else {}
    return f"catperm_{json_hash({'scope': scope, 'event_id': invocation.get('event_id'), 'tab': active_tab.get('tab_id_hash'), 'url': active_tab.get('url_hash') or active_tab.get('url')})[:16]}"


def _chrome_active_tab_payload_id(scope: dict[str, str], payload: dict[str, Any]) -> str:
    return f"catpay_{json_hash({'scope': scope, 'payload': payload})[:16]}"


def _chrome_active_tab_policy_decision_id(scope: dict[str, str], payload_id: str, reason_codes: list[str]) -> str:
    return f"catpol_{json_hash({'scope': scope, 'payload_id': payload_id, 'reason_codes': reason_codes})[:16]}"


def _chrome_active_tab_summary_id(scope: dict[str, str], payload_id: str, policy_decision_id: str) -> str:
    return f"catsum_{json_hash({'scope': scope, 'payload_id': payload_id, 'policy_decision_id': policy_decision_id})[:16]}"


def _chrome_auto_capture_config_id(scope: dict[str, str], config: dict[str, Any]) -> str:
    return f"cacfg_{json_hash({'scope': scope, 'config': config})[:16]}"


def _chrome_auto_capture_trigger_id(scope: dict[str, str], trigger_payload: dict[str, Any]) -> str:
    trigger = trigger_payload.get("trigger", {}) if isinstance(trigger_payload.get("trigger"), dict) else {}
    page = trigger_payload.get("page", {}) if isinstance(trigger_payload.get("page"), dict) else {}
    return f"catrig_{json_hash({'scope': scope, 'idempotency_key': trigger.get('idempotency_key'), 'event_id': trigger.get('event_id'), 'url_hash': page.get('url_hash') or page.get('url')})[:16]}"


def _chrome_auto_capture_policy_decision_id(
    scope: dict[str, str],
    trigger_id: str,
    reason_codes: list[str],
) -> str:
    return f"capol_{json_hash({'scope': scope, 'trigger_id': trigger_id, 'reason_codes': reason_codes})[:16]}"


def _chrome_auto_capture_summary_id(scope: dict[str, str], trigger_id: str, policy_decision_id: str) -> str:
    return f"casum_{json_hash({'scope': scope, 'trigger_id': trigger_id, 'policy_decision_id': policy_decision_id})[:16]}"


def _chrome_sensitive_page_policy_decision_id(scope: dict[str, str], case_payload: dict[str, Any]) -> str:
    case_id = str(case_payload.get("case_id") or "")
    page = case_payload.get("page", {}) if isinstance(case_payload.get("page"), dict) else {}
    return f"csppol_{json_hash({'scope': scope, 'case_id': case_id, 'page': page, 'case_hash': json_hash(case_payload)})[:16]}"


def _chrome_sensitive_page_degraded_payload_id(
    scope: dict[str, str],
    policy_decision_id: str,
    case_payload: dict[str, Any],
) -> str:
    return f"cspdeg_{json_hash({'scope': scope, 'policy_decision_id': policy_decision_id, 'case_hash': json_hash(case_payload)})[:16]}"


def _chrome_sensitive_page_history_item_id(scope: dict[str, str], policy_decision_id: str) -> str:
    return f"csphis_{json_hash({'scope': scope, 'policy_decision_id': policy_decision_id})[:16]}"


def _capture_inbox_item_id(scope: dict[str, str], source_id: str, summary_id: str) -> str:
    return f"capin_{json_hash({'scope': scope, 'source_id': source_id, 'summary_id': summary_id})[:16]}"


def _capture_lifecycle_source_state_id(
    scope: dict[str, str],
    target_kind: str,
    source_id: str,
    target_id: str | None,
) -> str:
    return f"caplife_{json_hash({'scope': scope, 'target_kind': target_kind, 'source_id': source_id, 'target_id': target_id or source_id})[:16]}"


def _capture_lifecycle_decision_id(
    scope: dict[str, str],
    action: str,
    target_kind: str,
    source_id: str,
    target_id: str | None,
    decision_basis: dict[str, Any],
) -> str:
    return f"capldec_{json_hash({'scope': scope, 'action': action, 'target_kind': target_kind, 'source_id': source_id, 'target_id': target_id or source_id, 'decision_basis': decision_basis})[:16]}"


def _capture_lifecycle_export_id(scope: dict[str, str], source_id: str, export_basis: dict[str, Any]) -> str:
    return f"caplexp_{json_hash({'scope': scope, 'source_id': source_id, 'export_basis': export_basis})[:16]}"


def _capture_lifecycle_deletion_receipt_id(
    scope: dict[str, str],
    source_id: str,
    mode: str,
    deletion_basis: dict[str, Any],
) -> str:
    return f"capldel_{json_hash({'scope': scope, 'source_id': source_id, 'mode': mode, 'deletion_basis': deletion_basis})[:16]}"


def _capture_result_review_id(scope: dict[str, str], result_id: str, decision: str) -> str:
    return f"caprev_{json_hash({'scope': scope, 'result_id': result_id, 'decision': decision})[:16]}"


def _watch_observation_id(scope: dict[str, str], result_key: str, observation: dict[str, Any]) -> str:
    return f"wobs_{json_hash({'scope': scope, 'result_key': result_key, 'observation': observation})[:16]}"


def _watch_inference_id(scope: dict[str, str], result_key: str, inference: dict[str, Any]) -> str:
    return f"winf_{json_hash({'scope': scope, 'result_key': result_key, 'inference': inference})[:16]}"


def _watch_result_id(
    scope: dict[str, str],
    result_key: str,
    observation_ids: list[str],
    inference_ids: list[str],
) -> str:
    return f"wres_{json_hash({'scope': scope, 'result_key': result_key, 'observation_ids': observation_ids, 'inference_ids': inference_ids})[:16]}"


def _watch_result_correction_id(
    scope: dict[str, str],
    watch_result_id: str,
    inference_id: str,
    corrected_hypothesis: str,
) -> str:
    return f"wcor_{json_hash({'scope': scope, 'watch_result_id': watch_result_id, 'inference_id': inference_id, 'corrected_hypothesis': corrected_hypothesis})[:16]}"


def _watch_result_review_id(scope: dict[str, str], watch_result_id: str, decision: str) -> str:
    return f"wrev_{json_hash({'scope': scope, 'watch_result_id': watch_result_id, 'decision': decision})[:16]}"


def _connector_action_preflight_id(scope: dict[str, str], action_id: str, case_id: str) -> str:
    return f"capf_{json_hash({'scope': scope, 'action_id': action_id, 'case_id': case_id})[:16]}"


def _connector_action_preflight_review_id(scope: dict[str, str], action_id: str, preflight_id: str) -> str:
    return f"capfr_{json_hash({'scope': scope, 'action_id': action_id, 'preflight_id': preflight_id})[:16]}"


def _activity_sample_batch_id(scope: dict[str, str], source_id: str, samples: list[dict[str, Any]]) -> str:
    sample_keys = [str(sample.get("sample_id") or sample.get("event_key") or "") for sample in samples]
    return f"casb_{json_hash({'scope': scope, 'source_id': source_id, 'sample_keys': sample_keys})[:16]}"


def _activity_sessionization_id(scope: dict[str, str], sample_batch_id: str, algorithm_version: str) -> str:
    return f"csessrun_{json_hash({'scope': scope, 'sample_batch_id': sample_batch_id, 'algorithm_version': algorithm_version})[:16]}"


def _activity_session_projection_id(
    scope: dict[str, str],
    sample_batch_id: str,
    started_at: str,
    ended_at: str,
    sample_ids: list[str],
) -> str:
    return f"actsess_{json_hash({'scope': scope, 'sample_batch_id': sample_batch_id, 'started_at': started_at, 'ended_at': ended_at, 'sample_ids': sample_ids})[:16]}"


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return normalized or "watch-rule"


def _watch_rule_id(scope: dict[str, str], rule_key: str) -> str:
    return f"wrule_{json_hash({'scope': scope, 'rule_key': rule_key})[:16]}"


def _watch_rule_version_id(
    scope: dict[str, str],
    watch_rule_id: str,
    version_number: int,
    definition_hash: str,
) -> str:
    return f"wrver_{json_hash({'scope': scope, 'watch_rule_id': watch_rule_id, 'version_number': version_number, 'definition_hash': definition_hash})[:16]}"


def _watch_rule_policy_decision_id(
    scope: dict[str, str],
    watch_rule_id: str,
    version_number: int,
    action: str,
    decision_basis: dict[str, Any],
) -> str:
    return f"wrpdec_{json_hash({'scope': scope, 'watch_rule_id': watch_rule_id, 'version_number': version_number, 'action': action, 'decision_basis': decision_basis})[:16]}"


def _watch_rule_evaluation_trace_id(
    scope: dict[str, str],
    watch_rule_id: str,
    watch_rule_version_id: str,
    source_evidence_refs: list[str],
) -> str:
    return f"wreval_{json_hash({'scope': scope, 'watch_rule_id': watch_rule_id, 'watch_rule_version_id': watch_rule_version_id, 'source_evidence_refs': source_evidence_refs})[:16]}"


def _parse_iso_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _format_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _source_cursor_value(delivery: dict[str, Any]) -> str:
    payload = delivery.get("payload") if isinstance(delivery.get("payload"), dict) else {}
    return str(payload.get("updated_at") or _source_revision(delivery) or delivery.get("provider_event_id") or "")


def _delivery_retry_state_id(scope: dict[str, str], contract_version_id: str, delivery: dict[str, Any]) -> str:
    return f"cret_{json_hash({'scope': scope, 'contract_version_id': contract_version_id, 'delivery_id': delivery.get('delivery_id'), 'provider_event_id': delivery.get('provider_event_id'), 'projection_id': delivery.get('projection_id')})[:16]}"


def _delivery_quarantine_id(retry_state_id: str) -> str:
    return f"cquar_{json_hash({'retry_state_id': retry_state_id})[:16]}"


def _quarantine_replay_attempt_id(quarantine_id: str, attempt_count: int) -> str:
    return f"cqrp_{json_hash({'quarantine_id': quarantine_id, 'attempt_count': attempt_count})[:16]}"


def _raw_access_request_id(
    scope: dict[str, str],
    contract_version_id: str,
    evidence_ref_id: str,
    purpose: str,
    classification: str,
    ttl_seconds: int,
    max_reads: int,
) -> str:
    return f"crawreq_{json_hash({'scope': scope, 'contract_version_id': contract_version_id, 'evidence_ref_id': evidence_ref_id, 'purpose': purpose, 'classification': classification, 'ttl_seconds': ttl_seconds, 'max_reads': max_reads})[:16]}"


def _raw_access_grant_id(raw_access_request_id: str) -> str:
    return f"crawgrant_{json_hash({'raw_access_request_id': raw_access_request_id})[:16]}"


def _untrusted_content_review_id(scope: dict[str, str], contract_version_id: str, delivery_id: str, artifact_id: str) -> str:
    return f"cuntrust_{json_hash({'scope': scope, 'contract_version_id': contract_version_id, 'delivery_id': delivery_id, 'artifact_id': artifact_id})[:16]}"


def _retry_delay_seconds(attempt_no: int) -> int:
    return min(60 * (2 ** max(attempt_no - 1, 0)), 3600)


def _safe_delivery_ref(delivery: dict[str, Any]) -> dict[str, Any]:
    return {
        "delivery_id": delivery.get("delivery_id"),
        "projection_id": delivery.get("projection_id"),
        "provider_event_id": delivery.get("provider_event_id"),
        "provider_pack_id": delivery.get("provider_pack_id"),
        "common_capability": delivery.get("common_capability"),
        "projection_type": delivery.get("projection_type"),
        "source_summary": delivery.get("source_summary") if isinstance(delivery.get("source_summary"), dict) else {},
        "raw_provider_payload_included": False,
    }


def _source_ref_allowed(source_summary: dict[str, Any], selected_resources: list[str]) -> bool:
    if not selected_resources:
        return True
    source_ref = str(source_summary.get("source_ref") or "")
    object_ref = str(source_summary.get("object_ref") or "")
    return source_ref in selected_resources or any(object_ref.startswith(f"{resource}:") for resource in selected_resources)


def _selected_resource_scope(policy_request: dict[str, Any]) -> dict[str, Any]:
    selected_resources = [
        str(resource)
        for resource in policy_request.get("selected_resources", [])
        if isinstance(resource, str) and resource.strip()
    ]
    catalog = [
        item
        for item in policy_request.get("provider_resource_catalog", [])
        if isinstance(item, dict) and isinstance(item.get("source_ref"), str)
    ]
    available_count = len(catalog) if catalog else len(selected_resources)
    selected_set = set(selected_resources)
    catalog_selected = [item for item in catalog if str(item.get("source_ref")) in selected_set]
    selected_count = len(catalog_selected) if catalog else len(selected_resources)
    unselected_count = max(available_count - selected_count, 0)
    return {
        "schema_version": "cs.connector_selected_resource_scope.v1",
        "provider_kind": "github",
        "selection_mode": "explicit_selected_repositories",
        "namespace_scoped": True,
        "versioned": True,
        "selection_version_id": f"csel_{json_hash({'selected_resources': selected_resources, 'available_count': available_count})[:16]}",
        "available_resource_count": available_count,
        "selected_resource_count": selected_count,
        "unselected_resource_count": unselected_count,
        "selected_resources": selected_resources,
        "visible_to_product_resources": selected_resources,
        "unselected_resources_hidden_from_product": True,
        "organization_wide_fallback_enabled": False,
        "account_wide_fallback_enabled": False,
        "stores_opaque_source_refs_only": True,
        "credentials_exposed": False,
        "write_permissions_requested": False,
        "write_permissions_granted": False,
        "provider_mutation_capabilities": [],
        "requires_owner_review_to_expand": True,
    }


def _delivery_path_allowed(payload: dict[str, Any], source_summary: dict[str, Any], allowed_paths: list[str]) -> bool:
    path = payload.get("path") or source_summary.get("path")
    if not path:
        return True
    if not allowed_paths:
        return False
    return any(fnmatch.fnmatch(str(path), pattern) for pattern in allowed_paths)


def _policy_content_size_bytes(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def _declared_content_size_bytes(payload: dict[str, Any]) -> int:
    value = payload.get("size_bytes")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return _policy_content_size_bytes(payload)


def _path_matches_any(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def _mime_type_is_binary(mime_type: str) -> bool:
    if not mime_type:
        return False
    normalized = mime_type.lower()
    if normalized.startswith("text/") or normalized in TEXT_MIME_TYPES:
        return False
    return True


def _sensitive_marker_fingerprint(value: str) -> str:
    return f"sha256:{hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]}"


def _scan_sensitive_markers(text: str) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for marker_type, pattern in SENSITIVE_MARKER_PATTERNS:
        for match in pattern.finditer(text):
            matched = match.group(0)
            fingerprint = _sensitive_marker_fingerprint(matched)
            key = (marker_type, fingerprint)
            if key in seen:
                continue
            seen.add(key)
            findings.append({"marker_type": marker_type, "fingerprint": fingerprint, "length": len(matched)})
    return findings


def _redact_sensitive_markers(text: str) -> tuple[str, list[dict[str, Any]]]:
    findings = _scan_sensitive_markers(text)

    def replacement(match: re.Match[str]) -> str:
        return f"[REDACTED:{_sensitive_marker_fingerprint(match.group(0))}]"

    redacted = text
    for _, pattern in SENSITIVE_MARKER_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    return redacted, findings


def _redact_sensitive_payload(value: Any) -> tuple[Any, list[dict[str, Any]], list[str]]:
    findings: list[dict[str, Any]] = []
    redacted_fields: list[str] = []

    def visit(node: Any, path: str) -> Any:
        if isinstance(node, dict):
            return {key: visit(child, f"{path}.{key}") for key, child in node.items()}
        if isinstance(node, list):
            return [visit(child, f"{path}[{index}]") for index, child in enumerate(node)]
        if isinstance(node, str):
            redacted, node_findings = _redact_sensitive_markers(node)
            if node_findings:
                redacted_fields.append(path)
                findings.extend(node_findings)
            return redacted
        return node

    return visit(value, "payload"), findings, sorted(set(redacted_fields))


def _strip_content_fields(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        field: value
        for field, value in payload.items()
        if field in CONTENT_METADATA_FIELDS or field not in CONTENT_TEXT_FIELDS
    }


def _build_content_restriction_decision(
    *,
    delivery: dict[str, Any],
    requested_scope: dict[str, str],
    contract: dict[str, Any],
    setup_result: dict[str, Any],
    source_policy: dict[str, Any],
    normalized_payload: dict[str, Any],
    path_allowed: bool,
    content_size_bytes: int,
    declared_size_bytes: int,
) -> dict[str, Any]:
    payload = delivery.get("payload") if isinstance(delivery.get("payload"), dict) else {}
    source_summary = delivery.get("source_summary") if isinstance(delivery.get("source_summary"), dict) else {}
    path = str(payload.get("path") or source_summary.get("path") or "")
    mime_type = str(payload.get("mime_type") or "")
    max_content_bytes = source_policy.get("max_content_bytes")
    is_binary = bool(payload.get("binary")) or _mime_type_is_binary(mime_type)
    is_generated = bool(path and _path_matches_any(path, GENERATED_PATH_PATTERNS))
    redacted_payload, sensitive_findings, redacted_fields = _redact_sensitive_payload(normalized_payload)
    marker_types = sorted({finding["marker_type"] for finding in sensitive_findings})
    private_material_detected = "private_key_block" in marker_types

    action = "allow"
    durable_status = "full_permitted"
    reason_codes: list[str] = []
    safe_payload = redacted_payload if isinstance(redacted_payload, dict) else {}
    partial_status = "complete"

    if not path_allowed:
        action = "skip"
        durable_status = "skipped_before_product_state"
        safe_payload = {}
        partial_status = "skipped_path_denied"
        reason_codes.append("CS_CONNECTOR_CONTENT_PATH_SKIPPED")
    elif is_generated:
        action = "skip"
        durable_status = "skipped_before_product_state"
        safe_payload = {}
        partial_status = "skipped_generated_content"
        reason_codes.append("CS_CONNECTOR_CONTENT_GENERATED_SKIPPED")
    elif private_material_detected:
        action = "quarantine"
        durable_status = "quarantined_before_product_state"
        safe_payload = {}
        partial_status = "quarantined_sensitive_material"
        reason_codes.append("CS_CONNECTOR_CONTENT_SENSITIVE_MATERIAL_QUARANTINED")
    elif is_binary:
        action = "metadata_only"
        durable_status = "metadata_only_artifact"
        safe_payload = _strip_content_fields(safe_payload)
        partial_status = "metadata_only_binary"
        reason_codes.append("CS_CONNECTOR_CONTENT_BINARY_METADATA_ONLY")
    elif isinstance(max_content_bytes, int) and declared_size_bytes > max_content_bytes:
        action = "metadata_only"
        durable_status = "metadata_only_artifact"
        safe_payload = _strip_content_fields(safe_payload)
        partial_status = "metadata_only_size_limit"
        reason_codes.append("CS_CONNECTOR_CONTENT_SIZE_METADATA_ONLY")
    elif sensitive_findings:
        action = "redact"
        durable_status = "redacted_artifact"
        partial_status = "redacted_sensitive_markers"
        reason_codes.append("CS_CONNECTOR_CONTENT_REDACTED")

    return {
        "schema_version": CONTENT_RESTRICTION_DECISION_SCHEMA,
        "content_restriction_decision_id": _content_restriction_decision_id(
            requested_scope,
            contract["contract_version_id"],
            delivery,
        ),
        "scope": requested_scope,
        "contract_id": contract["contract_id"],
        "contract_version_id": contract["contract_version_id"],
        "setup_result_id": setup_result.get("setup_result_id"),
        "source_policy_id": source_policy.get("source_policy_id"),
        "delivery_id": delivery.get("delivery_id"),
        "projection_id": delivery.get("projection_id"),
        "provider_event_id": delivery.get("provider_event_id"),
        "projection_type": delivery.get("projection_type"),
        "source_external_id": _source_external_id(delivery),
        "source_revision": _source_revision(delivery),
        "path": path,
        "mime_type": mime_type,
        "allowed_paths": source_policy.get("allowed_paths", []),
        "path_allowed": path_allowed,
        "generated_content": is_generated,
        "binary_content": is_binary,
        "max_content_bytes": max_content_bytes,
        "content_size_bytes": content_size_bytes,
        "declared_size_bytes": declared_size_bytes,
        "action": action,
        "durable_content_status": durable_status,
        "partial_status": partial_status,
        "reason_codes": reason_codes,
        "normalized_payload": safe_payload,
        "sensitive_marker_scan": {
            "matches_detected": len(sensitive_findings),
            "marker_types": marker_types,
            "findings": sensitive_findings,
            "redacted_fields": redacted_fields,
            "raw_values_persisted": False,
            "raw_values_in_operator_output": False,
        },
        "redaction_applied": action == "redact",
        "metadata_only": action == "metadata_only",
        "quarantined": action == "quarantine",
        "skipped": action == "skip",
        "product_state_safe_to_use": action in {"allow", "redact", "metadata_only"},
        "artifact_may_be_created": action in {"allow", "redact", "metadata_only"},
        "receipt_may_be_created": action in {"allow", "redact", "metadata_only"},
        "raw_restricted_content_persisted": False,
        "raw_provider_payload_persisted": False,
        "source_remains_read_only": True,
        "helpful_resolution": "Use allowed paths, omit generated/private material, keep large or binary content metadata-only, and redact sensitive markers before Product state.",
        "created_at": utc_now(),
    }


def _merge_content_decision_refs(
    content_decision: dict[str, Any],
    *,
    evidence_refs: list[str] | None = None,
    audit_refs: list[str] | None = None,
    linked_records: dict[str, str | None] | None = None,
) -> dict[str, Any]:
    if evidence_refs:
        content_decision["evidence_refs"] = list(
            dict.fromkeys([*content_decision.get("evidence_refs", []), *evidence_refs])
        )
    if audit_refs:
        content_decision["audit_refs"] = list(dict.fromkeys([*content_decision.get("audit_refs", []), *audit_refs]))
    if linked_records:
        current_links = dict(content_decision.get("linked_records", {}))
        current_links.update({key: value for key, value in linked_records.items() if value})
        content_decision["linked_records"] = current_links
    return content_decision


def _restriction_summary(source_policy: dict[str, Any]) -> str:
    content_mode = source_policy.get("content_mode") or "metadata_only"
    if content_mode == "metadata_and_markdown_excerpt":
        return "Source Policy allows metadata and a markdown excerpt only; full body content remains omitted."
    if content_mode == "metadata_only":
        return "Source Policy allows metadata only; body content remains omitted."
    return f"Source Policy content mode {content_mode} is enforced before durable Product state."


def _activation_guidance(required_gaps: list[dict[str, Any]]) -> str | None:
    if not required_gaps:
        return None
    return "Choose a compatible Provider Pack, grant an approved permission, or change the contract before activation."


def _fixture_ref(contract: dict[str, Any]) -> str:
    source_path = Path(str(contract.get("source", {}).get("path", "connector_contract")))
    try:
        display_path = source_path.resolve().relative_to(Path.cwd().resolve())
    except ValueError:
        display_path = source_path
    return f"fixture:{display_path}"


def _capability_surface(capability: str) -> str:
    return CAPABILITY_SURFACES.get(capability, capability.replace("_", " ").replace(".", " "))


def _product_handler_contract_hash() -> str:
    return json_hash(PRODUCT_HANDLER_CONTRACT)


def _safe_status_explanation(setup_gap: dict[str, Any] | None, required_gaps: list[dict[str, Any]]) -> dict[str, Any] | None:
    if setup_gap:
        return {
            "schema_version": "cs.connector_status_explanation.v1",
            "reason_code": setup_gap["reason_code"],
            "cause": setup_gap["cause"],
            "impact": setup_gap["impact"],
            "resolution_steps": setup_gap["resolution_steps"],
            "safe_to_show_to_owner": True,
            "redaction": {
                "tokens_exposed": False,
                "secret_values_exposed": False,
                "raw_provider_response_exposed": False,
                "raw_local_paths_exposed": False,
                "direct_api_handles_exposed": False,
            },
        }
    if required_gaps:
        return {
            "schema_version": "cs.connector_status_explanation.v1",
            "reason_code": "CS_CONNECTOR_REQUIRED_CAPABILITY_MISSING",
            "cause": "A required connector capability has no compatible Provider Pack mapping.",
            "impact": "Connector activation is blocked until the required capability can be supplied.",
            "resolution_steps": [
                "Choose a compatible Provider Pack.",
                "Grant an approved permission.",
                "Change the contract to remove or make optional the unavailable capability.",
            ],
            "safe_to_show_to_owner": True,
            "redaction": {
                "tokens_exposed": False,
                "secret_values_exposed": False,
                "raw_provider_response_exposed": False,
                "raw_local_paths_exposed": False,
                "direct_api_handles_exposed": False,
            },
        }
    return None


def _list_subset(candidate: list[str], allowed: list[str]) -> bool:
    return set(candidate).issubset(set(allowed))


def _chrome_manifest_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if value in (None, ""):
        return []
    return [str(value)]


def _int_not_broader(candidate: int | None, base: int | None) -> bool:
    if candidate is None or base is None:
        return candidate == base
    return candidate <= base


def _normalized_source_policy_request(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    base_raw_access = base.get("raw_access", "denied")
    normalized = {
        "selected_resources": overrides.get("selected_resources") or base.get("selected_resources", []),
        "content_mode": overrides.get("content_mode") or base.get("content_mode", "metadata_only"),
        "max_content_bytes": overrides.get("max_content_bytes", base.get("max_content_bytes")),
        "allowed_paths": overrides.get("allowed_paths") or base.get("allowed_paths", []),
        "raw_access": base_raw_access if base_raw_access == RAW_ACCESS_ALLOWED_MODE else "denied",
        "raw_access_policy": base.get("raw_access_policy", {}) if isinstance(base.get("raw_access_policy"), dict) else {},
        "retention_days": overrides.get("retention_days", base.get("retention_days")),
    }
    return normalized


def _source_policy_broadening_issues(base: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if not _list_subset(candidate.get("selected_resources", []), base.get("selected_resources", [])):
        issues.append(
            {
                "code": "CS_CONNECTOR_SOURCE_POLICY_BROADENING_DENIED",
                "field": "selected_resources",
                "message": "Selected resources cannot broaden beyond the declared contract policy.",
            }
        )
    if not _list_subset(candidate.get("allowed_paths", []), base.get("allowed_paths", [])):
        issues.append(
            {
                "code": "CS_CONNECTOR_SOURCE_POLICY_BROADENING_DENIED",
                "field": "allowed_paths",
                "message": "Allowed paths cannot broaden beyond the declared contract policy.",
            }
        )
    if candidate.get("content_mode") != base.get("content_mode", "metadata_only"):
        issues.append(
            {
                "code": "CS_CONNECTOR_SOURCE_POLICY_BROADENING_DENIED",
                "field": "content_mode",
                "message": "Content mode cannot change during Source Policy confirmation.",
            }
        )
    if candidate.get("raw_access") != base.get("raw_access", "denied"):
        issues.append(
            {
                "code": "CS_CONNECTOR_SOURCE_POLICY_BROADENING_DENIED",
                "field": "raw_access",
                "message": "Raw access posture cannot broaden during Source Policy confirmation.",
            }
        )
    if not _int_not_broader(candidate.get("max_content_bytes"), base.get("max_content_bytes")):
        issues.append(
            {
                "code": "CS_CONNECTOR_SOURCE_POLICY_BROADENING_DENIED",
                "field": "max_content_bytes",
                "message": "Maximum content bytes cannot increase during Source Policy confirmation.",
            }
        )
    if not _int_not_broader(candidate.get("retention_days"), base.get("retention_days")):
        issues.append(
            {
                "code": "CS_CONNECTOR_SOURCE_POLICY_BROADENING_DENIED",
                "field": "retention_days",
                "message": "Retention days cannot increase during Source Policy confirmation.",
            }
        )
    return issues


def _source_policy_diff(base: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    changed: dict[str, dict[str, Any]] = {}
    narrowed: list[str] = []
    for field in ["selected_resources", "allowed_paths", "content_mode", "max_content_bytes", "raw_access", "raw_access_policy", "retention_days"]:
        before = base.get(field)
        after = candidate.get(field)
        if before == after:
            continue
        changed[field] = {"before": before, "after": after}
        if field in {"selected_resources", "allowed_paths"} and isinstance(before, list) and isinstance(after, list):
            if set(after).issubset(set(before)):
                narrowed.append(field)
        if field in {"max_content_bytes", "retention_days"} and isinstance(before, int) and isinstance(after, int) and after <= before:
            narrowed.append(field)
    return {
        "changed_fields": changed,
        "narrowed_fields": sorted(set(narrowed)),
        "broadened": False,
        "base_hash": json_hash(base),
        "candidate_hash": json_hash(candidate),
    }


def _github_write_action_findings(action: Any) -> list[str]:
    if not isinstance(action, dict):
        return []
    provider_text = " ".join(
        str(action.get(field, ""))
        for field in ["provider", "provider_kind", "target_provider", "source", "source_kind"]
    ).lower()
    method = str(action.get("method", "")).upper()
    endpoint = str(action.get("endpoint") or action.get("path") or action.get("url_path") or "").lower()
    capability_text = " ".join(
        str(action.get(field, ""))
        for field in ["common_capability", "capability", "action_type", "type", "operation", "name", "id"]
    ).lower()
    action_text = json.dumps(action, sort_keys=True).lower()
    source_control_context = (
        "github" in provider_text
        or "github" in action_text
        or "source_control" in provider_text
        or "source_control" in action_text
    )
    findings: list[str] = []
    if source_control_context and method in GITHUB_WRITE_HTTP_METHODS:
        findings.append(f"write_http_method:{method}")
    if method in GITHUB_WRITE_HTTP_METHODS and any(pattern in endpoint for pattern in GITHUB_WRITE_ENDPOINT_PATTERNS):
        findings.append(f"github_write_endpoint:{method} {endpoint}")
    for term in GITHUB_WRITE_FORBIDDEN_TERMS:
        if source_control_context and term in capability_text:
            findings.append(f"github_write_action:{term}")
        elif term in endpoint and method in GITHUB_WRITE_HTTP_METHODS:
            findings.append(f"github_write_endpoint_term:{term}")
    return sorted(set(findings))


def _github_write_cli_hits(root: Path) -> list[dict[str, str]]:
    main_path = root / "packages/cornerstone_cli/main.py"
    if not main_path.exists():
        return []
    text = main_path.read_text(errors="ignore")
    hits: list[dict[str, str]] = []
    for command in GITHUB_WRITE_FORBIDDEN_CLI_COMMANDS:
        if re.search(rf"add_parser\(\s*['\"]{re.escape(command)}['\"]", text):
            hits.append({"command": command, "source": str(main_path.relative_to(root))})
    return hits


def _github_write_endpoint_literal_hits(root: Path) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for rel_path in ["packages/cornerstone_cli/main.py", "packages/cornerstone_cli/runtime.py"]:
        path = root / rel_path
        if not path.exists():
            continue
        text = path.read_text(errors="ignore")
        for method in GITHUB_WRITE_HTTP_METHODS:
            for pattern in GITHUB_WRITE_ENDPOINT_PATTERNS:
                if method in text and pattern in text:
                    hits.append({"source": rel_path, "method": method, "endpoint_pattern": pattern})
    return hits


def github_write_guard_report(root: Path) -> dict[str, Any]:
    provider_pack_results: list[dict[str, Any]] = []
    provider_pack_write_mappings: list[dict[str, Any]] = []
    for pack_id, pack in FIXTURE_PROVIDER_PACKS.items():
        declared_actions = pack.get("declared_actions", [])
        action_findings = [
            {"index": index, "findings": findings}
            for index, action in enumerate(declared_actions)
            if (findings := _github_write_action_findings(action))
        ]
        capability_write_mappings = [
            capability
            for capability in pack.get("capabilities", {})
            if any(term in capability.lower() for term in GITHUB_WRITE_FORBIDDEN_TERMS)
        ]
        if action_findings or capability_write_mappings:
            provider_pack_write_mappings.append(
                {
                    "provider_pack_id": pack_id,
                    "action_findings": action_findings,
                    "capability_write_mappings": capability_write_mappings,
                }
            )
        provider_pack_results.append(
            {
                "provider_pack_id": pack_id,
                "transport": pack.get("transport"),
                "declared_action_count": len(declared_actions),
                "write_action_count": len(action_findings),
                "write_capability_count": len(capability_write_mappings),
                "write_mappings": action_findings + [{"capability": value} for value in capability_write_mappings],
            }
        )

    contract_results: list[dict[str, Any]] = []
    accepted_write_contract_count = 0
    for rel_path in ACTIVE_GITHUB_CONTRACT_FIXTURES:
        path = root / rel_path
        if not path.exists():
            contract_results.append({"path": rel_path, "status": "missing"})
            continue
        contract = read_contract_file(path)
        action_findings = [
            {"index": index, "findings": findings}
            for index, action in enumerate(contract.get("actions", []))
            if (findings := _github_write_action_findings(action))
        ]
        if action_findings:
            accepted_write_contract_count += 1
        contract_results.append(
            {
                "path": rel_path,
                "contract_id": contract.get("contract_id"),
                "action_count": len(contract.get("actions", [])),
                "write_action_count": len(action_findings),
                "write_action_findings": action_findings,
            }
        )

    controlled_attempts = [
        {
            **attempt,
            "status": "denied",
            "policy": "default_egress_deny",
            "direct_provider_access": False,
            "external_http_calls": 0,
            "provider_mutations": 0,
        }
        for attempt in GITHUB_WRITE_DIRECT_ATTEMPTS
    ]
    cli_hits = _github_write_cli_hits(root)
    endpoint_hits = _github_write_endpoint_literal_hits(root)
    source_control_actions_declared = sum(result["write_action_count"] for result in provider_pack_results) + sum(
        result.get("write_action_count", 0) for result in contract_results
    )
    negative = {
        "source_control_actions_declared": source_control_actions_declared,
        "provider_pack_write_mappings": len(provider_pack_write_mappings),
        "provider_mutations": sum(attempt["provider_mutations"] for attempt in controlled_attempts),
        "github_write_calls": sum(attempt["external_http_calls"] for attempt in controlled_attempts),
        "github_write_permissions_requested": 0,
        "github_write_cli_commands_exposed": len(cli_hits),
        "github_write_contracts_accepted": accepted_write_contract_count,
        "github_write_egress_allowed": len([attempt for attempt in controlled_attempts if attempt["status"] != "denied"]),
        "github_write_endpoint_literals": len(endpoint_hits),
    }
    return {
        "schema_version": GITHUB_WRITE_GUARD_SCHEMA,
        "provider": "github",
        "status": "pass" if all(value == 0 for value in negative.values()) else "fail",
        "provider_packs": provider_pack_results,
        "active_contracts": contract_results,
        "forbidden_cli_command_hits": cli_hits,
        "forbidden_endpoint_literal_hits": endpoint_hits,
        "controlled_egress_attempts": controlled_attempts,
        "negative_evidence": negative,
    }


def validate_connector_contract(contract: dict[str, Any], requested_scope: dict[str, str]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    if contract.get("schema_version") != SUPPORTED_CONTRACT_SCHEMA:
        issues.append(
            {
                "code": "CS_CONNECTOR_SCHEMA_UNSUPPORTED",
                "message": f"Contract schema must be {SUPPORTED_CONTRACT_SCHEMA}.",
                "path": "schema_version",
            }
        )
    if not contract.get("contract_id"):
        issues.append({"code": "CS_CONNECTOR_CONTRACT_ID_MISSING", "message": "contract_id is required.", "path": "contract_id"})
    if not contract.get("contract_version"):
        issues.append(
            {
                "code": "CS_CONNECTOR_CONTRACT_VERSION_MISSING",
                "message": "contract_version is required.",
                "path": "contract_version",
            }
        )
    scope = contract.get("scope")
    if not scope_complete(scope):
        issues.append({"code": "CS_CONNECTOR_SCOPE_INCOMPLETE", "message": "tenant/owner/namespace/workspace scope is required.", "path": "scope"})
    elif not _scope_matches(scope, requested_scope):
        issues.append({"code": "CS_CONNECTOR_SCOPE_MISMATCH", "message": "Contract scope must match the trusted CLI scope.", "path": "scope"})
    if not contract.get("purpose"):
        issues.append({"code": "CS_CONNECTOR_PURPOSE_MISSING", "message": "purpose is required.", "path": "purpose"})

    needs = contract.get("needs")
    if not isinstance(needs, list) or not needs:
        issues.append({"code": "CS_CONNECTOR_NEEDS_MISSING", "message": "At least one capability need is required.", "path": "needs"})
    else:
        for index, need in enumerate(needs):
            base = f"needs[{index}]"
            if not need.get("common_capability"):
                issues.append({"code": "CS_CONNECTOR_CAPABILITY_MISSING", "message": "common_capability is required.", "path": base})
            if "required" not in need:
                issues.append({"code": "CS_CONNECTOR_REQUIRED_FLAG_MISSING", "message": "required flag is required.", "path": base})
            projections = need.get("accepted_projection_types")
            if not isinstance(projections, list) or not projections:
                issues.append(
                    {
                        "code": "CS_CONNECTOR_PROJECTION_TYPES_MISSING",
                        "message": "accepted_projection_types must be a non-empty list.",
                        "path": base,
                    }
                )

    policy = contract.get("source_policy_request")
    if not isinstance(policy, dict):
        issues.append(
            {
                "code": "CS_CONNECTOR_SOURCE_POLICY_MISSING",
                "message": "source_policy_request is required.",
                "path": "source_policy_request",
            }
        )
    else:
        if not policy.get("selected_resources"):
            issues.append(
                {
                    "code": "CS_CONNECTOR_SELECTED_RESOURCES_MISSING",
                    "message": "source_policy_request.selected_resources is required.",
                    "path": "source_policy_request.selected_resources",
                }
            )
        raw_access = policy.get("raw_access", "denied")
        if raw_access not in {"denied", RAW_ACCESS_ALLOWED_MODE}:
            issues.append(
                {
                    "code": "CS_CONNECTOR_RAW_ACCESS_UNSUPPORTED",
                    "message": "raw_access must be denied or temporary_scoped.",
                    "path": "source_policy_request.raw_access",
                }
            )
        if raw_access == RAW_ACCESS_ALLOWED_MODE:
            raw_policy = policy.get("raw_access_policy")
            if not isinstance(raw_policy, dict):
                issues.append(
                    {
                        "code": "CS_CONNECTOR_RAW_ACCESS_POLICY_MISSING",
                        "message": "temporary_scoped raw access requires raw_access_policy.",
                        "path": "source_policy_request.raw_access_policy",
                    }
                )
            else:
                max_ttl = raw_policy.get("max_ttl_seconds")
                max_reads = raw_policy.get("max_reads")
                purposes = raw_policy.get("allowed_purposes")
                if not isinstance(max_ttl, int) or max_ttl <= 0:
                    issues.append(
                        {
                            "code": "CS_CONNECTOR_RAW_ACCESS_TTL_POLICY_INVALID",
                            "message": "raw_access_policy.max_ttl_seconds must be a positive integer.",
                            "path": "source_policy_request.raw_access_policy.max_ttl_seconds",
                        }
                    )
                if not isinstance(max_reads, int) or max_reads <= 0:
                    issues.append(
                        {
                            "code": "CS_CONNECTOR_RAW_ACCESS_READ_POLICY_INVALID",
                            "message": "raw_access_policy.max_reads must be a positive integer.",
                            "path": "source_policy_request.raw_access_policy.max_reads",
                        }
                    )
                if not isinstance(purposes, list) or not purposes:
                    issues.append(
                        {
                            "code": "CS_CONNECTOR_RAW_ACCESS_PURPOSE_POLICY_INVALID",
                            "message": "raw_access_policy.allowed_purposes must list approved purpose labels.",
                            "path": "source_policy_request.raw_access_policy.allowed_purposes",
                        }
                    )

    delivery = contract.get("delivery")
    if not isinstance(delivery, dict) or delivery.get("ack_required") is not True:
        issues.append(
            {
                "code": "CS_CONNECTOR_DELIVERY_ACK_REQUIRED",
                "message": "delivery.ack_required must be true.",
                "path": "delivery.ack_required",
            }
        )
    actions = contract.get("actions")
    if not isinstance(actions, list):
        issues.append({"code": "CS_CONNECTOR_ACTIONS_INVALID", "message": "actions must be a list.", "path": "actions"})
    else:
        for index, action in enumerate(actions):
            findings = _github_write_action_findings(action)
            if findings:
                issues.append(
                    {
                        "code": "CS_CONNECTOR_GITHUB_WRITE_ACTION_DENIED",
                        "message": "GitHub/source-control write actions are outside the read-only Connector Hub source scope.",
                        "path": f"actions[{index}]",
                        "findings": findings,
                    }
                )
    return issues


def provider_internal_findings(value: Any, path: str = "$") -> list[str]:
    findings: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = key.lower()
            if lowered in FORBIDDEN_PROVIDER_INTERNAL_KEYS:
                findings.append(f"{path}.{key}")
            findings.extend(provider_internal_findings(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            findings.extend(provider_internal_findings(child, f"{path}[{index}]"))
    return findings


def audit_product_surface() -> dict[str, Any]:
    normal_texts = [item["label"] for item in NORMAL_USER_PRODUCT_SURFACES] + CONNECTED_SOURCE_PRODUCT_COPY
    normal_hits = [
        {"term": term, "text": text}
        for term in FORBIDDEN_NORMAL_USER_TERMS
        for text in normal_texts
        if term.lower() in text.lower()
    ]
    return {
        "schema_version": "cs.connector_product_surface_audit.v1",
        "product_name": "CornerStone",
        "one_product_experience": True,
        "normal_user_navigation": NORMAL_USER_PRODUCT_SURFACES,
        "connected_source_surface": {
            "id": "connected_sources",
            "label": "Connected Sources",
            "owned_by_product": True,
            "requires_connectorhub_sub_product": False,
            "copy": CONNECTED_SOURCE_PRODUCT_COPY,
        },
        "admin_detail_surfaces": ADMIN_DETAIL_SURFACES,
        "progressive_disclosure": {
            "setup_gaps_admin_detail": True,
            "source_policy_admin_detail": True,
            "provider_pack_admin_detail": True,
            "retry_quarantine_admin_detail": True,
            "audit_admin_detail": True,
        },
        "native_cli": {
            "prefix": "cornerstone",
            "commands_begin_with_cornerstone": True,
            "connected_source_commands": [
                "cornerstone connector contract validate",
                "cornerstone connector setup plan",
                "cornerstone connector source-policy confirm",
                "cornerstone connector upgrade plan",
            ],
        },
        "forbidden_normal_user_terms": FORBIDDEN_NORMAL_USER_TERMS,
        "normal_user_forbidden_term_hits": normal_hits,
        "negative_counters": {
            "connectorhub_default_label_count": sum(
                1
                for text in normal_texts
                if "connectorhub" in text.lower() or "connector-hub" in text.lower()
            ),
            "connector_admin_first_required": 0,
            "sub_product_required": 0,
        },
    }


def connector_report_readiness_dimensions() -> dict[str, str]:
    return dict(REPORT_READINESS_DIMENSIONS)


def lint_connector_report_claims(report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("summary", {}) if isinstance(report, dict) else {}
    negative = report.get("negative_evidence", {}) if isinstance(report, dict) else {}
    dimensions = report.get("readiness_dimensions", {}) if isinstance(report, dict) else {}
    product_claim = str(summary.get("product_feature_claims", ""))
    scenario_results = report.get("scenario_results", []) if isinstance(report, dict) else []
    human_required = report.get("human_required", []) if isinstance(report, dict) else []
    forbidden_claim_terms = [
        "PRODUCTION_READY",
        "LIVE_PROVIDER_READY",
        "LIVE_READY",
        "PROD_READY",
    ]
    overclaim_issues: list[dict[str, Any]] = []
    if not product_claim.startswith("LOCAL_FIXTURE_"):
        overclaim_issues.append(
            {
                "code": "CS_CONNECTOR_REPORT_CLAIM_NOT_FIXTURE_SCOPED",
                "message": "Connector report product_feature_claims must remain local fixture-scoped.",
                "product_feature_claims": product_claim,
            }
        )
    for term in forbidden_claim_terms:
        if term in product_claim:
            overclaim_issues.append(
                {
                    "code": "CS_CONNECTOR_REPORT_FORBIDDEN_READINESS_TERM",
                    "message": "Connector report product_feature_claims implies live or production readiness.",
                    "term": term,
                }
            )
    expected_dimensions = connector_report_readiness_dimensions()
    for key, expected_value in expected_dimensions.items():
        if dimensions.get(key) != expected_value:
            overclaim_issues.append(
                {
                    "code": "CS_CONNECTOR_REPORT_READINESS_DIMENSION_MISMATCH",
                    "message": "Connector report readiness dimension is missing or incorrectly promoted.",
                    "dimension": key,
                    "expected": expected_value,
                    "actual": dimensions.get(key),
                }
            )
    if int(negative.get("production_readiness_overclaims", 0) or 0) != 0:
        overclaim_issues.append(
            {
                "code": "CS_CONNECTOR_REPORT_NEGATIVE_COUNTER_NONZERO",
                "message": "Connector report negative overclaim counter must remain zero.",
                "counter": "production_readiness_overclaims",
                "actual": negative.get("production_readiness_overclaims"),
            }
        )
    return {
        "schema_version": "cs.connector_report_lint.v1",
        "status": "pass" if not overclaim_issues else "fail",
        "product_feature_claims": product_claim,
        "scenario_count": len(scenario_results) if isinstance(scenario_results, list) else 0,
        "human_required_count": len(human_required) if isinstance(human_required, list) else 0,
        "readiness_dimensions": dimensions,
        "negative_overclaim_counter": len(overclaim_issues),
        "issues": overclaim_issues,
    }


class ConnectorRuntime:
    def __init__(self, store: LocalRuntimeStore) -> None:
        self.store = store
        self.root = store.state_dir / "connector"
        self.contract_dir = self.root / "contracts"
        self.latest_dir = self.root / "latest_contracts"
        self.setup_result_dir = self.root / "setup_results"
        self.source_policy_dir = self.root / "source_policies"
        self.upgrade_plan_dir = self.root / "upgrade_plans"
        self.delivery_receipt_dir = self.root / "delivery_receipts"
        self.projection_snapshot_dir = self.root / "projection_snapshots"
        self.evidence_link_dir = self.root / "evidence_links"
        self.ack_outbox_dir = self.root / "ack_outbox"
        self.delivery_retry_dir = self.root / "delivery_retries"
        self.quarantine_dir = self.root / "quarantine"
        self.delivery_dedupe_dir = self.root / "delivery_dedupe"
        self.content_version_dir = self.root / "content_versions"
        self.content_current_dir = self.root / "content_current"
        self.projection_policy_decision_dir = self.root / "projection_policy_decisions"
        self.content_restriction_decision_dir = self.root / "content_restriction_decisions"
        self.content_restricted_delivery_dir = self.root / "content_restricted_deliveries"
        self.raw_access_request_dir = self.root / "raw_access_requests"
        self.raw_access_grant_dir = self.root / "raw_access_grants"
        self.untrusted_content_review_dir = self.root / "untrusted_content_reviews"
        self.sync_signal_receipt_dir = self.root / "sync_signal_receipts"
        self.sync_cursor_dir = self.root / "sync_cursors"
        self.sync_reconciliation_dir = self.root / "sync_reconciliations"
        self.provider_failure_state_dir = self.root / "provider_failure_states"
        self.capture_permission_probe_dir = self.root / "capture_permission_probes"
        self.watch_source_consent_dir = self.root / "watch_source_consents"
        self.capture_guard_decision_dir = self.root / "capture_guard_decisions"
        self.capture_sample_dir = self.root / "capture_samples"
        self.activity_sample_batch_dir = self.root / "activity_sample_batches"
        self.activity_sessionization_dir = self.root / "activity_sessionizations"
        self.activity_session_dir = self.root / "activity_sessions"
        self.chrome_active_tab_permission_event_dir = self.root / "chrome_active_tab_permission_events"
        self.chrome_active_tab_payload_dir = self.root / "chrome_active_tab_payloads"
        self.chrome_active_tab_policy_decision_dir = self.root / "chrome_active_tab_policy_decisions"
        self.chrome_active_tab_summary_dir = self.root / "chrome_active_tab_summaries"
        self.chrome_auto_capture_config_dir = self.root / "chrome_auto_capture_configs"
        self.chrome_auto_capture_trigger_dir = self.root / "chrome_auto_capture_triggers"
        self.chrome_auto_capture_policy_decision_dir = self.root / "chrome_auto_capture_policy_decisions"
        self.chrome_auto_capture_summary_dir = self.root / "chrome_auto_capture_summaries"
        self.chrome_sensitive_page_policy_decision_dir = self.root / "chrome_sensitive_page_policy_decisions"
        self.chrome_sensitive_page_degraded_payload_dir = self.root / "chrome_sensitive_page_degraded_payloads"
        self.chrome_sensitive_page_history_item_dir = self.root / "chrome_sensitive_page_history_items"
        self.capture_inbox_item_dir = self.root / "capture_inbox_items"
        self.capture_lifecycle_source_state_dir = self.root / "capture_lifecycle_source_states"
        self.capture_lifecycle_decision_dir = self.root / "capture_lifecycle_decisions"
        self.capture_lifecycle_export_dir = self.root / "capture_lifecycle_exports"
        self.capture_lifecycle_deletion_receipt_dir = self.root / "capture_lifecycle_deletion_receipts"
        self.capture_result_review_dir = self.root / "capture_result_reviews"
        self.watch_observation_dir = self.root / "watch_observations"
        self.watch_inference_dir = self.root / "watch_inferences"
        self.watch_result_dir = self.root / "watch_results"
        self.watch_result_correction_dir = self.root / "watch_result_corrections"
        self.watch_result_review_dir = self.root / "watch_result_reviews"
        self.connector_action_preflight_dir = self.root / "connector_action_preflights"
        self.connector_action_preflight_review_dir = self.root / "connector_action_preflight_reviews"
        self.watch_rule_dir = self.root / "watch_rules"
        self.watch_rule_version_dir = self.root / "watch_rule_versions"
        self.watch_rule_policy_decision_dir = self.root / "watch_rule_policy_decisions"
        self.watch_rule_evaluation_trace_dir = self.root / "watch_rule_evaluation_traces"
        self.human_gate_package_dir = self.root / "human_gate_packages"
        self.human_gate_field_ref_contract_report_dir = self.root / "human_gate_field_ref_contract_reports"
        self.human_gate_evidence_packet_contract_report_dir = (
            self.root / "human_gate_evidence_packet_contract_reports"
        )
        self.human_gate_evidence_packet_file_contract_report_dir = (
            self.root / "human_gate_evidence_packet_file_contract_reports"
        )
        self.human_gate_evidence_packet_scaffold_report_dir = (
            self.root / "human_gate_evidence_packet_scaffold_reports"
        )
        self.human_gate_evidence_packet_validation_report_dir = (
            self.root / "human_gate_evidence_packet_validation_reports"
        )
        self.human_gate_evidence_packet_record_draft_report_dir = (
            self.root / "human_gate_evidence_packet_record_draft_reports"
        )
        self.human_gate_preflight_bundle_report_dir = self.root / "human_gate_preflight_bundle_reports"
        self.human_gate_readiness_report_dir = self.root / "human_gate_readiness_reports"
        self.human_gate_validation_handoff_dir = self.root / "human_gate_validation_handoffs"
        self.human_gate_record_validation_dir = self.root / "human_gate_record_validations"

    def _contract_path(self, contract_version_id: str) -> Path:
        return self.contract_dir / f"{contract_version_id}.json"

    def _latest_path(self, contract_id: str) -> Path:
        return self.latest_dir / f"{contract_id}.json"

    def _setup_result_path(self, setup_result_id: str) -> Path:
        return self.setup_result_dir / f"{setup_result_id}.json"

    def _source_policy_path(self, source_policy_id: str) -> Path:
        return self.source_policy_dir / f"{source_policy_id}.json"

    def _upgrade_plan_path(self, upgrade_plan_id: str) -> Path:
        return self.upgrade_plan_dir / f"{upgrade_plan_id}.json"

    def _delivery_receipt_path(self, receipt_id: str) -> Path:
        return self.delivery_receipt_dir / f"{receipt_id}.json"

    def _projection_snapshot_path(self, snapshot_id: str) -> Path:
        return self.projection_snapshot_dir / f"{snapshot_id}.json"

    def _evidence_link_path(self, evidence_link_id: str) -> Path:
        return self.evidence_link_dir / f"{evidence_link_id}.json"

    def _ack_outbox_path(self, ack_outbox_id: str) -> Path:
        return self.ack_outbox_dir / f"{ack_outbox_id}.json"

    def _delivery_retry_path(self, retry_state_id: str) -> Path:
        return self.delivery_retry_dir / f"{retry_state_id}.json"

    def _quarantine_path(self, quarantine_id: str) -> Path:
        return self.quarantine_dir / f"{quarantine_id}.json"

    def _delivery_dedupe_path(self, idempotency_key: str) -> Path:
        return self.delivery_dedupe_dir / f"{idempotency_key}.json"

    def _content_version_path(self, content_version_id: str) -> Path:
        return self.content_version_dir / f"{content_version_id}.json"

    def _content_current_path(self, content_source_key: str) -> Path:
        return self.content_current_dir / f"{content_source_key}.json"

    def _projection_policy_decision_path(self, policy_decision_id: str) -> Path:
        return self.projection_policy_decision_dir / f"{policy_decision_id}.json"

    def _content_restriction_decision_path(self, decision_id: str) -> Path:
        return self.content_restriction_decision_dir / f"{decision_id}.json"

    def _raw_access_request_path(self, raw_access_request_id: str) -> Path:
        return self.raw_access_request_dir / f"{raw_access_request_id}.json"

    def _raw_access_grant_path(self, raw_access_grant_id: str) -> Path:
        return self.raw_access_grant_dir / f"{raw_access_grant_id}.json"

    def _untrusted_content_review_path(self, review_id: str) -> Path:
        return self.untrusted_content_review_dir / f"{review_id}.json"

    def _sync_signal_receipt_path(self, signal_receipt_id: str) -> Path:
        return self.sync_signal_receipt_dir / f"{signal_receipt_id}.json"

    def _sync_cursor_path(self, cursor_storage_id: str) -> Path:
        return self.sync_cursor_dir / f"{cursor_storage_id}.json"

    def _sync_reconciliation_path(self, reconciliation_id: str) -> Path:
        return self.sync_reconciliation_dir / f"{reconciliation_id}.json"

    def _provider_failure_state_path(self, failure_state_id: str) -> Path:
        return self.provider_failure_state_dir / f"{failure_state_id}.json"

    def _capture_permission_probe_path(self, probe_id: str) -> Path:
        return self.capture_permission_probe_dir / f"{probe_id}.json"

    def _watch_source_consent_path(self, consent_record_id: str) -> Path:
        return self.watch_source_consent_dir / f"{consent_record_id}.json"

    def _capture_guard_decision_path(self, guard_decision_id: str) -> Path:
        return self.capture_guard_decision_dir / f"{guard_decision_id}.json"

    def _activity_sample_batch_path(self, sample_batch_id: str) -> Path:
        return self.activity_sample_batch_dir / f"{sample_batch_id}.json"

    def _activity_sessionization_path(self, sessionization_id: str) -> Path:
        return self.activity_sessionization_dir / f"{sessionization_id}.json"

    def _activity_session_path(self, activity_session_id: str) -> Path:
        return self.activity_session_dir / f"{activity_session_id}.json"

    def _chrome_active_tab_permission_event_path(self, permission_event_id: str) -> Path:
        return self.chrome_active_tab_permission_event_dir / f"{permission_event_id}.json"

    def _chrome_active_tab_payload_path(self, active_tab_payload_id: str) -> Path:
        return self.chrome_active_tab_payload_dir / f"{active_tab_payload_id}.json"

    def _chrome_active_tab_policy_decision_path(self, policy_decision_id: str) -> Path:
        return self.chrome_active_tab_policy_decision_dir / f"{policy_decision_id}.json"

    def _chrome_active_tab_summary_path(self, capture_summary_id: str) -> Path:
        return self.chrome_active_tab_summary_dir / f"{capture_summary_id}.json"

    def _chrome_auto_capture_config_path(self, config_id: str) -> Path:
        return self.chrome_auto_capture_config_dir / f"{config_id}.json"

    def _chrome_auto_capture_trigger_path(self, trigger_id: str) -> Path:
        return self.chrome_auto_capture_trigger_dir / f"{trigger_id}.json"

    def _chrome_auto_capture_policy_decision_path(self, policy_decision_id: str) -> Path:
        return self.chrome_auto_capture_policy_decision_dir / f"{policy_decision_id}.json"

    def _chrome_auto_capture_summary_path(self, capture_summary_id: str) -> Path:
        return self.chrome_auto_capture_summary_dir / f"{capture_summary_id}.json"

    def _chrome_sensitive_page_policy_decision_path(self, policy_decision_id: str) -> Path:
        return self.chrome_sensitive_page_policy_decision_dir / f"{policy_decision_id}.json"

    def _chrome_sensitive_page_degraded_payload_path(self, degraded_payload_id: str) -> Path:
        return self.chrome_sensitive_page_degraded_payload_dir / f"{degraded_payload_id}.json"

    def _chrome_sensitive_page_history_item_path(self, history_item_id: str) -> Path:
        return self.chrome_sensitive_page_history_item_dir / f"{history_item_id}.json"

    def _capture_inbox_item_path(self, capture_inbox_item_id: str) -> Path:
        return self.capture_inbox_item_dir / f"{capture_inbox_item_id}.json"

    def _capture_lifecycle_source_state_path(self, source_state_id: str) -> Path:
        return self.capture_lifecycle_source_state_dir / f"{source_state_id}.json"

    def _capture_lifecycle_decision_path(self, lifecycle_decision_id: str) -> Path:
        return self.capture_lifecycle_decision_dir / f"{lifecycle_decision_id}.json"

    def _capture_lifecycle_export_path(self, lifecycle_export_id: str) -> Path:
        return self.capture_lifecycle_export_dir / f"{lifecycle_export_id}.json"

    def _capture_lifecycle_deletion_receipt_path(self, deletion_receipt_id: str) -> Path:
        return self.capture_lifecycle_deletion_receipt_dir / f"{deletion_receipt_id}.json"

    def _capture_result_review_path(self, capture_result_review_id: str) -> Path:
        return self.capture_result_review_dir / f"{capture_result_review_id}.json"

    def _watch_observation_path(self, watch_observation_id: str) -> Path:
        return self.watch_observation_dir / f"{watch_observation_id}.json"

    def _watch_inference_path(self, watch_inference_id: str) -> Path:
        return self.watch_inference_dir / f"{watch_inference_id}.json"

    def _watch_result_path(self, watch_result_id: str) -> Path:
        return self.watch_result_dir / f"{watch_result_id}.json"

    def _watch_result_correction_path(self, watch_result_correction_id: str) -> Path:
        return self.watch_result_correction_dir / f"{watch_result_correction_id}.json"

    def _watch_result_review_path(self, watch_result_review_id: str) -> Path:
        return self.watch_result_review_dir / f"{watch_result_review_id}.json"

    def _connector_action_preflight_path(self, preflight_id: str) -> Path:
        return self.connector_action_preflight_dir / f"{preflight_id}.json"

    def _connector_action_preflight_review_path(self, review_id: str) -> Path:
        return self.connector_action_preflight_review_dir / f"{review_id}.json"

    def _watch_rule_path(self, watch_rule_id: str) -> Path:
        return self.watch_rule_dir / f"{watch_rule_id}.json"

    def _watch_rule_version_path(self, watch_rule_version_id: str) -> Path:
        return self.watch_rule_version_dir / f"{watch_rule_version_id}.json"

    def _watch_rule_policy_decision_path(self, policy_decision_id: str) -> Path:
        return self.watch_rule_policy_decision_dir / f"{policy_decision_id}.json"

    def _watch_rule_evaluation_trace_path(self, evaluation_trace_id: str) -> Path:
        return self.watch_rule_evaluation_trace_dir / f"{evaluation_trace_id}.json"

    def _load_setup_result(self, setup_result_id: str) -> dict[str, Any] | None:
        path = self._setup_result_path(setup_result_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _load_source_policy(self, source_policy_id: str) -> dict[str, Any] | None:
        path = self._source_policy_path(source_policy_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _load_delivery_receipt(self, receipt_id: str) -> dict[str, Any] | None:
        path = self._delivery_receipt_path(receipt_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _load_ack_outbox(self, ack_outbox_id: str) -> dict[str, Any] | None:
        path = self._ack_outbox_path(ack_outbox_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _load_delivery_retry(self, retry_state_id: str) -> dict[str, Any] | None:
        path = self._delivery_retry_path(retry_state_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _load_quarantine(self, quarantine_id: str) -> dict[str, Any] | None:
        path = self._quarantine_path(quarantine_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _load_projection_snapshot(self, snapshot_id: str) -> dict[str, Any] | None:
        path = self._projection_snapshot_path(snapshot_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _load_evidence_link(self, evidence_link_id: str) -> dict[str, Any] | None:
        path = self._evidence_link_path(evidence_link_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _load_delivery_dedupe(self, idempotency_key: str) -> dict[str, Any] | None:
        path = self._delivery_dedupe_path(idempotency_key)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _load_content_version(self, content_version_id: str) -> dict[str, Any] | None:
        path = self._content_version_path(content_version_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _load_content_current(self, content_source_key: str) -> dict[str, Any] | None:
        path = self._content_current_path(content_source_key)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _load_projection_policy_decision(self, policy_decision_id: str) -> dict[str, Any] | None:
        path = self._projection_policy_decision_path(policy_decision_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _load_watch_rule(self, watch_rule_id: str) -> dict[str, Any] | None:
        path = self._watch_rule_path(watch_rule_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _load_watch_rule_version(self, watch_rule_version_id: str) -> dict[str, Any] | None:
        path = self._watch_rule_version_path(watch_rule_version_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _load_untrusted_content_review(self, review_id: str) -> dict[str, Any] | None:
        path = self._untrusted_content_review_path(review_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _load_raw_access_grant(self, raw_access_grant_id: str) -> dict[str, Any] | None:
        path = self._raw_access_grant_path(raw_access_grant_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _load_sync_cursor(self, cursor_storage_id: str) -> dict[str, Any] | None:
        path = self._sync_cursor_path(cursor_storage_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _load_provider_failure_state(self, failure_state_id: str) -> dict[str, Any] | None:
        path = self._provider_failure_state_path(failure_state_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _audit_correlation_path(self, report_id: str) -> Path:
        return self.root / "audit_correlations" / f"{report_id}.json"

    def _human_gate_package_path(self, package_id: str) -> Path:
        return self.human_gate_package_dir / f"{package_id}.json"

    def _human_gate_field_ref_contract_report_path(self, report_id: str) -> Path:
        return self.human_gate_field_ref_contract_report_dir / f"{report_id}.json"

    def _human_gate_evidence_packet_contract_report_path(self, report_id: str) -> Path:
        return self.human_gate_evidence_packet_contract_report_dir / f"{report_id}.json"

    def _human_gate_evidence_packet_file_contract_report_path(self, report_id: str) -> Path:
        return self.human_gate_evidence_packet_file_contract_report_dir / f"{report_id}.json"

    def _human_gate_evidence_packet_scaffold_report_path(self, report_id: str) -> Path:
        return self.human_gate_evidence_packet_scaffold_report_dir / f"{report_id}.json"

    def _human_gate_evidence_packet_validation_report_path(self, report_id: str) -> Path:
        return self.human_gate_evidence_packet_validation_report_dir / f"{report_id}.json"

    def _human_gate_evidence_packet_record_draft_report_path(self, report_id: str) -> Path:
        return self.human_gate_evidence_packet_record_draft_report_dir / f"{report_id}.json"

    def _human_gate_preflight_bundle_report_path(self, report_id: str) -> Path:
        return self.human_gate_preflight_bundle_report_dir / f"{report_id}.json"

    def _human_gate_readiness_report_path(self, report_id: str) -> Path:
        return self.human_gate_readiness_report_dir / f"{report_id}.json"

    def _human_gate_next_path(self, next_id: str) -> Path:
        return self.human_gate_readiness_report_dir / f"{next_id}.json"

    def _human_gate_validation_handoff_path(self, handoff_id: str) -> Path:
        return self.human_gate_validation_handoff_dir / f"{handoff_id}.json"

    def _human_gate_record_validation_path(self, validation_id: str) -> Path:
        return self.human_gate_record_validation_dir / f"{validation_id}.json"

    def _list_human_gate_record_validations(self) -> list[dict[str, Any]]:
        if not self.human_gate_record_validation_dir.exists():
            return []
        validations: list[dict[str, Any]] = []
        for path in sorted(self.human_gate_record_validation_dir.glob("*.json")):
            try:
                validation = json.loads(path.read_text())
            except json.JSONDecodeError:
                continue
            if isinstance(validation, dict):
                validations.append(validation)
        return validations

    def _human_gate_readiness_report_is_current(self, report: dict[str, Any]) -> bool:
        rows = report.get("scenario_results")
        if not isinstance(rows, list):
            return False
        expected_scenario_ids = set(CONNECTOR_HUMAN_GATE_QUEUE_BY_SCENARIO)
        row_ids = {str(row.get("scenario_id")) for row in rows if isinstance(row, dict)}
        if row_ids != expected_scenario_ids:
            return False
        required_row_fields = [
            "source_requirement_ids",
            "source_requirement_count",
            "source_requirement_status",
            "source_requirement_claim_boundary",
            "package_command",
            "record_template_output_command",
            "record_validation_command",
            "record_validation_output_command",
            "scenario_delivery_unit_plan",
            "scenario_delivery_unit_plan_summary",
        ]
        for row in rows:
            if not isinstance(row, dict):
                return False
            if any(field not in row for field in required_row_fields):
                return False
            if not isinstance(row.get("source_requirement_ids"), list):
                return False
            if not isinstance(row.get("scenario_delivery_unit_plan_summary"), dict):
                return False
            scenario_id = str(row.get("scenario_id"))
            baseline = row.get("local_baseline_review_inputs")
            baseline_preflight_bundle = (
                baseline.get("preflight_bundle")
                if isinstance(baseline, dict) and isinstance(baseline.get("preflight_bundle"), dict)
                else None
            )
            baseline_preflight_bundle_alias = row.get("local_baseline_preflight_bundle")
            if scenario_id == "CS-CH-H04":
                if not isinstance(baseline, dict):
                    return False
                if not isinstance(baseline_preflight_bundle, dict):
                    return False
                if not isinstance(baseline_preflight_bundle_alias, dict):
                    return False
                if baseline_preflight_bundle_alias != baseline_preflight_bundle:
                    return False
            elif baseline is not None:
                return False
            elif baseline_preflight_bundle_alias is not None:
                return False
        return True

    def _load_latest_human_gate_readiness_report(
        self,
        requested_scope: dict[str, str],
    ) -> dict[str, Any] | None:
        if not self.human_gate_readiness_report_dir.exists():
            return None
        reports: list[dict[str, Any]] = []
        for path in self.human_gate_readiness_report_dir.glob("cshuman_report_*.json"):
            try:
                report = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if report.get("schema_version") != CONNECTOR_HUMAN_GATE_READINESS_REPORT_SCHEMA:
                continue
            if report.get("scope") != requested_scope:
                continue
            if not isinstance(report.get("created_at"), str):
                continue
            if not self._human_gate_readiness_report_is_current(report):
                continue
            reports.append(report)
        if not reports:
            return None
        return max(reports, key=lambda report: str(report["created_at"]))

    def _get_or_create_human_gate_readiness_report(
        self,
        requested_scope: dict[str, str],
        repo_root: Path,
    ) -> dict[str, Any]:
        existing_report = self._load_latest_human_gate_readiness_report(requested_scope)
        if existing_report is not None:
            return {"human_gate_readiness_report": existing_report, "audit_event": None}
        return self.create_human_gate_readiness_report(requested_scope, repo_root)

    def _connector_audit_affected_object_refs(self, event: dict[str, Any]) -> list[str]:
        refs: list[str] = []
        subject = event.get("subject") if isinstance(event.get("subject"), dict) else {}
        subject_type = str(subject.get("type") or "audit_subject")
        subject_id = str(subject.get("id") or "")
        if subject_id:
            refs.append(f"{subject_type}:{subject_id}")

        def collect(value: Any, parent_key: str = "") -> None:
            if isinstance(value, dict):
                for key, item in value.items():
                    key_text = str(key)
                    if (
                        key_text.endswith("_id")
                        and not isinstance(item, (dict, list))
                        and item is not None
                        and item != ""
                    ):
                        refs.append(f"{key_text[:-3]}:{item}")
                    elif key_text.endswith("_ids") and isinstance(item, list):
                        for entry in item:
                            if not isinstance(entry, (dict, list)) and entry is not None and entry != "":
                                refs.append(f"{key_text[:-4]}:{entry}")
                    collect(item, key_text)
            elif isinstance(value, list):
                for item in value:
                    collect(item, parent_key)

        collect(event.get("details", {}))
        return list(dict.fromkeys(refs))

    def _connector_audit_detail_leaks(self, value: Any, path: str = "details") -> list[dict[str, str]]:
        leaks: list[dict[str, str]] = []
        secret_markers = [
            "connectorhub-private-secret",
            "-----BEGIN PRIVATE KEY-----",
            "ghp_",
            "xoxb-",
            "Bearer ",
            "Authorization:",
        ]

        def present(item: Any) -> bool:
            return not (item is None or item is False or item == 0 or item == "")

        if isinstance(value, dict):
            for key, item in value.items():
                key_text = str(key)
                key_lower = key_text.lower()
                item_path = f"{path}.{key_text}"
                if key_lower in CONNECTOR_AUDIT_FORBIDDEN_DETAIL_KEYS and present(item):
                    leaks.append({"path": item_path, "reason": "forbidden_raw_or_secret_key"})
                if key_lower in CONNECTOR_AUDIT_BOOLEAN_LEAK_KEYS and present(item):
                    leaks.append({"path": item_path, "reason": "truthy_boundary_leak_flag"})
                leaks.extend(self._connector_audit_detail_leaks(item, item_path))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                leaks.extend(self._connector_audit_detail_leaks(item, f"{path}[{index}]"))
        elif isinstance(value, str):
            for marker in secret_markers:
                if marker in value:
                    leaks.append({"path": path, "reason": "secret_marker"})
        return leaks

    def correlate_connector_audit(self, requested_scope: dict[str, str]) -> dict[str, Any]:
        all_events = self.store._all_audit_events()
        connector_events = [
            event
            for event in all_events
            if str(event.get("event_type", "")).startswith("connector.")
            and {
                "tenant_id": event.get("tenant_id"),
                "owner_id": event.get("owner_id"),
                "namespace_id": event.get("namespace_id"),
                "workspace_id": event.get("workspace_id"),
            }
            == requested_scope
        ]
        event_types = {str(event.get("event_type")) for event in connector_events}
        required_family_presence = {
            family: sorted(event_type for event_type in expected if event_type in event_types)
            for family, expected in CONNECTOR_AUDIT_REQUIRED_EVENT_FAMILIES.items()
        }
        missing_required_families = [
            family for family, present_event_types in required_family_presence.items() if not present_event_types
        ]

        correlations: list[dict[str, Any]] = []
        uncorrelated_event_ids: list[str] = []
        detail_leaks: list[dict[str, str]] = []
        scope_mismatches: list[str] = []
        for event in connector_events:
            audit_event_id = str(event.get("event_id") or "")
            subject = event.get("subject") if isinstance(event.get("subject"), dict) else {}
            connector_event_id = str(subject.get("id") or "")
            affected_object_refs = self._connector_audit_affected_object_refs(event)
            if not audit_event_id or not connector_event_id or not affected_object_refs:
                uncorrelated_event_ids.append(audit_event_id or "<missing-audit-event-id>")
            event_scope = {
                "tenant_id": event.get("tenant_id"),
                "owner_id": event.get("owner_id"),
                "namespace_id": event.get("namespace_id"),
                "workspace_id": event.get("workspace_id"),
            }
            if event_scope != requested_scope:
                scope_mismatches.append(audit_event_id or connector_event_id or "<missing-id>")
            for leak in self._connector_audit_detail_leaks(event.get("details", {})):
                detail_leaks.append({"audit_event_id": audit_event_id, **leak})
            correlation_id = f"conn_audit_corr_{json_hash({'audit_event_id': audit_event_id, 'connector_event_id': connector_event_id})[:16]}"
            correlations.append(
                {
                    "schema_version": "cs.connector_audit_correlation_item.v1",
                    "correlation_id": correlation_id,
                    "connector_event_id": connector_event_id,
                    "cornerstone_audit_event_id": audit_event_id,
                    "event_type": event.get("event_type"),
                    "affected_object_refs": affected_object_refs,
                    "event_hash": event.get("event_hash"),
                    "previous_hash": event.get("previous_hash"),
                    "raw_payload_copied": False,
                    "secret_copied": False,
                }
            )

        correlation_ids = [item["correlation_id"] for item in correlations]
        duplicate_correlation_ids = sorted(
            correlation_id
            for correlation_id in set(correlation_ids)
            if correlation_ids.count(correlation_id) > 1
        )
        audit_integrity = self.store.verify_audit()
        negative_evidence = {
            "missing_required_event_families": len(missing_required_families),
            "uncorrelated_connector_events": len(uncorrelated_event_ids),
            "duplicate_correlation_ids": len(duplicate_correlation_ids),
            "scope_mismatches": len(scope_mismatches),
            "raw_payload_or_secret_leaks": len(detail_leaks),
            "audit_integrity_errors": len(audit_integrity.get("errors", [])),
        }
        checks = {
            "connector_events_present": bool(connector_events),
            "required_event_families_present": not missing_required_families,
            "correlation_ids_unique": not duplicate_correlation_ids,
            "every_connector_event_has_audit_and_object_refs": not uncorrelated_event_ids,
            "scope_consistent": not scope_mismatches,
            "raw_payloads_and_secrets_absent": not detail_leaks,
            "audit_integrity_verified": audit_integrity.get("status") == "success",
        }
        report_base = {
            "schema_version": CONNECTOR_AUDIT_CORRELATION_SCHEMA,
            "scope": requested_scope,
            "status": "success" if all(checks.values()) else "failed",
            "required_event_families": CONNECTOR_AUDIT_REQUIRED_EVENT_FAMILIES,
            "required_family_presence": required_family_presence,
            "missing_required_families": missing_required_families,
            "connector_event_count": len(connector_events),
            "correlated_event_count": len(correlations),
            "correlations": correlations,
            "uncorrelated_event_ids": uncorrelated_event_ids,
            "duplicate_correlation_ids": duplicate_correlation_ids,
            "scope_mismatches": scope_mismatches,
            "detail_leaks": detail_leaks,
            "audit_integrity": audit_integrity,
            "negative_evidence": negative_evidence,
            "checks": checks,
            "created_at": utc_now(),
        }
        report_id = f"caudit_{json_hash(report_base)[:16]}"
        report = dict(report_base)
        report["connector_audit_correlation_report_id"] = report_id
        _write_json(self._audit_correlation_path(report_id), report)
        return report

    def create_human_gate_package(self, requested_scope: dict[str, str], scenario_id: str, repo_root: Path) -> dict[str, Any]:
        definition = CONNECTOR_HUMAN_GATE_PACKAGE_DEFINITIONS.get(scenario_id)
        if definition is None:
            return {
                "schema_version": CONNECTOR_HUMAN_GATE_PACKAGE_SCHEMA,
                "scenario_id": scenario_id,
                "status": "unsupported",
                "errors": [
                    {
                        "code": "CS_CONNECTOR_HUMAN_GATE_UNSUPPORTED",
                        "message": "Supported ConnectorHub human-gate packages are CS-CH-H01 through CS-CH-H07.",
                    }
                ],
            }
        non_mutation_evidence = {
            "approval_collected_by_ai": False,
            "live_provider_calls_executed_by_package": 0,
            "provider_mutations_executed_by_package": 0,
            "external_mutations_executed_by_package": 0,
            "secret_material_read_by_package": False,
        }
        non_mutation_evidence.update(deepcopy(definition.get("non_mutation_evidence", {})))
        template_path = connector_human_template_path(scenario_id)
        template_structure = connector_human_template_structure(repo_root / template_path, scenario_id)
        execution_queue_item = deepcopy(CONNECTOR_HUMAN_GATE_QUEUE_BY_SCENARIO[scenario_id])
        proposed_record_template = connector_human_gate_record_template(
            scenario_id,
            definition,
            execution_queue_item,
        )
        package_command = f"cornerstone connector human-gate package --scenario {scenario_id} --json"
        scenario_delivery_unit_plan = connector_human_gate_delivery_unit_plan(
            scenario_id,
            definition,
            execution_queue_item,
            package_command=package_command,
            record_template_output_command=proposed_record_template["record_template_output_command"],
            validation_command=proposed_record_template["validation_command"],
            validation_output_command=proposed_record_template["validation_output_command"],
        )
        scenario_delivery_unit_plan_summary = connector_human_gate_delivery_unit_plan_summary(
            scenario_delivery_unit_plan
        )
        remaining_human_evidence_summary = connector_human_gate_remaining_evidence_summary(
            {
                "scenario_id": scenario_id,
                "required_human_fields": proposed_record_template["required_fields"],
                "required_evidence": proposed_record_template["required_evidence"],
                "release_impact": definition["release_impact"],
                "stop_or_reject_when": execution_queue_item["stop_or_reject_when"],
                "record_template_output_command": proposed_record_template[
                    "record_template_output_command"
                ],
                "record_validation_output_command": connector_human_gate_record_validation_command(
                    scenario_id,
                    record_placeholder="<json>",
                    output_placeholder="<redacted-validation-envelope.json>",
                ),
            }
        )
        local_baseline_review_inputs = connector_human_gate_local_baseline_review_inputs(repo_root, scenario_id)
        source_requirement_ids = connector_human_gate_source_requirement_ids(scenario_id)
        package_base = {
            "schema_version": CONNECTOR_HUMAN_GATE_PACKAGE_SCHEMA,
            "scenario_id": scenario_id,
            "title": definition["title"],
            "status": "human_review_required",
            "approval_status": "pending",
            "scope": requested_scope,
            "execution_queue_item": execution_queue_item,
            "review_order": execution_queue_item["order"],
            "depends_on_human_gates": execution_queue_item["depends_on"],
            "source_requirement_ids": source_requirement_ids,
            "source_requirement_count": len(source_requirement_ids),
            "source_requirement_status": "human_external_pending",
            "source_requirement_claim_boundary": CONNECTOR_HUMAN_GATE_SOURCE_REQUIREMENT_CLAIM_BOUNDARY,
            "gate_category": execution_queue_item["gate_category"],
            "stop_or_reject_when": execution_queue_item["stop_or_reject_when"],
            "template_path": str(template_path),
            "template_present": template_structure["template_present"],
            "template_structure_ready": template_structure["structure_ready"],
            "template_structure": template_structure,
            "proof_boundary": deepcopy(definition["proof_boundary"]),
            "senior_review_perspectives": deepcopy(definition["senior_review_perspectives"]),
            "rehearsal_checklist": deepcopy(definition["rehearsal_checklist"]),
            "required_human_record": deepcopy(definition["required_human_record"]),
            "record_template_output_command": proposed_record_template["record_template_output_command"],
            "record_validation_command": proposed_record_template["validation_command"],
            "record_validation_output_command": proposed_record_template["validation_output_command"],
            "validation_output_command": proposed_record_template["validation_output_command"],
            "remaining_human_evidence_summary": remaining_human_evidence_summary,
            "proposed_record_template": proposed_record_template,
            "scenario_delivery_unit_plan": scenario_delivery_unit_plan,
            "scenario_delivery_unit_plan_summary": scenario_delivery_unit_plan_summary,
            "non_mutation_evidence": non_mutation_evidence,
            "release_impact": definition["release_impact"],
            "created_at": utc_now(),
        }
        package_base.update(connector_human_gate_completion_boundary())
        if local_baseline_review_inputs is not None:
            package_base["local_baseline_review_inputs"] = local_baseline_review_inputs
            local_baseline_preflight_bundle = local_baseline_review_inputs.get("preflight_bundle")
            if isinstance(local_baseline_preflight_bundle, dict):
                package_base["local_baseline_preflight_bundle"] = deepcopy(local_baseline_preflight_bundle)
            package_base["required_human_delta"] = list(
                local_baseline_review_inputs.get("required_human_delta") or []
            )
            package_base["required_human_delta_count"] = len(
                package_base["required_human_delta"]
            )
        package_id = f"cshuman_{json_hash(package_base)[:16]}"
        package = dict(package_base)
        package["package_id"] = package_id
        _write_json(self._human_gate_package_path(package_id), package)
        event = self.store.append_audit(
            "connector.human_gate_package.created",
            requested_scope,
            {"type": "connector_human_gate_package", "id": package_id},
            {
                "scenario_id": package["scenario_id"],
                "approval_status": package["approval_status"],
                "review_order": package["review_order"],
                "depends_on_human_gates": package["depends_on_human_gates"],
                "template_structure_ready": package["template_structure_ready"],
                "product_claim_allowed": False,
                "live_provider_calls_executed_by_package": 0,
                "provider_mutations_executed_by_package": 0,
            },
        )
        return {"human_gate_package": package, "audit_event": event}

    def create_human_gate_field_ref_contract_report(
        self,
        requested_scope: dict[str, str],
        scenario_id: str,
    ) -> dict[str, Any]:
        field_ref_contract = connector_human_gate_field_ref_contract(scenario_id)
        if field_ref_contract is None:
            unsupported = {
                "schema_version": CONNECTOR_HUMAN_GATE_FIELD_REF_CONTRACT_REPORT_SCHEMA,
                "scenario_id": scenario_id,
                "status": "unsupported",
                "scope": requested_scope,
                "errors": [
                    {
                        "code": "CS_CONNECTOR_HUMAN_GATE_FIELD_REF_CONTRACT_UNSUPPORTED",
                        "message": "Only CS-CH-H04 currently has a ConnectorHub human-gate field-ref contract.",
                    }
                ],
            }
            unsupported.update(connector_human_gate_completion_boundary())
            return {"human_gate_field_ref_contract_report": unsupported, "audit_event": None}
        required_field_ref_fields = [
            str(item["field"])
            for item in field_ref_contract["required_field_ref_items"]
        ]
        accepted_ref_prefixes_by_field = {
            str(item["field"]): list(item.get("accepted_ref_prefixes") or [])
            for item in field_ref_contract["required_field_ref_items"]
        }
        accepted_container_by_field = {
            str(item["field"]): item.get("accepted_container")
            for item in field_ref_contract["required_field_ref_items"]
        }
        non_mutation_evidence = {
            "approval_collected_by_field_ref_contract": False,
            "human_decisions_recorded_by_field_ref_contract": 0,
            "commands_executed_by_field_ref_contract": 0,
            "live_provider_calls_executed_by_field_ref_contract": 0,
            "provider_mutations_executed_by_field_ref_contract": 0,
            "external_mutations_executed_by_field_ref_contract": 0,
            "secret_material_read_by_field_ref_contract": False,
            "raw_field_values_recorded_by_field_ref_contract": False,
            "raw_field_values_persisted_by_field_ref_contract": False,
        }
        report_base = {
            "schema_version": CONNECTOR_HUMAN_GATE_FIELD_REF_CONTRACT_REPORT_SCHEMA,
            "scenario_id": scenario_id,
            "status": "operator_preparation_only",
            "scope": requested_scope,
            "final_verdict": "HUMAN_REQUIRED",
            "weakest_applicable_scenario_result": "HUMAN_REQUIRED",
            "field_ref_contract": deepcopy(field_ref_contract),
            "field_ref_contract_schema_version": field_ref_contract["schema_version"],
            "required_field_ref_item_count": len(required_field_ref_fields),
            "required_field_ref_fields": required_field_ref_fields,
            "accepted_ref_prefixes_by_field": accepted_ref_prefixes_by_field,
            "accepted_container_by_field": accepted_container_by_field,
            "raw_field_values_persisted_by_validator": False,
            "raw_field_values_recorded_by_report": False,
            "invalid_value_report_shape": field_ref_contract["invalid_value_report_shape"],
            "non_mutation_evidence": non_mutation_evidence,
            "negative_evidence": {
                "human_rows_marked_pass_by_field_ref_contract": 0,
                "approvals_collected_by_field_ref_contract": 0,
                "product_claims_allowed_by_field_ref_contract": 0,
                "pass_claims_allowed_by_field_ref_contract": 0,
                "commands_executed_by_field_ref_contract": 0,
                "live_provider_calls_executed_by_field_ref_contract": 0,
                "provider_mutations_executed_by_field_ref_contract": 0,
                "external_mutations_executed_by_field_ref_contract": 0,
                "secret_material_read_by_field_ref_contract": 0,
                "raw_field_values_recorded_by_field_ref_contract": 0,
                "raw_field_values_persisted_by_field_ref_contract": 0,
            },
            "product_feature_claims": "CONNECTOR_HUB_H04_FIELD_REF_CONTRACT_PREPARED_HUMAN_EVIDENCE_REQUIRED",
            "operator_rule": (
                "This report exposes accepted evidence-reference shapes for H04 production-like proof. "
                "It does not include, record, validate, or persist submitted field values."
            ),
            "created_at": utc_now(),
        }
        report_base.update(connector_human_gate_completion_boundary())
        report_id = f"cshuman_fieldref_{json_hash(report_base)[:16]}"
        report = dict(report_base)
        report["field_ref_contract_report_id"] = report_id
        _write_json(self._human_gate_field_ref_contract_report_path(report_id), report)
        event = self.store.append_audit(
            "connector.human_gate_field_ref_contract.reported",
            requested_scope,
            {"type": "connector_human_gate_field_ref_contract_report", "id": report_id},
            {
                "scenario_id": scenario_id,
                "required_field_ref_item_count": report["required_field_ref_item_count"],
                "final_verdict": report["final_verdict"],
                "product_claim_allowed": False,
                "raw_field_values_recorded_by_field_ref_contract": False,
                "raw_field_values_persisted_by_field_ref_contract": False,
                "live_provider_calls_executed_by_field_ref_contract": 0,
                "provider_mutations_executed_by_field_ref_contract": 0,
            },
        )
        return {"human_gate_field_ref_contract_report": report, "audit_event": event}

    def create_human_gate_evidence_packet_contract_report(
        self,
        requested_scope: dict[str, str],
        scenario_id: str,
    ) -> dict[str, Any]:
        if scenario_id not in CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_FILE_ITEMS_BY_SCENARIO:
            unsupported = {
                "schema_version": CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_CONTRACT_REPORT_SCHEMA,
                "scenario_id": scenario_id,
                "status": "unsupported",
                "scope": requested_scope,
                "errors": [
                    {
                        "code": "CS_CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_CONTRACT_UNSUPPORTED",
                        "message": (
                            "This ConnectorHub human gate does not yet have an evidence-packet "
                            "contract file mapping."
                        ),
                    }
                ],
            }
            unsupported.update(connector_human_gate_completion_boundary())
            return {"human_gate_evidence_packet_contract_report": unsupported, "audit_event": None}
        definition = CONNECTOR_HUMAN_GATE_PACKAGE_DEFINITIONS[scenario_id]
        execution_queue_item = deepcopy(CONNECTOR_HUMAN_GATE_QUEUE_BY_SCENARIO[scenario_id])
        proposed_record_template = connector_human_gate_record_template(
            scenario_id,
            definition,
            execution_queue_item,
        )
        required_manifest = deepcopy(
            proposed_record_template["required_evidence_packet_manifest"]
        )
        required_indexes = [
            int(item["required_evidence_index"])
            for item in required_manifest
        ]
        required_labels = [
            str(item["required_evidence"])
            for item in required_manifest
        ]
        allowed_redaction_statuses = list(
            proposed_record_template["allowed_redaction_statuses"]
        )
        evidence_packet_contract = {
            "schema_version": CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_CONTRACT_SCHEMA,
            "scenario_id": scenario_id,
            "status": "operator_preparation_only",
            "validation_scope": "evidence_packet_manifest_shape_only",
            "required_evidence_packet_manifest": required_manifest,
            "required_evidence_packet_manifest_count": len(required_manifest),
            "required_evidence_indexes": required_indexes,
            "required_evidence_labels": required_labels,
            "evidence_ref_required": True,
            "evidence_ref_uniqueness_required": True,
            "redaction_status_required": True,
            "allowed_redaction_statuses": allowed_redaction_statuses,
            "raw_evidence_ref_values_recorded_by_report": False,
            "evidence_packet_manifest_values_persisted_by_validator": False,
            "invalid_value_report_shape": "field_names_and_required_evidence_indexes_only",
        }
        non_mutation_evidence = {
            "approval_collected_by_evidence_packet_contract": False,
            "human_decisions_recorded_by_evidence_packet_contract": 0,
            "commands_executed_by_evidence_packet_contract": 0,
            "live_provider_calls_executed_by_evidence_packet_contract": 0,
            "provider_mutations_executed_by_evidence_packet_contract": 0,
            "external_mutations_executed_by_evidence_packet_contract": 0,
            "secret_material_read_by_evidence_packet_contract": False,
            "raw_evidence_ref_values_recorded_by_evidence_packet_contract": False,
            "evidence_packet_manifest_values_persisted_by_evidence_packet_contract": False,
        }
        report_base = {
            "schema_version": CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_CONTRACT_REPORT_SCHEMA,
            "scenario_id": scenario_id,
            "status": "operator_preparation_only",
            "scope": requested_scope,
            "final_verdict": "HUMAN_REQUIRED",
            "weakest_applicable_scenario_result": "HUMAN_REQUIRED",
            "evidence_packet_contract": evidence_packet_contract,
            "evidence_packet_contract_schema_version": evidence_packet_contract["schema_version"],
            "required_evidence_packet_manifest_count": len(required_manifest),
            "required_evidence_indexes": required_indexes,
            "required_evidence_labels": required_labels,
            "allowed_redaction_statuses": allowed_redaction_statuses,
            "evidence_ref_required": True,
            "evidence_ref_uniqueness_required": True,
            "redaction_status_required": True,
            "raw_evidence_ref_values_recorded_by_report": False,
            "evidence_packet_manifest_values_persisted_by_validator": False,
            "invalid_value_report_shape": evidence_packet_contract["invalid_value_report_shape"],
            "non_mutation_evidence": non_mutation_evidence,
            "negative_evidence": {
                "human_rows_marked_pass_by_evidence_packet_contract": 0,
                "approvals_collected_by_evidence_packet_contract": 0,
                "product_claims_allowed_by_evidence_packet_contract": 0,
                "pass_claims_allowed_by_evidence_packet_contract": 0,
                "commands_executed_by_evidence_packet_contract": 0,
                "live_provider_calls_executed_by_evidence_packet_contract": 0,
                "provider_mutations_executed_by_evidence_packet_contract": 0,
                "external_mutations_executed_by_evidence_packet_contract": 0,
                "secret_material_read_by_evidence_packet_contract": 0,
                "raw_evidence_ref_values_recorded_by_evidence_packet_contract": 0,
                "evidence_packet_manifest_values_persisted_by_evidence_packet_contract": 0,
            },
            "product_feature_claims": connector_human_gate_evidence_packet_product_feature_claim(
                scenario_id,
                "CONTRACT",
            ),
            "operator_rule": (
                f"This report exposes required {scenario_id} evidence-packet manifest rows and redaction "
                "statuses only. It does not include, record, validate, or persist submitted "
                "evidence refs or human decisions."
            ),
            "created_at": utc_now(),
        }
        report_base.update(connector_human_gate_completion_boundary())
        report_id = f"cshuman_evidencepacket_{json_hash(report_base)[:16]}"
        report = dict(report_base)
        report["evidence_packet_contract_report_id"] = report_id
        _write_json(self._human_gate_evidence_packet_contract_report_path(report_id), report)
        event = self.store.append_audit(
            "connector.human_gate_evidence_packet_contract.reported",
            requested_scope,
            {"type": "connector_human_gate_evidence_packet_contract_report", "id": report_id},
            {
                "scenario_id": scenario_id,
                "required_evidence_packet_manifest_count": report[
                    "required_evidence_packet_manifest_count"
                ],
                "final_verdict": report["final_verdict"],
                "product_claim_allowed": False,
                "raw_evidence_ref_values_recorded_by_evidence_packet_contract": False,
                "evidence_packet_manifest_values_persisted_by_evidence_packet_contract": False,
                "live_provider_calls_executed_by_evidence_packet_contract": 0,
                "provider_mutations_executed_by_evidence_packet_contract": 0,
            },
        )
        return {"human_gate_evidence_packet_contract_report": report, "audit_event": event}

    def create_human_gate_evidence_packet_file_contract_report(
        self,
        requested_scope: dict[str, str],
        scenario_id: str,
    ) -> dict[str, Any]:
        evidence_packet_file_contract = connector_human_gate_evidence_packet_file_contract(scenario_id)
        if evidence_packet_file_contract is None:
            unsupported = {
                "schema_version": CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_FILE_CONTRACT_REPORT_SCHEMA,
                "scenario_id": scenario_id,
                "status": "unsupported",
                "scope": requested_scope,
                "errors": [
                    {
                        "code": "CS_CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_FILE_CONTRACT_UNSUPPORTED",
                        "message": (
                            "This ConnectorHub human gate does not yet have an evidence-packet "
                            "file contract."
                        ),
                    }
                ],
            }
            unsupported.update(connector_human_gate_completion_boundary())
            return {"human_gate_evidence_packet_file_contract_report": unsupported, "audit_event": None}
        required_packet_files = deepcopy(evidence_packet_file_contract["required_packet_files"])
        required_packet_file_names = [
            str(item["packet_file"])
            for item in required_packet_files
        ]
        packet_file_scaffold_plan = deepcopy(
            evidence_packet_file_contract["packet_file_scaffold_plan"]
        )
        packet_file_scaffold_commands = [
            str(item["command"])
            for item in packet_file_scaffold_plan
        ]
        non_mutation_evidence = {
            "approval_collected_by_evidence_packet_file_contract": False,
            "human_decisions_recorded_by_evidence_packet_file_contract": 0,
            "commands_executed_by_evidence_packet_file_contract": 0,
            "live_provider_calls_executed_by_evidence_packet_file_contract": 0,
            "provider_mutations_executed_by_evidence_packet_file_contract": 0,
            "external_mutations_executed_by_evidence_packet_file_contract": 0,
            "secret_material_read_by_evidence_packet_file_contract": False,
            "raw_packet_file_contents_recorded_by_evidence_packet_file_contract": False,
            "packet_file_contents_persisted_by_evidence_packet_file_contract": False,
        }
        report_base = {
            "schema_version": CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_FILE_CONTRACT_REPORT_SCHEMA,
            "scenario_id": scenario_id,
            "status": "operator_preparation_only",
            "scope": requested_scope,
            "final_verdict": "HUMAN_REQUIRED",
            "weakest_applicable_scenario_result": "HUMAN_REQUIRED",
            "evidence_packet_file_contract": evidence_packet_file_contract,
            "evidence_packet_file_contract_schema_version": (
                evidence_packet_file_contract["schema_version"]
            ),
            "required_packet_file_count": len(required_packet_files),
            "required_packet_file_names": required_packet_file_names,
            "required_packet_files": required_packet_files,
            "raw_packet_file_contents_recorded_by_report": False,
            "packet_file_contents_persisted_by_report": False,
            "packet_file_contents_persisted_by_validator": False,
            "review_input_only": True,
            "acceptance_sufficient": False,
            "product_claim_allowed": False,
            "pass_claim_allowed": False,
            "invalid_value_report_shape": evidence_packet_file_contract[
                "invalid_value_report_shape"
            ],
            "packet_file_scaffold_plan_available": True,
            "packet_file_scaffold_directory": (
                evidence_packet_file_contract["packet_file_scaffold_directory"]
            ),
            "packet_file_scaffold_plan": packet_file_scaffold_plan,
            "packet_file_scaffold_commands": packet_file_scaffold_commands,
            "packet_file_scaffold_command_count": len(packet_file_scaffold_plan),
            "packet_file_scaffold_plan_executed_by_report": False,
            "packet_file_scaffold_plan_review_input_only": True,
            "packet_file_scaffold_plan_acceptance_sufficient": False,
            "non_mutation_evidence": non_mutation_evidence,
            "negative_evidence": {
                "human_rows_marked_pass_by_evidence_packet_file_contract": 0,
                "approvals_collected_by_evidence_packet_file_contract": 0,
                "product_claims_allowed_by_evidence_packet_file_contract": 0,
                "pass_claims_allowed_by_evidence_packet_file_contract": 0,
                "commands_executed_by_evidence_packet_file_contract": 0,
                "live_provider_calls_executed_by_evidence_packet_file_contract": 0,
                "provider_mutations_executed_by_evidence_packet_file_contract": 0,
                "external_mutations_executed_by_evidence_packet_file_contract": 0,
                "secret_material_read_by_evidence_packet_file_contract": 0,
                "raw_packet_file_contents_recorded_by_evidence_packet_file_contract": 0,
                "packet_file_contents_persisted_by_evidence_packet_file_contract": 0,
                "packet_file_scaffold_commands_executed_by_evidence_packet_file_contract": 0,
            },
            "product_feature_claims": (
                connector_human_gate_evidence_packet_product_feature_claim(
                    scenario_id,
                    "FILE_CONTRACT",
                )
            ),
            "operator_rule": (
                f"This report exposes the required {scenario_id} acceptance evidence packet filenames "
                "and required content categories plus a non-executed scaffold command plan "
                "only. It does not run scaffold commands, read, record, validate, or persist "
                "packet file contents or human decisions."
            ),
            "created_at": utc_now(),
        }
        report_base.update(connector_human_gate_completion_boundary())
        report_id = f"cshuman_evidencepacketfile_{json_hash(report_base)[:16]}"
        report = dict(report_base)
        report["evidence_packet_file_contract_report_id"] = report_id
        _write_json(self._human_gate_evidence_packet_file_contract_report_path(report_id), report)
        event = self.store.append_audit(
            "connector.human_gate_evidence_packet_file_contract.reported",
            requested_scope,
            {"type": "connector_human_gate_evidence_packet_file_contract_report", "id": report_id},
            {
                "scenario_id": scenario_id,
                "required_packet_file_count": report["required_packet_file_count"],
                "final_verdict": report["final_verdict"],
                "product_claim_allowed": False,
                "raw_packet_file_contents_recorded_by_evidence_packet_file_contract": False,
                "packet_file_contents_persisted_by_evidence_packet_file_contract": False,
                "packet_file_scaffold_command_count": report[
                    "packet_file_scaffold_command_count"
                ],
                "packet_file_scaffold_plan_executed_by_report": False,
                "live_provider_calls_executed_by_evidence_packet_file_contract": 0,
                "provider_mutations_executed_by_evidence_packet_file_contract": 0,
            },
        )
        return {"human_gate_evidence_packet_file_contract_report": report, "audit_event": event}

    def create_human_gate_evidence_packet_scaffold_report(
        self,
        requested_scope: dict[str, str],
        scenario_id: str,
        packet_dir: Path,
        write_files: bool,
    ) -> dict[str, Any]:
        evidence_packet_scaffold = connector_human_gate_evidence_packet_scaffold(scenario_id)
        if evidence_packet_scaffold is None:
            unsupported = {
                "schema_version": CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_SCAFFOLD_REPORT_SCHEMA,
                "scenario_id": scenario_id,
                "status": "unsupported",
                "scope": requested_scope,
                "errors": [
                    {
                        "code": "CS_CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_SCAFFOLD_UNSUPPORTED",
                        "message": (
                            "This ConnectorHub human gate does not yet have an evidence-packet "
                            "scaffold."
                        ),
                    }
                ],
            }
            unsupported.update(connector_human_gate_completion_boundary())
            return {"human_gate_evidence_packet_scaffold_report": unsupported, "audit_event": None}

        packet_directory = str(packet_dir)
        scaffold_templates = deepcopy(evidence_packet_scaffold["scaffold_templates"])
        target_files = [
            packet_dir / str(item["packet_file"])
            for item in scaffold_templates
        ]
        existing_files = [
            path.as_posix()
            for path in target_files
            if path.exists()
        ]
        if write_files and existing_files:
            blocked = {
                "schema_version": CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_SCAFFOLD_REPORT_SCHEMA,
                "scenario_id": scenario_id,
                "status": "write_blocked_existing_files",
                "scope": requested_scope,
                "final_verdict": "HUMAN_REQUIRED",
                "weakest_applicable_scenario_result": "HUMAN_REQUIRED",
                "packet_directory": packet_directory,
                "write_requested": True,
                "write_executed": False,
                "existing_packet_files": existing_files,
                "errors": [
                    {
                        "code": "CS_CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_SCAFFOLD_EXISTS",
                        "message": f"Refusing to overwrite existing {scenario_id} packet template files.",
                    }
                ],
            }
            blocked.update(connector_human_gate_completion_boundary())
            return {"human_gate_evidence_packet_scaffold_report": blocked, "audit_event": None}

        written_packet_files: list[dict[str, Any]] = []
        if write_files:
            packet_dir.mkdir(parents=True, exist_ok=True)
            for item in scaffold_templates:
                packet_file = str(item["packet_file"])
                required_contents = str(item["required_contents"])
                content = connector_human_gate_evidence_packet_scaffold_template_content(
                    packet_file,
                    required_contents,
                    scenario_id,
                )
                target_path = packet_dir / packet_file
                target_path.write_text(content)
                written_packet_files.append(
                    {
                        "packet_file": packet_file,
                        "path": target_path.as_posix(),
                        "template_content_sha256": hashlib.sha256(
                            content.encode("utf-8")
                        ).hexdigest(),
                        "template_content_line_count": len(content.splitlines()),
                        "template_only": True,
                        "human_evidence_recorded_by_template": False,
                    }
                )

        non_mutation_evidence = {
            "approval_collected_by_evidence_packet_scaffold": False,
            "human_decisions_recorded_by_evidence_packet_scaffold": 0,
            "commands_executed_by_evidence_packet_scaffold": 0,
            "live_provider_calls_executed_by_evidence_packet_scaffold": 0,
            "provider_mutations_executed_by_evidence_packet_scaffold": 0,
            "external_mutations_executed_by_evidence_packet_scaffold": 0,
            "secret_material_read_by_evidence_packet_scaffold": False,
            "packet_file_contents_read_by_evidence_packet_scaffold": False,
            "human_evidence_recorded_by_evidence_packet_scaffold": False,
            "template_contents_included_in_report": False,
            "local_template_files_written_by_evidence_packet_scaffold": (
                len(written_packet_files)
            ),
        }
        report_base = {
            "schema_version": CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_SCAFFOLD_REPORT_SCHEMA,
            "scenario_id": scenario_id,
            "status": "operator_preparation_only",
            "scope": requested_scope,
            "final_verdict": "HUMAN_REQUIRED",
            "weakest_applicable_scenario_result": "HUMAN_REQUIRED",
            "evidence_packet_scaffold": evidence_packet_scaffold,
            "evidence_packet_scaffold_schema_version": (
                evidence_packet_scaffold["schema_version"]
            ),
            "packet_directory": packet_directory,
            "write_requested": write_files,
            "write_executed": write_files,
            "scaffold_template_count": len(scaffold_templates),
            "scaffold_templates": scaffold_templates,
            "written_packet_files": written_packet_files,
            "written_packet_file_count": len(written_packet_files),
            "template_contents_included_in_report": False,
            "packet_file_contents_read_by_scaffold": False,
            "human_evidence_recorded_by_scaffold": False,
            "review_input_only": True,
            "acceptance_sufficient": False,
            "product_claim_allowed": False,
            "pass_claim_allowed": False,
            "non_mutation_evidence": non_mutation_evidence,
            "negative_evidence": {
                "human_rows_marked_pass_by_evidence_packet_scaffold": 0,
                "approvals_collected_by_evidence_packet_scaffold": 0,
                "product_claims_allowed_by_evidence_packet_scaffold": 0,
                "pass_claims_allowed_by_evidence_packet_scaffold": 0,
                "commands_executed_by_evidence_packet_scaffold": 0,
                "live_provider_calls_executed_by_evidence_packet_scaffold": 0,
                "provider_mutations_executed_by_evidence_packet_scaffold": 0,
                "external_mutations_executed_by_evidence_packet_scaffold": 0,
                "secret_material_read_by_evidence_packet_scaffold": 0,
                "packet_file_contents_read_by_evidence_packet_scaffold": 0,
                "human_evidence_recorded_by_evidence_packet_scaffold": 0,
                "template_contents_included_in_report": 0,
            },
            "product_feature_claims": (
                connector_human_gate_evidence_packet_product_feature_claim(
                    scenario_id,
                    "SCAFFOLD",
                )
            ),
            "operator_rule": (
                f"This report prepares blank {scenario_id} acceptance-packet templates only. "
                "It does not read packet evidence, collect approval, call live providers, "
                f"execute external mutations, or mark {scenario_id} accepted."
            ),
            "created_at": utc_now(),
        }
        report_base.update(connector_human_gate_completion_boundary())
        report_id = f"cshuman_evidencepacketscaffold_{json_hash(report_base)[:16]}"
        report = dict(report_base)
        report["evidence_packet_scaffold_report_id"] = report_id
        _write_json(self._human_gate_evidence_packet_scaffold_report_path(report_id), report)
        event = self.store.append_audit(
            "connector.human_gate_evidence_packet_scaffold.reported",
            requested_scope,
            {"type": "connector_human_gate_evidence_packet_scaffold_report", "id": report_id},
            {
                "scenario_id": scenario_id,
                "scaffold_template_count": report["scaffold_template_count"],
                "write_requested": write_files,
                "write_executed": write_files,
                "written_packet_file_count": report["written_packet_file_count"],
                "final_verdict": report["final_verdict"],
                "product_claim_allowed": False,
                "pass_claim_allowed": False,
                "live_provider_calls_executed_by_evidence_packet_scaffold": 0,
                "provider_mutations_executed_by_evidence_packet_scaffold": 0,
                "external_mutations_executed_by_evidence_packet_scaffold": 0,
            },
        )
        return {"human_gate_evidence_packet_scaffold_report": report, "audit_event": event}

    def create_human_gate_evidence_packet_validation_report(
        self,
        requested_scope: dict[str, str],
        scenario_id: str,
        packet_dir: Path,
    ) -> dict[str, Any]:
        evidence_packet_file_contract = connector_human_gate_evidence_packet_file_contract(scenario_id)
        evidence_packet_scaffold = connector_human_gate_evidence_packet_scaffold(scenario_id)
        if evidence_packet_file_contract is None or evidence_packet_scaffold is None:
            unsupported = {
                "schema_version": CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_VALIDATION_REPORT_SCHEMA,
                "scenario_id": scenario_id,
                "status": "unsupported",
                "scope": requested_scope,
                "errors": [
                    {
                        "code": "CS_CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_VALIDATION_UNSUPPORTED",
                        "message": (
                            "This ConnectorHub human gate does not yet have an evidence-packet "
                            "validation report."
                        ),
                    }
                ],
            }
            unsupported.update(connector_human_gate_completion_boundary())
            return {"human_gate_evidence_packet_validation_report": unsupported, "audit_event": None}

        required_packet_files = deepcopy(evidence_packet_file_contract["required_packet_files"])
        expected_templates = {
            str(item["packet_file"]): item
            for item in evidence_packet_scaffold["scaffold_templates"]
        }
        packet_directory = str(packet_dir)
        packet_directory_exists = packet_dir.exists() and packet_dir.is_dir()
        observed_packet_files: list[dict[str, Any]] = []
        missing_packet_file_names: list[str] = []
        empty_packet_file_names: list[str] = []
        template_only_packet_file_names: list[str] = []
        hashed_packet_file_names: list[str] = []
        structural_errors: list[dict[str, Any]] = []

        if not packet_directory_exists:
            structural_errors.append(
                    {
                        "code": "CS_CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_DIR_NOT_FOUND",
                        "message": f"{scenario_id} evidence packet directory was not found.",
                        "packet_directory": packet_directory,
                    }
                )

        for item in required_packet_files:
            packet_file = str(item["packet_file"])
            target_path = packet_dir / packet_file
            entry: dict[str, Any] = {
                "packet_file": packet_file,
                "required": True,
                "required_contents": item["required_contents"],
                "present": False,
                "size_bytes": None,
                "sha256": None,
                "line_count": None,
                "matches_blank_template": False,
                "raw_contents_included_in_report": False,
                "raw_contents_persisted_by_validator": False,
                "acceptance_sufficient": False,
                "product_claim_allowed": False,
                "pass_claim_allowed": False,
            }
            if not packet_directory_exists or not target_path.exists():
                missing_packet_file_names.append(packet_file)
                structural_errors.append(
                    {
                        "code": "CS_CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_FILE_MISSING",
                        "packet_file": packet_file,
                    }
                )
                observed_packet_files.append(entry)
                continue
            if not target_path.is_file():
                missing_packet_file_names.append(packet_file)
                structural_errors.append(
                    {
                        "code": "CS_CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_FILE_NOT_REGULAR",
                        "packet_file": packet_file,
                    }
                )
                observed_packet_files.append(entry)
                continue
            metadata = connector_human_gate_packet_file_metadata(target_path)
            template_hash = expected_templates[packet_file]["template_content_sha256"]
            matches_blank_template = metadata["sha256"] == template_hash
            entry.update(
                {
                    "present": True,
                    "size_bytes": metadata["size_bytes"],
                    "sha256": metadata["sha256"],
                    "line_count": metadata["line_count"],
                    "matches_blank_template": matches_blank_template,
                }
            )
            hashed_packet_file_names.append(packet_file)
            if metadata["size_bytes"] <= 0:
                empty_packet_file_names.append(packet_file)
                structural_errors.append(
                    {
                        "code": "CS_CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_FILE_EMPTY",
                        "packet_file": packet_file,
                    }
                )
            if matches_blank_template:
                template_only_packet_file_names.append(packet_file)
                structural_errors.append(
                    {
                        "code": "CS_CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_FILE_TEMPLATE_ONLY",
                        "packet_file": packet_file,
                    }
                )
            observed_packet_files.append(entry)

        structurally_complete = (
            packet_directory_exists
            and not missing_packet_file_names
            and not empty_packet_file_names
            and not template_only_packet_file_names
        )
        report_status = (
            "packet_structurally_complete"
            if structurally_complete
            else "packet_structural_issues"
        )
        if not packet_directory_exists:
            report_status = "packet_not_submitted"
        non_mutation_evidence = {
            "approval_collected_by_evidence_packet_validation": False,
            "human_decisions_recorded_by_evidence_packet_validation": 0,
            "commands_executed_by_evidence_packet_validation": 0,
            "live_provider_calls_executed_by_evidence_packet_validation": 0,
            "provider_mutations_executed_by_evidence_packet_validation": 0,
            "external_mutations_executed_by_evidence_packet_validation": 0,
            "raw_packet_file_contents_recorded_by_evidence_packet_validation": False,
            "packet_file_contents_persisted_by_evidence_packet_validation": False,
            "packet_file_hashes_recorded_by_evidence_packet_validation": (
                len(hashed_packet_file_names)
            ),
            "human_acceptance_collected_by_evidence_packet_validation": False,
        }
        report_base = {
            "schema_version": CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_VALIDATION_REPORT_SCHEMA,
            "scenario_id": scenario_id,
            "status": report_status,
            "scope": requested_scope,
            "final_verdict": "HUMAN_REQUIRED",
            "weakest_applicable_scenario_result": "HUMAN_REQUIRED",
            "packet_directory": packet_directory,
            "packet_directory_exists": packet_directory_exists,
            "evidence_packet_file_contract_schema_version": (
                evidence_packet_file_contract["schema_version"]
            ),
            "evidence_packet_scaffold_schema_version": (
                evidence_packet_scaffold["schema_version"]
            ),
            "required_packet_file_count": len(required_packet_files),
            "observed_packet_file_count": len(
                [entry for entry in observed_packet_files if entry["present"]]
            ),
            "hashed_packet_file_count": len(hashed_packet_file_names),
            "hashed_packet_file_names": hashed_packet_file_names,
            "missing_packet_file_count": len(missing_packet_file_names),
            "missing_packet_file_names": missing_packet_file_names,
            "empty_packet_file_count": len(empty_packet_file_names),
            "empty_packet_file_names": empty_packet_file_names,
            "template_only_packet_file_count": len(template_only_packet_file_names),
            "template_only_packet_file_names": template_only_packet_file_names,
            "packet_files": observed_packet_files,
            "raw_packet_file_contents_included_in_report": False,
            "raw_packet_file_contents_recorded_by_validator": False,
            "packet_file_contents_persisted_by_validator": False,
            "packet_file_hashes_recorded_by_validator": True,
            "packet_structurally_complete": structurally_complete,
            "review_input_only": True,
            "acceptance_sufficient": False,
            "dependency_unlock_allowed_by_packet_validator": False,
            "product_claim_allowed": False,
            "pass_claim_allowed": False,
            "structural_errors": structural_errors,
            "non_mutation_evidence": non_mutation_evidence,
            "negative_evidence": {
                "human_rows_marked_pass_by_evidence_packet_validation": 0,
                "approvals_collected_by_evidence_packet_validation": 0,
                "product_claims_allowed_by_evidence_packet_validation": 0,
                "pass_claims_allowed_by_evidence_packet_validation": 0,
                "dependency_unlocks_allowed_by_evidence_packet_validation": 0,
                "live_provider_calls_executed_by_evidence_packet_validation": 0,
                "provider_mutations_executed_by_evidence_packet_validation": 0,
                "external_mutations_executed_by_evidence_packet_validation": 0,
                "raw_packet_file_contents_recorded_by_evidence_packet_validation": 0,
                "packet_file_contents_persisted_by_evidence_packet_validation": 0,
            },
            "product_feature_claims": (
                connector_human_gate_evidence_packet_product_feature_claim(
                    scenario_id,
                    "VALIDATION",
                )
            ),
            "operator_rule": (
                f"This report validates {scenario_id} acceptance-packet file presence, non-empty status, "
                "and blank-template replacement by recording metadata and hashes only. It does "
                "not record packet contents, collect approval, unlock dependencies, or mark "
                f"{scenario_id} accepted."
            ),
            "created_at": utc_now(),
        }
        report_base.update(connector_human_gate_completion_boundary())
        report_id = f"cshuman_evidencepacketvalidation_{json_hash(report_base)[:16]}"
        report = dict(report_base)
        report["evidence_packet_validation_report_id"] = report_id
        _write_json(self._human_gate_evidence_packet_validation_report_path(report_id), report)
        event = self.store.append_audit(
            "connector.human_gate_evidence_packet_validation.reported",
            requested_scope,
            {"type": "connector_human_gate_evidence_packet_validation_report", "id": report_id},
            {
                "scenario_id": scenario_id,
                "status": report["status"],
                "packet_structurally_complete": structurally_complete,
                "required_packet_file_count": report["required_packet_file_count"],
                "observed_packet_file_count": report["observed_packet_file_count"],
                "missing_packet_file_count": report["missing_packet_file_count"],
                "template_only_packet_file_count": report["template_only_packet_file_count"],
                "final_verdict": report["final_verdict"],
                "product_claim_allowed": False,
                "pass_claim_allowed": False,
                "dependency_unlock_allowed_by_packet_validator": False,
                "live_provider_calls_executed_by_evidence_packet_validation": 0,
                "provider_mutations_executed_by_evidence_packet_validation": 0,
                "external_mutations_executed_by_evidence_packet_validation": 0,
            },
        )
        return {"human_gate_evidence_packet_validation_report": report, "audit_event": event}

    def create_human_gate_evidence_packet_record_draft_report(
        self,
        requested_scope: dict[str, str],
        scenario_id: str,
        packet_dir: Path,
    ) -> dict[str, Any]:
        validation_result = self.create_human_gate_evidence_packet_validation_report(
            requested_scope,
            scenario_id,
            packet_dir,
        )
        packet_validation_report = validation_result["human_gate_evidence_packet_validation_report"]
        if packet_validation_report.get("status") == "unsupported":
            unsupported = {
                "schema_version": CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_RECORD_DRAFT_REPORT_SCHEMA,
                "scenario_id": scenario_id,
                "status": "unsupported",
                "scope": requested_scope,
                "packet_validation_report": packet_validation_report,
                "errors": [
                    {
                        "code": "CS_CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_RECORD_DRAFT_UNSUPPORTED",
                        "message": (
                            "This ConnectorHub human gate does not yet have an evidence-packet "
                            "record draft report."
                        ),
                    }
                ],
            }
            unsupported.update(connector_human_gate_completion_boundary())
            return {
                "human_gate_evidence_packet_record_draft_report": unsupported,
                "packet_validation_audit_event": validation_result.get("audit_event"),
                "audit_event": None,
            }

        draft_record = connector_human_gate_record_draft_from_packet(packet_validation_report)
        draft_available = isinstance(draft_record, dict)
        report_status = (
            "draft_record_requires_human_completion"
            if draft_available
            else "packet_not_ready_for_record_draft"
        )
        non_mutation_evidence = {
            "commands_executed_by_evidence_packet_record_draft": 0,
            "live_provider_calls_executed_by_evidence_packet_record_draft": 0,
            "provider_mutations_executed_by_evidence_packet_record_draft": 0,
            "external_mutations_executed_by_evidence_packet_record_draft": 0,
            "raw_packet_file_contents_recorded_by_evidence_packet_record_draft": False,
            "packet_file_contents_persisted_by_evidence_packet_record_draft": False,
            "human_acceptance_collected_by_evidence_packet_record_draft": False,
            "human_decision_recorded_by_evidence_packet_record_draft": False,
        }
        report_base = {
            "schema_version": CONNECTOR_HUMAN_GATE_EVIDENCE_PACKET_RECORD_DRAFT_REPORT_SCHEMA,
            "scenario_id": scenario_id,
            "status": report_status,
            "scope": requested_scope,
            "final_verdict": "HUMAN_REQUIRED",
            "weakest_applicable_scenario_result": "HUMAN_REQUIRED",
            "packet_validation_report_id": packet_validation_report.get(
                "evidence_packet_validation_report_id"
            ),
            "packet_validation_status": packet_validation_report.get("status"),
            "packet_structurally_complete": packet_validation_report.get(
                "packet_structurally_complete"
            )
            is True,
            "required_packet_file_count": packet_validation_report.get(
                "required_packet_file_count"
            ),
            "hashed_packet_file_count": packet_validation_report.get("hashed_packet_file_count"),
            "draft_record_available": draft_available,
            "draft_record": draft_record,
            "draft_record_output_allowed": draft_available,
            "draft_record_output_written_by_runtime": False,
            "draft_record_intentionally_incomplete_fields": [
                "decision",
                "reviewer",
                "review_timestamp",
                "senior_review_perspective_findings",
            ],
            "draft_record_expected_validation_status_before_human_completion": (
                "record_structurally_invalid"
            ),
            "draft_record_validation_output_command": connector_human_gate_record_validation_command(
                scenario_id,
                record_placeholder="<reviewer-record-draft.json>",
                output_placeholder="<redacted-validation-envelope.json>",
            ),
            "draft_record_human_completion_required": True,
            "draft_record_human_decision_required": True,
            "raw_packet_file_contents_included_in_report": False,
            "raw_packet_file_contents_recorded_by_draft": False,
            "packet_file_contents_persisted_by_draft": False,
            "packet_file_hashes_used_by_draft": draft_available,
            "packet_directory_path_recorded_by_draft": False,
            "review_input_only": True,
            "acceptance_sufficient": False,
            "dependency_unlock_allowed_by_record_draft": False,
            "product_claim_allowed": False,
            "pass_claim_allowed": False,
            "non_mutation_evidence": non_mutation_evidence,
            "negative_evidence": {
                "human_rows_marked_pass_by_evidence_packet_record_draft": 0,
                "approvals_collected_by_evidence_packet_record_draft": 0,
                "human_decisions_recorded_by_evidence_packet_record_draft": 0,
                "product_claims_allowed_by_evidence_packet_record_draft": 0,
                "pass_claims_allowed_by_evidence_packet_record_draft": 0,
                "dependency_unlocks_allowed_by_evidence_packet_record_draft": 0,
                "live_provider_calls_executed_by_evidence_packet_record_draft": 0,
                "provider_mutations_executed_by_evidence_packet_record_draft": 0,
                "external_mutations_executed_by_evidence_packet_record_draft": 0,
                "raw_packet_file_contents_recorded_by_evidence_packet_record_draft": 0,
                "packet_file_contents_persisted_by_evidence_packet_record_draft": 0,
            },
            "product_feature_claims": (
                connector_human_gate_evidence_packet_product_feature_claim(
                    scenario_id,
                    "RECORD_DRAFT",
                )
            ),
            "operator_rule": (
                f"This report derives a reviewer-record draft from {scenario_id} packet file hashes only. "
                "It leaves decision, reviewer, timestamp, and senior-review findings empty so "
                "a human must complete and validate the record before any dependency can unlock."
            ),
            "created_at": utc_now(),
        }
        report_base.update(connector_human_gate_completion_boundary())
        report_id = f"cshuman_evidencepacketrecorddraft_{json_hash(report_base)[:16]}"
        report = dict(report_base)
        report["evidence_packet_record_draft_report_id"] = report_id
        _write_json(self._human_gate_evidence_packet_record_draft_report_path(report_id), report)
        event = self.store.append_audit(
            "connector.human_gate_evidence_packet_record_draft.reported",
            requested_scope,
            {"type": "connector_human_gate_evidence_packet_record_draft_report", "id": report_id},
            {
                "scenario_id": scenario_id,
                "status": report["status"],
                "packet_validation_report_id": report["packet_validation_report_id"],
                "draft_record_available": draft_available,
                "final_verdict": report["final_verdict"],
                "product_claim_allowed": False,
                "pass_claim_allowed": False,
                "dependency_unlock_allowed_by_record_draft": False,
                "live_provider_calls_executed_by_evidence_packet_record_draft": 0,
                "provider_mutations_executed_by_evidence_packet_record_draft": 0,
                "external_mutations_executed_by_evidence_packet_record_draft": 0,
            },
        )
        return {
            "human_gate_evidence_packet_record_draft_report": report,
            "packet_validation_audit_event": validation_result.get("audit_event"),
            "audit_event": event,
        }

    def create_human_gate_preflight_bundle_report(
        self,
        requested_scope: dict[str, str],
        scenario_id: str,
        repo_root: Path,
    ) -> dict[str, Any]:
        local_baseline_review_inputs = connector_human_gate_local_baseline_review_inputs(repo_root, scenario_id)
        if local_baseline_review_inputs is None:
            unsupported = {
                "schema_version": CONNECTOR_HUMAN_GATE_PREFLIGHT_BUNDLE_REPORT_SCHEMA,
                "scenario_id": scenario_id,
                "status": "unsupported",
                "scope": requested_scope,
                "errors": [
                    {
                        "code": "CS_CONNECTOR_HUMAN_GATE_PREFLIGHT_BUNDLE_UNSUPPORTED",
                        "message": "Only CS-CH-H04 currently has a ConnectorHub human-gate preflight bundle.",
                    }
                ],
            }
            unsupported.update(connector_human_gate_completion_boundary())
            return {"human_gate_preflight_bundle_report": unsupported, "audit_event": None}
        preflight_bundle = deepcopy(local_baseline_review_inputs["preflight_bundle"])
        non_mutation_evidence = {
            "approval_collected_by_preflight_bundle": False,
            "human_decisions_recorded_by_preflight_bundle": 0,
            "commands_executed_by_preflight_bundle": 0,
            "live_provider_calls_executed_by_preflight_bundle": 0,
            "provider_mutations_executed_by_preflight_bundle": 0,
            "external_mutations_executed_by_preflight_bundle": 0,
            "secret_material_read_by_preflight_bundle": False,
            "record_bodies_persisted_by_preflight_bundle": False,
        }
        report_base = {
            "schema_version": CONNECTOR_HUMAN_GATE_PREFLIGHT_BUNDLE_REPORT_SCHEMA,
            "scenario_id": scenario_id,
            "status": "operator_preparation_only",
            "scope": requested_scope,
            "final_verdict": "HUMAN_REQUIRED",
            "weakest_applicable_scenario_result": "HUMAN_REQUIRED",
            "local_baseline_review_inputs_schema_version": local_baseline_review_inputs["schema_version"],
            "local_baseline_acceptance_sufficient": local_baseline_review_inputs["acceptance_sufficient"],
            "local_baseline_product_claim_allowed": local_baseline_review_inputs["product_claim_allowed"],
            "local_baseline_pass_claim_allowed": local_baseline_review_inputs["pass_claim_allowed"],
            "preflight_bundle": preflight_bundle,
            "required_human_delta": list(local_baseline_review_inputs["required_human_delta"]),
            "recommended_preflight_commands": list(
                local_baseline_review_inputs["recommended_preflight_commands"]
            ),
            "recommended_preflight_command_plan": deepcopy(
                local_baseline_review_inputs["recommended_preflight_command_plan"]
            ),
            "non_mutation_evidence": non_mutation_evidence,
            "negative_evidence": {
                "human_rows_marked_pass_by_preflight_bundle": 0,
                "approvals_collected_by_preflight_bundle": 0,
                "product_claims_allowed_by_preflight_bundle": 0,
                "pass_claims_allowed_by_preflight_bundle": 0,
                "commands_executed_by_preflight_bundle": 0,
                "live_provider_calls_executed_by_preflight_bundle": 0,
                "provider_mutations_executed_by_preflight_bundle": 0,
                "external_mutations_executed_by_preflight_bundle": 0,
                "secret_material_read_by_preflight_bundle": 0,
                "record_bodies_persisted_by_preflight_bundle": 0,
            },
            "product_feature_claims": "CONNECTOR_HUB_H04_PREFLIGHT_BUNDLE_PREPARED_HUMAN_EVIDENCE_REQUIRED",
            "operator_rule": (
                "This report exposes H04 local comparison inputs only. Run the command plan, collect "
                "fresh production-like evidence, and validate a dated human reviewer record before "
                "claiming dependency unlock."
            ),
            "created_at": utc_now(),
        }
        report_base.update(connector_human_gate_completion_boundary())
        report_id = f"cshuman_preflight_{json_hash(report_base)[:16]}"
        report = dict(report_base)
        report["preflight_bundle_report_id"] = report_id
        _write_json(self._human_gate_preflight_bundle_report_path(report_id), report)
        event = self.store.append_audit(
            "connector.human_gate_preflight_bundle.reported",
            requested_scope,
            {"type": "connector_human_gate_preflight_bundle_report", "id": report_id},
            {
                "scenario_id": scenario_id,
                "current_report_count": preflight_bundle["current_report_count"],
                "ready_report_count": preflight_bundle["ready_report_count"],
                "command_plan_count": preflight_bundle["command_plan_count"],
                "final_verdict": report["final_verdict"],
                "product_claim_allowed": False,
                "live_provider_calls_executed_by_preflight_bundle": 0,
                "provider_mutations_executed_by_preflight_bundle": 0,
            },
        )
        return {"human_gate_preflight_bundle_report": report, "audit_event": event}

    def create_human_gate_readiness_report(self, requested_scope: dict[str, str], repo_root: Path) -> dict[str, Any]:
        scenario_rows: list[dict[str, Any]] = []
        record_validations = self._list_human_gate_record_validations()
        validations_by_scenario: dict[str, list[dict[str, Any]]] = {}
        for validation in record_validations:
            scenario_id = str(validation.get("scenario_id") or "")
            if scenario_id:
                validations_by_scenario.setdefault(scenario_id, []).append(validation)
        for validations in validations_by_scenario.values():
            validations.sort(key=lambda item: str(item.get("created_at") or ""))

        scenarios_with_structurally_valid_record_validation = sorted(
            scenario_id
            for scenario_id, validations in validations_by_scenario.items()
            if any(connector_human_gate_validation_is_structurally_valid(validation) for validation in validations)
        )
        structurally_valid_record_validation_scenarios = set(scenarios_with_structurally_valid_record_validation)
        scenarios_with_dependency_unlock_record_validation = sorted(
            scenario_id
            for scenario_id, validations in validations_by_scenario.items()
            if any(connector_human_gate_validation_allows_dependency_unlock(validation) for validation in validations)
        )
        dependency_unlock_record_validation_scenarios = set(scenarios_with_dependency_unlock_record_validation)
        dependency_unlock_denied_record_validations = [
            str(validation.get("validation_id") or validation.get("scenario_id") or "unknown")
            for validation in record_validations
            if connector_human_gate_validation_is_structurally_valid(validation)
            and not connector_human_gate_validation_allows_dependency_unlock(validation)
        ]
        for scenario_id, definition in sorted(CONNECTOR_HUMAN_GATE_PACKAGE_DEFINITIONS.items()):
            template_path = connector_human_template_path(scenario_id)
            template_structure = connector_human_template_structure(repo_root / template_path, scenario_id)
            execution_queue_item = deepcopy(CONNECTOR_HUMAN_GATE_QUEUE_BY_SCENARIO[scenario_id])
            source_requirement_ids = connector_human_gate_source_requirement_ids(scenario_id)
            proof_boundary = deepcopy(definition["proof_boundary"])
            required_human_record = deepcopy(definition["required_human_record"])
            proposed_record_template = connector_human_gate_record_template(
                scenario_id,
                definition,
                execution_queue_item,
            )
            package_command = f"cornerstone connector human-gate package --scenario {scenario_id} --json"
            scenario_delivery_unit_plan = connector_human_gate_delivery_unit_plan(
                scenario_id,
                definition,
                execution_queue_item,
                package_command=package_command,
                record_template_output_command=proposed_record_template["record_template_output_command"],
                validation_command=proposed_record_template["validation_command"],
                validation_output_command=proposed_record_template["validation_output_command"],
            )
            scenario_delivery_unit_plan_summary = connector_human_gate_delivery_unit_plan_summary(
                scenario_delivery_unit_plan
            )
            local_baseline_review_inputs = connector_human_gate_local_baseline_review_inputs(repo_root, scenario_id)
            local_baseline_preflight_bundle = (
                local_baseline_review_inputs.get("preflight_bundle")
                if isinstance(local_baseline_review_inputs, dict)
                and isinstance(local_baseline_review_inputs.get("preflight_bundle"), dict)
                else None
            )
            scenario_validations = validations_by_scenario.get(scenario_id, [])
            latest_validation = scenario_validations[-1] if scenario_validations else None
            latest_validation_summary = None
            if latest_validation is not None:
                latest_validation_summary = {
                    "validation_id": latest_validation.get("validation_id"),
                    "status": latest_validation.get("status"),
                    "validation_scope": latest_validation.get("validation_scope"),
                    "record_file_sha256": (latest_validation.get("record_file") or {}).get("sha256"),
                    "missing_required_fields": latest_validation.get("missing_required_fields", []),
                    "empty_required_fields": latest_validation.get("empty_required_fields", []),
                    "invalid_field_formats": latest_validation.get("invalid_field_formats", []),
                    "field_ref_contract_present": latest_validation.get("field_ref_contract_present") is True,
                    "invalid_required_field_ref_shapes": latest_validation.get(
                        "invalid_required_field_ref_shapes",
                        [],
                    ),
                    "senior_review_perspective_findings_complete": latest_validation.get(
                        "senior_review_perspective_findings_complete"
                    )
                    is True,
                    "missing_senior_review_perspectives": latest_validation.get(
                        "missing_senior_review_perspectives",
                        [],
                    ),
                    "empty_senior_review_perspectives": latest_validation.get(
                        "empty_senior_review_perspectives",
                        [],
                    ),
                    "invalid_senior_review_perspective_roles": latest_validation.get(
                        "invalid_senior_review_perspective_roles",
                        [],
                    ),
                    "evidence_packet_manifest_complete": latest_validation.get(
                        "evidence_packet_manifest_complete"
                    )
                    is True,
                    "missing_evidence_packet_manifest_items": latest_validation.get(
                        "missing_evidence_packet_manifest_items",
                        [],
                    ),
                    "empty_evidence_packet_manifest_items": latest_validation.get(
                        "empty_evidence_packet_manifest_items",
                        [],
                    ),
                    "invalid_evidence_packet_manifest_items": latest_validation.get(
                        "invalid_evidence_packet_manifest_items",
                        [],
                    ),
                    "duplicate_evidence_packet_manifest_items": latest_validation.get(
                        "duplicate_evidence_packet_manifest_items",
                        [],
                    ),
                    "duplicate_evidence_packet_manifest_ref_fingerprints": latest_validation.get(
                        "duplicate_evidence_packet_manifest_ref_fingerprints",
                        [],
                    ),
                    "missing_dependency_human_gates": latest_validation.get("missing_dependency_human_gates", []),
                    "invalid_dependency_human_gate_refs": latest_validation.get("invalid_dependency_human_gate_refs", []),
                    "valid_dependency_human_gate_refs": latest_validation.get("valid_dependency_human_gate_refs", []),
                    "dependency_unlock_allowed_by_validator": latest_validation.get(
                        "dependency_unlock_allowed_by_validator"
                    )
                    is True,
                    "dependency_unlock_blocked_reason": latest_validation.get("dependency_unlock_blocked_reason"),
                    "sensitive_marker_findings": len(latest_validation.get("sensitive_marker_findings", [])),
                    "structural_errors": latest_validation.get("structural_errors", []),
                    "product_claim_allowed": latest_validation.get("product_claim_allowed") is True,
                    "pass_claim_allowed_by_validator": latest_validation.get("pass_claim_allowed_by_validator") is True,
                    "matrix_status_after_validation": latest_validation.get("matrix_status_after_validation"),
                }
            depends_on_human_gates_with_structurally_valid_record_validation = [
                dependency
                for dependency in execution_queue_item["depends_on"]
                if dependency in structurally_valid_record_validation_scenarios
            ]
            depends_on_human_gates_missing_structurally_valid_record_validation = [
                dependency
                for dependency in execution_queue_item["depends_on"]
                if dependency not in structurally_valid_record_validation_scenarios
            ]
            depends_on_human_gates_with_dependency_unlock_record_validation = [
                dependency
                for dependency in execution_queue_item["depends_on"]
                if dependency in dependency_unlock_record_validation_scenarios
            ]
            depends_on_human_gates_missing_dependency_unlock_record_validation = [
                dependency
                for dependency in execution_queue_item["depends_on"]
                if dependency not in dependency_unlock_record_validation_scenarios
            ]
            if not execution_queue_item["depends_on"]:
                depends_on_human_gate_record_validation_status = "not_applicable"
            elif depends_on_human_gates_missing_dependency_unlock_record_validation:
                depends_on_human_gate_record_validation_status = "missing_dependency_validations"
            else:
                depends_on_human_gate_record_validation_status = "ready"
            row_structurally_valid_record_validation_present = any(
                connector_human_gate_validation_is_structurally_valid(validation)
                for validation in scenario_validations
            )
            row_dependency_unlock_record_validation_present = any(
                connector_human_gate_validation_allows_dependency_unlock(validation)
                for validation in scenario_validations
            )
            scenario_row = {
                "scenario_id": scenario_id,
                "title": definition["title"],
                "status": "HUMAN_REQUIRED",
                "owner": "Human",
                "execution_queue_item": execution_queue_item,
                "review_order": execution_queue_item["order"],
                "depends_on_human_gates": execution_queue_item["depends_on"],
                "source_requirement_ids": source_requirement_ids,
                "source_requirement_count": len(source_requirement_ids),
                "source_requirement_status": "human_external_pending",
                "source_requirement_claim_boundary": CONNECTOR_HUMAN_GATE_SOURCE_REQUIREMENT_CLAIM_BOUNDARY,
                "gate_category": execution_queue_item["gate_category"],
                "stop_or_reject_when": execution_queue_item["stop_or_reject_when"],
                "package_supported": True,
                "package_command": package_command,
                "record_template_output_command": connector_human_gate_record_template_output_command(
                    scenario_id,
                    output_placeholder="<reviewer-record-template.json>",
                ),
                "record_validation_command": connector_human_gate_record_validation_command(
                    scenario_id,
                    record_placeholder="<json>",
                ),
                "record_validation_output_command": connector_human_gate_record_validation_command(
                    scenario_id,
                    record_placeholder="<json>",
                    output_placeholder="<redacted-validation-envelope.json>",
                ),
                "record_validation_count": len(scenario_validations),
                "record_validation_status": latest_validation.get("status") if latest_validation else "not_submitted",
                "structurally_valid_record_validation_present": row_structurally_valid_record_validation_present,
                "dependency_unlock_record_validation_present": row_dependency_unlock_record_validation_present,
                "depends_on_human_gate_record_validation_status": depends_on_human_gate_record_validation_status,
                "depends_on_human_gate_record_validations_ready": (
                    depends_on_human_gate_record_validation_status != "missing_dependency_validations"
                ),
                "depends_on_human_gates_with_structurally_valid_record_validation": (
                    depends_on_human_gates_with_structurally_valid_record_validation
                ),
                "depends_on_human_gates_missing_structurally_valid_record_validation": (
                    depends_on_human_gates_missing_structurally_valid_record_validation
                ),
                "depends_on_human_gates_with_dependency_unlock_record_validation": (
                    depends_on_human_gates_with_dependency_unlock_record_validation
                ),
                "depends_on_human_gates_missing_dependency_unlock_record_validation": (
                    depends_on_human_gates_missing_dependency_unlock_record_validation
                ),
                "latest_record_validation": latest_validation_summary,
                "template_path": str(template_path),
                "template_present": template_structure["template_present"],
                "template_structure_ready": template_structure["structure_ready"],
                "template_structure": template_structure,
                "proof_boundary": proof_boundary,
                "product_claim_allowed": proof_boundary.get("product_claim_allowed") is True,
                "pass_claim_allowed_without_human_record": proof_boundary.get("pass_claim_allowed_without_human_record") is True,
                "senior_review_perspective_count": len(definition["senior_review_perspectives"]),
                "rehearsal_check_count": len(definition["rehearsal_checklist"]),
                "required_human_fields": required_human_record.get("required_fields", []),
                "required_evidence": required_human_record.get("required_evidence", []),
                "proposed_record_template": proposed_record_template,
                "scenario_delivery_unit_plan": scenario_delivery_unit_plan,
                "scenario_delivery_unit_plan_summary": scenario_delivery_unit_plan_summary,
                "release_impact": definition["release_impact"],
            }
            if local_baseline_review_inputs is not None:
                scenario_row["local_baseline_review_inputs"] = local_baseline_review_inputs
            if local_baseline_preflight_bundle is not None:
                scenario_row["local_baseline_preflight_bundle"] = deepcopy(local_baseline_preflight_bundle)
            scenario_rows.append(scenario_row)
        missing_templates = [
            row["template_path"]
            for row in scenario_rows
            if not row["template_present"]
        ]
        template_structure_missing = [
            row["scenario_id"]
            for row in scenario_rows
            if not row["template_structure_ready"]
        ]
        scenario_first_runbook_missing = [
            row["scenario_id"]
            for row in scenario_rows
            if not row["template_structure"]["has_scenario_first_runbook_or_study"]
        ]
        evidence_packet_missing = [
            row["scenario_id"]
            for row in scenario_rows
            if not row["template_structure"]["has_acceptance_evidence_packet"]
        ]
        senior_review_perspectives_missing = [
            row["scenario_id"]
            for row in scenario_rows
            if not row["template_structure"]["has_senior_review_perspectives"]
        ]
        reject_conditions_missing = [
            row["scenario_id"]
            for row in scenario_rows
            if not row["template_structure"]["has_reject_conditions"]
        ]
        no_pass_boundary_missing = [
            row["scenario_id"]
            for row in scenario_rows
            if not row["template_structure"]["has_no_pass_boundary"]
        ]
        row_status_issues = [
            row["scenario_id"]
            for row in scenario_rows
            if row["status"] != "HUMAN_REQUIRED"
            or row["product_claim_allowed"]
            or row["pass_claim_allowed_without_human_record"]
            or not row["package_supported"]
            or row["senior_review_perspective_count"] < 6
            or not row["required_human_fields"]
            or not row["required_evidence"]
            or not row["template_structure_ready"]
        ]
        record_validation_scenarios = sorted(validations_by_scenario)
        record_validation_product_claims_allowed = [
            str(validation.get("validation_id") or validation.get("scenario_id") or "unknown")
            for validation in record_validations
            if validation.get("product_claim_allowed") is True
        ]
        record_validation_pass_claims_allowed = [
            str(validation.get("validation_id") or validation.get("scenario_id") or "unknown")
            for validation in record_validations
            if validation.get("pass_claim_allowed_by_validator") is True
        ]
        record_validation_body_persisted = [
            str(validation.get("validation_id") or validation.get("scenario_id") or "unknown")
            for validation in record_validations
            if (validation.get("non_mutation_evidence") or {}).get("record_body_persisted_by_validator") is not False
            or (validation.get("negative_evidence") or {}).get("record_body_persisted_by_validator", 0) != 0
        ]
        scenario_delivery_unit_plan_missing = [
            row["scenario_id"]
            for row in scenario_rows
            if not row["scenario_delivery_unit_plan_summary"]["scenario_delivery_unit_plan_ready"]
        ]
        scenario_delivery_unit_plan_product_claims_allowed = [
            row["scenario_id"]
            for row in scenario_rows
            if row["scenario_delivery_unit_plan_summary"]["scenario_delivery_unit_plan_product_claim_allowed"]
        ]
        scenario_delivery_unit_plan_pass_claims_allowed = [
            row["scenario_id"]
            for row in scenario_rows
            if row["scenario_delivery_unit_plan_summary"]["scenario_delivery_unit_plan_pass_claim_allowed"]
        ]
        scenario_delivery_unit_plan_approvals_collected = [
            row["scenario_id"]
            for row in scenario_rows
            if row["scenario_delivery_unit_plan_summary"]["scenario_delivery_unit_plan_approval_collected"]
        ]
        scenario_delivery_unit_plan_dependency_unlock_allowed = [
            row["scenario_id"]
            for row in scenario_rows
            if row["scenario_delivery_unit_plan_summary"]["scenario_delivery_unit_plan_dependency_unlock_allowed"]
        ]
        report_base = {
            "schema_version": CONNECTOR_HUMAN_GATE_READINESS_REPORT_SCHEMA,
            "status": "human_review_required",
            "scope": requested_scope,
            "scenario_count": len(scenario_rows),
            "human_required_count": len([row for row in scenario_rows if row["status"] == "HUMAN_REQUIRED"]),
            "package_supported_count": len([row for row in scenario_rows if row["package_supported"]]),
            "template_present_count": len([row for row in scenario_rows if row["template_present"]]),
            "execution_queue": deepcopy(CONNECTOR_HUMAN_GATE_EXECUTION_QUEUE),
            "execution_queue_scenario_order": [item["scenario_id"] for item in CONNECTOR_HUMAN_GATE_EXECUTION_QUEUE],
            "record_validation_summary": {
                "validation_count": len(record_validations),
                "structurally_valid_count": len(
                    [
                        validation
                        for validation in record_validations
                        if connector_human_gate_validation_is_structurally_valid(validation)
                    ]
                ),
                "structurally_invalid_count": len(
                    [
                        validation
                        for validation in record_validations
                        if validation.get("status") == "record_structurally_invalid"
                    ]
                ),
                "senior_review_perspective_findings_complete_count": len(
                    [
                        validation
                        for validation in record_validations
                        if validation.get("senior_review_perspective_findings_complete") is True
                    ]
                ),
                "senior_review_perspective_findings_incomplete_count": len(
                    [
                        validation
                        for validation in record_validations
                        if validation.get("senior_review_perspective_findings_complete") is not True
                    ]
                ),
                "evidence_packet_manifest_complete_count": len(
                    [
                        validation
                        for validation in record_validations
                        if validation.get("evidence_packet_manifest_complete") is True
                    ]
                ),
                "evidence_packet_manifest_incomplete_count": len(
                    [
                        validation
                        for validation in record_validations
                        if validation.get("evidence_packet_manifest_complete") is not True
                    ]
                ),
                "scenarios_with_record_validation": record_validation_scenarios,
                "scenarios_with_structurally_valid_record_validation": scenarios_with_structurally_valid_record_validation,
                "scenarios_missing_structurally_valid_record_validation": [
                    row["scenario_id"]
                    for row in scenario_rows
                    if row["scenario_id"] not in scenarios_with_structurally_valid_record_validation
                ],
                "dependency_unlock_allowed_count": len(
                    [
                        validation
                        for validation in record_validations
                        if connector_human_gate_validation_allows_dependency_unlock(validation)
                    ]
                ),
                "dependency_unlock_denied_count": len(dependency_unlock_denied_record_validations),
                "dependency_unlock_denied_record_validations": dependency_unlock_denied_record_validations,
                "scenarios_with_dependency_unlock_record_validation": scenarios_with_dependency_unlock_record_validation,
                "scenarios_missing_dependency_unlock_record_validation": [
                    row["scenario_id"]
                    for row in scenario_rows
                    if row["scenario_id"] not in scenarios_with_dependency_unlock_record_validation
                ],
                "depends_on_human_gate_record_validation_ready_rows": [
                    row["scenario_id"]
                    for row in scenario_rows
                    if row["depends_on_human_gate_record_validation_status"] == "ready"
                ],
                "depends_on_human_gate_record_validation_missing_rows": [
                    row["scenario_id"]
                    for row in scenario_rows
                    if row["depends_on_human_gate_record_validation_status"] == "missing_dependency_validations"
                ],
                "depends_on_human_gate_record_validation_not_applicable_rows": [
                    row["scenario_id"]
                    for row in scenario_rows
                    if row["depends_on_human_gate_record_validation_status"] == "not_applicable"
                ],
                "depends_on_human_gate_record_validation_ready_count": len(
                    [
                        row
                        for row in scenario_rows
                        if row["depends_on_human_gate_record_validation_status"] == "ready"
                    ]
                ),
                "depends_on_human_gate_record_validation_missing_count": len(
                    [
                        row
                        for row in scenario_rows
                        if row["depends_on_human_gate_record_validation_status"] == "missing_dependency_validations"
                    ]
                ),
                "depends_on_human_gate_record_validation_not_applicable_count": len(
                    [
                        row
                        for row in scenario_rows
                        if row["depends_on_human_gate_record_validation_status"] == "not_applicable"
                    ]
                ),
                "product_claims_allowed_by_validations": len(record_validation_product_claims_allowed),
                "pass_claims_allowed_by_validations": len(record_validation_pass_claims_allowed),
                "record_bodies_persisted_by_validations": len(record_validation_body_persisted),
            },
            "template_structure_ready_count": len([row for row in scenario_rows if row["template_structure_ready"]]),
            "scenario_first_runbook_ready_count": len(
                [
                    row
                    for row in scenario_rows
                    if row["template_structure"]["has_scenario_first_runbook_or_study"]
                ]
            ),
            "evidence_packet_ready_count": len(
                [row for row in scenario_rows if row["template_structure"]["has_acceptance_evidence_packet"]]
            ),
            "senior_review_perspectives_ready_count": len(
                [
                    row
                    for row in scenario_rows
                    if row["template_structure"]["has_senior_review_perspectives"]
                ]
            ),
            "reject_conditions_ready_count": len(
                [row for row in scenario_rows if row["template_structure"]["has_reject_conditions"]]
            ),
            "no_pass_boundary_ready_count": len(
                [row for row in scenario_rows if row["template_structure"]["has_no_pass_boundary"]]
            ),
            "scenario_delivery_unit_plan_ready_count": len(
                [
                    row
                    for row in scenario_rows
                    if row["scenario_delivery_unit_plan_summary"]["scenario_delivery_unit_plan_ready"]
                ]
            ),
            "scenario_delivery_unit_plan_missing": scenario_delivery_unit_plan_missing,
            "scenario_delivery_unit_plan_product_claims_allowed": scenario_delivery_unit_plan_product_claims_allowed,
            "scenario_delivery_unit_plan_pass_claims_allowed": scenario_delivery_unit_plan_pass_claims_allowed,
            "scenario_delivery_unit_plan_approvals_collected": scenario_delivery_unit_plan_approvals_collected,
            "scenario_delivery_unit_plan_dependency_unlock_allowed": (
                scenario_delivery_unit_plan_dependency_unlock_allowed
            ),
            "scenario_results": scenario_rows,
            "non_mutation_evidence": {
                "approval_collected_by_ai": False,
                "human_decisions_recorded_by_report": 0,
                "live_provider_calls_executed_by_report": 0,
                "provider_mutations_executed_by_report": 0,
                "external_mutations_executed_by_report": 0,
                "secret_material_read_by_report": False,
                "human_record_validations_executed_by_report": len(record_validations),
                "record_bodies_persisted_by_report": False,
            },
            "negative_evidence": {
                "human_rows_marked_pass_by_report": len([row for row in scenario_rows if row["status"] == "PASS"]),
                "package_support_missing": len([row for row in scenario_rows if not row["package_supported"]]),
                "template_files_missing": len(missing_templates),
                "template_structure_missing": len(template_structure_missing),
                "scenario_first_runbook_missing": len(scenario_first_runbook_missing),
                "evidence_packet_missing": len(evidence_packet_missing),
                "senior_review_perspectives_missing": len(senior_review_perspectives_missing),
                "reject_conditions_missing": len(reject_conditions_missing),
                "no_pass_boundary_missing": len(no_pass_boundary_missing),
                "scenario_delivery_unit_plan_missing": len(scenario_delivery_unit_plan_missing),
                "scenario_delivery_unit_plan_product_claims_allowed": len(
                    scenario_delivery_unit_plan_product_claims_allowed
                ),
                "scenario_delivery_unit_plan_pass_claims_allowed": len(
                    scenario_delivery_unit_plan_pass_claims_allowed
                ),
                "scenario_delivery_unit_plan_approvals_collected": len(
                    scenario_delivery_unit_plan_approvals_collected
                ),
                "scenario_delivery_unit_plan_dependency_unlock_allowed": len(
                    scenario_delivery_unit_plan_dependency_unlock_allowed
                ),
                "product_claims_allowed_by_report": len([row for row in scenario_rows if row["product_claim_allowed"]]),
                "pass_without_human_record_allowed_by_report": len(
                    [row for row in scenario_rows if row["pass_claim_allowed_without_human_record"]]
                ),
                "row_status_or_contract_issues": len(row_status_issues),
                "approvals_collected_by_report": 0,
                "live_provider_calls_executed_by_report": 0,
                "provider_mutations_executed_by_report": 0,
                "external_mutations_executed_by_report": 0,
                "secret_material_read_by_report": 0,
                "product_claims_allowed_by_validations": len(record_validation_product_claims_allowed),
                "pass_claims_allowed_by_validations": len(record_validation_pass_claims_allowed),
                "record_bodies_persisted_by_validations": len(record_validation_body_persisted),
            },
            "missing_templates": missing_templates,
            "row_status_issues": row_status_issues,
            "final_verdict": "HUMAN_REQUIRED",
            "weakest_applicable_scenario_result": "HUMAN_REQUIRED",
            "product_feature_claims": "CONNECTOR_HUB_HUMAN_GATES_PREPARED_HUMAN_EVIDENCE_REQUIRED",
            "required_before_pass": [
                "Attach dated human records for H01 through H07 before changing any human row to PASS.",
                "Keep live-provider, physical-device, browser privacy, production-like VS2, live Action, UX/trust, and recovery evidence separate.",
            ],
            "created_at": utc_now(),
        }
        report_base.update(connector_human_gate_completion_boundary())
        report_id = f"cshuman_report_{json_hash(report_base)[:16]}"
        report = dict(report_base)
        report["report_id"] = report_id
        _write_json(self._human_gate_readiness_report_path(report_id), report)
        event = self.store.append_audit(
            "connector.human_gate_readiness.reported",
            requested_scope,
            {"type": "connector_human_gate_readiness_report", "id": report_id},
            {
                "scenario_count": report["scenario_count"],
                "human_required_count": report["human_required_count"],
                "template_present_count": report["template_present_count"],
                "template_structure_ready_count": report["template_structure_ready_count"],
                "final_verdict": report["final_verdict"],
                "product_claim_allowed": False,
                "live_provider_calls_executed_by_report": 0,
                "provider_mutations_executed_by_report": 0,
            },
        )
        return {"human_gate_readiness_report": report, "audit_event": event}

    def create_human_gate_next_report(self, requested_scope: dict[str, str], repo_root: Path) -> dict[str, Any]:
        readiness_result = self._get_or_create_human_gate_readiness_report(requested_scope, repo_root)
        readiness_report = readiness_result["human_gate_readiness_report"]
        scenario_rows = sorted(
            readiness_report["scenario_results"],
            key=lambda row: int(row["review_order"]),
        )
        completed_rows = [
            row for row in scenario_rows if row["dependency_unlock_record_validation_present"]
        ]
        ready_rows = [
            row
            for row in scenario_rows
            if not row["dependency_unlock_record_validation_present"]
            and row["depends_on_human_gate_record_validations_ready"]
            and row["template_structure_ready"]
            and row["package_supported"]
        ]
        blocked_rows = [
            row
            for row in scenario_rows
            if not row["dependency_unlock_record_validation_present"]
            and not row["depends_on_human_gate_record_validations_ready"]
        ]
        next_row = ready_rows[0] if ready_rows else None
        validation_summary = readiness_report.get("record_validation_summary", {})
        next_delivery_unit_plan_summary = (
            deepcopy(next_row.get("scenario_delivery_unit_plan_summary"))
            if next_row
            else None
        )
        next_remaining_human_evidence_summary = (
            connector_human_gate_remaining_evidence_summary(next_row)
            if next_row
            else None
        )
        source_requirement_human_pending_ids = connector_human_gate_unique_source_requirement_ids(scenario_rows)
        next_latest_validation = (
            next_row.get("latest_record_validation")
            if next_row and isinstance(next_row.get("latest_record_validation"), dict)
            else None
        )
        next_local_baseline_review_inputs = (
            next_row.get("local_baseline_review_inputs")
            if next_row and isinstance(next_row.get("local_baseline_review_inputs"), dict)
            else None
        )
        next_local_baseline_preflight_bundle = (
            next_local_baseline_review_inputs.get("preflight_bundle")
            if isinstance(next_local_baseline_review_inputs, dict)
            and isinstance(next_local_baseline_review_inputs.get("preflight_bundle"), dict)
            else None
        )
        next_base = {
            "schema_version": CONNECTOR_HUMAN_GATE_NEXT_SCHEMA,
            "status": (
                "next_ready"
                if next_row is not None
                else "all_human_gate_dependency_unlock_records_present"
                if len(completed_rows) == len(scenario_rows)
                else "no_dependency_ready_human_gate"
            ),
            "scope": requested_scope,
            "final_verdict": "HUMAN_REQUIRED",
            "weakest_applicable_scenario_result": readiness_report["weakest_applicable_scenario_result"],
            "scenario_count": readiness_report["scenario_count"],
            "human_required_count": readiness_report["human_required_count"],
            "completed_human_gate_record_validation_count": len(completed_rows),
            "completed_human_gate_record_validation_scenarios": [row["scenario_id"] for row in completed_rows],
            "completed_dependency_unlock_record_validation_count": len(completed_rows),
            "completed_dependency_unlock_record_validation_scenarios": [row["scenario_id"] for row in completed_rows],
            "record_validation_count": validation_summary.get("validation_count", 0),
            "structurally_valid_record_validation_count": validation_summary.get("structurally_valid_count", 0),
            "dependency_unlock_allowed_count": validation_summary.get("dependency_unlock_allowed_count", 0),
            "dependency_unlock_denied_count": validation_summary.get("dependency_unlock_denied_count", 0),
            "next_scenario_id": next_row["scenario_id"] if next_row else None,
            "next_review_order": next_row["review_order"] if next_row else None,
            "next_gate_category": next_row["gate_category"] if next_row else None,
            "next_title": next_row["title"] if next_row else None,
            "next_depends_on_human_gates": next_row["depends_on_human_gates"] if next_row else [],
            "next_source_requirement_ids": next_row["source_requirement_ids"] if next_row else [],
            "next_source_requirement_count": next_row["source_requirement_count"] if next_row else 0,
            "source_requirement_human_pending_ids": source_requirement_human_pending_ids,
            "source_requirement_human_pending_count": len(source_requirement_human_pending_ids),
            "source_requirement_claim_boundary": CONNECTOR_HUMAN_GATE_SOURCE_REQUIREMENT_CLAIM_BOUNDARY,
            "next_package_command": next_row["package_command"] if next_row else None,
            "next_record_template_output_command": (
                next_row["record_template_output_command"] if next_row else None
            ),
            "next_record_validation_command": next_row["record_validation_command"] if next_row else None,
            "next_record_validation_output_command": next_row["record_validation_output_command"] if next_row else None,
            "next_record_validation_count": next_row["record_validation_count"] if next_row else 0,
            "next_record_validation_status": (
                next_row["record_validation_status"] if next_row else "not_submitted"
            ),
            "next_latest_record_validation_ref": (
                f"connector_human_gate_record_validation:{next_latest_validation['validation_id']}"
                if next_latest_validation
                and isinstance(next_latest_validation.get("validation_id"), str)
                else None
            ),
            "next_latest_record_validation_dependency_unlock_allowed": (
                bool(next_latest_validation.get("dependency_unlock_allowed_by_validator"))
                if next_latest_validation
                else False
            ),
            "next_latest_record_validation_issue_summary": (
                connector_human_gate_validation_issue_summary(next_latest_validation)
                if next_latest_validation
                else None
            ),
            "next_record_redaction_guidance": (
                deepcopy((next_row["proposed_record_template"] or {}).get("redaction_guidance"))
                if next_row
                else None
            ),
            "next_reviewer_checklist": (
                deepcopy((next_row["proposed_record_template"] or {}).get("reviewer_checklist"))
                if next_row
                else None
            ),
            "next_scenario_delivery_unit_plan": (
                deepcopy(next_row.get("scenario_delivery_unit_plan"))
                if next_row
                else None
            ),
            "next_scenario_delivery_unit_plan_summary": next_delivery_unit_plan_summary,
            "next_scenario_delivery_unit_plan_ready": (
                (next_delivery_unit_plan_summary or {}).get("scenario_delivery_unit_plan_ready") is True
            ),
            "next_scenario_delivery_unit_plan_lifecycle_step_count": (
                (next_delivery_unit_plan_summary or {}).get("scenario_delivery_unit_plan_lifecycle_step_count", 0)
            ),
            "next_scenario_delivery_unit_plan_senior_review_perspective_count": (
                (next_delivery_unit_plan_summary or {}).get(
                    "scenario_delivery_unit_plan_senior_review_perspective_count",
                    0,
                )
            ),
            "next_scenario_delivery_unit_plan_product_claim_allowed": (
                (next_delivery_unit_plan_summary or {}).get("scenario_delivery_unit_plan_product_claim_allowed")
                is True
            ),
            "next_scenario_delivery_unit_plan_pass_claim_allowed": (
                (next_delivery_unit_plan_summary or {}).get("scenario_delivery_unit_plan_pass_claim_allowed")
                is True
            ),
            "next_scenario_delivery_unit_plan_approval_collected": (
                (next_delivery_unit_plan_summary or {}).get("scenario_delivery_unit_plan_approval_collected")
                is True
            ),
            "next_scenario_delivery_unit_plan_dependency_unlock_allowed": (
                (next_delivery_unit_plan_summary or {}).get(
                    "scenario_delivery_unit_plan_dependency_unlock_allowed"
                )
                is True
            ),
            "next_template_path": next_row["template_path"] if next_row else None,
            "next_required_human_fields": next_row["required_human_fields"] if next_row else [],
            "next_required_evidence": next_row["required_evidence"] if next_row else [],
            "next_release_impact": next_row["release_impact"] if next_row else None,
            "next_stop_or_reject_when": next_row["stop_or_reject_when"] if next_row else None,
            "next_remaining_human_evidence_summary": next_remaining_human_evidence_summary,
            "next_remaining_human_field_count": (
                next_remaining_human_evidence_summary["required_human_field_count"]
                if next_remaining_human_evidence_summary
                else 0
            ),
            "next_remaining_human_evidence_count": (
                next_remaining_human_evidence_summary["required_evidence_count"]
                if next_remaining_human_evidence_summary
                else 0
            ),
            "next_remaining_human_evidence_claim_boundary": (
                CONNECTOR_HUMAN_GATE_REMAINING_EVIDENCE_CLAIM_BOUNDARY
            ),
            "next_required_human_delta": (
                next_local_baseline_review_inputs.get("required_human_delta", [])
                if next_local_baseline_review_inputs is not None
                else []
            ),
            "next_recommended_preflight_commands": (
                next_local_baseline_review_inputs.get("recommended_preflight_commands", [])
                if next_local_baseline_review_inputs is not None
                else []
            ),
            "next_recommended_preflight_command_plan": (
                next_local_baseline_review_inputs.get("recommended_preflight_command_plan", [])
                if next_local_baseline_review_inputs is not None
                else []
            ),
            "next_local_baseline_review_inputs": (
                deepcopy(next_local_baseline_review_inputs)
                if next_local_baseline_review_inputs is not None
                else None
            ),
            "next_local_baseline_preflight_bundle": (
                deepcopy(next_local_baseline_preflight_bundle)
                if next_local_baseline_preflight_bundle is not None
                else None
            ),
            "ready_scenario_ids": [row["scenario_id"] for row in ready_rows],
            "blocked_scenario_ids": [row["scenario_id"] for row in blocked_rows],
            "blocked_by_missing_dependency_validations": {
                row["scenario_id"]: row["depends_on_human_gates_missing_dependency_unlock_record_validation"]
                for row in blocked_rows
            },
            "blocked_by_missing_dependency_unlock_validations": {
                row["scenario_id"]: row["depends_on_human_gates_missing_dependency_unlock_record_validation"]
                for row in blocked_rows
            },
            "readiness_report_id": readiness_report["report_id"],
            "readiness_report_ref": f"connector_human_gate_readiness_report:{readiness_report['report_id']}",
            "non_mutation_evidence": {
                "approval_collected_by_ai": False,
                "human_decisions_recorded_by_next": 0,
                "live_provider_calls_executed_by_next": 0,
                "provider_mutations_executed_by_next": 0,
                "external_mutations_executed_by_next": 0,
                "secret_material_read_by_next": False,
                "human_record_validations_executed_by_next": 0,
                "record_bodies_persisted_by_next": False,
            },
            "negative_evidence": {
                "human_rows_marked_pass_by_next": 0,
                "approvals_collected_by_next": 0,
                "live_provider_calls_executed_by_next": 0,
                "provider_mutations_executed_by_next": 0,
                "external_mutations_executed_by_next": 0,
                "secret_material_read_by_next": 0,
                "record_bodies_persisted_by_next": 0,
                "product_claims_allowed_by_next": 0,
                "pass_claims_allowed_by_next": 0,
            },
            "product_feature_claims": readiness_report["product_feature_claims"],
            "required_before_pass": readiness_report["required_before_pass"],
            "created_at": utc_now(),
        }
        next_base.update(connector_human_gate_completion_boundary())
        next_id = f"cshuman_next_{json_hash(next_base)[:16]}"
        next_report = dict(next_base)
        next_report["next_id"] = next_id
        _write_json(self._human_gate_next_path(next_id), next_report)
        event = self.store.append_audit(
            "connector.human_gate_next.selected",
            requested_scope,
            {"type": "connector_human_gate_next", "id": next_id},
            {
                "next_scenario_id": next_report["next_scenario_id"],
                "ready_scenario_ids": next_report["ready_scenario_ids"],
                "blocked_scenario_ids": next_report["blocked_scenario_ids"],
                "completed_human_gate_record_validation_count": next_report[
                    "completed_human_gate_record_validation_count"
                ],
                "final_verdict": next_report["final_verdict"],
                "product_claim_allowed": False,
                "live_provider_calls_executed_by_next": 0,
                "provider_mutations_executed_by_next": 0,
            },
        )
        return {
            "human_gate_next": next_report,
            "human_gate_readiness_report": readiness_report,
            "readiness_audit_event": readiness_result.get("audit_event"),
            "audit_event": event,
        }

    def create_human_gate_validation_handoff(self, requested_scope: dict[str, str], repo_root: Path) -> dict[str, Any]:
        readiness_result = self._get_or_create_human_gate_readiness_report(requested_scope, repo_root)
        readiness_report = readiness_result["human_gate_readiness_report"]
        scenario_rows = sorted(
            readiness_report["scenario_results"],
            key=lambda row: int(row["review_order"]),
        )
        validation_summary = readiness_report.get("record_validation_summary", {})
        completed_rows = [
            row for row in scenario_rows if row["dependency_unlock_record_validation_present"]
        ]
        ready_rows = [
            row
            for row in scenario_rows
            if not row["dependency_unlock_record_validation_present"]
            and row["depends_on_human_gate_record_validations_ready"]
            and row["template_structure_ready"]
            and row["package_supported"]
        ]
        blocked_rows = [
            row
            for row in scenario_rows
            if not row["dependency_unlock_record_validation_present"]
            and not row["depends_on_human_gate_record_validations_ready"]
        ]
        next_row = ready_rows[0] if ready_rows else None
        rows_by_scenario_id = {row["scenario_id"]: row for row in scenario_rows}
        source_requirement_human_pending_ids = connector_human_gate_unique_source_requirement_ids(scenario_rows)
        handoff_rows = []
        for row in scenario_rows:
            required_human_fields = list(row.get("required_human_fields") or [])
            required_evidence = list(row.get("required_evidence") or [])
            remaining_human_evidence_summary = connector_human_gate_remaining_evidence_summary(row)
            latest_validation = (
                row.get("latest_record_validation")
                if isinstance(row.get("latest_record_validation"), dict)
                else None
            )
            validation_id = (
                str(latest_validation.get("validation_id"))
                if latest_validation and latest_validation.get("validation_id")
                else None
            )
            proposed_record_template = (
                row.get("proposed_record_template")
                if isinstance(row.get("proposed_record_template"), dict)
                else {}
            )
            redaction_guidance = (
                proposed_record_template.get("redaction_guidance")
                if isinstance(proposed_record_template.get("redaction_guidance"), dict)
                else {}
            )
            redaction_sensitive_marker_policy = (
                redaction_guidance.get("sensitive_marker_policy")
                if isinstance(redaction_guidance.get("sensitive_marker_policy"), dict)
                else {}
            )
            redaction_field_guidance = (
                redaction_guidance.get("field_guidance")
                if isinstance(redaction_guidance.get("field_guidance"), dict)
                else {}
            )
            redaction_evidence_manifest = (
                redaction_guidance.get("evidence_packet_manifest")
                if isinstance(redaction_guidance.get("evidence_packet_manifest"), dict)
                else {}
            )
            redaction_dependency_refs = (
                redaction_guidance.get("dependency_human_gate_refs")
                if isinstance(redaction_guidance.get("dependency_human_gate_refs"), dict)
                else {}
            )
            reviewer_checklist = (
                proposed_record_template.get("reviewer_checklist")
                if isinstance(proposed_record_template.get("reviewer_checklist"), dict)
                else {}
            )
            checklist_required_field_items = (
                reviewer_checklist.get("required_field_items")
                if isinstance(reviewer_checklist.get("required_field_items"), list)
                else []
            )
            checklist_senior_review_items = (
                reviewer_checklist.get("senior_review_perspective_items")
                if isinstance(reviewer_checklist.get("senior_review_perspective_items"), list)
                else []
            )
            checklist_evidence_manifest_items = (
                reviewer_checklist.get("evidence_packet_manifest_items")
                if isinstance(reviewer_checklist.get("evidence_packet_manifest_items"), list)
                else []
            )
            checklist_dependency_ref_items = (
                reviewer_checklist.get("dependency_human_gate_ref_items")
                if isinstance(reviewer_checklist.get("dependency_human_gate_ref_items"), list)
                else []
            )
            local_baseline = (
                row.get("local_baseline_review_inputs")
                if isinstance(row.get("local_baseline_review_inputs"), dict)
                else None
            )
            local_baseline_summary = None
            preflight_bundle = None
            if local_baseline is not None:
                preflight_bundle = (
                    local_baseline.get("preflight_bundle")
                    if isinstance(local_baseline.get("preflight_bundle"), dict)
                    else None
                )
                report_summaries = []
                for report in local_baseline.get("reports") or []:
                    if not isinstance(report, dict):
                        continue
                    report_summary = {
                        "path": report.get("path"),
                        "status": report.get("status"),
                        "schema_version": report.get("schema_version"),
                        "command": report.get("command"),
                        "scenario_count": report.get("scenario_count"),
                        "summary": report.get("summary"),
                        "sha256": report.get("sha256"),
                        "review_input_only": report.get("review_input_only"),
                        "acceptance_sufficient": report.get("acceptance_sufficient"),
                        "product_claim_allowed": report.get("product_claim_allowed"),
                        "pass_claim_allowed": report.get("pass_claim_allowed"),
                        "claim_boundary": report.get("claim_boundary"),
                    }
                    report_summaries.append(
                        {key: value for key, value in report_summary.items() if value is not None}
                    )
                local_baseline_summary = {
                    "schema_version": local_baseline.get("schema_version"),
                    "status": local_baseline.get("status"),
                    "report_count": len(local_baseline.get("reports") or []),
                    "report_paths": [
                        report.get("path")
                        for report in local_baseline.get("reports") or []
                        if isinstance(report, dict) and report.get("path")
                    ],
                    "report_summaries": report_summaries,
                    "required_human_delta_count": len(local_baseline.get("required_human_delta") or []),
                    "recommended_preflight_command_count": len(
                        local_baseline.get("recommended_preflight_commands") or []
                    ),
                    "recommended_preflight_command_plan_schema_version": local_baseline.get(
                        "recommended_preflight_command_plan_schema_version"
                    ),
                    "recommended_preflight_command_plan_count": len(
                        local_baseline.get("recommended_preflight_command_plan") or []
                    ),
                    "recommended_preflight_command_plan": deepcopy(
                        local_baseline.get("recommended_preflight_command_plan") or []
                    ),
                    "preflight_bundle": deepcopy(preflight_bundle) if preflight_bundle else None,
                    "preflight_bundle_schema_version": (
                        preflight_bundle.get("schema_version") if preflight_bundle else None
                    ),
                    "preflight_bundle_ready_report_count": (
                        preflight_bundle.get("ready_report_count") if preflight_bundle else None
                    ),
                    "preflight_bundle_acceptance_sufficient": (
                        preflight_bundle.get("acceptance_sufficient") if preflight_bundle else None
                    ),
                    "acceptance_sufficient": local_baseline.get("acceptance_sufficient") is True,
                    "product_claim_allowed": local_baseline.get("product_claim_allowed") is True,
                    "pass_claim_allowed": local_baseline.get("pass_claim_allowed") is True,
                }
            missing_dependency_ids = row["depends_on_human_gates_missing_dependency_unlock_record_validation"]
            blocked_dependency_details = []
            for dependency_id in missing_dependency_ids:
                dependency_row = rows_by_scenario_id.get(dependency_id)
                if dependency_row is None:
                    continue
                blocked_dependency_details.append(
                    {
                        "scenario_id": dependency_id,
                        "package_command": dependency_row["package_command"],
                        "record_template_output_command": dependency_row[
                            "record_template_output_command"
                        ],
                        "validate_record_command": dependency_row["record_validation_command"],
                        "validate_record_output_command": dependency_row[
                            "record_validation_output_command"
                        ],
                        "accepted_ref_prefix": "connector_human_gate_record_validation:",
                        "required_status": "structurally_valid_accept_record",
                        "unlock_rule": (
                            "Only structurally valid ACCEPT validation refs unlock dependent H gates."
                        ),
                    }
                )
            handoff_rows.append(
                {
                    "scenario_id": row["scenario_id"],
                    "review_order": row["review_order"],
                    "status": "HUMAN_REQUIRED",
                    "owner": "Human",
                    "gate_category": row["gate_category"],
                    "depends_on_human_gates": row["depends_on_human_gates"],
                    "source_requirement_ids": row["source_requirement_ids"],
                    "source_requirement_count": row["source_requirement_count"],
                    "source_requirement_status": row["source_requirement_status"],
                    "source_requirement_claim_boundary": row["source_requirement_claim_boundary"],
                    "stop_or_reject_when": row["stop_or_reject_when"],
                    "dependency_status": row["depends_on_human_gate_record_validation_status"],
                    "dependency_ready": row["depends_on_human_gate_record_validations_ready"],
                    "blocked_by_missing_dependency_unlock_validations": row[
                        "depends_on_human_gates_missing_dependency_unlock_record_validation"
                    ],
                    "blocked_dependency_details": blocked_dependency_details,
                    "record_validation_count": row["record_validation_count"],
                    "record_validation_status": row["record_validation_status"],
                    "structurally_valid_record_validation_present": row[
                        "structurally_valid_record_validation_present"
                    ],
                    "dependency_unlock_record_validation_present": row[
                        "dependency_unlock_record_validation_present"
                    ],
                    "latest_record_validation_ref": (
                        f"connector_human_gate_record_validation:{validation_id}"
                        if validation_id
                        else None
                    ),
                    "latest_record_validation_dependency_unlock_allowed": (
                        latest_validation.get("dependency_unlock_allowed_by_validator") is True
                        if latest_validation
                        else False
                    ),
                    "latest_record_validation_issue_summary": (
                        connector_human_gate_validation_issue_summary(latest_validation)
                    ),
                    "reviewer_commands": {
                        "package": row["package_command"],
                        "record_template_output": row["record_template_output_command"],
                        "validate_record": row["record_validation_command"],
                        "validate_record_output": row["record_validation_output_command"],
                    },
                    "remaining_human_evidence_summary": remaining_human_evidence_summary,
                    "required_human_fields": required_human_fields,
                    "required_human_field_count": len(required_human_fields),
                    "required_evidence": required_evidence,
                    "required_evidence_count": len(required_evidence),
                    "release_impact": row["release_impact"],
                    "senior_review_perspective_count": row["senior_review_perspective_count"],
                    "scenario_delivery_unit_plan_ready": row[
                        "scenario_delivery_unit_plan_summary"
                    ]["scenario_delivery_unit_plan_ready"],
                    "redaction_guidance_schema_version": redaction_guidance.get("schema_version"),
                    "redaction_guidance_status": redaction_guidance.get("status"),
                    "allowed_redaction_statuses": proposed_record_template.get("allowed_redaction_statuses", []),
                    "redaction_guidance_sensitive_marker_policy_schema_version": (
                        redaction_sensitive_marker_policy.get("schema_version")
                    ),
                    "redaction_guidance_sensitive_marker_fingerprints_only": (
                        redaction_sensitive_marker_policy.get("fingerprints_only")
                    ),
                    "redaction_guidance_raw_secret_values_allowed": (
                        redaction_guidance.get("raw_secret_values_allowed")
                    ),
                    "redaction_guidance_raw_provider_payloads_allowed": (
                        redaction_guidance.get("raw_provider_payloads_allowed")
                    ),
                    "redaction_guidance_raw_evidence_values_allowed": (
                        redaction_guidance.get("raw_evidence_values_allowed")
                    ),
                    "redaction_guidance_raw_record_body_persisted_by_validator": (
                        redaction_guidance.get("raw_record_body_persisted_by_validator")
                    ),
                    "redaction_guidance_raw_record_path_persisted_by_validator": (
                        redaction_guidance.get("raw_record_path_persisted_by_validator")
                    ),
                    "redaction_guidance_required_field_count": len(redaction_field_guidance),
                    "redaction_guidance_required_evidence_count": (
                        redaction_evidence_manifest.get("required_evidence_count")
                    ),
                    "redaction_guidance_dependency_human_gate_count": len(
                        redaction_dependency_refs.get("required_gates") or []
                    ),
                    "reviewer_checklist_schema_version": reviewer_checklist.get("schema_version"),
                    "reviewer_checklist_status": reviewer_checklist.get("status"),
                    "reviewer_checklist_required_field_item_count": len(
                        checklist_required_field_items
                    ),
                    "reviewer_checklist_senior_review_perspective_count": len(
                        checklist_senior_review_items
                    ),
                    "reviewer_checklist_evidence_packet_manifest_item_count": len(
                        checklist_evidence_manifest_items
                    ),
                    "reviewer_checklist_dependency_human_gate_ref_count": len(
                        checklist_dependency_ref_items
                    ),
                    "reviewer_checklist_reviewer_record_validation_required": (
                        reviewer_checklist.get("reviewer_record_validation_required")
                    ),
                    "reviewer_checklist_record_template_output_command": (
                        reviewer_checklist.get("record_template_output_command")
                    ),
                    "reviewer_checklist_validation_output_command": reviewer_checklist.get(
                        "validation_output_command"
                    ),
                    "reviewer_checklist_evidence_packet_workflow_command_count": (
                        reviewer_checklist.get("evidence_packet_workflow_command_count")
                    ),
                    "reviewer_checklist_evidence_packet_workflow_claim_boundary": (
                        reviewer_checklist.get("evidence_packet_workflow_claim_boundary")
                    ),
                    "reviewer_checklist_product_claim_allowed": (
                        reviewer_checklist.get("product_claim_allowed_by_checklist") is True
                    ),
                    "reviewer_checklist_pass_claim_allowed": (
                        reviewer_checklist.get("pass_claim_allowed_by_checklist") is True
                    ),
                    "local_baseline_review_input_summary": local_baseline_summary,
                    "local_baseline_preflight_bundle": (
                        deepcopy(preflight_bundle) if preflight_bundle else None
                    ),
                    "product_claim_allowed": False,
                    "pass_claim_allowed": False,
                    "approval_collected": False,
                }
            )
        handoff_base = {
            "schema_version": CONNECTOR_HUMAN_GATE_VALIDATION_HANDOFF_SCHEMA,
            "status": "human_validation_handoff_ready",
            "scope": requested_scope,
            "final_verdict": "HUMAN_REQUIRED",
            "weakest_applicable_scenario_result": readiness_report["weakest_applicable_scenario_result"],
            "readiness_report_id": readiness_report["report_id"],
            "readiness_report_ref": f"connector_human_gate_readiness_report:{readiness_report['report_id']}",
            "scenario_count": readiness_report["scenario_count"],
            "human_required_count": readiness_report["human_required_count"],
            "source_requirement_human_pending_ids": source_requirement_human_pending_ids,
            "source_requirement_human_pending_count": len(source_requirement_human_pending_ids),
            "source_requirement_claim_boundary": CONNECTOR_HUMAN_GATE_SOURCE_REQUIREMENT_CLAIM_BOUNDARY,
            "remaining_human_evidence_summary_count": len(handoff_rows),
            "remaining_human_field_total": sum(
                row["remaining_human_evidence_summary"]["required_human_field_count"]
                for row in handoff_rows
            ),
            "remaining_human_evidence_total": sum(
                row["remaining_human_evidence_summary"]["required_evidence_count"]
                for row in handoff_rows
            ),
            "remaining_human_evidence_claim_boundary": CONNECTOR_HUMAN_GATE_REMAINING_EVIDENCE_CLAIM_BOUNDARY,
            "execution_queue_scenario_order": readiness_report["execution_queue_scenario_order"],
            "next_scenario_id": next_row["scenario_id"] if next_row else None,
            "ready_scenario_ids": [row["scenario_id"] for row in ready_rows],
            "blocked_scenario_ids": [row["scenario_id"] for row in blocked_rows],
            "completed_dependency_unlock_record_validation_scenarios": [
                row["scenario_id"] for row in completed_rows
            ],
            "record_validation_count": validation_summary.get("validation_count", 0),
            "structurally_valid_record_validation_count": validation_summary.get("structurally_valid_count", 0),
            "dependency_unlock_allowed_count": validation_summary.get("dependency_unlock_allowed_count", 0),
            "dependency_unlock_denied_count": validation_summary.get("dependency_unlock_denied_count", 0),
            "scenario_validation_handoff_rows": handoff_rows,
            "operator_rule": (
                "Use the next_scenario_id row first, fill only redacted/public-safe reviewer records, "
                "run validate-record with --output, and treat validation refs as dependency-unlock "
                "inputs only when they are structurally valid ACCEPT records. This handoff cannot "
                "collect approval or promote HUMAN_REQUIRED rows."
            ),
            "non_mutation_evidence": {
                "approval_collected_by_handoff": False,
                "human_decisions_recorded_by_handoff": 0,
                "live_provider_calls_executed_by_handoff": 0,
                "provider_mutations_executed_by_handoff": 0,
                "external_mutations_executed_by_handoff": 0,
                "secret_material_read_by_handoff": False,
                "record_bodies_persisted_by_handoff": False,
            },
            "negative_evidence": {
                "human_rows_marked_pass_by_handoff": 0,
                "approvals_collected_by_handoff": 0,
                "product_claims_allowed_by_handoff": 0,
                "pass_claims_allowed_by_handoff": 0,
                "live_provider_calls_executed_by_handoff": 0,
                "provider_mutations_executed_by_handoff": 0,
                "external_mutations_executed_by_handoff": 0,
                "secret_material_read_by_handoff": 0,
                "record_bodies_persisted_by_handoff": 0,
            },
            "product_feature_claims": readiness_report["product_feature_claims"],
            "required_before_pass": readiness_report["required_before_pass"],
            "created_at": utc_now(),
        }
        handoff_base.update(connector_human_gate_completion_boundary())
        handoff_id = f"cshandoff_{json_hash(handoff_base)[:16]}"
        handoff = dict(handoff_base)
        handoff["handoff_id"] = handoff_id
        _write_json(self._human_gate_validation_handoff_path(handoff_id), handoff)
        event = self.store.append_audit(
            "connector.human_gate_validation_handoff.created",
            requested_scope,
            {"type": "connector_human_gate_validation_handoff", "id": handoff_id},
            {
                "scenario_count": handoff["scenario_count"],
                "next_scenario_id": handoff["next_scenario_id"],
                "record_validation_count": handoff["record_validation_count"],
                "dependency_unlock_allowed_count": handoff["dependency_unlock_allowed_count"],
                "final_verdict": handoff["final_verdict"],
                "product_claim_allowed": False,
                "live_provider_calls_executed_by_handoff": 0,
                "provider_mutations_executed_by_handoff": 0,
            },
        )
        return {
            "human_gate_validation_handoff": handoff,
            "human_gate_readiness_report": readiness_report,
            "readiness_audit_event": readiness_result.get("audit_event"),
            "audit_event": event,
        }

    def validate_human_gate_record(
        self,
        requested_scope: dict[str, str],
        scenario_id: str,
        record: dict[str, Any],
        record_file: Path,
        repo_root: Path,
    ) -> dict[str, Any]:
        definition = CONNECTOR_HUMAN_GATE_PACKAGE_DEFINITIONS.get(scenario_id)
        if definition is None:
            return {
                "schema_version": CONNECTOR_HUMAN_GATE_RECORD_VALIDATION_SCHEMA,
                "scenario_id": scenario_id,
                "status": "unsupported",
                "errors": [
                    {
                        "code": "CS_CONNECTOR_HUMAN_GATE_UNSUPPORTED",
                        "message": "Supported ConnectorHub human-gate record validation is CS-CH-H01 through CS-CH-H07.",
                    }
                ],
            }

        record_scenario_id = str(record.get("scenario_id") or "")
        required_human_record = deepcopy(definition["required_human_record"])
        required_fields = list(required_human_record.get("required_fields", []))
        required_evidence = list(required_human_record.get("required_evidence", []))
        required_evidence_packet_manifest = [
            {
                "required_evidence_index": index,
                "required_evidence": str(evidence),
            }
            for index, evidence in enumerate(required_evidence, start=1)
        ]
        required_evidence_packet_manifest_indexes = [
            item["required_evidence_index"]
            for item in required_evidence_packet_manifest
        ]
        required_evidence_by_index = {
            item["required_evidence_index"]: item["required_evidence"]
            for item in required_evidence_packet_manifest
        }
        decision_values = set(required_human_record.get("decision_values", []))
        decision = str(record.get("decision") or "")
        required_perspective_roles = [
            str(perspective["role"])
            for perspective in definition.get("senior_review_perspectives", [])
            if perspective.get("role")
        ]

        def is_empty(value: Any) -> bool:
            return value is None or value == "" or value == [] or value == {}

        missing_required_fields = [field for field in required_fields if field not in record]
        empty_required_fields = [
            field
            for field in required_fields
            if field in record and is_empty(record.get(field))
        ]
        invalid_field_formats = []
        if "review_timestamp" in record and not is_empty(record.get("review_timestamp")):
            if not connector_human_review_timestamp_valid(record.get("review_timestamp")):
                invalid_field_formats.append("review_timestamp")
        field_ref_contract = connector_human_gate_field_ref_contract(scenario_id)
        invalid_required_field_ref_shapes = connector_human_gate_invalid_field_ref_shapes(
            record,
            field_ref_contract,
        )
        perspective_findings = record.get("senior_review_perspective_findings")
        perspective_findings_present = isinstance(perspective_findings, dict)
        if perspective_findings_present:
            provided_perspective_roles = sorted(str(role) for role in perspective_findings)
            missing_senior_review_perspectives = [
                role
                for role in required_perspective_roles
                if role not in perspective_findings
            ]
            empty_senior_review_perspectives = [
                role
                for role in required_perspective_roles
                if role in perspective_findings and is_empty(perspective_findings.get(role))
            ]
            invalid_senior_review_perspective_roles = [
                role
                for role in provided_perspective_roles
                if role not in required_perspective_roles
            ]
        else:
            provided_perspective_roles = []
            missing_senior_review_perspectives = list(required_perspective_roles)
            empty_senior_review_perspectives = []
            invalid_senior_review_perspective_roles = []
        senior_review_perspective_findings_complete = (
            perspective_findings_present
            and not missing_senior_review_perspectives
            and not empty_senior_review_perspectives
            and not invalid_senior_review_perspective_roles
        )
        evidence_packet_manifest = record.get("evidence_packet_manifest")
        evidence_packet_manifest_present = isinstance(evidence_packet_manifest, list)
        provided_evidence_packet_manifest_indexes: list[int] = []
        missing_evidence_packet_manifest_items: list[int] = []
        empty_evidence_packet_manifest_items: list[int] = []
        invalid_evidence_packet_manifest_items: list[str] = []
        duplicate_evidence_packet_manifest_items: list[int] = []
        duplicate_evidence_packet_manifest_ref_fingerprints: list[str] = []
        if evidence_packet_manifest_present:
            seen_manifest_indexes: set[int] = set()
            seen_evidence_refs: dict[str, int] = {}
            for position, item in enumerate(evidence_packet_manifest, start=1):
                if not isinstance(item, dict):
                    invalid_evidence_packet_manifest_items.append(f"entry_{position}_not_object")
                    continue
                index = item.get("required_evidence_index")
                if not isinstance(index, int) or index not in required_evidence_packet_manifest_indexes:
                    invalid_evidence_packet_manifest_items.append(
                        f"entry_{position}_invalid_required_evidence_index"
                    )
                    continue
                provided_evidence_packet_manifest_indexes.append(index)
                if index in seen_manifest_indexes:
                    duplicate_evidence_packet_manifest_items.append(index)
                seen_manifest_indexes.add(index)
                if item.get("required_evidence") != required_evidence_by_index.get(index):
                    invalid_evidence_packet_manifest_items.append(
                        f"entry_{position}_required_evidence_mismatch"
                    )
                if is_empty(item.get("evidence_ref")) or is_empty(item.get("redaction_status")):
                    empty_evidence_packet_manifest_items.append(index)
                elif item.get("redaction_status") not in CONNECTOR_HUMAN_GATE_ALLOWED_REDACTION_STATUSES:
                    invalid_evidence_packet_manifest_items.append(
                        f"entry_{position}_invalid_redaction_status"
                    )
                evidence_ref = item.get("evidence_ref")
                if isinstance(evidence_ref, str) and evidence_ref.strip():
                    normalized_evidence_ref = evidence_ref.strip()
                    if normalized_evidence_ref in seen_evidence_refs:
                        duplicate_evidence_packet_manifest_ref_fingerprints.append(
                            f"sha256:{hashlib.sha256(normalized_evidence_ref.encode('utf-8')).hexdigest()[:16]}"
                        )
                    else:
                        seen_evidence_refs[normalized_evidence_ref] = index
            missing_evidence_packet_manifest_items = [
                index
                for index in required_evidence_packet_manifest_indexes
                if index not in seen_manifest_indexes
            ]
            provided_evidence_packet_manifest_indexes = sorted(set(provided_evidence_packet_manifest_indexes))
            duplicate_evidence_packet_manifest_items = sorted(set(duplicate_evidence_packet_manifest_items))
            duplicate_evidence_packet_manifest_ref_fingerprints = sorted(
                set(duplicate_evidence_packet_manifest_ref_fingerprints)
            )
            empty_evidence_packet_manifest_items = sorted(set(empty_evidence_packet_manifest_items))
        else:
            missing_evidence_packet_manifest_items = list(required_evidence_packet_manifest_indexes)
        evidence_packet_manifest_complete = (
            evidence_packet_manifest_present
            and not missing_evidence_packet_manifest_items
            and not empty_evidence_packet_manifest_items
            and not invalid_evidence_packet_manifest_items
            and not duplicate_evidence_packet_manifest_items
            and not duplicate_evidence_packet_manifest_ref_fingerprints
        )
        execution_queue_item = deepcopy(CONNECTOR_HUMAN_GATE_QUEUE_BY_SCENARIO[scenario_id])
        dependency_refs = record.get("dependency_human_gate_refs")
        dependency_ref_values_by_gate: dict[str, Any] = {}
        if isinstance(dependency_refs, dict):
            dependency_ref_values_by_gate = {
                str(key): value
                for key, value in dependency_refs.items()
                if not is_empty(value)
            }
        elif isinstance(dependency_refs, list):
            dependency_ref_values_by_gate = {
                str(item): item
                for item in dependency_refs
                if not is_empty(item)
            }
        else:
            dependency_ref_values_by_gate = {}
        dependency_ref_ids = sorted(dependency_ref_values_by_gate)
        existing_validations_by_id = {
            str(validation.get("validation_id")): validation
            for validation in self._list_human_gate_record_validations()
            if validation.get("validation_id")
        }
        valid_dependency_human_gate_refs: list[str] = []
        invalid_dependency_human_gate_refs: list[str] = []
        dependency_human_gate_validation_refs: dict[str, str] = {}
        for dependency in execution_queue_item["depends_on"]:
            if dependency not in dependency_ref_values_by_gate:
                continue
            validation_ref = connector_human_gate_validation_ref_id(dependency_ref_values_by_gate[dependency])
            if validation_ref:
                dependency_human_gate_validation_refs[dependency] = validation_ref
            validation = existing_validations_by_id.get(validation_ref)
            if (
                validation_ref
                and validation
                and validation.get("scenario_id") == dependency
                and connector_human_gate_validation_allows_dependency_unlock(validation)
            ):
                valid_dependency_human_gate_refs.append(dependency)
            else:
                invalid_dependency_human_gate_refs.append(dependency)
        missing_dependency_human_gates = [
            dependency
            for dependency in execution_queue_item["depends_on"]
            if dependency not in dependency_ref_ids
        ]
        _, sensitive_findings, sensitive_fields = _redact_sensitive_payload(record)
        redaction_guidance = connector_human_gate_redaction_guidance(
            scenario_id,
            definition,
            execution_queue_item,
        )
        scenario_matches = record_scenario_id == scenario_id
        decision_allowed = decision in decision_values
        structural_errors = []
        if not scenario_matches:
            structural_errors.append("scenario_id_mismatch")
        if not decision_allowed:
            structural_errors.append("decision_not_allowed")
        if missing_required_fields:
            structural_errors.append("missing_required_fields")
        if empty_required_fields:
            structural_errors.append("empty_required_fields")
        if invalid_field_formats:
            structural_errors.append("invalid_field_formats")
        if invalid_required_field_ref_shapes:
            structural_errors.append("invalid_required_field_ref_shapes")
        if not perspective_findings_present:
            structural_errors.append("missing_senior_review_perspective_findings")
        if missing_senior_review_perspectives:
            structural_errors.append("missing_senior_review_perspectives")
        if empty_senior_review_perspectives:
            structural_errors.append("empty_senior_review_perspectives")
        if invalid_senior_review_perspective_roles:
            structural_errors.append("invalid_senior_review_perspective_roles")
        if not evidence_packet_manifest_present:
            structural_errors.append("missing_evidence_packet_manifest")
        if missing_evidence_packet_manifest_items:
            structural_errors.append("missing_evidence_packet_manifest_items")
        if empty_evidence_packet_manifest_items:
            structural_errors.append("empty_evidence_packet_manifest_items")
        if invalid_evidence_packet_manifest_items:
            structural_errors.append("invalid_evidence_packet_manifest_items")
        if duplicate_evidence_packet_manifest_items:
            structural_errors.append("duplicate_evidence_packet_manifest_items")
        if duplicate_evidence_packet_manifest_ref_fingerprints:
            structural_errors.append("duplicate_evidence_packet_manifest_refs")
        if missing_dependency_human_gates:
            structural_errors.append("missing_dependency_human_gate_refs")
        if invalid_dependency_human_gate_refs:
            structural_errors.append("invalid_dependency_human_gate_refs")
        if sensitive_findings:
            structural_errors.append("sensitive_marker_detected")

        validation_status = "record_structurally_valid" if not structural_errors else "record_structurally_invalid"
        dependency_unlock_allowed_by_validator = validation_status == "record_structurally_valid" and decision == "ACCEPT"
        if dependency_unlock_allowed_by_validator:
            dependency_unlock_blocked_reason = None
        elif validation_status != "record_structurally_valid":
            dependency_unlock_blocked_reason = "structural_errors"
        else:
            dependency_unlock_blocked_reason = "decision_not_accept"
        record_file_ref = {
            "sha256": hashlib.sha256(record_file.read_bytes()).hexdigest() if record_file.exists() else None,
            "path_sha256": hashlib.sha256(str(record_file).encode("utf-8")).hexdigest(),
            "path_recorded_by_validator": False,
        }
        validation_base = {
            "schema_version": CONNECTOR_HUMAN_GATE_RECORD_VALIDATION_SCHEMA,
            "scenario_id": scenario_id,
            "record_scenario_id": record_scenario_id,
            "title": definition["title"],
            "status": validation_status,
            "scope": requested_scope,
            "validation_scope": "structure_and_safety_only",
            "record_file": record_file_ref,
            "provided_fields": sorted(str(key) for key in record.keys()),
            "review_order": execution_queue_item["order"],
            "depends_on_human_gates": execution_queue_item["depends_on"],
            "dependency_human_gate_refs_present": dependency_ref_ids,
            "dependency_human_gate_validation_refs": dependency_human_gate_validation_refs,
            "valid_dependency_human_gate_refs": valid_dependency_human_gate_refs,
            "invalid_dependency_human_gate_refs": invalid_dependency_human_gate_refs,
            "missing_dependency_human_gates": missing_dependency_human_gates,
            "gate_category": execution_queue_item["gate_category"],
            "stop_or_reject_when": execution_queue_item["stop_or_reject_when"],
            "decision_present": not is_empty(record.get("decision")),
            "decision_allowed": decision_allowed,
            "decision_recorded_by_validator": False,
            "dependency_unlock_allowed_by_validator": dependency_unlock_allowed_by_validator,
            "dependency_unlock_blocked_reason": dependency_unlock_blocked_reason,
            "allowed_decision_values": sorted(decision_values),
            "scenario_matches": scenario_matches,
            "required_fields": required_fields,
            "missing_required_fields": missing_required_fields,
            "empty_required_fields": empty_required_fields,
            "invalid_field_formats": invalid_field_formats,
            "field_ref_contract": field_ref_contract,
            "field_ref_contract_present": field_ref_contract is not None,
            "invalid_required_field_ref_shapes": invalid_required_field_ref_shapes,
            "field_ref_values_recorded_by_validator": False,
            "required_senior_review_perspectives": required_perspective_roles,
            "provided_senior_review_perspective_roles": provided_perspective_roles,
            "senior_review_perspective_findings_present": perspective_findings_present,
            "senior_review_perspective_findings_complete": senior_review_perspective_findings_complete,
            "missing_senior_review_perspectives": missing_senior_review_perspectives,
            "empty_senior_review_perspectives": empty_senior_review_perspectives,
            "invalid_senior_review_perspective_roles": invalid_senior_review_perspective_roles,
            "senior_review_perspective_findings_recorded_by_validator": False,
            "required_evidence_packet_manifest": required_evidence_packet_manifest,
            "required_evidence_packet_manifest_indexes": required_evidence_packet_manifest_indexes,
            "allowed_redaction_statuses": list(CONNECTOR_HUMAN_GATE_ALLOWED_REDACTION_STATUSES),
            "provided_evidence_packet_manifest_indexes": provided_evidence_packet_manifest_indexes,
            "evidence_packet_manifest_present": evidence_packet_manifest_present,
            "evidence_packet_manifest_complete": evidence_packet_manifest_complete,
            "missing_evidence_packet_manifest_items": missing_evidence_packet_manifest_items,
            "empty_evidence_packet_manifest_items": empty_evidence_packet_manifest_items,
            "invalid_evidence_packet_manifest_items": invalid_evidence_packet_manifest_items,
            "duplicate_evidence_packet_manifest_items": duplicate_evidence_packet_manifest_items,
            "duplicate_evidence_packet_manifest_ref_fingerprints": duplicate_evidence_packet_manifest_ref_fingerprints,
            "evidence_packet_manifest_recorded_by_validator": False,
            "format_rules": {
                "review_timestamp": "ISO-8601 timestamp with timezone, for example 2026-06-24T12:00:00Z",
            },
            "required_evidence": required_evidence,
            "sensitive_marker_findings": sensitive_findings,
            "sensitive_fields": sensitive_fields,
            "redaction_guidance": redaction_guidance,
            "structural_errors": structural_errors,
            "template_path": str(connector_human_template_path(scenario_id)),
            "template_structure": connector_human_template_structure(repo_root / connector_human_template_path(scenario_id), scenario_id),
            "product_claim_allowed": False,
            "pass_claim_allowed_by_validator": False,
            "matrix_status_after_validation": "HUMAN_REQUIRED",
            "weakest_applicable_scenario_result": "HUMAN_REQUIRED",
            "promotion_rule": "This validator checks record structure and safety only. A human owner must attach dated external evidence and explicitly promote the scenario before any H row can move toward PASS.",
            "non_mutation_evidence": {
                "approval_collected_by_validator": False,
                "human_decision_recorded_by_validator": False,
                "live_provider_calls_executed_by_validator": 0,
                "provider_mutations_executed_by_validator": 0,
                "external_mutations_executed_by_validator": 0,
                "record_body_persisted_by_validator": False,
                "record_path_persisted_by_validator": False,
                "field_ref_values_persisted_by_validator": False,
                "senior_review_perspective_findings_persisted_by_validator": False,
                "evidence_packet_manifest_values_persisted_by_validator": False,
            },
            "negative_evidence": {
                "human_rows_marked_pass_by_validator": 0,
                "product_claims_allowed_by_validator": 0,
                "pass_without_owner_promotion_allowed_by_validator": 0,
                "live_provider_calls_executed_by_validator": 0,
                "provider_mutations_executed_by_validator": 0,
                "external_mutations_executed_by_validator": 0,
                "record_body_persisted_by_validator": 0,
                "record_path_persisted_by_validator": 0,
                "human_decision_value_persisted_by_validator": 0,
                "senior_review_perspective_findings_persisted_by_validator": 0,
                "evidence_packet_manifest_values_persisted_by_validator": 0,
                "sensitive_marker_findings": len(sensitive_findings),
                "missing_required_fields": len(missing_required_fields),
                "empty_required_fields": len(empty_required_fields),
                "invalid_field_formats": len(invalid_field_formats),
                "invalid_required_field_ref_shapes": len(invalid_required_field_ref_shapes),
                "field_ref_values_persisted_by_validator": 0,
                "missing_senior_review_perspectives": len(missing_senior_review_perspectives),
                "empty_senior_review_perspectives": len(empty_senior_review_perspectives),
                "invalid_senior_review_perspective_roles": len(invalid_senior_review_perspective_roles),
                "missing_evidence_packet_manifest_items": len(missing_evidence_packet_manifest_items),
                "empty_evidence_packet_manifest_items": len(empty_evidence_packet_manifest_items),
                "invalid_evidence_packet_manifest_items": len(invalid_evidence_packet_manifest_items),
                "duplicate_evidence_packet_manifest_items": len(duplicate_evidence_packet_manifest_items),
                "duplicate_evidence_packet_manifest_refs": len(duplicate_evidence_packet_manifest_ref_fingerprints),
                "missing_dependency_human_gates": len(missing_dependency_human_gates),
                "invalid_dependency_human_gate_refs": len(invalid_dependency_human_gate_refs),
            },
            "created_at": utc_now(),
        }
        validation_base.update(connector_human_gate_completion_boundary())
        validation_id = f"cshval_{json_hash(validation_base)[:16]}"
        validation = dict(validation_base)
        validation["validation_id"] = validation_id
        _write_json(self._human_gate_record_validation_path(validation_id), validation)
        event = self.store.append_audit(
            "connector.human_gate_record.validated",
            requested_scope,
            {"type": "connector_human_gate_record_validation", "id": validation_id},
            {
                "scenario_id": scenario_id,
                "status": validation_status,
                "validation_scope": validation["validation_scope"],
                "review_order": validation["review_order"],
                "depends_on_human_gates": validation["depends_on_human_gates"],
                "missing_required_fields": missing_required_fields,
                "invalid_required_field_ref_shapes": invalid_required_field_ref_shapes,
                "missing_dependency_human_gates": missing_dependency_human_gates,
                "invalid_dependency_human_gate_refs": invalid_dependency_human_gate_refs,
                "senior_review_perspective_findings_complete": senior_review_perspective_findings_complete,
                "missing_senior_review_perspectives": missing_senior_review_perspectives,
                "empty_senior_review_perspectives": empty_senior_review_perspectives,
                "invalid_senior_review_perspective_roles": invalid_senior_review_perspective_roles,
                "evidence_packet_manifest_complete": evidence_packet_manifest_complete,
                "missing_evidence_packet_manifest_items": len(missing_evidence_packet_manifest_items),
                "empty_evidence_packet_manifest_items": len(empty_evidence_packet_manifest_items),
                "invalid_evidence_packet_manifest_items": len(invalid_evidence_packet_manifest_items),
                "duplicate_evidence_packet_manifest_items": len(duplicate_evidence_packet_manifest_items),
                "duplicate_evidence_packet_manifest_refs": len(duplicate_evidence_packet_manifest_ref_fingerprints),
                "sensitive_marker_findings": len(sensitive_findings),
                "dependency_unlock_allowed_by_validator": dependency_unlock_allowed_by_validator,
                "dependency_unlock_blocked_reason": dependency_unlock_blocked_reason,
                "product_claim_allowed": False,
                "pass_claim_allowed_by_validator": False,
            },
        )
        return {"human_gate_record_validation": validation, "audit_event": event}

    def register_contract(
        self,
        contract: dict[str, Any],
        requested_scope: dict[str, str],
        source_path: Path,
    ) -> dict[str, Any]:
        issues = validate_connector_contract(contract, requested_scope)
        if issues:
            return {"status": "failed", "issues": issues}

        contract_version_id = _contract_version_id(contract)
        record = {
            "schema_version": CONTRACT_RECORD_SCHEMA,
            "contract_id": contract["contract_id"],
            "contract_version_id": contract_version_id,
            "contract_version": contract["contract_version"],
            "status": "draft_validated",
            "scope": requested_scope,
            "purpose": contract["purpose"],
            "needs": contract["needs"],
            "source_policy_request": contract["source_policy_request"],
            "delivery": contract["delivery"],
            "actions": contract["actions"],
            "connector_port": {
                "name": "CornerStone ConnectorPort",
                "adapter": "local_fixture_connectorhub_adapter",
                "product_depends_on_provider_sdk": False,
                "transport_replaceable": True,
            },
            "source": {
                "path": str(source_path),
                "sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            },
            "created_at": utc_now(),
        }
        _write_json(self._contract_path(contract_version_id), record)
        _write_json(
            self._latest_path(record["contract_id"]),
            {"contract_id": record["contract_id"], "contract_version_id": contract_version_id},
        )
        audit_event = self.store.append_audit(
            "connector.contract.validated",
            requested_scope,
            {"type": "connector_contract", "id": contract_version_id},
            {
                "contract_id": record["contract_id"],
                "schema_version": record["schema_version"],
                "provider_internals_exposed": False,
            },
        )
        return {"status": "success", "contract": record, "audit_event": audit_event}

    def load_contract(self, contract_id: str, contract_version_id: str | None = None) -> dict[str, Any] | None:
        selected_version = contract_version_id
        if selected_version is None:
            latest_path = self._latest_path(contract_id)
            if not latest_path.exists():
                return None
            selected_version = json.loads(latest_path.read_text()).get("contract_version_id")
        if not selected_version:
            return None
        path = self._contract_path(selected_version)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _latest_setup_result(
        self,
        contract: dict[str, Any],
        requested_scope: dict[str, str],
        provider_pack_id: str,
    ) -> dict[str, Any] | None:
        if not self.setup_result_dir.exists():
            return None
        candidates: list[dict[str, Any]] = []
        for path in self.setup_result_dir.glob("*.json"):
            setup = json.loads(path.read_text())
            if setup.get("contract_id") != contract.get("contract_id"):
                continue
            if setup.get("contract_version_id") != contract.get("contract_version_id"):
                continue
            if setup.get("scope") != requested_scope:
                continue
            policy = setup.get("source_policy_snapshot", {})
            if policy.get("selected_provider_pack_id") != provider_pack_id:
                continue
            candidates.append(setup)
        if not candidates:
            return None
        candidates.sort(key=lambda item: item.get("created_at", ""))
        latest = dict(candidates[-1])
        latest_policy = self._latest_source_policy(contract, requested_scope, provider_pack_id)
        if latest_policy is not None:
            latest["source_policy_snapshot"] = latest_policy
        return latest

    def _latest_source_policy(
        self,
        contract: dict[str, Any],
        requested_scope: dict[str, str],
        provider_pack_id: str,
    ) -> dict[str, Any] | None:
        if not self.source_policy_dir.exists():
            return None
        candidates: list[dict[str, Any]] = []
        for path in self.source_policy_dir.glob("*.json"):
            source_policy = json.loads(path.read_text())
            if source_policy.get("contract_version_id") != contract.get("contract_version_id"):
                continue
            if source_policy.get("scope") != requested_scope:
                continue
            if source_policy.get("selected_provider_pack_id") not in {None, provider_pack_id}:
                continue
            if provider_pack_id not in source_policy.get("provider_pack_ids", [provider_pack_id]):
                continue
            candidates.append(source_policy)
        if not candidates:
            return None
        candidates.sort(
            key=lambda item: item.get("confirmation", {}).get("confirmed_at")
            or item.get("created_at")
            or ""
        )
        return candidates[-1]

    def _evaluate_projection_source_policy(
        self,
        delivery: dict[str, Any],
        contract: dict[str, Any],
        setup_result: dict[str, Any],
        requested_scope: dict[str, str],
    ) -> tuple[dict[str, Any], list[dict[str, str]]]:
        source_policy = setup_result.get("source_policy_snapshot", {})
        payload = delivery.get("payload") if isinstance(delivery.get("payload"), dict) else {}
        source_summary = delivery.get("source_summary") if isinstance(delivery.get("source_summary"), dict) else {}
        evidence_ref = delivery.get("evidence_ref") if isinstance(delivery.get("evidence_ref"), dict) else {}
        permitted = evidence_ref.get("permitted_fields") if isinstance(evidence_ref.get("permitted_fields"), list) else []
        permitted_fields = sorted({str(field) for field in permitted if isinstance(field, str) and field})
        system_fields = {"updated_at", "source_revision", "revision"}
        allowed_fields = sorted(set(permitted_fields) | system_fields)
        payload_fields = sorted(str(field) for field in payload.keys())
        forbidden_body_fields = sorted(
            {
                field
                for field in payload_fields
                if field
                in {
                    "body",
                    "body_markdown",
                    "content",
                    "diff",
                    "full_body",
                    "patch",
                    "raw_body",
                    "raw_content",
                }
            }
        )
        fields_outside_contract = sorted(field for field in payload_fields if field not in set(allowed_fields))
        excluded_fields = sorted(set(fields_outside_contract) | set(forbidden_body_fields))
        normalized_payload = {field: payload[field] for field in payload_fields if field in set(allowed_fields)}
        content_size_bytes = _policy_content_size_bytes(payload)
        declared_size_bytes = _declared_content_size_bytes(payload)
        max_content_bytes = source_policy.get("max_content_bytes")
        selected_resources = [
            str(resource)
            for resource in source_policy.get("selected_resources", [])
            if isinstance(resource, str)
        ]
        allowed_paths = [str(path) for path in source_policy.get("allowed_paths", []) if isinstance(path, str)]
        selected_resource_allowed = _source_ref_allowed(source_summary, selected_resources)
        path_allowed = _delivery_path_allowed(payload, source_summary, allowed_paths)
        raw_access_denied = source_policy.get("raw_access") == "denied"
        issues: list[dict[str, str]] = []
        if excluded_fields:
            issues.append(
                {
                    "code": "CS_CONNECTOR_SOURCE_POLICY_FIELD_FORBIDDEN",
                    "message": "Projection contains fields outside the active Source Policy.",
                    "path": "payload",
                }
            )
        if isinstance(max_content_bytes, int) and content_size_bytes > max_content_bytes:
            issues.append(
                {
                    "code": "CS_CONNECTOR_SOURCE_POLICY_CONTENT_TOO_LARGE",
                    "message": "Projection content exceeds the active Source Policy byte limit.",
                    "path": "payload",
                }
            )
        if not selected_resource_allowed:
            issues.append(
                {
                    "code": "CS_CONNECTOR_SOURCE_POLICY_RESOURCE_DENIED",
                    "message": "Projection source is outside selected resources.",
                    "path": "source_summary.source_ref",
                }
            )
        if not path_allowed:
            issues.append(
                {
                    "code": "CS_CONNECTOR_SOURCE_POLICY_PATH_DENIED",
                    "message": "Projection path is outside allowed paths.",
                    "path": "payload.path",
                }
            )
        if not raw_access_denied:
            issues.append(
                {
                    "code": "CS_CONNECTOR_SOURCE_POLICY_RAW_ACCESS_DENIED",
                    "message": "Projection ingest requires raw access to remain denied.",
                    "path": "source_policy_snapshot.raw_access",
                }
            )
        content_decision = _build_content_restriction_decision(
            delivery=delivery,
            requested_scope=requested_scope,
            contract=contract,
            setup_result=setup_result,
            source_policy=source_policy,
            normalized_payload=normalized_payload,
            path_allowed=path_allowed,
            content_size_bytes=content_size_bytes,
            declared_size_bytes=declared_size_bytes,
        )
        if content_decision["action"] == "skip" and not any(
            issue["code"] == "CS_CONNECTOR_SOURCE_POLICY_PATH_DENIED" for issue in issues
        ):
            issues.append(
                {
                    "code": content_decision["reason_codes"][0],
                    "message": "Projection content is skipped by repository content restrictions before Product state.",
                    "path": "payload.path",
                }
            )
        if content_decision["action"] == "quarantine":
            issues.append(
                {
                    "code": content_decision["reason_codes"][0],
                    "message": "Projection content contains sensitive material that must be quarantined before Product state.",
                    "path": "payload",
                }
            )

        decision_id = _projection_policy_decision_id(requested_scope, contract["contract_version_id"], delivery)
        decision_status = "deny" if issues else "allow"
        safe_normalized_payload = content_decision["normalized_payload"] if not issues else {}
        enforcement_action = content_decision["action"] if content_decision["action"] in {"redact", "metadata_only"} else "normalize"
        if issues:
            enforcement_action = "quarantine" if content_decision["action"] == "quarantine" else "block"
        decision = {
            "schema_version": PROJECTION_POLICY_DECISION_SCHEMA,
            "policy_decision_id": decision_id,
            "decision": decision_status,
            "enforcement_action": enforcement_action,
            "scope": requested_scope,
            "contract_id": contract["contract_id"],
            "contract_version_id": contract["contract_version_id"],
            "setup_result_id": setup_result.get("setup_result_id"),
            "source_policy_id": source_policy.get("source_policy_id"),
            "delivery_id": delivery.get("delivery_id"),
            "projection_id": delivery.get("projection_id"),
            "provider_event_id": delivery.get("provider_event_id"),
            "projection_type": delivery.get("projection_type"),
            "source_external_id": _source_external_id(delivery),
            "source_revision": _source_revision(delivery),
            "content_mode": source_policy.get("content_mode", "metadata_only"),
            "selected_resources": selected_resources,
            "selected_resource_allowed": selected_resource_allowed,
            "allowed_paths": allowed_paths,
            "path_allowed": path_allowed,
            "max_content_bytes": max_content_bytes,
            "content_size_bytes": content_size_bytes,
            "declared_size_bytes": declared_size_bytes,
            "permitted_fields": permitted_fields,
            "included_fields": sorted(safe_normalized_payload.keys()),
            "excluded_fields": excluded_fields,
            "forbidden_body_fields": forbidden_body_fields,
            "normalized_projection": {
                "schema_version": "cs.connector_normalized_projection.v1",
                "projection_id": delivery.get("projection_id"),
                "projection_type": delivery.get("projection_type"),
                "source_summary": source_summary,
                "payload": safe_normalized_payload,
            },
            "body_restriction": {
                "full_body_allowed": False,
                "preview_field": "body_markdown_excerpt" if "body_markdown_excerpt" in permitted_fields else None,
                "full_body_fields_present": forbidden_body_fields,
            },
            "restriction_summary": _restriction_summary(source_policy),
            "raw_access_allowed": False,
            "raw_content_persisted": False,
            "raw_provider_payload_persisted": False,
            "content_restriction_decision_id": content_decision["content_restriction_decision_id"],
            "content_restriction_decision": content_decision,
            "product_state_safe_to_use": not issues,
            "helpful_error": None
            if not issues
            else "Projection violates Source Policy; remove forbidden fields, select an allowed source/path, or reduce content before delivery.",
            "issue_codes": [issue["code"] for issue in issues],
            "created_at": utc_now(),
        }
        return decision, issues

    def _delivery_validation_issues(
        self,
        delivery: dict[str, Any],
        contract: dict[str, Any],
        setup_result: dict[str, Any],
        requested_scope: dict[str, str],
    ) -> list[dict[str, str]]:
        issues: list[dict[str, str]] = []
        if delivery.get("schema_version") != SUPPORTED_DELIVERY_SCHEMA:
            issues.append(
                {
                    "code": "CS_CONNECTOR_DELIVERY_SCHEMA_UNSUPPORTED",
                    "message": f"Delivery schema must be {SUPPORTED_DELIVERY_SCHEMA}.",
                    "path": "schema_version",
                }
            )
        for field in ["delivery_id", "projection_id", "provider_pack_id", "provider_event_id", "common_capability", "projection_type"]:
            if not delivery.get(field):
                issues.append(
                    {
                        "code": "CS_CONNECTOR_DELIVERY_FIELD_MISSING",
                        "message": f"{field} is required.",
                        "path": field,
                    }
                )
        if delivery.get("contract_id") != contract.get("contract_id"):
            issues.append(
                {
                    "code": "CS_CONNECTOR_DELIVERY_CONTRACT_MISMATCH",
                    "message": "Delivery contract_id must match the selected connector contract.",
                    "path": "contract_id",
                }
            )
        scope = delivery.get("scope")
        if not scope_complete(scope):
            issues.append(
                {
                    "code": "CS_CONNECTOR_DELIVERY_SCOPE_INCOMPLETE",
                    "message": "Delivery tenant/owner/namespace/workspace scope is required.",
                    "path": "scope",
                }
            )
        elif scope != requested_scope or scope != contract.get("scope"):
            issues.append(
                {
                    "code": "CS_CONNECTOR_DELIVERY_SCOPE_MISMATCH",
                    "message": "Delivery scope must match the trusted CLI scope and contract scope.",
                    "path": "scope",
                }
            )
        if setup_result.get("activation_allowed") is not True:
            issues.append(
                {
                    "code": "CS_CONNECTOR_SETUP_NOT_ACTIVE",
                    "message": "Delivery cannot be ingested until setup planning allows activation.",
                    "path": "setup_result.activation_allowed",
                }
            )
        source_policy = setup_result.get("source_policy_snapshot", {})
        if source_policy.get("raw_access") != "denied":
            issues.append(
                {
                    "code": "CS_CONNECTOR_DELIVERY_RAW_ACCESS_NOT_DENIED",
                    "message": "Local Projection Delivery ingest requires raw provider access to remain denied.",
                    "path": "source_policy_snapshot.raw_access",
                }
            )
        mappings = setup_result.get("mappings", [])
        matching_mapping = next(
            (
                mapping
                for mapping in mappings
                if mapping.get("common_capability") == delivery.get("common_capability")
                and mapping.get("provider_pack_id") == delivery.get("provider_pack_id")
            ),
            None,
        )
        if not matching_mapping:
            issues.append(
                {
                    "code": "CS_CONNECTOR_DELIVERY_CAPABILITY_UNMAPPED",
                    "message": "Delivery capability is not mapped by the active Setup Result.",
                    "path": "common_capability",
                }
            )
        elif delivery.get("projection_type") not in matching_mapping.get("projection_types", []):
            issues.append(
                {
                    "code": "CS_CONNECTOR_DELIVERY_PROJECTION_UNSUPPORTED",
                    "message": "Delivery projection type is not accepted by the active mapping.",
                    "path": "projection_type",
                }
            )
        if "raw_provider_payload" in delivery:
            issues.append(
                {
                    "code": "CS_CONNECTOR_DELIVERY_RAW_PROVIDER_PAYLOAD_FORBIDDEN",
                    "message": "Raw provider payloads must not enter Product delivery fixtures.",
                    "path": "raw_provider_payload",
                }
            )
        if not isinstance(delivery.get("source_summary"), dict):
            issues.append(
                {
                    "code": "CS_CONNECTOR_DELIVERY_SOURCE_SUMMARY_MISSING",
                    "message": "source_summary metadata is required.",
                    "path": "source_summary",
                }
            )
        if not isinstance(delivery.get("evidence_ref"), dict):
            issues.append(
                {
                    "code": "CS_CONNECTOR_DELIVERY_EVIDENCE_REF_MISSING",
                    "message": "evidence_ref metadata is required.",
                    "path": "evidence_ref",
                }
            )
        else:
            evidence_policy_id = delivery["evidence_ref"].get("source_policy_id")
            if evidence_policy_id and evidence_policy_id != source_policy.get("source_policy_id"):
                issues.append(
                    {
                        "code": "CS_CONNECTOR_DELIVERY_SOURCE_POLICY_MISMATCH",
                        "message": "Delivery EvidenceRef source policy must match the active Setup Result Source Policy.",
                        "path": "evidence_ref.source_policy_id",
                    }
                )
        for provider_path in provider_internal_findings(delivery):
            issues.append(
                {
                    "code": "CS_CONNECTOR_DELIVERY_PROVIDER_INTERNAL_EXPOSED",
                    "message": "Delivery exposes provider internal or credential-shaped data.",
                    "path": provider_path,
                }
            )
        return issues

    def _write_content_restricted_delivery(self, delivery: dict[str, Any], content_decision: dict[str, Any]) -> Path:
        sanitized_delivery = json.loads(json.dumps(delivery))
        sanitized_delivery["payload"] = content_decision.get("normalized_payload", {})
        sanitized_delivery["content_restriction"] = {
            "schema_version": CONTENT_RESTRICTION_DECISION_SCHEMA,
            "content_restriction_decision_id": content_decision["content_restriction_decision_id"],
            "action": content_decision["action"],
            "durable_content_status": content_decision["durable_content_status"],
            "partial_status": content_decision["partial_status"],
            "reason_codes": content_decision["reason_codes"],
            "sensitive_marker_scan": content_decision["sensitive_marker_scan"],
            "raw_restricted_content_persisted": False,
            "raw_provider_payload_persisted": False,
        }
        path = self.content_restricted_delivery_dir / f"{content_decision['content_restriction_decision_id']}.delivery.json"
        _write_json(path, sanitized_delivery)
        return path

    def _record_content_quarantine(
        self,
        *,
        delivery: dict[str, Any],
        requested_scope: dict[str, str],
        contract: dict[str, Any],
        setup_result: dict[str, Any],
        policy_decision: dict[str, Any],
        issues: list[dict[str, Any]],
        prior_audit_events: list[dict[str, Any]],
    ) -> dict[str, Any]:
        content_decision = policy_decision.get("content_restriction_decision", {})
        source_policy = setup_result.get("source_policy_snapshot", {})
        retry_state_id = _delivery_retry_state_id(requested_scope, contract["contract_version_id"], delivery)
        quarantine_id = _delivery_quarantine_id(retry_state_id)
        now = utc_now()
        issue_codes = [str(issue.get("code", "CS_CONNECTOR_CONTENT_QUARANTINED")) for issue in issues]
        reason_code = issue_codes[0] if issue_codes else "CS_CONNECTOR_CONTENT_QUARANTINED"
        retry_state = {
            "schema_version": DELIVERY_RETRY_STATE_SCHEMA,
            "retry_state_id": retry_state_id,
            "status": "quarantined",
            "delivery_id": delivery.get("delivery_id"),
            "projection_id": delivery.get("projection_id"),
            "scope": requested_scope,
            "contract_id": contract["contract_id"],
            "contract_version_id": contract["contract_version_id"],
            "setup_result_id": setup_result.get("setup_result_id"),
            "source_policy_id": source_policy.get("source_policy_id"),
            "provider_pack_id": delivery.get("provider_pack_id"),
            "provider_event_id": delivery.get("provider_event_id"),
            "common_capability": delivery.get("common_capability"),
            "projection_type": delivery.get("projection_type"),
            "delivery_ref": _safe_delivery_ref(delivery),
            "failure_class": "content_restriction",
            "attempt_count": 1,
            "last_reason_code": reason_code,
            "attempts": [
                {
                    "attempt_no": 1,
                    "failure_class": "content_restriction",
                    "reason_code": reason_code,
                    "issue_codes": issue_codes,
                    "retryable": False,
                    "delay_seconds": 0,
                    "next_retry_at": None,
                    "raw_provider_payload_persisted": False,
                    "at": now,
                }
            ],
            "retry_schedule": [],
            "next_retry_at": None,
            "unrelated_streams_blocked": False,
            "raw_provider_payload_persisted": False,
            "content_restriction_decision_id": content_decision.get("content_restriction_decision_id"),
            "quarantine_id": quarantine_id,
            "created_at": now,
            "updated_at": now,
        }
        quarantine = {
            "schema_version": DELIVERY_QUARANTINE_SCHEMA,
            "quarantine_id": quarantine_id,
            "retry_state_id": retry_state_id,
            "status": "quarantined",
            "failure_class": "content_restriction",
            "reason_code": reason_code,
            "issue_codes": issue_codes,
            "attempt_count": 1,
            "quarantine_after_attempts": 1,
            "retry_policy": "not_retryable_sensitive_content",
            "delivery_id": delivery.get("delivery_id"),
            "projection_id": delivery.get("projection_id"),
            "scope": requested_scope,
            "contract_id": contract["contract_id"],
            "contract_version_id": contract["contract_version_id"],
            "setup_result_id": setup_result.get("setup_result_id"),
            "source_policy_id": source_policy.get("source_policy_id"),
            "provider_pack_id": delivery.get("provider_pack_id"),
            "provider_event_id": delivery.get("provider_event_id"),
            "common_capability": delivery.get("common_capability"),
            "projection_type": delivery.get("projection_type"),
            "delivery_ref": _safe_delivery_ref(delivery),
            "content_restriction_decision_id": content_decision.get("content_restriction_decision_id"),
            "redacted_error": {
                "reason_code": reason_code,
                "issue_codes": issue_codes,
                "message": "Repository content was quarantined before Product state because sensitive material was detected.",
                "raw_provider_payload_included": False,
                "secret_values_included": False,
            },
            "source_health_impact": {
                "affected_source": delivery.get("source_summary", {}).get("source_ref")
                if isinstance(delivery.get("source_summary"), dict)
                else None,
                "affected_capability": delivery.get("common_capability"),
                "freshness_state": "degraded_for_affected_content",
                "unrelated_streams_continue": True,
                "unrelated_streams_blocked": False,
            },
            "safe_diagnostics": {
                "raw_provider_payload_persisted": False,
                "raw_provider_payload_in_operator_output": False,
                "provider_credentials_exposed": False,
                "sensitive_marker_scan": content_decision.get("sensitive_marker_scan", {}),
                "resolution": "Remove or rotate sensitive material in the source, then replay a sanitized projection.",
            },
            "failure_evidence_preserved": True,
            "created_at": now,
            "updated_at": now,
            "replay_attempts": [],
        }
        _write_json(self._delivery_retry_path(retry_state_id), retry_state)
        _write_json(self._quarantine_path(quarantine_id), quarantine)
        audit_event = self.store.append_audit(
            "connector.content_restriction.quarantined",
            requested_scope,
            {"type": "connector_delivery_quarantine", "id": quarantine_id},
            {
                "delivery_id": delivery.get("delivery_id"),
                "projection_id": delivery.get("projection_id"),
                "content_restriction_decision_id": content_decision.get("content_restriction_decision_id"),
                "reason_code": reason_code,
                "raw_provider_payload_persisted": False,
                "unrelated_streams_blocked": False,
            },
        )
        audit_refs = list(
            dict.fromkeys([*policy_decision.get("audit_refs", []), f"audit:{audit_event['event_id']}"])
        )
        _merge_content_decision_refs(
            content_decision,
            evidence_refs=[
                f"connector_delivery_retry_state:{retry_state_id}",
                f"connector_delivery_quarantine:{quarantine_id}",
            ],
            audit_refs=audit_refs,
            linked_records={
                "delivery_retry_state_id": retry_state_id,
                "quarantine_id": quarantine_id,
            },
        )
        policy_decision["content_restriction_decision"] = content_decision
        policy_decision["audit_refs"] = audit_refs
        _write_json(self._projection_policy_decision_path(policy_decision["policy_decision_id"]), policy_decision)
        _write_json(
            self._content_restriction_decision_path(content_decision["content_restriction_decision_id"]),
            content_decision,
        )
        return {
            "status": "quarantined",
            "contract": contract,
            "setup_result": setup_result,
            "source_policy": source_policy,
            "connector_projection_policy_decision": policy_decision,
            "connector_content_restriction_decision": content_decision,
            "delivery_retry_state": retry_state,
            "delivery_quarantine": quarantine,
            "audit_event": audit_event,
            "audit_events": [*prior_audit_events, audit_event],
            "issues": issues,
        }

    def _record_untrusted_content_review(
        self,
        *,
        delivery: dict[str, Any],
        requested_scope: dict[str, str],
        contract: dict[str, Any],
        setup_result: dict[str, Any],
        source_policy: dict[str, Any],
        artifact: dict[str, Any],
        receipt_id: str,
        snapshot_id: str,
        evidence_link_id: str,
        policy_decision: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        review_text = json.dumps(
            {
                "source_summary": delivery.get("source_summary", {}),
                "payload": delivery.get("payload", {}),
                "normalized_projection": policy_decision.get("normalized_projection", {}),
            },
            sort_keys=True,
        )
        blocked_attempts = detect_unsafe_instructions(review_text)
        source_summary = delivery.get("source_summary") if isinstance(delivery.get("source_summary"), dict) else {}
        evidence_ref = delivery.get("evidence_ref") if isinstance(delivery.get("evidence_ref"), dict) else {}
        counters = {
            "tool_calls_created": 0,
            "action_cards_created_from_untrusted_artifact": 0,
            "workflow_runs_created_from_untrusted_artifact": 0,
            "connector_actions_triggered_from_content": 0,
            "provider_calls_triggered_from_content": 0,
            "shell_calls_triggered_from_content": 0,
            "external_http_calls": 0,
            "memory_promotions_from_untrusted_artifact": 0,
            "policy_overrides_from_untrusted_artifact": 0,
            "authority_expansions_from_untrusted_artifact": 0,
        }
        review_base = {
            "schema_version": UNTRUSTED_CONTENT_REVIEW_SCHEMA,
            "status": "blocked_attempt_recorded" if blocked_attempts else "reviewed",
            "scope": requested_scope,
            "contract_id": contract["contract_id"],
            "contract_version_id": contract["contract_version_id"],
            "setup_result_id": setup_result["setup_result_id"],
            "source_policy_id": source_policy["source_policy_id"],
            "delivery_id": delivery["delivery_id"],
            "projection_id": delivery["projection_id"],
            "delivery_receipt_id": receipt_id,
            "projection_snapshot_id": snapshot_id,
            "evidence_link_id": evidence_link_id,
            "artifact_id": artifact["artifact_id"],
            "evidence_ref_id": evidence_ref.get("evidence_ref_id"),
            "source_external_id": _source_external_id(delivery),
            "source_trust_label": source_summary.get("trust_state") or "untrusted_connector_content",
            "artifact_trust_state": artifact.get("trust_state"),
            "untrusted_evidence_label_present": artifact.get("trust_state") == "untrusted"
            and source_summary.get("trust_state") == "untrusted_connector_content",
            "unsafe_instruction_detected": bool(blocked_attempts),
            "blocked_attempt_count": len(blocked_attempts),
            "blocked_attempts": blocked_attempts,
            "content_handling": {
                "treated_as_system_instruction": False,
                "treated_as_policy_authority": False,
                "quoted_or_summarized_as_evidence_only": True,
                "claim_must_distinguish_quoted_instruction": True,
            },
            "required_authority_gates": {
                "explicit_product_intent_required": True,
                "role_or_mission_authority_required": True,
                "policy_decision_required": True,
                "declared_connector_capability_required": True,
                "owner_approval_required_for_external_or_risky_action": True,
            },
            "negative_evidence": counters,
            "raw_provider_payload_persisted": False,
            "created_at": utc_now(),
        }
        review_id = _untrusted_content_review_id(
            requested_scope,
            contract["contract_version_id"],
            str(delivery.get("delivery_id", "")),
            artifact["artifact_id"],
        )
        review = dict(review_base)
        review["review_id"] = review_id
        audit_event = self.store.append_audit(
            "connector.untrusted_content.blocked" if blocked_attempts else "connector.untrusted_content.reviewed",
            requested_scope,
            {"type": "connector_untrusted_content_review", "id": review_id},
            {
                "delivery_id": delivery["delivery_id"],
                "projection_id": delivery["projection_id"],
                "artifact_id": artifact["artifact_id"],
                "unsafe_instruction_detected": bool(blocked_attempts),
                "blocked_attempt_count": len(blocked_attempts),
                **counters,
            },
        )
        review["audit_refs"] = [f"audit:{audit_event['event_id']}"]
        _write_json(self._untrusted_content_review_path(review_id), review)
        safety = dict(artifact.get("safety", {}))
        existing_blocked_attempts = list(safety.get("blocked_attempts", []))
        for blocked_attempt in blocked_attempts:
            if blocked_attempt not in existing_blocked_attempts:
                existing_blocked_attempts.append(blocked_attempt)
        safety.update(
            {
                "untrusted_evidence": True,
                "unsafe_instruction_detected": bool(existing_blocked_attempts),
                "blocked_attempt_count": len(existing_blocked_attempts),
                "blocked_attempts": existing_blocked_attempts,
                **counters,
                "authority_expanded": False,
                "connector_untrusted_content_review_id": review_id,
            }
        )
        artifact["safety"] = safety
        artifact["connector_untrusted_content_review"] = {
            "review_id": review_id,
            "status": review["status"],
            "unsafe_instruction_detected": review["unsafe_instruction_detected"],
            "blocked_attempt_count": review["blocked_attempt_count"],
            "tool_calls_created": 0,
            "action_cards_created_from_untrusted_artifact": 0,
            "external_http_calls": 0,
            "authority_expanded": False,
        }
        artifact.setdefault("provenance", {}).setdefault("connector_refs", {})[
            "untrusted_content_review_id"
        ] = review_id
        artifact.setdefault("connector_delivery", {})["untrusted_content_review_id"] = review_id
        return review, audit_event

    def ingest_projection_delivery(
        self,
        delivery: dict[str, Any],
        requested_scope: dict[str, str],
        source_path: Path,
        contract_id: str,
        contract_version_id: str | None = None,
    ) -> dict[str, Any]:
        selected_contract_id = contract_id or delivery.get("contract_id")
        contract = self.load_contract(selected_contract_id, contract_version_id=contract_version_id)
        if contract is None:
            return {"status": "not_found", "contract_id": selected_contract_id}
        if contract.get("scope") != requested_scope:
            return {"status": "scope_denied", "resource_scope": contract.get("scope")}

        provider_pack_id = str(delivery.get("provider_pack_id", ""))
        setup_result = self._latest_setup_result(contract, requested_scope, provider_pack_id)
        if setup_result is None:
            return {
                "status": "setup_missing",
                "contract": contract,
                "issues": [
                    {
                        "code": "CS_CONNECTOR_SETUP_NOT_FOUND",
                        "message": "A scoped Setup Result must exist before Projection Delivery ingest.",
                        "path": "setup_result",
                    }
                ],
            }

        issues = self._delivery_validation_issues(delivery, contract, setup_result, requested_scope)
        policy_decision, policy_issues = self._evaluate_projection_source_policy(
            delivery,
            contract,
            setup_result,
            requested_scope,
        )
        issues.extend(policy_issues)
        policy_event = self.store.append_audit(
            "connector.source_policy.enforced",
            requested_scope,
            {"type": "connector_projection_policy_decision", "id": policy_decision["policy_decision_id"]},
            {
                "delivery_id": delivery.get("delivery_id"),
                "projection_id": delivery.get("projection_id"),
                "decision": policy_decision["decision"],
                "enforcement_action": policy_decision["enforcement_action"],
                "source_policy_id": policy_decision.get("source_policy_id"),
                "issue_codes": policy_decision["issue_codes"],
                "raw_content_persisted": False,
                "raw_provider_payload_persisted": False,
            },
        )
        policy_decision["audit_refs"] = [f"audit:{policy_event['event_id']}"]
        content_decision = policy_decision.get("content_restriction_decision", {})
        if content_decision:
            evidence_ref = delivery.get("evidence_ref") if isinstance(delivery.get("evidence_ref"), dict) else {}
            _merge_content_decision_refs(
                content_decision,
                evidence_refs=[
                    ref
                    for ref in [
                        f"connector_content_restriction_decision:{content_decision['content_restriction_decision_id']}",
                        f"connector_projection_policy_decision:{policy_decision['policy_decision_id']}",
                        f"connector_contract:{contract['contract_version_id']}",
                        f"connector_setup_result:{setup_result.get('setup_result_id')}",
                        f"connector_source_policy:{policy_decision.get('source_policy_id')}",
                        f"connector_delivery:{delivery.get('delivery_id')}",
                        f"connector_projection:{delivery.get('projection_id')}",
                        f"connector_evidence_ref:{evidence_ref.get('evidence_ref_id')}"
                        if evidence_ref.get("evidence_ref_id")
                        else None,
                    ]
                    if ref
                ],
                audit_refs=policy_decision["audit_refs"],
                linked_records={
                    "policy_decision_id": policy_decision["policy_decision_id"],
                    "contract_version_id": contract["contract_version_id"],
                    "setup_result_id": setup_result.get("setup_result_id"),
                    "source_policy_id": policy_decision.get("source_policy_id"),
                    "delivery_id": str(delivery.get("delivery_id") or ""),
                    "projection_id": str(delivery.get("projection_id") or ""),
                },
            )
        _write_json(self._projection_policy_decision_path(policy_decision["policy_decision_id"]), policy_decision)
        if content_decision:
            _write_json(
                self._content_restriction_decision_path(content_decision["content_restriction_decision_id"]),
                content_decision,
            )
        if content_decision.get("action") == "quarantine":
            return self._record_content_quarantine(
                delivery=delivery,
                requested_scope=requested_scope,
                contract=contract,
                setup_result=setup_result,
                policy_decision=policy_decision,
                issues=issues,
                prior_audit_events=[policy_event],
            )
        if issues:
            reject_event = self.store.append_audit(
                "connector.delivery.rejected",
                requested_scope,
                {"type": "connector_delivery", "id": str(delivery.get("delivery_id", ""))},
                {"contract_version_id": contract["contract_version_id"], "issue_codes": [issue["code"] for issue in issues]},
            )
            if content_decision:
                reject_audit_refs = [f"audit:{policy_event['event_id']}", f"audit:{reject_event['event_id']}"]
                _merge_content_decision_refs(
                    content_decision,
                    audit_refs=reject_audit_refs,
                    linked_records={"rejected_delivery_id": str(delivery.get("delivery_id") or "")},
                )
                policy_decision["content_restriction_decision"] = content_decision
                policy_decision["audit_refs"] = reject_audit_refs
                _write_json(self._projection_policy_decision_path(policy_decision["policy_decision_id"]), policy_decision)
                _write_json(
                    self._content_restriction_decision_path(content_decision["content_restriction_decision_id"]),
                    content_decision,
                )
            return {
                "status": "failed",
                "contract": contract,
                "setup_result": setup_result,
                "source_policy": setup_result.get("source_policy_snapshot", {}),
                "issues": issues,
                "audit_event": reject_event,
                "audit_events": [policy_event, reject_event],
                "connector_projection_policy_decision": policy_decision,
            }

        source_external_id = _source_external_id(delivery)
        source_revision = _source_revision(delivery)
        source_content_hash = _source_content_hash(delivery)
        idempotency_key = _delivery_idempotency_key(requested_scope, contract["contract_version_id"], delivery)
        content_source_key = _content_source_key(requested_scope, contract["contract_version_id"], source_external_id)
        content_version_id = _content_version_id(requested_scope, contract["contract_version_id"], source_external_id, source_content_hash)
        existing_dedupe = self._load_delivery_dedupe(idempotency_key)
        if existing_dedupe:
            artifact = self.store.get_artifact(str(existing_dedupe.get("artifact_id")), requested_scope)
            receipt = self._load_delivery_receipt(str(existing_dedupe.get("delivery_receipt_id", "")))
            projection_snapshot = self._load_projection_snapshot(str(existing_dedupe.get("projection_snapshot_id", "")))
            evidence_link = self._load_evidence_link(str(existing_dedupe.get("evidence_link_id", "")))
            content_version = self._load_content_version(str(existing_dedupe.get("content_version_id", "")))
            if artifact and receipt and projection_snapshot and evidence_link and content_version:
                review_id = str(
                    receipt.get("untrusted_content_review", {}).get("review_id")
                    or artifact.get("provenance", {}).get("connector_refs", {}).get("untrusted_content_review_id")
                    or artifact.get("connector_delivery", {}).get("untrusted_content_review_id")
                    or ""
                )
                untrusted_review = self._load_untrusted_content_review(review_id) if review_id else None
                now = utc_now()
                observed_delivery_ids = list(existing_dedupe.get("observed_delivery_ids", []))
                if delivery["delivery_id"] not in observed_delivery_ids:
                    observed_delivery_ids.append(delivery["delivery_id"])
                provider_event_ids = list(existing_dedupe.get("provider_event_ids", []))
                if delivery["provider_event_id"] not in provider_event_ids:
                    provider_event_ids.append(delivery["provider_event_id"])
                source_revisions = list(existing_dedupe.get("source_revisions", []))
                if source_revision and source_revision not in source_revisions:
                    source_revisions.append(source_revision)
                existing_dedupe.update(
                    {
                        "status": "deduplicated",
                        "observed_delivery_ids": observed_delivery_ids,
                        "provider_event_ids": provider_event_ids,
                        "source_revisions": source_revisions,
                        "delivery_count": len(observed_delivery_ids),
                        "duplicate_delivery_count": max(len(observed_delivery_ids) - 1, 0),
                        "last_seen_delivery_id": delivery["delivery_id"],
                        "last_seen_provider_event_id": delivery["provider_event_id"],
                        "last_seen_source_revision": source_revision,
                        "no_new_artifact_created": True,
                        "no_duplicate_active_truth_created": True,
                        "updated_at": now,
                    }
                )
                _write_json(self._delivery_dedupe_path(idempotency_key), existing_dedupe)
                audit_event = self.store.append_audit(
                    "connector.delivery.deduplicated",
                    requested_scope,
                    {"type": "connector_delivery_dedupe_state", "id": idempotency_key},
                    {
                        "canonical_delivery_receipt_id": receipt["delivery_receipt_id"],
                        "artifact_id": artifact["artifact_id"],
                        "content_version_id": content_version["content_version_id"],
                        "source_external_id": source_external_id,
                        "source_content_hash": source_content_hash,
                        "no_new_artifact_created": True,
                        "no_duplicate_active_truth_created": True,
                    },
                )
                audit_refs = [f"audit:{policy_event['event_id']}", f"audit:{audit_event['event_id']}"]
                if content_decision:
                    _merge_content_decision_refs(
                        content_decision,
                        evidence_refs=[
                            f"artifact:{artifact['artifact_id']}",
                            f"storage:{artifact['original_storage_ref']}",
                            f"connector_delivery_receipt:{receipt['delivery_receipt_id']}",
                            f"connector_projection_snapshot:{projection_snapshot['projection_snapshot_id']}",
                            f"connector_evidence_link:{evidence_link['evidence_link_id']}",
                            f"connector_content_version:{content_version['content_version_id']}",
                            f"connector_delivery_dedupe_state:{idempotency_key}",
                        ],
                        audit_refs=audit_refs,
                        linked_records={
                            "artifact_id": artifact["artifact_id"],
                            "storage_ref": artifact["original_storage_ref"],
                            "delivery_receipt_id": receipt["delivery_receipt_id"],
                            "projection_snapshot_id": projection_snapshot["projection_snapshot_id"],
                            "evidence_link_id": evidence_link["evidence_link_id"],
                            "content_version_id": content_version["content_version_id"],
                            "delivery_idempotency_key": idempotency_key,
                        },
                    )
                    policy_decision["content_restriction_decision"] = content_decision
                    policy_decision["audit_refs"] = audit_refs
                    _write_json(self._projection_policy_decision_path(policy_decision["policy_decision_id"]), policy_decision)
                    _write_json(
                        self._content_restriction_decision_path(content_decision["content_restriction_decision_id"]),
                        content_decision,
                    )
                return {
                    "status": "success",
                    "contract": contract,
                    "setup_result": setup_result,
                    "source_policy": setup_result["source_policy_snapshot"],
                    "artifact": artifact,
                    "delivery_receipt": receipt,
                    "projection_snapshot": projection_snapshot,
                    "evidence_link": evidence_link,
                    "audit_event": audit_event,
                    "audit_events": [policy_event, audit_event],
                    "deduplicated": True,
                    "connector_projection_policy_decision": policy_decision,
                    "connector_content_restriction_decision": content_decision,
                    "connector_untrusted_content_review": untrusted_review,
                    "connector_delivery_dedupe_state": existing_dedupe,
                    "connector_content_version": content_version,
                }

        current_pointer = self._load_content_current(content_source_key)
        previous_version = (
            self._load_content_version(str(current_pointer.get("current_content_version_id", "")))
            if current_pointer and current_pointer.get("source_content_hash") != source_content_hash
            else None
        )
        predecessor_version_id = str(previous_version.get("content_version_id")) if previous_version else None
        predecessor_artifact_id = str(previous_version.get("artifact_id")) if previous_version else None
        version_ordinal = int(previous_version.get("version_ordinal", 0)) + 1 if previous_version else 1
        lineage_from = f"artifact:{predecessor_artifact_id}" if predecessor_artifact_id else f"connector_projection:{delivery['projection_id']}"

        artifact_source_path = source_path
        if content_decision.get("action") in {"redact", "metadata_only"}:
            artifact_source_path = self._write_content_restricted_delivery(delivery, content_decision)
        artifact_result = self.store.ingest_artifact(
            artifact_source_path,
            tenant_id=requested_scope["tenant_id"],
            owner_id=requested_scope["owner_id"],
            namespace_id=requested_scope["namespace_id"],
            workspace_id=requested_scope["workspace_id"],
            source="connector_projection_delivery",
            media_type=DELIVERY_ARTIFACT_MEDIA_TYPE,
            derived_mode="unsupported",
            trust="untrusted",
            lineage_from=lineage_from,
        )
        artifact = dict(artifact_result["artifact"])
        checksum = artifact["checksum_sha256"]
        source_policy = setup_result["source_policy_snapshot"]
        receipt_id = _delivery_receipt_id(idempotency_key, artifact["artifact_id"])
        snapshot_id = _projection_snapshot_id(delivery["projection_id"], checksum)
        evidence_link_id = _evidence_link_id(receipt_id, artifact["artifact_id"])

        artifact["source"] = {
            **artifact.get("source", {}),
            "type": "connector_projection_delivery",
            "delivery_id": delivery["delivery_id"],
            "projection_id": delivery["projection_id"],
            "provider_event_id": delivery["provider_event_id"],
            "source_external_id": source_external_id,
            "source_revision": source_revision,
            "source_content_hash": source_content_hash,
        }
        artifact["connector_delivery"] = {
            "schema_version": "cs.connector_delivery_artifact_link.v1",
            "delivery_id": delivery["delivery_id"],
            "projection_id": delivery["projection_id"],
            "delivery_idempotency_key": idempotency_key,
            "contract_id": contract["contract_id"],
            "contract_version_id": contract["contract_version_id"],
            "setup_result_id": setup_result["setup_result_id"],
            "source_policy_id": source_policy["source_policy_id"],
            "policy_decision_id": policy_decision["policy_decision_id"],
            "provider_pack_id": delivery["provider_pack_id"],
            "provider_event_id": delivery["provider_event_id"],
            "common_capability": delivery["common_capability"],
            "projection_type": delivery["projection_type"],
            "evidence_ref_id": delivery["evidence_ref"]["evidence_ref_id"],
            "source_summary": delivery["source_summary"],
            "source_policy_enforcement": {
                "decision": policy_decision["decision"],
                "content_mode": policy_decision["content_mode"],
                "included_fields": policy_decision["included_fields"],
                "excluded_fields": policy_decision["excluded_fields"],
                "restriction_summary": policy_decision["restriction_summary"],
                "content_restriction_decision_id": content_decision.get("content_restriction_decision_id"),
                "content_policy_action": content_decision.get("action"),
                "durable_content_status": content_decision.get("durable_content_status"),
                "partial_status": content_decision.get("partial_status"),
                "redaction_applied": content_decision.get("redaction_applied") is True,
                "metadata_only": content_decision.get("metadata_only") is True,
                "raw_content_persisted": False,
            },
            "source_external_id": source_external_id,
            "source_revision": source_revision,
            "source_content_hash": source_content_hash,
            "content_source_key": content_source_key,
            "content_version_id": content_version_id,
            "predecessor_content_version_id": predecessor_version_id,
            "predecessor_artifact_id": predecessor_artifact_id,
            "exact_envelope_preserved": artifact_source_path == source_path,
            "artifact_input_sanitized": artifact_source_path != source_path,
            "raw_provider_payload_stored_in_product_state": False,
        }
        artifact["provenance"] = {
            **artifact.get("provenance", {}),
            "connector_refs": {
                "delivery_receipt_id": receipt_id,
                "projection_snapshot_id": snapshot_id,
                "evidence_link_id": evidence_link_id,
                "setup_result_id": setup_result["setup_result_id"],
                "source_policy_id": source_policy["source_policy_id"],
                "policy_decision_id": policy_decision["policy_decision_id"],
            },
        }
        _write_json(self.store.artifact_path(artifact["artifact_id"], requested_scope), artifact)

        connector_audit_event = self.store.append_audit(
            "connector.delivery.archived",
            requested_scope,
            {"type": "connector_delivery_receipt", "id": receipt_id},
            {
                "delivery_id": delivery["delivery_id"],
                "projection_id": delivery["projection_id"],
                "artifact_id": artifact["artifact_id"],
                "checksum_sha256": checksum,
                "source_policy_id": source_policy["source_policy_id"],
                "setup_result_id": setup_result["setup_result_id"],
                "acknowledged": False,
                "product_interpretation_before_archive_commit": False,
            },
        )
        audit_events = [policy_event] + list(artifact_result.get("audit_events", [artifact_result["audit_event"]])) + [connector_audit_event]
        audit_refs = [f"audit:{event['event_id']}" for event in audit_events]
        receipt = {
            "schema_version": DELIVERY_RECEIPT_SCHEMA,
            "delivery_receipt_id": receipt_id,
            "delivery_id": delivery["delivery_id"],
            "projection_id": delivery["projection_id"],
            "scope": requested_scope,
            "contract_id": contract["contract_id"],
            "contract_version_id": contract["contract_version_id"],
            "setup_result_id": setup_result["setup_result_id"],
            "source_policy_id": source_policy["source_policy_id"],
            "policy_decision_id": policy_decision["policy_decision_id"],
            "provider_pack_id": delivery["provider_pack_id"],
            "provider_event_id": delivery["provider_event_id"],
            "common_capability": delivery["common_capability"],
            "projection_type": delivery["projection_type"],
            "artifact_id": artifact["artifact_id"],
            "artifact_checksum_sha256": checksum,
            "envelope_sha256": checksum,
            "delivery_idempotency_key": idempotency_key,
            "source_external_id": source_external_id,
            "source_revision": source_revision,
            "source_content_hash": source_content_hash,
            "content_source_key": content_source_key,
            "content_version_id": content_version_id,
            "predecessor_content_version_id": predecessor_version_id,
            "predecessor_artifact_id": predecessor_artifact_id,
            "original_storage_ref": artifact["original_storage_ref"],
            "source_summary": delivery["source_summary"],
            "evidence_ref": delivery["evidence_ref"],
            "source_policy_enforcement": {
                "decision": policy_decision["decision"],
                "content_mode": policy_decision["content_mode"],
                "included_fields": policy_decision["included_fields"],
                "excluded_fields": policy_decision["excluded_fields"],
                "restriction_summary": policy_decision["restriction_summary"],
                "content_restriction_decision_id": content_decision.get("content_restriction_decision_id"),
                "content_policy_action": content_decision.get("action"),
                "durable_content_status": content_decision.get("durable_content_status"),
                "partial_status": content_decision.get("partial_status"),
                "redaction_applied": content_decision.get("redaction_applied") is True,
                "metadata_only": content_decision.get("metadata_only") is True,
                "raw_content_persisted": False,
            },
            "commit_state": "artifact_committed",
            "acknowledgement_state": "not_acknowledged_by_cs_ch_007",
            "product_interpretation": {
                "before_archive_commit": False,
                "handlers_receive_committed_artifact_only": True,
            },
            "raw_provider_payload": {
                "exposed_to_product": False,
                "stored_in_product_state": False,
            },
            "provider_call_ledger": {"during_ingest": 0, "external_http_calls": 0},
            "audit_refs": audit_refs,
            "created_at": utc_now(),
        }
        projection_snapshot = {
            "schema_version": PROJECTION_SNAPSHOT_SCHEMA,
            "projection_snapshot_id": snapshot_id,
            "projection_id": delivery["projection_id"],
            "delivery_id": delivery["delivery_id"],
            "scope": requested_scope,
            "artifact_id": artifact["artifact_id"],
            "envelope_sha256": checksum,
            "contract_version_id": contract["contract_version_id"],
            "setup_result_id": setup_result["setup_result_id"],
            "source_policy_id": source_policy["source_policy_id"],
            "policy_decision_id": policy_decision["policy_decision_id"],
            "provider_pack_id": delivery["provider_pack_id"],
            "provider_event_id": delivery["provider_event_id"],
            "common_capability": delivery["common_capability"],
            "projection_type": delivery["projection_type"],
            "delivery_idempotency_key": idempotency_key,
            "source_external_id": source_external_id,
            "source_revision": source_revision,
            "source_content_hash": source_content_hash,
            "content_version_id": content_version_id,
            "predecessor_content_version_id": predecessor_version_id,
            "source_summary": delivery["source_summary"],
            "evidence_ref": delivery["evidence_ref"],
            "normalized_projection": policy_decision["normalized_projection"],
            "source_policy_enforcement": {
                "decision": policy_decision["decision"],
                "content_mode": policy_decision["content_mode"],
                "included_fields": policy_decision["included_fields"],
                "excluded_fields": policy_decision["excluded_fields"],
                "restriction_summary": policy_decision["restriction_summary"],
                "content_restriction_decision_id": content_decision.get("content_restriction_decision_id"),
                "content_policy_action": content_decision.get("action"),
                "durable_content_status": content_decision.get("durable_content_status"),
                "partial_status": content_decision.get("partial_status"),
                "redaction_applied": content_decision.get("redaction_applied") is True,
                "metadata_only": content_decision.get("metadata_only") is True,
                "raw_content_persisted": False,
            },
            "created_at": utc_now(),
        }
        evidence_link = {
            "schema_version": EVIDENCE_LINK_SCHEMA,
            "evidence_link_id": evidence_link_id,
            "scope": requested_scope,
            "artifact_id": artifact["artifact_id"],
            "delivery_receipt_id": receipt_id,
            "projection_snapshot_id": snapshot_id,
            "projection_id": delivery["projection_id"],
            "delivery_idempotency_key": idempotency_key,
            "source_external_id": source_external_id,
            "source_revision": source_revision,
            "source_content_hash": source_content_hash,
            "content_version_id": content_version_id,
            "predecessor_content_version_id": predecessor_version_id,
            "source_policy_id": source_policy["source_policy_id"],
            "policy_decision_id": policy_decision["policy_decision_id"],
            "content_restriction_decision_id": content_decision.get("content_restriction_decision_id"),
            "setup_result_id": setup_result["setup_result_id"],
            "contract_version_id": contract["contract_version_id"],
            "evidence_ref": delivery["evidence_ref"],
            "normalized_projection": policy_decision["normalized_projection"],
            "created_at": utc_now(),
        }
        untrusted_review, untrusted_review_event = self._record_untrusted_content_review(
            delivery=delivery,
            requested_scope=requested_scope,
            contract=contract,
            setup_result=setup_result,
            source_policy=source_policy,
            artifact=artifact,
            receipt_id=receipt_id,
            snapshot_id=snapshot_id,
            evidence_link_id=evidence_link_id,
            policy_decision=policy_decision,
        )
        untrusted_review_ref = {
            "review_id": untrusted_review["review_id"],
            "status": untrusted_review["status"],
            "source_trust_label": untrusted_review["source_trust_label"],
            "unsafe_instruction_detected": untrusted_review["unsafe_instruction_detected"],
            "blocked_attempt_count": untrusted_review["blocked_attempt_count"],
            "treated_as_system_instruction": False,
            "quoted_or_summarized_as_evidence_only": True,
            "tool_calls_created": 0,
            "action_cards_created_from_untrusted_artifact": 0,
            "workflow_runs_created_from_untrusted_artifact": 0,
            "external_http_calls": 0,
            "memory_promotions_from_untrusted_artifact": 0,
            "policy_overrides_from_untrusted_artifact": 0,
            "authority_expanded": False,
        }
        receipt["untrusted_content_review"] = untrusted_review_ref
        projection_snapshot["untrusted_content_review"] = untrusted_review_ref
        evidence_link["untrusted_content_review"] = untrusted_review_ref
        content_version = {
            "schema_version": CONTENT_VERSION_SCHEMA,
            "content_version_id": content_version_id,
            "content_source_key": content_source_key,
            "scope": requested_scope,
            "contract_id": contract["contract_id"],
            "contract_version_id": contract["contract_version_id"],
            "setup_result_id": setup_result["setup_result_id"],
            "source_policy_id": source_policy["source_policy_id"],
            "policy_decision_id": policy_decision["policy_decision_id"],
            "content_restriction_decision_id": content_decision.get("content_restriction_decision_id"),
            "provider_pack_id": delivery["provider_pack_id"],
            "source_external_id": source_external_id,
            "source_revision": source_revision,
            "source_content_hash": source_content_hash,
            "projection_type": delivery["projection_type"],
            "artifact_id": artifact["artifact_id"],
            "delivery_receipt_id": receipt_id,
            "projection_snapshot_id": snapshot_id,
            "evidence_link_id": evidence_link_id,
            "provider_event_ids": [delivery["provider_event_id"]],
            "delivery_idempotency_key": idempotency_key,
            "predecessor_content_version_id": predecessor_version_id,
            "predecessor_artifact_id": predecessor_artifact_id,
            "version_ordinal": version_ordinal,
            "historical_evidence_mutated": False,
            "raw_provider_payload_stored_in_product_state": False,
            "raw_content_persisted": False,
            "content_policy_action": content_decision.get("action"),
            "durable_content_status": content_decision.get("durable_content_status"),
            "partial_status": content_decision.get("partial_status"),
            "source_policy_restriction_summary": policy_decision["restriction_summary"],
            "created_at": utc_now(),
        }
        current_record = {
            "schema_version": CONTENT_CURRENT_SCHEMA,
            "content_source_key": content_source_key,
            "scope": requested_scope,
            "contract_id": contract["contract_id"],
            "contract_version_id": contract["contract_version_id"],
            "source_external_id": source_external_id,
            "current_content_version_id": content_version_id,
            "current_artifact_id": artifact["artifact_id"],
            "current_policy_decision_id": policy_decision["policy_decision_id"],
            "current_source_revision": source_revision,
            "source_content_hash": source_content_hash,
            "previous_content_version_id": predecessor_version_id,
            "previous_artifact_id": predecessor_artifact_id,
            "version_count": version_ordinal,
            "one_current_logical_truth": True,
            "historical_evidence_mutated": False,
            "updated_at": utc_now(),
        }
        dedupe_state = {
            "schema_version": DELIVERY_DEDUPE_SCHEMA,
            "delivery_idempotency_key": idempotency_key,
            "status": "canonical",
            "scope": requested_scope,
            "contract_id": contract["contract_id"],
            "contract_version_id": contract["contract_version_id"],
            "setup_result_id": setup_result["setup_result_id"],
            "source_policy_id": source_policy["source_policy_id"],
            "policy_decision_id": policy_decision["policy_decision_id"],
            "provider_pack_id": delivery["provider_pack_id"],
            "source_external_id": source_external_id,
            "source_revision": source_revision,
            "source_content_hash": source_content_hash,
            "projection_type": delivery["projection_type"],
            "canonical_delivery_id": delivery["delivery_id"],
            "observed_delivery_ids": [delivery["delivery_id"]],
            "provider_event_ids": [delivery["provider_event_id"]],
            "source_revisions": [source_revision] if source_revision else [],
            "delivery_count": 1,
            "duplicate_delivery_count": 0,
            "artifact_id": artifact["artifact_id"],
            "delivery_receipt_id": receipt_id,
            "projection_snapshot_id": snapshot_id,
            "evidence_link_id": evidence_link_id,
            "content_version_id": content_version_id,
            "policy_decision_id": policy_decision["policy_decision_id"],
            "no_new_artifact_created": False,
            "no_duplicate_active_truth_created": True,
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        content_event = self.store.append_audit(
            "connector.content_version.created",
            requested_scope,
            {"type": "connector_content_version", "id": content_version_id},
            {
                "source_external_id": source_external_id,
                "source_revision": source_revision,
                "source_content_hash": source_content_hash,
                "artifact_id": artifact["artifact_id"],
                "predecessor_content_version_id": predecessor_version_id,
                "historical_evidence_mutated": False,
            },
        )
        current_event = self.store.append_audit(
            "connector.content_current.advanced",
            requested_scope,
            {"type": "connector_content_current", "id": content_source_key},
            {
                "source_external_id": source_external_id,
                "current_content_version_id": content_version_id,
                "previous_content_version_id": predecessor_version_id,
                "one_current_logical_truth": True,
            },
        )
        audit_events = [policy_event] + list(artifact_result.get("audit_events", [artifact_result["audit_event"]])) + [
            connector_audit_event,
            untrusted_review_event,
            content_event,
            current_event,
        ]
        audit_refs = [f"audit:{event['event_id']}" for event in audit_events]
        if content_decision:
            _merge_content_decision_refs(
                content_decision,
                evidence_refs=[
                    f"artifact:{artifact['artifact_id']}",
                    f"storage:{artifact['original_storage_ref']}",
                    f"connector_delivery_receipt:{receipt_id}",
                    f"connector_projection_snapshot:{snapshot_id}",
                    f"connector_evidence_link:{evidence_link_id}",
                    f"connector_content_version:{content_version_id}",
                    f"connector_content_current:{content_source_key}",
                    f"connector_delivery_dedupe_state:{idempotency_key}",
                    f"connector_untrusted_content_review:{untrusted_review['review_id']}",
                ],
                audit_refs=audit_refs,
                linked_records={
                    "artifact_id": artifact["artifact_id"],
                    "storage_ref": artifact["original_storage_ref"],
                    "delivery_receipt_id": receipt_id,
                    "projection_snapshot_id": snapshot_id,
                    "evidence_link_id": evidence_link_id,
                    "content_version_id": content_version_id,
                    "content_source_key": content_source_key,
                    "delivery_idempotency_key": idempotency_key,
                    "untrusted_content_review_id": untrusted_review["review_id"],
                },
            )
        receipt["audit_refs"] = audit_refs
        evidence_link["audit_refs"] = audit_refs
        content_version["audit_refs"] = audit_refs
        dedupe_state["audit_refs"] = audit_refs
        current_record["audit_refs"] = audit_refs
        policy_decision["audit_refs"] = audit_refs
        untrusted_review["audit_refs"] = list(
            dict.fromkeys([*untrusted_review.get("audit_refs", []), *audit_refs])
        )
        _write_json(self._projection_policy_decision_path(policy_decision["policy_decision_id"]), policy_decision)
        if content_decision:
            _write_json(
                self._content_restriction_decision_path(content_decision["content_restriction_decision_id"]),
                content_decision,
            )
        _write_json(self._untrusted_content_review_path(untrusted_review["review_id"]), untrusted_review)
        _write_json(self.store.artifact_path(artifact["artifact_id"], requested_scope), artifact)
        _write_json(self._delivery_receipt_path(receipt_id), receipt)
        _write_json(self._projection_snapshot_path(snapshot_id), projection_snapshot)
        _write_json(self._evidence_link_path(evidence_link_id), evidence_link)
        _write_json(self._content_version_path(content_version_id), content_version)
        _write_json(self._content_current_path(content_source_key), current_record)
        _write_json(self._delivery_dedupe_path(idempotency_key), dedupe_state)
        return {
            "status": "success",
            "contract": contract,
            "setup_result": setup_result,
            "source_policy": source_policy,
            "artifact": artifact,
            "delivery_receipt": receipt,
            "projection_snapshot": projection_snapshot,
            "evidence_link": evidence_link,
            "audit_event": connector_audit_event,
            "audit_events": audit_events,
            "deduplicated": artifact_result.get("deduplicated", False),
            "connector_projection_policy_decision": policy_decision,
            "connector_content_restriction_decision": content_decision,
            "connector_untrusted_content_review": untrusted_review,
            "connector_delivery_dedupe_state": dedupe_state,
            "connector_content_version": content_version,
            "connector_content_current": current_record,
        }

    def _record_delivery_failure(
        self,
        delivery: dict[str, Any],
        requested_scope: dict[str, str],
        contract: dict[str, Any],
        setup_result: dict[str, Any],
        failure_class: str,
        issues: list[dict[str, Any]],
    ) -> dict[str, Any]:
        retry_policy = contract.get("delivery", {}) if isinstance(contract.get("delivery"), dict) else {}
        quarantine_after_attempts = int(retry_policy.get("quarantine_after_attempts") or 3)
        retry_state_id = _delivery_retry_state_id(requested_scope, contract["contract_version_id"], delivery)
        existing = self._load_delivery_retry(retry_state_id)
        now = utc_now()
        attempts = list(existing.get("attempts", [])) if existing else []
        attempt_no = len(attempts) + 1
        delay_seconds = _retry_delay_seconds(attempt_no)
        issue_codes = [str(issue.get("code", "CS_CONNECTOR_DELIVERY_FAILURE")) for issue in issues]
        reason_code = issue_codes[0] if issue_codes else "CS_CONNECTOR_DELIVERY_FAILURE"
        retry_schedule = list(existing.get("retry_schedule", [])) if existing else []
        next_retry_at = utc_after(delay_seconds)
        attempt = {
            "attempt_no": attempt_no,
            "failure_class": failure_class,
            "reason_code": reason_code,
            "issue_codes": issue_codes,
            "retryable": attempt_no < quarantine_after_attempts,
            "delay_seconds": delay_seconds,
            "next_retry_at": next_retry_at,
            "raw_provider_payload_persisted": False,
            "at": now,
        }
        attempts.append(attempt)
        retry_schedule.append(
            {
                "attempt_no": attempt_no,
                "delay_seconds": delay_seconds,
                "next_retry_at": next_retry_at,
                "bounded": True,
            }
        )
        source_policy = setup_result.get("source_policy_snapshot", {})
        retry_state = existing or {
            "schema_version": DELIVERY_RETRY_STATE_SCHEMA,
            "retry_state_id": retry_state_id,
            "delivery_id": delivery.get("delivery_id"),
            "projection_id": delivery.get("projection_id"),
            "scope": requested_scope,
            "contract_id": contract["contract_id"],
            "contract_version_id": contract["contract_version_id"],
            "setup_result_id": setup_result.get("setup_result_id"),
            "source_policy_id": source_policy.get("source_policy_id"),
            "provider_pack_id": delivery.get("provider_pack_id"),
            "provider_event_id": delivery.get("provider_event_id"),
            "common_capability": delivery.get("common_capability"),
            "projection_type": delivery.get("projection_type"),
            "delivery_ref": _safe_delivery_ref(delivery),
            "retry_policy": retry_policy.get("retry_policy", "bounded_exponential"),
            "quarantine_after_attempts": quarantine_after_attempts,
            "created_at": now,
        }
        retry_state.update(
            {
                "status": "quarantined" if attempt_no >= quarantine_after_attempts else "retry_scheduled",
                "failure_class": failure_class,
                "attempt_count": attempt_no,
                "last_reason_code": reason_code,
                "attempts": attempts,
                "retry_schedule": retry_schedule,
                "next_retry_at": None if attempt_no >= quarantine_after_attempts else next_retry_at,
                "unrelated_streams_blocked": False,
                "raw_provider_payload_persisted": False,
                "updated_at": now,
            }
        )
        _write_json(self._delivery_retry_path(retry_state_id), retry_state)

        if attempt_no < quarantine_after_attempts:
            audit_event = self.store.append_audit(
                "connector.delivery.retry_scheduled",
                requested_scope,
                {"type": "connector_delivery_retry_state", "id": retry_state_id},
                {
                    "delivery_id": delivery.get("delivery_id"),
                    "attempt_no": attempt_no,
                    "failure_class": failure_class,
                    "reason_code": reason_code,
                    "delay_seconds": delay_seconds,
                    "raw_provider_payload_persisted": False,
                    "unrelated_streams_blocked": False,
                },
            )
            return {
                "status": "retry_scheduled",
                "contract": contract,
                "setup_result": setup_result,
                "delivery_retry_state": retry_state,
                "audit_event": audit_event,
                "audit_events": [audit_event],
                "issues": issues,
            }

        quarantine_id = _delivery_quarantine_id(retry_state_id)
        existing_quarantine = self._load_quarantine(quarantine_id)
        quarantine = existing_quarantine or {
            "schema_version": DELIVERY_QUARANTINE_SCHEMA,
            "quarantine_id": quarantine_id,
            "retry_state_id": retry_state_id,
            "delivery_id": delivery.get("delivery_id"),
            "projection_id": delivery.get("projection_id"),
            "scope": requested_scope,
            "contract_id": contract["contract_id"],
            "contract_version_id": contract["contract_version_id"],
            "setup_result_id": setup_result.get("setup_result_id"),
            "source_policy_id": source_policy.get("source_policy_id"),
            "provider_pack_id": delivery.get("provider_pack_id"),
            "provider_event_id": delivery.get("provider_event_id"),
            "common_capability": delivery.get("common_capability"),
            "projection_type": delivery.get("projection_type"),
            "delivery_ref": _safe_delivery_ref(delivery),
            "created_at": now,
            "replay_attempts": [],
        }
        quarantine.update(
            {
                "status": "quarantined",
                "failure_class": failure_class,
                "reason_code": reason_code,
                "attempt_count": attempt_no,
                "quarantine_after_attempts": quarantine_after_attempts,
                "retry_policy": retry_state.get("retry_policy"),
                "redacted_error": {
                    "reason_code": reason_code,
                    "issue_codes": issue_codes,
                    "message": "Delivery processing failed and was quarantined with raw payload omitted.",
                    "raw_provider_payload_included": False,
                    "secret_values_included": False,
                },
                "source_health_impact": {
                    "affected_source": delivery.get("source_summary", {}).get("source_ref")
                    if isinstance(delivery.get("source_summary"), dict)
                    else None,
                    "affected_capability": delivery.get("common_capability"),
                    "freshness_state": "degraded_for_affected_delivery",
                    "unrelated_streams_continue": True,
                    "unrelated_streams_blocked": False,
                },
                "safe_diagnostics": {
                    "raw_provider_payload_persisted": False,
                    "raw_provider_payload_in_operator_output": False,
                    "provider_credentials_exposed": False,
                    "resolution": "Inspect connector diagnostics in the Hub, correct the source payload or mapping, then request replay.",
                },
                "failure_evidence_preserved": True,
                "updated_at": now,
            }
        )
        _write_json(self._quarantine_path(quarantine_id), quarantine)
        retry_state["quarantine_id"] = quarantine_id
        _write_json(self._delivery_retry_path(retry_state_id), retry_state)
        audit_event = self.store.append_audit(
            "connector.delivery.quarantined",
            requested_scope,
            {"type": "connector_delivery_quarantine", "id": quarantine_id},
            {
                "delivery_id": delivery.get("delivery_id"),
                "retry_state_id": retry_state_id,
                "attempt_count": attempt_no,
                "failure_class": failure_class,
                "reason_code": reason_code,
                "raw_provider_payload_persisted": False,
                "unrelated_streams_blocked": False,
            },
        )
        return {
            "status": "quarantined",
            "contract": contract,
            "setup_result": setup_result,
            "delivery_retry_state": retry_state,
            "delivery_quarantine": quarantine,
            "audit_event": audit_event,
            "audit_events": [audit_event],
            "issues": issues,
        }

    def _mark_delivery_retry_resolved(
        self,
        delivery: dict[str, Any],
        requested_scope: dict[str, str],
        contract: dict[str, Any],
        artifact_id: str,
    ) -> dict[str, Any] | None:
        retry_state_id = _delivery_retry_state_id(requested_scope, contract["contract_version_id"], delivery)
        retry_state = self._load_delivery_retry(retry_state_id)
        if retry_state is None or retry_state.get("status") == "resolved":
            return retry_state
        now = utc_now()
        retry_state.update(
            {
                "status": "resolved",
                "resolved_at": now,
                "resolved_by_artifact_id": artifact_id,
                "next_retry_at": None,
                "unrelated_streams_blocked": False,
                "updated_at": now,
            }
        )
        _write_json(self._delivery_retry_path(retry_state_id), retry_state)
        self.store.append_audit(
            "connector.delivery.retry_resolved",
            requested_scope,
            {"type": "connector_delivery_retry_state", "id": retry_state_id},
            {"delivery_id": delivery.get("delivery_id"), "artifact_id": artifact_id},
        )
        return retry_state

    def list_quarantine(self, requested_scope: dict[str, str]) -> dict[str, Any]:
        items: list[dict[str, Any]] = []
        if self.quarantine_dir.exists():
            for path in sorted(self.quarantine_dir.glob("*.json")):
                item = json.loads(path.read_text())
                if item.get("scope") == requested_scope:
                    items.append(item)
        listing = {
            "schema_version": QUARANTINE_LIST_SCHEMA,
            "scope": requested_scope,
            "status": "success",
            "quarantine_count": len(items),
            "open_quarantine_count": len([item for item in items if item.get("status") == "quarantined"]),
            "items": items,
            "checked_at": utc_now(),
        }
        audit_event = self.store.append_audit(
            "connector.quarantine.listed",
            requested_scope,
            {"type": "connector_quarantine_list", "id": "latest"},
            {"quarantine_count": listing["quarantine_count"], "open_quarantine_count": listing["open_quarantine_count"]},
        )
        listing["audit_refs"] = [f"audit:{audit_event['event_id']}"]
        return {"status": "success", "connector_quarantine_list": listing, "audit_event": audit_event}

    def replay_quarantine(self, quarantine_id: str, requested_scope: dict[str, str]) -> dict[str, Any]:
        quarantine = self._load_quarantine(quarantine_id)
        if quarantine is None:
            return {"status": "not_found", "quarantine_id": quarantine_id}
        if quarantine.get("scope") != requested_scope:
            return {"status": "scope_denied", "resource_scope": quarantine.get("scope")}
        retry_state = self._load_delivery_retry(str(quarantine.get("retry_state_id", ""))) or {}
        replay_attempts = list(quarantine.get("replay_attempts", []))
        attempt_id = _quarantine_replay_attempt_id(quarantine_id, len(replay_attempts) + 1)
        now = utc_now()
        replay_attempt = {
            "replay_attempt_id": attempt_id,
            "requested_at": now,
            "requested_by_owner_id": requested_scope["owner_id"],
            "status": "requested",
            "failure_evidence_preserved": True,
        }
        replay_attempts.append(replay_attempt)
        quarantine.update(
            {
                "status": "replay_requested",
                "replay_attempts": replay_attempts,
                "failure_evidence_preserved": True,
                "updated_at": now,
            }
        )
        _write_json(self._quarantine_path(quarantine_id), quarantine)
        if retry_state:
            retry_state.update({"status": "replay_requested", "updated_at": now, "quarantine_id": quarantine_id})
            _write_json(self._delivery_retry_path(str(retry_state["retry_state_id"])), retry_state)
        audit_event = self.store.append_audit(
            "connector.quarantine.replay_requested",
            requested_scope,
            {"type": "connector_delivery_quarantine", "id": quarantine_id},
            {
                "replay_attempt_id": attempt_id,
                "retry_state_id": quarantine.get("retry_state_id"),
                "failure_evidence_preserved": True,
                "raw_provider_payload_persisted": False,
            },
        )
        return {
            "status": "success",
            "delivery_quarantine": quarantine,
            "delivery_retry_state": retry_state,
            "replay_attempt": replay_attempt,
            "audit_event": audit_event,
        }

    def show_content_lineage(self, contract_id: str, source_external_id: str, requested_scope: dict[str, str]) -> dict[str, Any]:
        contract = self.load_contract(contract_id)
        if contract is None:
            return {"status": "not_found", "contract_id": contract_id}
        if contract.get("scope") != requested_scope:
            return {"status": "scope_denied", "resource_scope": contract.get("scope")}

        content_source_key = _content_source_key(requested_scope, contract["contract_version_id"], source_external_id)
        current_record = self._load_content_current(content_source_key)
        if current_record is None:
            return {"status": "not_found", "source_external_id": source_external_id}

        versions: list[dict[str, Any]] = []
        if self.content_version_dir.exists():
            for path in sorted(self.content_version_dir.glob("*.json")):
                version = json.loads(path.read_text())
                if version.get("scope") != requested_scope:
                    continue
                if version.get("contract_version_id") != contract["contract_version_id"]:
                    continue
                if version.get("source_external_id") != source_external_id:
                    continue
                versions.append(version)
        versions.sort(key=lambda item: int(item.get("version_ordinal", 0)))
        current_version_id = current_record.get("current_content_version_id")
        current_versions = [version for version in versions if version.get("content_version_id") == current_version_id]
        duplicate_active_truth_count = 0 if len(current_versions) == 1 and current_record.get("one_current_logical_truth") is True else 1
        historical_mutation_count = len([version for version in versions if version.get("historical_evidence_mutated") is not False])
        audit_event = self.store.append_audit(
            "connector.content_lineage.queried",
            requested_scope,
            {"type": "connector_content_current", "id": content_source_key},
            {
                "contract_id": contract_id,
                "source_external_id": source_external_id,
                "version_count": len(versions),
                "current_content_version_id": current_version_id,
                "duplicate_active_truth_count": duplicate_active_truth_count,
                "historical_mutation_count": historical_mutation_count,
            },
        )
        lineage = {
            "schema_version": CONTENT_LINEAGE_SCHEMA,
            "content_source_key": content_source_key,
            "scope": requested_scope,
            "contract_id": contract["contract_id"],
            "contract_version_id": contract["contract_version_id"],
            "source_external_id": source_external_id,
            "current_content_version_id": current_version_id,
            "current_artifact_id": current_record.get("current_artifact_id"),
            "current_source_revision": current_record.get("current_source_revision"),
            "version_count": len(versions),
            "versions": versions,
            "one_current_logical_truth": duplicate_active_truth_count == 0,
            "duplicate_active_truth_count": duplicate_active_truth_count,
            "historical_evidence_mutation_count": historical_mutation_count,
            "audit_refs": [f"audit:{audit_event['event_id']}"],
            "created_at": utc_now(),
        }
        return {
            "status": "success",
            "contract": contract,
            "connector_content_lineage": lineage,
            "connector_content_current": current_record,
            "audit_event": audit_event,
        }

    def process_projection_delivery(
        self,
        delivery: dict[str, Any],
        requested_scope: dict[str, str],
        source_path: Path,
        contract_id: str,
        contract_version_id: str | None = None,
        fault_mode: str = "none",
        failure_mode: str = "none",
    ) -> dict[str, Any]:
        if fault_mode not in {"none", "before_commit", "after_commit_before_ack", "after_ack"}:
            return {
                "status": "failed",
                "issues": [
                    {
                        "code": "CS_CONNECTOR_DELIVERY_FAULT_MODE_INVALID",
                        "message": "fault_mode must be none, before_commit, after_commit_before_ack, or after_ack.",
                        "path": "fault_mode",
                    }
                ],
            }
        if failure_mode not in {"none", "transient"}:
            return {
                "status": "failed",
                "issues": [
                    {
                        "code": "CS_CONNECTOR_DELIVERY_FAILURE_MODE_INVALID",
                        "message": "failure_mode must be none or transient.",
                        "path": "failure_mode",
                    }
                ],
            }

        selected_contract_id = contract_id or delivery.get("contract_id")
        contract = self.load_contract(selected_contract_id, contract_version_id=contract_version_id)
        if contract is None:
            return {"status": "not_found", "contract_id": selected_contract_id}
        if contract.get("scope") != requested_scope:
            return {"status": "scope_denied", "resource_scope": contract.get("scope")}

        provider_pack_id = str(delivery.get("provider_pack_id", ""))
        setup_result = self._latest_setup_result(contract, requested_scope, provider_pack_id)
        if setup_result is None:
            return {
                "status": "setup_missing",
                "contract": contract,
                "issues": [
                    {
                        "code": "CS_CONNECTOR_SETUP_NOT_FOUND",
                        "message": "A scoped Setup Result must exist before Projection Delivery processing.",
                        "path": "setup_result",
                    }
                ],
            }

        issues = self._delivery_validation_issues(delivery, contract, setup_result, requested_scope)
        if issues:
            return self._record_delivery_failure(
                delivery,
                requested_scope,
                contract,
                setup_result,
                "poison",
                issues,
            )
        if failure_mode == "transient":
            return self._record_delivery_failure(
                delivery,
                requested_scope,
                contract,
                setup_result,
                "transient",
                [
                    {
                        "code": "CS_CONNECTOR_DELIVERY_TRANSIENT_FAILURE",
                        "message": "Simulated transient connector/provider failure; retry was scheduled by bounded policy.",
                        "path": "delivery",
                    }
                ],
            )

        if fault_mode == "before_commit":
            return {
                "status": "interrupted",
                "crash_point": "before_commit",
                "contract": contract,
                "setup_result": setup_result,
                "acknowledgement": {
                    "ack_sent": False,
                    "acknowledged_without_artifact": False,
                    "durable_commit_completed": False,
                    "reason": "simulated_crash_before_archive_commit",
                },
                "audit_events": [],
                "errors": [
                    {
                        "code": "CS_CONNECTOR_DELIVERY_SIMULATED_CRASH_BEFORE_COMMIT",
                        "message": "Simulated crash before durable archive commit; no ack was sent.",
                    }
                ],
            }

        archive_result = self.ingest_projection_delivery(
            delivery,
            requested_scope,
            source_path,
            contract_id,
            contract_version_id=contract_version_id,
        )
        if archive_result.get("status") != "success":
            return archive_result

        artifact = archive_result["artifact"]
        receipt = dict(archive_result["delivery_receipt"])
        projection_snapshot = archive_result["projection_snapshot"]
        evidence_link = archive_result["evidence_link"]
        idempotency_key = _delivery_idempotency_key(requested_scope, contract["contract_version_id"], delivery)
        ack_outbox_id = _ack_outbox_id(receipt["delivery_receipt_id"])
        existing_outbox = self._load_ack_outbox(ack_outbox_id)
        now = utc_now()
        outbox = existing_outbox or {
            "schema_version": ACK_OUTBOX_SCHEMA,
            "ack_outbox_id": ack_outbox_id,
            "delivery_receipt_id": receipt["delivery_receipt_id"],
            "delivery_id": delivery["delivery_id"],
            "projection_id": delivery["projection_id"],
            "scope": requested_scope,
            "contract_id": contract["contract_id"],
            "contract_version_id": contract["contract_version_id"],
            "setup_result_id": archive_result["setup_result"]["setup_result_id"],
            "source_policy_id": archive_result["source_policy"]["source_policy_id"],
            "artifact_id": artifact["artifact_id"],
            "artifact_checksum_sha256": artifact["checksum_sha256"],
            "idempotency_key": idempotency_key,
            "status": "pending",
            "ack_sent": False,
            "attempts": [],
            "created_at": now,
        }
        outbox.update(
            {
                "artifact_id": artifact["artifact_id"],
                "artifact_checksum_sha256": artifact["checksum_sha256"],
                "idempotency_key": idempotency_key,
                "updated_at": now,
            }
        )
        _write_json(self._ack_outbox_path(ack_outbox_id), outbox)

        receipt.update(
            {
                "idempotency_key": idempotency_key,
                "ack_outbox_id": ack_outbox_id,
                "acknowledgement_state": "pending_after_commit",
                "ack_timeline": [
                    {"state": "archive_committed", "at": now},
                    {"state": "ack_outbox_pending", "at": now},
                ],
            }
        )
        _write_json(self._delivery_receipt_path(receipt["delivery_receipt_id"]), receipt)
        outbox_event = self.store.append_audit(
            "connector.delivery.ack_outbox.created",
            requested_scope,
            {"type": "connector_ack_outbox", "id": ack_outbox_id},
            {
                "delivery_receipt_id": receipt["delivery_receipt_id"],
                "artifact_id": artifact["artifact_id"],
                "idempotency_key": idempotency_key,
                "ack_sent": False,
            },
        )
        audit_events = list(archive_result.get("audit_events", [])) + [outbox_event]

        if fault_mode == "after_commit_before_ack":
            return {
                "status": "interrupted",
                "crash_point": "after_commit_before_ack",
                "contract": contract,
                "setup_result": archive_result["setup_result"],
                "source_policy": archive_result["source_policy"],
                "artifact": artifact,
                "delivery_receipt": receipt,
                "projection_snapshot": projection_snapshot,
                "evidence_link": evidence_link,
                "ack_outbox": outbox,
                "audit_event": outbox_event,
                "audit_events": audit_events,
                "deduplicated": archive_result.get("deduplicated", False),
                "connector_delivery_dedupe_state": archive_result.get("connector_delivery_dedupe_state"),
                "connector_content_version": archive_result.get("connector_content_version"),
                "connector_content_current": archive_result.get("connector_content_current"),
                "connector_projection_policy_decision": archive_result.get("connector_projection_policy_decision"),
                "connector_content_restriction_decision": archive_result.get("connector_content_restriction_decision"),
                "connector_untrusted_content_review": archive_result.get("connector_untrusted_content_review"),
                "acknowledgement": {
                    "ack_sent": False,
                    "acknowledged_without_artifact": False,
                    "durable_commit_completed": True,
                    "reason": "simulated_crash_after_archive_commit_before_ack",
                },
                "errors": [
                    {
                        "code": "CS_CONNECTOR_DELIVERY_SIMULATED_CRASH_AFTER_COMMIT_BEFORE_ACK",
                        "message": "Simulated crash after durable archive commit and before ack send; pending outbox remains replayable.",
                    }
                ],
            }

        already_acknowledged = outbox.get("status") == "acknowledged" and outbox.get("ack_sent") is True
        if already_acknowledged:
            ack_event = self.store.append_audit(
                "connector.delivery.ack_replayed",
                requested_scope,
                {"type": "connector_ack_outbox", "id": ack_outbox_id},
                {
                    "delivery_receipt_id": receipt["delivery_receipt_id"],
                    "artifact_id": artifact["artifact_id"],
                    "idempotency_key": idempotency_key,
                    "duplicate_downstream_effect": False,
                },
            )
            audit_events.append(ack_event)
            resolved_retry_state = self._mark_delivery_retry_resolved(
                delivery,
                requested_scope,
                contract,
                artifact["artifact_id"],
            )
            receipt.update(
                {
                    "acknowledgement_state": "already_acknowledged_after_commit",
                    "ack_sent_at": outbox.get("acknowledged_at"),
                    "ack_timeline": [
                        {"state": "archive_committed", "at": receipt.get("created_at", now)},
                        {"state": "ack_already_sent", "at": outbox.get("acknowledged_at", now)},
                        {"state": "redelivery_reconciled", "at": utc_now()},
                    ],
                }
            )
            _write_json(self._delivery_receipt_path(receipt["delivery_receipt_id"]), receipt)
            return {
                "status": "success",
                "contract": contract,
                "setup_result": archive_result["setup_result"],
                "source_policy": archive_result["source_policy"],
                "artifact": artifact,
                "delivery_receipt": receipt,
                "projection_snapshot": projection_snapshot,
                "evidence_link": evidence_link,
                "ack_outbox": outbox,
                "audit_event": ack_event,
                "audit_events": audit_events,
                "deduplicated": archive_result.get("deduplicated", False),
                "connector_delivery_dedupe_state": archive_result.get("connector_delivery_dedupe_state"),
                "connector_content_version": archive_result.get("connector_content_version"),
                "connector_content_current": archive_result.get("connector_content_current"),
                "connector_projection_policy_decision": archive_result.get("connector_projection_policy_decision"),
                "connector_content_restriction_decision": archive_result.get("connector_content_restriction_decision"),
                "connector_untrusted_content_review": archive_result.get("connector_untrusted_content_review"),
                "acknowledgement": {
                    "ack_sent": True,
                    "acknowledged_without_artifact": False,
                    "durable_commit_completed": True,
                    "replayed": True,
                    "duplicate_downstream_effect": False,
                },
                "connector_delivery_retry_state": resolved_retry_state,
            }

        ack_at = utc_now()
        attempts = list(outbox.get("attempts", []))
        attempts.append(
            {
                "attempt_no": len(attempts) + 1,
                "sent_at": ack_at,
                "result": "acknowledged",
                "transport": "local_fixture_connectorhub_ack",
            }
        )
        outbox.update(
            {
                "status": "acknowledged",
                "ack_sent": True,
                "acknowledged_at": ack_at,
                "attempts": attempts,
                "updated_at": ack_at,
            }
        )
        _write_json(self._ack_outbox_path(ack_outbox_id), outbox)
        receipt.update(
            {
                "acknowledgement_state": "acknowledged_after_commit",
                "ack_sent_at": ack_at,
                "ack_timeline": [
                    {"state": "archive_committed", "at": receipt.get("created_at", now)},
                    {"state": "ack_outbox_pending", "at": now},
                    {"state": "ack_sent_after_commit", "at": ack_at},
                ],
            }
        )
        _write_json(self._delivery_receipt_path(receipt["delivery_receipt_id"]), receipt)
        ack_event = self.store.append_audit(
            "connector.delivery.acknowledged",
            requested_scope,
            {"type": "connector_ack_outbox", "id": ack_outbox_id},
            {
                "delivery_receipt_id": receipt["delivery_receipt_id"],
                "artifact_id": artifact["artifact_id"],
                "idempotency_key": idempotency_key,
                "ack_sent_after_commit": True,
                "duplicate_downstream_effect": False,
            },
        )
        audit_events.append(ack_event)
        resolved_retry_state = self._mark_delivery_retry_resolved(
            delivery,
            requested_scope,
            contract,
            artifact["artifact_id"],
        )
        response_status = "interrupted" if fault_mode == "after_ack" else "success"
        response = {
            "status": response_status,
            "contract": contract,
            "setup_result": archive_result["setup_result"],
            "source_policy": archive_result["source_policy"],
            "artifact": artifact,
            "delivery_receipt": receipt,
            "projection_snapshot": projection_snapshot,
            "evidence_link": evidence_link,
            "ack_outbox": outbox,
            "audit_event": ack_event,
            "audit_events": audit_events,
            "deduplicated": archive_result.get("deduplicated", False),
            "connector_delivery_dedupe_state": archive_result.get("connector_delivery_dedupe_state"),
            "connector_content_version": archive_result.get("connector_content_version"),
            "connector_content_current": archive_result.get("connector_content_current"),
            "connector_projection_policy_decision": archive_result.get("connector_projection_policy_decision"),
            "connector_content_restriction_decision": archive_result.get("connector_content_restriction_decision"),
            "connector_untrusted_content_review": archive_result.get("connector_untrusted_content_review"),
            "acknowledgement": {
                "ack_sent": True,
                "acknowledged_without_artifact": False,
                "durable_commit_completed": True,
                "replayed": False,
                "duplicate_downstream_effect": False,
            },
            "connector_delivery_retry_state": resolved_retry_state,
        }
        if fault_mode == "after_ack":
            response["crash_point"] = "after_ack"
            response["errors"] = [
                {
                    "code": "CS_CONNECTOR_DELIVERY_SIMULATED_CRASH_AFTER_ACK",
                    "message": "Simulated crash after ack send; replay must reconcile without duplicate downstream effects.",
                }
            ]
        return response

    def _record_sync_signal_receipt(
        self,
        *,
        delivery: dict[str, Any],
        requested_scope: dict[str, str],
        contract: dict[str, Any],
        setup_result: dict[str, Any] | None,
        signal: str,
        cursor_id: str,
        cursor_value: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        signal_receipt_id = _sync_signal_receipt_id(requested_scope, contract["contract_id"], delivery, signal)
        sync_metadata = _sync_metadata(delivery)
        origin_verified = sync_metadata.get("webhook_origin_verified") is True if signal == "webhook" else False
        signature_verified = sync_metadata.get("webhook_signature_verified") is True if signal == "webhook" else False
        signal_receipt = {
            "schema_version": SYNC_SIGNAL_RECEIPT_SCHEMA,
            "signal_receipt_id": signal_receipt_id,
            "scope": requested_scope,
            "contract_id": contract["contract_id"],
            "contract_version_id": contract["contract_version_id"],
            "setup_result_id": (setup_result or {}).get("setup_result_id"),
            "signal": signal,
            "origin_verified_inside_connector_boundary": origin_verified,
            "webhook_signature_verified": signature_verified,
            "poll_cursor_source": signal == "poll",
            "delivery_id": delivery.get("delivery_id"),
            "projection_id": delivery.get("projection_id"),
            "provider_pack_id": delivery.get("provider_pack_id"),
            "provider_event_id": delivery.get("provider_event_id"),
            "provider_event_key": _sync_provider_event_key(delivery),
            "provider_event_key_parts": _sync_provider_event_key_parts(delivery),
            "source_external_id": _source_external_id(delivery),
            "source_revision": _source_revision(delivery),
            "source_content_hash": _source_content_hash(delivery),
            "cursor_id": cursor_id,
            "cursor_value": cursor_value,
            "raw_provider_payload_persisted": False,
            "external_http_calls": 0,
            "created_at": utc_now(),
        }
        audit_event = self.store.append_audit(
            "connector.sync.webhook_received" if signal == "webhook" else "connector.sync.poll_observed",
            requested_scope,
            {"type": "connector_sync_signal_receipt", "id": signal_receipt_id},
            {
                "signal": signal,
                "delivery_id": delivery.get("delivery_id"),
                "provider_event_key": signal_receipt["provider_event_key"],
                "cursor_id": cursor_id,
                "cursor_value": cursor_value,
                "origin_verified_inside_connector_boundary": signal_receipt["origin_verified_inside_connector_boundary"],
                "raw_provider_payload_persisted": False,
            },
        )
        signal_receipt["audit_refs"] = [f"audit:{audit_event['event_id']}"]
        _write_json(self._sync_signal_receipt_path(signal_receipt_id), signal_receipt)
        return signal_receipt, audit_event

    def _advance_sync_cursor(
        self,
        *,
        delivery: dict[str, Any],
        requested_scope: dict[str, str],
        contract: dict[str, Any],
        setup_result: dict[str, Any],
        signal: str,
        cursor_id: str,
        cursor_value: str,
        process_result: dict[str, Any],
        signal_receipt: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        cursor_storage_id = _sync_cursor_storage_id(requested_scope, contract["contract_id"], cursor_id)
        existing = self._load_sync_cursor(cursor_storage_id)
        receipt = process_result["delivery_receipt"]
        artifact = process_result["artifact"]
        content_version = process_result.get("connector_content_version", {})
        provider_event_key = _sync_provider_event_key(delivery)
        previous_value = str((existing or {}).get("cursor_value") or "")
        monotonic = not previous_value or cursor_value >= previous_value
        advanced = not previous_value or cursor_value > previous_value
        if not monotonic:
            reason = "out_of_order_source_revision_observed"
        elif not advanced:
            reason = "duplicate_or_replay_observed"
        else:
            reason = "advanced_after_durable_commit"
        observation = {
            "signal": signal,
            "delivery_id": delivery.get("delivery_id"),
            "delivery_receipt_id": receipt["delivery_receipt_id"],
            "artifact_id": artifact["artifact_id"],
            "content_version_id": content_version.get("content_version_id"),
            "provider_event_id": delivery.get("provider_event_id"),
            "provider_event_key": provider_event_key,
            "source_external_id": receipt.get("source_external_id"),
            "source_revision": receipt.get("source_revision"),
            "source_content_hash": receipt.get("source_content_hash"),
            "cursor_value": cursor_value,
            "previous_cursor_value": previous_value or None,
            "advanced": advanced,
            "monotonic": monotonic,
            "reason": reason,
            "durable_commit_completed": True,
            "cursor_advanced_before_commit": False,
            "signal_receipt_id": (signal_receipt or {}).get("signal_receipt_id"),
            "observed_at": utc_now(),
        }
        history = list((existing or {}).get("history", []))
        history.append(observation)
        processed_receipt_ids = list((existing or {}).get("processed_delivery_receipt_ids", []))
        if receipt["delivery_receipt_id"] not in processed_receipt_ids:
            processed_receipt_ids.append(receipt["delivery_receipt_id"])
        provider_event_keys = list((existing or {}).get("provider_event_keys", []))
        if provider_event_key not in provider_event_keys:
            provider_event_keys.append(provider_event_key)
        signals_seen = list((existing or {}).get("signals_seen", []))
        if signal not in signals_seen:
            signals_seen.append(signal)
        cursor = {
            "schema_version": SYNC_CURSOR_SCHEMA,
            "cursor_storage_id": cursor_storage_id,
            "cursor_id": cursor_id,
            "scope": requested_scope,
            "contract_id": contract["contract_id"],
            "contract_version_id": contract["contract_version_id"],
            "setup_result_id": setup_result["setup_result_id"],
            "source_ref": receipt.get("source_summary", {}).get("source_ref"),
            "projection_type": receipt.get("projection_type"),
            "cursor_value": cursor_value if advanced or not existing else previous_value,
            "last_advanced_cursor_value": cursor_value if advanced or not existing else (existing or {}).get("last_advanced_cursor_value"),
            "last_source_revision": receipt.get("source_revision") if advanced or not existing else (existing or {}).get("last_source_revision"),
            "last_provider_event_key": provider_event_key if advanced or not existing else (existing or {}).get("last_provider_event_key"),
            "last_delivery_receipt_id": receipt["delivery_receipt_id"] if advanced or not existing else (existing or {}).get("last_delivery_receipt_id"),
            "processed_delivery_receipt_ids": processed_receipt_ids,
            "provider_event_keys": provider_event_keys,
            "signals_seen": signals_seen,
            "history": history,
            "history_count": len(history),
            "advanced_after_durable_commit_count": len([item for item in history if item.get("advanced") is True and item.get("durable_commit_completed") is True]),
            "duplicate_or_replay_count": len([item for item in history if item.get("reason") == "duplicate_or_replay_observed"]),
            "out_of_order_count": len([item for item in history if item.get("reason") == "out_of_order_source_revision_observed"]),
            "cursor_advanced_before_commit_count": 0,
            "raw_provider_payload_persisted": False,
            "updated_at": utc_now(),
        }
        if existing and existing.get("created_at"):
            cursor["created_at"] = existing["created_at"]
        else:
            cursor["created_at"] = cursor["updated_at"]
        audit_event = self.store.append_audit(
            "connector.sync.cursor_advanced" if advanced else "connector.sync.cursor_observed",
            requested_scope,
            {"type": "connector_sync_cursor", "id": cursor_storage_id},
            {
                "cursor_id": cursor_id,
                "cursor_value": cursor["cursor_value"],
                "observed_cursor_value": cursor_value,
                "advanced": advanced,
                "monotonic": monotonic,
                "reason": reason,
                "delivery_receipt_id": receipt["delivery_receipt_id"],
                "artifact_id": artifact["artifact_id"],
                "cursor_advanced_before_commit": False,
            },
        )
        cursor["audit_refs"] = list(dict.fromkeys([*(existing or {}).get("audit_refs", []), f"audit:{audit_event['event_id']}"]))
        _write_json(self._sync_cursor_path(cursor_storage_id), cursor)
        return cursor, audit_event

    def process_incremental_sync(
        self,
        delivery: dict[str, Any],
        requested_scope: dict[str, str],
        source_path: Path,
        contract_id: str,
        *,
        signal: str,
        cursor_id: str,
        cursor_value: str | None = None,
        contract_version_id: str | None = None,
        sync_fault_mode: str = "none",
    ) -> dict[str, Any]:
        if signal not in {"webhook", "poll"}:
            return {
                "status": "failed",
                "issues": [
                    {
                        "code": "CS_CONNECTOR_SYNC_SIGNAL_INVALID",
                        "message": "signal must be webhook or poll.",
                        "path": "signal",
                    }
                ],
            }
        if sync_fault_mode not in {"none", "before_commit", "after_commit_before_cursor", "after_cursor"}:
            return {
                "status": "failed",
                "issues": [
                    {
                        "code": "CS_CONNECTOR_SYNC_FAULT_MODE_INVALID",
                        "message": "fault_mode must be none, before_commit, after_commit_before_cursor, or after_cursor.",
                        "path": "fault_mode",
                    }
                ],
            }
        selected_contract_id = contract_id or delivery.get("contract_id")
        contract = self.load_contract(selected_contract_id, contract_version_id=contract_version_id)
        if contract is None:
            return {"status": "not_found", "contract_id": selected_contract_id}
        if contract.get("scope") != requested_scope:
            return {"status": "scope_denied", "resource_scope": contract.get("scope")}
        provider_pack_id = str(delivery.get("provider_pack_id", ""))
        setup_result = self._latest_setup_result(contract, requested_scope, provider_pack_id)
        resolved_cursor_value = cursor_value or _source_cursor_value(delivery)
        signal_receipt: dict[str, Any] | None = None
        signal_event: dict[str, Any] | None = None
        if signal == "webhook":
            signal_receipt, signal_event = self._record_sync_signal_receipt(
                delivery=delivery,
                requested_scope=requested_scope,
                contract=contract,
                setup_result=setup_result,
                signal=signal,
                cursor_id=cursor_id,
                cursor_value=resolved_cursor_value,
            )
            if (
                signal_receipt.get("origin_verified_inside_connector_boundary") is not True
                or signal_receipt.get("webhook_signature_verified") is not True
            ):
                issues = []
                if signal_receipt.get("origin_verified_inside_connector_boundary") is not True:
                    issues.append(
                        {
                            "code": "CS_CONNECTOR_WEBHOOK_ORIGIN_UNVERIFIED",
                            "message": "Webhook origin was not verified inside the Connector boundary.",
                            "path": "sync.webhook_origin_verified",
                        }
                    )
                if signal_receipt.get("webhook_signature_verified") is not True:
                    issues.append(
                        {
                            "code": "CS_CONNECTOR_WEBHOOK_SIGNATURE_INVALID",
                            "message": "Webhook signature was not verified inside the Connector boundary.",
                            "path": "sync.webhook_signature_verified",
                        }
                    )
                return {
                    "status": "failed",
                    "issues": issues,
                    "connector_sync_signal_receipt": signal_receipt,
                    "sync_signal": signal,
                    "sync_cursor_id": cursor_id,
                    "sync_cursor_value": resolved_cursor_value,
                    "sync_provider_event_key": _sync_provider_event_key(delivery),
                    "cursor_update": {
                        "status": "not_advanced",
                        "cursor_id": cursor_id,
                        "cursor_value": resolved_cursor_value,
                        "reason": "webhook_boundary_verification_failed",
                        "durable_commit_completed": False,
                        "cursor_advanced_before_commit": False,
                    },
                    "audit_events": [signal_event] if signal_event else [],
                }
        delivery_fault_mode = "before_commit" if sync_fault_mode == "before_commit" else "none"
        process_result = self.process_projection_delivery(
            delivery,
            requested_scope,
            source_path,
            contract_id,
            contract_version_id=contract_version_id,
            fault_mode=delivery_fault_mode,
        )
        audit_events = list(process_result.get("audit_events", []))
        if signal_event:
            audit_events = [signal_event, *audit_events]
        process_result["audit_events"] = audit_events
        if signal_receipt:
            process_result["connector_sync_signal_receipt"] = signal_receipt
        process_result["sync_signal"] = signal
        process_result["sync_cursor_id"] = cursor_id
        process_result["sync_cursor_value"] = resolved_cursor_value
        process_result["sync_provider_event_key"] = _sync_provider_event_key(delivery)
        if process_result.get("status") != "success":
            if sync_fault_mode == "before_commit":
                process_result["cursor_update"] = {
                    "status": "not_advanced",
                    "cursor_id": cursor_id,
                    "cursor_value": resolved_cursor_value,
                    "reason": "durable_commit_not_completed",
                    "cursor_advanced_before_commit": False,
                }
            return process_result
        if sync_fault_mode == "after_commit_before_cursor":
            process_result["status"] = "interrupted"
            process_result["crash_point"] = "after_commit_before_cursor"
            process_result["cursor_update"] = {
                "status": "not_advanced",
                "cursor_id": cursor_id,
                "cursor_value": resolved_cursor_value,
                "reason": "simulated_crash_after_commit_before_cursor",
                "durable_commit_completed": True,
                "cursor_advanced_before_commit": False,
            }
            process_result.setdefault("errors", []).append(
                {
                    "code": "CS_CONNECTOR_SYNC_SIMULATED_CRASH_BEFORE_CURSOR",
                    "message": "Simulated crash after durable delivery commit and before cursor advancement.",
                }
            )
            return process_result
        cursor, cursor_event = self._advance_sync_cursor(
            delivery=delivery,
            requested_scope=requested_scope,
            contract=contract,
            setup_result=process_result["setup_result"],
            signal=signal,
            cursor_id=cursor_id,
            cursor_value=resolved_cursor_value,
            process_result=process_result,
            signal_receipt=signal_receipt,
        )
        process_result["connector_sync_cursor"] = cursor
        process_result["cursor_update"] = {
            "status": "advanced" if cursor["history"][-1]["advanced"] else "observed",
            "cursor_id": cursor_id,
            "cursor_value": cursor["cursor_value"],
            "observed_cursor_value": resolved_cursor_value,
            "reason": cursor["history"][-1]["reason"],
            "durable_commit_completed": True,
            "cursor_advanced_before_commit": False,
        }
        process_result["audit_events"] = [*process_result.get("audit_events", []), cursor_event]
        if sync_fault_mode == "after_cursor":
            process_result["status"] = "interrupted"
            process_result["crash_point"] = "after_cursor"
            process_result.setdefault("errors", []).append(
                {
                    "code": "CS_CONNECTOR_SYNC_SIMULATED_CRASH_AFTER_CURSOR",
                    "message": "Simulated crash after cursor advancement; replay must reconcile without duplicate active truth.",
                }
            )
        return process_result

    def reconcile_incremental_sync(self, requested_scope: dict[str, str], cursor_id: str | None = None) -> dict[str, Any]:
        cursors: list[dict[str, Any]] = []
        if self.sync_cursor_dir.exists():
            for path in sorted(self.sync_cursor_dir.glob("*.json")):
                cursor = json.loads(path.read_text())
                if cursor.get("scope") == requested_scope and (cursor_id is None or cursor.get("cursor_id") == cursor_id):
                    cursors.append(cursor)
        receipts: list[dict[str, Any]] = []
        if self.delivery_receipt_dir.exists():
            for path in sorted(self.delivery_receipt_dir.glob("*.json")):
                receipt = json.loads(path.read_text())
                if receipt.get("scope") == requested_scope:
                    receipts.append(receipt)
        dedupe_states: list[dict[str, Any]] = []
        if self.delivery_dedupe_dir.exists():
            for path in sorted(self.delivery_dedupe_dir.glob("*.json")):
                dedupe = json.loads(path.read_text())
                if dedupe.get("scope") == requested_scope:
                    dedupe_states.append(dedupe)
        artifact_ids = {str(receipt.get("artifact_id")) for receipt in receipts if receipt.get("artifact_id")}
        artifacts_by_id = {
            artifact_id: self.store.get_artifact(artifact_id, requested_scope)
            for artifact_id in artifact_ids
        }
        artifacts_by_id = {artifact_id: artifact for artifact_id, artifact in artifacts_by_id.items() if artifact}
        receipt_by_id = {receipt.get("delivery_receipt_id"): receipt for receipt in receipts}
        cursor_history = [item for cursor in cursors for item in cursor.get("history", [])]
        cursor_delivery_receipt_ids = {
            item.get("delivery_receipt_id")
            for item in cursor_history
            if item.get("delivery_receipt_id")
        }
        missing_receipt = [
            item
            for item in cursor_history
            if item.get("delivery_receipt_id") not in receipt_by_id
        ]
        missing_artifact = [
            item
            for item in cursor_history
            if item.get("artifact_id") not in artifacts_by_id
        ]
        unobserved_delivery_receipts = [
            receipt
            for receipt in receipts
            if receipt.get("delivery_receipt_id") not in cursor_delivery_receipt_ids
        ]
        duplicate_artifacts_by_key: dict[str, set[str]] = {}
        for receipt in receipts:
            key = str(receipt.get("idempotency_key") or receipt.get("delivery_idempotency_key") or "")
            artifact_id = str(receipt.get("artifact_id") or "")
            if key and artifact_id:
                duplicate_artifacts_by_key.setdefault(key, set()).add(artifact_id)
        duplicate_logical_artifacts = {
            key: sorted(values)
            for key, values in duplicate_artifacts_by_key.items()
            if len(values) > 1
        }
        cursor_advanced_before_commit_count = len(
            [
                item
                for item in cursor_history
                if item.get("cursor_advanced_before_commit") is True
                or item.get("durable_commit_completed") is not True
            ]
        )
        reconciliation_id = _sync_reconciliation_id(requested_scope, cursor_id)
        reconciliation = {
            "schema_version": SYNC_RECONCILIATION_SCHEMA,
            "sync_reconciliation_id": reconciliation_id,
            "scope": requested_scope,
            "cursor_id": cursor_id,
            "status": "success"
            if not missing_receipt
            and not missing_artifact
            and not unobserved_delivery_receipts
            and not duplicate_logical_artifacts
            and cursor_advanced_before_commit_count == 0
            else "failed",
            "cursor_count": len(cursors),
            "cursor_observation_count": len(cursor_history),
            "delivery_receipt_count": len(receipts),
            "artifact_count": len(artifacts_by_id),
            "dedupe_record_count": len(dedupe_states),
            "missing_cursor_receipt_count": len(missing_receipt),
            "missing_cursor_artifact_count": len(missing_artifact),
            "unobserved_delivery_receipt_count": len(unobserved_delivery_receipts),
            "unobserved_delivery_receipt_ids": [
                str(receipt.get("delivery_receipt_id"))
                for receipt in unobserved_delivery_receipts
                if receipt.get("delivery_receipt_id")
            ],
            "duplicate_logical_artifact_count": len(duplicate_logical_artifacts),
            "duplicate_logical_artifacts": duplicate_logical_artifacts,
            "cursor_advanced_before_commit_count": cursor_advanced_before_commit_count,
            "duplicate_or_replay_observation_count": len([item for item in cursor_history if item.get("reason") == "duplicate_or_replay_observed"]),
            "out_of_order_observation_count": len([item for item in cursor_history if item.get("reason") == "out_of_order_source_revision_observed"]),
            "signals_seen": sorted({signal for cursor in cursors for signal in cursor.get("signals_seen", [])}),
            "provider_event_keys": sorted({key for cursor in cursors for key in cursor.get("provider_event_keys", [])}),
            "checked_at": utc_now(),
        }
        reconciliation["sync_lag_metrics"] = {
            "unobserved_delivery_receipt_count": reconciliation["unobserved_delivery_receipt_count"],
            "cursor_observation_count": reconciliation["cursor_observation_count"],
            "duplicate_or_replay_observation_count": reconciliation["duplicate_or_replay_observation_count"],
            "out_of_order_observation_count": reconciliation["out_of_order_observation_count"],
            "duplicate_logical_artifact_count": reconciliation["duplicate_logical_artifact_count"],
        }
        audit_event = self.store.append_audit(
            "connector.sync.cursor_reconciled",
            requested_scope,
            {"type": "connector_sync_reconciliation", "id": reconciliation_id},
            {
                "status": reconciliation["status"],
                "cursor_count": reconciliation["cursor_count"],
                "cursor_advanced_before_commit_count": cursor_advanced_before_commit_count,
                "duplicate_logical_artifact_count": reconciliation["duplicate_logical_artifact_count"],
            },
        )
        reconciliation["audit_refs"] = [f"audit:{audit_event['event_id']}"]
        _write_json(self._sync_reconciliation_path(reconciliation_id), reconciliation)
        return {"status": reconciliation["status"], "connector_sync_reconciliation": reconciliation, "connector_sync_cursors": cursors, "audit_event": audit_event}

    def _capture_negative_evidence(self) -> dict[str, int]:
        return dict(CAPTURE_NEGATIVE_EVIDENCE_TEMPLATE)

    def _chrome_active_tab_negative_evidence(self) -> dict[str, int]:
        return dict(CHROME_ACTIVE_TAB_NEGATIVE_EVIDENCE_TEMPLATE)

    def _chrome_auto_capture_negative_evidence(self) -> dict[str, int]:
        return dict(CHROME_AUTO_CAPTURE_NEGATIVE_EVIDENCE_TEMPLATE)

    def _chrome_sensitive_page_negative_evidence(self) -> dict[str, int]:
        return dict(CHROME_SENSITIVE_PAGE_NEGATIVE_EVIDENCE_TEMPLATE)

    def _latest_watch_source_consent(self, requested_scope: dict[str, str], source_id: str) -> dict[str, Any] | None:
        if not self.watch_source_consent_dir.exists():
            return None
        records: list[dict[str, Any]] = []
        for path in sorted(self.watch_source_consent_dir.glob("*.json")):
            record = json.loads(path.read_text())
            if record.get("scope") == requested_scope and record.get("source_id") == source_id:
                records.append(record)
        if not records:
            return None
        records.sort(key=lambda record: str(record.get("created_at") or ""))
        return records[-1]

    def _latest_chrome_auto_capture_config(
        self,
        requested_scope: dict[str, str],
        source_id: str,
    ) -> dict[str, Any] | None:
        if not self.chrome_auto_capture_config_dir.exists():
            return None
        records: list[dict[str, Any]] = []
        for path in sorted(self.chrome_auto_capture_config_dir.glob("*.json")):
            record = json.loads(path.read_text())
            if record.get("scope") == requested_scope and record.get("source_id") == source_id:
                records.append(record)
        if not records:
            return None
        records.sort(key=lambda record: str(record.get("created_at") or ""))
        return records[-1]

    def _latest_capture_permission_probe(
        self,
        requested_scope: dict[str, str],
        source_id: str,
        platform: str,
        permission_state: str,
    ) -> dict[str, Any] | None:
        if not self.capture_permission_probe_dir.exists():
            return None
        records: list[dict[str, Any]] = []
        for path in sorted(self.capture_permission_probe_dir.glob("*.json")):
            record = json.loads(path.read_text())
            if (
                record.get("scope") == requested_scope
                and record.get("source_id") == source_id
                and record.get("platform") == platform
                and record.get("platform_permission_state") == permission_state
            ):
                records.append(record)
        if not records:
            return None
        records.sort(key=lambda record: str(record.get("probed_at") or ""))
        return records[-1]

    def _capture_setup_diagnostics(self, requested_scope: dict[str, str], source_id: str) -> dict[str, Any]:
        return {
            "source_id": source_id,
            "source_kind": "watch_agent_macos_activity",
            "disabled_by_default": True,
            "data_categories_explained": [
                "frontmost_application_identifier_hash",
                "activity_timestamp",
                "idle_gap_marker",
                "session_boundary_hint",
            ],
            "excluded_categories": [
                "screenshots",
                "keystrokes",
                "clipboard_values",
                "cookies",
                "browser_history",
                "form_values",
                "raw_window_titles",
            ],
            "sampling_interval_explained": "bounded local activity samples only after consent and platform permission",
            "privacy_mode": "metadata_only_local_fixture",
            "retention": {"mode": "local_fixture", "default_days": 30, "owner_controls_visible": True},
            "namespace_scope": requested_scope,
            "pause_delete_explained": True,
            "physical_device_acceptance": "HUMAN_REQUIRED",
        }

    def probe_capture_permission(
        self,
        *,
        requested_scope: dict[str, str],
        source_id: str,
        platform: str,
        platform_permission_state: str,
    ) -> dict[str, Any]:
        if platform not in SUPPORTED_CAPTURE_PLATFORMS:
            return {
                "status": "failed",
                "issues": [
                    {
                        "code": "CS_CONNECTOR_CAPTURE_PLATFORM_UNSUPPORTED",
                        "message": "Capture permission probe supports only declared local fixture platforms.",
                        "path": "platform",
                    }
                ],
            }
        if platform_permission_state not in CAPTURE_PLATFORM_PERMISSION_STATES:
            return {
                "status": "failed",
                "issues": [
                    {
                        "code": "CS_CONNECTOR_CAPTURE_PERMISSION_STATE_INVALID",
                        "message": "platform_permission_state must be one of the supported fixture states.",
                        "path": "platform_permission_state",
                    }
                ],
            }

        probe_id = _capture_permission_probe_id(requested_scope, source_id, platform, platform_permission_state)
        supported_host = platform_permission_state != "unsupported_host"
        permission_active = platform_permission_state == "granted"
        probe = {
            "schema_version": CAPTURE_PERMISSION_PROBE_SCHEMA,
            "permission_probe_id": probe_id,
            "scope": requested_scope,
            "source_id": source_id,
            "platform": platform,
            "platform_supported": supported_host,
            "platform_permission_state": platform_permission_state,
            "permission_active_for_local_fixture": permission_active,
            "production_permission_proof": "HUMAN_REQUIRED",
            "permission_probe_only": True,
            "capture_enabled": False,
            "capture_attempted": False,
            "capture_samples_created": 0,
            "screenshots_created": 0,
            "raw_window_titles_collected": 0,
            "external_http_calls": 0,
            "provider_mutations": 0,
            "setup_diagnostics": self._capture_setup_diagnostics(requested_scope, source_id),
            "status_explanation": {
                "reason_code": "CS_CONNECTOR_CAPTURE_PERMISSION_GRANTED"
                if permission_active
                else (
                    "CS_CONNECTOR_CAPTURE_HOST_UNSUPPORTED"
                    if platform_permission_state == "unsupported_host"
                    else "CS_CONNECTOR_CAPTURE_PLATFORM_PERMISSION_MISSING"
                ),
                "cause": "This local fixture records platform permission state without attempting capture.",
                "impact": "Collection remains off until explicit consent and platform permission are both active.",
                "resolution_steps": [
                    "Review the data categories and privacy controls.",
                    "Grant explicit owner consent in CornerStone.",
                    "Grant the required macOS permission on a physical supported Mac before live capture.",
                ],
                "safe_to_show_to_owner": True,
            },
            "negative_evidence": self._capture_negative_evidence(),
            "evidence_refs": [f"connector_capture_permission_probe:{probe_id}"],
            "audit_refs": [],
            "probed_at": utc_now(),
        }
        audit_event = self.store.append_audit(
            "connector.capture.permission_probed",
            requested_scope,
            {"type": "connector_capture_permission_probe", "id": probe_id},
            {
                "source_id": source_id,
                "platform": platform,
                "platform_permission_state": platform_permission_state,
                "capture_attempted": False,
            },
        )
        probe["audit_refs"] = [f"audit:{audit_event['event_id']}"]
        _write_json(self._capture_permission_probe_path(probe_id), probe)
        return {"status": "success", "connector_capture_permission_probe": probe, "audit_event": audit_event}

    def record_watch_source_consent(
        self,
        *,
        requested_scope: dict[str, str],
        source_id: str,
        decision: str,
        purpose: str,
    ) -> dict[str, Any]:
        if decision not in CAPTURE_CONSENT_DECISIONS:
            return {
                "status": "failed",
                "issues": [
                    {
                        "code": "CS_CONNECTOR_CAPTURE_CONSENT_DECISION_INVALID",
                        "message": "consent decision must be granted, denied, or revoked.",
                        "path": "decision",
                    }
                ],
            }
        if not purpose.strip():
            return {
                "status": "failed",
                "issues": [
                    {
                        "code": "CS_CONNECTOR_CAPTURE_CONSENT_PURPOSE_REQUIRED",
                        "message": "A purpose is required before recording Watch source consent.",
                        "path": "purpose",
                    }
                ],
            }

        consent_record_id = _watch_source_consent_id(requested_scope, source_id, decision, purpose)
        active = decision == "granted"
        consent = {
            "schema_version": WATCH_SOURCE_CONSENT_SCHEMA,
            "watch_source_consent_id": consent_record_id,
            "scope": requested_scope,
            "source_id": source_id,
            "decision": decision,
            "active": active,
            "explicit_owner_consent": active,
            "purpose": purpose,
            "consent_scope": "owner_personal_activity_local_fixture",
            "platform_permission_required": True,
            "platform_permission_granted_by_this_record": False,
            "capture_enabled": False,
            "collection_started": False,
            "capture_attempted": False,
            "setup_diagnostics": self._capture_setup_diagnostics(requested_scope, source_id),
            "negative_evidence": self._capture_negative_evidence(),
            "evidence_refs": [f"connector_watch_source_consent:{consent_record_id}"],
            "audit_refs": [],
            "created_at": utc_now(),
        }
        audit_event = self.store.append_audit(
            "connector.capture.consent_recorded",
            requested_scope,
            {"type": "connector_watch_source_consent", "id": consent_record_id},
            {
                "source_id": source_id,
                "decision": decision,
                "explicit_owner_consent": active,
                "collection_started": False,
            },
        )
        consent["audit_refs"] = [f"audit:{audit_event['event_id']}"]
        _write_json(self._watch_source_consent_path(consent_record_id), consent)
        return {"status": "success", "connector_watch_source_consent": consent, "audit_event": audit_event}

    def evaluate_capture_guard(
        self,
        *,
        requested_scope: dict[str, str],
        source_id: str,
        platform: str,
        platform_permission_state: str,
    ) -> dict[str, Any]:
        if platform not in SUPPORTED_CAPTURE_PLATFORMS:
            return {
                "status": "failed",
                "issues": [
                    {
                        "code": "CS_CONNECTOR_CAPTURE_PLATFORM_UNSUPPORTED",
                        "message": "Capture guard supports only declared local fixture platforms.",
                        "path": "platform",
                    }
                ],
            }
        if platform_permission_state not in CAPTURE_PLATFORM_PERMISSION_STATES:
            return {
                "status": "failed",
                "issues": [
                    {
                        "code": "CS_CONNECTOR_CAPTURE_PERMISSION_STATE_INVALID",
                        "message": "platform_permission_state must be one of the supported fixture states.",
                        "path": "platform_permission_state",
                    }
                ],
            }

        consent = self._latest_watch_source_consent(requested_scope, source_id)
        permission_probe = self._latest_capture_permission_probe(
            requested_scope,
            source_id,
            platform,
            platform_permission_state,
        )
        consent_active = bool(consent and consent.get("active") is True and consent.get("explicit_owner_consent") is True)
        platform_supported = platform_permission_state != "unsupported_host"
        platform_permission_active = platform_permission_state == "granted"
        capture_enabled = consent_active and platform_supported and platform_permission_active
        reason_codes: list[str] = []
        if not consent_active:
            reason_codes.append("CS_CONNECTOR_CAPTURE_CONSENT_MISSING")
        if not platform_supported:
            reason_codes.append("CS_CONNECTOR_CAPTURE_HOST_UNSUPPORTED")
        elif not platform_permission_active:
            reason_codes.append("CS_CONNECTOR_CAPTURE_PLATFORM_PERMISSION_MISSING")
        if capture_enabled:
            reason_codes.append("CS_CONNECTOR_CAPTURE_READY")

        consent_record_id = str(consent.get("watch_source_consent_id")) if consent else None
        guard_decision_id = _capture_guard_decision_id(
            requested_scope,
            source_id,
            platform,
            platform_permission_state,
            consent_record_id,
        )
        guard = {
            "schema_version": CAPTURE_GUARD_DECISION_SCHEMA,
            "capture_guard_decision_id": guard_decision_id,
            "scope": requested_scope,
            "source_id": source_id,
            "platform": platform,
            "status": "ready" if capture_enabled else "blocked",
            "capture_enabled": capture_enabled,
            "capture_allowed_for_future_collection": capture_enabled,
            "collection_started_by_this_command": False,
            "capture_attempted": False,
            "capture_samples_created": 0,
            "artifacts_created": 0,
            "screenshots_created": 0,
            "raw_window_titles_collected": 0,
            "external_http_calls": 0,
            "provider_mutations": 0,
            "consent": {
                "record_id": consent_record_id,
                "active": consent_active,
                "explicit_owner_consent": consent_active,
                "decision": consent.get("decision") if consent else "missing",
                "distinct_from_platform_permission": True,
            },
            "platform_permission": {
                "probe_id": permission_probe.get("permission_probe_id") if permission_probe else None,
                "state": platform_permission_state,
                "active_for_local_fixture": platform_permission_active,
                "platform_supported": platform_supported,
                "production_permission_proof": "HUMAN_REQUIRED",
                "environment_flag_treated_as_production_proof": False,
            },
            "reason_codes": reason_codes,
            "setup_diagnostics": self._capture_setup_diagnostics(requested_scope, source_id),
            "status_explanation": {
                "reason_code": reason_codes[0] if not capture_enabled else "CS_CONNECTOR_CAPTURE_READY",
                "cause": "Capture requires distinct owner consent and platform permission records.",
                "impact": "No activity samples are collected until both gates are active.",
                "resolution_steps": [
                    "Record explicit owner consent after reviewing data categories and privacy controls.",
                    "Verify the required macOS permission on a supported physical host.",
                    "Re-run capture guard evaluation before starting collection.",
                ],
                "safe_to_show_to_owner": True,
            },
            "negative_evidence": self._capture_negative_evidence(),
            "evidence_refs": [
                ref
                for ref in [
                    f"connector_capture_guard_decision:{guard_decision_id}",
                    f"connector_watch_source_consent:{consent_record_id}" if consent_record_id else None,
                    f"connector_capture_permission_probe:{permission_probe.get('permission_probe_id')}"
                    if permission_probe
                    else None,
                ]
                if ref
            ],
            "audit_refs": [],
            "evaluated_at": utc_now(),
        }
        audit_event = self.store.append_audit(
            "connector.capture.guard_evaluated",
            requested_scope,
            {"type": "connector_capture_guard_decision", "id": guard_decision_id},
            {
                "source_id": source_id,
                "platform": platform,
                "capture_enabled": capture_enabled,
                "collection_started_by_this_command": False,
                "reason_codes": reason_codes,
            },
        )
        guard["audit_refs"] = [f"audit:{audit_event['event_id']}"]
        _write_json(self._capture_guard_decision_path(guard_decision_id), guard)
        return {"status": "success", "connector_capture_guard_decision": guard, "audit_event": audit_event}

    def _chrome_active_tab_payload_issues(
        self,
        *,
        requested_scope: dict[str, str],
        payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        if payload.get("schema_version") != CHROME_ACTIVE_TAB_INPUT_SCHEMA:
            issues.append(
                {
                    "code": "CS_CHROME_ACTIVE_TAB_PAYLOAD_SCHEMA_INVALID",
                    "message": f"Chrome active-tab payloads must use {CHROME_ACTIVE_TAB_INPUT_SCHEMA}.",
                    "path": "schema_version",
                }
            )
        if payload.get("scope") != requested_scope:
            issues.append(
                {
                    "code": "CS_SCOPE_DENIED",
                    "message": "Chrome active-tab payload scope must match the trusted CLI scope.",
                    "path": "scope",
                    "resource_scope": payload.get("scope"),
                }
            )
        source_id = str(payload.get("source_id") or "")
        if source_id != CHROME_ACTIVE_TAB_DEFAULT_SOURCE_ID:
            issues.append(
                {
                    "code": "CS_CHROME_ACTIVE_TAB_SOURCE_INVALID",
                    "message": "Chrome active-tab capture must use the declared chrome_active_tab source.",
                    "path": "source_id",
                }
            )
        for field in ["extension", "invocation", "active_tab", "bounded_payload", "preflight"]:
            if not isinstance(payload.get(field), dict):
                issues.append(
                    {
                        "code": "CS_CHROME_ACTIVE_TAB_SECTION_REQUIRED",
                        "message": f"Chrome active-tab payload requires an object field: {field}.",
                        "path": field,
                    }
                )
        return issues

    def _chrome_active_tab_policy_reason_codes(
        self,
        *,
        payload: dict[str, Any],
        consent: dict[str, Any] | None,
    ) -> list[str]:
        reason_codes: list[str] = []
        extension = payload.get("extension", {}) if isinstance(payload.get("extension"), dict) else {}
        invocation = payload.get("invocation", {}) if isinstance(payload.get("invocation"), dict) else {}
        active_tab = payload.get("active_tab", {}) if isinstance(payload.get("active_tab"), dict) else {}
        bounded_payload = payload.get("bounded_payload", {}) if isinstance(payload.get("bounded_payload"), dict) else {}
        preflight = payload.get("preflight", {}) if isinstance(payload.get("preflight"), dict) else {}
        permissions = _chrome_manifest_list(extension.get("permissions"))
        host_permissions = _chrome_manifest_list(extension.get("host_permissions"))
        optional_host_permissions = _chrome_manifest_list(extension.get("optional_host_permissions"))
        all_permissions = permissions + host_permissions + optional_host_permissions
        text_clip = str(bounded_payload.get("text_clip") or "")
        max_clip = int(bounded_payload.get("max_text_clip_chars") or CHROME_ACTIVE_TAB_MAX_TEXT_CLIP_CHARS)
        consent_active = bool(consent and consent.get("active") and consent.get("explicit_owner_consent"))
        if not consent_active:
            reason_codes.append("CS_CHROME_ACTIVE_TAB_CONSENT_MISSING")
        if "activeTab" not in permissions:
            reason_codes.append("CS_CHROME_ACTIVE_TAB_PERMISSION_MISSING")
        if "<all_urls>" in all_permissions or extension.get("broad_all_urls_permission") is True:
            reason_codes.append("CS_CHROME_ACTIVE_TAB_BROAD_PERMISSION_DENIED")
        if invocation.get("user_gesture") is not True:
            reason_codes.append("CS_CHROME_ACTIVE_TAB_USER_GESTURE_REQUIRED")
        if invocation.get("explicit_confirmation") is not True:
            reason_codes.append("CS_CHROME_ACTIVE_TAB_CONFIRMATION_REQUIRED")
        if invocation.get("popup_open_only") is True:
            reason_codes.append("CS_CHROME_ACTIVE_TAB_POPUP_ONLY_DENIED")
        if invocation.get("active_tab_permission_granted") is not True:
            reason_codes.append("CS_CHROME_ACTIVE_TAB_TEMPORARY_PERMISSION_REQUIRED")
        if invocation.get("active_tab_only") is not True or active_tab.get("active") is not True:
            reason_codes.append("CS_CHROME_ACTIVE_TAB_ONLY_ACTIVE_PAGE_ALLOWED")
        if active_tab.get("browser_internal") is True:
            reason_codes.append("CS_CHROME_ACTIVE_TAB_BROWSER_INTERNAL_BLOCKED")
        if active_tab.get("sensitive_surface") is True:
            reason_codes.append("CS_CHROME_ACTIVE_TAB_SENSITIVE_SURFACE_BLOCKED")
        if preflight.get("decision") != "allow":
            reason_codes.append("CS_CHROME_ACTIVE_TAB_PREFLIGHT_BLOCKED")
        if len(text_clip) > max_clip or len(text_clip) > CHROME_ACTIVE_TAB_MAX_TEXT_CLIP_CHARS:
            reason_codes.append("CS_CHROME_ACTIVE_TAB_PAYLOAD_TOO_LARGE")
        raw_flags = [
            "raw_html_included",
            "cookies_included",
            "local_storage_included",
            "session_storage_included",
            "screenshot_included",
            "form_values_included",
            "browser_history_included",
        ]
        for flag in raw_flags:
            if bounded_payload.get(flag) is True:
                reason_codes.append("CS_CHROME_ACTIVE_TAB_RAW_BROWSER_DATA_DENIED")
                break
        if not reason_codes:
            reason_codes.append("CS_CHROME_ACTIVE_TAB_POLICY_ALLOW")
        return reason_codes

    def process_chrome_active_tab_capture(
        self,
        *,
        requested_scope: dict[str, str],
        payload: dict[str, Any],
        source_path: str,
    ) -> dict[str, Any]:
        issues = self._chrome_active_tab_payload_issues(requested_scope=requested_scope, payload=payload)
        if any(issue.get("code") == "CS_SCOPE_DENIED" for issue in issues):
            return {"status": "scope_denied", "issues": issues, "resource_scope": payload.get("scope")}
        if issues:
            return {"status": "failed", "issues": issues}

        source_id = str(payload.get("source_id") or CHROME_ACTIVE_TAB_DEFAULT_SOURCE_ID)
        consent = self._latest_watch_source_consent(requested_scope, source_id)
        reason_codes = self._chrome_active_tab_policy_reason_codes(payload=payload, consent=consent)
        allowed = reason_codes == ["CS_CHROME_ACTIVE_TAB_POLICY_ALLOW"]
        extension = payload.get("extension", {})
        invocation = payload.get("invocation", {})
        active_tab = payload.get("active_tab", {})
        bounded_payload = payload.get("bounded_payload", {})
        preflight = payload.get("preflight", {})
        summary_input = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
        text_clip = str(bounded_payload.get("text_clip") or "")
        text_clip_hash = hashlib.sha256(text_clip.encode("utf-8")).hexdigest() if text_clip else None
        permission_event_id = _chrome_active_tab_permission_event_id(requested_scope, payload)
        payload_id = _chrome_active_tab_payload_id(requested_scope, payload)
        policy_decision_id = _chrome_active_tab_policy_decision_id(requested_scope, payload_id, reason_codes)
        negative_evidence = self._chrome_active_tab_negative_evidence()
        permission_event = {
            "schema_version": CHROME_ACTIVE_TAB_PERMISSION_EVENT_SCHEMA,
            "permission_event_id": permission_event_id,
            "scope": requested_scope,
            "source_id": source_id,
            "extension_manifest_version": extension.get("manifest_version"),
            "extension_id_hash": extension.get("extension_id_hash"),
            "permissions": _chrome_manifest_list(extension.get("permissions")),
            "host_permissions": _chrome_manifest_list(extension.get("host_permissions")),
            "optional_host_permissions": _chrome_manifest_list(extension.get("optional_host_permissions")),
            "active_tab_temporary_access": invocation.get("active_tab_permission_granted") is True,
            "user_gesture": invocation.get("user_gesture") is True,
            "popup_open_only": invocation.get("popup_open_only") is True,
            "explicit_confirmation": invocation.get("explicit_confirmation") is True,
            "active_tab_only": invocation.get("active_tab_only") is True,
            "broad_all_urls_permission": "<all_urls>"
            in (
                _chrome_manifest_list(extension.get("permissions"))
                + _chrome_manifest_list(extension.get("host_permissions"))
                + _chrome_manifest_list(extension.get("optional_host_permissions"))
            ),
            "manual_browser_proof": "HUMAN_REQUIRED",
            "negative_evidence": negative_evidence,
            "evidence_refs": [f"chrome_active_tab_permission_event:{permission_event_id}"],
            "audit_refs": [],
            "created_at": utc_now(),
        }
        sanitized_payload = {
            "schema_version": CHROME_ACTIVE_TAB_PAYLOAD_SCHEMA,
            "active_tab_payload_id": payload_id,
            "scope": requested_scope,
            "source_id": source_id,
            "input_schema_version": payload.get("schema_version"),
            "source_path": source_path,
            "input_payload_hash": json_hash(payload),
            "active_tab": {
                "tab_id_hash": active_tab.get("tab_id_hash"),
                "window_id_hash": active_tab.get("window_id_hash"),
                "url_hash": active_tab.get("url_hash") or hashlib.sha256(str(active_tab.get("url") or "").encode("utf-8")).hexdigest(),
                "origin": active_tab.get("origin"),
                "title_hash": active_tab.get("title_hash"),
                "active": active_tab.get("active") is True,
                "browser_internal": active_tab.get("browser_internal") is True,
                "sensitive_surface": active_tab.get("sensitive_surface") is True,
                "incognito": active_tab.get("incognito") is True,
            },
            "preflight_decision": preflight.get("decision"),
            "preflight_reason_codes": preflight.get("reason_codes", []),
            "bounded_payload": {
                "text_clip_hash": text_clip_hash,
                "text_clip_char_count": len(text_clip),
                "max_text_clip_chars": bounded_payload.get("max_text_clip_chars") or CHROME_ACTIVE_TAB_MAX_TEXT_CLIP_CHARS,
                "raw_text_stored": False,
                "raw_html_stored": False,
                "cookies_collected": False,
                "local_storage_collected": False,
                "session_storage_collected": False,
                "screenshots_collected": False,
                "form_values_collected": False,
                "browser_history_collected": False,
                "text_clip_discarded_after_summary": True,
            },
            "negative_evidence": negative_evidence,
            "evidence_refs": [
                f"chrome_active_tab_payload:{payload_id}",
                f"chrome_active_tab_permission_event:{permission_event_id}",
            ],
            "audit_refs": [],
            "received_at": utc_now(),
        }
        policy_decision = {
            "schema_version": CHROME_ACTIVE_TAB_POLICY_DECISION_SCHEMA,
            "policy_decision_id": policy_decision_id,
            "scope": requested_scope,
            "source_id": source_id,
            "active_tab_payload_id": payload_id,
            "permission_event_id": permission_event_id,
            "decision": "allow" if allowed else "deny",
            "reason_codes": reason_codes,
            "server_revalidated": True,
            "checks": {
                "scope_matches_trusted_request": payload.get("scope") == requested_scope,
                "consent_active": bool(consent and consent.get("active") and consent.get("explicit_owner_consent")),
                "active_tab_permission_present": invocation.get("active_tab_permission_granted") is True,
                "active_tab_only": invocation.get("active_tab_only") is True and active_tab.get("active") is True,
                "user_gesture_present": invocation.get("user_gesture") is True,
                "explicit_confirmation_present": invocation.get("explicit_confirmation") is True,
                "popup_only_denied": invocation.get("popup_open_only") is not True,
                "no_broad_all_urls_permission": permission_event["broad_all_urls_permission"] is False,
                "bounded_payload": len(text_clip) <= int(bounded_payload.get("max_text_clip_chars") or CHROME_ACTIVE_TAB_MAX_TEXT_CLIP_CHARS)
                and len(text_clip) <= CHROME_ACTIVE_TAB_MAX_TEXT_CLIP_CHARS,
                "raw_browser_data_absent": all(
                    bounded_payload.get(flag) is not True
                    for flag in [
                        "raw_html_included",
                        "cookies_included",
                        "local_storage_included",
                        "session_storage_included",
                        "screenshot_included",
                        "form_values_included",
                        "browser_history_included",
                    ]
                ),
            },
            "negative_evidence": negative_evidence,
            "evidence_refs": [
                f"chrome_active_tab_policy_decision:{policy_decision_id}",
                f"chrome_active_tab_payload:{payload_id}",
                f"chrome_active_tab_permission_event:{permission_event_id}",
                f"connector_watch_source_consent:{consent.get('watch_source_consent_id')}" if consent else "connector_watch_source_consent:missing",
            ],
            "audit_refs": [],
            "decided_at": utc_now(),
        }
        audit_event = self.store.append_audit(
            "connector.capture.chrome_active_tab_policy_decided",
            requested_scope,
            {"type": "chrome_active_tab_policy_decision", "id": policy_decision_id},
            {
                "source_id": source_id,
                "decision": policy_decision["decision"],
                "reason_codes": reason_codes,
                "raw_text_stored": False,
                "raw_html_stored": False,
            },
        )
        audit_ref = f"audit:{audit_event['event_id']}"
        permission_event["audit_refs"] = [audit_ref]
        sanitized_payload["audit_refs"] = [audit_ref]
        policy_decision["audit_refs"] = [audit_ref]
        _write_json(self._chrome_active_tab_permission_event_path(permission_event_id), permission_event)
        _write_json(self._chrome_active_tab_payload_path(payload_id), sanitized_payload)
        _write_json(self._chrome_active_tab_policy_decision_path(policy_decision_id), policy_decision)
        result: dict[str, Any] = {
            "status": "success" if allowed else "policy_denied",
            "chrome_active_tab_permission_event": permission_event,
            "chrome_active_tab_payload": sanitized_payload,
            "chrome_active_tab_policy_decision": policy_decision,
            "audit_event": audit_event,
        }
        if not allowed:
            return result

        summary_id = _chrome_active_tab_summary_id(requested_scope, payload_id, policy_decision_id)
        inbox_item_id = _capture_inbox_item_id(requested_scope, source_id, summary_id)
        summary = {
            "schema_version": CHROME_ACTIVE_TAB_CAPTURE_SUMMARY_SCHEMA,
            "capture_summary_id": summary_id,
            "scope": requested_scope,
            "source_id": source_id,
            "active_tab_payload_id": payload_id,
            "policy_decision_id": policy_decision_id,
            "summary_text": str(summary_input.get("summary_text") or "Active tab captured as summary-only evidence."),
            "summary_kind": "chrome_active_tab_summary_only",
            "source_origin": sanitized_payload["active_tab"]["origin"],
            "source_url_hash": sanitized_payload["active_tab"]["url_hash"],
            "text_clip_hash": text_clip_hash,
            "raw_text_stored": False,
            "raw_html_stored": False,
            "cookies_collected": False,
            "local_storage_collected": False,
            "session_storage_collected": False,
            "screenshots_collected": False,
            "form_values_collected": False,
            "browser_history_collected": False,
            "trust_state": "untrusted_browser_evidence",
            "confidence": summary_input.get("confidence", {"score": 0.72, "caveats": ["local_fixture_summary_only"]}),
            "negative_evidence": negative_evidence,
            "evidence_refs": [
                f"chrome_active_tab_capture_summary:{summary_id}",
                f"chrome_active_tab_policy_decision:{policy_decision_id}",
                f"chrome_active_tab_payload:{payload_id}",
            ],
            "audit_refs": [audit_ref],
            "created_at": utc_now(),
        }
        inbox_item = {
            "schema_version": CAPTURE_INBOX_ITEM_SCHEMA,
            "capture_inbox_item_id": inbox_item_id,
            "scope": requested_scope,
            "source_id": source_id,
            "status": "pending_review",
            "item_kind": "chrome_active_tab_capture",
            "summary_id": summary_id,
            "policy_decision_id": policy_decision_id,
            "source_origin": sanitized_payload["active_tab"]["origin"],
            "source_url_hash": sanitized_payload["active_tab"]["url_hash"],
            "raw_text_stored": False,
            "raw_html_stored": False,
            "owner_review_required": True,
            "can_save_as_evidence": True,
            "can_dismiss": True,
            "negative_evidence": negative_evidence,
            "evidence_refs": [
                f"capture_inbox_item:{inbox_item_id}",
                f"chrome_active_tab_capture_summary:{summary_id}",
            ],
            "audit_refs": [audit_ref],
            "created_at": utc_now(),
        }
        _write_json(self._chrome_active_tab_summary_path(summary_id), summary)
        _write_json(self._capture_inbox_item_path(inbox_item_id), inbox_item)
        result["chrome_active_tab_capture_summary"] = summary
        result["capture_inbox_item"] = inbox_item
        return result

    def _chrome_auto_capture_config_issues(
        self,
        *,
        requested_scope: dict[str, str],
        config: dict[str, Any],
    ) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        if config.get("schema_version") != CHROME_AUTO_CAPTURE_CONFIG_INPUT_SCHEMA:
            issues.append(
                {
                    "code": "CS_CHROME_AUTO_CAPTURE_CONFIG_SCHEMA_INVALID",
                    "message": f"Chrome auto-capture config must use {CHROME_AUTO_CAPTURE_CONFIG_INPUT_SCHEMA}.",
                    "path": "schema_version",
                }
            )
        if config.get("scope") != requested_scope:
            issues.append(
                {
                    "code": "CS_SCOPE_DENIED",
                    "message": "Chrome auto-capture config scope must match the trusted CLI scope.",
                    "path": "scope",
                    "resource_scope": config.get("scope"),
                }
            )
        if str(config.get("source_id") or "") != CHROME_AUTO_CAPTURE_DEFAULT_SOURCE_ID:
            issues.append(
                {
                    "code": "CS_CHROME_AUTO_CAPTURE_SOURCE_INVALID",
                    "message": "Chrome auto capture must use the declared chrome_auto_capture source.",
                    "path": "source_id",
                }
            )
        for field in ["owner_rule", "site_allowance", "source_pack_allowance", "browser_permission", "limits"]:
            if not isinstance(config.get(field), dict):
                issues.append(
                    {
                        "code": "CS_CHROME_AUTO_CAPTURE_CONFIG_SECTION_REQUIRED",
                        "message": f"Chrome auto-capture config requires an object field: {field}.",
                        "path": field,
                    }
                )
        return issues

    def record_chrome_auto_capture_config(
        self,
        *,
        requested_scope: dict[str, str],
        config: dict[str, Any],
        source_path: str,
    ) -> dict[str, Any]:
        issues = self._chrome_auto_capture_config_issues(requested_scope=requested_scope, config=config)
        if any(issue.get("code") == "CS_SCOPE_DENIED" for issue in issues):
            return {"status": "scope_denied", "issues": issues, "resource_scope": config.get("scope")}
        if issues:
            return {"status": "failed", "issues": issues}

        source_id = str(config.get("source_id") or CHROME_AUTO_CAPTURE_DEFAULT_SOURCE_ID)
        owner_rule = config.get("owner_rule", {})
        site_allowance = config.get("site_allowance", {})
        source_pack_allowance = config.get("source_pack_allowance", {})
        browser_permission = config.get("browser_permission", {})
        limits = config.get("limits", {})
        config_id = _chrome_auto_capture_config_id(requested_scope, config)
        negative_evidence = self._chrome_auto_capture_negative_evidence()
        allowed_origins = [str(origin) for origin in site_allowance.get("allowed_origins", [])]
        allowed_url_hashes = [str(url_hash) for url_hash in site_allowance.get("allowed_url_hashes", [])]
        allowed_trigger_types = [str(trigger_type) for trigger_type in owner_rule.get("allowed_trigger_types", [])]
        browser_permission_granted = (
            browser_permission.get("optional_host_permission_granted") is True
            and browser_permission.get("broad_all_urls_permission") is not True
        )
        auto_capture_enabled = (
            owner_rule.get("confirmed") is True
            and site_allowance.get("site_allowed") is True
            and bool(allowed_origins or allowed_url_hashes)
            and source_pack_allowance.get("allowed") is True
            and browser_permission_granted
            and bool(allowed_trigger_types)
        )
        record = {
            "schema_version": CHROME_AUTO_CAPTURE_CONFIG_SCHEMA,
            "auto_capture_config_id": config_id,
            "scope": requested_scope,
            "source_id": source_id,
            "source_path": source_path,
            "input_schema_version": config.get("schema_version"),
            "input_config_hash": json_hash(config),
            "config_version": str(config.get("config_version") or ""),
            "consent_version": str(config.get("consent_version") or ""),
            "status": "ready" if auto_capture_enabled else "blocked",
            "auto_capture_enabled": auto_capture_enabled,
            "two_sided_consent": {
                "owner_rule_confirmed": owner_rule.get("confirmed") is True,
                "site_allowance_present": site_allowance.get("site_allowed") is True,
                "source_pack_allowance_present": source_pack_allowance.get("allowed") is True,
                "browser_permission_granted": browser_permission_granted,
            },
            "owner_rule": {
                "rule_id": owner_rule.get("rule_id"),
                "confirmed": owner_rule.get("confirmed") is True,
                "active_page_only": owner_rule.get("active_page_only") is not False,
                "allowed_trigger_types": allowed_trigger_types,
            },
            "site_allowance": {
                "allowed_origins": allowed_origins,
                "allowed_url_hashes": allowed_url_hashes,
                "blocked_origins": [str(origin) for origin in site_allowance.get("blocked_origins", [])],
                "site_allowed": site_allowance.get("site_allowed") is True,
            },
            "source_pack_allowance": {
                "source_pack_id": source_pack_allowance.get("source_pack_id"),
                "allowed": source_pack_allowance.get("allowed") is True,
                "policy_snapshot_ref": source_pack_allowance.get("policy_snapshot_ref"),
            },
            "browser_permission": {
                "optional_host_permission_granted": browser_permission.get("optional_host_permission_granted") is True,
                "host_permissions": _chrome_manifest_list(browser_permission.get("host_permissions")),
                "broad_all_urls_permission": browser_permission.get("broad_all_urls_permission") is True,
                "manual_browser_proof": "HUMAN_REQUIRED",
            },
            "limits": {
                "throttle_seconds": int(limits.get("throttle_seconds") or 0),
                "max_captures_per_session": int(limits.get("max_captures_per_session") or 0),
            },
            "diagnostics": {
                "safe_to_show_to_owner": True,
                "auto_capture_enabled": auto_capture_enabled,
                "reason_codes": ["CS_CHROME_AUTO_CAPTURE_CONFIG_READY"]
                if auto_capture_enabled
                else ["CS_CHROME_AUTO_CAPTURE_CONFIG_NOT_READY"],
                "human_browser_privacy_review": "HUMAN_REQUIRED",
            },
            "negative_evidence": negative_evidence,
            "evidence_refs": [f"chrome_auto_capture_config:{config_id}"],
            "audit_refs": [],
            "created_at": utc_now(),
        }
        audit_event = self.store.append_audit(
            "connector.capture.chrome_auto_capture_config_recorded",
            requested_scope,
            {"type": "chrome_auto_capture_config", "id": config_id},
            {
                "source_id": source_id,
                "auto_capture_enabled": auto_capture_enabled,
                "config_version": record["config_version"],
                "source_pack_id": record["source_pack_allowance"]["source_pack_id"],
            },
        )
        record["audit_refs"] = [f"audit:{audit_event['event_id']}"]
        _write_json(self._chrome_auto_capture_config_path(config_id), record)
        return {"status": "success", "chrome_auto_capture_config": record, "audit_event": audit_event}

    def _chrome_auto_capture_trigger_issues(
        self,
        *,
        requested_scope: dict[str, str],
        payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        if payload.get("schema_version") != CHROME_AUTO_CAPTURE_TRIGGER_INPUT_SCHEMA:
            issues.append(
                {
                    "code": "CS_CHROME_AUTO_CAPTURE_TRIGGER_SCHEMA_INVALID",
                    "message": f"Chrome auto-capture triggers must use {CHROME_AUTO_CAPTURE_TRIGGER_INPUT_SCHEMA}.",
                    "path": "schema_version",
                }
            )
        if payload.get("scope") != requested_scope:
            issues.append(
                {
                    "code": "CS_SCOPE_DENIED",
                    "message": "Chrome auto-capture trigger scope must match the trusted CLI scope.",
                    "path": "scope",
                    "resource_scope": payload.get("scope"),
                }
            )
        if str(payload.get("source_id") or "") != CHROME_AUTO_CAPTURE_DEFAULT_SOURCE_ID:
            issues.append(
                {
                    "code": "CS_CHROME_AUTO_CAPTURE_SOURCE_INVALID",
                    "message": "Chrome auto capture must use the declared chrome_auto_capture source.",
                    "path": "source_id",
                }
            )
        for field in ["extension", "trigger", "page", "bounded_payload"]:
            if not isinstance(payload.get(field), dict):
                issues.append(
                    {
                        "code": "CS_CHROME_AUTO_CAPTURE_TRIGGER_SECTION_REQUIRED",
                        "message": f"Chrome auto-capture trigger requires an object field: {field}.",
                        "path": field,
                    }
                )
        return issues

    def _chrome_auto_capture_summary_exists(self, trigger_id: str) -> bool:
        if not self.chrome_auto_capture_summary_dir.exists():
            return False
        for path in self.chrome_auto_capture_summary_dir.glob("*.json"):
            record = json.loads(path.read_text())
            if record.get("auto_capture_trigger_id") == trigger_id:
                return True
        return False

    def _chrome_auto_capture_policy_reason_codes(
        self,
        *,
        payload: dict[str, Any],
        consent: dict[str, Any] | None,
        config: dict[str, Any] | None,
        duplicate_trigger: bool,
    ) -> list[str]:
        reason_codes: list[str] = []
        extension = payload.get("extension", {}) if isinstance(payload.get("extension"), dict) else {}
        trigger = payload.get("trigger", {}) if isinstance(payload.get("trigger"), dict) else {}
        page = payload.get("page", {}) if isinstance(payload.get("page"), dict) else {}
        bounded_payload = payload.get("bounded_payload", {}) if isinstance(payload.get("bounded_payload"), dict) else {}
        consent_active = bool(consent and consent.get("active") and consent.get("explicit_owner_consent"))
        if not consent_active:
            reason_codes.append("CS_CHROME_AUTO_CAPTURE_CONSENT_MISSING")
        if not config:
            reason_codes.append("CS_CHROME_AUTO_CAPTURE_CONFIG_MISSING")
        elif config.get("auto_capture_enabled") is not True or config.get("status") != "ready":
            reason_codes.append("CS_CHROME_AUTO_CAPTURE_CONFIG_NOT_READY")

        owner_rule = config.get("owner_rule", {}) if isinstance(config, dict) else {}
        site_allowance = config.get("site_allowance", {}) if isinstance(config, dict) else {}
        source_pack_allowance = config.get("source_pack_allowance", {}) if isinstance(config, dict) else {}
        browser_permission = config.get("browser_permission", {}) if isinstance(config, dict) else {}
        limits = config.get("limits", {}) if isinstance(config, dict) else {}
        allowed_origins = set(str(origin) for origin in site_allowance.get("allowed_origins", []))
        allowed_url_hashes = set(str(url_hash) for url_hash in site_allowance.get("allowed_url_hashes", []))
        page_url_hash = str(page.get("url_hash") or hashlib.sha256(str(page.get("url") or "").encode("utf-8")).hexdigest())
        site_allowed = str(page.get("origin") or "") in allowed_origins or page_url_hash in allowed_url_hashes
        source_pack_allowed = (
            source_pack_allowance.get("allowed") is True
            and str(payload.get("source_pack_id") or "") == str(source_pack_allowance.get("source_pack_id") or "")
        )
        browser_permission_granted = (
            browser_permission.get("optional_host_permission_granted") is True
            and browser_permission.get("broad_all_urls_permission") is not True
            and extension.get("broad_all_urls_permission") is not True
            and "<all_urls>" not in _chrome_manifest_list(extension.get("host_permissions"))
            and "<all_urls>" not in _chrome_manifest_list(extension.get("optional_host_permissions"))
        )
        if owner_rule.get("confirmed") is not True:
            reason_codes.append("CS_CHROME_AUTO_CAPTURE_OWNER_RULE_MISSING")
        if not site_allowed:
            reason_codes.append("CS_CHROME_AUTO_CAPTURE_SITE_NOT_ALLOWED")
        if not source_pack_allowed:
            reason_codes.append("CS_CHROME_AUTO_CAPTURE_SOURCE_PACK_NOT_ALLOWED")
        if not browser_permission_granted:
            reason_codes.append("CS_CHROME_AUTO_CAPTURE_BROWSER_PERMISSION_MISSING")
        if config and str(payload.get("consent_version") or "") != str(config.get("consent_version") or ""):
            reason_codes.append("CS_CHROME_AUTO_CAPTURE_CONSENT_VERSION_MISMATCH")
        if config and str(payload.get("config_version") or "") != str(config.get("config_version") or ""):
            reason_codes.append("CS_CHROME_AUTO_CAPTURE_CONFIG_VERSION_MISMATCH")
        if str(trigger.get("trigger_type") or "") not in set(str(item) for item in owner_rule.get("allowed_trigger_types", [])):
            reason_codes.append("CS_CHROME_AUTO_CAPTURE_TRIGGER_NOT_ALLOWED")
        if page.get("active") is not True:
            reason_codes.append("CS_CHROME_AUTO_CAPTURE_ACTIVE_PAGE_REQUIRED")
        if page.get("browser_internal") is True or page.get("sensitive_surface") is True:
            reason_codes.append("CS_CHROME_AUTO_CAPTURE_SENSITIVE_PAGE_BLOCKED")
        if page.get("incognito") is True:
            reason_codes.append("CS_CHROME_AUTO_CAPTURE_INCOGNITO_BLOCKED")
        if trigger.get("within_throttle_window") is True:
            reason_codes.append("CS_CHROME_AUTO_CAPTURE_THROTTLED")
        if int(trigger.get("session_capture_count") or 0) >= int(limits.get("max_captures_per_session") or 1):
            reason_codes.append("CS_CHROME_AUTO_CAPTURE_SESSION_LIMIT_REACHED")
        if duplicate_trigger:
            reason_codes.append("CS_CHROME_AUTO_CAPTURE_IDEMPOTENCY_DUPLICATE")
        max_clip = int(bounded_payload.get("max_text_clip_chars") or CHROME_ACTIVE_TAB_MAX_TEXT_CLIP_CHARS)
        if len(str(bounded_payload.get("text_clip") or "")) > max_clip or len(str(bounded_payload.get("text_clip") or "")) > CHROME_ACTIVE_TAB_MAX_TEXT_CLIP_CHARS:
            reason_codes.append("CS_CHROME_AUTO_CAPTURE_PAYLOAD_TOO_LARGE")
        for flag in [
            "raw_html_included",
            "cookies_included",
            "local_storage_included",
            "session_storage_included",
            "screenshot_included",
            "form_values_included",
            "browser_history_included",
        ]:
            if bounded_payload.get(flag) is True:
                reason_codes.append("CS_CHROME_AUTO_CAPTURE_RAW_BROWSER_DATA_DENIED")
                break
        if not reason_codes:
            reason_codes.append("CS_CHROME_AUTO_CAPTURE_POLICY_ALLOW")
        return reason_codes

    def process_chrome_auto_capture_trigger(
        self,
        *,
        requested_scope: dict[str, str],
        payload: dict[str, Any],
        source_path: str,
    ) -> dict[str, Any]:
        issues = self._chrome_auto_capture_trigger_issues(requested_scope=requested_scope, payload=payload)
        if any(issue.get("code") == "CS_SCOPE_DENIED" for issue in issues):
            return {"status": "scope_denied", "issues": issues, "resource_scope": payload.get("scope")}
        if issues:
            return {"status": "failed", "issues": issues}

        source_id = str(payload.get("source_id") or CHROME_AUTO_CAPTURE_DEFAULT_SOURCE_ID)
        consent = self._latest_watch_source_consent(requested_scope, source_id)
        config = self._latest_chrome_auto_capture_config(requested_scope, source_id)
        trigger_id = _chrome_auto_capture_trigger_id(requested_scope, payload)
        duplicate_trigger = self._chrome_auto_capture_summary_exists(trigger_id)
        reason_codes = self._chrome_auto_capture_policy_reason_codes(
            payload=payload,
            consent=consent,
            config=config,
            duplicate_trigger=duplicate_trigger,
        )
        allowed = reason_codes == ["CS_CHROME_AUTO_CAPTURE_POLICY_ALLOW"]
        extension = payload.get("extension", {})
        trigger = payload.get("trigger", {})
        page = payload.get("page", {})
        bounded_payload = payload.get("bounded_payload", {})
        summary_input = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
        text_clip = str(bounded_payload.get("text_clip") or "")
        text_clip_hash = hashlib.sha256(text_clip.encode("utf-8")).hexdigest() if text_clip else None
        page_url_hash = str(page.get("url_hash") or hashlib.sha256(str(page.get("url") or "").encode("utf-8")).hexdigest())
        page_origin_hash = hashlib.sha256(str(page.get("origin") or "").encode("utf-8")).hexdigest() if page.get("origin") else None
        config_id = str(config.get("auto_capture_config_id")) if config else None
        policy_decision_id = _chrome_auto_capture_policy_decision_id(requested_scope, trigger_id, reason_codes)
        negative_evidence = self._chrome_auto_capture_negative_evidence()
        trigger_record = {
            "schema_version": CHROME_AUTO_CAPTURE_TRIGGER_SCHEMA,
            "auto_capture_trigger_id": trigger_id,
            "scope": requested_scope,
            "source_id": source_id,
            "source_path": source_path,
            "input_schema_version": payload.get("schema_version"),
            "input_payload_hash": json_hash(payload),
            "auto_capture_config_id": config_id,
            "config_version": str(payload.get("config_version") or ""),
            "consent_version": str(payload.get("consent_version") or ""),
            "source_pack_id": str(payload.get("source_pack_id") or ""),
            "extension": {
                "extension_id_hash": extension.get("extension_id_hash"),
                "manifest_version": extension.get("manifest_version"),
                "permissions": _chrome_manifest_list(extension.get("permissions")),
                "host_permission_hashes": [
                    hashlib.sha256(permission.encode("utf-8")).hexdigest()
                    for permission in _chrome_manifest_list(extension.get("host_permissions"))
                ],
                "optional_host_permission_hashes": [
                    hashlib.sha256(permission.encode("utf-8")).hexdigest()
                    for permission in _chrome_manifest_list(extension.get("optional_host_permissions"))
                ],
                "broad_all_urls_permission": extension.get("broad_all_urls_permission") is True,
            },
            "trigger": {
                "event_id": trigger.get("event_id"),
                "trigger_type": trigger.get("trigger_type"),
                "idempotency_key_hash": hashlib.sha256(str(trigger.get("idempotency_key") or "").encode("utf-8")).hexdigest(),
                "within_throttle_window": trigger.get("within_throttle_window") is True,
                "session_capture_count": int(trigger.get("session_capture_count") or 0),
                "session_id_hash": trigger.get("session_id_hash"),
            },
            "page": {
                "url_hash": page_url_hash,
                "origin": page.get("origin") if allowed else None,
                "origin_hash": page_origin_hash,
                "title_hash": page.get("title_hash"),
                "active": page.get("active") is True,
                "browser_internal": page.get("browser_internal") is True,
                "sensitive_surface": page.get("sensitive_surface") is True,
                "incognito": page.get("incognito") is True,
            },
            "bounded_payload": {
                "text_clip_hash": text_clip_hash,
                "text_clip_char_count": len(text_clip),
                "max_text_clip_chars": bounded_payload.get("max_text_clip_chars") or CHROME_ACTIVE_TAB_MAX_TEXT_CLIP_CHARS,
                "raw_text_stored": False,
                "raw_html_stored": False,
                "cookies_collected": False,
                "local_storage_collected": False,
                "session_storage_collected": False,
                "screenshots_collected": False,
                "form_values_collected": False,
                "browser_history_collected": False,
                "text_clip_discarded_after_summary": True,
            },
            "negative_evidence": negative_evidence,
            "evidence_refs": [f"chrome_auto_capture_trigger:{trigger_id}"],
            "audit_refs": [],
            "received_at": utc_now(),
        }
        config_owner_rule = config.get("owner_rule", {}) if isinstance(config, dict) else {}
        config_site_allowance = config.get("site_allowance", {}) if isinstance(config, dict) else {}
        config_source_pack = config.get("source_pack_allowance", {}) if isinstance(config, dict) else {}
        config_browser_permission = config.get("browser_permission", {}) if isinstance(config, dict) else {}
        config_limits = config.get("limits", {}) if isinstance(config, dict) else {}
        policy_decision = {
            "schema_version": CHROME_AUTO_CAPTURE_POLICY_DECISION_SCHEMA,
            "policy_decision_id": policy_decision_id,
            "scope": requested_scope,
            "source_id": source_id,
            "auto_capture_trigger_id": trigger_id,
            "auto_capture_config_id": config_id,
            "decision": "allow" if allowed else "deny",
            "reason_codes": reason_codes,
            "server_revalidated": True,
            "checks": {
                "scope_matches_trusted_request": payload.get("scope") == requested_scope,
                "consent_active": bool(consent and consent.get("active") and consent.get("explicit_owner_consent")),
                "config_ready": bool(config and config.get("auto_capture_enabled") is True and config.get("status") == "ready"),
                "owner_rule_confirmed": config_owner_rule.get("confirmed") is True,
                "site_allowed": str(page.get("origin") or "") in set(config_site_allowance.get("allowed_origins", []))
                or page_url_hash in set(config_site_allowance.get("allowed_url_hashes", [])),
                "source_pack_allowed": config_source_pack.get("allowed") is True
                and str(payload.get("source_pack_id") or "") == str(config_source_pack.get("source_pack_id") or ""),
                "browser_permission_granted": config_browser_permission.get("optional_host_permission_granted") is True
                and config_browser_permission.get("broad_all_urls_permission") is not True
                and extension.get("broad_all_urls_permission") is not True,
                "consent_version_matches": bool(config)
                and str(payload.get("consent_version") or "") == str(config.get("consent_version") or ""),
                "config_version_matches": bool(config)
                and str(payload.get("config_version") or "") == str(config.get("config_version") or ""),
                "trigger_type_allowed": str(trigger.get("trigger_type") or "")
                in set(str(item) for item in config_owner_rule.get("allowed_trigger_types", [])),
                "active_allowed_page": page.get("active") is True
                and page.get("browser_internal") is not True
                and page.get("sensitive_surface") is not True
                and page.get("incognito") is not True,
                "throttle_passed": trigger.get("within_throttle_window") is not True,
                "session_limit_passed": int(trigger.get("session_capture_count") or 0)
                < int(config_limits.get("max_captures_per_session") or 1),
                "idempotency_unique": duplicate_trigger is False,
                "raw_browser_data_absent": all(
                    bounded_payload.get(flag) is not True
                    for flag in [
                        "raw_html_included",
                        "cookies_included",
                        "local_storage_included",
                        "session_storage_included",
                        "screenshot_included",
                        "form_values_included",
                        "browser_history_included",
                    ]
                ),
            },
            "diagnostics": {
                "safe_to_show_to_owner": True,
                "trigger_type": trigger.get("trigger_type"),
                "page_origin_hash": page_origin_hash,
                "config_version": str(config.get("config_version") or "") if config else None,
                "consent_version": str(config.get("consent_version") or "") if config else None,
                "throttle_seconds": int(config_limits.get("throttle_seconds") or 0),
                "max_captures_per_session": int(config_limits.get("max_captures_per_session") or 0),
                "human_browser_privacy_review": "HUMAN_REQUIRED",
            },
            "negative_evidence": negative_evidence,
            "evidence_refs": [
                f"chrome_auto_capture_policy_decision:{policy_decision_id}",
                f"chrome_auto_capture_trigger:{trigger_id}",
                f"chrome_auto_capture_config:{config_id}" if config_id else "chrome_auto_capture_config:missing",
                f"connector_watch_source_consent:{consent.get('watch_source_consent_id')}" if consent else "connector_watch_source_consent:missing",
            ],
            "audit_refs": [],
            "decided_at": utc_now(),
        }
        audit_event = self.store.append_audit(
            "connector.capture.chrome_auto_capture_policy_decided",
            requested_scope,
            {"type": "chrome_auto_capture_policy_decision", "id": policy_decision_id},
            {
                "source_id": source_id,
                "decision": policy_decision["decision"],
                "reason_codes": reason_codes,
                "auto_capture_config_id": config_id,
            },
        )
        audit_ref = f"audit:{audit_event['event_id']}"
        trigger_record["audit_refs"] = [audit_ref]
        policy_decision["audit_refs"] = [audit_ref]
        _write_json(self._chrome_auto_capture_trigger_path(trigger_id), trigger_record)
        _write_json(self._chrome_auto_capture_policy_decision_path(policy_decision_id), policy_decision)
        result: dict[str, Any] = {
            "status": "success" if allowed else "policy_denied",
            "chrome_auto_capture_trigger": trigger_record,
            "chrome_auto_capture_policy_decision": policy_decision,
            "audit_event": audit_event,
        }
        if not allowed:
            return result

        summary_id = _chrome_auto_capture_summary_id(requested_scope, trigger_id, policy_decision_id)
        inbox_item_id = _capture_inbox_item_id(requested_scope, source_id, summary_id)
        summary = {
            "schema_version": CHROME_AUTO_CAPTURE_SUMMARY_SCHEMA,
            "capture_summary_id": summary_id,
            "scope": requested_scope,
            "source_id": source_id,
            "auto_capture_trigger_id": trigger_id,
            "auto_capture_config_id": config_id,
            "policy_decision_id": policy_decision_id,
            "summary_text": str(summary_input.get("summary_text") or "Auto-captured page summarized as review-only evidence."),
            "summary_kind": "chrome_auto_capture_summary_only",
            "source_origin": page.get("origin"),
            "source_url_hash": page_url_hash,
            "text_clip_hash": text_clip_hash,
            "raw_text_stored": False,
            "raw_html_stored": False,
            "cookies_collected": False,
            "local_storage_collected": False,
            "session_storage_collected": False,
            "screenshots_collected": False,
            "form_values_collected": False,
            "browser_history_collected": False,
            "trust_state": "untrusted_browser_evidence",
            "confidence": summary_input.get("confidence", {"score": 0.7, "caveats": ["local_fixture_auto_capture"]}),
            "negative_evidence": negative_evidence,
            "evidence_refs": [
                f"chrome_auto_capture_summary:{summary_id}",
                f"chrome_auto_capture_policy_decision:{policy_decision_id}",
                f"chrome_auto_capture_trigger:{trigger_id}",
            ],
            "audit_refs": [audit_ref],
            "created_at": utc_now(),
        }
        inbox_item = {
            "schema_version": CAPTURE_INBOX_ITEM_SCHEMA,
            "capture_inbox_item_id": inbox_item_id,
            "scope": requested_scope,
            "source_id": source_id,
            "status": "pending_review",
            "item_kind": "chrome_auto_capture",
            "summary_id": summary_id,
            "policy_decision_id": policy_decision_id,
            "source_origin": page.get("origin"),
            "source_url_hash": page_url_hash,
            "raw_text_stored": False,
            "raw_html_stored": False,
            "owner_review_required": True,
            "can_save_as_evidence": True,
            "can_dismiss": True,
            "negative_evidence": negative_evidence,
            "evidence_refs": [
                f"capture_inbox_item:{inbox_item_id}",
                f"chrome_auto_capture_summary:{summary_id}",
            ],
            "audit_refs": [audit_ref],
            "created_at": utc_now(),
        }
        _write_json(self._chrome_auto_capture_summary_path(summary_id), summary)
        _write_json(self._capture_inbox_item_path(inbox_item_id), inbox_item)
        result["chrome_auto_capture_summary"] = summary
        result["capture_inbox_item"] = inbox_item
        return result

    def _chrome_sensitive_page_payload_issues(
        self,
        *,
        requested_scope: dict[str, str],
        payload: dict[str, Any],
    ) -> list[dict[str, Any]]:
        issues: list[dict[str, Any]] = []
        if payload.get("schema_version") != CHROME_SENSITIVE_PAGE_INPUT_SCHEMA:
            issues.append(
                {
                    "code": "CS_CHROME_SENSITIVE_PAGE_SCHEMA_INVALID",
                    "message": f"Chrome sensitive-page policy evaluation must use {CHROME_SENSITIVE_PAGE_INPUT_SCHEMA}.",
                    "path": "schema_version",
                }
            )
        if payload.get("scope") != requested_scope:
            issues.append(
                {
                    "code": "CS_SCOPE_DENIED",
                    "message": "Chrome sensitive-page policy scope must match the trusted CLI scope.",
                    "path": "scope",
                    "resource_scope": payload.get("scope"),
                }
            )
        if str(payload.get("source_id") or "") != CHROME_SENSITIVE_PAGE_DEFAULT_SOURCE_ID:
            issues.append(
                {
                    "code": "CS_CHROME_SENSITIVE_PAGE_SOURCE_INVALID",
                    "message": "Chrome sensitive-page policy must use the declared chrome_sensitive_page source.",
                    "path": "source_id",
                }
            )
        cases = payload.get("cases")
        if not isinstance(cases, list) or not cases:
            issues.append(
                {
                    "code": "CS_CHROME_SENSITIVE_PAGE_CASES_REQUIRED",
                    "message": "Chrome sensitive-page policy evaluation requires one or more cases.",
                    "path": "cases",
                }
            )
        elif any(not isinstance(case, dict) for case in cases):
            issues.append(
                {
                    "code": "CS_CHROME_SENSITIVE_PAGE_CASE_INVALID",
                    "message": "Every Chrome sensitive-page case must be an object.",
                    "path": "cases",
                }
            )
        return issues

    def _chrome_sensitive_page_case_reason_codes(self, case_payload: dict[str, Any]) -> list[str]:
        reason_codes: list[str] = []
        page = case_payload.get("page", {}) if isinstance(case_payload.get("page"), dict) else {}
        signals = case_payload.get("sensitive_signals", {}) if isinstance(case_payload.get("sensitive_signals"), dict) else {}
        preflight = case_payload.get("client_preflight", {}) if isinstance(case_payload.get("client_preflight"), dict) else {}
        bounded_payload = case_payload.get("bounded_payload", {}) if isinstance(case_payload.get("bounded_payload"), dict) else {}
        scheme = str(page.get("scheme") or "").lower()
        page_classification = str(page.get("page_classification") or "").lower()
        if str(preflight.get("decision") or "") == "block":
            reason_codes.append("CS_CHROME_SENSITIVE_CLIENT_BLOCK_PRESERVED")
        if page.get("browser_internal") is True:
            reason_codes.append("CS_CHROME_SENSITIVE_BROWSER_INTERNAL_BLOCKED")
        if page.get("incognito") is True or signals.get("private_account") is True or page_classification == "private_account":
            reason_codes.append("CS_CHROME_SENSITIVE_PRIVATE_CONTEXT_BLOCKED")
        if scheme and scheme not in {"http", "https"}:
            reason_codes.append("CS_CHROME_SENSITIVE_UNSUPPORTED_SCHEME_BLOCKED")
        if signals.get("password_field_count", 0):
            reason_codes.append("CS_CHROME_SENSITIVE_PASSWORD_FIELD_BLOCKED")
        if signals.get("payment_field_count", 0):
            reason_codes.append("CS_CHROME_SENSITIVE_PAYMENT_FIELD_BLOCKED")
        if (
            signals.get("secret_field_count", 0)
            or signals.get("token_like_count", 0)
            or signals.get("secret_canary_present") is True
        ):
            reason_codes.append("CS_CHROME_SENSITIVE_SECRET_FIELD_BLOCKED")
        if any(
            bounded_payload.get(flag) is True
            for flag in [
                "raw_text_included",
                "raw_html_included",
                "cookies_included",
                "local_storage_included",
                "session_storage_included",
                "screenshot_included",
                "form_values_included",
                "browser_history_included",
            ]
        ):
            reason_codes.append("CS_CHROME_SENSITIVE_RAW_BROWSER_DATA_BLOCKED")
        false_safe = preflight.get("decision") == "allow" and any(
            [
                page.get("browser_internal") is True,
                page.get("incognito") is True,
                page.get("sensitive_surface") is True,
                signals.get("password_field_count", 0),
                signals.get("payment_field_count", 0),
                signals.get("secret_field_count", 0),
                signals.get("token_like_count", 0),
                signals.get("secret_canary_present") is True,
                signals.get("private_account") is True,
            ]
        )
        if false_safe:
            reason_codes.append("CS_CHROME_SENSITIVE_BACKEND_RECHECK_BLOCKED_FALSE_SAFE")
        if signals.get("compose_surface") is True or signals.get("unknown_editable_surface") is True:
            reason_codes.append("CS_CHROME_SENSITIVE_EDITABLE_SURFACE_DEGRADED")
        if signals.get("oversized_page") is True or int(bounded_payload.get("page_size_bytes") or 0) > int(
            bounded_payload.get("max_page_size_bytes") or 0
        ) > 0:
            reason_codes.append("CS_CHROME_SENSITIVE_OVERSIZED_PAGE_DEGRADED")
        return list(dict.fromkeys(reason_codes or ["CS_CHROME_SENSITIVE_POLICY_ALLOW"]))

    @staticmethod
    def _chrome_sensitive_page_final_decision(reason_codes: list[str], client_decision: str) -> str:
        block_codes = {
            "CS_CHROME_SENSITIVE_CLIENT_BLOCK_PRESERVED",
            "CS_CHROME_SENSITIVE_BROWSER_INTERNAL_BLOCKED",
            "CS_CHROME_SENSITIVE_PRIVATE_CONTEXT_BLOCKED",
            "CS_CHROME_SENSITIVE_UNSUPPORTED_SCHEME_BLOCKED",
            "CS_CHROME_SENSITIVE_PASSWORD_FIELD_BLOCKED",
            "CS_CHROME_SENSITIVE_PAYMENT_FIELD_BLOCKED",
            "CS_CHROME_SENSITIVE_SECRET_FIELD_BLOCKED",
            "CS_CHROME_SENSITIVE_RAW_BROWSER_DATA_BLOCKED",
            "CS_CHROME_SENSITIVE_BACKEND_RECHECK_BLOCKED_FALSE_SAFE",
        }
        if any(reason_code in block_codes for reason_code in reason_codes):
            return "block"
        if any(reason_code.endswith("_DEGRADED") for reason_code in reason_codes) or client_decision == "degraded":
            return "degraded"
        return "allow"

    @staticmethod
    def _chrome_sensitive_page_restriction_rank(decision: str) -> int:
        return {"allow": 0, "degraded": 1, "block": 2}.get(decision, 0)

    def process_chrome_sensitive_page_policy(
        self,
        *,
        requested_scope: dict[str, str],
        payload: dict[str, Any],
        source_path: str,
    ) -> dict[str, Any]:
        issues = self._chrome_sensitive_page_payload_issues(requested_scope=requested_scope, payload=payload)
        if any(issue.get("code") == "CS_SCOPE_DENIED" for issue in issues):
            return {"status": "scope_denied", "issues": issues, "resource_scope": payload.get("scope")}
        if issues:
            return {"status": "failed", "issues": issues}

        source_id = str(payload.get("source_id") or CHROME_SENSITIVE_PAGE_DEFAULT_SOURCE_ID)
        policy_decisions: list[dict[str, Any]] = []
        degraded_payloads: list[dict[str, Any]] = []
        history_items: list[dict[str, Any]] = []
        negative_evidence = self._chrome_sensitive_page_negative_evidence()
        for index, case_payload in enumerate(payload.get("cases", [])):
            page = case_payload.get("page", {}) if isinstance(case_payload.get("page"), dict) else {}
            signals = (
                case_payload.get("sensitive_signals", {})
                if isinstance(case_payload.get("sensitive_signals"), dict)
                else {}
            )
            preflight = (
                case_payload.get("client_preflight", {})
                if isinstance(case_payload.get("client_preflight"), dict)
                else {}
            )
            bounded_payload = (
                case_payload.get("bounded_payload", {})
                if isinstance(case_payload.get("bounded_payload"), dict)
                else {}
            )
            reason_codes = self._chrome_sensitive_page_case_reason_codes(case_payload)
            client_decision = str(preflight.get("decision") or "allow")
            final_decision = self._chrome_sensitive_page_final_decision(reason_codes, client_decision)
            policy_decision_id = _chrome_sensitive_page_policy_decision_id(requested_scope, case_payload)
            history_item_id = _chrome_sensitive_page_history_item_id(requested_scope, policy_decision_id)
            url_value = str(page.get("url") or "")
            origin_value = str(page.get("origin") or "")
            title_value = str(page.get("title") or "")
            url_hash = str(page.get("url_hash") or hashlib.sha256(url_value.encode("utf-8")).hexdigest())
            origin_hash = str(page.get("origin_hash") or hashlib.sha256(origin_value.encode("utf-8")).hexdigest())
            title_hash = str(page.get("title_hash") or hashlib.sha256(title_value.encode("utf-8")).hexdigest())
            backend_restriction_preserved_or_increased = self._chrome_sensitive_page_restriction_rank(
                final_decision
            ) >= self._chrome_sensitive_page_restriction_rank(client_decision)
            policy_decision = {
                "schema_version": CHROME_SENSITIVE_PAGE_POLICY_DECISION_SCHEMA,
                "policy_decision_id": policy_decision_id,
                "scope": requested_scope,
                "source_id": source_id,
                "source_path": source_path,
                "case_id": str(case_payload.get("case_id") or f"case-{index + 1}"),
                "input_schema_version": payload.get("schema_version"),
                "input_case_hash": json_hash(case_payload),
                "client_preflight_decision": client_decision,
                "decision": final_decision,
                "reason_codes": reason_codes,
                "server_revalidated": True,
                "backend_restriction_preserved_or_increased": backend_restriction_preserved_or_increased,
                "page": {
                    "url_hash": url_hash,
                    "origin_hash": origin_hash,
                    "title_hash": title_hash,
                    "scheme": page.get("scheme"),
                    "page_classification": page.get("page_classification"),
                    "browser_internal": page.get("browser_internal") is True,
                    "incognito": page.get("incognito") is True,
                    "sensitive_surface": page.get("sensitive_surface") is True,
                },
                "signal_summary": {
                    "password_field_count": int(signals.get("password_field_count") or 0),
                    "payment_field_count": int(signals.get("payment_field_count") or 0),
                    "secret_field_count": int(signals.get("secret_field_count") or 0),
                    "token_like_count": int(signals.get("token_like_count") or 0),
                    "compose_surface": signals.get("compose_surface") is True,
                    "unknown_editable_surface": signals.get("unknown_editable_surface") is True,
                    "private_account": signals.get("private_account") is True,
                    "oversized_page": signals.get("oversized_page") is True,
                },
                "checks": {
                    "scope_matches_trusted_request": payload.get("scope") == requested_scope,
                    "client_block_not_downgraded": client_decision != "block" or final_decision == "block",
                    "backend_revalidated_sensitive_signals": True,
                    "backend_false_safe_blocked": (
                        "CS_CHROME_SENSITIVE_BACKEND_RECHECK_BLOCKED_FALSE_SAFE" not in reason_codes
                        or final_decision == "block"
                    ),
                    "raw_browser_data_absent_from_output": True,
                    "content_sent_to_models": False,
                    "searchable_content_artifact_created": False,
                    "capture_inbox_item_created": False,
                    "owner_visible_history_item_created": True,
                },
                "negative_evidence": negative_evidence,
                "evidence_refs": [f"chrome_sensitive_page_policy_decision:{policy_decision_id}"],
                "audit_refs": [],
                "decided_at": utc_now(),
            }
            degraded_payload: dict[str, Any] | None = None
            if final_decision == "degraded":
                degraded_payload_id = _chrome_sensitive_page_degraded_payload_id(
                    requested_scope,
                    policy_decision_id,
                    case_payload,
                )
                degraded_payload = {
                    "schema_version": CHROME_SENSITIVE_PAGE_DEGRADED_PAYLOAD_SCHEMA,
                    "degraded_payload_id": degraded_payload_id,
                    "scope": requested_scope,
                    "source_id": source_id,
                    "policy_decision_id": policy_decision_id,
                    "case_id": policy_decision["case_id"],
                    "page_metadata": {
                        "url_hash": url_hash,
                        "origin_hash": origin_hash,
                        "title_hash": title_hash,
                        "page_classification": page.get("page_classification"),
                        "visible_text_hash": str(bounded_payload.get("visible_text_hash") or ""),
                    },
                    "restriction": "metadata_hash_only",
                    "raw_text_stored": False,
                    "raw_html_stored": False,
                    "cookies_collected": False,
                    "local_storage_collected": False,
                    "session_storage_collected": False,
                    "screenshots_collected": False,
                    "form_values_collected": False,
                    "browser_history_collected": False,
                    "content_sent_to_models": False,
                    "searchable_content_artifact_created": False,
                    "capture_inbox_item_created": False,
                    "negative_evidence": negative_evidence,
                    "evidence_refs": [
                        f"chrome_sensitive_page_degraded_payload:{degraded_payload_id}",
                        f"chrome_sensitive_page_policy_decision:{policy_decision_id}",
                    ],
                    "audit_refs": [],
                    "created_at": utc_now(),
                }
                degraded_payloads.append(degraded_payload)
            history_item = {
                "schema_version": CHROME_SENSITIVE_PAGE_HISTORY_ITEM_SCHEMA,
                "history_item_id": history_item_id,
                "scope": requested_scope,
                "source_id": source_id,
                "policy_decision_id": policy_decision_id,
                "degraded_payload_id": degraded_payload.get("degraded_payload_id") if degraded_payload else None,
                "case_id": policy_decision["case_id"],
                "status": final_decision,
                "capture_inbox_surface": "history_only",
                "ui_explanation": (
                    "Capture was blocked because the page appears sensitive."
                    if final_decision == "block"
                    else "Capture was degraded to hash-only metadata because the page may contain sensitive editable or oversized content."
                ),
                "safe_manual_alternative": "Manually create an evidence note with only non-sensitive context and attach approved references.",
                "can_save_as_evidence": False,
                "raw_text_stored": False,
                "raw_html_stored": False,
                "content_sent_to_models": False,
                "searchable_content_artifact_created": False,
                "capture_inbox_item_created": False,
                "negative_evidence": negative_evidence,
                "evidence_refs": [
                    f"chrome_sensitive_page_history_item:{history_item_id}",
                    f"chrome_sensitive_page_policy_decision:{policy_decision_id}",
                ],
                "audit_refs": [],
                "created_at": utc_now(),
            }
            audit_event = self.store.append_audit(
                "connector.capture.chrome_sensitive_page_policy_decided",
                requested_scope,
                {"type": "chrome_sensitive_page_policy_decision", "id": policy_decision_id},
                {
                    "source_id": source_id,
                    "case_id": policy_decision["case_id"],
                    "decision": final_decision,
                    "reason_codes": reason_codes,
                    "content_sent_to_models": False,
                    "searchable_content_artifact_created": False,
                },
            )
            audit_ref = f"audit:{audit_event['event_id']}"
            policy_decision["audit_refs"] = [audit_ref]
            history_item["audit_refs"] = [audit_ref]
            if degraded_payload:
                degraded_payload["audit_refs"] = [audit_ref]
                _write_json(
                    self._chrome_sensitive_page_degraded_payload_path(degraded_payload["degraded_payload_id"]),
                    degraded_payload,
                )
            _write_json(self._chrome_sensitive_page_policy_decision_path(policy_decision_id), policy_decision)
            _write_json(self._chrome_sensitive_page_history_item_path(history_item_id), history_item)
            policy_decisions.append(policy_decision)
            history_items.append(history_item)

        allowed_count = sum(1 for item in policy_decisions if item.get("decision") == "allow")
        blocked_count = sum(1 for item in policy_decisions if item.get("decision") == "block")
        degraded_count = sum(1 for item in policy_decisions if item.get("decision") == "degraded")
        summary = {
            "case_count": len(policy_decisions),
            "blocked_count": blocked_count,
            "degraded_count": degraded_count,
            "allowed_count": allowed_count,
            "backend_revalidated_count": sum(1 for item in policy_decisions if item.get("server_revalidated") is True),
            "client_block_downgrade_count": sum(
                1
                for item in policy_decisions
                if item.get("client_preflight_decision") == "block" and item.get("decision") != "block"
            ),
            "false_safe_bypass_count": sum(
                1
                for item in policy_decisions
                if "CS_CHROME_SENSITIVE_BACKEND_RECHECK_BLOCKED_FALSE_SAFE" in item.get("reason_codes", [])
                and item.get("decision") != "block"
            ),
            "content_sent_to_models": False,
            "searchable_content_artifacts_created": 0,
            "capture_inbox_items_created": 0,
            "raw_text_persisted": False,
            "raw_html_persisted": False,
            "manual_browser_privacy_review": "HUMAN_REQUIRED",
        }
        return {
            "status": "success",
            "chrome_sensitive_page_policy_decisions": policy_decisions,
            "chrome_sensitive_page_degraded_payloads": degraded_payloads,
            "chrome_sensitive_page_history_items": history_items,
            "chrome_sensitive_page_summary": summary,
            "negative_evidence": negative_evidence,
        }

    def _capture_lifecycle_negative_evidence(self) -> dict[str, int]:
        return dict(CAPTURE_LIFECYCLE_NEGATIVE_EVIDENCE_TEMPLATE)

    def _capture_lifecycle_source_states(
        self,
        requested_scope: dict[str, str],
        *,
        source_id: str | None = None,
        target_kind: str | None = None,
    ) -> list[dict[str, Any]]:
        if not self.capture_lifecycle_source_state_dir.exists():
            return []
        states: list[dict[str, Any]] = []
        for path in sorted(self.capture_lifecycle_source_state_dir.glob("*.json")):
            record = json.loads(path.read_text())
            if record.get("scope") != requested_scope:
                continue
            if source_id is not None and record.get("source_id") != source_id:
                continue
            if target_kind is not None and record.get("target_kind") != target_kind:
                continue
            states.append(record)
        states.sort(key=lambda record: str(record.get("updated_at") or record.get("created_at") or ""))
        return states

    def _latest_capture_lifecycle_source_state(
        self,
        requested_scope: dict[str, str],
        *,
        source_id: str,
        target_kind: str,
        target_id: str | None = None,
    ) -> dict[str, Any] | None:
        target_key = target_id or source_id
        states = [
            state
            for state in self._capture_lifecycle_source_states(
                requested_scope,
                source_id=source_id,
                target_kind=target_kind,
            )
            if str(state.get("target_id") or state.get("source_id")) == target_key
        ]
        return states[-1] if states else None

    @staticmethod
    def _capture_lifecycle_item_counts(items: list[dict[str, Any]]) -> dict[str, int]:
        eligible = [
            item
            for item in items
            if item.get("status") != "deleted"
            and (item.get("eligible_for_delete") is True or item.get("deletion_eligibility") == "eligible_derived")
        ]
        retained = [
            item
            for item in items
            if item.get("retention_requirement") in {"immutable_evidence", "audit_obligation"}
            and item.get("status") != "deleted"
        ]
        anonymizable = [
            item
            for item in items
            if item.get("status") != "deleted" and item.get("anonymize_on_delete") is True
        ]
        deleted = [item for item in items if item.get("status") == "deleted"]
        return {
            "total": len(items),
            "eligible_for_delete": len(eligible),
            "retained_for_audit_or_evidence": len(retained),
            "anonymizable": len(anonymizable),
            "deleted": len(deleted),
        }

    def _capture_lifecycle_seed_issues(
        self,
        *,
        payload: dict[str, Any],
        requested_scope: dict[str, str],
    ) -> list[dict[str, str]]:
        issues: list[dict[str, str]] = []
        if payload.get("schema_version") != CAPTURE_LIFECYCLE_SEED_SCHEMA:
            issues.append(
                {
                    "code": "CS_CAPTURE_LIFECYCLE_SEED_SCHEMA_INVALID",
                    "message": f"Capture lifecycle fixtures must use {CAPTURE_LIFECYCLE_SEED_SCHEMA}.",
                    "path": "schema_version",
                }
            )
        fixture_scope = payload.get("scope")
        if not scope_complete(fixture_scope):
            issues.append(
                {
                    "code": "CS_CAPTURE_LIFECYCLE_SCOPE_INCOMPLETE",
                    "message": "Capture lifecycle fixture scope is required.",
                    "path": "scope",
                }
            )
        elif fixture_scope != requested_scope:
            issues.append(
                {
                    "code": "CS_CAPTURE_LIFECYCLE_SCOPE_MISMATCH",
                    "message": "Capture lifecycle fixture scope must match the trusted CLI scope.",
                    "path": "scope",
                }
            )
        sources = payload.get("sources")
        if not isinstance(sources, list) or not sources:
            issues.append(
                {
                    "code": "CS_CAPTURE_LIFECYCLE_SOURCES_REQUIRED",
                    "message": "At least one captured source state is required.",
                    "path": "sources",
                }
            )
        else:
            for index, source in enumerate(sources):
                if not isinstance(source, dict):
                    issues.append(
                        {
                            "code": "CS_CAPTURE_LIFECYCLE_SOURCE_INVALID",
                            "message": "Every source entry must be an object.",
                            "path": f"sources[{index}]",
                        }
                    )
                    continue
                if str(source.get("target_kind") or "source") not in CAPTURE_LIFECYCLE_TARGET_KINDS:
                    issues.append(
                        {
                            "code": "CS_CAPTURE_LIFECYCLE_TARGET_KIND_INVALID",
                            "message": "Lifecycle target_kind must be source, watch_rule, or global.",
                            "path": f"sources[{index}].target_kind",
                        }
                    )
                if not str(source.get("source_id") or "").strip():
                    issues.append(
                        {
                            "code": "CS_CAPTURE_LIFECYCLE_SOURCE_ID_REQUIRED",
                            "message": "Each lifecycle source requires source_id.",
                            "path": f"sources[{index}].source_id",
                        }
                    )
                status = str(source.get("status") or "active")
                if status not in CAPTURE_LIFECYCLE_STATUSES:
                    issues.append(
                        {
                            "code": "CS_CAPTURE_LIFECYCLE_STATUS_INVALID",
                            "message": "Lifecycle source status must be active, paused, revoked, or disabled.",
                            "path": f"sources[{index}].status",
                        }
                    )
        return issues

    def seed_capture_lifecycle_state(
        self,
        *,
        requested_scope: dict[str, str],
        payload: dict[str, Any],
        source_path: str,
    ) -> dict[str, Any]:
        issues = self._capture_lifecycle_seed_issues(payload=payload, requested_scope=requested_scope)
        if any(issue.get("code") == "CS_CAPTURE_LIFECYCLE_SCOPE_MISMATCH" for issue in issues):
            return {"status": "scope_denied", "issues": issues, "resource_scope": payload.get("scope")}
        if issues:
            return {"status": "failed", "issues": issues}

        source_states: list[dict[str, Any]] = []
        negative_evidence = self._capture_lifecycle_negative_evidence()
        for source in payload.get("sources", []):
            source_id = str(source.get("source_id"))
            target_kind = str(source.get("target_kind") or "source")
            target_id = str(source.get("target_id") or source_id)
            source_state_id = _capture_lifecycle_source_state_id(requested_scope, target_kind, source_id, target_id)
            local_state_items = [
                dict(item)
                for item in source.get("local_state_items", [])
                if isinstance(item, dict)
            ]
            capture_results = [
                dict(result)
                for result in source.get("capture_results", [])
                if isinstance(result, dict)
            ]
            item_counts = self._capture_lifecycle_item_counts(local_state_items)
            status = str(source.get("status") or "active")
            source_state = {
                "schema_version": CAPTURE_LIFECYCLE_SOURCE_STATE_SCHEMA,
                "source_state_id": source_state_id,
                "scope": requested_scope,
                "source_id": source_id,
                "source_type": str(source.get("source_type") or "capture_source"),
                "target_kind": target_kind,
                "target_id": target_id,
                "display_name": str(source.get("display_name") or source_id),
                "status": status,
                "collection_enabled": bool(source.get("collection_enabled", status == "active")),
                "active_consent": bool(source.get("active_consent", status == "active")),
                "capability_active": bool(source.get("capability_active", status == "active")),
                "configuration_retained": True,
                "requires_new_consent": status == "revoked",
                "retention_days": int(source.get("retention_days") or 30),
                "retention_policy": {
                    "immutable_evidence_retained": True,
                    "audit_retained": True,
                    "eligible_derived_state_deleteable": True,
                    "retention_boundary_visible": True,
                },
                "collected_state_summary": {
                    "sample_count": int(source.get("sample_count") or item_counts["total"]),
                    "session_count": int(source.get("session_count") or 0),
                    "capture_result_count": len(capture_results),
                    "local_state_item_counts": item_counts,
                    "raw_content_retained": False,
                    "raw_browser_payload_retained": False,
                },
                "local_state_items": local_state_items,
                "capture_results": capture_results,
                "watch_rule_refs": [str(ref) for ref in source.get("watch_rule_refs", []) if isinstance(ref, str)],
                "lifecycle_history": [
                    {
                        "event": "seeded",
                        "from_status": None,
                        "to_status": status,
                        "source_path": source_path,
                        "at": utc_now(),
                    }
                ],
                "negative_evidence": negative_evidence,
                "evidence_refs": [f"capture_lifecycle_source_state:{source_state_id}"],
                "audit_refs": [],
                "created_at": utc_now(),
                "updated_at": utc_now(),
            }
            audit_event = self.store.append_audit(
                "connector.capture.lifecycle_state_seeded",
                requested_scope,
                {"type": "capture_lifecycle_source_state", "id": source_state_id},
                {
                    "source_id": source_id,
                    "target_kind": target_kind,
                    "status": status,
                    "eligible_for_delete": item_counts["eligible_for_delete"],
                },
            )
            source_state["audit_refs"] = [f"audit:{audit_event['event_id']}"]
            _write_json(self._capture_lifecycle_source_state_path(source_state_id), source_state)
            source_states.append(source_state)

        return {
            "status": "success",
            "capture_lifecycle_source_states": source_states,
            "negative_evidence": negative_evidence,
        }

    def apply_capture_lifecycle_action(
        self,
        *,
        requested_scope: dict[str, str],
        action: str,
        source_id: str,
        target_kind: str,
        target_id: str | None = None,
        reason: str = "",
        retention_days: int | None = None,
    ) -> dict[str, Any]:
        if action not in CAPTURE_LIFECYCLE_ACTIONS:
            return {"status": "failed", "issues": [{"code": "CS_CAPTURE_LIFECYCLE_ACTION_INVALID", "message": "Unsupported capture lifecycle action.", "path": "action"}]}
        if target_kind not in CAPTURE_LIFECYCLE_TARGET_KINDS:
            return {"status": "failed", "issues": [{"code": "CS_CAPTURE_LIFECYCLE_TARGET_KIND_INVALID", "message": "target_kind must be source, watch_rule, or global.", "path": "target_kind"}]}
        if action == "retention" and (retention_days is None or retention_days <= 0):
            return {"status": "failed", "issues": [{"code": "CS_CAPTURE_LIFECYCLE_RETENTION_DAYS_INVALID", "message": "retention_days must be a positive integer.", "path": "retention_days"}]}
        target_key = target_id or ("global_collection" if target_kind == "global" else source_id)
        source_state = self._latest_capture_lifecycle_source_state(
            requested_scope,
            source_id=source_id,
            target_kind=target_kind,
            target_id=target_key,
        )
        if source_state is None:
            source_state_id = _capture_lifecycle_source_state_id(requested_scope, target_kind, source_id, target_key)
            source_state = {
                "schema_version": CAPTURE_LIFECYCLE_SOURCE_STATE_SCHEMA,
                "source_state_id": source_state_id,
                "scope": requested_scope,
                "source_id": source_id,
                "source_type": "capture_source",
                "target_kind": target_kind,
                "target_id": target_key,
                "display_name": source_id,
                "status": "active",
                "collection_enabled": True,
                "active_consent": True,
                "capability_active": True,
                "configuration_retained": True,
                "requires_new_consent": False,
                "retention_days": 30,
                "retention_policy": {
                    "immutable_evidence_retained": True,
                    "audit_retained": True,
                    "eligible_derived_state_deleteable": True,
                    "retention_boundary_visible": True,
                },
                "collected_state_summary": {
                    "sample_count": 0,
                    "session_count": 0,
                    "capture_result_count": 0,
                    "local_state_item_counts": self._capture_lifecycle_item_counts([]),
                    "raw_content_retained": False,
                    "raw_browser_payload_retained": False,
                },
                "local_state_items": [],
                "capture_results": [],
                "watch_rule_refs": [],
                "lifecycle_history": [],
                "negative_evidence": self._capture_lifecycle_negative_evidence(),
                "evidence_refs": [f"capture_lifecycle_source_state:{source_state_id}"],
                "audit_refs": [],
                "created_at": utc_now(),
                "updated_at": utc_now(),
            }
        previous_status = str(source_state.get("status") or "active")
        if action == "pause":
            next_status = "paused"
            source_state["collection_enabled"] = False
            source_state["capability_active"] = False
            source_state["paused_at"] = utc_now()
            reason_codes = ["CS_CAPTURE_LIFECYCLE_PAUSED"]
        elif action == "resume":
            if previous_status == "revoked":
                return {
                    "status": "failed",
                    "issues": [
                        {
                            "code": "CS_CAPTURE_LIFECYCLE_REVOKED_RECONSENT_REQUIRED",
                            "message": "Revoked capture sources require new consent instead of resume.",
                            "path": "status",
                        }
                    ],
                }
            next_status = "active"
            source_state["collection_enabled"] = True
            source_state["capability_active"] = True
            source_state["resumed_at"] = utc_now()
            reason_codes = ["CS_CAPTURE_LIFECYCLE_RESUMED"]
        elif action == "revoke":
            next_status = "revoked"
            source_state["collection_enabled"] = False
            source_state["capability_active"] = False
            source_state["active_consent"] = False
            source_state["requires_new_consent"] = True
            source_state["revoked_at"] = utc_now()
            reason_codes = ["CS_CAPTURE_LIFECYCLE_REVOKED"]
        else:
            next_status = previous_status
            source_state["retention_days"] = int(retention_days or source_state.get("retention_days") or 30)
            source_state["retention_changed_at"] = utc_now()
            source_state["retention_policy"]["retention_boundary_visible"] = True
            reason_codes = ["CS_CAPTURE_LIFECYCLE_RETENTION_UPDATED"]

        source_state["status"] = next_status
        source_state["configuration_retained"] = True
        source_state["lifecycle_history"] = [
            *source_state.get("lifecycle_history", []),
            {
                "event": action,
                "from_status": previous_status,
                "to_status": next_status,
                "reason": reason,
                "retention_days": source_state.get("retention_days"),
                "configuration_retained": True,
                "at": utc_now(),
            },
        ]
        source_state["collected_state_summary"]["local_state_item_counts"] = self._capture_lifecycle_item_counts(
            source_state.get("local_state_items", [])
        )
        source_state["updated_at"] = utc_now()

        decision_basis = {
            "previous_status": previous_status,
            "next_status": next_status,
            "reason": reason,
            "retention_days": source_state.get("retention_days"),
        }
        lifecycle_decision_id = _capture_lifecycle_decision_id(
            requested_scope,
            action,
            target_kind,
            source_id,
            target_key,
            decision_basis,
        )
        decision = {
            "schema_version": CAPTURE_LIFECYCLE_DECISION_SCHEMA,
            "lifecycle_decision_id": lifecycle_decision_id,
            "scope": requested_scope,
            "source_id": source_id,
            "target_kind": target_kind,
            "target_id": target_key,
            "action": action,
            "decision": "allow",
            "previous_status": previous_status,
            "resulting_status": next_status,
            "collection_enabled": source_state["collection_enabled"],
            "future_capture_allowed": source_state["collection_enabled"] and next_status == "active",
            "configuration_retained": True,
            "requires_new_consent": source_state.get("requires_new_consent") is True,
            "retention_days": source_state.get("retention_days"),
            "reason_codes": reason_codes,
            "checks": {
                "decision_persisted": True,
                "pause_or_revoke_stops_future_capture": action not in {"pause", "revoke"}
                or source_state["collection_enabled"] is False,
                "configuration_retained": True,
                "retention_boundary_visible": source_state.get("retention_policy", {}).get("retention_boundary_visible") is True,
            },
            "negative_evidence": self._capture_lifecycle_negative_evidence(),
            "evidence_refs": [
                f"capture_lifecycle_decision:{lifecycle_decision_id}",
                f"capture_lifecycle_source_state:{source_state['source_state_id']}",
            ],
            "audit_refs": [],
            "decided_at": utc_now(),
        }
        audit_event = self.store.append_audit(
            f"connector.capture.lifecycle_{action}",
            requested_scope,
            {"type": "capture_lifecycle_decision", "id": lifecycle_decision_id},
            {
                "source_id": source_id,
                "target_kind": target_kind,
                "previous_status": previous_status,
                "resulting_status": next_status,
                "collection_enabled": source_state["collection_enabled"],
                "retention_days": source_state.get("retention_days"),
            },
        )
        audit_ref = f"audit:{audit_event['event_id']}"
        decision["audit_refs"] = [audit_ref]
        source_state["audit_refs"] = list(dict.fromkeys([*source_state.get("audit_refs", []), audit_ref]))
        _write_json(self._capture_lifecycle_source_state_path(source_state["source_state_id"]), source_state)
        _write_json(self._capture_lifecycle_decision_path(lifecycle_decision_id), decision)
        return {
            "status": "success",
            "capture_lifecycle_source_state": source_state,
            "capture_lifecycle_decision": decision,
            "audit_event": audit_event,
        }

    def attempt_capture_lifecycle_sample(
        self,
        *,
        requested_scope: dict[str, str],
        source_id: str,
        event_id: str,
    ) -> dict[str, Any]:
        states = self._capture_lifecycle_source_states(requested_scope)
        relevant_states = [
            state
            for state in states
            if state.get("source_id") == source_id or state.get("target_kind") == "global"
        ]
        blocking_states = [
            state
            for state in relevant_states
            if state.get("status") in {"paused", "revoked", "disabled"} or state.get("collection_enabled") is False
        ]
        blocking_states.sort(
            key=lambda state: {"revoked": 0, "disabled": 1, "paused": 2, "active": 3}.get(str(state.get("status")), 4)
        )
        blocker = blocking_states[0] if blocking_states else None
        reason_code = "CS_CAPTURE_LIFECYCLE_READY"
        if blocker:
            status = str(blocker.get("status") or "paused")
            reason_code = (
                "CS_CAPTURE_LIFECYCLE_REVOKED_BLOCKS_SAMPLE"
                if status == "revoked"
                else (
                    "CS_CAPTURE_LIFECYCLE_DELETED_STATE_BLOCKS_SAMPLE"
                    if status == "disabled"
                    else "CS_CAPTURE_LIFECYCLE_PAUSED_BLOCKS_SAMPLE"
                )
            )
        lifecycle_decision_id = _capture_lifecycle_decision_id(
            requested_scope,
            "sample_attempt",
            str(blocker.get("target_kind") if blocker else "source"),
            source_id,
            str(blocker.get("target_id") if blocker else source_id),
            {"event_id": event_id, "reason_code": reason_code},
        )
        allowed = blocker is None
        decision = {
            "schema_version": CAPTURE_LIFECYCLE_DECISION_SCHEMA,
            "lifecycle_decision_id": lifecycle_decision_id,
            "scope": requested_scope,
            "source_id": source_id,
            "target_kind": str(blocker.get("target_kind") if blocker else "source"),
            "target_id": str(blocker.get("target_id") if blocker else source_id),
            "action": "sample_attempt",
            "decision": "allow" if allowed else "deny",
            "event_id": event_id,
            "collection_enabled": allowed,
            "future_capture_allowed": allowed,
            "sample_created": False,
            "capture_samples_created": 0,
            "blocked_by_status": blocker.get("status") if blocker else None,
            "reason_codes": [reason_code],
            "checks": {
                "paused_or_revoked_state_checked": True,
                "sample_file_created": False,
                "denied_before_sample_creation": not allowed,
            },
            "negative_evidence": self._capture_lifecycle_negative_evidence(),
            "evidence_refs": [f"capture_lifecycle_decision:{lifecycle_decision_id}"],
            "audit_refs": [],
            "decided_at": utc_now(),
        }
        audit_event = self.store.append_audit(
            "connector.capture.lifecycle_sample_attempted",
            requested_scope,
            {"type": "capture_lifecycle_decision", "id": lifecycle_decision_id},
            {
                "source_id": source_id,
                "event_id": event_id,
                "decision": decision["decision"],
                "reason_codes": decision["reason_codes"],
                "sample_created": False,
            },
        )
        decision["audit_refs"] = [f"audit:{audit_event['event_id']}"]
        _write_json(self._capture_lifecycle_decision_path(lifecycle_decision_id), decision)
        return {"status": "success", "capture_lifecycle_decision": decision, "audit_event": audit_event}

    def export_capture_lifecycle_state(
        self,
        *,
        requested_scope: dict[str, str],
        source_id: str,
        include_history: bool,
    ) -> dict[str, Any]:
        source_states = self._capture_lifecycle_source_states(requested_scope, source_id=source_id)
        if not source_states:
            return {"status": "not_found"}
        export_basis = {
            "source_id": source_id,
            "state_ids": [state.get("source_state_id") for state in source_states],
            "include_history": include_history,
        }
        lifecycle_export_id = _capture_lifecycle_export_id(requested_scope, source_id, export_basis)
        redacted_states: list[dict[str, Any]] = []
        for state in source_states:
            redacted_states.append(
                {
                    "source_state_id": state.get("source_state_id"),
                    "source_id": state.get("source_id"),
                    "source_type": state.get("source_type"),
                    "target_kind": state.get("target_kind"),
                    "target_id": state.get("target_id"),
                    "status": state.get("status"),
                    "collection_enabled": state.get("collection_enabled") is True,
                    "configuration_retained": state.get("configuration_retained") is True,
                    "requires_new_consent": state.get("requires_new_consent") is True,
                    "retention_days": state.get("retention_days"),
                    "retention_policy": state.get("retention_policy", {}),
                    "collected_state_summary": state.get("collected_state_summary", {}),
                    "capture_results": [
                        {
                            "result_id": result.get("result_id"),
                            "kind": result.get("kind"),
                            "status": result.get("status"),
                            "evidence_refs": result.get("evidence_refs", []),
                            "raw_content_included": False,
                        }
                        for result in state.get("capture_results", [])
                        if isinstance(result, dict)
                    ],
                    "lifecycle_history": state.get("lifecycle_history", []) if include_history else [],
                    "audit_refs": state.get("audit_refs", []),
                    "evidence_refs": state.get("evidence_refs", []),
                }
            )
        lifecycle_export = {
            "schema_version": CAPTURE_LIFECYCLE_EXPORT_SCHEMA,
            "lifecycle_export_id": lifecycle_export_id,
            "scope": requested_scope,
            "source_id": source_id,
            "status": "ready",
            "source_scope": {"scope": requested_scope, "source_id": source_id},
            "scoped_to_requested_source": all(state.get("source_id") == source_id for state in redacted_states),
            "redacted": True,
            "permission_aware": True,
            "raw_content_included": False,
            "raw_browser_payload_included": False,
            "credential_values_included": False,
            "states": redacted_states,
            "state_count": len(redacted_states),
            "audit_metadata_included": True,
            "retention_metadata_included": True,
            "history_included": include_history,
            "negative_evidence": self._capture_lifecycle_negative_evidence(),
            "evidence_refs": [f"capture_lifecycle_export:{lifecycle_export_id}"],
            "audit_refs": [],
            "exported_at": utc_now(),
        }
        audit_event = self.store.append_audit(
            "connector.capture.lifecycle_exported",
            requested_scope,
            {"type": "capture_lifecycle_export", "id": lifecycle_export_id},
            {
                "source_id": source_id,
                "state_count": len(redacted_states),
                "redacted": True,
                "raw_content_included": False,
            },
        )
        lifecycle_export["audit_refs"] = [f"audit:{audit_event['event_id']}"]
        _write_json(self._capture_lifecycle_export_path(lifecycle_export_id), lifecycle_export)
        return {"status": "success", "capture_lifecycle_export": lifecycle_export, "audit_event": audit_event}

    def review_capture_result(
        self,
        *,
        requested_scope: dict[str, str],
        result_id: str,
        decision: str,
        note: str = "",
    ) -> dict[str, Any]:
        if decision not in CAPTURE_RESULT_REVIEW_DECISIONS:
            return {"status": "failed", "issues": [{"code": "CS_CAPTURE_RESULT_REVIEW_DECISION_INVALID", "message": "decision must be save or dismiss.", "path": "decision"}]}
        matched_result: dict[str, Any] | None = None
        matched_state: dict[str, Any] | None = None
        for state in self._capture_lifecycle_source_states(requested_scope):
            for result in state.get("capture_results", []):
                if isinstance(result, dict) and result.get("result_id") == result_id:
                    matched_result = result
                    matched_state = state
                    break
            if matched_result:
                break
        if matched_result is None or matched_state is None:
            return {"status": "not_found"}
        review_id = _capture_result_review_id(requested_scope, result_id, decision)
        saved_evidence_ref = f"evidence:capture_result:{result_id}" if decision == "save" else None
        review = {
            "schema_version": CAPTURE_RESULT_REVIEW_SCHEMA,
            "capture_result_review_id": review_id,
            "scope": requested_scope,
            "source_id": matched_state.get("source_id"),
            "source_state_id": matched_state.get("source_state_id"),
            "result_id": result_id,
            "decision": decision,
            "status": "saved" if decision == "save" else "dismissed",
            "note": note,
            "saved_as_evidence": decision == "save",
            "saved_evidence_ref": saved_evidence_ref,
            "dismissed_from_inbox": decision == "dismiss",
            "raw_content_stored": False,
            "raw_browser_payload_stored": False,
            "negative_evidence": self._capture_lifecycle_negative_evidence(),
            "evidence_refs": [
                ref
                for ref in [
                    f"capture_result_review:{review_id}",
                    saved_evidence_ref,
                    *matched_result.get("evidence_refs", []),
                ]
                if ref
            ],
            "audit_refs": [],
            "reviewed_at": utc_now(),
        }
        for result in matched_state.get("capture_results", []):
            if isinstance(result, dict) and result.get("result_id") == result_id:
                result["status"] = review["status"]
                result["saved_evidence_ref"] = saved_evidence_ref
                result["reviewed_at"] = review["reviewed_at"]
        matched_state["updated_at"] = utc_now()
        audit_event = self.store.append_audit(
            "connector.capture.result_reviewed",
            requested_scope,
            {"type": "capture_result_review", "id": review_id},
            {
                "source_id": matched_state.get("source_id"),
                "result_id": result_id,
                "decision": decision,
                "raw_content_stored": False,
            },
        )
        review["audit_refs"] = [f"audit:{audit_event['event_id']}"]
        matched_state["audit_refs"] = list(dict.fromkeys([*matched_state.get("audit_refs", []), f"audit:{audit_event['event_id']}"]))
        _write_json(self._capture_lifecycle_source_state_path(matched_state["source_state_id"]), matched_state)
        _write_json(self._capture_result_review_path(review_id), review)
        return {"status": "success", "capture_result_review": review, "capture_lifecycle_source_state": matched_state, "audit_event": audit_event}

    def delete_capture_lifecycle_state(
        self,
        *,
        requested_scope: dict[str, str],
        source_id: str,
        execute: bool,
        authorized: bool,
        reason: str = "",
    ) -> dict[str, Any]:
        source_states = self._capture_lifecycle_source_states(requested_scope, source_id=source_id)
        if not source_states:
            return {"status": "not_found"}
        if execute and not authorized:
            deletion_basis = {"source_id": source_id, "execute": execute, "authorized": False, "reason": reason}
            receipt_id = _capture_lifecycle_deletion_receipt_id(requested_scope, source_id, "execute_denied", deletion_basis)
            receipt = {
                "schema_version": CAPTURE_LIFECYCLE_DELETION_RECEIPT_SCHEMA,
                "deletion_receipt_id": receipt_id,
                "scope": requested_scope,
                "source_id": source_id,
                "status": "denied",
                "mode": "execute",
                "authorized": False,
                "reason_codes": ["CS_CAPTURE_LIFECYCLE_DELETE_REQUIRES_AUTHORIZATION"],
                "eligible_deleted_count": 0,
                "audit_records_deleted": 0,
                "negative_evidence": {**self._capture_lifecycle_negative_evidence(), "unauthorized_delete_executions": 1},
                "evidence_refs": [f"capture_lifecycle_deletion_receipt:{receipt_id}"],
                "audit_refs": [],
                "created_at": utc_now(),
            }
            audit_event = self.store.append_audit(
                "connector.capture.lifecycle_delete_denied",
                requested_scope,
                {"type": "capture_lifecycle_deletion_receipt", "id": receipt_id},
                {"source_id": source_id, "authorized": False},
            )
            receipt["audit_refs"] = [f"audit:{audit_event['event_id']}"]
            _write_json(self._capture_lifecycle_deletion_receipt_path(receipt_id), receipt)
            return {"status": "success", "capture_lifecycle_deletion_receipt": receipt, "audit_event": audit_event}

        before_counts = [self._capture_lifecycle_item_counts(state.get("local_state_items", [])) for state in source_states]
        will_delete = sum(count["eligible_for_delete"] for count in before_counts)
        will_retain = sum(count["retained_for_audit_or_evidence"] for count in before_counts)
        will_anonymize = sum(count["anonymizable"] for count in before_counts)
        deletion_basis = {
            "source_id": source_id,
            "execute": execute,
            "authorized": authorized,
            "state_ids": [state.get("source_state_id") for state in source_states],
            "will_delete": will_delete,
            "will_retain": will_retain,
            "will_anonymize": will_anonymize,
        }
        mode = "execute" if execute else "dry_run"
        receipt_id = _capture_lifecycle_deletion_receipt_id(requested_scope, source_id, mode, deletion_basis)
        updated_states: list[dict[str, Any]] = []
        if execute:
            for state in source_states:
                previous_status = str(state.get("status") or "")
                for item in state.get("local_state_items", []):
                    if not isinstance(item, dict):
                        continue
                    if item.get("eligible_for_delete") is True or item.get("deletion_eligibility") == "eligible_derived":
                        item["status"] = "deleted"
                        item["deleted_at"] = utc_now()
                        item["payload_hash_retained"] = True
                    elif item.get("anonymize_on_delete") is True:
                        item["status"] = "anonymized"
                        item["anonymized_at"] = utc_now()
                state["status"] = "disabled"
                state["collection_enabled"] = False
                state["capability_active"] = False
                state["configuration_retained"] = True
                state["retention_policy"]["audit_retained"] = True
                state["retention_policy"]["immutable_evidence_retained"] = True
                state["collected_state_summary"]["local_state_item_counts"] = self._capture_lifecycle_item_counts(
                    state.get("local_state_items", [])
                )
                state["lifecycle_history"] = [
                    *state.get("lifecycle_history", []),
                    {
                        "event": "delete_execute",
                        "from_status": previous_status,
                        "to_status": "disabled",
                        "authorized": authorized,
                        "configuration_retained": True,
                        "at": utc_now(),
                    },
                ]
                state["updated_at"] = utc_now()
                updated_states.append(state)
        after_counts = [
            self._capture_lifecycle_item_counts(state.get("local_state_items", []))
            for state in (updated_states if execute else source_states)
        ]
        receipt = {
            "schema_version": CAPTURE_LIFECYCLE_DELETION_RECEIPT_SCHEMA,
            "deletion_receipt_id": receipt_id,
            "scope": requested_scope,
            "source_id": source_id,
            "status": "executed" if execute else "dry_run",
            "mode": mode,
            "authorized": bool(authorized),
            "dry_run": not execute,
            "reason": reason,
            "will_delete": will_delete,
            "will_disable_sources": len(source_states),
            "will_retain": will_retain,
            "will_anonymize": will_anonymize,
            "eligible_deleted_count": sum(count["deleted"] for count in after_counts) if execute else 0,
            "eligible_remaining_after_execute": sum(count["eligible_for_delete"] for count in after_counts) if execute else will_delete,
            "audit_records_deleted": 0,
            "retained_audit_explanation": (
                "Audit events and immutable evidence references are retained even when eligible local capture history is deleted or anonymized."
            ),
            "delete_everything_promised": False,
            "misleading_delete_everything_promise": False,
            "raw_content_deleted_or_exported": False,
            "negative_evidence": self._capture_lifecycle_negative_evidence(),
            "evidence_refs": [f"capture_lifecycle_deletion_receipt:{receipt_id}"],
            "audit_refs": [],
            "created_at": utc_now(),
        }
        audit_event = self.store.append_audit(
            "connector.capture.lifecycle_deleted" if execute else "connector.capture.lifecycle_delete_dry_run",
            requested_scope,
            {"type": "capture_lifecycle_deletion_receipt", "id": receipt_id},
            {
                "source_id": source_id,
                "mode": mode,
                "authorized": bool(authorized),
                "will_delete": will_delete,
                "eligible_deleted_count": receipt["eligible_deleted_count"],
                "audit_records_deleted": 0,
            },
        )
        audit_ref = f"audit:{audit_event['event_id']}"
        receipt["audit_refs"] = [audit_ref]
        if execute:
            for state in updated_states:
                state["audit_refs"] = list(dict.fromkeys([*state.get("audit_refs", []), audit_ref]))
                _write_json(self._capture_lifecycle_source_state_path(state["source_state_id"]), state)
        _write_json(self._capture_lifecycle_deletion_receipt_path(receipt_id), receipt)
        return {
            "status": "success",
            "capture_lifecycle_deletion_receipt": receipt,
            "capture_lifecycle_source_states": updated_states if execute else source_states,
            "audit_event": audit_event,
        }

    def _activity_negative_evidence(self, samples: list[dict[str, Any]]) -> dict[str, int]:
        negative = dict(ACTIVITY_SESSION_NEGATIVE_EVIDENCE_TEMPLATE)
        negative["raw_window_titles_stored"] = sum(1 for sample in samples if sample.get("raw_title_stored") is True)
        negative["full_urls_stored"] = sum(1 for sample in samples if sample.get("full_url_stored") is True)
        negative["keystrokes_collected"] = sum(1 for sample in samples if sample.get("keystrokes_collected") is True)
        negative["clipboard_values_collected"] = sum(1 for sample in samples if sample.get("clipboard_values_collected") is True)
        negative["screenshots_collected"] = sum(1 for sample in samples if sample.get("screenshot_captured") is True)
        negative["cookies_collected"] = sum(1 for sample in samples if sample.get("cookies_collected") is True)
        negative["browser_history_collected"] = sum(1 for sample in samples if sample.get("browser_history_collected") is True)
        return negative

    def _build_activity_session_projection(
        self,
        *,
        requested_scope: dict[str, str],
        sample_batch_id: str,
        session_index: int,
        session_samples: list[dict[str, Any]],
        boundary_reason: str,
        sample_interval_seconds: int,
        source_refs: dict[str, Any],
        filtered_noise_count: int,
    ) -> dict[str, Any]:
        started = _parse_iso_utc(str(session_samples[0]["observed_at"]))
        last_observed = _parse_iso_utc(str(session_samples[-1]["observed_at"]))
        ended = last_observed + timedelta(seconds=sample_interval_seconds)
        sample_ids = [str(sample.get("sample_id")) for sample in session_samples]
        app_categories = [str(sample.get("app_category") or "unknown") for sample in session_samples]
        app_ref_hashes = [str(sample.get("app_ref_hash") or "unknown") for sample in session_samples]
        project_hints = sorted({str(sample.get("project_hint")) for sample in session_samples if sample.get("project_hint")})
        app_switch_count = sum(
            1
            for index in range(1, len(session_samples))
            if app_categories[index] != app_categories[index - 1] or app_ref_hashes[index] != app_ref_hashes[index - 1]
        )
        caveats = ["metadata_only_sessionization"]
        if boundary_reason != "first_session":
            caveats.append(boundary_reason)
        if app_switch_count:
            caveats.append("contains_app_switches")
        if len(session_samples) <= 1:
            caveats.append("sparse_sample_count")
        if not project_hints:
            caveats.append("no_explicit_project_hint")
        if filtered_noise_count:
            caveats.append("low_information_samples_filtered_in_batch")
        base_confidence = 0.86 if len(session_samples) >= 3 and project_hints else 0.74 if len(session_samples) >= 2 else 0.55
        if app_switch_count:
            base_confidence -= 0.04
        if not project_hints:
            base_confidence -= 0.08
        confidence_score = max(0.0, round(base_confidence, 2))
        activity_session_id = _activity_session_projection_id(
            requested_scope,
            sample_batch_id,
            _format_utc(started),
            _format_utc(ended),
            sample_ids,
        )
        return {
            "schema_version": ACTIVITY_SESSION_PROJECTION_SCHEMA,
            "projection_type": "activity_session.v1",
            "activity_session_id": activity_session_id,
            "session_index": session_index,
            "scope": requested_scope,
            "sample_batch_id": sample_batch_id,
            "bounded": True,
            "started_at": _format_utc(started),
            "ended_at": _format_utc(ended),
            "duration_seconds": int((ended - started).total_seconds()),
            "source_sample_ids": sample_ids,
            "observed_facts": {
                "sample_count": len(session_samples),
                "app_categories": sorted(set(app_categories)),
                "app_ref_hashes": sorted(set(app_ref_hashes)),
                "app_switch_count": app_switch_count,
                "project_hints": project_hints,
                "first_sample_at": _format_utc(started),
                "last_sample_at": _format_utc(last_observed),
            },
            "inference": {
                "unsupported_intent_claim_present": False,
                "inference_stored_as_observed_fact": False,
                "intent_claim": None,
                "inferred_work_mode": None,
                "correction_required_before_memory_or_mission_use": True,
            },
            "confidence": {
                "score": confidence_score,
                "basis": [
                    "sample_count",
                    "time_contiguity",
                    "project_hint" if project_hints else "no_project_hint",
                    "app_switch_count",
                ],
                "caveats": caveats,
            },
            "privacy": {
                "privacy_mode": "metadata_only_local_fixture",
                "raw_titles_stored": False,
                "full_urls_stored": False,
                "keystrokes_collected": False,
                "clipboard_values_collected": False,
                "screenshots_collected": False,
                "cookies_collected": False,
                "browser_history_collected": False,
            },
            "source_refs": source_refs,
            "evidence_refs": [f"connector_activity_session:{activity_session_id}"],
            "audit_refs": [],
        }

    def sessionize_activity_samples(
        self,
        *,
        requested_scope: dict[str, str],
        sample_batch: dict[str, Any],
        source_path: str,
    ) -> dict[str, Any]:
        if sample_batch.get("schema_version") != ACTIVITY_SAMPLE_BATCH_SCHEMA:
            return {
                "status": "failed",
                "issues": [
                    {
                        "code": "CS_CONNECTOR_ACTIVITY_SAMPLE_BATCH_SCHEMA_INVALID",
                        "message": "Activity sample batch must use cs.connector_activity_sample_batch.v1.",
                        "path": "schema_version",
                    }
                ],
            }
        source_id = str(sample_batch.get("source_id") or CAPTURE_DEFAULT_SOURCE_ID)
        batch_scope = sample_batch.get("scope")
        if isinstance(batch_scope, dict) and batch_scope != requested_scope:
            return {"status": "scope_denied", "resource_scope": batch_scope}
        samples = sample_batch.get("samples")
        if not isinstance(samples, list) or not samples:
            return {
                "status": "failed",
                "issues": [
                    {
                        "code": "CS_CONNECTOR_ACTIVITY_SAMPLES_REQUIRED",
                        "message": "Activity sample batch must include at least one sample.",
                        "path": "samples",
                    }
                ],
            }
        if not all(isinstance(sample, dict) for sample in samples):
            return {
                "status": "failed",
                "issues": [
                    {
                        "code": "CS_CONNECTOR_ACTIVITY_SAMPLE_INVALID",
                        "message": "Every activity sample must be an object.",
                        "path": "samples",
                    }
                ],
            }

        sample_interval_seconds = int(sample_batch.get("sample_interval_seconds") or ACTIVITY_DEFAULT_SAMPLE_INTERVAL_SECONDS)
        idle_gap_threshold_seconds = int(
            sample_batch.get("idle_gap_threshold_seconds") or ACTIVITY_DEFAULT_IDLE_GAP_THRESHOLD_SECONDS
        )
        source_refs = sample_batch.get("source_refs") if isinstance(sample_batch.get("source_refs"), dict) else {}
        negative = self._activity_negative_evidence(samples)
        parsed_samples: list[dict[str, Any]] = []
        for index, sample in enumerate(samples):
            if not sample.get("sample_id") or not sample.get("observed_at"):
                return {
                    "status": "failed",
                    "issues": [
                        {
                            "code": "CS_CONNECTOR_ACTIVITY_SAMPLE_REQUIRED_FIELD_MISSING",
                            "message": "Activity samples require sample_id and observed_at.",
                            "path": f"samples[{index}]",
                        }
                    ],
                }
            parsed = dict(sample)
            parsed["_observed_dt"] = _parse_iso_utc(str(sample["observed_at"]))
            parsed["_event_key"] = str(sample.get("event_key") or sample.get("sample_id"))
            parsed_samples.append(parsed)
        parsed_samples.sort(key=lambda sample: (sample["_observed_dt"], str(sample.get("sample_id"))))

        seen_event_keys: set[str] = set()
        retained_samples: list[dict[str, Any]] = []
        duplicate_sample_ids: list[str] = []
        idle_sample_ids: list[str] = []
        noise_sample_ids: list[str] = []
        for sample in parsed_samples:
            sample_id = str(sample.get("sample_id"))
            event_key = str(sample["_event_key"])
            if event_key in seen_event_keys:
                duplicate_sample_ids.append(sample_id)
                continue
            seen_event_keys.add(event_key)
            if sample.get("activity_state") == "idle":
                idle_sample_ids.append(sample_id)
                continue
            if sample.get("low_information") is True or sample.get("activity_state") == "noise":
                noise_sample_ids.append(sample_id)
                continue
            retained_samples.append(sample)

        sessions: list[tuple[list[dict[str, Any]], str]] = []
        current: list[dict[str, Any]] = []
        current_boundary = "first_session"
        last_dt: datetime | None = None
        last_project_hint = ""
        for sample in retained_samples:
            observed_dt = sample["_observed_dt"]
            project_hint = str(sample.get("project_hint") or "")
            boundary_reason = ""
            if current and last_dt is not None:
                gap_seconds = int((observed_dt - last_dt).total_seconds())
                if gap_seconds > idle_gap_threshold_seconds:
                    boundary_reason = "idle_gap_boundary"
                elif project_hint and last_project_hint and project_hint != last_project_hint:
                    boundary_reason = "project_hint_changed_boundary"
            if current and boundary_reason:
                sessions.append((current, current_boundary))
                current = []
                current_boundary = boundary_reason
            current.append(sample)
            last_dt = observed_dt
            if project_hint:
                last_project_hint = project_hint
        if current:
            sessions.append((current, current_boundary))

        sample_batch_id = _activity_sample_batch_id(requested_scope, source_id, samples)
        sessionization_id = _activity_sessionization_id(requested_scope, sample_batch_id, ACTIVITY_SESSIONIZER_ALGORITHM)
        sanitized_samples = [
            {
                key: value
                for key, value in sample.items()
                if not key.startswith("_")
                and key
                not in {
                    "raw_window_title",
                    "full_url",
                    "keystrokes",
                    "clipboard_value",
                    "screenshot",
                    "cookies",
                    "browser_history",
                }
            }
            for sample in parsed_samples
        ]
        batch_record = {
            "schema_version": ACTIVITY_SAMPLE_BATCH_SCHEMA,
            "activity_sample_batch_id": sample_batch_id,
            "scope": requested_scope,
            "source_id": source_id,
            "platform": sample_batch.get("platform", "macos"),
            "privacy_mode": sample_batch.get("privacy_mode", "metadata_only_local_fixture"),
            "sample_interval_seconds": sample_interval_seconds,
            "idle_gap_threshold_seconds": idle_gap_threshold_seconds,
            "source_refs": source_refs,
            "sample_count": len(samples),
            "sanitized_samples": sanitized_samples,
            "raw_text_stored": False,
            "raw_html_stored": False,
            "created_from": {"path": source_path, "sha256": hashlib.sha256(json.dumps(sample_batch, sort_keys=True).encode()).hexdigest()},
            "evidence_refs": [f"connector_activity_sample_batch:{sample_batch_id}"],
            "audit_refs": [],
            "created_at": utc_now(),
        }
        session_projections = [
            self._build_activity_session_projection(
                requested_scope=requested_scope,
                sample_batch_id=sample_batch_id,
                session_index=index + 1,
                session_samples=session_samples,
                boundary_reason=boundary_reason,
                sample_interval_seconds=sample_interval_seconds,
                source_refs=source_refs,
                filtered_noise_count=len(noise_sample_ids),
            )
            for index, (session_samples, boundary_reason) in enumerate(sessions)
        ]
        sessionization = {
            "schema_version": ACTIVITY_SESSIONIZATION_SCHEMA,
            "activity_sessionization_id": sessionization_id,
            "scope": requested_scope,
            "source_id": source_id,
            "sample_batch_id": sample_batch_id,
            "algorithm": {
                "name": "CornerStone deterministic activity sessionizer",
                "version": ACTIVITY_SESSIONIZER_ALGORITHM,
                "idle_gap_threshold_seconds": idle_gap_threshold_seconds,
                "sample_interval_seconds": sample_interval_seconds,
                "dedupe_key": "event_key_or_sample_id",
                "noise_strategy": "drop_low_information_samples_with_metrics",
                "intent_inference": "disabled",
            },
            "input_metrics": {
                "sample_count": len(samples),
                "unique_sample_count": len(samples) - len(duplicate_sample_ids),
                "retained_active_sample_count": len(retained_samples),
                "duplicate_sample_count": len(duplicate_sample_ids),
                "idle_gap_sample_count": len(idle_sample_ids),
                "low_information_sample_count": len(noise_sample_ids),
                "session_count": len(session_projections),
            },
            "filtered_samples": {
                "duplicates": duplicate_sample_ids,
                "idle_gap_markers": idle_sample_ids,
                "low_information_noise": noise_sample_ids,
            },
            "retention_record": {
                "mode": "local_fixture",
                "source_retention_days": int(sample_batch.get("retention_days") or 30),
                "raw_samples_retained": False,
                "sanitized_samples_retained": True,
            },
            "session_projection_schema": ACTIVITY_SESSION_PROJECTION_SCHEMA,
            "session_projection_ids": [session["activity_session_id"] for session in session_projections],
            "unsupported_intent_claim_present": False,
            "inference_stored_as_observed_fact": False,
            "negative_evidence": negative,
            "source_refs": source_refs,
            "evidence_refs": [
                f"connector_activity_sessionization:{sessionization_id}",
                f"connector_activity_sample_batch:{sample_batch_id}",
                *[f"connector_activity_session:{session['activity_session_id']}" for session in session_projections],
            ],
            "audit_refs": [],
            "created_at": utc_now(),
        }
        audit_event = self.store.append_audit(
            "connector.capture.activity_sessionized",
            requested_scope,
            {"type": "connector_activity_sessionization", "id": sessionization_id},
            {
                "source_id": source_id,
                "sample_count": len(samples),
                "session_count": len(session_projections),
                "unsupported_intent_claim_present": False,
            },
        )
        audit_ref = f"audit:{audit_event['event_id']}"
        batch_record["audit_refs"] = [audit_ref]
        sessionization["audit_refs"] = [audit_ref]
        for session in session_projections:
            session["audit_refs"] = [audit_ref]
        _write_json(self._activity_sample_batch_path(sample_batch_id), batch_record)
        _write_json(self._activity_sessionization_path(sessionization_id), sessionization)
        for session in session_projections:
            _write_json(self._activity_session_path(session["activity_session_id"]), session)
        return {
            "status": "success",
            "connector_activity_sample_batch": batch_record,
            "connector_activity_sessionization": sessionization,
            "activity_session_projections": session_projections,
            "audit_event": audit_event,
        }

    def _watch_result_negative_evidence(self) -> dict[str, int]:
        return dict(WATCH_RESULT_NEGATIVE_EVIDENCE_TEMPLATE)

    def _load_watch_result(self, watch_result_id: str) -> dict[str, Any] | None:
        path = self._watch_result_path(watch_result_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _load_watch_inference(self, watch_inference_id: str) -> dict[str, Any] | None:
        path = self._watch_inference_path(watch_inference_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    def _watch_result_observation_hash(self, watch_result: dict[str, Any]) -> str:
        observations: list[dict[str, Any]] = []
        for observation_id in watch_result.get("watch_observation_ids", []):
            path = self._watch_observation_path(str(observation_id))
            if path.exists():
                observations.append(json.loads(path.read_text()))
        return json_hash(observations)

    def _watch_result_fixture_issues(
        self,
        *,
        payload: dict[str, Any],
        requested_scope: dict[str, str],
    ) -> list[dict[str, str]]:
        issues: list[dict[str, str]] = []
        if payload.get("schema_version") != WATCH_RESULT_INPUT_SCHEMA:
            issues.append(
                {
                    "code": "CS_WATCH_RESULT_FIXTURE_SCHEMA_INVALID",
                    "message": f"Watch Result fixtures must use {WATCH_RESULT_INPUT_SCHEMA}.",
                    "path": "schema_version",
                }
            )
        fixture_scope = payload.get("scope")
        if not scope_complete(fixture_scope):
            issues.append(
                {
                    "code": "CS_WATCH_RESULT_SCOPE_INCOMPLETE",
                    "message": "Watch Result fixture scope is required.",
                    "path": "scope",
                }
            )
        elif fixture_scope != requested_scope:
            issues.append(
                {
                    "code": "CS_SCOPE_DENIED",
                    "message": "Watch Result fixture scope must match the trusted CLI scope.",
                    "path": "scope",
                }
            )
        if not str(payload.get("watch_result_key") or "").strip():
            issues.append(
                {
                    "code": "CS_WATCH_RESULT_KEY_REQUIRED",
                    "message": "watch_result_key is required.",
                    "path": "watch_result_key",
                }
            )
        observations = payload.get("observations")
        if not isinstance(observations, list) or len(observations) < 1:
            issues.append(
                {
                    "code": "CS_WATCH_RESULT_OBSERVATIONS_REQUIRED",
                    "message": "At least one observation is required.",
                    "path": "observations",
                }
            )
        else:
            for index, observation in enumerate(observations):
                if not isinstance(observation, dict):
                    issues.append(
                        {
                            "code": "CS_WATCH_RESULT_OBSERVATION_INVALID",
                            "message": "Every observation must be an object.",
                            "path": f"observations[{index}]",
                        }
                    )
                    continue
                if not observation.get("observed_facts"):
                    issues.append(
                        {
                            "code": "CS_WATCH_RESULT_OBSERVED_FACTS_REQUIRED",
                            "message": "Observation records require observed_facts.",
                            "path": f"observations[{index}].observed_facts",
                        }
                    )
                if not observation.get("evidence_refs"):
                    issues.append(
                        {
                            "code": "CS_WATCH_RESULT_OBSERVATION_EVIDENCE_REQUIRED",
                            "message": "Observation records require evidence_refs.",
                            "path": f"observations[{index}].evidence_refs",
                        }
                    )
        inferences = payload.get("inferences")
        if not isinstance(inferences, list) or len(inferences) < 1:
            issues.append(
                {
                    "code": "CS_WATCH_RESULT_INFERENCES_REQUIRED",
                    "message": "At least one inference candidate is required.",
                    "path": "inferences",
                }
            )
        else:
            for index, inference in enumerate(inferences):
                if not isinstance(inference, dict):
                    issues.append(
                        {
                            "code": "CS_WATCH_RESULT_INFERENCE_INVALID",
                            "message": "Every inference must be an object.",
                            "path": f"inferences[{index}]",
                        }
                    )
                    continue
                if not str(inference.get("hypothesis") or "").strip():
                    issues.append(
                        {
                            "code": "CS_WATCH_RESULT_HYPOTHESIS_REQUIRED",
                            "message": "Inference candidates require a hypothesis.",
                            "path": f"inferences[{index}].hypothesis",
                        }
                    )
                if not inference.get("evidence_refs"):
                    issues.append(
                        {
                            "code": "CS_WATCH_RESULT_INFERENCE_EVIDENCE_REQUIRED",
                            "message": "Inference candidates require evidence_refs.",
                            "path": f"inferences[{index}].evidence_refs",
                        }
                    )
        if not isinstance(payload.get("proposal"), dict):
            issues.append(
                {
                    "code": "CS_WATCH_RESULT_PROPOSAL_REQUIRED",
                    "message": "A non-executing proposal object is required.",
                    "path": "proposal",
                }
            )
        return issues

    def build_watch_result(
        self,
        *,
        requested_scope: dict[str, str],
        fixture: dict[str, Any],
        source_path: str,
    ) -> dict[str, Any]:
        issues = self._watch_result_fixture_issues(payload=fixture, requested_scope=requested_scope)
        if any(issue.get("code") == "CS_SCOPE_DENIED" for issue in issues):
            return {"status": "scope_denied", "issues": issues, "resource_scope": fixture.get("scope")}
        if issues:
            return {"status": "failed", "issues": issues}

        result_key = str(fixture.get("watch_result_key"))
        negative = self._watch_result_negative_evidence()
        observation_records: list[dict[str, Any]] = []
        for index, observation in enumerate(fixture.get("observations", [])):
            observed_facts = observation.get("observed_facts", {})
            observation_id = _watch_observation_id(requested_scope, result_key, observation)
            record = {
                "schema_version": WATCH_OBSERVATION_SCHEMA,
                "watch_observation_id": observation_id,
                "scope": requested_scope,
                "watch_result_key": result_key,
                "section": "Observation",
                "source_kind": str(observation.get("source_kind") or "connector_evidence"),
                "source_id": str(observation.get("source_id") or ""),
                "observed_at": str(observation.get("observed_at") or fixture.get("created_at") or utc_now()),
                "observed_facts": observed_facts,
                "source_restrictions": observation.get("source_restrictions", []),
                "evidence_refs": list(observation.get("evidence_refs", [])),
                "source_refs": observation.get("source_refs", {}),
                "connector_delivered_evidence_only": True,
                "observed_only": True,
                "contains_hypothesis": False,
                "contains_proposal": False,
                "inference_fields_absent": True,
                "inferred_intent_labeled_as_observed": False,
                "negative_evidence": negative,
                "observation_hash": json_hash(
                    {
                        "scope": requested_scope,
                        "index": index,
                        "observed_facts": observed_facts,
                        "evidence_refs": observation.get("evidence_refs", []),
                    }
                ),
                "audit_refs": [],
                "created_at": utc_now(),
            }
            observation_records.append(record)

        inference_records: list[dict[str, Any]] = []
        for index, inference in enumerate(fixture.get("inferences", [])):
            confidence_score = float(inference.get("confidence", {}).get("score", inference.get("confidence_score", 0.0)) or 0.0)
            unsupported = inference.get("unsupported") is True
            low_confidence = confidence_score < 0.7
            trust_state = "draft_hypothesis" if unsupported or low_confidence else "review_candidate"
            inference_id = _watch_inference_id(requested_scope, result_key, inference)
            record = {
                "schema_version": WATCH_INFERENCE_SCHEMA,
                "watch_inference_id": inference_id,
                "scope": requested_scope,
                "watch_result_key": result_key,
                "section": "Inference",
                "hypothesis": str(inference.get("hypothesis") or ""),
                "confidence": {
                    "score": confidence_score,
                    "label": "low" if low_confidence else "medium",
                    "threshold_for_memory_approval": 0.7,
                },
                "trust_state": trust_state,
                "model": inference.get(
                    "model",
                    {
                        "provider": "local_test",
                        "name": "deterministic_watch_result_fixture",
                        "version": "v1",
                    },
                ),
                "caveats": list(inference.get("caveats", [])),
                "alternatives": list(inference.get("alternatives", [])),
                "evidence_refs": list(inference.get("evidence_refs", [])),
                "observation_refs": list(inference.get("observation_refs", [])),
                "unsupported": unsupported,
                "low_confidence": low_confidence,
                "stored_as_observed_fact": False,
                "eligible_for_approved_memory": bool(inference.get("eligible_for_approved_memory") is True)
                and not low_confidence
                and not unsupported,
                "requires_owner_review": True,
                "negative_evidence": negative,
                "audit_refs": [],
                "created_at": utc_now(),
            }
            inference_records.append(record)

        watch_result_id = _watch_result_id(
            requested_scope,
            result_key,
            [record["watch_observation_id"] for record in observation_records],
            [record["watch_inference_id"] for record in inference_records],
        )
        proposal_fixture = fixture.get("proposal", {})
        proposal = {
            "section": "Proposed",
            "proposal_kind": str(proposal_fixture.get("kind") or "draft_memory"),
            "summary": str(proposal_fixture.get("summary") or ""),
            "required_authority": str(proposal_fixture.get("required_authority") or "owner_review"),
            "risk": str(proposal_fixture.get("risk") or "medium"),
            "action_card_required": proposal_fixture.get("action_card_required") is True,
            "would_execute": False,
            "executed": False,
            "workflow_run_started": False,
            "provider_mutation": False,
            "external_call": False,
        }
        evidence_caveats = list(fixture.get("evidence_caveats", []))
        watch_result = {
            "schema_version": WATCH_RESULT_SCHEMA,
            "watch_result_id": watch_result_id,
            "scope": requested_scope,
            "watch_result_key": result_key,
            "title": str(fixture.get("title") or "Watch Result"),
            "status": "draft_review_required",
            "trust_state": "draft_hypothesis"
            if any(record.get("trust_state") == "draft_hypothesis" for record in inference_records)
            else "review_candidate",
            "section_order": ["Observation", "Inference", "Evidence/Caveats", "Proposed"],
            "sections": {
                "Observation": [record["watch_observation_id"] for record in observation_records],
                "Inference": [record["watch_inference_id"] for record in inference_records],
                "Evidence/Caveats": evidence_caveats,
                "Proposed": proposal,
            },
            "watch_observation_ids": [record["watch_observation_id"] for record in observation_records],
            "watch_inference_ids": [record["watch_inference_id"] for record in inference_records],
            "evidence_caveats": evidence_caveats,
            "proposal": proposal,
            "checks": {
                "sections_separated": True,
                "connector_delivered_evidence_only": True,
                "product_intelligence_created_inference": True,
                "observed_records_contain_no_hypothesis": all(
                    record.get("contains_hypothesis") is False for record in observation_records
                ),
                "inferences_not_observed_facts": all(
                    record.get("stored_as_observed_fact") is False for record in inference_records
                ),
                "unsupported_or_low_confidence_stays_draft": all(
                    record.get("trust_state") == "draft_hypothesis"
                    for record in inference_records
                    if record.get("unsupported") is True or record.get("low_confidence") is True
                ),
                "proposal_non_executing": proposal["executed"] is False
                and proposal["workflow_run_started"] is False
                and proposal["provider_mutation"] is False,
            },
            "negative_evidence": negative,
            "source_path": source_path,
            "source_sha256": hashlib.sha256(json.dumps(fixture, sort_keys=True).encode()).hexdigest(),
            "evidence_refs": [
                f"watch_result:{watch_result_id}",
                *[f"watch_observation:{record['watch_observation_id']}" for record in observation_records],
                *[f"watch_inference:{record['watch_inference_id']}" for record in inference_records],
            ],
            "audit_refs": [],
            "created_at": utc_now(),
        }
        audit_event = self.store.append_audit(
            "watch.result.built",
            requested_scope,
            {"type": "watch_result", "id": watch_result_id},
            {
                "observation_count": len(observation_records),
                "inference_count": len(inference_records),
                "trust_state": watch_result["trust_state"],
                "proposal_executed": False,
            },
        )
        audit_ref = f"audit:{audit_event['event_id']}"
        watch_result["audit_refs"] = [audit_ref]
        for record in [*observation_records, *inference_records]:
            record["audit_refs"] = [audit_ref]
        for record in observation_records:
            _write_json(self._watch_observation_path(record["watch_observation_id"]), record)
        for record in inference_records:
            _write_json(self._watch_inference_path(record["watch_inference_id"]), record)
        _write_json(self._watch_result_path(watch_result_id), watch_result)
        return {
            "status": "success",
            "watch_observations": observation_records,
            "watch_inferences": inference_records,
            "watch_result": watch_result,
            "audit_event": audit_event,
        }

    def correct_watch_result(
        self,
        *,
        requested_scope: dict[str, str],
        watch_result_id: str,
        inference_id: str,
        corrected_hypothesis: str,
        reason: str = "",
    ) -> dict[str, Any]:
        watch_result = self._load_watch_result(watch_result_id)
        if watch_result is None:
            return {"status": "not_found"}
        if watch_result.get("scope") != requested_scope:
            return {"status": "scope_denied", "resource_scope": watch_result.get("scope")}
        inference = self._load_watch_inference(inference_id)
        if inference is None or inference_id not in watch_result.get("watch_inference_ids", []):
            return {"status": "not_found"}
        observation_hash_before = self._watch_result_observation_hash(watch_result)
        correction_id = _watch_result_correction_id(
            requested_scope,
            watch_result_id,
            inference_id,
            corrected_hypothesis,
        )
        inference["previous_hypothesis"] = inference.get("hypothesis")
        inference["hypothesis"] = corrected_hypothesis
        inference["trust_state"] = "draft_hypothesis"
        inference["requires_owner_review"] = True
        inference["correction_refs"] = list(
            dict.fromkeys([*inference.get("correction_refs", []), f"watch_result_correction:{correction_id}"])
        )
        inference["updated_at"] = utc_now()
        _write_json(self._watch_inference_path(inference_id), inference)
        observation_hash_after = self._watch_result_observation_hash(watch_result)
        negative = self._watch_result_negative_evidence()
        if observation_hash_before != observation_hash_after:
            negative["observation_mutated_by_correction"] = 1
        correction = {
            "schema_version": WATCH_RESULT_CORRECTION_SCHEMA,
            "watch_result_correction_id": correction_id,
            "scope": requested_scope,
            "watch_result_id": watch_result_id,
            "watch_inference_id": inference_id,
            "previous_hypothesis": inference.get("previous_hypothesis"),
            "corrected_hypothesis": corrected_hypothesis,
            "reason": reason,
            "observation_hash_before": observation_hash_before,
            "observation_hash_after": observation_hash_after,
            "observation_immutable": observation_hash_before == observation_hash_after,
            "changed_section": "Inference",
            "observation_section_changed": False,
            "negative_evidence": negative,
            "evidence_refs": [f"watch_result_correction:{correction_id}", f"watch_result:{watch_result_id}"],
            "audit_refs": [],
            "created_at": utc_now(),
        }
        watch_result["status"] = "corrected_review_required"
        watch_result["trust_state"] = "draft_hypothesis"
        watch_result["correction_refs"] = list(
            dict.fromkeys([*watch_result.get("correction_refs", []), f"watch_result_correction:{correction_id}"])
        )
        watch_result["updated_at"] = utc_now()
        audit_event = self.store.append_audit(
            "watch.result.corrected",
            requested_scope,
            {"type": "watch_result_correction", "id": correction_id},
            {
                "watch_result_id": watch_result_id,
                "watch_inference_id": inference_id,
                "observation_immutable": correction["observation_immutable"],
            },
        )
        audit_ref = f"audit:{audit_event['event_id']}"
        correction["audit_refs"] = [audit_ref]
        watch_result["audit_refs"] = list(dict.fromkeys([*watch_result.get("audit_refs", []), audit_ref]))
        inference["audit_refs"] = list(dict.fromkeys([*inference.get("audit_refs", []), audit_ref]))
        _write_json(self._watch_result_correction_path(correction_id), correction)
        _write_json(self._watch_inference_path(inference_id), inference)
        _write_json(self._watch_result_path(watch_result_id), watch_result)
        return {
            "status": "success",
            "watch_result": watch_result,
            "watch_inference": inference,
            "watch_result_correction": correction,
            "audit_event": audit_event,
        }

    def review_watch_result(
        self,
        *,
        requested_scope: dict[str, str],
        watch_result_id: str,
        decision: str,
        note: str = "",
    ) -> dict[str, Any]:
        if decision not in WATCH_RESULT_REVIEW_DECISIONS:
            return {"status": "failed", "issues": [{"code": "CS_WATCH_RESULT_REVIEW_DECISION_INVALID", "message": "Invalid Watch Result review decision.", "path": "decision"}]}
        watch_result = self._load_watch_result(watch_result_id)
        if watch_result is None:
            return {"status": "not_found"}
        if watch_result.get("scope") != requested_scope:
            return {"status": "scope_denied", "resource_scope": watch_result.get("scope")}
        review_id = _watch_result_review_id(requested_scope, watch_result_id, decision)
        review_status = {
            "save_draft_memory": "draft_memory_saved",
            "dismiss": "dismissed",
            "create_claim_draft": "claim_draft_prepared",
            "open_mission_draft": "mission_draft_prepared",
        }[decision]
        review = {
            "schema_version": WATCH_RESULT_REVIEW_SCHEMA,
            "watch_result_review_id": review_id,
            "scope": requested_scope,
            "watch_result_id": watch_result_id,
            "decision": decision,
            "status": review_status,
            "note": note,
            "draft_memory_saved": decision == "save_draft_memory",
            "approved_memory_created": False,
            "claim_created": False,
            "mission_opened": False,
            "action_card_created": False,
            "proposal_executed": False,
            "requires_separate_promotion": decision in {"save_draft_memory", "create_claim_draft", "open_mission_draft"},
            "negative_evidence": self._watch_result_negative_evidence(),
            "evidence_refs": [f"watch_result_review:{review_id}", f"watch_result:{watch_result_id}"],
            "audit_refs": [],
            "reviewed_at": utc_now(),
        }
        watch_result["status"] = review_status
        watch_result["review_refs"] = list(
            dict.fromkeys([*watch_result.get("review_refs", []), f"watch_result_review:{review_id}"])
        )
        watch_result["updated_at"] = utc_now()
        audit_event = self.store.append_audit(
            "watch.result.reviewed",
            requested_scope,
            {"type": "watch_result_review", "id": review_id},
            {
                "watch_result_id": watch_result_id,
                "decision": decision,
                "approved_memory_created": False,
                "proposal_executed": False,
            },
        )
        audit_ref = f"audit:{audit_event['event_id']}"
        review["audit_refs"] = [audit_ref]
        watch_result["audit_refs"] = list(dict.fromkeys([*watch_result.get("audit_refs", []), audit_ref]))
        _write_json(self._watch_result_review_path(review_id), review)
        _write_json(self._watch_result_path(watch_result_id), watch_result)
        return {"status": "success", "watch_result": watch_result, "watch_result_review": review, "audit_event": audit_event}

    def attempt_watch_result_memory_approval(
        self,
        *,
        requested_scope: dict[str, str],
        watch_result_id: str,
        inference_id: str,
    ) -> dict[str, Any]:
        watch_result = self._load_watch_result(watch_result_id)
        if watch_result is None:
            return {"status": "not_found"}
        if watch_result.get("scope") != requested_scope:
            return {"status": "scope_denied", "resource_scope": watch_result.get("scope")}
        inference = self._load_watch_inference(inference_id)
        if inference is None or inference_id not in watch_result.get("watch_inference_ids", []):
            return {"status": "not_found"}
        denied = (
            inference.get("low_confidence") is True
            or inference.get("unsupported") is True
            or inference.get("trust_state") == "draft_hypothesis"
            or inference.get("eligible_for_approved_memory") is not True
        )
        review_id = _watch_result_review_id(requested_scope, watch_result_id, f"approve_memory:{inference_id}")
        negative = self._watch_result_negative_evidence()
        review = {
            "schema_version": WATCH_RESULT_REVIEW_SCHEMA,
            "watch_result_review_id": review_id,
            "scope": requested_scope,
            "watch_result_id": watch_result_id,
            "watch_inference_id": inference_id,
            "decision": "approve_memory",
            "status": "denied" if denied else "approved",
            "reason_codes": [
                "CS_WATCH_RESULT_LOW_CONFIDENCE_MEMORY_APPROVAL_DENIED"
                if inference.get("low_confidence") is True
                else "CS_WATCH_RESULT_UNSUPPORTED_INFERENCE_MEMORY_APPROVAL_DENIED"
                if inference.get("unsupported") is True
                else "CS_WATCH_RESULT_OWNER_REVIEW_REQUIRED"
                if denied
                else "CS_WATCH_RESULT_MEMORY_APPROVAL_ALLOWED"
            ],
            "approved_memory_created": False,
            "draft_memory_saved": False,
            "claim_created": False,
            "mission_opened": False,
            "action_card_created": False,
            "proposal_executed": False,
            "negative_evidence": negative,
            "evidence_refs": [
                f"watch_result_review:{review_id}",
                f"watch_result:{watch_result_id}",
                f"watch_inference:{inference_id}",
            ],
            "audit_refs": [],
            "reviewed_at": utc_now(),
        }
        audit_event = self.store.append_audit(
            "watch.result.memory_approval_denied" if denied else "watch.result.memory_approval_allowed",
            requested_scope,
            {"type": "watch_result_review", "id": review_id},
            {
                "watch_result_id": watch_result_id,
                "watch_inference_id": inference_id,
                "denied": denied,
                "approved_memory_created": False,
            },
        )
        audit_ref = f"audit:{audit_event['event_id']}"
        review["audit_refs"] = [audit_ref]
        watch_result["review_refs"] = list(
            dict.fromkeys([*watch_result.get("review_refs", []), f"watch_result_review:{review_id}"])
        )
        watch_result["audit_refs"] = list(dict.fromkeys([*watch_result.get("audit_refs", []), audit_ref]))
        watch_result["updated_at"] = utc_now()
        _write_json(self._watch_result_review_path(review_id), review)
        _write_json(self._watch_result_path(watch_result_id), watch_result)
        return {
            "status": "memory_approval_denied" if denied else "success",
            "watch_result": watch_result,
            "watch_inference": inference,
            "watch_result_review": review,
            "audit_event": audit_event,
        }

    def _connector_action_preflight_negative_evidence(self) -> dict[str, int]:
        return dict(CONNECTOR_ACTION_PREFLIGHT_NEGATIVE_EVIDENCE_TEMPLATE)

    def run_action_preflight(
        self,
        *,
        requested_scope: dict[str, str],
        action_id: str,
        fixture_path: Path,
        case_id: str,
    ) -> dict[str, Any]:
        action = self.store.get_action(action_id)
        if action is None:
            return {"status": "not_found", "resource": "action"}
        if action.get("scope") != requested_scope:
            return {"status": "scope_denied", "resource_scope": action.get("scope")}
        if not fixture_path.exists():
            return {
                "status": "not_found",
                "resource": "fixture",
                "issues": [{"code": "CS_CONNECTOR_ACTION_PREFLIGHT_FIXTURE_NOT_FOUND", "message": "Preflight fixture was not found.", "path": str(fixture_path)}],
            }
        fixture = json.loads(fixture_path.read_text())
        if fixture.get("schema_version") != CONNECTOR_ACTION_PREFLIGHT_FIXTURE_SCHEMA:
            return {
                "status": "failed",
                "issues": [
                    {
                        "code": "CS_CONNECTOR_ACTION_PREFLIGHT_FIXTURE_SCHEMA_INVALID",
                        "message": f"Preflight fixtures must use {CONNECTOR_ACTION_PREFLIGHT_FIXTURE_SCHEMA}.",
                        "path": "schema_version",
                    }
                ],
            }
        if fixture.get("scope") != requested_scope:
            return {"status": "scope_denied", "resource_scope": fixture.get("scope")}
        cases = fixture.get("cases") if isinstance(fixture.get("cases"), list) else []
        case = next((item for item in cases if isinstance(item, dict) and item.get("case_id") == case_id), None)
        if case is None:
            return {
                "status": "not_found",
                "resource": "case",
                "issues": [{"code": "CS_CONNECTOR_ACTION_PREFLIGHT_CASE_NOT_FOUND", "message": "Preflight case was not found.", "path": "case_id"}],
            }

        connector_meta = fixture.get("connector", {}) if isinstance(fixture.get("connector"), dict) else {}
        declaration = fixture.get("action_declaration", {}) if isinstance(fixture.get("action_declaration"), dict) else {}
        action_connector = str(action.get("connector_boundary", {}).get("connector", ""))
        target = str(case.get("target") or action.get("dry_run", {}).get("expected_impact", {}).get("target") or "")
        connector_kind = str(case.get("provider_kind") or connector_meta.get("provider_kind") or "unknown")
        is_github_read_only = (
            connector_kind == "github"
            or "github" in action_connector.lower()
            or target.startswith("github:")
            or str(case.get("action_type", "")).startswith("issue.")
        )
        idempotency_key = str(case.get("idempotency_key") or "")
        gate_results = {
            "non_github_connector": not is_github_read_only,
            "declared_action": case.get("declared_action") is True,
            "provider_supported": case.get("provider_supported") is True,
            "source_policy_allows": case.get("source_policy_allows") is True,
            "permission_granted": case.get("permission_granted") is True,
            "input_schema_valid": case.get("input_schema_valid") is True,
            "idempotency_key_present": bool(idempotency_key.strip()),
        }
        reason_codes: list[str] = []
        if not gate_results["non_github_connector"]:
            reason_codes.append("CS_CONNECTOR_ACTION_PREFLIGHT_GITHUB_READ_ONLY_DENIED")
        if not gate_results["declared_action"]:
            reason_codes.append("CS_CONNECTOR_ACTION_PREFLIGHT_UNDECLARED_ACTION")
        if not gate_results["provider_supported"]:
            reason_codes.append("CS_CONNECTOR_ACTION_PREFLIGHT_PROVIDER_UNSUPPORTED")
        if not gate_results["source_policy_allows"]:
            reason_codes.append("CS_CONNECTOR_ACTION_PREFLIGHT_SOURCE_POLICY_DENIED")
        if not gate_results["permission_granted"]:
            reason_codes.append("CS_CONNECTOR_ACTION_PREFLIGHT_PERMISSION_MISSING")
        if not gate_results["input_schema_valid"]:
            reason_codes.append("CS_CONNECTOR_ACTION_PREFLIGHT_INPUT_INVALID")
        if not gate_results["idempotency_key_present"]:
            reason_codes.append("CS_CONNECTOR_ACTION_PREFLIGHT_IDEMPOTENCY_REQUIRED")
        decision = "allow" if not reason_codes else "deny"
        preflight_id = _connector_action_preflight_id(requested_scope, action_id, case_id)
        expected_calls = declaration.get("expected_calls", []) if decision == "allow" else []
        negative = self._connector_action_preflight_negative_evidence()
        preflight = {
            "schema_version": CONNECTOR_ACTION_PREFLIGHT_SCHEMA,
            "connector_action_preflight_id": preflight_id,
            "scope": requested_scope,
            "action_id": action_id,
            "action_binding": action_preflight_binding_for_action(action),
            "mission_id": action.get("mission_id"),
            "claim_id": action.get("source_claim_id"),
            "case_id": case_id,
            "status": "preflight_allowed" if decision == "allow" else "preflight_denied",
            "decision": decision,
            "reason_codes": reason_codes,
            "connector_kind": connector_kind,
            "connector_id": connector_meta.get("connector_id"),
            "provider_pack_ref": connector_meta.get("provider_pack_id"),
            "action_type": case.get("action_type"),
            "target": target,
            "gate_results": gate_results,
            "policy_input": {
                "source_policy_ref": connector_meta.get("source_policy_ref"),
                "selected_resources": connector_meta.get("selected_resources", []),
                "required_permission": declaration.get("required_permission"),
                "idempotency_key": idempotency_key if idempotency_key else None,
                "product_policy_ref": f"policy:{action.get('policy_decision', {}).get('id')}",
            },
            "risk": {
                "product_risk": action.get("risk"),
                "connector_risk": declaration.get("risk"),
                "approval_required": True,
                "preflight_counts_as_approval": False,
            },
            "call_ledger": {
                "expected_provider_calls": expected_calls,
                "expected_provider_call_count": len(expected_calls),
                "real_provider_calls": [],
                "real_provider_call_count": 0,
                "external_http_calls": 0,
                "provider_mutations": 0,
            },
            "input_schema": {
                "required_fields": declaration.get("required_input_fields", []),
                "valid": gate_results["input_schema_valid"],
                "input_fingerprint": hashlib.sha256(
                    json.dumps(case.get("input", {}), sort_keys=True).encode("utf-8")
                ).hexdigest(),
            },
            "safe_resolution_path": [
                "Use a declared non-GitHub connector action.",
                "Confirm provider support and Source Policy allowance.",
                "Grant the required permission through ConnectorHub.",
                "Supply valid input and a stable idempotency key.",
                "Request Product approval separately before execution.",
            ],
            "negative_evidence": negative,
            "evidence_refs": [
                f"connector_action_preflight:{preflight_id}",
                f"action:{action_id}",
                *action.get("evidence", {}).get("artifact_refs", []),
            ],
            "audit_refs": [],
            "created_at": utc_now(),
        }
        review_id = _connector_action_preflight_review_id(requested_scope, action_id, preflight_id)
        product_policy = action.get("policy_decision", {})
        product_denied = product_policy.get("decision") == "deny"
        blocked = decision != "allow" or product_denied
        review = {
            "schema_version": CONNECTOR_ACTION_PREFLIGHT_REVIEW_SCHEMA,
            "connector_action_preflight_review_id": review_id,
            "scope": requested_scope,
            "action_id": action_id,
            "preflight_id": preflight_id,
            "status": "blocked" if blocked else "owner_review_required",
            "section_order": [
                "Product Impact",
                "Connector Feasibility",
                "Permissions",
                "Source Policy",
                "Risk",
                "Idempotency",
                "Expected Calls",
                "Evidence",
                "Approval",
            ],
            "product_impact": {
                "diff": action.get("dry_run", {}).get("diff", {}),
                "expected_impact": action.get("dry_run", {}).get("expected_impact", {}),
                "policy_decision": product_policy,
            },
            "connector_feasibility": {
                "decision": decision,
                "reason_codes": reason_codes,
                "gate_results": gate_results,
                "provider_support_status": "supported" if gate_results["provider_supported"] else "unsupported",
            },
            "permissions": {
                "required_permission": declaration.get("required_permission"),
                "granted": gate_results["permission_granted"],
            },
            "source_policy": {
                "source_policy_ref": connector_meta.get("source_policy_ref"),
                "allows_action": gate_results["source_policy_allows"],
                "raw_access": connector_meta.get("raw_access"),
            },
            "risk": {
                "product_risk": action.get("risk"),
                "connector_risk": declaration.get("risk"),
                "approval_required": True,
                "approval_status": action.get("approval", {}).get("status"),
                "preflight_counts_as_approval": False,
            },
            "idempotency": {
                "required": declaration.get("required_idempotency") is True,
                "key_present": gate_results["idempotency_key_present"],
                "idempotency_key_fingerprint": hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
                if idempotency_key
                else None,
            },
            "expected_calls": expected_calls,
            "call_ledger": preflight["call_ledger"],
            "evidence_refs": list(dict.fromkeys(preflight["evidence_refs"] + action.get("evidence_refs", []))),
            "approval": {
                "required": True,
                "status": "pending",
                "preflight_is_approval": False,
                "execution_allowed": False,
            },
            "no_side_effects": {
                "dry_run_only": True,
                "execution_result_created": False,
                "workflow_run_started": False,
                "external_http_calls": 0,
                "provider_mutations": 0,
                "real_provider_calls": 0,
            },
            "negative_evidence": negative,
            "audit_refs": [],
            "created_at": utc_now(),
        }
        updated_action = dict(action)
        updated_action["connector_preflight"] = {
            "preflight_ref": f"connector_action_preflight:{preflight_id}",
            "review_ref": f"connector_action_preflight_review:{review_id}",
            "status": review["status"],
            "decision": decision,
            "case_id": case_id,
            "preflight_counts_as_approval": False,
        }
        updated_action["dry_run"] = dict(updated_action.get("dry_run", {}))
        updated_action["dry_run"]["connector_preflight_ref"] = f"connector_action_preflight:{preflight_id}"
        updated_action["approval"] = {
            **updated_action.get("approval", {}),
            "status": updated_action.get("approval", {}).get("status", "pending"),
            "preflight_counts_as_approval": False,
        }
        updated_action["execution"] = {
            **updated_action.get("execution", {}),
            "can_execute_now": False,
            "preflight_blocks_execution": blocked,
            "connector_preflight_status": review["status"],
        }
        updated_action["updated_at"] = utc_now()
        audit_event = self.store.append_audit(
            "connector.action_preflight.completed",
            requested_scope,
            {"type": "connector_action_preflight", "id": preflight_id},
            {
                "action_id": action_id,
                "case_id": case_id,
                "decision": decision,
                "external_http_calls": 0,
                "provider_mutations": 0,
                "preflight_counts_as_approval": False,
            },
        )
        audit_ref = f"audit:{audit_event['event_id']}"
        preflight["audit_refs"] = [audit_ref]
        review["audit_refs"] = [audit_ref]
        _write_json(self._connector_action_preflight_path(preflight_id), preflight)
        _write_json(self._connector_action_preflight_review_path(review_id), review)
        _write_json(self.store.action_path(action_id), updated_action)
        return {
            "status": "success" if decision == "allow" else "preflight_denied",
            "action_card": updated_action,
            "connector_action_preflight": preflight,
            "connector_action_preflight_review": review,
            "audit_event": audit_event,
        }

    def _watch_negative_evidence(self) -> dict[str, int]:
        return dict(WATCH_RULE_NEGATIVE_EVIDENCE_TEMPLATE)

    def _watch_rule_source_refs(self, definition: dict[str, Any]) -> list[str]:
        sources = definition.get("sources") if isinstance(definition.get("sources"), list) else []
        refs: list[str] = []
        for source in sources:
            if isinstance(source, dict) and isinstance(source.get("source_ref"), str) and source["source_ref"].strip():
                refs.append(source["source_ref"])
        return refs

    def _watch_rule_connector_contract_refs(self, definition: dict[str, Any]) -> list[str]:
        refs = [str(ref) for ref in definition.get("connector_contract_refs", []) if isinstance(ref, str) and ref.strip()]
        for source in definition.get("sources", []):
            if isinstance(source, dict) and isinstance(source.get("connector_contract_ref"), str):
                refs.append(source["connector_contract_ref"])
        return sorted(set(refs))

    def _watch_rule_source_policy_refs(self, definition: dict[str, Any]) -> list[str]:
        refs = [str(ref) for ref in definition.get("source_policy_refs", []) if isinstance(ref, str) and ref.strip()]
        for source in definition.get("sources", []):
            if isinstance(source, dict) and isinstance(source.get("source_policy_ref"), str):
                refs.append(source["source_policy_ref"])
        return sorted(set(refs))

    def _watch_rule_definition_issues(
        self,
        *,
        definition: dict[str, Any],
        requested_scope: dict[str, str],
    ) -> list[dict[str, str]]:
        issues: list[dict[str, str]] = []
        if definition.get("schema_version") != "cs.watch_rule_definition.v1":
            issues.append(
                {
                    "code": "CS_WATCH_RULE_DEFINITION_SCHEMA_INVALID",
                    "message": "Watch Rule definitions must use cs.watch_rule_definition.v1.",
                    "path": "schema_version",
                }
            )
        scope = definition.get("scope")
        if not scope_complete(scope):
            issues.append(
                {
                    "code": "CS_WATCH_RULE_SCOPE_INCOMPLETE",
                    "message": "Watch Rule tenant/owner/namespace/workspace scope is required.",
                    "path": "scope",
                }
            )
        elif scope != requested_scope:
            issues.append(
                {
                    "code": "CS_WATCH_RULE_SCOPE_MISMATCH",
                    "message": "Watch Rule scope must match the trusted CLI scope.",
                    "path": "scope",
                }
            )
        if str(definition.get("owner_id") or requested_scope.get("owner_id")) != requested_scope.get("owner_id"):
            issues.append(
                {
                    "code": "CS_WATCH_RULE_OWNER_MISMATCH",
                    "message": "Watch Rule owner must match the trusted owner scope.",
                    "path": "owner_id",
                }
            )
        sources = definition.get("sources")
        if not isinstance(sources, list) or not sources:
            issues.append(
                {
                    "code": "CS_WATCH_RULE_SOURCES_REQUIRED",
                    "message": "At least one explicit source is required.",
                    "path": "sources",
                }
            )
        else:
            for index, source in enumerate(sources):
                if not isinstance(source, dict) or not source.get("source_ref"):
                    issues.append(
                        {
                            "code": "CS_WATCH_RULE_SOURCE_REF_REQUIRED",
                            "message": "Every Watch Rule source requires a source_ref.",
                            "path": f"sources[{index}].source_ref",
                        }
                    )
        if not self._watch_rule_connector_contract_refs(definition):
            issues.append(
                {
                    "code": "CS_WATCH_RULE_CONNECTOR_CONTRACT_REQUIRED",
                    "message": "Watch Rule creation requires Connector Capability Contract refs.",
                    "path": "connector_contract_refs",
                }
            )
        if not self._watch_rule_source_policy_refs(definition):
            issues.append(
                {
                    "code": "CS_WATCH_RULE_SOURCE_POLICY_REQUIRED",
                    "message": "Watch Rule creation requires Connector Source Policy refs.",
                    "path": "source_policy_refs",
                }
            )
        if not isinstance(definition.get("match_criteria"), dict) or not definition["match_criteria"]:
            issues.append(
                {
                    "code": "CS_WATCH_RULE_MATCH_CRITERIA_REQUIRED",
                    "message": "Explicit match criteria are required.",
                    "path": "match_criteria",
                }
            )
        if not isinstance(definition.get("schedule"), dict) or not definition["schedule"]:
            issues.append(
                {
                    "code": "CS_WATCH_RULE_SCHEDULE_REQUIRED",
                    "message": "A schedule or trigger window is required.",
                    "path": "schedule",
                }
            )
        allowed_outputs = definition.get("allowed_outputs")
        forbidden_outputs = {
            "external_action_execution",
            "provider_mutation",
            "workflow_execution",
            "connector_action_execution",
        }
        if not isinstance(allowed_outputs, list) or not allowed_outputs:
            issues.append(
                {
                    "code": "CS_WATCH_RULE_ALLOWED_OUTPUTS_REQUIRED",
                    "message": "Watch Rule allowed outputs must be explicit.",
                    "path": "allowed_outputs",
                }
            )
        elif any(str(output) in forbidden_outputs for output in allowed_outputs):
            issues.append(
                {
                    "code": "CS_WATCH_RULE_EXTERNAL_ACTION_AUTHORITY_FORBIDDEN",
                    "message": "Watch Rules cannot authorize external Action execution by themselves.",
                    "path": "allowed_outputs",
                }
            )
        authority = definition.get("authority") if isinstance(definition.get("authority"), dict) else {}
        if authority.get("external_action_execution") is True or authority.get("provider_mutation") is True:
            issues.append(
                {
                    "code": "CS_WATCH_RULE_EXTERNAL_ACTION_AUTHORITY_FORBIDDEN",
                    "message": "Natural-language rule text cannot expand action or provider mutation authority.",
                    "path": "authority",
                }
            )
        retention_days = definition.get("retention_days")
        if not isinstance(retention_days, int) or retention_days <= 0:
            issues.append(
                {
                    "code": "CS_WATCH_RULE_RETENTION_REQUIRED",
                    "message": "Watch Rule retention_days must be a positive integer.",
                    "path": "retention_days",
                }
            )
        return issues

    def _build_watch_rule_policy_decision(
        self,
        *,
        requested_scope: dict[str, str],
        watch_rule: dict[str, Any],
        watch_rule_version: dict[str, Any],
        action: str,
        decision: str,
        reason_codes: list[str],
        source_readiness: str,
        owner_confirmed: bool = False,
    ) -> dict[str, Any]:
        version_number = int(watch_rule_version.get("version_number", 0) or 0)
        decision_basis = {
            "source_readiness": source_readiness,
            "owner_confirmed": owner_confirmed,
            "reason_codes": reason_codes,
        }
        policy_decision_id = _watch_rule_policy_decision_id(
            requested_scope,
            watch_rule["watch_rule_id"],
            version_number,
            action,
            decision_basis,
        )
        source_refs = watch_rule_version.get("source_refs", [])
        return {
            "schema_version": WATCH_RULE_POLICY_DECISION_SCHEMA,
            "policy_decision_id": policy_decision_id,
            "watch_rule_id": watch_rule["watch_rule_id"],
            "watch_rule_version_id": watch_rule_version["watch_rule_version_id"],
            "version_number": version_number,
            "scope": requested_scope,
            "action": action,
            "decision": decision,
            "reason_codes": reason_codes,
            "source_readiness": source_readiness,
            "activation_allowed": action == "activate" and decision == "allow",
            "owner_confirmed": owner_confirmed,
            "checks": {
                "scope_owner_complete": scope_complete(requested_scope),
                "source_limited": bool(source_refs),
                "connector_contract_refs_present": bool(watch_rule_version.get("connector_contract_refs")),
                "source_policy_refs_present": bool(watch_rule_version.get("source_policy_refs")),
                "source_permissions_ready": source_readiness == "ready" if action == "activate" else True,
                "natural_language_authority_expanded": False,
                "external_action_authority_granted": False,
                "provider_mutation_authority_granted": False,
                "requires_connectorhub_source_fulfillment": True,
            },
            "negative_evidence": self._watch_negative_evidence(),
            "evidence_refs": [
                f"watch_rule_policy_decision:{policy_decision_id}",
                f"watch_rule:{watch_rule['watch_rule_id']}",
                f"watch_rule_version:{watch_rule_version['watch_rule_version_id']}",
            ],
            "audit_refs": [],
            "decided_at": utc_now(),
        }

    def _watch_rule_version_record(
        self,
        *,
        requested_scope: dict[str, str],
        watch_rule_id: str,
        definition: dict[str, Any],
        version_number: int,
        status: str,
        previous_version_id: str | None,
        version_diff: dict[str, Any],
        source_path: str,
    ) -> dict[str, Any]:
        definition_hash = json_hash(definition)
        watch_rule_version_id = _watch_rule_version_id(
            requested_scope,
            watch_rule_id,
            version_number,
            definition_hash,
        )
        return {
            "schema_version": WATCH_RULE_VERSION_SCHEMA,
            "watch_rule_version_id": watch_rule_version_id,
            "watch_rule_id": watch_rule_id,
            "version_number": version_number,
            "status": status,
            "scope": requested_scope,
            "definition_hash": definition_hash,
            "definition_snapshot": definition,
            "goal": definition.get("goal"),
            "source_refs": self._watch_rule_source_refs(definition),
            "connector_contract_refs": self._watch_rule_connector_contract_refs(definition),
            "source_policy_refs": self._watch_rule_source_policy_refs(definition),
            "match_criteria": definition.get("match_criteria", {}),
            "schedule": definition.get("schedule", {}),
            "sensitivity": definition.get("sensitivity", "standard"),
            "retention_days": definition.get("retention_days"),
            "allowed_outputs": definition.get("allowed_outputs", []),
            "authority": {
                "external_action_execution_allowed": False,
                "provider_mutation_allowed": False,
                "workflow_execution_requires_action_path": True,
                "natural_language_authority_ignored": True,
            },
            "previous_version_id": previous_version_id,
            "version_diff": version_diff,
            "created_from": {"path": source_path, "sha256": json_hash(definition)},
            "evidence_refs": [f"watch_rule_version:{watch_rule_version_id}"],
            "audit_refs": [],
            "created_at": utc_now(),
        }

    def create_watch_rule(
        self,
        *,
        requested_scope: dict[str, str],
        definition: dict[str, Any],
        source_path: str,
    ) -> dict[str, Any]:
        issues = self._watch_rule_definition_issues(definition=definition, requested_scope=requested_scope)
        if issues:
            return {"status": "failed", "issues": issues}
        rule_key = str(definition.get("watch_rule_slug") or definition.get("title") or definition.get("goal") or "watch-rule")
        watch_rule_id = _watch_rule_id(requested_scope, _slug(rule_key))
        if self._watch_rule_path(watch_rule_id).exists():
            return {
                "status": "failed",
                "issues": [
                    {
                        "code": "CS_WATCH_RULE_ALREADY_EXISTS",
                        "message": "A Watch Rule with the same scoped key already exists.",
                        "path": "watch_rule_slug",
                    }
                ],
            }
        version = self._watch_rule_version_record(
            requested_scope=requested_scope,
            watch_rule_id=watch_rule_id,
            definition=definition,
            version_number=1,
            status="draft",
            previous_version_id=None,
            version_diff={"change_type": "create", "changed_fields": sorted(definition.keys())},
            source_path=source_path,
        )
        watch_rule = {
            "schema_version": WATCH_RULE_SCHEMA,
            "watch_rule_id": watch_rule_id,
            "status": "draft",
            "lifecycle_state": "draft",
            "scope": requested_scope,
            "owner_id": requested_scope["owner_id"],
            "namespace_id": requested_scope["namespace_id"],
            "workspace_id": requested_scope["workspace_id"],
            "title": definition.get("title"),
            "goal": definition.get("goal"),
            "source_refs": version["source_refs"],
            "connector_contract_refs": version["connector_contract_refs"],
            "source_policy_refs": version["source_policy_refs"],
            "allowed_outputs": version["allowed_outputs"],
            "current_version_id": version["watch_rule_version_id"],
            "current_version_number": 1,
            "active_version_id": None,
            "pending_version_id": version["watch_rule_version_id"],
            "version_count": 1,
            "external_action_authority": False,
            "can_authorize_external_action_execution": False,
            "connectorhub_role": "fulfills_declared_source_capabilities_only",
            "negative_evidence": self._watch_negative_evidence(),
            "change_history": [
                {
                    "event": "created",
                    "from_status": None,
                    "to_status": "draft",
                    "watch_rule_version_id": version["watch_rule_version_id"],
                    "at": utc_now(),
                }
            ],
            "evidence_refs": [f"watch_rule:{watch_rule_id}", f"watch_rule_version:{version['watch_rule_version_id']}"],
            "audit_refs": [],
            "created_at": utc_now(),
            "updated_at": utc_now(),
        }
        policy_decision = self._build_watch_rule_policy_decision(
            requested_scope=requested_scope,
            watch_rule=watch_rule,
            watch_rule_version=version,
            action="create",
            decision="allow",
            reason_codes=["CS_WATCH_RULE_DRAFT_CREATED"],
            source_readiness="not_evaluated",
        )
        audit_event = self.store.append_audit(
            "watch.rule.created",
            requested_scope,
            {"type": "watch_rule", "id": watch_rule_id},
            {
                "watch_rule_version_id": version["watch_rule_version_id"],
                "status": "draft",
                "external_action_authority": False,
            },
        )
        audit_ref = f"audit:{audit_event['event_id']}"
        watch_rule["audit_refs"] = [audit_ref]
        watch_rule["last_policy_decision_id"] = policy_decision["policy_decision_id"]
        watch_rule["policy_decision_refs"] = [f"policy:{policy_decision['policy_decision_id']}"]
        version["audit_refs"] = [audit_ref]
        policy_decision["audit_refs"] = [audit_ref]
        _write_json(self._watch_rule_path(watch_rule_id), watch_rule)
        _write_json(self._watch_rule_version_path(version["watch_rule_version_id"]), version)
        _write_json(self._watch_rule_policy_decision_path(policy_decision["policy_decision_id"]), policy_decision)
        return {
            "status": "success",
            "watch_rule": watch_rule,
            "watch_rule_version": version,
            "watch_rule_policy_decision": policy_decision,
            "audit_event": audit_event,
        }

    def show_watch_rule(self, *, requested_scope: dict[str, str], watch_rule_id: str) -> dict[str, Any]:
        watch_rule = self._load_watch_rule(watch_rule_id)
        if watch_rule is None:
            return {"status": "not_found"}
        if watch_rule.get("scope") != requested_scope:
            return {"status": "scope_denied", "resource_scope": watch_rule.get("scope")}
        current_version = self._load_watch_rule_version(str(watch_rule.get("current_version_id"))) or {}
        active_version = (
            self._load_watch_rule_version(str(watch_rule.get("active_version_id")))
            if watch_rule.get("active_version_id")
            else None
        )
        audit_event = self.store.append_audit(
            "watch.rule.read",
            requested_scope,
            {"type": "watch_rule", "id": watch_rule_id},
            {"status": watch_rule.get("status"), "current_version_id": watch_rule.get("current_version_id")},
        )
        return {
            "status": "success",
            "watch_rule": watch_rule,
            "watch_rule_version": current_version,
            "active_watch_rule_version": active_version,
            "audit_event": audit_event,
        }

    def activate_watch_rule(
        self,
        *,
        requested_scope: dict[str, str],
        watch_rule_id: str,
        source_readiness: str,
    ) -> dict[str, Any]:
        if source_readiness not in WATCH_SOURCE_READINESS_STATES:
            return {
                "status": "failed",
                "issues": [
                    {
                        "code": "CS_WATCH_RULE_SOURCE_READINESS_INVALID",
                        "message": "source_readiness must be ready or missing.",
                        "path": "source_readiness",
                    }
                ],
            }
        watch_rule = self._load_watch_rule(watch_rule_id)
        if watch_rule is None:
            return {"status": "not_found"}
        if watch_rule.get("scope") != requested_scope:
            return {"status": "scope_denied", "resource_scope": watch_rule.get("scope")}
        version = self._load_watch_rule_version(str(watch_rule.get("current_version_id")))
        if version is None:
            return {"status": "not_found"}
        if source_readiness != "ready":
            policy_decision = self._build_watch_rule_policy_decision(
                requested_scope=requested_scope,
                watch_rule=watch_rule,
                watch_rule_version=version,
                action="activate",
                decision="deny",
                reason_codes=["CS_WATCH_RULE_SOURCE_NOT_READY"],
                source_readiness=source_readiness,
            )
            audit_event = self.store.append_audit(
                "watch.rule.activation_denied",
                requested_scope,
                {"type": "watch_rule", "id": watch_rule_id},
                {
                    "watch_rule_version_id": version["watch_rule_version_id"],
                    "source_readiness": source_readiness,
                    "reason_codes": policy_decision["reason_codes"],
                },
            )
            audit_ref = f"audit:{audit_event['event_id']}"
            policy_decision["audit_refs"] = [audit_ref]
            watch_rule["last_policy_decision_id"] = policy_decision["policy_decision_id"]
            watch_rule["policy_decision_refs"] = list(
                dict.fromkeys([*watch_rule.get("policy_decision_refs", []), f"policy:{policy_decision['policy_decision_id']}"])
            )
            watch_rule["change_history"].append(
                {
                    "event": "activation_denied",
                    "from_status": watch_rule.get("status"),
                    "to_status": watch_rule.get("status"),
                    "watch_rule_version_id": version["watch_rule_version_id"],
                    "reason_codes": policy_decision["reason_codes"],
                    "at": utc_now(),
                }
            )
            watch_rule["updated_at"] = utc_now()
            _write_json(self._watch_rule_path(watch_rule_id), watch_rule)
            _write_json(self._watch_rule_policy_decision_path(policy_decision["policy_decision_id"]), policy_decision)
            return {
                "status": "source_not_ready",
                "watch_rule": watch_rule,
                "watch_rule_version": version,
                "watch_rule_policy_decision": policy_decision,
                "audit_event": audit_event,
            }
        previous_status = str(watch_rule.get("status"))
        version["status"] = "active"
        version["activated_at"] = utc_now()
        watch_rule["status"] = "active"
        watch_rule["lifecycle_state"] = "active"
        watch_rule["active_version_id"] = version["watch_rule_version_id"]
        watch_rule["pending_version_id"] = None
        watch_rule["change_history"].append(
            {
                "event": "activated",
                "from_status": previous_status,
                "to_status": "active",
                "watch_rule_version_id": version["watch_rule_version_id"],
                "at": utc_now(),
            }
        )
        watch_rule["updated_at"] = utc_now()
        policy_decision = self._build_watch_rule_policy_decision(
            requested_scope=requested_scope,
            watch_rule=watch_rule,
            watch_rule_version=version,
            action="activate",
            decision="allow",
            reason_codes=["CS_WATCH_RULE_SOURCES_READY"],
            source_readiness=source_readiness,
        )
        audit_event = self.store.append_audit(
            "watch.rule.activated",
            requested_scope,
            {"type": "watch_rule", "id": watch_rule_id},
            {
                "watch_rule_version_id": version["watch_rule_version_id"],
                "source_readiness": source_readiness,
                "external_action_authority": False,
            },
        )
        audit_ref = f"audit:{audit_event['event_id']}"
        version["audit_refs"] = list(dict.fromkeys([*version.get("audit_refs", []), audit_ref]))
        policy_decision["audit_refs"] = [audit_ref]
        watch_rule["audit_refs"] = list(dict.fromkeys([*watch_rule.get("audit_refs", []), audit_ref]))
        watch_rule["last_policy_decision_id"] = policy_decision["policy_decision_id"]
        watch_rule["policy_decision_refs"] = list(
            dict.fromkeys([*watch_rule.get("policy_decision_refs", []), f"policy:{policy_decision['policy_decision_id']}"])
        )
        _write_json(self._watch_rule_path(watch_rule_id), watch_rule)
        _write_json(self._watch_rule_version_path(version["watch_rule_version_id"]), version)
        _write_json(self._watch_rule_policy_decision_path(policy_decision["policy_decision_id"]), policy_decision)
        return {
            "status": "success",
            "watch_rule": watch_rule,
            "watch_rule_version": version,
            "watch_rule_policy_decision": policy_decision,
            "audit_event": audit_event,
        }

    def transition_watch_rule(
        self,
        *,
        requested_scope: dict[str, str],
        watch_rule_id: str,
        transition: str,
    ) -> dict[str, Any]:
        target_status = {
            "pause": "paused",
            "resume": "active",
            "delete": "deleted",
        }.get(transition)
        if target_status is None:
            return {"status": "failed", "issues": [{"code": "CS_WATCH_RULE_TRANSITION_INVALID", "message": "Unknown Watch Rule transition.", "path": "transition"}]}
        watch_rule = self._load_watch_rule(watch_rule_id)
        if watch_rule is None:
            return {"status": "not_found"}
        if watch_rule.get("scope") != requested_scope:
            return {"status": "scope_denied", "resource_scope": watch_rule.get("scope")}
        active_version_id = watch_rule.get("active_version_id") or watch_rule.get("current_version_id")
        version = self._load_watch_rule_version(str(active_version_id)) or {}
        previous_status = str(watch_rule.get("status"))
        if transition == "resume" and not watch_rule.get("active_version_id"):
            return {
                "status": "failed",
                "issues": [
                    {
                        "code": "CS_WATCH_RULE_ACTIVE_VERSION_REQUIRED",
                        "message": "A paused Watch Rule can resume only when an active version exists.",
                        "path": "active_version_id",
                    }
                ],
            }
        watch_rule["status"] = target_status
        watch_rule["lifecycle_state"] = "disabled" if transition == "delete" else target_status
        if transition == "delete":
            watch_rule["physical_delete_performed"] = False
            watch_rule["retained_for_audit"] = True
        watch_rule["change_history"].append(
            {
                "event": transition,
                "from_status": previous_status,
                "to_status": target_status,
                "watch_rule_version_id": active_version_id,
                "physical_delete_performed": False if transition == "delete" else None,
                "at": utc_now(),
            }
        )
        watch_rule["updated_at"] = utc_now()
        audit_event = self.store.append_audit(
            f"watch.rule.{transition}d" if transition != "pause" else "watch.rule.paused",
            requested_scope,
            {"type": "watch_rule", "id": watch_rule_id},
            {
                "from_status": previous_status,
                "to_status": target_status,
                "watch_rule_version_id": active_version_id,
                "physical_delete_performed": False if transition == "delete" else None,
            },
        )
        audit_ref = f"audit:{audit_event['event_id']}"
        watch_rule["audit_refs"] = list(dict.fromkeys([*watch_rule.get("audit_refs", []), audit_ref]))
        _write_json(self._watch_rule_path(watch_rule_id), watch_rule)
        return {
            "status": "success",
            "watch_rule": watch_rule,
            "watch_rule_version": version,
            "audit_event": audit_event,
        }

    def _watch_rule_changed_fields(self, previous: dict[str, Any], current: dict[str, Any]) -> list[str]:
        fields = [
            "goal",
            "sources",
            "match_criteria",
            "schedule",
            "sensitivity",
            "retention_days",
            "allowed_outputs",
            "connector_contract_refs",
            "source_policy_refs",
        ]
        return [field for field in fields if previous.get(field) != current.get(field)]

    def _watch_rule_broadening_fields(self, previous: dict[str, Any], current: dict[str, Any]) -> list[str]:
        broadened: list[str] = []
        previous_sources = set(self._watch_rule_source_refs(previous))
        current_sources = set(self._watch_rule_source_refs(current))
        if current_sources - previous_sources:
            broadened.append("sources")
        previous_outputs = {str(output) for output in previous.get("allowed_outputs", [])}
        current_outputs = {str(output) for output in current.get("allowed_outputs", [])}
        if current_outputs - previous_outputs:
            broadened.append("allowed_outputs")
        if int(current.get("retention_days", 0) or 0) > int(previous.get("retention_days", 0) or 0):
            broadened.append("retention_days")
        return broadened

    def edit_watch_rule(
        self,
        *,
        requested_scope: dict[str, str],
        watch_rule_id: str,
        definition: dict[str, Any],
        source_path: str,
        owner_confirmed: bool,
    ) -> dict[str, Any]:
        watch_rule = self._load_watch_rule(watch_rule_id)
        if watch_rule is None:
            return {"status": "not_found"}
        if watch_rule.get("scope") != requested_scope:
            return {"status": "scope_denied", "resource_scope": watch_rule.get("scope")}
        issues = self._watch_rule_definition_issues(definition=definition, requested_scope=requested_scope)
        if issues:
            return {"status": "failed", "issues": issues}
        previous_version = self._load_watch_rule_version(str(watch_rule.get("current_version_id")))
        if previous_version is None:
            return {"status": "not_found"}
        previous_definition = previous_version.get("definition_snapshot", {})
        changed_fields = self._watch_rule_changed_fields(previous_definition, definition)
        broadened_fields = self._watch_rule_broadening_fields(previous_definition, definition)
        if broadened_fields and not owner_confirmed:
            policy_decision = self._build_watch_rule_policy_decision(
                requested_scope=requested_scope,
                watch_rule=watch_rule,
                watch_rule_version=previous_version,
                action="edit",
                decision="deny",
                reason_codes=["CS_WATCH_RULE_BROADENING_REQUIRES_CONFIRMATION"],
                source_readiness="not_evaluated",
                owner_confirmed=False,
            )
            audit_event = self.store.append_audit(
                "watch.rule.edit_denied",
                requested_scope,
                {"type": "watch_rule", "id": watch_rule_id},
                {
                    "previous_version_id": previous_version["watch_rule_version_id"],
                    "broadened_fields": broadened_fields,
                    "owner_confirmed": False,
                },
            )
            audit_ref = f"audit:{audit_event['event_id']}"
            policy_decision["audit_refs"] = [audit_ref]
            _write_json(self._watch_rule_policy_decision_path(policy_decision["policy_decision_id"]), policy_decision)
            return {
                "status": "broadening_requires_confirmation",
                "watch_rule": watch_rule,
                "watch_rule_version": previous_version,
                "watch_rule_policy_decision": policy_decision,
                "audit_event": audit_event,
                "broadened_fields": broadened_fields,
            }
        next_version_number = int(watch_rule.get("version_count", 1) or 1) + 1
        version = self._watch_rule_version_record(
            requested_scope=requested_scope,
            watch_rule_id=watch_rule_id,
            definition=definition,
            version_number=next_version_number,
            status="draft",
            previous_version_id=previous_version["watch_rule_version_id"],
            version_diff={
                "change_type": "edit",
                "changed_fields": changed_fields,
                "broadened_fields": broadened_fields,
                "owner_confirmed_for_broadening": bool(owner_confirmed and broadened_fields),
            },
            source_path=source_path,
        )
        previous_status = str(watch_rule.get("status"))
        watch_rule["current_version_id"] = version["watch_rule_version_id"]
        watch_rule["current_version_number"] = next_version_number
        watch_rule["pending_version_id"] = version["watch_rule_version_id"]
        watch_rule["version_count"] = next_version_number
        watch_rule["status"] = "active_pending_review" if watch_rule.get("active_version_id") else "draft"
        watch_rule["lifecycle_state"] = "active_pending_review" if watch_rule.get("active_version_id") else "draft"
        watch_rule["source_refs"] = version["source_refs"]
        watch_rule["connector_contract_refs"] = version["connector_contract_refs"]
        watch_rule["source_policy_refs"] = version["source_policy_refs"]
        watch_rule["allowed_outputs"] = version["allowed_outputs"]
        watch_rule["change_history"].append(
            {
                "event": "edited",
                "from_status": previous_status,
                "to_status": watch_rule["status"],
                "previous_version_id": previous_version["watch_rule_version_id"],
                "watch_rule_version_id": version["watch_rule_version_id"],
                "changed_fields": changed_fields,
                "broadened_fields": broadened_fields,
                "at": utc_now(),
            }
        )
        watch_rule["updated_at"] = utc_now()
        policy_decision = self._build_watch_rule_policy_decision(
            requested_scope=requested_scope,
            watch_rule=watch_rule,
            watch_rule_version=version,
            action="edit",
            decision="allow",
            reason_codes=["CS_WATCH_RULE_VERSION_CREATED"],
            source_readiness="not_evaluated",
            owner_confirmed=owner_confirmed,
        )
        audit_event = self.store.append_audit(
            "watch.rule.edited",
            requested_scope,
            {"type": "watch_rule", "id": watch_rule_id},
            {
                "previous_version_id": previous_version["watch_rule_version_id"],
                "watch_rule_version_id": version["watch_rule_version_id"],
                "changed_fields": changed_fields,
                "broadened_fields": broadened_fields,
            },
        )
        audit_ref = f"audit:{audit_event['event_id']}"
        version["audit_refs"] = [audit_ref]
        watch_rule["audit_refs"] = list(dict.fromkeys([*watch_rule.get("audit_refs", []), audit_ref]))
        watch_rule["last_policy_decision_id"] = policy_decision["policy_decision_id"]
        watch_rule["policy_decision_refs"] = list(
            dict.fromkeys([*watch_rule.get("policy_decision_refs", []), f"policy:{policy_decision['policy_decision_id']}"])
        )
        policy_decision["audit_refs"] = [audit_ref]
        _write_json(self._watch_rule_path(watch_rule_id), watch_rule)
        _write_json(self._watch_rule_version_path(version["watch_rule_version_id"]), version)
        _write_json(self._watch_rule_policy_decision_path(policy_decision["policy_decision_id"]), policy_decision)
        return {
            "status": "success",
            "watch_rule": watch_rule,
            "watch_rule_version": version,
            "previous_watch_rule_version": previous_version,
            "watch_rule_policy_decision": policy_decision,
            "audit_event": audit_event,
        }

    def evaluate_watch_rule(
        self,
        *,
        requested_scope: dict[str, str],
        watch_rule_id: str,
        source_evidence_refs: list[str],
    ) -> dict[str, Any]:
        watch_rule = self._load_watch_rule(watch_rule_id)
        if watch_rule is None:
            return {"status": "not_found"}
        if watch_rule.get("scope") != requested_scope:
            return {"status": "scope_denied", "resource_scope": watch_rule.get("scope")}
        version_id = str(watch_rule.get("active_version_id") or watch_rule.get("current_version_id"))
        version = self._load_watch_rule_version(version_id)
        if version is None:
            return {"status": "not_found"}
        source_refs = source_evidence_refs or [f"source:{source_ref}" for source_ref in version.get("source_refs", [])]
        evaluation_trace_id = _watch_rule_evaluation_trace_id(
            requested_scope,
            watch_rule_id,
            version["watch_rule_version_id"],
            source_refs,
        )
        trace = {
            "schema_version": WATCH_RULE_EVALUATION_TRACE_SCHEMA,
            "evaluation_trace_id": evaluation_trace_id,
            "watch_rule_id": watch_rule_id,
            "watch_rule_version_id": version["watch_rule_version_id"],
            "version_number": version["version_number"],
            "scope": requested_scope,
            "status": "matched",
            "source_evidence_refs": source_refs,
            "source_refs": version.get("source_refs", []),
            "match_criteria": version.get("match_criteria", {}),
            "generated_outputs": ["watch_result"],
            "external_action_authority": False,
            "provider_mutation_authority": False,
            "action_card_created": False,
            "source_evidence_required": True,
            "negative_evidence": self._watch_negative_evidence(),
            "evidence_refs": [
                f"watch_rule_evaluation_trace:{evaluation_trace_id}",
                f"watch_rule:{watch_rule_id}",
                f"watch_rule_version:{version['watch_rule_version_id']}",
            ],
            "audit_refs": [],
            "evaluated_at": utc_now(),
        }
        audit_event = self.store.append_audit(
            "watch.rule.evaluated",
            requested_scope,
            {"type": "watch_rule_evaluation_trace", "id": evaluation_trace_id},
            {
                "watch_rule_id": watch_rule_id,
                "watch_rule_version_id": version["watch_rule_version_id"],
                "external_action_authority": False,
            },
        )
        audit_ref = f"audit:{audit_event['event_id']}"
        trace["audit_refs"] = [audit_ref]
        _write_json(self._watch_rule_evaluation_trace_path(evaluation_trace_id), trace)
        return {
            "status": "success",
            "watch_rule": watch_rule,
            "watch_rule_version": version,
            "watch_rule_evaluation_trace": trace,
            "audit_event": audit_event,
        }

    def simulate_github_provider_failure(
        self,
        *,
        requested_scope: dict[str, str],
        contract_id: str,
        source_ref: str,
        failure_mode: str,
        provider_pack_id: str = "local_source_control_readonly.v1",
    ) -> dict[str, Any]:
        fixture = GITHUB_PROVIDER_FAILURE_FIXTURES.get(failure_mode)
        if fixture is None:
            return {
                "status": "failed",
                "issues": [
                    {
                        "code": "CS_CONNECTOR_GITHUB_FAILURE_MODE_INVALID",
                        "message": f"failure_mode must be one of: {', '.join(GITHUB_PROVIDER_FAILURE_MODES)}.",
                        "path": "failure_mode",
                    }
                ],
            }
        contract = self.load_contract(contract_id)
        if contract is None:
            return {"status": "not_found", "contract_id": contract_id}
        if contract.get("scope") != requested_scope:
            return {"status": "scope_denied", "resource_scope": contract.get("scope")}
        setup_result = self._latest_setup_result(contract, requested_scope, provider_pack_id)
        if setup_result is None:
            return {
                "status": "setup_missing",
                "issues": [
                    {
                        "code": "CS_CONNECTOR_SETUP_RESULT_MISSING",
                        "message": "Connector setup must exist before provider failure state can be recorded.",
                        "path": "connector_setup_result",
                    }
                ],
            }
        source_policy = setup_result.get("source_policy_snapshot") or {}
        selected_resources = list(source_policy.get("selected_resources") or contract.get("source_policy_request", {}).get("selected_resources") or [])
        if source_ref not in selected_resources:
            return {
                "status": "source_denied",
                "issues": [
                    {
                        "code": "CS_CONNECTOR_SOURCE_NOT_SELECTED",
                        "message": "Provider failure state can only be recorded for an owner-selected source.",
                        "path": "source_ref",
                    }
                ],
            }

        prior_receipts: list[dict[str, Any]] = []
        if self.delivery_receipt_dir.exists():
            for path in sorted(self.delivery_receipt_dir.glob("*.json")):
                receipt = json.loads(path.read_text())
                receipt_source = receipt.get("source_summary") if isinstance(receipt.get("source_summary"), dict) else {}
                source_external_id = str(receipt.get("source_external_id") or "")
                if (
                    receipt.get("scope") == requested_scope
                    and receipt.get("contract_id") == contract_id
                    and (receipt_source.get("source_ref") == source_ref or source_external_id.startswith(f"{source_ref}:"))
                ):
                    prior_receipts.append(receipt)
        prior_artifacts = [
            artifact
            for artifact in (
                self.store.get_artifact(str(receipt.get("artifact_id")), requested_scope)
                for receipt in prior_receipts
                if receipt.get("artifact_id")
            )
            if artifact is not None
        ]
        retry_after_seconds = fixture.get("retry_after_seconds")
        retry_scheduled = isinstance(retry_after_seconds, int) and retry_after_seconds > 0
        now = utc_now()
        failure_state_id = _provider_failure_state_id(requested_scope, contract_id, source_ref, failure_mode)
        prior_evidence_refs: list[str] = []
        for receipt in prior_receipts:
            delivery_receipt_id = receipt.get("delivery_receipt_id")
            artifact_id = receipt.get("artifact_id")
            if delivery_receipt_id:
                prior_evidence_refs.append(f"connector_delivery_receipt:{delivery_receipt_id}")
            if artifact_id:
                prior_evidence_refs.append(f"artifact:{artifact_id}")
        failure_state = {
            "schema_version": PROVIDER_FAILURE_STATE_SCHEMA,
            "provider_failure_state_id": failure_state_id,
            "provider": "github",
            "provider_pack_id": provider_pack_id,
            "failure_mode": failure_mode,
            "reason_code": fixture["reason_code"],
            "message": fixture["message"],
            "scope": requested_scope,
            "contract_id": contract["contract_id"],
            "contract_version_id": contract["contract_version_id"],
            "setup_result_id": setup_result.get("setup_result_id"),
            "source_policy_id": source_policy.get("source_policy_id"),
            "source_ref": source_ref,
            "source_health": {
                "state": fixture["health_state"],
                "source_availability": fixture["source_availability"],
                "recoverability": fixture["recoverability"],
                "visible_in_health": True,
                "visible_in_mission_control": True,
            },
            "setup_gap": {
                "state": fixture["setup_state"],
                "permanent": bool(fixture["setup_gap_permanent"]),
                "reason_code": fixture["reason_code"],
                "stream_state": fixture["stream_state"],
                "owner_action_required": bool(fixture["owner_action_required"]),
                "requires_new_verification": bool(fixture["requires_new_verification"]),
            },
            "retry_policy": {
                "status": "scheduled" if retry_scheduled else "not_scheduled",
                "retry_after_seconds": retry_after_seconds,
                "next_retry_at": utc_after(int(retry_after_seconds)) if retry_scheduled else None,
                "tight_retry_loop_prevented": True,
                "max_attempts_before_quarantine": 3,
                "backoff_family": "bounded_exponential" if retry_scheduled else "not_applicable",
            },
            "freshness": {
                "state": fixture["freshness_state"],
                "current_data_claim_allowed": bool(fixture["current_data_claim_allowed"]),
                "fresh_sync_claim_allowed": False,
                "warning_required": True,
                "warning_text": f"GitHub source {source_ref} is {fixture['freshness_state']} because {fixture['reason_code']}.",
            },
            "ingestion_control": {
                "stream_state": fixture["stream_state"],
                "future_ingestion_allowed": bool(fixture["future_ingestion_allowed"]),
                "stop_removed_scope": failure_mode == "repository_removed",
                "suspend_affected_streams": failure_mode == "permission_revoked",
                "cursor_advanced": False,
                "provider_ack_sent": False,
                "external_http_calls": 0,
                "provider_mutations": 0,
            },
            "recovery_path": {
                "owner_action_required": bool(fixture["owner_action_required"]),
                "requires_new_verification": bool(fixture["requires_new_verification"]),
                "automatic_retry_allowed": fixture["recoverability"] == "automatic_retry",
                "automatic_reconnect_allowed": False,
                "recommended_owner_action": "wait_for_retry"
                if fixture["recoverability"] == "automatic_retry"
                else "reconnect_or_reselect_source",
            },
            "existing_evidence": {
                "delivery_receipt_count_before": len(prior_receipts),
                "artifact_count_before": len(prior_artifacts),
                "delivery_receipt_ids": [receipt["delivery_receipt_id"] for receipt in prior_receipts if receipt.get("delivery_receipt_id")],
                "artifact_ids": [artifact["artifact_id"] for artifact in prior_artifacts if artifact.get("artifact_id")],
                "existing_artifacts_preserved": True,
                "delete_existing_evidence": False,
                "mark_existing_evidence_stale_or_unavailable": True,
            },
            "surface_warnings": {
                "search_result_warning": {
                    "source_ref": source_ref,
                    "source_freshness": fixture["freshness_state"],
                    "source_availability": fixture["source_availability"],
                    "current_data_claim_allowed": bool(fixture["current_data_claim_allowed"]),
                    "warning_required": True,
                },
                "claim_warning": {
                    "requires_freshness_qualification": True,
                    "unsupported_fresh_claim_denied": True,
                    "current_data_claim_allowed": bool(fixture["current_data_claim_allowed"]),
                },
                "mission_control_status": fixture["health_state"],
                "setup_result_status": fixture["setup_state"],
            },
            "negative_evidence": {
                "silent_data_deletions": 0,
                "fresh_sync_claims_while_suspended": 0,
                "tight_retry_loops": 0,
                "fabricated_current_data": 0,
                "removed_repository_future_ingestions": 0,
                "revoked_permission_streams_active": 0,
                "reconnect_without_owner_action": 0,
                "external_http_calls": 0,
                "provider_mutations": 0,
            },
            "evidence_refs": [
                f"connector_contract:{contract['contract_version_id']}",
                f"connector_setup_result:{setup_result['setup_result_id']}",
                f"connector_provider_failure_state:{failure_state_id}",
                *prior_evidence_refs,
            ],
            "created_at": now,
            "updated_at": now,
        }
        audit_event = self.store.append_audit(
            "connector.github_failure_state.recorded",
            requested_scope,
            {"type": "connector_provider_failure_state", "id": failure_state_id},
            {
                "provider": "github",
                "failure_mode": failure_mode,
                "reason_code": fixture["reason_code"],
                "source_ref": source_ref,
                "stream_state": fixture["stream_state"],
                "future_ingestion_allowed": bool(fixture["future_ingestion_allowed"]),
                "existing_artifact_count": len(prior_artifacts),
                "external_http_calls": 0,
                "provider_mutations": 0,
            },
        )
        failure_state["audit_refs"] = [f"audit:{audit_event['event_id']}"]
        _write_json(self._provider_failure_state_path(failure_state_id), failure_state)
        return {
            "status": "success",
            "connector_provider_failure_state": failure_state,
            "audit_event": audit_event,
        }

    def reconcile_delivery_ack_state(self, requested_scope: dict[str, str]) -> dict[str, Any]:
        receipts: list[dict[str, Any]] = []
        if self.delivery_receipt_dir.exists():
            for path in sorted(self.delivery_receipt_dir.glob("*.json")):
                receipt = json.loads(path.read_text())
                if receipt.get("scope") == requested_scope:
                    receipts.append(receipt)
        outboxes: list[dict[str, Any]] = []
        if self.ack_outbox_dir.exists():
            for path in sorted(self.ack_outbox_dir.glob("*.json")):
                outbox = json.loads(path.read_text())
                if outbox.get("scope") == requested_scope:
                    outboxes.append(outbox)

        receipt_by_id = {receipt.get("delivery_receipt_id"): receipt for receipt in receipts}
        artifacts_by_id: dict[str, dict[str, Any]] = {}
        for receipt in receipts:
            artifact_id = str(receipt.get("artifact_id", ""))
            artifact = self.store.get_artifact(artifact_id, requested_scope) if artifact_id else None
            if artifact:
                artifacts_by_id[artifact_id] = artifact

        acknowledged_without_artifact = [
            outbox
            for outbox in outboxes
            if outbox.get("status") == "acknowledged"
            and outbox.get("artifact_id") not in artifacts_by_id
        ]
        orphan_artifacts = [
            artifact_id
            for artifact_id in artifacts_by_id
            if not any(receipt.get("artifact_id") == artifact_id for receipt in receipts)
        ]
        artifacts_by_key: dict[str, set[str]] = {}
        for receipt in receipts:
            key = str(receipt.get("idempotency_key") or receipt.get("delivery_id") or "")
            artifacts_by_key.setdefault(key, set()).add(str(receipt.get("artifact_id", "")))
        duplicate_logical_artifacts = {
            key: sorted(artifact_ids)
            for key, artifact_ids in artifacts_by_key.items()
            if key and len({artifact_id for artifact_id in artifact_ids if artifact_id}) > 1
        }
        pending_outboxes = [outbox for outbox in outboxes if outbox.get("status") == "pending"]
        reconciliation = {
            "schema_version": ACK_RECONCILIATION_SCHEMA,
            "scope": requested_scope,
            "status": "success"
            if not acknowledged_without_artifact and not orphan_artifacts and not duplicate_logical_artifacts
            else "failed",
            "receipt_count": len(receipts),
            "ack_outbox_count": len(outboxes),
            "artifact_count": len(artifacts_by_id),
            "pending_ack_count": len(pending_outboxes),
            "acknowledged_without_artifact_count": len(acknowledged_without_artifact),
            "orphan_artifact_count": len(orphan_artifacts),
            "duplicate_logical_artifact_count": len(duplicate_logical_artifacts),
            "duplicate_logical_artifacts": duplicate_logical_artifacts,
            "checked_at": utc_now(),
        }
        audit_event = self.store.append_audit(
            "connector.delivery.ack_reconciled",
            requested_scope,
            {"type": "connector_ack_reconciliation", "id": "latest"},
            {
                "status": reconciliation["status"],
                "receipt_count": reconciliation["receipt_count"],
                "ack_outbox_count": reconciliation["ack_outbox_count"],
                "acknowledged_without_artifact_count": reconciliation["acknowledged_without_artifact_count"],
                "duplicate_logical_artifact_count": reconciliation["duplicate_logical_artifact_count"],
            },
        )
        reconciliation["audit_refs"] = [f"audit:{audit_event['event_id']}"]
        return {"status": reconciliation["status"], "ack_reconciliation": reconciliation, "audit_event": audit_event}

    def show_untrusted_content_review(
        self,
        requested_scope: dict[str, str],
        *,
        delivery_receipt_id: str | None = None,
        review_id: str | None = None,
    ) -> dict[str, Any]:
        if not delivery_receipt_id and not review_id:
            return {
                "status": "failed",
                "issues": [
                    {
                        "code": "CS_CONNECTOR_UNTRUSTED_REVIEW_SELECTOR_REQUIRED",
                        "message": "delivery_receipt_id or review_id is required to inspect an untrusted content review.",
                        "path": "delivery_receipt_id",
                    }
                ],
            }

        receipt = self._load_delivery_receipt(delivery_receipt_id) if delivery_receipt_id else None
        if delivery_receipt_id and receipt is None:
            return {"status": "not_found", "resource": "connector_delivery_receipt", "id": delivery_receipt_id}
        if receipt and receipt.get("scope") != requested_scope:
            return {"status": "scope_denied", "resource_scope": receipt.get("scope")}

        selected_review_id = str(
            review_id
            or (receipt.get("untrusted_content_review", {}).get("review_id") if receipt else "")
            or ""
        )
        if not selected_review_id:
            return {
                "status": "not_found",
                "resource": "connector_untrusted_content_review",
                "id": "",
                "issues": [
                    {
                        "code": "CS_CONNECTOR_UNTRUSTED_REVIEW_NOT_LINKED",
                        "message": "The selected Connector Delivery receipt has no untrusted content review link.",
                        "path": "delivery_receipt_id",
                    }
                ],
            }

        review = self._load_untrusted_content_review(selected_review_id)
        if review is None:
            return {"status": "not_found", "resource": "connector_untrusted_content_review", "id": selected_review_id}
        if review.get("scope") != requested_scope:
            return {"status": "scope_denied", "resource_scope": review.get("scope")}

        if receipt is None:
            receipt = self._load_delivery_receipt(str(review.get("delivery_receipt_id", "")))
            if receipt and receipt.get("scope") != requested_scope:
                return {"status": "scope_denied", "resource_scope": receipt.get("scope")}
        artifact_id = str(review.get("artifact_id") or (receipt or {}).get("artifact_id") or "")
        artifact = self.store.get_artifact(artifact_id, requested_scope) if artifact_id else None
        audit_event = self.store.append_audit(
            "connector.untrusted_content.review_read",
            requested_scope,
            {"type": "connector_untrusted_content_review", "id": review["review_id"]},
            {
                "delivery_receipt_id": review.get("delivery_receipt_id"),
                "artifact_id": review.get("artifact_id"),
                "unsafe_instruction_detected": review.get("unsafe_instruction_detected"),
                "blocked_attempt_count": review.get("blocked_attempt_count"),
            },
        )
        review["audit_refs"] = list(
            dict.fromkeys([*review.get("audit_refs", []), f"audit:{audit_event['event_id']}"])
        )
        _write_json(self._untrusted_content_review_path(review["review_id"]), review)
        return {
            "status": "success",
            "connector_untrusted_content_review": review,
            "delivery_receipt": receipt,
            "artifact": artifact,
            "audit_event": audit_event,
        }

    def assemble_evidence_bundle(
        self,
        requested_scope: dict[str, str],
        delivery_receipt_id: str | None = None,
        evidence_ref_id: str | None = None,
        query: str | None = None,
    ) -> dict[str, Any]:
        if evidence_ref_id and not delivery_receipt_id:
            audit_event = self.store.append_audit(
                "connector.evidence_bundle.denied",
                requested_scope,
                {"type": "connector_evidence_ref", "id": evidence_ref_id},
                {
                    "reason": "evidence_ref_only",
                    "required": "Evidence Bundle assembly requires an immutable Artifact and Connector Delivery receipt.",
                    "evidence_ref_only_approved_truth": False,
                },
            )
            return {
                "status": "evidence_ref_only",
                "evidence_ref_id": evidence_ref_id,
                "audit_event": audit_event,
                "issues": [
                    {
                        "code": "CS_CONNECTOR_EVIDENCE_REF_ONLY_UNSUPPORTED",
                        "message": "EvidenceRef metadata is provenance input only; assemble an Evidence Bundle from a committed Artifact and Delivery receipt.",
                        "path": "evidence_ref_id",
                    }
                ],
            }
        if not delivery_receipt_id:
            return {
                "status": "failed",
                "issues": [
                    {
                        "code": "CS_CONNECTOR_DELIVERY_RECEIPT_REQUIRED",
                        "message": "delivery_receipt_id is required to assemble a connector Evidence Bundle.",
                        "path": "delivery_receipt_id",
                    }
                ],
            }

        receipt = self._load_delivery_receipt(delivery_receipt_id)
        if receipt is None:
            return {"status": "not_found", "resource": "connector_delivery_receipt", "id": delivery_receipt_id}
        if receipt.get("scope") != requested_scope:
            return {"status": "scope_denied", "resource_scope": receipt.get("scope")}

        artifact_id = str(receipt.get("artifact_id", ""))
        artifact = self.store.get_artifact(artifact_id, requested_scope) if artifact_id else None
        artifact_connector_refs = (
            artifact.get("provenance", {}).get("connector_refs", {})
            if artifact and isinstance(artifact.get("provenance"), dict)
            else {}
        )
        setup_result_id = str(receipt.get("setup_result_id") or artifact_connector_refs.get("setup_result_id") or "")
        source_policy_id = str(receipt.get("source_policy_id") or artifact_connector_refs.get("source_policy_id") or "")
        projection_snapshot_id = str(
            receipt.get("projection_snapshot_id")
            or artifact_connector_refs.get("projection_snapshot_id")
            or ""
        )
        policy_decision_id = str(receipt.get("policy_decision_id") or artifact_connector_refs.get("policy_decision_id") or "")
        setup_result = self._load_setup_result(setup_result_id)
        source_policy = self._load_source_policy(source_policy_id)
        projection_snapshot = self._load_projection_snapshot(projection_snapshot_id)
        evidence_link_id = str(
            artifact_connector_refs.get("evidence_link_id", "")
        ) if artifact else _evidence_link_id(delivery_receipt_id, artifact_id)
        evidence_link = self._load_evidence_link(evidence_link_id)
        policy_decision = self._load_projection_policy_decision(policy_decision_id)
        content_version = self._load_content_version(str(receipt.get("content_version_id", "")))
        untrusted_review_id = str(
            receipt.get("untrusted_content_review", {}).get("review_id")
            or artifact_connector_refs.get("untrusted_content_review_id")
            or (
                artifact.get("connector_delivery", {}).get("untrusted_content_review_id")
                if artifact and isinstance(artifact.get("connector_delivery"), dict)
                else ""
            )
            or ""
        )
        untrusted_review = self._load_untrusted_content_review(untrusted_review_id) if untrusted_review_id else None
        if untrusted_review and untrusted_review.get("scope") != requested_scope:
            return {"status": "scope_denied", "resource_scope": untrusted_review.get("scope")}

        missing = []
        if artifact is None:
            missing.append({"resource": "artifact", "id": artifact_id})
        if setup_result is None:
            missing.append({"resource": "connector_setup_result", "id": setup_result_id})
        if source_policy is None:
            missing.append({"resource": "connector_source_policy", "id": source_policy_id})
        if projection_snapshot is None:
            missing.append({"resource": "connector_projection_snapshot", "id": projection_snapshot_id})
        if evidence_link is None:
            missing.append({"resource": "connector_evidence_link", "id": evidence_link_id})
        if policy_decision is None:
            missing.append({"resource": "connector_projection_policy_decision", "id": policy_decision_id})
        if untrusted_review_id and untrusted_review is None:
            missing.append({"resource": "connector_untrusted_content_review", "id": untrusted_review_id})
        if missing:
            audit_event = self.store.append_audit(
                "connector.evidence_bundle.denied",
                requested_scope,
                {"type": "connector_delivery_receipt", "id": delivery_receipt_id},
                {"reason": "missing_required_evidence_link", "missing": missing},
            )
            return {
                "status": "evidence_missing",
                "delivery_receipt": receipt,
                "missing": missing,
                "audit_event": audit_event,
                "issues": [
                    {
                        "code": "CS_CONNECTOR_EVIDENCE_CHAIN_INCOMPLETE",
                        "message": "Connector Evidence Bundle requires Artifact, Delivery, Setup Result, Source Policy, EvidenceRef, policy decision, and Projection snapshot links.",
                        "path": "delivery_receipt_id",
                    }
                ],
            }

        assert artifact is not None
        assert setup_result is not None
        assert source_policy is not None
        assert projection_snapshot is not None
        assert evidence_link is not None
        assert policy_decision is not None

        normalized_payload = projection_snapshot.get("normalized_projection", {}).get("payload", {})
        source_summary = receipt.get("source_summary", {})
        title = str(source_summary.get("title") or source_summary.get("source_ref") or receipt.get("source_external_id") or "")
        excerpt = ""
        if isinstance(normalized_payload, dict):
            excerpt = str(
                normalized_payload.get("body_markdown_excerpt")
                or normalized_payload.get("title")
                or normalized_payload.get("summary")
                or ""
            )
        snippet = " ".join(part for part in [title, excerpt] if part).replace("\n", " ").strip()[:240]
        if not snippet:
            snippet = f"Connector evidence for {receipt.get('source_external_id', delivery_receipt_id)}"

        query_text = query or f"connector evidence {receipt.get('source_external_id', delivery_receipt_id)}"
        connector_refs = {
            "artifact_id": artifact["artifact_id"],
            "delivery_receipt_id": receipt["delivery_receipt_id"],
            "projection_snapshot_id": projection_snapshot["projection_snapshot_id"],
            "evidence_link_id": evidence_link["evidence_link_id"],
            "setup_result_id": setup_result["setup_result_id"],
            "source_policy_id": source_policy["source_policy_id"],
            "policy_decision_id": policy_decision["policy_decision_id"],
            "evidence_ref_id": receipt.get("evidence_ref", {}).get("evidence_ref_id"),
            "content_version_id": receipt.get("content_version_id"),
        }
        if untrusted_review:
            connector_refs["untrusted_content_review_id"] = untrusted_review["review_id"]
        evidence_refs = [
            f"artifact:{artifact['artifact_id']}",
            f"connector_delivery_receipt:{receipt['delivery_receipt_id']}",
            f"connector_projection_snapshot:{projection_snapshot['projection_snapshot_id']}",
            f"connector_evidence_link:{evidence_link['evidence_link_id']}",
            f"connector_setup_result:{setup_result['setup_result_id']}",
            f"connector_source_policy:{source_policy['source_policy_id']}",
            f"connector_projection_policy_decision:{policy_decision['policy_decision_id']}",
            f"connector_evidence_ref:{receipt.get('evidence_ref', {}).get('evidence_ref_id')}",
        ]
        if content_version:
            evidence_refs.append(f"connector_content_version:{content_version['content_version_id']}")
        if untrusted_review:
            evidence_refs.append(f"connector_untrusted_content_review:{untrusted_review['review_id']}")
        search_snapshot_base = {
            "schema_version": "cs.search_snapshot.v0",
            "query": query_text,
            "filters": requested_scope,
            "result_count": 1,
            "results": [
                {
                    "artifact_id": artifact["artifact_id"],
                    "score": 100,
                    "snippet": snippet,
                    "source": "connector_projection_delivery",
                    "source_external_id": receipt.get("source_external_id"),
                    "source_revision": receipt.get("source_revision"),
                    "connector_refs": connector_refs,
                    "evidence_refs": evidence_refs,
                }
            ],
            "connector_snapshot": True,
            "created_at": utc_now(),
            "duration_ms": 0.0,
        }
        search_snapshot_id = f"search_{json_hash(search_snapshot_base)[:16]}"
        search_snapshot = dict(search_snapshot_base)
        search_snapshot["search_snapshot_id"] = search_snapshot_id
        _write_json(self.store.search_snapshot_path(search_snapshot_id), search_snapshot)
        search_event = self.store.append_audit(
            "search.snapshot.created",
            requested_scope,
            {"type": "search_snapshot", "id": search_snapshot_id},
            {
                "query": query_text,
                "result_count": 1,
                "source": "connector_projection_delivery",
                "delivery_receipt_id": receipt["delivery_receipt_id"],
            },
        )

        artifact_checksum_matches = artifact.get("checksum_sha256") == receipt.get("artifact_checksum_sha256")
        untrusted_negative = (
            untrusted_review.get("negative_evidence", {})
            if untrusted_review and isinstance(untrusted_review.get("negative_evidence"), dict)
            else {}
        )
        untrusted_content_handling = (
            untrusted_review.get("content_handling", {})
            if untrusted_review and isinstance(untrusted_review.get("content_handling"), dict)
            else {}
        )
        untrusted_zero_keys = [
            "tool_calls_created",
            "action_cards_created_from_untrusted_artifact",
            "workflow_runs_created_from_untrusted_artifact",
            "connector_actions_triggered_from_content",
            "provider_calls_triggered_from_content",
            "shell_calls_triggered_from_content",
            "external_http_calls",
            "memory_promotions_from_untrusted_artifact",
            "policy_overrides_from_untrusted_artifact",
            "authority_expansions_from_untrusted_artifact",
        ]
        untrusted_trust_boundary = {
            "source_trust_label": untrusted_review.get("source_trust_label") if untrusted_review else None,
            "artifact_trust_state": artifact.get("trust_state"),
            "untrusted_evidence_label_present": bool(
                untrusted_review
                and artifact.get("trust_state") == "untrusted"
                and untrusted_review.get("source_trust_label") == "untrusted_connector_content"
            ),
            "unsafe_instruction_detected": bool(untrusted_review and untrusted_review.get("unsafe_instruction_detected")),
            "treated_as_system_instruction": untrusted_content_handling.get("treated_as_system_instruction"),
            "quoted_or_summarized_as_evidence_only": untrusted_content_handling.get("quoted_or_summarized_as_evidence_only"),
            "tool_action_egress_counters_zero": bool(
                untrusted_review and all(untrusted_negative.get(key) == 0 for key in untrusted_zero_keys)
            ),
            "memory_promotion_blocked": bool(
                untrusted_review and untrusted_negative.get("memory_promotions_from_untrusted_artifact") == 0
            ),
            "policy_override_blocked": bool(
                untrusted_review and untrusted_negative.get("policy_overrides_from_untrusted_artifact") == 0
            ),
            "authority_expansion_blocked": bool(
                untrusted_review and untrusted_negative.get("authority_expansions_from_untrusted_artifact") == 0
            ),
        }
        coverage = {
            "artifact_ref_present": bool(artifact.get("artifact_id")),
            "artifact_checksum_matches": artifact_checksum_matches,
            "delivery_receipt_ref_present": True,
            "setup_result_ref_present": True,
            "source_policy_ref_present": True,
            "evidence_ref_metadata_present": bool(receipt.get("evidence_ref", {}).get("evidence_ref_id")),
            "projection_snapshot_ref_present": True,
            "policy_decision_ref_present": True,
            "search_snapshot_ref_present": True,
            "audit_refs_present": bool(receipt.get("audit_refs")),
            "evidence_ref_alone_is_original": False,
            "inaccessible_phantom_evidence_count": 0,
            "raw_provider_payload_included": False,
            "raw_access_handle_included": False,
            "untrusted_evidence_label_present": untrusted_trust_boundary["untrusted_evidence_label_present"],
            "unsafe_instruction_treated_as_evidence_only": bool(
                untrusted_review
                and untrusted_content_handling.get("treated_as_system_instruction") is False
                and untrusted_content_handling.get("quoted_or_summarized_as_evidence_only") is True
            ),
            "tool_action_egress_counters_zero": untrusted_trust_boundary["tool_action_egress_counters_zero"],
            "memory_promotion_blocked": untrusted_trust_boundary["memory_promotion_blocked"],
            "policy_override_blocked": untrusted_trust_boundary["policy_override_blocked"],
            "authority_expansion_blocked": untrusted_trust_boundary["authority_expansion_blocked"],
        }
        connector_bundle_link = {
            "schema_version": CONNECTOR_EVIDENCE_BUNDLE_LINK_SCHEMA,
            "delivery_receipt_id": receipt["delivery_receipt_id"],
            "artifact_id": artifact["artifact_id"],
            "artifact_checksum_sha256": artifact.get("checksum_sha256"),
            "source_external_id": receipt.get("source_external_id"),
            "source_revision": receipt.get("source_revision"),
            "provider_event_id": receipt.get("provider_event_id"),
            "projection_type": receipt.get("projection_type"),
            "connector_refs": connector_refs,
            "coverage": coverage,
            "source_policy_restriction_summary": policy_decision.get("restriction_summary"),
            "untrusted_content_review": {
                "review_id": untrusted_review["review_id"],
                "status": untrusted_review.get("status"),
                "unsafe_instruction_detected": untrusted_review.get("unsafe_instruction_detected"),
                "blocked_attempt_count": untrusted_review.get("blocked_attempt_count"),
            } if untrusted_review else None,
            "trust_boundary": untrusted_trust_boundary,
        }
        evidence_item = {
            "artifact_id": artifact["artifact_id"],
            "search_snapshot_id": search_snapshot_id,
            "snippet": snippet,
            "original_storage_ref": artifact.get("original_storage_ref"),
            "derived_text_ref": artifact.get("derived", {}).get("text_ref"),
            "source": artifact.get("source"),
            "provenance": artifact.get("provenance"),
            "artifact_checksum_sha256": artifact.get("checksum_sha256"),
            "connector": connector_bundle_link,
            "connector_delivery_receipt": {
                "delivery_receipt_id": receipt["delivery_receipt_id"],
                "delivery_id": receipt.get("delivery_id"),
                "projection_id": receipt.get("projection_id"),
                "artifact_id": receipt.get("artifact_id"),
                "artifact_checksum_sha256": receipt.get("artifact_checksum_sha256"),
                "source_external_id": receipt.get("source_external_id"),
                "source_revision": receipt.get("source_revision"),
                "acknowledgement_state": receipt.get("acknowledgement_state"),
            },
            "connector_evidence_ref": receipt.get("evidence_ref", {}),
            "connector_source_policy": {
                "source_policy_id": source_policy["source_policy_id"],
                "content_mode": source_policy.get("content_mode"),
                "raw_access": source_policy.get("raw_access"),
                "selected_resources": source_policy.get("selected_resources", []),
                "allowed_paths": source_policy.get("allowed_paths", []),
                "max_content_bytes": source_policy.get("max_content_bytes"),
            },
            "connector_setup_result": {
                "setup_result_id": setup_result["setup_result_id"],
                "readiness": setup_result.get("readiness"),
                "activation_allowed": setup_result.get("activation_allowed"),
            },
            "connector_policy_decision": {
                "policy_decision_id": policy_decision["policy_decision_id"],
                "decision": policy_decision.get("decision"),
                "enforcement_action": policy_decision.get("enforcement_action"),
                "included_fields": policy_decision.get("included_fields", []),
                "excluded_fields": policy_decision.get("excluded_fields", []),
                "raw_content_persisted": False,
            },
            "connector_untrusted_content_review": untrusted_review,
            "trust_boundary": untrusted_trust_boundary,
        }
        bundle_base = {
            "schema_version": "cs.evidence_bundle.v0",
            "origin": "connector_projection_delivery",
            "search_snapshot_id": search_snapshot_id,
            "query": query_text,
            "filters": requested_scope,
            "result_snapshot": {
                "search_snapshot_id": search_snapshot_id,
                "query": query_text,
                "filters": requested_scope,
                "result_count": 1,
                "results": search_snapshot["results"],
            },
            "evidence_items": [evidence_item],
            "connector_evidence_bundle": connector_bundle_link,
            "coverage": coverage,
            "raw_provider_payload_available": False,
            "raw_access_default_denied": source_policy.get("raw_access") == "denied",
            "created_at": utc_now(),
        }
        bundle_id = f"evb_{json_hash(bundle_base)[:16]}"
        bundle = dict(bundle_base)
        bundle["evidence_bundle_id"] = bundle_id
        bundle_event = self.store.append_audit(
            "evidence_bundle.created",
            requested_scope,
            {"type": "evidence_bundle", "id": bundle_id},
            {
                "search_snapshot_id": search_snapshot_id,
                "evidence_item_count": 1,
                "origin": "connector_projection_delivery",
                "delivery_receipt_id": receipt["delivery_receipt_id"],
            },
        )
        connector_event = self.store.append_audit(
            "connector.evidence_bundle.assembled",
            requested_scope,
            {"type": "evidence_bundle", "id": bundle_id},
            {
                "delivery_receipt_id": receipt["delivery_receipt_id"],
                "artifact_id": artifact["artifact_id"],
                "evidence_ref_id": receipt.get("evidence_ref", {}).get("evidence_ref_id"),
                "coverage": coverage,
            },
        )
        audit_refs = list(dict.fromkeys([
            *receipt.get("audit_refs", []),
            f"audit:{search_event['event_id']}",
            f"audit:{bundle_event['event_id']}",
            f"audit:{connector_event['event_id']}",
        ]))
        bundle["audit_refs"] = audit_refs
        bundle["connector_evidence_bundle"]["audit_refs"] = audit_refs
        _write_json(self.store.evidence_bundle_path(bundle_id), bundle)
        return {
            "status": "success",
            "evidence_bundle": bundle,
            "search_snapshot": search_snapshot,
            "connector_evidence_bundle_link": connector_bundle_link,
            "delivery_receipt": receipt,
            "artifact": artifact,
            "setup_result": setup_result,
            "source_policy": source_policy,
            "projection_snapshot": projection_snapshot,
            "evidence_link": evidence_link,
            "policy_decision": policy_decision,
            "connector_untrusted_content_review": untrusted_review,
            "audit_event": connector_event,
            "audit_events": [search_event, bundle_event, connector_event],
        }

    def request_raw_access(
        self,
        requested_scope: dict[str, str],
        contract_id: str,
        evidence_ref_id: str,
        purpose: str,
        classification: str,
        ttl_seconds: int,
        max_reads: int,
        human_approved: bool,
        source_external_id: str | None = None,
        contract_version_id: str | None = None,
    ) -> dict[str, Any]:
        contract = self.load_contract(contract_id, contract_version_id=contract_version_id)
        if contract is None:
            return {"status": "not_found", "resource": "connector_contract", "id": contract_id}
        if contract.get("scope") != requested_scope:
            return {"status": "scope_denied", "resource_scope": contract.get("scope")}

        provider_pack_id = str(contract.get("source_policy_request", {}).get("provider_pack_id") or "local_source_control_readonly.v1")
        setup_result = self._latest_setup_result(contract, requested_scope, provider_pack_id)
        source_policy = setup_result.get("source_policy_snapshot", {}) if setup_result else {}
        raw_policy = source_policy.get("raw_access_policy") if isinstance(source_policy.get("raw_access_policy"), dict) else {}
        request_id = _raw_access_request_id(
            requested_scope,
            contract["contract_version_id"],
            evidence_ref_id,
            purpose,
            classification,
            ttl_seconds,
            max_reads,
        )
        base_request = {
            "schema_version": RAW_ACCESS_REQUEST_SCHEMA,
            "raw_access_request_id": request_id,
            "scope": requested_scope,
            "contract_id": contract["contract_id"],
            "contract_version_id": contract["contract_version_id"],
            "setup_result_id": setup_result.get("setup_result_id") if setup_result else None,
            "source_policy_id": source_policy.get("source_policy_id"),
            "evidence_ref_id": evidence_ref_id,
            "source_external_id": source_external_id,
            "purpose": purpose,
            "classification": classification,
            "requested_ttl_seconds": ttl_seconds,
            "requested_max_reads": max_reads,
            "human_approved": human_approved,
            "raw_content_copied_to_product_records": False,
            "raw_provider_payload_persisted": False,
            "raw_access_handle_in_logs": False,
            "created_at": utc_now(),
        }
        issues: list[dict[str, str]] = []
        if setup_result is None:
            issues.append(
                {
                    "code": "CS_CONNECTOR_SETUP_NOT_FOUND",
                    "message": "Raw access requires a scoped Setup Result and Source Policy.",
                    "path": "setup_result",
                }
            )
        if source_policy.get("raw_access") != RAW_ACCESS_ALLOWED_MODE:
            issues.append(
                {
                    "code": "CS_CONNECTOR_RAW_ACCESS_DENIED_BY_DEFAULT",
                    "message": "Raw access is denied unless the connector contract and Source Policy explicitly allow temporary scoped access.",
                    "path": "source_policy.raw_access",
                }
            )
        if not human_approved:
            issues.append(
                {
                    "code": "CS_CONNECTOR_RAW_ACCESS_APPROVAL_REQUIRED",
                    "message": "Temporary raw access requires explicit authorized human approval.",
                    "path": "human_approved",
                }
            )
        if classification not in RAW_ACCESS_CLASSIFICATIONS:
            issues.append(
                {
                    "code": "CS_CONNECTOR_RAW_ACCESS_CLASSIFICATION_UNSUPPORTED",
                    "message": "classification must be internal, confidential, or restricted.",
                    "path": "classification",
                }
            )
        allowed_purposes = [str(item) for item in raw_policy.get("allowed_purposes", []) if isinstance(item, str)]
        if allowed_purposes and purpose not in allowed_purposes:
            issues.append(
                {
                    "code": "CS_CONNECTOR_RAW_ACCESS_PURPOSE_DENIED",
                    "message": "Raw access purpose is not allowed by the active Source Policy.",
                    "path": "purpose",
                }
            )
        max_ttl = int(raw_policy.get("max_ttl_seconds") or 0)
        if ttl_seconds <= 0 or (max_ttl and ttl_seconds > max_ttl):
            issues.append(
                {
                    "code": "CS_CONNECTOR_RAW_ACCESS_TTL_DENIED",
                    "message": "Requested raw access TTL exceeds the active Source Policy limit.",
                    "path": "ttl_seconds",
                }
            )
        policy_max_reads = int(raw_policy.get("max_reads") or 0)
        if max_reads <= 0 or (policy_max_reads and max_reads > policy_max_reads):
            issues.append(
                {
                    "code": "CS_CONNECTOR_RAW_ACCESS_READ_LIMIT_DENIED",
                    "message": "Requested raw access read count exceeds the active Source Policy limit.",
                    "path": "max_reads",
                }
            )
        if not evidence_ref_id:
            issues.append(
                {
                    "code": "CS_CONNECTOR_RAW_ACCESS_EVIDENCE_REF_REQUIRED",
                    "message": "Temporary raw access must be bound to one EvidenceRef.",
                    "path": "evidence_ref_id",
                }
            )

        if issues:
            request_record = {
                **base_request,
                "decision": "deny",
                "status": "denied",
                "issue_codes": [issue["code"] for issue in issues],
                "reusable_raw_handle": False,
            }
            audit_event = self.store.append_audit(
                "connector.raw_access.denied",
                requested_scope,
                {"type": "connector_raw_access_request", "id": request_id},
                {
                    "contract_version_id": contract["contract_version_id"],
                    "evidence_ref_id": evidence_ref_id,
                    "issue_codes": request_record["issue_codes"],
                    "raw_access_handle_in_logs": False,
                    "raw_content_copied_to_product_records": False,
                },
            )
            request_record["audit_refs"] = [f"audit:{audit_event['event_id']}"]
            _write_json(self._raw_access_request_path(request_id), request_record)
            return {
                "status": "denied",
                "raw_access_request": request_record,
                "issues": issues,
                "audit_event": audit_event,
            }

        grant_id = _raw_access_grant_id(request_id)
        expires_at = utc_after(ttl_seconds)
        reference_fingerprint = json_hash(
            {
                "grant_id": grant_id,
                "request_id": request_id,
                "scope": requested_scope,
                "evidence_ref_id": evidence_ref_id,
                "purpose": purpose,
            }
        )
        grant = {
            "schema_version": RAW_ACCESS_GRANT_SCHEMA,
            "raw_access_grant_id": grant_id,
            "raw_access_request_id": request_id,
            "scope": requested_scope,
            "contract_id": contract["contract_id"],
            "contract_version_id": contract["contract_version_id"],
            "setup_result_id": setup_result["setup_result_id"] if setup_result else None,
            "source_policy_id": source_policy.get("source_policy_id"),
            "evidence_ref_id": evidence_ref_id,
            "source_external_id": source_external_id,
            "purpose": purpose,
            "classification": classification,
            "status": "active",
            "expires_at": expires_at,
            "ttl_seconds": ttl_seconds,
            "max_reads": max_reads,
            "remaining_reads": max_reads,
            "read_count": 0,
            "revoked": False,
            "revoked_at": None,
            "opaque_reference_fingerprint": reference_fingerprint,
            "opaque_reference_exposed": False,
            "reusable_raw_handle": False,
            "raw_access_handle_in_logs": False,
            "raw_content_copied_to_product_records": False,
            "raw_provider_payload_persisted": False,
            "redaction": {
                "raw_content_returned_to_product": False,
                "provider_payload_returned_to_product": False,
                "opaque_reference_redacted_in_output": True,
                "safe_to_show_to_owner": True,
            },
            "access_events": [],
            "created_at": utc_now(),
        }
        request_record = {
            **base_request,
            "decision": "grant",
            "status": "granted",
            "raw_access_grant_id": grant_id,
            "expires_at": expires_at,
            "policy_limits": {
                "max_ttl_seconds": max_ttl,
                "max_reads": policy_max_reads,
                "allowed_purposes": allowed_purposes,
            },
            "reusable_raw_handle": False,
        }
        audit_event = self.store.append_audit(
            "connector.raw_access.granted",
            requested_scope,
            {"type": "connector_raw_access_grant", "id": grant_id},
            {
                "raw_access_request_id": request_id,
                "evidence_ref_id": evidence_ref_id,
                "expires_at": expires_at,
                "max_reads": max_reads,
                "opaque_reference_fingerprint": reference_fingerprint,
                "raw_access_handle_in_logs": False,
                "raw_content_copied_to_product_records": False,
            },
        )
        audit_refs = [f"audit:{audit_event['event_id']}"]
        request_record["audit_refs"] = audit_refs
        grant["audit_refs"] = audit_refs
        _write_json(self._raw_access_request_path(request_id), request_record)
        _write_json(self._raw_access_grant_path(grant_id), grant)
        return {
            "status": "success",
            "raw_access_request": request_record,
            "raw_access_grant": grant,
            "setup_result": setup_result,
            "source_policy": source_policy,
            "audit_event": audit_event,
        }

    def read_raw_access(
        self,
        raw_access_grant_id: str,
        requested_scope: dict[str, str],
        at: str | None = None,
    ) -> dict[str, Any]:
        grant = self._load_raw_access_grant(raw_access_grant_id)
        if grant is None:
            return {"status": "not_found", "resource": "connector_raw_access_grant", "id": raw_access_grant_id}
        if grant.get("scope") != requested_scope:
            return {"status": "scope_denied", "resource_scope": grant.get("scope")}
        checked_at = at or utc_now()
        denial_code: str | None = None
        if grant.get("revoked") is True or grant.get("status") == "revoked":
            denial_code = "CS_CONNECTOR_RAW_ACCESS_REVOKED"
            grant["status"] = "revoked"
        elif checked_at >= str(grant.get("expires_at", "")):
            denial_code = "CS_CONNECTOR_RAW_ACCESS_EXPIRED"
            grant["status"] = "expired"
        elif int(grant.get("remaining_reads", 0) or 0) <= 0:
            denial_code = "CS_CONNECTOR_RAW_ACCESS_READ_LIMIT_EXHAUSTED"
            grant["status"] = "exhausted"

        if denial_code:
            audit_event = self.store.append_audit(
                "connector.raw_access.read_denied",
                requested_scope,
                {"type": "connector_raw_access_grant", "id": raw_access_grant_id},
                {
                    "reason_code": denial_code,
                    "checked_at": checked_at,
                    "raw_access_handle_in_logs": False,
                    "raw_content_returned": False,
                },
            )
            grant.setdefault("access_events", []).append(
                {
                    "event": "read_denied",
                    "reason_code": denial_code,
                    "at": checked_at,
                    "raw_content_returned": False,
                }
            )
            grant["audit_refs"] = list(dict.fromkeys([*grant.get("audit_refs", []), f"audit:{audit_event['event_id']}"]))
            _write_json(self._raw_access_grant_path(raw_access_grant_id), grant)
            return {
                "status": "denied",
                "raw_access_grant": grant,
                "issues": [
                    {
                        "code": denial_code,
                        "message": "Temporary raw access is unavailable for this grant.",
                        "path": "raw_access_grant_id",
                    }
                ],
                "audit_event": audit_event,
            }

        grant["read_count"] = int(grant.get("read_count", 0) or 0) + 1
        grant["remaining_reads"] = max(int(grant.get("remaining_reads", 0) or 0) - 1, 0)
        if grant["remaining_reads"] == 0:
            grant["status"] = "exhausted"
        audit_event = self.store.append_audit(
            "connector.raw_access.read",
            requested_scope,
            {"type": "connector_raw_access_grant", "id": raw_access_grant_id},
            {
                "read_count": grant["read_count"],
                "remaining_reads": grant["remaining_reads"],
                "raw_access_handle_in_logs": False,
                "raw_content_returned": False,
            },
        )
        grant.setdefault("access_events", []).append(
            {
                "event": "read",
                "at": checked_at,
                "read_count": grant["read_count"],
                "remaining_reads": grant["remaining_reads"],
                "raw_content_returned": False,
            }
        )
        grant["audit_refs"] = list(dict.fromkeys([*grant.get("audit_refs", []), f"audit:{audit_event['event_id']}"]))
        _write_json(self._raw_access_grant_path(raw_access_grant_id), grant)
        return {
            "status": "success",
            "raw_access_grant": grant,
            "raw_access_result": {
                "schema_version": "cs.connector_raw_access_read_result.v1",
                "raw_access_grant_id": raw_access_grant_id,
                "status": "redacted_success",
                "evidence_ref_id": grant.get("evidence_ref_id"),
                "classification": grant.get("classification"),
                "read_count": grant["read_count"],
                "remaining_reads": grant["remaining_reads"],
                "raw_content_returned": False,
                "provider_payload_returned": False,
                "redacted_summary": "Raw provider content is mediated inside ConnectorHub and omitted from Product output in the local fixture.",
            },
            "audit_event": audit_event,
        }

    def revoke_raw_access(
        self,
        raw_access_grant_id: str,
        requested_scope: dict[str, str],
        reason: str,
    ) -> dict[str, Any]:
        grant = self._load_raw_access_grant(raw_access_grant_id)
        if grant is None:
            return {"status": "not_found", "resource": "connector_raw_access_grant", "id": raw_access_grant_id}
        if grant.get("scope") != requested_scope:
            return {"status": "scope_denied", "resource_scope": grant.get("scope")}
        grant["status"] = "revoked"
        grant["revoked"] = True
        grant["revoked_at"] = utc_now()
        grant["revocation_reason"] = reason
        grant["remaining_reads"] = 0
        audit_event = self.store.append_audit(
            "connector.raw_access.revoked",
            requested_scope,
            {"type": "connector_raw_access_grant", "id": raw_access_grant_id},
            {
                "reason": reason,
                "raw_access_handle_in_logs": False,
                "raw_content_returned": False,
            },
        )
        grant["audit_refs"] = list(dict.fromkeys([*grant.get("audit_refs", []), f"audit:{audit_event['event_id']}"]))
        _write_json(self._raw_access_grant_path(raw_access_grant_id), grant)
        return {"status": "success", "raw_access_grant": grant, "audit_event": audit_event}

    def export_raw_access_metadata(
        self,
        raw_access_grant_id: str,
        requested_scope: dict[str, str],
    ) -> dict[str, Any]:
        grant = self._load_raw_access_grant(raw_access_grant_id)
        if grant is None:
            return {"status": "not_found", "resource": "connector_raw_access_grant", "id": raw_access_grant_id}
        if grant.get("scope") != requested_scope:
            return {"status": "scope_denied", "resource_scope": grant.get("scope")}
        export = {
            "schema_version": RAW_ACCESS_EXPORT_SCHEMA,
            "raw_access_grant_id": grant["raw_access_grant_id"],
            "raw_access_request_id": grant["raw_access_request_id"],
            "scope": requested_scope,
            "contract_id": grant.get("contract_id"),
            "contract_version_id": grant.get("contract_version_id"),
            "source_policy_id": grant.get("source_policy_id"),
            "evidence_ref_id": grant.get("evidence_ref_id"),
            "source_external_id": grant.get("source_external_id"),
            "purpose": grant.get("purpose"),
            "classification": grant.get("classification"),
            "status": grant.get("status"),
            "expires_at": grant.get("expires_at"),
            "max_reads": grant.get("max_reads"),
            "read_count": grant.get("read_count"),
            "remaining_reads": grant.get("remaining_reads"),
            "revoked": grant.get("revoked"),
            "opaque_reference_fingerprint": grant.get("opaque_reference_fingerprint"),
            "opaque_reference_exposed": False,
            "raw_content_included": False,
            "raw_provider_payload_included": False,
            "raw_access_handle_included": False,
            "audit_refs": grant.get("audit_refs", []),
        }
        audit_event = self.store.append_audit(
            "connector.raw_access.metadata_exported",
            requested_scope,
            {"type": "connector_raw_access_grant", "id": raw_access_grant_id},
            {
                "raw_content_included": False,
                "raw_provider_payload_included": False,
                "raw_access_handle_included": False,
            },
        )
        export["audit_refs"] = list(dict.fromkeys([*export["audit_refs"], f"audit:{audit_event['event_id']}"]))
        return {"status": "success", "raw_access_export": export, "audit_event": audit_event}

    def plan_setup(
        self,
        contract_id: str,
        requested_scope: dict[str, str],
        contract_version_id: str | None = None,
        provider_pack_id: str | None = None,
    ) -> dict[str, Any]:
        contract = self.load_contract(contract_id, contract_version_id=contract_version_id)
        if contract is None:
            return {"status": "not_found", "contract_id": contract_id}
        if contract.get("scope") != requested_scope:
            return {"status": "scope_denied", "resource_scope": contract.get("scope")}

        mappings: list[dict[str, Any]] = []
        gaps: list[dict[str, Any]] = []
        selected_provider_pack_id = (
            provider_pack_id
            or contract["source_policy_request"].get("provider_pack_id")
            or "local_source_control_readonly.v1"
        )
        provider_pack = FIXTURE_PROVIDER_PACKS.get(
            selected_provider_pack_id,
            {
                "provider_pack_id": selected_provider_pack_id,
                "display_name": selected_provider_pack_id,
                "transport": "unavailable",
                "capabilities": {},
                "declared_actions": [],
                "provider_calls_before_activation": 0,
            },
        )
        supported = provider_pack["capabilities"]
        setup_gap = provider_pack.get("setup_gap")
        planned_capabilities: list[dict[str, Any]] = []
        for need in contract["needs"]:
            capability = need["common_capability"]
            accepted = set(need.get("accepted_projection_types", []))
            if setup_gap:
                planned_capabilities.append(
                    {
                        "common_capability": capability,
                        "required": bool(need["required"]),
                        "support_status": "unavailable",
                        "projection_types": [],
                        "provider_pack_id": provider_pack["provider_pack_id"],
                    }
                )
                gaps.append(
                    {
                        "common_capability": capability,
                        "required": bool(need["required"]),
                        "reason_code": setup_gap["reason_code"],
                        "resolution": "Grant the missing read-only permission or choose an already authorized compatible Provider Pack.",
                    }
                )
                continue
            supported_projection_types = [projection for projection in supported.get(capability, []) if projection in accepted]
            if supported_projection_types:
                planned_capabilities.append(
                    {
                        "common_capability": capability,
                        "required": bool(need["required"]),
                        "support_status": "available",
                        "projection_types": supported_projection_types,
                        "provider_pack_id": provider_pack["provider_pack_id"],
                    }
                )
                mappings.append(
                    {
                        "common_capability": capability,
                        "required": bool(need["required"]),
                        "provider_pack_id": provider_pack["provider_pack_id"],
                        "projection_types": supported_projection_types,
                        "stream_id": _stream_id(contract["contract_version_id"], capability),
                    }
                )
                continue
            planned_capabilities.append(
                {
                    "common_capability": capability,
                    "required": bool(need["required"]),
                    "support_status": "unavailable",
                    "projection_types": [],
                    "provider_pack_id": None,
                }
            )
            gaps.append(
                {
                    "common_capability": capability,
                    "required": bool(need["required"]),
                    "reason_code": "CS_CONNECTOR_CAPABILITY_UNAVAILABLE",
                    "resolution": "Choose a compatible Provider Pack, grant an approved permission, or change the contract.",
                }
            )

        required_gaps = [gap for gap in gaps if gap["required"]]
        activation_allowed = not required_gaps
        policy_request = contract["source_policy_request"]
        source_policy_identity = {**policy_request, "selected_provider_pack_id": provider_pack["provider_pack_id"]}
        source_policy_id = _source_policy_id(contract["contract_version_id"], source_policy_identity)
        selected_resource_scope = _selected_resource_scope(policy_request)
        source_policy = {
            "schema_version": SOURCE_POLICY_SCHEMA,
            "source_policy_id": source_policy_id,
            "contract_version_id": contract["contract_version_id"],
            "scope": requested_scope,
            "selected_resources": policy_request.get("selected_resources", []),
            "selected_resource_scope": selected_resource_scope,
            "content_mode": policy_request.get("content_mode", "metadata_only"),
            "max_content_bytes": policy_request.get("max_content_bytes"),
            "allowed_paths": policy_request.get("allowed_paths", []),
            "raw_access": policy_request.get("raw_access", "denied"),
            "raw_access_policy": policy_request.get("raw_access_policy", {})
            if isinstance(policy_request.get("raw_access_policy"), dict)
            else {},
            "retention_days": policy_request.get("retention_days"),
            "constraints_never_broadened_silently": True,
            "provider_pack_ids": sorted({mapping["provider_pack_id"] for mapping in mappings}),
            "selected_provider_pack_id": provider_pack["provider_pack_id"],
        }
        readiness = "blocked" if required_gaps else ("ready_with_gaps" if gaps else "ready")
        delivery_streams = [mapping["stream_id"] for mapping in mappings] if activation_allowed else []
        feature_availability: list[dict[str, Any]] = []
        disabled_surfaces: list[dict[str, Any]] = []
        for planned in planned_capabilities:
            surface = _capability_surface(planned["common_capability"])
            if planned["support_status"] == "available" and activation_allowed:
                availability = {
                    "common_capability": planned["common_capability"],
                    "surface": surface,
                    "required": planned["required"],
                    "enabled": True,
                    "state": "enabled",
                    "reason_code": None,
                    "projection_types": planned["projection_types"],
                    "provider_pack_id": planned["provider_pack_id"],
                }
            elif planned["support_status"] == "available":
                availability = {
                    "common_capability": planned["common_capability"],
                    "surface": surface,
                    "required": planned["required"],
                    "enabled": False,
                    "state": "disabled_blocked_activation",
                    "reason_code": "CS_CONNECTOR_REQUIRED_CAPABILITY_MISSING",
                    "projection_types": planned["projection_types"],
                    "provider_pack_id": planned["provider_pack_id"],
                }
            else:
                availability = {
                    "common_capability": planned["common_capability"],
                    "surface": surface,
                    "required": planned["required"],
                    "enabled": False,
                    "state": "disabled_required_missing" if planned["required"] else "disabled_optional_unavailable",
                    "reason_code": "CS_CONNECTOR_CAPABILITY_UNAVAILABLE"
                    if planned["required"]
                    else "CS_CONNECTOR_OPTIONAL_CAPABILITY_UNAVAILABLE",
                    "projection_types": [],
                    "provider_pack_id": None,
                }
            feature_availability.append(availability)
            if not availability["enabled"]:
                disabled_surfaces.append(
                    {
                        "surface": surface,
                        "common_capability": availability["common_capability"],
                        "reason_code": availability["reason_code"],
                        "required": availability["required"],
                        "guidance": "Unavailable optional surfaces stay disabled while available capabilities remain usable."
                        if not availability["required"] and activation_allowed
                        else _activation_guidance(required_gaps),
                    }
                )
        setup_result_id = _setup_result_id(contract["contract_version_id"], source_policy, mappings)
        product_projection_contract = [
            {
                "common_capability": mapping["common_capability"],
                "projection_types": mapping["projection_types"],
                "required": mapping["required"],
            }
            for mapping in mappings
        ]
        product_object_preview = {
            "schema_version": "cs.connected_source.preview.v1",
            "handler_family": PRODUCT_HANDLER_CONTRACT["handler_family"],
            "capabilities": product_projection_contract,
            "requires_provider_sdk": False,
        }
        setup_result = {
            "schema_version": SETUP_RESULT_SCHEMA,
            "setup_result_id": setup_result_id,
            "contract_id": contract["contract_id"],
            "contract_version_id": contract["contract_version_id"],
            "scope": requested_scope,
            "readiness": readiness,
            "activation_state": (
                "blocked_permission_required"
                if required_gaps and setup_gap
                else ("blocked_required_capability_missing" if required_gaps else "planned_ready")
            ),
            "activation_allowed": activation_allowed,
            "activation_blockers": required_gaps,
            "activation_guidance": _activation_guidance(required_gaps),
            "blocked_reason_code": setup_gap["reason_code"] if required_gaps and setup_gap else (
                "CS_CONNECTOR_REQUIRED_CAPABILITY_MISSING" if required_gaps else None
            ),
            "required_capabilities_available": activation_allowed,
            "mappings": mappings,
            "delivery_streams": delivery_streams,
            "selected_resource_scope": selected_resource_scope,
            "feature_availability": feature_availability,
            "disabled_surfaces": disabled_surfaces,
            "status_explanation": _safe_status_explanation(setup_gap, required_gaps),
            "product_handler_contract": {
                **PRODUCT_HANDLER_CONTRACT,
                "handler_contract_hash": _product_handler_contract_hash(),
            },
            "product_projection_contract": product_projection_contract,
            "product_object_preview": product_object_preview,
            "gaps": gaps,
            "warnings": (
                ["Required connector capabilities are unavailable; activation is blocked."]
                if required_gaps
                else (["Optional connector capabilities are unavailable."] if gaps else [])
            ),
            "source_policy_snapshot": source_policy,
            "verification_refs": [
                _fixture_ref(contract),
                f"provider_pack:{provider_pack['provider_pack_id']}",
            ],
            "provider_call_ledger": {
                "before_activation": provider_pack["provider_calls_before_activation"],
                "during_plan": 0,
            },
            "app_facing_boundaries": {
                "provider_tokens_exposed": False,
                "provider_clients_exposed": False,
                "raw_local_paths_exposed": False,
                "direct_api_handles_exposed": False,
                "product_depends_on_provider_sdk": False,
            },
            "created_at": utc_now(),
        }
        _write_json(self._source_policy_path(source_policy_id), source_policy)
        _write_json(self._setup_result_path(setup_result_id), setup_result)
        audit_event = self.store.append_audit(
            "connector.setup.planned",
            requested_scope,
            {"type": "connector_setup_result", "id": setup_result_id},
            {
                "contract_version_id": contract["contract_version_id"],
                "readiness": readiness,
                "required_gap_count": len(required_gaps),
                "provider_calls_before_activation": 0,
            },
        )
        return {
            "status": "success" if not required_gaps else "blocked",
            "contract": contract,
            "setup_result": setup_result,
            "source_policy": source_policy,
            "audit_event": audit_event,
        }

    def confirm_source_policy(
        self,
        contract_id: str,
        requested_scope: dict[str, str],
        overrides: dict[str, Any],
        contract_version_id: str | None = None,
    ) -> dict[str, Any]:
        contract = self.load_contract(contract_id, contract_version_id=contract_version_id)
        if contract is None:
            return {"status": "not_found", "contract_id": contract_id}
        if contract.get("scope") != requested_scope:
            return {"status": "scope_denied", "resource_scope": contract.get("scope")}

        base_policy = _normalized_source_policy_request(contract["source_policy_request"], {})
        candidate_policy_request = _normalized_source_policy_request(base_policy, overrides)
        issues = _source_policy_broadening_issues(base_policy, candidate_policy_request)
        if issues:
            audit_event = self.store.append_audit(
                "connector.source_policy.broadening_denied",
                requested_scope,
                {"type": "connector_contract", "id": contract["contract_version_id"]},
                {
                    "contract_version_id": contract["contract_version_id"],
                    "issue_fields": [issue.get("field") for issue in issues],
                    "broadened": True,
                    "source_policy_created": False,
                },
            )
            return {
                "status": "denied",
                "contract": contract,
                "base_policy_request": base_policy,
                "candidate_policy_request": candidate_policy_request,
                "issues": issues,
                "audit_event": audit_event,
            }

        source_policy_id = _source_policy_id(contract["contract_version_id"], candidate_policy_request)
        diff = _source_policy_diff(base_policy, candidate_policy_request)
        confirmation_kind = "owner_override" if diff["changed_fields"] else "owner_confirmed"
        source_policy = {
            "schema_version": SOURCE_POLICY_SCHEMA,
            "source_policy_id": source_policy_id,
            "contract_version_id": contract["contract_version_id"],
            "scope": requested_scope,
            "selected_resources": candidate_policy_request.get("selected_resources", []),
            "content_mode": candidate_policy_request.get("content_mode", "metadata_only"),
            "max_content_bytes": candidate_policy_request.get("max_content_bytes"),
            "allowed_paths": candidate_policy_request.get("allowed_paths", []),
            "raw_access": candidate_policy_request.get("raw_access", "denied"),
            "raw_access_policy": candidate_policy_request.get("raw_access_policy", {})
            if isinstance(candidate_policy_request.get("raw_access_policy"), dict)
            else {},
            "retention_days": candidate_policy_request.get("retention_days"),
            "constraints_never_broadened_silently": True,
            "provider_pack_ids": ["local_source_control_readonly.v1"],
            "confirmation": {
                "schema_version": "cs.connector_source_policy_confirmation.v1",
                "kind": confirmation_kind,
                "owner_confirmed": True,
                "confirmed_by": requested_scope["owner_id"],
                "confirmed_at": utc_now(),
                "base_policy_hash": diff["base_hash"],
                "confirmed_policy_hash": diff["candidate_hash"],
                "silent_broadening": False,
            },
            "compatibility_decision": {
                "status": "compatible",
                "broadened": False,
                "narrowed_fields": diff["narrowed_fields"],
            },
            "source_policy_diff": diff,
        }
        _write_json(self._source_policy_path(source_policy_id), source_policy)
        audit_event = self.store.append_audit(
            "connector.source_policy.confirmed",
            requested_scope,
            {"type": "connector_source_policy", "id": source_policy_id},
            {
                "contract_version_id": contract["contract_version_id"],
                "confirmation_kind": confirmation_kind,
                "broadened": False,
                "narrowed_fields": diff["narrowed_fields"],
            },
        )
        return {
            "status": "success",
            "contract": contract,
            "source_policy": source_policy,
            "source_policy_diff": diff,
            "audit_event": audit_event,
        }

    def plan_upgrade(
        self,
        contract_id: str,
        requested_scope: dict[str, str],
        target_provider_pack_id: str,
        contract_version_id: str | None = None,
    ) -> dict[str, Any]:
        contract = self.load_contract(contract_id, contract_version_id=contract_version_id)
        if contract is None:
            return {"status": "not_found", "contract_id": contract_id}
        if contract.get("scope") != requested_scope:
            return {"status": "scope_denied", "resource_scope": contract.get("scope")}

        current_provider_pack_id = contract["source_policy_request"].get("provider_pack_id") or "local_source_control_readonly.v1"
        target_pack = FIXTURE_PROVIDER_PACKS.get(target_provider_pack_id)
        supported = target_pack.get("capabilities", {}) if target_pack else {}
        incompatible: list[dict[str, Any]] = []
        compatible_mappings: list[dict[str, Any]] = []
        for need in contract["needs"]:
            capability = need["common_capability"]
            accepted = set(need.get("accepted_projection_types", []))
            projections = [projection for projection in supported.get(capability, []) if projection in accepted]
            if projections:
                compatible_mappings.append(
                    {
                        "common_capability": capability,
                        "required": bool(need["required"]),
                        "projection_types": projections,
                    }
                )
            elif need["required"]:
                incompatible.append(
                    {
                        "common_capability": capability,
                        "required": True,
                        "reason_code": "CS_CONNECTOR_PROVIDER_PACK_INCOMPATIBLE",
                        "accepted_projection_types": sorted(accepted),
                    }
                )
        compatible = bool(target_pack) and not incompatible
        upgrade_plan = {
            "schema_version": "cs.connector_upgrade_plan.v1",
            "upgrade_plan_id": f"cup_{json_hash({'contract': contract['contract_version_id'], 'target': target_provider_pack_id})[:16]}",
            "contract_id": contract["contract_id"],
            "contract_version_id": contract["contract_version_id"],
            "scope": requested_scope,
            "current_provider_pack_id": current_provider_pack_id,
            "target_provider_pack_id": target_provider_pack_id,
            "pinned_versions_remain_active": True,
            "activation_blocked_until_reviewed": True,
            "rollback_available": True,
            "compatibility": {
                "status": "compatible" if compatible else "incompatible",
                "compatible_mappings": compatible_mappings,
                "incompatible_items": incompatible,
            },
            "migration_plan": {
                "required": current_provider_pack_id != target_provider_pack_id,
                "steps": [
                    "Review provider-pack diff.",
                    "Run fixture scenario verification.",
                    "Confirm Source Policy for the target provider pack.",
                    "Keep the current pinned provider active until review completes.",
                ],
                "rollback": {
                    "available": True,
                    "provider_pack_id": current_provider_pack_id,
                },
            },
            "provider_pack_diff": {
                "current": current_provider_pack_id,
                "target": target_provider_pack_id,
                "target_known": bool(target_pack),
                "product_handler_contract_hash": _product_handler_contract_hash(),
            },
            "created_at": utc_now(),
        }
        _write_json(self._upgrade_plan_path(upgrade_plan["upgrade_plan_id"]), upgrade_plan)
        audit_event = self.store.append_audit(
            "connector.upgrade.planned",
            requested_scope,
            {"type": "connector_upgrade_plan", "id": upgrade_plan["upgrade_plan_id"]},
            {
                "contract_version_id": contract["contract_version_id"],
                "target_provider_pack_id": target_provider_pack_id,
                "compatibility": upgrade_plan["compatibility"]["status"],
                "activation_blocked_until_reviewed": True,
            },
        )
        return {"status": "success", "contract": contract, "upgrade_plan": upgrade_plan, "audit_event": audit_event}
