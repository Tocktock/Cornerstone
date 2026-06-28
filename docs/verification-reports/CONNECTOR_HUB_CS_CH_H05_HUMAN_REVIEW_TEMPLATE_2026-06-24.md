# Connector Hub CS-CH-H05 Human Review Template - 2026-06-24

**Owner:** JiYong / Tars
**Scenario:** `CS-CH-H05`
**Gate:** Human live non-GitHub Action execution
**Status:** PENDING HUMAN REVIEW

## Review Target

Decide whether one separately approved live declared Action against a reversible non-GitHub target changes external state exactly once and re-ingests the outcome as evidence.

This template prepares the human evidence shape only. It does not mark `CS-CH-H05` as `PASS`; the scenario remains `HUMAN_REQUIRED` until authorized live mutation evidence exists.

## Preparation Command

```sh
PATH="$PWD:$PATH" cornerstone connector human-gate package --scenario CS-CH-H05 --state-dir tmp/manual-connector-h05 --json
```

Current non-GitHub live Action evidence-packet workflow:

```sh
cornerstone connector human-gate evidence-packet-contract --scenario CS-CH-H05 --json
cornerstone connector human-gate evidence-packet-file-contract --scenario CS-CH-H05 --json
cornerstone connector human-gate evidence-packet-scaffold --scenario CS-CH-H05 --packet-dir <h05-live-action-packet-dir> --json --write
cornerstone connector human-gate evidence-packet-validate --scenario CS-CH-H05 --packet-dir <h05-live-action-packet-dir> --json
cornerstone connector human-gate evidence-packet-record-draft --scenario CS-CH-H05 --packet-dir <h05-live-action-packet-dir> --json --record-output <reviewer-record-draft.json>
cornerstone connector human-gate validate-record --scenario CS-CH-H05 --record-file <filled-reviewer-record.json> --json --output <redacted-validation-envelope.json>
```

Workflow boundary:

```text
schema_version=cs.connector_human_gate_h05_evidence_packet_workflow.v1
claim_boundary=h05_live_action_packet_workflow_is_operator_handoff_not_human_acceptance
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
scenario_id=CS-CH-H05
status=human_review_required
approval_status=pending
live_external_mutation_verified=HUMAN_REQUIRED
github_actions_excluded=true
product_claim_allowed=false
pass_claim_allowed_without_human_record=false
github_actions_executed_by_package=0
```

## Current Machine-Readable Handoff Snapshot

This snapshot is derived from `reports/scenario/connectorhub-human-gate-package-cs-ch-h05-2026-06-24.json`. It is operator-preparation only: `final_verdict=HUMAN_REQUIRED`, `goal_completion_claim_blocked=true`, and `full_goal_completion_allowed=false`.

| Handoff artifact | Purpose |
|---|---|
| `reports/scenario/connectorhub-human-gate-package-cs-ch-h05-2026-06-24.json` | Pinned H05 package with required evidence, dependency refs, and blank reviewer-record template. |
| `reports/scenario/connectorhub-human-gate-readiness-2026-06-24.json` | Pinned all-human-gate readiness rollup; this is not approval evidence. |
| `reports/scenario/connectorhub-human-gate-validation-handoff-2026-06-24.json` | Pinned ordered validation handoff with the H05 row and dependency blockers. |
| `reports/scenario/connectorhub-human-gate-record-template-cs-ch-h05-2026-06-24.json` | Pinned blank H05 reviewer-record template; this is not approval evidence. |
| `reports/scenario/connectorhub-human-gate-validation-blank-cs-ch-h05-2026-06-24.json` | Pinned blocked validation envelope proving the blank template remains invalid until a human fills required evidence. |

## Reviewer Record Submission Checklist

The blank reviewer record is still invalid until a human fills the generated `record_template` with redacted evidence refs and runs the structural validator. No row below is approval evidence, product acceptance, or a `PASS` claim.

| Submission item | Template source | Required action | Validator guard |
|---|---|---|---|
| Blank reviewer template | `reports/scenario/connectorhub-human-gate-record-template-cs-ch-h05-2026-06-24.json` | Write a blank template with `cornerstone connector human-gate package --scenario CS-CH-H05 --json --record-template-output <reviewer-record-template.json>`, then fill it from the authorized live-action evidence packet. | The blank template is preparation data only; `blank_template_requires_human_evidence` keeps H05 `HUMAN_REQUIRED`. |
| Redacted validation envelope | `proposed_record_template.validation_output_command` | Submit the filled JSON with `cornerstone connector human-gate validate-record --scenario CS-CH-H05 --record-file <filled-json> --json --output <redacted-validation-envelope.json>`. | Redacted structural validation only; no raw record body, raw record path, decision value, senior-review finding text, or evidence-packet manifest values are persisted. |
| Required fields | `proposed_record_template.required_fields` | Fill `reviewer`, `review_timestamp`, `approved_provider`, `reversible_test_target`, `rollback_or_compensation_plan`, `approval_ref`, `redacted_request_result`, `provider_receipt`, `idempotency_evidence`, `audit_refs`, and `issues_or_exceptions`. | Missing fields keep the validator envelope blocked and keep final verdict `HUMAN_REQUIRED`. |
| Evidence packet manifest | `proposed_record_template.required_evidence_packet_manifest` | Attach one distinct redacted evidence ref for `Non-GitHub provider and reversible target approval.`, `Pre-execution rollback or compensation plan.`, `Redacted request/result and provider receipt/state evidence.`, and `Idempotency and audit refs proving exactly-once execution.` | Each row must keep the matching required-evidence label and use only `redacted`, `public_safe`, or `no_sensitive_material` as `redaction_status`. |
| Senior review perspectives | `proposed_record_template.required_senior_review_perspectives` | Fill `product_value`, `domain_architecture`, `data_contract`, `reliability_observability`, `security_privacy`, and `testability_migration` findings. | Finding text is not persisted by the validator; missing perspectives remain structural issues. |
| Dependency refs | `proposed_record_template.dependency_human_gates` | Attach structurally valid ACCEPT `connector_human_gate_record_validation:<id>` refs for `CS-CH-H04` and `CS-CH-H07`. | REJECT records do not unlock dependents; missing dependency refs keep H05 blocked. |

## Scenario-First Execution Runbook

1. Freeze the live-action scope before any credential or provider access: record the non-GitHub provider, provider account/workspace, reversible test target, declared connector capability/action type, CornerStone workspace/namespace, reviewer/approver, rollback or compensation plan, and evidence directory. Reject the review if the target is GitHub, production-critical, irreversible, ambiguous, or outside the approved owner scope.
2. Verify pre-execution safety envelope: attach the Evidence Bundle, ActionCard dry-run, ConnectorHub Action Preflight, Product policy decision, Source Policy snapshot, expected provider calls, predicted diff/impact, risk label, required permissions, idempotency key, and audit refs. Reject if any gate is stale, missing, unsupported, permission-denied, policy-denied, or side-effecting during preflight.
3. Record explicit human approval immediately before execution: attach approval ref, approver authority, timestamp, approved action digest, expected target state delta, and rollback/compensation acknowledgement. Reject if approval is implicit, reused from another action, granted by an unauthorized reviewer, or not bound to the final request digest.
4. Execute exactly one declared non-GitHub Action through the governed WorkflowRun and ConnectorHub execution path. Attach the redacted request, execution command or UI/API transcript, WorkflowRun id, Action Result id, provider receipt, idempotency record, audit refs, and before/after provider-state proof. Reject if Product or an agent uses a direct provider client, if the action bypasses ConnectorHub, or if any undeclared provider endpoint is reached.
5. Verify exactly-once behavior: replay the same idempotency key and same request digest, then submit or simulate a conflicting-intent retry when safe. Attach replay result, conflict denial or reconciliation note, durable counts, and provider-state delta proof. Reject if the same request creates a second side effect, conflicting intent executes, or idempotency scope is not tied to owner/workspace/provider/action.
6. Re-ingest the outcome as CornerStone evidence: attach outcome Artifact/Evidence Bundle refs, connected outcome record, result summary, provider receipt linkage, audit chain, and any follow-up Claim/Mission state if created. Reject if the outcome remains only in provider state or if CornerStone records unsupported claims without evidence refs.
7. Execute rollback or compensation only when the approved test plan requires it; otherwise record why no rollback was needed. Attach rollback/compensation transcript, provider receipt, state proof, audit refs, and residual-risk note. Reject if a claimed rollback has no provider evidence or if the plan promises atomic rollback for a provider that cannot guarantee it.
8. Verify exclusion, redaction, and custody boundaries: attach GitHub-exclusion proof, direct-provider-bypass scan, credential/secret scan, redacted request/result, provider receipt redaction, and evidence manifest/checksum. Reject if GitHub is used, raw credentials are exposed, raw provider payloads are logged unnecessarily, or any unapproved private data appears in the evidence packet.

## Acceptance Evidence Packet

| Artifact | Required content | Human result |
|---|---|---|
| `live-action-scope.md` | Non-GitHub provider, account/workspace, reversible target, capability/action type, CornerStone workspace/namespace, reviewer/approver, rollback or compensation plan, evidence directory. | PENDING |
| `pre-execution-safety-envelope.json` | Evidence Bundle, ActionCard dry-run, ConnectorHub preflight, policy decision, Source Policy, expected calls, predicted diff, risk, permissions, idempotency key, audit refs. | PENDING |
| `approval-record.md` | Approver authority, timestamp, approved action digest, expected state delta, rollback/compensation acknowledgement. | PENDING |
| `execution-transcript.json` | Governed WorkflowRun/ConnectorHub execution transcript, redacted request, Action Result, provider receipt, idempotency record, audit refs. | PENDING |
| `provider-state-delta.md` | Before/after provider-state proof, exactly one intended external mutation, no unrelated state changes. | PENDING |
| `idempotency-replay-proof.json` | Same-key replay result, duplicate side-effect count, conflict denial or reconciliation note, durable counts. | PENDING |
| `outcome-reingest-proof.json` | Outcome Artifact/Evidence Bundle, connected outcome record, provider receipt linkage, audit chain, follow-up Claim/Mission state if any. | PENDING |
| `rollback-compensation-proof.md` | Rollback/compensation execution or explicit not-needed rationale, provider receipt, final state proof, residual-risk note. | PENDING |
| `github-exclusion-proof.json` | Provider is not GitHub, GitHub write actions/endpoints/calls are absent, release read-only invariant remains intact. | PENDING |
| `boundary-redaction-report.json` | Direct-provider-bypass scan, credential/secret scan, redacted request/result review, evidence manifest or checksum. | PENDING |
| `review-decision.md` | Reviewer decision, timestamp, accept/reject rationale, exceptions, follow-ups. | PENDING |

## Redaction And Handling Rules

- Do not paste raw provider credentials, access tokens, private customer data, payment data, private messages, or unredacted provider payloads into this template or companion evidence.
- Use redacted provider account, target, and receipt labels while preserving stable hashes, request digests, idempotency keys, audit refs, and before/after state identifiers needed for review.
- Prefer a reversible synthetic or sandbox target. If the target is live but not synthetic, the reviewer must record why the business risk is acceptable and how compensation is verified.
- Treat GitHub usage, missing approval, direct provider-client execution, stale preflight, missing idempotency, duplicate side effect, missing provider receipt, missing outcome re-ingest, or leaked secrets/private payloads as reject conditions.

## Senior Review Perspectives

| Perspective | H05 acceptance question |
|---|---|
| Product value | Does the governed live Action create useful outcome evidence without normalizing autonomous external mutation? |
| Domain architecture | Does execution stay behind declared actions, policy, approval, idempotency, and ConnectorHub-mediated provider access? |
| Data contract | Are ActionCard, policy decision, approval, result receipt, outcome evidence, idempotency, and audit refs attached? |
| Reliability and observability | Is provider receipt/state evidence paired with a rollback or compensation plan for the reversible target? |
| Security and privacy | Are GitHub usage, missing approval, non-reversible targets, and credential leaks absent? |
| Testability and migration | Do live execution semantics match the `CS-CH-029` through `CS-CH-033` fixture contracts? |

## Decision Record

```text
Decision: ACCEPT / REJECT
Reviewer:
Review timestamp:
Approved non-GitHub provider:
Reversible test target:
Evidence location:
Issues or exceptions:
```

## Evidence Checklist

| Required evidence | Human result | Notes |
|---|---|---|
| Non-GitHub provider and reversible target are approved before execution. | PENDING | |
| Rollback or compensation plan is attached before execution. | PENDING | |
| ActionCard, policy decision, and human approval refs are attached. | PENDING | |
| Redacted request/result and provider receipt/state evidence are attached. | PENDING | |
| Idempotency proof shows exactly-once execution. | PENDING | |
| Outcome is re-ingested as CornerStone evidence with audit refs. | PENDING | |
| No GitHub action target, write endpoint, or GitHub mutation appears. | PENDING | |

## Acceptance Statement

Accept only if the Action follows `CS-CH-029` through `CS-CH-033`, mutates exactly once, and remains reversible or compensable. Reject if GitHub is used.

## Boundary

This review blocks live Action readiness. GitHub remains excluded from this mutation gate.
