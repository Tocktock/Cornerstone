# Connector Hub CS-CH-H01 Human Review Template - 2026-06-24

**Owner:** JiYong / Tars
**Scenario:** `CS-CH-H01`
**Gate:** Human GitHub read-only rehearsal
**Status:** PENDING HUMAN REVIEW

## Review Target

Decide whether a real GitHub installation or equivalent least-privilege connection proves live selected-repository read-only readiness for CornerStone Connected Sources.

This template closes only the human evidence collection shape. It does not mark `CS-CH-H01` as `PASS`; the scenario remains `HUMAN_REQUIRED` until a dated human record with redacted live evidence exists.

## Preparation Command

Create the non-mutating review package:

```sh
PATH="$PWD:$PATH" cornerstone connector human-gate package --scenario CS-CH-H01 --state-dir tmp/manual-connector-h01 --json
```

Expected package properties:

```text
schema_version=cs.connector_human_gate_package.v1
scenario_id=CS-CH-H01
status=human_review_required
approval_status=pending
live_provider_read_verified=HUMAN_REQUIRED
live_provider_write_verified=OUT_OF_SCOPE_READ_ONLY
product_claim_allowed=false
pass_claim_allowed_without_human_record=false
live_provider_calls_executed_by_package=0
provider_mutations_executed_by_package=0
github_write_calls_by_package=0
```

## Current Machine-Readable Handoff Snapshot

This snapshot is derived from `reports/scenario/connectorhub-human-gate-package-cs-ch-h01-2026-06-24.json`. It is operator-preparation only: `final_verdict=HUMAN_REQUIRED`, `goal_completion_claim_blocked=true`, and `full_goal_completion_allowed=false`.

| Handoff artifact | Purpose |
|---|---|
| `reports/scenario/connectorhub-human-gate-package-cs-ch-h01-2026-06-24.json` | Pinned H01 package with required evidence, dependency refs, and blank reviewer-record template. |
| `reports/scenario/connectorhub-human-gate-readiness-2026-06-24.json` | Pinned all-human-gate readiness rollup; this is not approval evidence. |
| `reports/scenario/connectorhub-human-gate-validation-handoff-2026-06-24.json` | Pinned ordered validation handoff with the H01 row and dependency blockers. |
| `reports/scenario/connectorhub-human-gate-record-template-cs-ch-h01-2026-06-24.json` | Pinned blank H01 reviewer-record template; this is not approval evidence. |
| `reports/scenario/connectorhub-human-gate-validation-blank-cs-ch-h01-2026-06-24.json` | Pinned blocked validation envelope proving the blank template remains invalid until a human fills required evidence. |

Current GitHub read-only evidence-packet workflow:

This workflow is operator-preparation only. It records packet file hashes and redacted refs, but it does not record raw packet contents, collect human acceptance, unlock dependencies, or promote `CS-CH-H01` to `PASS`.

| Step | Command | Output role |
|---:|---|---|
| 1 | `cornerstone connector human-gate evidence-packet-contract --scenario CS-CH-H01 --json` | Required evidence manifest and redaction contract. |
| 2 | `cornerstone connector human-gate evidence-packet-file-contract --scenario CS-CH-H01 --json` | Required GitHub read-only packet file list and content expectations. |
| 3 | `cornerstone connector human-gate evidence-packet-scaffold --scenario CS-CH-H01 --packet-dir <h01-github-readonly-packet-dir> --json --write` | Blank local packet templates without overwriting existing evidence. |
| 4 | `cornerstone connector human-gate evidence-packet-validate --scenario CS-CH-H01 --packet-dir <h01-github-readonly-packet-dir> --json` | Hash-only packet validation envelope. |
| 5 | `cornerstone connector human-gate evidence-packet-record-draft --scenario CS-CH-H01 --packet-dir <h01-github-readonly-packet-dir> --json --record-output <reviewer-record-draft.json>` | Hash-only reviewer record draft; human decision fields remain human-owned. |
| 6 | `cornerstone connector human-gate validate-record --scenario CS-CH-H01 --record-file <filled-reviewer-record.json> --json --output <redacted-validation-envelope.json>` | Redacted structural validation envelope for the completed reviewer record. |

Boundary flags: `schema_version=cs.connector_human_gate_h01_evidence_packet_workflow.v1`, `claim_boundary=h01_github_readonly_packet_workflow_is_operator_handoff_not_human_acceptance`, `acceptance_sufficient=false`, `product_claim_allowed=false`, `pass_claim_allowed=false`, `dependency_unlock_allowed_by_workflow=false`, `human_acceptance_collected_by_workflow=false`, `raw_packet_file_contents_recorded_by_workflow=false`, and `packet_file_contents_persisted_by_workflow=false`.

## Reviewer Record Submission Checklist

The blank reviewer record is still invalid until a human fills the generated `record_template` with redacted evidence refs and runs the structural validator. No row below is approval evidence, product acceptance, or a `PASS` claim.

| Submission item | Template source | Required action | Validator guard |
|---|---|---|---|
| Blank reviewer template | `reports/scenario/connectorhub-human-gate-record-template-cs-ch-h01-2026-06-24.json` | Write a blank template with `cornerstone connector human-gate package --scenario CS-CH-H01 --json --record-template-output <reviewer-record-template.json>`, then fill it from the live read-only evidence packet. | The blank template is preparation data only; `blank_template_requires_human_evidence` keeps H01 `HUMAN_REQUIRED`. |
| Redacted validation envelope | `proposed_record_template.validation_output_command` | Submit the filled JSON with `cornerstone connector human-gate validate-record --scenario CS-CH-H01 --record-file <filled-json> --json --output <redacted-validation-envelope.json>`. | Redacted structural validation only; no raw record body, raw record path, decision value, senior-review finding text, or evidence-packet manifest values are persisted. |
| Required fields | `proposed_record_template.required_fields` | Fill `reviewer`, `review_timestamp`, `github_app_installation_id_redacted`, `selected_repositories`, `permission_snapshot`, `call_ledger`, `delivery_refs`, `audit_refs`, `zero_write_proof`, and `issues_or_exceptions`. | Missing fields keep the validator envelope blocked and keep final verdict `HUMAN_REQUIRED`. |
| Evidence packet manifest | `proposed_record_template.required_evidence_packet_manifest` | Attach one distinct redacted evidence ref for `Redacted GitHub App or equivalent least-privilege installation permission snapshot.`, `Selected repository list and explicit confirmation that no unselected repositories were ingested.`, `Read-only call ledger showing zero write-capable HTTP methods or endpoints.`, `Connector Delivery, Artifact/Evidence, and audit refs produced from selected repositories.`, and `Independent zero-write proof covering permissions, declared actions, routes, UI/CLI, and observed calls.` | Each row must keep the matching required-evidence label and use only `redacted`, `public_safe`, or `no_sensitive_material` as `redaction_status`. |
| Senior review perspectives | `proposed_record_template.required_senior_review_perspectives` | Fill `product_value`, `domain_architecture`, `data_contract`, `reliability_observability`, `security_privacy`, and `testability_migration` findings. | Finding text is not persisted by the validator; missing perspectives remain structural issues. |
| Dependency refs | `proposed_record_template.dependency_human_gates` | Attach structurally valid ACCEPT `connector_human_gate_record_validation:<id>` refs for `CS-CH-H04` and `CS-CH-H07`. | REJECT records do not unlock dependents; missing dependency refs keep H01 blocked. |

## Scenario-First Execution Runbook

Run this gate after the production-like security and recovery prerequisites are accepted or explicitly bounded. This review proves only live selected-repository GitHub read behavior. It must not prove or imply GitHub write readiness, organization-wide GitHub access, production topology readiness, or release approval.

| Step | Operator action | Evidence to attach | Reject immediately if |
|---:|---|---|---|
| 1 | Freeze the reviewed scope: tenant, owner, namespace, workspace, GitHub installation/account label, selected repositories, ConnectorPort version, Source Policy version, and evidence directory. | Scope manifest, selected repository list, Source Policy snapshot, reviewer, timestamp. | Scope is ambiguous, repository selection is broader than approved, or selected/unselected repos cannot be distinguished. |
| 2 | Capture the GitHub permission snapshot before any live read. | Redacted GitHub App or equivalent permission screenshot/export, selected-repository permission mode, granted scopes. | Any issue/comment/label/contents write, admin, settings, branch, release, merge, or organization-wide write scope is present. |
| 3 | Run one live selected-repository read path through ConnectorPort/ConnectorHub. | Redacted call ledger, connector app/source id, request id, response metadata, Delivery refs. | The Product layer calls GitHub directly, provider credentials leave ConnectorHub, or the call touches an unselected repository. |
| 4 | Archive the resulting Delivery into CornerStone evidence state. | Delivery receipt, Artifact or intake ref, Evidence Bundle ref if assembled, audit refs. | Delivery is acknowledged before durable commit, lacks source scope, or stores raw provider payload beyond approved policy. |
| 5 | Exercise unselected-repository and selection-broadening denial paths. | Denial transcript, policy decision ids, audit refs, zero Artifact/receipt/ack counts for unselected resources. | Unselected repositories appear in outputs, fallback organization access is used, or broadening is accepted silently. |
| 6 | Run and attach zero-write proof for the same reviewed scope. | Local zero-write guard output, live call ledger summary, denied write-path evidence if attempted safely in a non-mutating way. | Any write-capable endpoint is called, any mutation method appears, or any write Provider Pack/CLI/Product path is available for GitHub. |
| 7 | Verify redaction and audit correlation across setup, Source Policy, delivery/evidence, denial, and zero-write records. | Audit correlation ids, redaction scan result, evidence manifest/hash, reviewer note. | Tokens, authorization headers, private keys, raw provider payloads, raw access handles, or uncorrelated audit events appear. |

## Acceptance Evidence Packet

Attach a dated packet with these files or equivalent redacted artifacts:

| Artifact | Required contents |
|---|---|
| `github-scope.md` | Tenant/owner/namespace/workspace, GitHub installation/account label, selected repositories, unselected repository count/labels, Source Policy version, reviewer, timestamp. |
| `permission-snapshot.md` | Redacted permission screenshot/export, granted scopes, selected-repository mode, explicit statement that write/admin scopes are absent. |
| `read-call-ledger.json` | Live read request ids, endpoints or operation labels, HTTP methods, selected repository refs, response metadata, zero mutation markers. |
| `delivery-evidence.json` | Delivery refs, Artifact/intake refs, Evidence Bundle refs where applicable, Source Policy refs, audit refs. |
| `unselected-denial.txt` | Denial transcript, policy decision ids, zero unselected Artifact/receipt/ack counts, no organization-wide fallback proof. |
| `zero-write-proof.json` | GitHub write guard output, provider-pack/contract/CLI/runtime scan result, call ledger `github_write_calls=0`. |
| `redaction-audit-report.json` | Token/header/key/raw-payload scan result, audit correlation ids, evidence manifest hash/checksum. |
| `review-decision.md` | ACCEPT or REJECT, reviewer, timestamp, evidence packet path, issues/exceptions, release impact. |

## Redaction And Handling Rules

- Do not paste GitHub tokens, private keys, authorization headers, raw provider payloads, private repository contents, credential-bearing URLs, or raw access handles into this template.
- Use stable redacted labels for account, installation, repository, request, and audit identifiers so the packet can be correlated without exposing private values.
- If the live read touches private repository content, include only metadata, hashes, file/path classes, or redacted excerpts allowed by the Source Policy.
- Any observed write-capable permission or write call makes this review a rejection, even if the selected-repository read path also succeeds.

## Senior Review Perspectives

| Perspective | H01 acceptance question |
|---|---|
| Product value | Does live GitHub evidence support a CornerStone connected-source workflow without exposing ConnectorHub as a second product? |
| Domain architecture | Did Product remain behind ConnectorPort, with credentials and provider clients inside the ConnectorHub boundary? |
| Data contract | Are selected repositories, Projection/Delivery refs, Artifact/Evidence refs, and audit refs present for every accepted live item? |
| Reliability and observability | Is there a redacted call ledger, freshness/failure state, delivery receipt evidence, and audit trail that can be replayed? |
| Security and privacy | Are write permissions, write endpoints, credential leaks, unselected repository access, and raw provider payload exposure all absent? |
| Testability and migration | Does the live rehearsal match the local `CS-CH-015` through `CS-CH-020` fixture contracts without changing product semantics? |

## Decision Record

```text
Decision: ACCEPT / REJECT
Reviewer:
Review timestamp:
GitHub installation/account identifier, redacted:
Selected repositories:
Evidence location:
Issues or exceptions:
```

## Evidence Checklist

| Required evidence | Human result | Notes |
|---|---|---|
| Redacted GitHub App or equivalent installation permission snapshot is attached. | PENDING | |
| Permission snapshot shows read-only selected-repository access and no write/admin permissions. | PENDING | |
| Selected repository list is attached and scoped to the intended tenant/owner/namespace/workspace. | PENDING | |
| Unselected repositories are absent from CornerStone outputs and provider call evidence. | PENDING | |
| One selected-repository live-read path produces Connector Delivery refs. | PENDING | |
| The same path produces Artifact or Evidence refs in CornerStone. | PENDING | |
| Audit refs exist for setup, source policy, delivery/evidence, and zero-write guard evidence. | PENDING | |
| Redacted provider/API call ledger is attached. | PENDING | |
| Call ledger shows `github_write_calls=0`. | PENDING | |
| Call ledger shows no write-capable HTTP method or write endpoint. | PENDING | |
| No token, private key, authorization header, raw provider payload, or raw access handle appears in evidence. | PENDING | |
| Local zero-write guard evidence is attached for the same source scope. | PENDING | |
| Live evidence does not claim production tenant/RLS/OPA/network readiness. | PENDING | |

## Acceptance Statement

Use this if accepted:

```text
I accept CS-CH-H01 for live GitHub read-only rehearsal.
The attached redacted evidence proves selected-repository live-read behavior, Delivery/Evidence/Audit correlation, and zero GitHub write capability or calls for the reviewed scope.
This acceptance does not claim production security readiness, production topology readiness, human UX/privacy acceptance, or GitHub write readiness.
```

## Rejection Statement

Use this if rejected:

```text
I reject CS-CH-H01.
The live GitHub read-only rehearsal does not yet prove the required boundary.
Blocking issues:
1.
2.
3.
```

## Boundary

This human review can only accept live GitHub read-only rehearsal for `CS-CH-H01`. It must not claim:

- GitHub write readiness;
- production RLS, OPA, egress, backup/restore, or security readiness;
- physical macOS or Chrome privacy acceptance;
- human usability acceptance;
- production release approval.
