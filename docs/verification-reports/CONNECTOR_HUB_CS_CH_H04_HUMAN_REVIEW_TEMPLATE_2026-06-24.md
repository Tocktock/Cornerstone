# Connector Hub CS-CH-H04 Human Review Template - 2026-06-24

**Owner:** JiYong / Tars
**Scenario:** `CS-CH-H04`
**Gate:** Human production-like VS2 integrated security proof
**Status:** PENDING HUMAN REVIEW

## Review Target

Decide whether a production-like topology proves connected-source tenant isolation, policy enforcement, default-deny egress, backup/restore, and audit integrity under real request context.

This template prepares the human evidence shape only. It does not mark `CS-CH-H04` as `PASS`; the scenario remains `HUMAN_REQUIRED` until integrated environment evidence exists.

## Preparation Command

```sh
PATH="$PWD:$PATH" cornerstone connector human-gate package --scenario CS-CH-H04 --state-dir tmp/manual-connector-h04 --json
PATH="$PWD:$PATH" cornerstone connector human-gate field-ref-contract --scenario CS-CH-H04 --state-dir tmp/manual-connector-h04 --json
PATH="$PWD:$PATH" cornerstone connector human-gate evidence-packet-contract --scenario CS-CH-H04 --state-dir tmp/manual-connector-h04 --json
PATH="$PWD:$PATH" cornerstone connector human-gate evidence-packet-file-contract --scenario CS-CH-H04 --state-dir tmp/manual-connector-h04 --json
PATH="$PWD:$PATH" cornerstone connector human-gate evidence-packet-scaffold --scenario CS-CH-H04 --packet-dir <h04-acceptance-packet-dir> --state-dir tmp/manual-connector-h04 --json
PATH="$PWD:$PATH" cornerstone connector human-gate evidence-packet-validate --scenario CS-CH-H04 --packet-dir <h04-acceptance-packet-dir> --state-dir tmp/manual-connector-h04 --json
PATH="$PWD:$PATH" cornerstone connector human-gate evidence-packet-record-draft --scenario CS-CH-H04 --packet-dir <h04-acceptance-packet-dir> --record-output <reviewer-record-draft.json> --state-dir tmp/manual-connector-h04 --json
PATH="$PWD:$PATH" cornerstone connector human-gate preflight-bundle --scenario CS-CH-H04 --state-dir tmp/manual-connector-h04 --json
```

Expected package properties:

```text
schema_version=cs.connector_human_gate_package.v1
scenario_id=CS-CH-H04
status=human_review_required
approval_status=pending
production_like_request_context_verified=HUMAN_REQUIRED
production_tenancy_policy_egress_verified=HUMAN_REQUIRED
production_readiness_verified=NOT_VERIFIED
product_claim_allowed=false
pass_claim_allowed_without_human_record=false
```

Expected field-ref-contract properties:

```text
schema_version=cs.connector_human_gate_field_ref_contract_report.v1
scenario_id=CS-CH-H04
status=operator_preparation_only
final_verdict=HUMAN_REQUIRED
required_field_ref_item_count=7
raw_field_values_recorded_by_report=false
raw_field_values_persisted_by_validator=false
invalid_value_report_shape=field_names_only
commands_executed_by_field_ref_contract=0
live_provider_calls_executed_by_field_ref_contract=0
provider_mutations_executed_by_field_ref_contract=0
external_mutations_executed_by_field_ref_contract=0
```

Expected evidence-packet-contract properties:

```text
schema_version=cs.connector_human_gate_evidence_packet_contract_report.v1
scenario_id=CS-CH-H04
status=operator_preparation_only
final_verdict=HUMAN_REQUIRED
required_evidence_packet_manifest_count=4
allowed_redaction_statuses=redacted,public_safe,no_sensitive_material
raw_evidence_ref_values_recorded_by_report=false
evidence_packet_manifest_values_persisted_by_validator=false
invalid_value_report_shape=field_names_and_required_evidence_indexes_only
commands_executed_by_evidence_packet_contract=0
live_provider_calls_executed_by_evidence_packet_contract=0
provider_mutations_executed_by_evidence_packet_contract=0
external_mutations_executed_by_evidence_packet_contract=0
```

Expected evidence-packet-file-contract properties:

```text
schema_version=cs.connector_human_gate_evidence_packet_file_contract_report.v1
scenario_id=CS-CH-H04
status=operator_preparation_only
final_verdict=HUMAN_REQUIRED
required_packet_file_count=8
raw_packet_file_contents_recorded_by_report=false
packet_file_contents_persisted_by_report=false
packet_file_contents_persisted_by_validator=false
review_input_only=true
acceptance_sufficient=false
product_claim_allowed=false
pass_claim_allowed=false
packet_file_scaffold_plan_available=true
packet_file_scaffold_directory=<h04-acceptance-packet-dir>
packet_file_scaffold_command_count=9
packet_file_scaffold_plan_executed_by_report=false
packet_file_scaffold_plan_review_input_only=true
packet_file_scaffold_plan_acceptance_sufficient=false
commands_executed_by_evidence_packet_file_contract=0
live_provider_calls_executed_by_evidence_packet_file_contract=0
provider_mutations_executed_by_evidence_packet_file_contract=0
external_mutations_executed_by_evidence_packet_file_contract=0
```

Expected evidence-packet-scaffold properties:

```text
schema_version=cs.connector_human_gate_evidence_packet_scaffold_report.v1
scenario_id=CS-CH-H04
status=operator_preparation_only
final_verdict=HUMAN_REQUIRED
scaffold_template_count=8
write_requested=false
write_executed=false
template_contents_included_in_report=false
packet_file_contents_read_by_scaffold=false
human_evidence_recorded_by_scaffold=false
review_input_only=true
acceptance_sufficient=false
product_claim_allowed=false
pass_claim_allowed=false
commands_executed_by_evidence_packet_scaffold=0
live_provider_calls_executed_by_evidence_packet_scaffold=0
provider_mutations_executed_by_evidence_packet_scaffold=0
external_mutations_executed_by_evidence_packet_scaffold=0
local_template_files_written_by_evidence_packet_scaffold=0
```

Expected evidence-packet-validation properties:

```text
schema_version=cs.connector_human_gate_evidence_packet_validation_report.v1
scenario_id=CS-CH-H04
status=packet_not_submitted
final_verdict=HUMAN_REQUIRED
packet_structurally_complete=false
raw_packet_file_contents_included_in_report=false
raw_packet_file_contents_recorded_by_validator=false
packet_file_contents_persisted_by_validator=false
packet_file_hashes_recorded_by_validator=true
dependency_unlock_allowed_by_packet_validator=false
commands_executed_by_evidence_packet_validation=0
live_provider_calls_executed_by_evidence_packet_validation=0
provider_mutations_executed_by_evidence_packet_validation=0
external_mutations_executed_by_evidence_packet_validation=0
```

Expected evidence-packet-record-draft properties:

```text
schema_version=cs.connector_human_gate_evidence_packet_record_draft_report.v1
scenario_id=CS-CH-H04
status=packet_not_ready_for_record_draft
final_verdict=HUMAN_REQUIRED
draft_record_available=false
draft_record_expected_validation_status_before_human_completion=record_structurally_invalid
raw_packet_file_contents_included_in_report=false
raw_packet_file_contents_recorded_by_draft=false
packet_file_contents_persisted_by_draft=false
dependency_unlock_allowed_by_record_draft=false
commands_executed_by_evidence_packet_record_draft=0
live_provider_calls_executed_by_evidence_packet_record_draft=0
provider_mutations_executed_by_evidence_packet_record_draft=0
external_mutations_executed_by_evidence_packet_record_draft=0
```

Expected preflight-bundle properties:

```text
schema_version=cs.connector_human_gate_preflight_bundle_report.v1
scenario_id=CS-CH-H04
status=operator_preparation_only
final_verdict=HUMAN_REQUIRED
acceptance_sufficient=false
product_claim_allowed=false
pass_claim_allowed=false
commands_executed_by_preflight_bundle=0
live_provider_calls_executed_by_preflight_bundle=0
provider_mutations_executed_by_preflight_bundle=0
external_mutations_executed_by_preflight_bundle=0
```

## Local Baseline Review Inputs

The H04 package includes a `local_baseline_review_inputs` object for operator comparison. It references the current local VS2 and ConnectorHub dependency reports:

- `reports/security/vs2-local-security-proof.json`
- `reports/scenario/vs2-policy-tenancy-egress-2026-06-19.json`
- `reports/network/vs2-egress-proof.json`
- `reports/security/vs2-local-range.json`
- `reports/scenario/connector-contract-adapter/scenarios/CS-CH-036.json`

These files are review inputs only. They are not sufficient for H04 acceptance and cannot prove production-like RequestContext, PostgreSQL/RLS, OPA, egress, backup/restore, or audit readiness. The reviewer must attach fresh production-like evidence in the Acceptance Evidence Packet below.

`cornerstone connector human-gate next --json` mirrors these H04 comparison inputs as `next_local_baseline_review_inputs` with `next_required_human_delta` and `next_recommended_preflight_commands` while keeping `acceptance_sufficient=false`; this is an operator handoff, not H04 acceptance.

`cornerstone connector human-gate field-ref-contract --scenario CS-CH-H04 --json` exposes accepted ref prefixes and container names for production-like security proof fields only. It does not include submitted field values, persist raw refs, call live providers, collect approval, record human decisions, or mark H04 as accepted.

`cornerstone connector human-gate evidence-packet-contract --scenario CS-CH-H04 --json` exposes required evidence-packet manifest indexes, labels, and allowed redaction statuses only. It does not include submitted evidence refs, persist evidence-packet values, call live providers, collect approval, record human decisions, or mark H04 as accepted.

`cornerstone connector human-gate evidence-packet-file-contract --scenario CS-CH-H04 --json` exposes the required acceptance-packet file names, required content categories, and a non-executed scaffold plan only. The scaffold plan starts with `mkdir -p <h04-acceptance-packet-dir>` and then one `touch <h04-acceptance-packet-dir>/<packet-file>` command per required file. The report keeps `packet_file_scaffold_plan_executed_by_report=false` and `packet_file_scaffold_plan_acceptance_sufficient=false`; it does not run scaffold commands, read packet files, include packet file contents, persist packet file contents, call live providers, collect approval, record human decisions, or mark H04 as accepted.

`cornerstone connector human-gate evidence-packet-scaffold --scenario CS-CH-H04 --packet-dir <h04-acceptance-packet-dir> --json` exposes blank template hashes for the eight required acceptance-packet files without writing files by default. Add `--write` only to create local blank templates; the command refuses to overwrite existing packet files, keeps `template_contents_included_in_report=false`, does not read packet file contents, does not record human evidence, and cannot mark H04 as accepted.

`cornerstone connector human-gate evidence-packet-validate --scenario CS-CH-H04 --packet-dir <h04-acceptance-packet-dir> --json` checks whether the eight required acceptance-packet files are present, non-empty, and no longer equal to the blank scaffold templates. It records metadata and hashes only; it does not include or persist packet file contents, does not collect approval, keeps `dependency_unlock_allowed_by_packet_validator=false`, and cannot mark H04 as accepted.

`cornerstone connector human-gate evidence-packet-record-draft --scenario CS-CH-H04 --packet-dir <h04-acceptance-packet-dir> --json --record-output <reviewer-record-draft.json>` can turn a structurally complete acceptance packet into a hash-only reviewer-record draft. It records no packet contents, leaves human-only fields blank, sets `dependency_unlock_allowed_by_record_draft=false`, and the generated draft must still validate as `record_structurally_invalid` until a human supplies decision, reviewer, timestamp, and senior-review findings.

`cornerstone connector human-gate preflight-bundle --scenario CS-CH-H04 --json` exposes the same local comparison bundle as a first-class non-mutating CLI artifact. It does not run the recommended commands, call live providers, collect approval, record human decisions, or mark H04 as accepted.

When filling the generated `proposed_record_template.evidence_packet_manifest`, keep each `required_evidence` label attached to its `required_evidence_index`, attach a distinct redacted `evidence_ref` for each required evidence row, and use only `redacted`, `public_safe`, or `no_sensitive_material` as `redaction_status`. The validator rejects index-only rows, duplicate evidence refs, mismatched evidence labels, unsupported redaction statuses, and any sensitive marker findings while still leaving H04 `HUMAN_REQUIRED`. Duplicate evidence refs appear in validation handoff issue summaries as `duplicate_evidence_packet_manifest_refs` with short SHA-256 fingerprints only; raw refs remain omitted.

The generated package and `cornerstone connector human-gate next --json` handoff include `redaction_guidance.sensitive_marker_policy` with marker categories for GitHub-token-like values, AWS-key-like values, private-key blocks, and scenario canaries. Validation findings expose only `marker_type`, `fingerprint`, and `length`; raw match values are not returned or persisted.

## Current Machine-Readable Handoff Snapshot

This snapshot is derived from `reports/scenario/connectorhub-human-gate-package-cs-ch-h04-2026-06-24.json`. It is operator-preparation only: `final_verdict=HUMAN_REQUIRED`, `goal_completion_claim_blocked=true`, and `full_goal_completion_allowed=false`. The local baseline subset is comparison input only: `acceptance_sufficient=false`, `product_claim_allowed=false`, and `pass_claim_allowed=false`.

| Handoff artifact | Purpose |
|---|---|
| `reports/scenario/connectorhub-human-gate-package-cs-ch-h04-2026-06-24.json` | Pinned H04 package with local baseline comparison inputs and blank reviewer-record template. |
| `reports/scenario/connectorhub-human-gate-field-ref-contract-cs-ch-h04-2026-06-24.json` | Pinned first-class H04 field-ref-contract envelope; this is operator-preparation only and records no submitted field values or approval evidence. |
| `reports/scenario/connectorhub-human-gate-evidence-packet-contract-cs-ch-h04-2026-06-24.json` | Pinned first-class H04 evidence-packet-contract envelope; this is operator-preparation only and records no submitted evidence refs or approval evidence. |
| `reports/scenario/connectorhub-human-gate-evidence-packet-file-contract-cs-ch-h04-2026-06-24.json` | Pinned first-class H04 evidence-packet-file-contract envelope; this is operator-preparation only and records no packet file contents or approval evidence. |
| `reports/scenario/connectorhub-human-gate-evidence-packet-scaffold-cs-ch-h04-2026-06-24.json` | Pinned first-class H04 evidence-packet-scaffold dry-run envelope; this records template hashes only, writes no files in the pinned run, and is not approval evidence. |
| `reports/scenario/connectorhub-human-gate-evidence-packet-validation-cs-ch-h04-2026-06-24.json` | Pinned first-class H04 evidence-packet-validation envelope; this records that the placeholder packet is not submitted, records no packet contents, and is not approval evidence. |
| `reports/scenario/connectorhub-human-gate-evidence-packet-record-draft-cs-ch-h04-2026-06-24.json` | Pinned first-class H04 evidence-packet-record-draft envelope; this records the placeholder packet is not ready for a draft, records no packet contents, and cannot unlock dependencies. |
| `reports/scenario/connectorhub-human-gate-preflight-bundle-cs-ch-h04-2026-06-24.json` | Pinned first-class H04 preflight-bundle envelope; this is operator-preparation only and not approval evidence. |
| `reports/scenario/connectorhub-human-gate-readiness-2026-06-24.json` | Pinned all-human-gate readiness rollup; this is not approval evidence. |
| `reports/scenario/connectorhub-human-gate-next-2026-06-24.json` | Pinned next-gate selector showing H04 as the first dependency-ready human gate. |
| `reports/scenario/connectorhub-human-gate-validation-handoff-2026-06-24.json` | Pinned ordered validation handoff for all H rows. |
| `reports/scenario/connectorhub-human-gate-record-template-cs-ch-h04-2026-06-24.json` | Pinned blank H04 reviewer-record template; this is not approval evidence. |
| `reports/scenario/connectorhub-human-gate-validation-blank-cs-ch-h04-2026-06-24.json` | Pinned blocked validation envelope proving the blank template remains invalid until a human fills required evidence. |

Current local baseline reports:

| Report | Status | Scenario count | SHA-256 | Review input | Acceptance sufficient | Product claim | PASS claim | Claim boundary |
|---|---|---:|---|---|---|---|---|---|
| `reports/security/vs2-local-security-proof.json` | `success` | 93 | `841b4ed7ec0cdafa6ec6f395a131e484690beb31508af98a30ddeb64f182c3a9` | `true` | `false` | `false` | `false` | `h04_local_baseline_snapshot_is_review_input_not_human_acceptance` |
| `reports/scenario/vs2-policy-tenancy-egress-2026-06-19.json` | `success` | 93 | `976e21f4f0142d788e06885a32d57a3fbb0e1672dc3b4dd7759afd57148a994b` | `true` | `false` | `false` | `false` | `h04_local_baseline_snapshot_is_review_input_not_human_acceptance` |
| `reports/network/vs2-egress-proof.json` | `passed` | n/a | `26505a8b3db9d44ad99657a35cf38e4cc4bcd8dffd3e145fcb6b49a29fdde9ec` | `true` | `false` | `false` | `false` | `h04_local_baseline_snapshot_is_review_input_not_human_acceptance` |
| `reports/security/vs2-local-range.json` | `passed` | n/a | `e6257b6076598b1376573c649d2b8c59b205ae2b99787cc44c368431675c9eff` | `true` | `false` | `false` | `false` | `h04_local_baseline_snapshot_is_review_input_not_human_acceptance` |
| `reports/scenario/connector-contract-adapter/scenarios/CS-CH-036.json` | `success` | 1 | `8b91ccb9a307c8783ed28ca3182e80c0eab69b13e12d67c826523eec4f0d18a8` | `true` | `false` | `false` | `false` | `h04_local_baseline_snapshot_is_review_input_not_human_acceptance` |

Required human delta:

- Production-like topology identifier and trusted RequestContext transcript.
- Scenario-specific PostgreSQL/RLS and OPA transcripts from the reviewed environment.
- Network default-deny and governed-egress transcripts from the reviewed topology.
- Backup/restore evidence and audit-integrity report from the reviewed environment.
- Dated ACCEPT or REJECT decision with redacted evidence packet manifest.

2026-06-27 owner substitution decision:

JiYong/Tars observed that the served local UI only exposes the disclaimer `Local VS1 proof only. This page does not claim production readiness, live connector readiness, or human acceptance.` JiYong/Tars cannot personally verify required human delta items 2 through 4 and approved replacing those manual review actions with local integration-test evidence for local readiness only.

| Required delta item | Owner decision | Local replacement command | Claim boundary |
|---:|---|---|---|
| 1 | Not replaced. The local page disclaimer is evidence that no production-like topology or human acceptance is claimed. | n/a | `production_like_topology_still_human_required` |
| 2 | Replace manual personal verification with local integration evidence only. | `cornerstone scenario verify vs2-policy-tenancy-egress --reuse-vs2-local-proof-report reports/security/vs2-local-security-proof.json --json` | `local_integration_substitution_is_not_h04_acceptance` |
| 3 | Replace manual personal verification with local integration evidence only. | `cornerstone scenario verify vs2-policy-tenancy-egress --reuse-vs2-local-proof-report reports/security/vs2-local-security-proof.json --json`; `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-036 --json` | `local_integration_substitution_is_not_h04_acceptance` |
| 4 | Replace manual personal verification with local integration evidence only. | `cornerstone scenario verify vs2-policy-tenancy-egress --reuse-vs2-local-proof-report reports/security/vs2-local-security-proof.json --json` | `local_integration_substitution_is_not_h04_acceptance` |
| 5 | Not replaced. A dated ACCEPT or REJECT record is still required before H04 can move out of `HUMAN_REQUIRED`. | n/a | `dated_human_decision_still_required` |

This substitution preserves `acceptance_sufficient=false`, `product_claim_allowed=false`, and `pass_claim_allowed=false` for H04.

Recommended preflight commands:

- `cornerstone security vs2-local-proof --json`
- `cornerstone scenario verify vs2-policy-tenancy-egress --reuse-vs2-local-proof-report reports/security/vs2-local-security-proof.json --json`
- `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-036 --json`

Recommended preflight command plan:

Each row below is review input only: `review_input_only=true`, `acceptance_sufficient=false`, `product_claim_allowed=false`, and `pass_claim_allowed=false`.

| Step | Operator phase | Purpose | Command | Expected report paths | Boundary flags |
|---:|---|---|---|---|---|
| 1 | `refresh_local_vs2_baseline_inputs` | Refresh the current local VS2 proof inputs before H04 review without treating local proof as production-like acceptance. | `cornerstone security vs2-local-proof --json` | `reports/security/vs2-local-security-proof.json`; `reports/network/vs2-egress-proof.json`; `reports/security/vs2-local-range.json` | `claim_boundary=h04_local_baseline_preflight_is_review_input_not_human_acceptance` |
| 2 | `refresh_vs2_scenario_report` | Refresh the local VS2 scenario report that H04 reviewers compare against the production-like environment transcript. | `cornerstone scenario verify vs2-policy-tenancy-egress --reuse-vs2-local-proof-report reports/security/vs2-local-security-proof.json --json` | `reports/security/vs2-local-security-proof.json`; `reports/scenario/vs2-policy-tenancy-egress-2026-06-19.json` | `claim_boundary=h04_local_baseline_preflight_is_review_input_not_human_acceptance` |
| 3 | `refresh_connectorhub_dependency_report` | Refresh the ConnectorHub CS-CH-036 dependency report that remains local fixture evidence until H04/H07 human proof exists. | `cornerstone scenario verify connector-contract-adapter --scenario CS-CH-036 --json` | `reports/scenario/connector-contract-adapter/scenarios/CS-CH-036.json` | `claim_boundary=h04_local_baseline_preflight_is_review_input_not_human_acceptance` |

Current local preflight bundle:

The package also carries `preflight_bundle` as one machine-checkable review-input object. It is the local comparison bundle for the five report fingerprints above, not an approval record and not H04 acceptance evidence.

| Bundle field | Current value |
|---|---|
| `schema_version` | `cs.connector_human_gate_local_baseline_preflight_bundle.v1` |
| `status` | `operator_preparation_only` |
| `baseline_scope` | `local_ai_verifiable_vs2_and_connectorhub_dependency_proof` |
| `required_human_delta_count` | `5` |
| `command_plan_count` | `3` |
| `current_report_count` | `5` |
| `ready_report_count` | `5` |
| `command_plan_expected_report_path_count` | `5` |
| `review_input_only` | `true` |
| `acceptance_sufficient` | `false` |
| `product_claim_allowed` | `false` |
| `pass_claim_allowed` | `false` |
| `commands_executed_by_bundle` | `0` |
| `live_provider_calls_executed_by_bundle` | `0` |
| `provider_mutations_executed_by_bundle` | `0` |
| `external_mutations_executed_by_bundle` | `0` |
| `human_acceptance_collected_by_bundle` | `false` |
| `claim_boundary` | `h04_local_baseline_preflight_bundle_is_review_input_not_human_acceptance` |

## Reviewer Record Submission Checklist

The blank reviewer record is still invalid until a human fills the generated `record_template` with redacted evidence refs and runs the structural validator. No row below is approval evidence, product acceptance, or a `PASS` claim.

| Submission item | Template source | Required action | Validator guard |
|---|---|---|---|
| Blank reviewer template | `reports/scenario/connectorhub-human-gate-record-template-cs-ch-h04-2026-06-24.json` | Write a blank template with `cornerstone connector human-gate package --scenario CS-CH-H04 --json --record-template-output <reviewer-record-template.json>`, then fill it from the production-like evidence packet. | The blank template is preparation data only; `blank_template_requires_human_evidence` keeps H04 `HUMAN_REQUIRED`. |
| Field-ref contract | `reports/scenario/connectorhub-human-gate-field-ref-contract-cs-ch-h04-2026-06-24.json` | Inspect accepted evidence-ref prefixes and container names with `cornerstone connector human-gate field-ref-contract --scenario CS-CH-H04 --json` before filling the reviewer record. | The contract reports field names and accepted ref shapes only; submitted values are not recorded or accepted by this artifact. |
| Evidence-packet contract | `reports/scenario/connectorhub-human-gate-evidence-packet-contract-cs-ch-h04-2026-06-24.json` | Inspect required evidence-packet manifest rows with `cornerstone connector human-gate evidence-packet-contract --scenario CS-CH-H04 --json` before attaching reviewer evidence refs. | The contract reports required labels, indexes, and redaction statuses only; submitted evidence refs are not recorded or accepted by this artifact. |
| Evidence-packet file contract | `reports/scenario/connectorhub-human-gate-evidence-packet-file-contract-cs-ch-h04-2026-06-24.json` | Inspect required acceptance-packet filenames and the non-executed scaffold plan with `cornerstone connector human-gate evidence-packet-file-contract --scenario CS-CH-H04 --json` before assembling packet files. | The contract reports file names, required content categories, and `packet_file_scaffold_command_count=9` only; scaffold commands are not executed, and packet file contents are not read, recorded, or accepted by this artifact. |
| Evidence-packet scaffold | `reports/scenario/connectorhub-human-gate-evidence-packet-scaffold-cs-ch-h04-2026-06-24.json` | Optionally run `cornerstone connector human-gate evidence-packet-scaffold --scenario CS-CH-H04 --packet-dir <h04-acceptance-packet-dir> --json --write` to create blank local packet templates before filling evidence. | The scaffold writes blank templates only on explicit `--write`, refuses overwrites, records no packet content or human evidence, and keeps H04 `HUMAN_REQUIRED`. |
| Evidence-packet validation | `reports/scenario/connectorhub-human-gate-evidence-packet-validation-cs-ch-h04-2026-06-24.json` | After filling the packet files, run `cornerstone connector human-gate evidence-packet-validate --scenario CS-CH-H04 --packet-dir <h04-acceptance-packet-dir> --json` to check presence, non-empty files, and blank-template replacement. | The validator records file metadata and hashes only, records no packet contents, allows no dependency unlock, and keeps H04 `HUMAN_REQUIRED` until a dated reviewer record is validated. |
| Evidence-packet record draft | `reports/scenario/connectorhub-human-gate-evidence-packet-record-draft-cs-ch-h04-2026-06-24.json` | After packet validation is structurally complete, run `cornerstone connector human-gate evidence-packet-record-draft --scenario CS-CH-H04 --packet-dir <h04-acceptance-packet-dir> --json --record-output <reviewer-record-draft.json>` to prepare hash-only field refs. | The draft records no packet contents, leaves human-only fields blank, validates as structurally invalid before human completion, and allows no dependency unlock. |
| Redacted validation envelope | `proposed_record_template.validation_output_command` | Submit the filled JSON with `cornerstone connector human-gate validate-record --scenario CS-CH-H04 --record-file <filled-json> --json --output <redacted-validation-envelope.json>`. | Redacted structural validation only; no raw record body, raw record path, decision value, senior-review finding text, or evidence-packet manifest values are persisted. |
| Required fields | `proposed_record_template.required_fields` | Fill `reviewer`, `review_timestamp`, `environment_topology_ref`, `request_context_proof`, `db_policy_transcripts`, `network_egress_transcripts`, `backup_restore_evidence`, `audit_integrity_report`, `evidence_manifest_ref`, and `findings_or_exceptions`. | Missing fields keep the validator envelope blocked and keep final verdict `HUMAN_REQUIRED`. |
| Evidence packet manifest | `proposed_record_template.required_evidence_packet_manifest` | Attach one distinct redacted evidence ref for `Production-like topology description and trusted RequestContext proof.`, `Scenario-specific PostgreSQL/RLS and OPA transcripts.`, `Network default-deny and governed-egress transcripts.`, and `Backup/restore, evidence manifest, and audit integrity reports.` | Each row must keep the matching required-evidence label and use only `redacted`, `public_safe`, or `no_sensitive_material` as `redaction_status`. |
| Senior review perspectives | `proposed_record_template.required_senior_review_perspectives` | Fill `product_value`, `domain_architecture`, `data_contract`, `reliability_observability`, `security_privacy`, and `testability_migration` findings. | Finding text is not persisted by the validator; missing perspectives remain structural issues. |
| Dependency refs | `proposed_record_template.dependency_human_gates` | H04 has no dependency human-gate refs to attach before submission. | Later dependency-bearing H gates require structurally valid ACCEPT `connector_human_gate_record_validation:<id>` refs; REJECT records do not unlock dependents. |

## Scenario-First Execution Runbook

Run this gate as the first human/external ConnectorHub proof surface. Later recovery, live GitHub, live Action, device/browser, and usability gates depend on this environment proving that connected-source operations preserve CornerStone scope, policy, egress, recovery, and audit boundaries under a production-like RequestContext.

| Step | Operator action | Evidence to attach | Reject immediately if |
|---:|---|---|---|
| 1 | Freeze the environment under review: namespace/workspace ids, service versions, policy bundle version, database revision, network topology, and evidence directory. | Environment topology reference, version snapshot, evidence manifest path, reviewer identity, timestamp. | The environment cannot be uniquely identified or differs from the reviewed deployment during the run. |
| 2 | Generate a trusted RequestContext and run one allowed connected-source read path inside the controlled namespace. | RequestContext trace, scoped connector app id, owner/namespace/workspace fields, policy decision id, audit ref. | Any request lacks scope fields, uses implicit global scope, or crosses namespace boundaries. |
| 3 | Run the corresponding PostgreSQL/RLS product-path checks for allowed and denied namespace cases. | SQL transcript or DB check report with redacted identifiers, allowed row proof, denied row proof, RLS policy refs. | A denied namespace can read/write data, or the transcript exposes secrets or private payloads. |
| 4 | Run OPA/policy-client decisions for allowed source access, denied source access, and denied action/egress escalation. | Policy input/output transcripts, bundle digest, decision ids, denial reasons. | Policy decisions are missing, non-deterministic, stale, or silently broaden source/action authority. |
| 5 | Run network default-deny and governed-egress checks for ConnectorHub, Product/API, worker, and tool/agent runtime paths. | Egress transcript showing allowed ConnectorHub destination and denied bypass attempts, gateway/audit refs. | Product/API, worker, tool, or agent runtime can bypass ConnectorHub or reach an undeclared destination. |
| 6 | Execute backup/restore or restore-readiness proof sufficient for this integrated security gate. | Backup id, restore transcript, restored checksum or sampled state proof, recovery caveats. | Restore cannot be verified, backup identity is ambiguous, or recovery evidence is local-only when production-like proof is required. |
| 7 | Verify audit integrity across setup, source policy, delivery/access, policy decision, egress, and recovery events. | Audit event ids, tamper/evidence manifest check, correlation ids, redacted exception list. | Any required audit ref is missing, mutable, uncorrelated, or contains unredacted secret/provider material. |

## Acceptance Evidence Packet

Attach a dated packet with these files or equivalent redacted artifacts:

The file checklist is also available as a non-mutating CLI/report through `cornerstone connector human-gate evidence-packet-file-contract --scenario CS-CH-H04 --json`; that artifact lists required file names, required content categories, and a non-executed scaffold plan only. The plan includes `mkdir -p <h04-acceptance-packet-dir>` plus one `touch <h04-acceptance-packet-dir>/<packet-file>` command per required file, but `packet_file_scaffold_plan_executed_by_report=false` and `packet_file_scaffold_plan_acceptance_sufficient=false` keep it from accepting H04. The optional `evidence-packet-scaffold --write` command creates blank local templates for these files only; its report keeps `template_contents_included_in_report=false`, `packet_file_contents_read_by_scaffold=false`, and `human_evidence_recorded_by_scaffold=false`, so the reviewer still must fill and submit dated evidence separately. The `evidence-packet-validate` command can then check packet presence, non-empty files, and blank-template replacement by hashes only; it keeps `raw_packet_file_contents_included_in_report=false`, `packet_file_contents_persisted_by_validator=false`, and `dependency_unlock_allowed_by_packet_validator=false`. When the packet is structurally complete, `evidence-packet-record-draft --record-output <reviewer-record-draft.json>` can prepare hash-only field refs, but the reviewer must still fill the human-only decision, reviewer, timestamp, and senior-review findings before `validate-record` can unlock anything.

| Artifact | Required contents |
|---|---|
| `environment-topology.md` | Scope identifiers, service versions, deployment or compose/task references, policy bundle version, DB revision, network topology, reviewer, timestamp. |
| `request-context-trace.json` | Trusted RequestContext fields, connector app/source ids, owner/namespace/workspace, evidence refs, audit refs. |
| `postgres-rls-transcript.txt` | Allowed and denied product-path cases, RLS policy refs, redacted row identifiers, zero cross-namespace read/write finding. |
| `opa-policy-transcript.json` | Allowed and denied decisions, bundle digest, input shape, decision ids, denial reasons. |
| `egress-transcript.txt` | ConnectorHub allowed path, Product/API denied path, worker denied path, tool/agent denied path, gateway/audit refs. |
| `backup-restore-evidence.md` | Backup id, restore command or run reference, restored checksum/sample proof, recovery limitations. |
| `audit-integrity-report.json` | Audit event ids, correlation ids, tamper/evidence manifest result, redacted exceptions. |
| `review-decision.md` | ACCEPT or REJECT, reviewer, timestamp, evidence packet path, exceptions, release impact. |

## Redaction And Handling Rules

- Do not paste secrets, provider tokens, private keys, raw provider payloads, private database rows, or credential-bearing URLs into this template.
- Replace private identifiers with stable redacted labels that still allow trace correlation across transcripts.
- Keep raw evidence in the approved evidence location named in the Decision Record; this template should only include paths, hashes, short findings, and redacted excerpts.
- If a proof step needs a production secret, record only the secret reference name and the fact that access was performed through the approved runtime boundary.
- If the validator reports `sensitive_marker_detected`, replace the matching material with a redacted evidence reference or public-safe summary before resubmitting the reviewer record.

## Senior Review Perspectives

| Perspective | H04 acceptance question |
|---|---|
| Product value | Do connected-source release claims stay bounded by tenant, policy, egress, and recovery evidence? |
| Domain architecture | Do Product, Archive, and Connector boundaries hold under real RequestContext and deployment topology? |
| Data contract | Are scenario-specific DB, policy, network, backup, restore, and audit transcripts attached? |
| Reliability and observability | Are generated VS2 reports reviewed as inputs alongside direct report, transcript, and audit integrity evidence? |
| Security and privacy | Are RLS, OPA, egress deny, secret custody, and audit integrity proven in the production-like surface rather than simulated? |
| Testability and migration | Is this gate kept separate from local VS2 proof and ConnectorHub fixture PASS claims? |

## Decision Record

```text
Decision: ACCEPT / REJECT
Reviewer:
Review timestamp:
Environment topology reference:
Evidence location:
Findings or exceptions:
```

## Evidence Checklist

| Required evidence | Human result | Notes |
|---|---|---|
| Production-like topology and trusted RequestContext proof are attached. | PENDING | |
| PostgreSQL/RLS product-path transcripts are attached. | PENDING | |
| OPA policy-client transcripts are attached. | PENDING | |
| Network default-deny and governed-egress transcripts are attached. | PENDING | |
| Backup/restore evidence is attached. | PENDING | |
| Audit integrity and evidence manifest reports are attached. | PENDING | |
| Any remaining VS2 remediation gaps are listed explicitly. | PENDING | |

## Acceptance Statement

Accept only if real integrated evidence closes the tenant, policy, egress, recovery, and audit paths. Reject simulated or local-only substitutes when production-like proof is required.

## Boundary

This review blocks production connected-source and security readiness. Local ConnectorHub fixture PASS and local VS2 topology proof remain separate surfaces.
