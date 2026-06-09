#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

export PATH="$ROOT:$PATH"

fail() {
  printf 'FAIL: %s\n' "$1" >&2
  exit 1
}

cornerstone --help >/dev/null

version_json=$(mktemp)
health_json=$(mktemp)
ready_json=$(mktemp)
list_json=$(mktemp)
coverage_json=$(mktemp)
verify_json=$(mktemp)
fixtures_json=$(mktemp)
artifacts_json=$(mktemp)
security_json=$(mktemp)
search_json=$(mktemp)
understanding_json=$(mktemp)
namespace_json=$(mktemp)
audit_json=$(mktemp)
universal_json=$(mktemp)
claim_json=$(mktemp)
policy_json=$(mktemp)
guardrails_json=$(mktemp)
brief_json=$(mktemp)
mission_action_json=$(mktemp)
detail_json=$(mktemp)
conversation_json=$(mktemp)
product_loop_json=$(mktemp)
memory_truth_json=$(mktemp)
tenant_security_json=$(mktemp)
product_domain_json=$(mktemp)
claim_collaboration_json=$(mktemp)
memory_wiki_json=$(mktemp)
learning_experience_json=$(mktemp)
understanding_ontology_json=$(mktemp)
extension_ecosystem_json=$(mktemp)
trap 'rm -f "$version_json" "$health_json" "$ready_json" "$list_json" "$coverage_json" "$verify_json" "$fixtures_json" "$artifacts_json" "$security_json" "$search_json" "$understanding_json" "$namespace_json" "$audit_json" "$universal_json" "$claim_json" "$policy_json" "$guardrails_json" "$brief_json" "$mission_action_json" "$detail_json" "$conversation_json" "$product_loop_json" "$memory_truth_json" "$tenant_security_json" "$product_domain_json" "$claim_collaboration_json" "$memory_wiki_json" "$learning_experience_json" "$understanding_ontology_json" "$extension_ecosystem_json"' EXIT

cornerstone version --json > "$version_json"
python3 -m json.tool "$version_json" >/dev/null
grep -q '"schema_version": "cs.cli.v0"' "$version_json" || fail "version JSON missing schema version"

cornerstone health --json > "$health_json"
python3 -m json.tool "$health_json" >/dev/null
grep -q '"status": "success"' "$health_json" || fail "health JSON did not succeed"

set +e
cornerstone ready --json > "$ready_json"
ready_code=$?
set -e
[ "$ready_code" -eq 4 ] || fail "ready should exit 4 while product runtime is not ready; got $ready_code"
python3 -m json.tool "$ready_json" >/dev/null
grep -q '"status": "not_ready"' "$ready_json" || fail "ready JSON did not report not_ready"

cornerstone scenario list --set full --json > "$list_json"
python3 -m json.tool "$list_json" >/dev/null
grep -q '"count": 206' "$list_json" || fail "full scenario list did not return 206 rows"

cornerstone scenario coverage --json > "$coverage_json"
python3 -m json.tool "$coverage_json" >/dev/null
grep -q '"ok": true' "$coverage_json" || fail "scenario coverage failed"
python3 scripts/generate_scenario_verification_matrix.py --check
python3 scripts/verify_scenario_matrix.py

cornerstone scenario verify vs0-scaffold --json > "$verify_json"
python3 -m json.tool "$verify_json" >/dev/null
grep -q '"scenario_set": "vs0-scaffold"' "$verify_json" || fail "vs0-scaffold report missing scenario set"
grep -q '"blocking": 0' "$verify_json" || fail "vs0-scaffold report has blocking scenarios"

cornerstone scenario verify vs0-fixtures --corpus fixtures/vs0 --model-provider local_test --json > "$fixtures_json"
python3 -m json.tool "$fixtures_json" >/dev/null
grep -q '"scenario_set": "vs0-fixtures"' "$fixtures_json" || fail "vs0-fixtures report missing scenario set"
grep -q '"blocking": 0' "$fixtures_json" || fail "vs0-fixtures report has blocking scenarios"
grep -q '"product_feature_claims": "NOT_VERIFIED"' "$fixtures_json" || fail "vs0-fixtures must not claim product feature PASS"
grep -q '"external_http_calls": 0' "$fixtures_json" || fail "vs0-fixtures reported external HTTP calls"

cornerstone scenario verify vs0-artifacts --json > "$artifacts_json"
python3 -m json.tool "$artifacts_json" >/dev/null
grep -q '"scenario_set": "vs0-artifacts"' "$artifacts_json" || fail "vs0-artifacts report missing scenario set"
grep -q '"blocking": 0' "$artifacts_json" || fail "vs0-artifacts report has blocking scenarios"
grep -q '"id": "CS-ARCH-001"' "$artifacts_json" || fail "vs0-artifacts missing CS-ARCH-001"
grep -q '"product_feature_claims": "PARTIAL_VS0_ARTIFACTS_ONLY"' "$artifacts_json" || fail "vs0-artifacts overclaimed product feature scope"

cornerstone scenario verify vs0-security --json > "$security_json"
python3 -m json.tool "$security_json" >/dev/null
grep -q '"scenario_set": "vs0-security"' "$security_json" || fail "vs0-security report missing scenario set"
grep -q '"blocking": 0' "$security_json" || fail "vs0-security report has blocking scenarios"
grep -q '"unredacted_secret_occurrences": 0' "$security_json" || fail "vs0-security leaked unredacted secret occurrences"
grep -q '"product_feature_claims": "PARTIAL_VS0_SECURITY_ONLY"' "$security_json" || fail "vs0-security overclaimed product feature scope"

cornerstone scenario verify vs0-search-evidence --json > "$search_json"
python3 -m json.tool "$search_json" >/dev/null
grep -q '"scenario_set": "vs0-search-evidence"' "$search_json" || fail "vs0-search-evidence report missing scenario set"
grep -q '"blocking": 0' "$search_json" || fail "vs0-search-evidence report has blocking scenarios"
grep -q '"pass": 3' "$search_json" || fail "vs0-search-evidence did not pass exactly three scenarios"
grep -q '"id": "CS-ARCH-008"' "$search_json" || fail "vs0-search-evidence missing CS-ARCH-008"
grep -q '"id": "CS-ARCH-009"' "$search_json" || fail "vs0-search-evidence missing CS-ARCH-009"
grep -q '"id": "CS-UND-001"' "$search_json" || fail "vs0-search-evidence missing CS-UND-001"
grep -q '"claim_id": "claim_' "$search_json" || fail "vs0-search-evidence missing claim evidence"
grep -q '"evidence_viewer_id": "viewer_' "$search_json" || fail "vs0-search-evidence missing evidence viewer"
grep -q '"product_feature_claims": "PARTIAL_VS0_SEARCH_EVIDENCE_ONLY"' "$search_json" || fail "vs0-search-evidence overclaimed product feature scope"

cornerstone scenario verify vs0-search-understanding --json > "$understanding_json"
python3 -m json.tool "$understanding_json" >/dev/null
grep -q '"scenario_set": "vs0-search-understanding"' "$understanding_json" || fail "vs0-search-understanding report missing scenario set"
grep -q '"blocking": 0' "$understanding_json" || fail "vs0-search-understanding report has blocking scenarios"
grep -q '"pass": 2' "$understanding_json" || fail "vs0-search-understanding did not pass exactly two scenarios"
grep -q '"id": "CS-UND-002"' "$understanding_json" || fail "vs0-search-understanding missing CS-UND-002"
grep -q '"id": "CS-UND-003"' "$understanding_json" || fail "vs0-search-understanding missing CS-UND-003"
if grep -q '"id": "CS-UND-004"' "$understanding_json"; then
  fail "vs0-search-understanding must not claim CS-UND-004"
fi
grep -q '"type": "semantic_alias"' "$understanding_json" || fail "vs0-search-understanding missing semantic match reason"
grep -q '"cross_workspace_results": 0' "$understanding_json" || fail "vs0-search-understanding reported cross-workspace results"
grep -q '"project_result_count": 1' "$understanding_json" || fail "vs0-search-understanding missing project workspace result"
grep -q '"same_content_scope_collisions": 0' "$understanding_json" || fail "vs0-search-understanding reported same-content scope collision"
grep -q '"product_feature_claims": "PARTIAL_VS0_SEARCH_UNDERSTANDING_ONLY"' "$understanding_json" || fail "vs0-search-understanding overclaimed product feature scope"

cornerstone scenario verify vs0-namespace-isolation --json > "$namespace_json"
python3 -m json.tool "$namespace_json" >/dev/null
grep -q '"scenario_set": "vs0-namespace-isolation"' "$namespace_json" || fail "vs0-namespace-isolation report missing scenario set"
grep -q '"blocking": 0' "$namespace_json" || fail "vs0-namespace-isolation report has blocking scenarios"
grep -q '"pass": 2' "$namespace_json" || fail "vs0-namespace-isolation did not pass exactly two scenarios"
grep -q '"id": "CS-NS-001"' "$namespace_json" || fail "vs0-namespace-isolation missing CS-NS-001"
grep -q '"id": "CS-NS-003"' "$namespace_json" || fail "vs0-namespace-isolation missing CS-NS-003"
if grep -q '"id": "CS-NS-002"' "$namespace_json"; then
  fail "vs0-namespace-isolation must not claim CS-NS-002"
fi
if grep -q '"id": "CS-NS-004"' "$namespace_json"; then
  fail "vs0-namespace-isolation must not claim CS-NS-004"
fi
if grep -q '"id": "CS-SEC-004"' "$namespace_json"; then
  fail "vs0-namespace-isolation must not claim CS-SEC-004"
fi
if grep -q '"id": "CS-REG-006"' "$namespace_json"; then
  fail "vs0-namespace-isolation must not claim CS-REG-006"
fi
grep -q '"ownerless_records": 0' "$namespace_json" || fail "vs0-namespace-isolation reported ownerless records"
grep -q '"cross_namespace_results": 0' "$namespace_json" || fail "vs0-namespace-isolation reported cross-namespace results"
grep -q '"cross_scope_access_allowed": 0' "$namespace_json" || fail "vs0-namespace-isolation allowed cross-scope access"
grep -q '"implicit_promotions": 0' "$namespace_json" || fail "vs0-namespace-isolation reported implicit promotions"
grep -q '"product_feature_claims": "PARTIAL_VS0_NAMESPACE_ISOLATION_ONLY"' "$namespace_json" || fail "vs0-namespace-isolation overclaimed product feature scope"

cornerstone scenario verify vs0-audit-ledger --json > "$audit_json"
python3 -m json.tool "$audit_json" >/dev/null
grep -q '"scenario_set": "vs0-audit-ledger"' "$audit_json" || fail "vs0-audit-ledger report missing scenario set"
grep -q '"blocking": 0' "$audit_json" || fail "vs0-audit-ledger report has blocking scenarios"
grep -q '"pass": 1' "$audit_json" || fail "vs0-audit-ledger did not pass exactly one scenario"
grep -q '"id": "CS-SEC-006"' "$audit_json" || fail "vs0-audit-ledger missing CS-SEC-006"
grep -q '"claim.approved"' "$audit_json" || fail "vs0-audit-ledger missing claim approval audit event"
grep -q '"policy.egress.denied"' "$audit_json" || fail "vs0-audit-ledger missing egress denial audit event"
grep -q '"policy.sandbox_access.denied"' "$audit_json" || fail "vs0-audit-ledger missing sandbox denial audit event"
grep -q '"tamper_detection_exit_code": 5' "$audit_json" || fail "vs0-audit-ledger did not capture tamper failure exit code"
grep -q '"code": "AUDIT_EVENT_HASH_MISMATCH"' "$audit_json" || fail "vs0-audit-ledger did not detect audit event hash mismatch"
grep -q '"missing_required_event_types": 0' "$audit_json" || fail "vs0-audit-ledger missed required event types"
grep -q '"events_without_scope": 0' "$audit_json" || fail "vs0-audit-ledger reported events without scope"
grep -q '"events_without_hashes": 0' "$audit_json" || fail "vs0-audit-ledger reported events without hashes"
grep -q '"events_without_review_details": 0' "$audit_json" || fail "vs0-audit-ledger reported events without review details"
grep -q '"tamper_accepted": 0' "$audit_json" || fail "vs0-audit-ledger accepted tampering"
grep -q '"product_feature_claims": "PARTIAL_VS0_AUDIT_LEDGER_ONLY"' "$audit_json" || fail "vs0-audit-ledger overclaimed product feature scope"

cornerstone scenario verify vs0-universal-core --json > "$universal_json"
python3 -m json.tool "$universal_json" >/dev/null
grep -q '"scenario_set": "vs0-universal-core"' "$universal_json" || fail "vs0-universal-core report missing scenario set"
grep -q '"blocking": 0' "$universal_json" || fail "vs0-universal-core report has blocking scenarios"
grep -q '"pass": 1' "$universal_json" || fail "vs0-universal-core did not pass exactly one scenario"
grep -q '"id": "CS-REG-004"' "$universal_json" || fail "vs0-universal-core missing CS-REG-004"
grep -q '"logistics_terms_found": 0' "$universal_json" || fail "vs0-universal-core found logistics terms"
grep -q '"generic_fixture_failures": 0' "$universal_json" || fail "vs0-universal-core reported generic fixture failure"
grep -q '"product_feature_claims": "PARTIAL_VS0_UNIVERSAL_CORE_ONLY"' "$universal_json" || fail "vs0-universal-core overclaimed product feature scope"

cornerstone scenario verify vs0-claim-evidence --json > "$claim_json"
python3 -m json.tool "$claim_json" >/dev/null
grep -q '"scenario_set": "vs0-claim-evidence"' "$claim_json" || fail "vs0-claim-evidence report missing scenario set"
grep -q '"blocking": 0' "$claim_json" || fail "vs0-claim-evidence report has blocking scenarios"
grep -q '"pass": 2' "$claim_json" || fail "vs0-claim-evidence did not pass exactly two scenarios"
grep -q '"id": "CS-CLAIM-006"' "$claim_json" || fail "vs0-claim-evidence missing CS-CLAIM-006"
grep -q '"id": "CS-CLAIM-007"' "$claim_json" || fail "vs0-claim-evidence missing CS-CLAIM-007"
grep -q '"CS_CLAIM_EVIDENCE_REQUIRED"' "$claim_json" || fail "vs0-claim-evidence missing evidence-required denial"
grep -q '"approved_claim_status": "approved"' "$claim_json" || fail "vs0-claim-evidence missing approved claim status"
grep -q '"unsupported_approval_allowed": 0' "$claim_json" || fail "vs0-claim-evidence allowed unsupported approval"
grep -q '"evidence_claim_approval_blocked": 0' "$claim_json" || fail "vs0-claim-evidence blocked evidence-backed approval"
grep -q '"autonomous_action_allowed_from_claim": 0' "$claim_json" || fail "vs0-claim-evidence allowed autonomous action from claim"
grep -q '"product_feature_claims": "PARTIAL_VS0_CLAIM_EVIDENCE_ONLY"' "$claim_json" || fail "vs0-claim-evidence overclaimed product feature scope"

cornerstone scenario verify vs0-security-policy --json > "$policy_json"
python3 -m json.tool "$policy_json" >/dev/null
grep -q '"scenario_set": "vs0-security-policy"' "$policy_json" || fail "vs0-security-policy report missing scenario set"
grep -q '"blocking": 0' "$policy_json" || fail "vs0-security-policy report has blocking scenarios"
grep -q '"pass": 2' "$policy_json" || fail "vs0-security-policy did not pass exactly two scenarios"
grep -q '"id": "CS-SEC-002"' "$policy_json" || fail "vs0-security-policy missing CS-SEC-002"
grep -q '"id": "CS-SEC-003"' "$policy_json" || fail "vs0-security-policy missing CS-SEC-003"
grep -q '"CS_EGRESS_DENIED"' "$policy_json" || fail "vs0-security-policy missing egress denial"
grep -q '"CS_SANDBOX_ACCESS_DENIED"' "$policy_json" || fail "vs0-security-policy missing sandbox denial"
grep -q '"external_http_calls": 0' "$policy_json" || fail "vs0-security-policy reported external HTTP calls"
grep -q '"egress_allowed": 0' "$policy_json" || fail "vs0-security-policy allowed egress"
grep -q '"host_operations_executed": 0' "$policy_json" || fail "vs0-security-policy executed host operations"
grep -q '"shell_commands_executed": 0' "$policy_json" || fail "vs0-security-policy executed shell commands"
grep -q '"filesystem_reads": 0' "$policy_json" || fail "vs0-security-policy read filesystem"
grep -q '"environment_reads": 0' "$policy_json" || fail "vs0-security-policy read environment"
grep -q '"sandbox_access_allowed": 0' "$policy_json" || fail "vs0-security-policy allowed sandbox access"
grep -q '"product_feature_claims": "PARTIAL_VS0_SECURITY_POLICY_ONLY"' "$policy_json" || fail "vs0-security-policy overclaimed product feature scope"

cornerstone scenario verify vs0-regression-guardrails --json > "$guardrails_json"
python3 -m json.tool "$guardrails_json" >/dev/null
grep -q '"scenario_set": "vs0-regression-guardrails"' "$guardrails_json" || fail "vs0-regression-guardrails report missing scenario set"
grep -q '"blocking": 0' "$guardrails_json" || fail "vs0-regression-guardrails report has blocking scenarios"
grep -q '"pass": 3' "$guardrails_json" || fail "vs0-regression-guardrails did not pass exactly three scenarios"
grep -q '"id": "CS-REG-016"' "$guardrails_json" || fail "vs0-regression-guardrails missing CS-REG-016"
grep -q '"id": "CS-REG-017"' "$guardrails_json" || fail "vs0-regression-guardrails missing CS-REG-017"
grep -q '"id": "CS-REG-018"' "$guardrails_json" || fail "vs0-regression-guardrails missing CS-REG-018"
grep -q '"evidence_guardrail_failed": 0' "$guardrails_json" || fail "vs0-regression-guardrails failed evidence guardrail"
grep -q '"audit_guardrail_failed": 0' "$guardrails_json" || fail "vs0-regression-guardrails failed audit guardrail"
grep -q '"security_guardrail_failed": 0' "$guardrails_json" || fail "vs0-regression-guardrails failed security guardrail"
grep -q '"product_feature_claims": "PARTIAL_VS0_REGRESSION_GUARDRAILS_ONLY"' "$guardrails_json" || fail "vs0-regression-guardrails overclaimed product feature scope"

cornerstone scenario verify vs0-briefing --json > "$brief_json"
python3 -m json.tool "$brief_json" >/dev/null
grep -q '"scenario_set": "vs0-briefing"' "$brief_json" || fail "vs0-briefing report missing scenario set"
grep -q '"blocking": 0' "$brief_json" || fail "vs0-briefing report has blocking scenarios"
grep -q '"pass": 4' "$brief_json" || fail "vs0-briefing did not pass exactly four scenarios"
grep -q '"id": "CS-PROD-004"' "$brief_json" || fail "vs0-briefing missing CS-PROD-004"
grep -q '"id": "CS-UND-005"' "$brief_json" || fail "vs0-briefing missing CS-UND-005"
grep -q '"id": "CS-CLAIM-002"' "$brief_json" || fail "vs0-briefing missing CS-CLAIM-002"
grep -q '"id": "CS-SEC-001"' "$brief_json" || fail "vs0-briefing missing CS-SEC-001"
grep -q '"brief_status": "evidence_backed"' "$brief_json" || fail "vs0-briefing missing evidence-backed brief"
grep -q '"brief_without_evidence": 0' "$brief_json" || fail "vs0-briefing created brief without evidence"
grep -q '"required_connector_setup": 0' "$brief_json" || fail "vs0-briefing required connector setup"
grep -q '"required_model_provider_setup": 0' "$brief_json" || fail "vs0-briefing required model provider setup"
grep -q '"required_ontology_setup": 0' "$brief_json" || fail "vs0-briefing required ontology setup"
grep -q '"missing_uncertainty": 0' "$brief_json" || fail "vs0-briefing missed uncertainty"
grep -q '"missing_next_steps": 0' "$brief_json" || fail "vs0-briefing missed next steps"
grep -q '"product_feature_claims": "PARTIAL_VS0_BRIEFING_ONLY"' "$brief_json" || fail "vs0-briefing overclaimed product feature scope"

cornerstone scenario verify vs0-mission-action --json > "$mission_action_json"
python3 -m json.tool "$mission_action_json" >/dev/null
grep -q '"scenario_set": "vs0-mission-action"' "$mission_action_json" || fail "vs0-mission-action report missing scenario set"
grep -q '"blocking": 0' "$mission_action_json" || fail "vs0-mission-action report has blocking scenarios"
grep -q '"pass": 16' "$mission_action_json" || fail "vs0-mission-action did not pass exactly sixteen scenarios"
grep -q '"id": "CS-CLAIM-010"' "$mission_action_json" || fail "vs0-mission-action missing CS-CLAIM-010"
grep -q '"id": "CS-AUTO-001"' "$mission_action_json" || fail "vs0-mission-action missing CS-AUTO-001"
grep -q '"id": "CS-AUTO-003"' "$mission_action_json" || fail "vs0-mission-action missing CS-AUTO-003"
grep -q '"id": "CS-AUTO-004"' "$mission_action_json" || fail "vs0-mission-action missing CS-AUTO-004"
grep -q '"id": "CS-AUTO-005"' "$mission_action_json" || fail "vs0-mission-action missing CS-AUTO-005"
grep -q '"id": "CS-AUTO-006"' "$mission_action_json" || fail "vs0-mission-action missing CS-AUTO-006"
grep -q '"id": "CS-AUTO-007"' "$mission_action_json" || fail "vs0-mission-action missing CS-AUTO-007"
grep -q '"id": "CS-AUTO-008"' "$mission_action_json" || fail "vs0-mission-action missing CS-AUTO-008"
grep -q '"id": "CS-AUTO-009"' "$mission_action_json" || fail "vs0-mission-action missing CS-AUTO-009"
grep -q '"id": "CS-AUTO-010"' "$mission_action_json" || fail "vs0-mission-action missing CS-AUTO-010"
grep -q '"id": "CS-AUTO-011"' "$mission_action_json" || fail "vs0-mission-action missing CS-AUTO-011"
grep -q '"id": "CS-REG-002"' "$mission_action_json" || fail "vs0-mission-action missing CS-REG-002"
grep -q '"id": "CS-REG-003"' "$mission_action_json" || fail "vs0-mission-action missing CS-REG-003"
grep -q '"id": "CS-REG-011"' "$mission_action_json" || fail "vs0-mission-action missing CS-REG-011"
grep -q '"id": "CS-REG-012"' "$mission_action_json" || fail "vs0-mission-action missing CS-REG-012"
grep -q '"id": "CS-AUTO-020"' "$mission_action_json" || fail "vs0-mission-action missing CS-AUTO-020"
grep -q '"workflow_action_path_required"' "$mission_action_json" || fail "vs0-mission-action missing direct-write denial"
grep -q '"low_risk_autopilot_allowed"' "$mission_action_json" || fail "vs0-mission-action missing low-risk allow policy"
grep -q '"high_risk_action_requires_approval"' "$mission_action_json" || fail "vs0-mission-action missing high-risk approval policy"
grep -q '"mission_contract_action_scope"' "$mission_action_json" || fail "vs0-mission-action missing mission-contract policy"
grep -q '"workspace_mode_no_autonomous_execution"' "$mission_action_json" || fail "vs0-mission-action missing manual mode denial"
grep -q '"workspace_mode_locked"' "$mission_action_json" || fail "vs0-mission-action missing locked mode denial"
grep -q '"real_external_http_calls": 0' "$mission_action_json" || fail "vs0-mission-action reported real external HTTP calls"
grep -q '"high_risk_executed_without_approval": 0' "$mission_action_json" || fail "vs0-mission-action executed high-risk action without approval"
grep -q '"out_of_contract_action_executed": 0' "$mission_action_json" || fail "vs0-mission-action executed out-of-contract action"
grep -q '"manual_mode_autonomous_execution": 0' "$mission_action_json" || fail "vs0-mission-action executed while manual"
grep -q '"locked_mode_autonomous_execution": 0' "$mission_action_json" || fail "vs0-mission-action executed while locked"
grep -q '"cross_scope_action_executed": 0' "$mission_action_json" || fail "vs0-mission-action executed cross-scope action"
grep -q '"direct_provider_write_allowed": 0' "$mission_action_json" || fail "vs0-mission-action allowed direct provider write"
grep -q '"connector_credentials_exposed": 0' "$mission_action_json" || fail "vs0-mission-action exposed connector credentials"
grep -q '"product_feature_claims": "PARTIAL_VS0_MISSION_ACTION_ONLY"' "$mission_action_json" || fail "vs0-mission-action overclaimed product feature scope"

cornerstone scenario verify vs0-detail-surfaces --json > "$detail_json"
python3 -m json.tool "$detail_json" >/dev/null
grep -q '"scenario_set": "vs0-detail-surfaces"' "$detail_json" || fail "vs0-detail-surfaces report missing scenario set"
grep -q '"blocking": 0' "$detail_json" || fail "vs0-detail-surfaces report has blocking scenarios"
grep -q '"pass": 5' "$detail_json" || fail "vs0-detail-surfaces did not pass exactly five scenarios"
grep -q '"id": "CS-UND-004"' "$detail_json" || fail "vs0-detail-surfaces missing CS-UND-004"
grep -q '"id": "CS-CLAIM-005"' "$detail_json" || fail "vs0-detail-surfaces missing CS-CLAIM-005"
grep -q '"id": "CS-CLAIM-008"' "$detail_json" || fail "vs0-detail-surfaces missing CS-CLAIM-008"
grep -q '"id": "CS-NS-002"' "$detail_json" || fail "vs0-detail-surfaces missing CS-NS-002"
grep -q '"id": "CS-SEC-005"' "$detail_json" || fail "vs0-detail-surfaces missing CS-SEC-005"
grep -q '"artifact_detail_missing_related_claims": 0' "$detail_json" || fail "vs0-detail-surfaces missed related claim detail"
grep -q '"artifact_detail_missing_related_missions": 0' "$detail_json" || fail "vs0-detail-surfaces missed related mission detail"
grep -q '"trust_ladder_missing_states": 0' "$detail_json" || fail "vs0-detail-surfaces missed trust ladder states"
grep -q '"evidence_viewer_missing_sources": 0' "$detail_json" || fail "vs0-detail-surfaces missed evidence viewer sources"
grep -q '"policy_denials_missing_resolution_path": 0' "$detail_json" || fail "vs0-detail-surfaces missed denial resolution paths"
grep -q '"policy_denials_without_audit": 0' "$detail_json" || fail "vs0-detail-surfaces missed denial audit refs"
grep -q '"workspace_boundary_implicit_cross_namespace_context": 0' "$detail_json" || fail "vs0-detail-surfaces allowed implicit cross-namespace context"
grep -q '"CS_CLAIM_EVIDENCE_REQUIRED"' "$detail_json" || fail "vs0-detail-surfaces missing claim evidence denial"
grep -q '"CS_ACTION_POLICY_DENIED"' "$detail_json" || fail "vs0-detail-surfaces missing action policy denial"
grep -q '"product_feature_claims": "PARTIAL_VS0_DETAIL_SURFACES_ONLY"' "$detail_json" || fail "vs0-detail-surfaces overclaimed product feature scope"

cornerstone scenario verify vs0-conversation-onboarding --json > "$conversation_json"
python3 -m json.tool "$conversation_json" >/dev/null
grep -q '"scenario_set": "vs0-conversation-onboarding"' "$conversation_json" || fail "vs0-conversation-onboarding report missing scenario set"
grep -q '"blocking": 0' "$conversation_json" || fail "vs0-conversation-onboarding report has blocking scenarios"
grep -q '"pass": 5' "$conversation_json" || fail "vs0-conversation-onboarding did not pass exactly five scenarios"
grep -q '"id": "CS-PROD-005"' "$conversation_json" || fail "vs0-conversation-onboarding missing CS-PROD-005"
grep -q '"id": "CS-CLAIM-001"' "$conversation_json" || fail "vs0-conversation-onboarding missing CS-CLAIM-001"
grep -q '"id": "CS-CLAIM-003"' "$conversation_json" || fail "vs0-conversation-onboarding missing CS-CLAIM-003"
grep -q '"id": "CS-CLAIM-004"' "$conversation_json" || fail "vs0-conversation-onboarding missing CS-CLAIM-004"
grep -q '"id": "CS-CLAIM-009"' "$conversation_json" || fail "vs0-conversation-onboarding missing CS-CLAIM-009"
grep -q '"source_artifact_source_type": "conversation_turn"' "$conversation_json" || fail "vs0-conversation-onboarding missing conversation artifact source"
grep -q '"promoted_claim_trust_state": "evidence_backed"' "$conversation_json" || fail "vs0-conversation-onboarding missing evidence-backed promoted claim"
grep -q '"created_from": "conversation.promote"' "$conversation_json" || fail "vs0-conversation-onboarding missing promotion provenance"
grep -q '"unsupported_answer_label": "insufficient_evidence"' "$conversation_json" || fail "vs0-conversation-onboarding missing insufficient-evidence label"
grep -q '"unsupported_answer_presented_as_fact": false' "$conversation_json" || fail "vs0-conversation-onboarding presented unsupported answer as fact"
grep -q '"unsupported_answer_supporting_result_count": 0' "$conversation_json" || fail "vs0-conversation-onboarding counted unsupported evidence as supporting"
grep -q '"pre_modeling_required": 0' "$conversation_json" || fail "vs0-conversation-onboarding required pre-modeling"
grep -q '"required_connector_setup": 0' "$conversation_json" || fail "vs0-conversation-onboarding required connector setup"
grep -q '"required_model_provider_setup": 0' "$conversation_json" || fail "vs0-conversation-onboarding required model provider setup"
grep -q '"required_ontology_setup": 0' "$conversation_json" || fail "vs0-conversation-onboarding required ontology setup"
grep -q '"forced_conversion": 0' "$conversation_json" || fail "vs0-conversation-onboarding forced conversion"
grep -q '"promoted_objects_without_scope": 0' "$conversation_json" || fail "vs0-conversation-onboarding promoted object without scope"
grep -q '"promoted_objects_without_evidence": 0' "$conversation_json" || fail "vs0-conversation-onboarding promoted object without evidence"
grep -q '"unsupported_assertions_presented_as_fact": 0' "$conversation_json" || fail "vs0-conversation-onboarding presented unsupported assertions as fact"
grep -q '"real_external_http_calls": 0' "$conversation_json" || fail "vs0-conversation-onboarding reported real external HTTP calls"
grep -q '"product_feature_claims": "PARTIAL_VS0_CONVERSATION_ONBOARDING_ONLY"' "$conversation_json" || fail "vs0-conversation-onboarding overclaimed product feature scope"

cornerstone scenario verify vs0-product-loop-identity --json > "$product_loop_json"
python3 -m json.tool "$product_loop_json" >/dev/null
grep -q '"scenario_set": "vs0-product-loop-identity"' "$product_loop_json" || fail "vs0-product-loop-identity report missing scenario set"
grep -q '"blocking": 0' "$product_loop_json" || fail "vs0-product-loop-identity report has blocking scenarios"
grep -q '"pass": 2' "$product_loop_json" || fail "vs0-product-loop-identity did not pass exactly two scenarios"
grep -q '"id": "CS-PROD-002"' "$product_loop_json" || fail "vs0-product-loop-identity missing CS-PROD-002"
grep -q '"id": "CS-REG-001"' "$product_loop_json" || fail "vs0-product-loop-identity missing CS-REG-001"
grep -q '"memory_status": "owner_approved"' "$product_loop_json" || fail "vs0-product-loop-identity missing owner-approved memory"
grep -q '"memory_truth_foundation": "archive_evidence"' "$product_loop_json" || fail "vs0-product-loop-identity missing archive evidence truth foundation"
grep -q '"learning_status": "recorded"' "$product_loop_json" || fail "vs0-product-loop-identity missing learning record"
grep -q '"learning_changes_user_or_org_truth": false' "$product_loop_json" || fail "vs0-product-loop-identity learning changed truth"
grep -q '"action_policy": "low_risk_autopilot_allowed"' "$product_loop_json" || fail "vs0-product-loop-identity missing governed action policy"
grep -q '"action_result_status": "success"' "$product_loop_json" || fail "vs0-product-loop-identity missing successful action"
grep -q '"missing_product_loop_surfaces": 0' "$product_loop_json" || fail "vs0-product-loop-identity missing product loop surfaces"
grep -q '"chatbot_only": 0' "$product_loop_json" || fail "vs0-product-loop-identity regressed to chatbot only"
grep -q '"file_search_only": 0' "$product_loop_json" || fail "vs0-product-loop-identity regressed to file search only"
grep -q '"connector_framework_only": 0' "$product_loop_json" || fail "vs0-product-loop-identity regressed to connector framework only"
grep -q '"automation_script_runner_only": 0' "$product_loop_json" || fail "vs0-product-loop-identity regressed to automation script runner only"
grep -q '"memory_without_evidence": 0' "$product_loop_json" || fail "vs0-product-loop-identity created memory without evidence"
grep -q '"learning_without_action_result": 0' "$product_loop_json" || fail "vs0-product-loop-identity recorded learning without action result"
grep -q '"real_external_http_calls": 0' "$product_loop_json" || fail "vs0-product-loop-identity reported real external HTTP calls"
grep -q '"product_feature_claims": "PARTIAL_VS0_PRODUCT_LOOP_IDENTITY_ONLY"' "$product_loop_json" || fail "vs0-product-loop-identity overclaimed product feature scope"

cornerstone scenario verify vs0-memory-truth-boundary --json > "$memory_truth_json"
python3 -m json.tool "$memory_truth_json" >/dev/null
grep -q '"scenario_set": "vs0-memory-truth-boundary"' "$memory_truth_json" || fail "vs0-memory-truth-boundary report missing scenario set"
grep -q '"blocking": 0' "$memory_truth_json" || fail "vs0-memory-truth-boundary report has blocking scenarios"
grep -q '"pass": 1' "$memory_truth_json" || fail "vs0-memory-truth-boundary did not pass exactly one scenario"
grep -q '"id": "CS-REG-005"' "$memory_truth_json" || fail "vs0-memory-truth-boundary missing CS-REG-005"
grep -q '"owner_memory_status": "owner_approved"' "$memory_truth_json" || fail "vs0-memory-truth-boundary missing owner-approved memory"
grep -q '"owner_memory_truth_foundation": "archive_evidence"' "$memory_truth_json" || fail "vs0-memory-truth-boundary missing archive evidence foundation"
grep -q '"raw_memory_status": "raw_agent_memory"' "$memory_truth_json" || fail "vs0-memory-truth-boundary missing raw agent memory"
grep -q '"raw_memory_canonical": false' "$memory_truth_json" || fail "vs0-memory-truth-boundary made raw memory canonical"
grep -q '"conflict_selected_truth_foundation": "archive_evidence"' "$memory_truth_json" || fail "vs0-memory-truth-boundary did not choose archive evidence"
grep -q '"conflict_raw_memory_used_as_truth": false' "$memory_truth_json" || fail "vs0-memory-truth-boundary used raw memory as truth"
grep -q '"conflict_answer_based_on": "archive_evidence"' "$memory_truth_json" || fail "vs0-memory-truth-boundary answer was not evidence based"
grep -q '"owner_memory_without_evidence": 0' "$memory_truth_json" || fail "vs0-memory-truth-boundary created owner memory without evidence"
grep -q '"raw_agent_memory_canonical": 0' "$memory_truth_json" || fail "vs0-memory-truth-boundary reported canonical raw memory"
grep -q '"raw_agent_memory_owner_approved": 0' "$memory_truth_json" || fail "vs0-memory-truth-boundary owner-approved raw memory"
grep -q '"conflict_selected_raw_memory": 0' "$memory_truth_json" || fail "vs0-memory-truth-boundary selected raw memory"
grep -q '"conflict_truth_foundation_not_archive_evidence": 0' "$memory_truth_json" || fail "vs0-memory-truth-boundary missed archive evidence truth foundation"
grep -q '"conflict_without_audit": 0' "$memory_truth_json" || fail "vs0-memory-truth-boundary missed audit"
grep -q '"real_external_http_calls": 0' "$memory_truth_json" || fail "vs0-memory-truth-boundary reported real external HTTP calls"
grep -q '"product_feature_claims": "PARTIAL_VS0_MEMORY_TRUTH_BOUNDARY_ONLY"' "$memory_truth_json" || fail "vs0-memory-truth-boundary overclaimed product feature scope"

cornerstone scenario verify vs0-tenant-security-boundary --json > "$tenant_security_json"
python3 -m json.tool "$tenant_security_json" >/dev/null
grep -q '"scenario_set": "vs0-tenant-security-boundary"' "$tenant_security_json" || fail "vs0-tenant-security-boundary report missing scenario set"
grep -q '"blocking": 0' "$tenant_security_json" || fail "vs0-tenant-security-boundary report has blocking scenarios"
grep -q '"pass": 3' "$tenant_security_json" || fail "vs0-tenant-security-boundary did not pass exactly three scenarios"
grep -q '"id": "CS-NS-004"' "$tenant_security_json" || fail "vs0-tenant-security-boundary missing CS-NS-004"
grep -q '"id": "CS-SEC-004"' "$tenant_security_json" || fail "vs0-tenant-security-boundary missing CS-SEC-004"
grep -q '"id": "CS-REG-006"' "$tenant_security_json" || fail "vs0-tenant-security-boundary missing CS-REG-006"
grep -q '"promotion_mode": "copy_with_provenance"' "$tenant_security_json" || fail "vs0-tenant-security-boundary missing copy-with-provenance promotion"
grep -q '"promotion_policy": "local_rbac_abac_matrix"' "$tenant_security_json" || fail "vs0-tenant-security-boundary missing promotion policy decision"
grep -q '"pre_promotion_answer_status": "insufficient_evidence"' "$tenant_security_json" || fail "vs0-tenant-security-boundary used memory before promotion"
grep -q '"direct_cross_scope_read_exit_code": 6' "$tenant_security_json" || fail "vs0-tenant-security-boundary did not deny direct cross-scope read"
grep -q '"post_promotion_answer_status": "answered"' "$tenant_security_json" || fail "vs0-tenant-security-boundary did not answer after promotion"
grep -q '"post_promotion_used_promoted_memory": true' "$tenant_security_json" || fail "vs0-tenant-security-boundary did not use promoted memory"
grep -q '"access_matrix_case_count": 7' "$tenant_security_json" || fail "vs0-tenant-security-boundary missing access matrix cases"
grep -q '"access_allow_count": 3' "$tenant_security_json" || fail "vs0-tenant-security-boundary missing allowed access cases"
grep -q '"access_deny_count": 4' "$tenant_security_json" || fail "vs0-tenant-security-boundary missing denied access cases"
grep -q '"pre_promotion_personal_memory_used": 0' "$tenant_security_json" || fail "vs0-tenant-security-boundary used personal memory before promotion"
grep -q '"direct_cross_scope_memory_read_allowed": 0' "$tenant_security_json" || fail "vs0-tenant-security-boundary allowed direct cross-scope memory read"
grep -q '"post_promotion_used_source_memory_directly": 0' "$tenant_security_json" || fail "vs0-tenant-security-boundary used source memory directly after promotion"
grep -q '"unauthorized_access_allowed": 0' "$tenant_security_json" || fail "vs0-tenant-security-boundary allowed unauthorized access"
grep -q '"policy_decisions_without_audit": 0' "$tenant_security_json" || fail "vs0-tenant-security-boundary missed policy audit refs"
grep -q '"promotion_without_provenance": 0' "$tenant_security_json" || fail "vs0-tenant-security-boundary missed promotion provenance"
grep -q '"promotion_without_evidence": 0' "$tenant_security_json" || fail "vs0-tenant-security-boundary missed promotion evidence"
grep -q '"real_external_http_calls": 0' "$tenant_security_json" || fail "vs0-tenant-security-boundary reported real external HTTP calls"
grep -q '"secret_reads": 0' "$tenant_security_json" || fail "vs0-tenant-security-boundary read secrets"
grep -q '"product_feature_claims": "PARTIAL_VS0_TENANT_SECURITY_BOUNDARY_ONLY"' "$tenant_security_json" || fail "vs0-tenant-security-boundary overclaimed product feature scope"

cornerstone scenario verify full-claim-collaboration --json > "$claim_collaboration_json"
python3 -m json.tool "$claim_collaboration_json" >/dev/null
grep -q '"scenario_set": "full-claim-collaboration"' "$claim_collaboration_json" || fail "full-claim-collaboration report missing scenario set"
grep -q '"blocking": 0' "$claim_collaboration_json" || fail "full-claim-collaboration report has blocking scenarios"
grep -q '"pass": 4' "$claim_collaboration_json" || fail "full-claim-collaboration did not pass exactly four scenarios"
grep -q '"id": "CS-CLAIM-011"' "$claim_collaboration_json" || fail "full-claim-collaboration missing CS-CLAIM-011"
grep -q '"id": "CS-CLAIM-012"' "$claim_collaboration_json" || fail "full-claim-collaboration missing CS-CLAIM-012"
grep -q '"id": "CS-CLAIM-013"' "$claim_collaboration_json" || fail "full-claim-collaboration missing CS-CLAIM-013"
grep -q '"id": "CS-CLAIM-014"' "$claim_collaboration_json" || fail "full-claim-collaboration missing CS-CLAIM-014"
grep -q '"claim_trust_state": "approved"' "$claim_collaboration_json" || fail "full-claim-collaboration missing approved claim"
grep -q '"capsule_trust_state": "approved"' "$claim_collaboration_json" || fail "full-claim-collaboration missing approved capsule trust state"
grep -q '"capsule_freshness_status": "current"' "$claim_collaboration_json" || fail "full-claim-collaboration missing capsule freshness"
grep -q '"decision_action_count": 1' "$claim_collaboration_json" || fail "full-claim-collaboration missing decision action"
grep -q '"decision_learning_history_count": 1' "$claim_collaboration_json" || fail "full-claim-collaboration missing decision learning history"
grep -q '"correction_source_type": "evidence_bundle"' "$claim_collaboration_json" || fail "full-claim-collaboration missing evidence-aware correction"
grep -q '"correction_provenance_preserved": true' "$claim_collaboration_json" || fail "full-claim-collaboration did not preserve provenance"
grep -q '"share_trust_state": "approved"' "$claim_collaboration_json" || fail "full-claim-collaboration missing approved share trust state"
grep -q '"capsule_without_evidence": 0' "$claim_collaboration_json" || fail "full-claim-collaboration reported capsule without evidence"
grep -q '"capsule_without_namespace": 0' "$claim_collaboration_json" || fail "full-claim-collaboration reported capsule without namespace"
grep -q '"decision_card_missing_required_fields": 0' "$claim_collaboration_json" || fail "full-claim-collaboration missing decision fields"
grep -q '"correction_silent_overwrite": 0' "$claim_collaboration_json" || fail "full-claim-collaboration silently overwrote correction target"
grep -q '"correction_without_learning_signal": 0' "$claim_collaboration_json" || fail "full-claim-collaboration missed correction learning signal"
grep -q '"share_hidden_trust_state": 0' "$claim_collaboration_json" || fail "full-claim-collaboration hid share trust state"
grep -q '"share_hidden_evidence": 0' "$claim_collaboration_json" || fail "full-claim-collaboration hid share evidence"
grep -q '"share_hidden_owner_or_scope": 0' "$claim_collaboration_json" || fail "full-claim-collaboration hid owner or scope"
grep -q '"real_external_http_calls": 0' "$claim_collaboration_json" || fail "full-claim-collaboration reported real external HTTP calls"
grep -q '"secret_reads": 0' "$claim_collaboration_json" || fail "full-claim-collaboration read secrets"
grep -q '"product_feature_claims": "PARTIAL_FULL_CLAIM_COLLABORATION_ONLY"' "$claim_collaboration_json" || fail "full-claim-collaboration overclaimed product feature scope"

cornerstone scenario verify full-memory-wiki --json > "$memory_wiki_json"
python3 -m json.tool "$memory_wiki_json" >/dev/null
grep -q '"scenario_set": "full-memory-wiki"' "$memory_wiki_json" || fail "full-memory-wiki report missing scenario set"
grep -q '"blocking": 0' "$memory_wiki_json" || fail "full-memory-wiki report has blocking scenarios"
grep -q '"pass": 18' "$memory_wiki_json" || fail "full-memory-wiki did not pass exactly eighteen scenarios"
for scenario_id in CS-MEM-001 CS-MEM-002 CS-MEM-003 CS-MEM-004 CS-MEM-005 CS-MEM-006 CS-MEM-007 CS-MEM-008 CS-MEM-009 CS-MEM-010 CS-MEM-011 CS-MEM-012 CS-MEM-013 CS-MEM-014 CS-MEM-015 CS-MEM-016 CS-MEM-017 CS-MEM-018; do
  grep -q "\"id\": \"$scenario_id\"" "$memory_wiki_json" || fail "full-memory-wiki missing $scenario_id"
done
grep -q '"quarantine_status": "quarantined"' "$memory_wiki_json" || fail "full-memory-wiki missing quarantine evidence"
grep -q '"status": "needs_review"' "$memory_wiki_json" || fail "full-memory-wiki missing freshness warning"
grep -q '"conflict_selected_truth_foundation": "archive_evidence"' "$memory_wiki_json" || fail "full-memory-wiki did not prefer archive evidence"
grep -q '"memory_without_evidence": 0' "$memory_wiki_json" || fail "full-memory-wiki reported memory without evidence"
grep -q '"raw_memory_used_as_truth": 0' "$memory_wiki_json" || fail "full-memory-wiki used raw memory as truth"
grep -q '"hidden_profile_created": 0' "$memory_wiki_json" || fail "full-memory-wiki created hidden profile"
grep -q '"temporary_session_memory_created": 0' "$memory_wiki_json" || fail "full-memory-wiki persisted temporary memory"
grep -q '"correction_silent_overwrite": 0' "$memory_wiki_json" || fail "full-memory-wiki silently overwrote correction"
grep -q '"forgotten_memory_used": 0' "$memory_wiki_json" || fail "full-memory-wiki used forgotten memory"
grep -q '"stale_memory_used_without_warning": 0' "$memory_wiki_json" || fail "full-memory-wiki used stale memory without warning"
grep -q '"untrusted_memory_promoted": 0' "$memory_wiki_json" || fail "full-memory-wiki promoted untrusted memory"
grep -q '"product_learning_changed_user_org_truth": 0' "$memory_wiki_json" || fail "full-memory-wiki let product learning change user/org truth"
grep -q '"cross_namespace_adaptation": 0' "$memory_wiki_json" || fail "full-memory-wiki adapted across namespaces"
grep -q '"export_missing_sources": 0' "$memory_wiki_json" || fail "full-memory-wiki export missed sources"
grep -q '"real_external_http_calls": 0' "$memory_wiki_json" || fail "full-memory-wiki reported real external HTTP calls"
grep -q '"secret_reads": 0' "$memory_wiki_json" || fail "full-memory-wiki read secrets"
grep -q '"product_feature_claims": "PARTIAL_FULL_MEMORY_WIKI_ONLY"' "$memory_wiki_json" || fail "full-memory-wiki overclaimed product feature scope"

cornerstone scenario verify full-learning-experience --json > "$learning_experience_json"
python3 -m json.tool "$learning_experience_json" >/dev/null
grep -q '"scenario_set": "full-learning-experience"' "$learning_experience_json" || fail "full-learning-experience report missing scenario set"
grep -q '"blocking": 0' "$learning_experience_json" || fail "full-learning-experience report has blocking scenarios"
grep -q '"pass": 18' "$learning_experience_json" || fail "full-learning-experience did not pass exactly eighteen scenarios"
for scenario_id in CS-LEARN-001 CS-LEARN-002 CS-LEARN-003 CS-LEARN-004 CS-LEARN-005 CS-LEARN-006 CS-LEARN-007 CS-LEARN-008 CS-LEARN-009 CS-LEARN-010 CS-LEARN-011 CS-LEARN-012 CS-LEARN-013 CS-LEARN-014 CS-LEARN-015 CS-LEARN-016 CS-LEARN-017 CS-LEARN-018; do
  grep -q "\"id\": \"$scenario_id\"" "$learning_experience_json" || fail "full-learning-experience missing $scenario_id"
done
grep -q '"experience_search_result_count": 1' "$learning_experience_json" || fail "full-learning-experience missing scoped experience search result"
grep -q '"recommendation_count": 1' "$learning_experience_json" || fail "full-learning-experience missing prior experience recommendation"
grep -q '"personal_org_search_result_count": 0' "$learning_experience_json" || fail "full-learning-experience leaked organization experience to personal scope"
grep -q '"org_search_result_count": 1' "$learning_experience_json" || fail "full-learning-experience missing organization scoped experience result"
grep -q '"trajectory_without_owner": 0' "$learning_experience_json" || fail "full-learning-experience recorded ownerless trajectory"
grep -q '"trajectory_without_audit": 0' "$learning_experience_json" || fail "full-learning-experience missed trajectory audit"
grep -q '"failed_trajectory_hidden": 0' "$learning_experience_json" || fail "full-learning-experience hid failed trajectory"
grep -q '"experience_cross_namespace_results": 0' "$learning_experience_json" || fail "full-learning-experience leaked cross-namespace experience"
grep -q '"lesson_auto_global_rule": 0' "$learning_experience_json" || fail "full-learning-experience auto-promoted lesson globally"
grep -q '"promotion_ladder_skipped": 0' "$learning_experience_json" || fail "full-learning-experience skipped promotion ladder"
grep -q '"broader_reuse_without_approval": 0' "$learning_experience_json" || fail "full-learning-experience allowed broader reuse without approval"
grep -q '"behavior_signal_overrode_outcome": 0' "$learning_experience_json" || fail "full-learning-experience let behavior signal override outcome"
grep -q '"model_eval_overrode_outcome": 0' "$learning_experience_json" || fail "full-learning-experience let model eval override outcome"
grep -q '"product_global_mutation": 0' "$learning_experience_json" || fail "full-learning-experience mutated product defaults"
grep -q '"local_adaptation_cross_namespace": 0' "$learning_experience_json" || fail "full-learning-experience adapted across namespaces"
grep -q '"bad_lesson_still_active": 0' "$learning_experience_json" || fail "full-learning-experience left bad lesson active"
grep -q '"experience_export_unredacted_raw": 0' "$learning_experience_json" || fail "full-learning-experience leaked raw export content"
grep -q '"connected_outcome_without_evidence": 0' "$learning_experience_json" || fail "full-learning-experience recorded connected outcome without evidence"
grep -q '"real_external_http_calls": 0' "$learning_experience_json" || fail "full-learning-experience reported real external HTTP calls"
grep -q '"secret_reads": 0' "$learning_experience_json" || fail "full-learning-experience read secrets"
grep -q '"product_feature_claims": "PARTIAL_FULL_LEARNING_EXPERIENCE_ONLY"' "$learning_experience_json" || fail "full-learning-experience overclaimed product feature scope"

cornerstone scenario verify full-understanding-ontology --json > "$understanding_ontology_json"
python3 -m json.tool "$understanding_ontology_json" >/dev/null
grep -q '"scenario_set": "full-understanding-ontology"' "$understanding_ontology_json" || fail "full-understanding-ontology report missing scenario set"
grep -q '"blocking": 0' "$understanding_ontology_json" || fail "full-understanding-ontology report has blocking scenarios"
grep -q '"pass": 7' "$understanding_ontology_json" || fail "full-understanding-ontology did not pass exactly seven scenarios"
grep -q '"id": "CS-UND-006"' "$understanding_ontology_json" || fail "full-understanding-ontology missing CS-UND-006"
grep -q '"id": "CS-UND-007"' "$understanding_ontology_json" || fail "full-understanding-ontology missing CS-UND-007"
grep -q '"id": "CS-UND-008"' "$understanding_ontology_json" || fail "full-understanding-ontology missing CS-UND-008"
grep -q '"id": "CS-UND-009"' "$understanding_ontology_json" || fail "full-understanding-ontology missing CS-UND-009"
grep -q '"id": "CS-UND-010"' "$understanding_ontology_json" || fail "full-understanding-ontology missing CS-UND-010"
grep -q '"id": "CS-UND-011"' "$understanding_ontology_json" || fail "full-understanding-ontology missing CS-UND-011"
grep -q '"id": "CS-UND-012"' "$understanding_ontology_json" || fail "full-understanding-ontology missing CS-UND-012"
grep -q '"suggestion_kinds":' "$understanding_ontology_json" || fail "full-understanding-ontology missing suggestion kinds"
grep -q '"object"' "$understanding_ontology_json" || fail "full-understanding-ontology missing object suggestions"
grep -q '"fact"' "$understanding_ontology_json" || fail "full-understanding-ontology missing fact suggestions"
grep -q '"event"' "$understanding_ontology_json" || fail "full-understanding-ontology missing event suggestions"
grep -q '"link"' "$understanding_ontology_json" || fail "full-understanding-ontology missing link suggestions"
grep -q '"staleness_status": "needs_review"' "$understanding_ontology_json" || fail "full-understanding-ontology missing stale context review warning"
grep -q '"from": 1' "$understanding_ontology_json" || fail "full-understanding-ontology missing ontology from-version"
grep -q '"to": 2' "$understanding_ontology_json" || fail "full-understanding-ontology missing ontology to-version"
grep -q '"approved_truth_without_promotion": 0' "$understanding_ontology_json" || fail "full-understanding-ontology approved truth without promotion"
grep -q '"suggestions_without_evidence": 0' "$understanding_ontology_json" || fail "full-understanding-ontology suggested without evidence"
grep -q '"silent_contradiction_choice": 0' "$understanding_ontology_json" || fail "full-understanding-ontology silently chose contradiction"
grep -q '"stale_truth_used_without_warning": 0' "$understanding_ontology_json" || fail "full-understanding-ontology used stale truth without warning"
grep -q '"unversioned_ontology_changes": 0' "$understanding_ontology_json" || fail "full-understanding-ontology changed ontology without version"
grep -q '"domain_specific_certainty_without_evidence": 0' "$understanding_ontology_json" || fail "full-understanding-ontology claimed unknown-domain certainty"
grep -q '"real_external_http_calls": 0' "$understanding_ontology_json" || fail "full-understanding-ontology reported real external HTTP calls"
grep -q '"secret_reads": 0' "$understanding_ontology_json" || fail "full-understanding-ontology read secrets"
grep -q '"product_feature_claims": "PARTIAL_FULL_UNDERSTANDING_ONTOLOGY_ONLY"' "$understanding_ontology_json" || fail "full-understanding-ontology overclaimed product feature scope"

cornerstone scenario verify full-extension-ecosystem --json > "$extension_ecosystem_json"
python3 -m json.tool "$extension_ecosystem_json" >/dev/null
grep -q '"scenario_set": "full-extension-ecosystem"' "$extension_ecosystem_json" || fail "full-extension-ecosystem report missing scenario set"
grep -q '"blocking": 0' "$extension_ecosystem_json" || fail "full-extension-ecosystem report has blocking scenarios"
grep -q '"pass": 20' "$extension_ecosystem_json" || fail "full-extension-ecosystem did not pass exactly twenty scenarios"
grep -q '"id": "CS-EXT-001"' "$extension_ecosystem_json" || fail "full-extension-ecosystem missing CS-EXT-001"
grep -q '"id": "CS-EXT-016"' "$extension_ecosystem_json" || fail "full-extension-ecosystem missing CS-EXT-016"
grep -q '"id": "CS-SEC-015"' "$extension_ecosystem_json" || fail "full-extension-ecosystem missing CS-SEC-015"
grep -q '"id": "CS-SEC-016"' "$extension_ecosystem_json" || fail "full-extension-ecosystem missing CS-SEC-016"
grep -q '"id": "CS-REG-014"' "$extension_ecosystem_json" || fail "full-extension-ecosystem missing CS-REG-014"
grep -q '"id": "CS-REG-015"' "$extension_ecosystem_json" || fail "full-extension-ecosystem missing CS-REG-015"
grep -q '"pack_id": "pack_ops_recovery_agent"' "$extension_ecosystem_json" || fail "full-extension-ecosystem missing trusted Agent Pack"
grep -q '"registry_sources": \[' "$extension_ecosystem_json" || fail "full-extension-ecosystem missing registry source evidence"
grep -q '"curated_certified"' "$extension_ecosystem_json" || fail "full-extension-ecosystem missing curated certified registry source"
grep -q '"untrusted_activation_exit_code": 8' "$extension_ecosystem_json" || fail "full-extension-ecosystem allowed untrusted activation"
grep -q '"direct_provider_import_exit_code": 8' "$extension_ecosystem_json" || fail "full-extension-ecosystem allowed direct provider pack import"
grep -q '"core_requires_pack": 0' "$extension_ecosystem_json" || fail "full-extension-ecosystem made core depend on pack"
grep -q '"install_granted_authority": 0' "$extension_ecosystem_json" || fail "full-extension-ecosystem install granted authority"
grep -q '"silent_activation": 0' "$extension_ecosystem_json" || fail "full-extension-ecosystem silently activated a pack"
grep -q '"silent_behavior_update": 0' "$extension_ecosystem_json" || fail "full-extension-ecosystem silently changed behavior"
grep -q '"direct_provider_access": 0' "$extension_ecosystem_json" || fail "full-extension-ecosystem allowed direct provider access"
grep -q '"extension_owned_credentials": 0' "$extension_ecosystem_json" || fail "full-extension-ecosystem allowed extension-owned credentials"
grep -q '"playbook_auto_globalized": 0' "$extension_ecosystem_json" || fail "full-extension-ecosystem auto-globalized a playbook"
grep -q '"behavior_patch_applied_without_review": 0' "$extension_ecosystem_json" || fail "full-extension-ecosystem applied behavior patch without review"
grep -q '"real_external_http_calls": 0' "$extension_ecosystem_json" || fail "full-extension-ecosystem reported real external HTTP calls"
grep -q '"secret_reads": 0' "$extension_ecosystem_json" || fail "full-extension-ecosystem read secrets"
grep -q '"product_feature_claims": "PARTIAL_FULL_EXTENSION_ECOSYSTEM_ONLY"' "$extension_ecosystem_json" || fail "full-extension-ecosystem overclaimed product feature scope"

cornerstone scenario verify vs0-product-domain-readiness --json > "$product_domain_json"
python3 -m json.tool "$product_domain_json" >/dev/null
grep -q '"scenario_set": "vs0-product-domain-readiness"' "$product_domain_json" || fail "vs0-product-domain-readiness report missing scenario set"
grep -q '"blocking": 0' "$product_domain_json" || fail "vs0-product-domain-readiness report has blocking scenarios"
grep -q '"pass": 3' "$product_domain_json" || fail "vs0-product-domain-readiness did not pass exactly three scenarios"
grep -q '"id": "CS-PROD-001"' "$product_domain_json" || fail "vs0-product-domain-readiness missing CS-PROD-001"
grep -q '"id": "CS-PROD-003"' "$product_domain_json" || fail "vs0-product-domain-readiness missing CS-PROD-003"
grep -q '"id": "CS-AUTO-002"' "$product_domain_json" || fail "vs0-product-domain-readiness missing CS-AUTO-002"
grep -q '"walkthrough_product_name": "CornerStone"' "$product_domain_json" || fail "vs0-product-domain-readiness missing CornerStone walkthrough"
grep -q '"walkthrough_one_service": true' "$product_domain_json" || fail "vs0-product-domain-readiness missing one-service walkthrough"
grep -q '"daily_user_requires_subsystem_knowledge": false' "$product_domain_json" || fail "vs0-product-domain-readiness required subsystem knowledge"
grep -q '"domain_count": 3' "$product_domain_json" || fail "vs0-product-domain-readiness missing three domains"
grep -q '"initial_workspace_mode": "assist"' "$product_domain_json" || fail "vs0-product-domain-readiness did not start conservatively"
grep -q '"recommendation": "recommend_autopilot"' "$product_domain_json" || fail "vs0-product-domain-readiness missing Autopilot recommendation"
grep -q '"recommended_mode": "autopilot"' "$product_domain_json" || fail "vs0-product-domain-readiness missing recommended mode"
grep -q '"mission_contract_required": true' "$product_domain_json" || fail "vs0-product-domain-readiness missing mission contract requirement"
grep -q '"successful_internal_task_count": 1' "$product_domain_json" || fail "vs0-product-domain-readiness missing successful internal task"
grep -q '"successful_playbook_count": 1' "$product_domain_json" || fail "vs0-product-domain-readiness missing successful playbook count"
grep -q '"subsystem_identity_required": 0' "$product_domain_json" || fail "vs0-product-domain-readiness required subsystem identity"
grep -q '"missing_navigation_items": 0' "$product_domain_json" || fail "vs0-product-domain-readiness missed navigation items"
grep -q '"logistics_required": 0' "$product_domain_json" || fail "vs0-product-domain-readiness required logistics terms"
grep -q '"domain_failures": 0' "$product_domain_json" || fail "vs0-product-domain-readiness reported domain failures"
grep -q '"readiness_recommended_without_history": 0' "$product_domain_json" || fail "vs0-product-domain-readiness recommended without history"
grep -q '"autopilot_authority_granted_without_mission_contract": 0' "$product_domain_json" || fail "vs0-product-domain-readiness granted authority without mission contract"
grep -q '"real_external_http_calls": 0' "$product_domain_json" || fail "vs0-product-domain-readiness reported real external HTTP calls"
grep -q '"product_feature_claims": "PARTIAL_VS0_PRODUCT_DOMAIN_READINESS_ONLY"' "$product_domain_json" || fail "vs0-product-domain-readiness overclaimed product feature scope"

python3 -m unittest discover -s tests -p 'test_*.py'

printf 'PASS: CornerStone scaffold CLI verified (version, health, honest ready, scenario list, coverage, vs0-scaffold verify, vs0-fixtures verify, vs0-artifacts verify, vs0-security verify, vs0-search-evidence verify, vs0-search-understanding verify, vs0-namespace-isolation verify, vs0-audit-ledger verify, vs0-universal-core verify, vs0-claim-evidence verify, vs0-security-policy verify, vs0-regression-guardrails verify, vs0-briefing verify, vs0-mission-action verify, vs0-detail-surfaces verify, vs0-conversation-onboarding verify, vs0-product-loop-identity verify, vs0-memory-truth-boundary verify, vs0-tenant-security-boundary verify, full-claim-collaboration verify, full-memory-wiki verify, full-learning-experience verify, full-understanding-ontology verify, full-extension-ecosystem verify, vs0-product-domain-readiness verify, unittest).\n'
