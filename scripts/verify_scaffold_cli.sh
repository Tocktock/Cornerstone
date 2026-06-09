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
trap 'rm -f "$version_json" "$health_json" "$ready_json" "$list_json" "$coverage_json" "$verify_json" "$fixtures_json" "$artifacts_json" "$security_json" "$search_json" "$understanding_json" "$namespace_json" "$audit_json" "$universal_json"' EXIT

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

python3 -m unittest discover -s tests -p 'test_*.py'

printf 'PASS: CornerStone scaffold CLI verified (version, health, honest ready, scenario list, coverage, vs0-scaffold verify, vs0-fixtures verify, vs0-artifacts verify, vs0-security verify, vs0-search-evidence verify, vs0-search-understanding verify, vs0-namespace-isolation verify, vs0-audit-ledger verify, vs0-universal-core verify, unittest).\n'
