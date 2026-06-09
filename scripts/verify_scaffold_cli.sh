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
trap 'rm -f "$version_json" "$health_json" "$ready_json" "$list_json" "$coverage_json" "$verify_json" "$fixtures_json" "$artifacts_json"' EXIT

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

python3 -m unittest discover -s tests -p 'test_*.py'

printf 'PASS: CornerStone scaffold CLI verified (version, health, honest ready, scenario list, coverage, vs0-scaffold verify, vs0-fixtures verify, vs0-artifacts verify, unittest).\n'
