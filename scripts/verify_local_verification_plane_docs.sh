#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$ROOT"

fail() {
  printf 'FAIL: %s\n' "$1" >&2
  exit 1
}

require_file() {
  [ -f "$1" ] || fail "missing required file: $1"
}

require_file "docs/scenario-contracts/LOCAL_VERIFICATION_PLANE_V0.md"
require_file "docs/scenario-contracts/CLI_NATIVE_FIRST_CONTRACT.md"
require_file "docs/sot/sot_manifest.yaml"
require_file "docs/scenario-contracts/SCENARIO_VERIFICATION_REPORT_TEMPLATE.md"

grep -q 'cornerstone scenario verify <contract> --json' docs/scenario-contracts/LOCAL_VERIFICATION_PLANE_V0.md || fail "local verification plane missing product-level scenario verify entrypoint"
grep -q 'It must never be the judge of scenario `PASS`' docs/scenario-contracts/LOCAL_VERIFICATION_PLANE_V0.md || fail "local verification plane missing LLM-not-judge rule"
grep -q 'deterministic validators' docs/scenario-contracts/LOCAL_VERIFICATION_PLANE_V0.md || fail "local verification plane missing deterministic validator language"
grep -q 'local_test' docs/scenario-contracts/LOCAL_VERIFICATION_PLANE_V0.md || fail "local verification plane missing local_test provider"
grep -q 'negative_evidence' docs/scenario-contracts/LOCAL_VERIFICATION_PLANE_V0.md || fail "local verification plane missing negative evidence contract"
grep -q 'SCENARIO_VERIFICATION_MATRIX.csv' docs/scenario-contracts/LOCAL_VERIFICATION_PLANE_V0.md || fail "local verification plane missing coverage matrix target"
grep -q 'make verify-local-fast' docs/scenario-contracts/LOCAL_VERIFICATION_PLANE_V0.md || fail "local verification plane missing planned fast verification command"
grep -q 'Local Verification Plane:' docs/scenario-contracts/LOCAL_VERIFICATION_PLANE_V0.md || fail "local verification plane missing final report addition"
grep -q 'Local Verification Plane:' docs/scenario-contracts/SCENARIO_VERIFICATION_REPORT_TEMPLATE.md || fail "verification report template missing local verification section"
grep -q 'local_verification_plane:' docs/sot/sot_manifest.yaml || fail "manifest missing local verification plane section"
grep -q 'LOCAL_VERIFICATION_PLANE_V0.md' AGENTS.md || fail "AGENTS missing local verification plane"
grep -q 'LOCAL_VERIFICATION_PLANE_V0.md' README.md || fail "README missing local verification plane"
grep -q 'LOCAL_VERIFICATION_PLANE_V0.md' docs/sot/README.md || fail "SoT README missing local verification plane"

section_count=$(grep -E '^## [0-9]+\.' docs/scenario-contracts/LOCAL_VERIFICATION_PLANE_V0.md | wc -l | tr -d ' ')
[ "$section_count" -ge "20" ] || fail "expected at least 20 numbered local verification sections, found $section_count"

printf 'PASS: CornerStone local verification plane docs verified (%s numbered sections; deterministic PASS gate documented).\n' "$section_count"
