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

require_file "docs/scenario-contracts/CLI_NATIVE_FIRST_CONTRACT.md"
require_file "docs/scenario-contracts/CLI_FEATURE_PARITY_MATRIX.csv"

grep -q 'No CLI, no feature PASS' docs/scenario-contracts/CLI_NATIVE_FIRST_CONTRACT.md || fail "CLI native-first hard gate missing"
grep -q 'Every shipped CornerStone feature must have a native' docs/scenario-contracts/CLI_NATIVE_FIRST_CONTRACT.md || fail "native CLI requirement missing"
grep -q 'Raw SQL, direct Python scripts' docs/scenario-contracts/CLI_NATIVE_FIRST_CONTRACT.md || fail "bypass prohibition missing"
grep -q 'CLI Parity:' docs/scenario-contracts/CLI_NATIVE_FIRST_CONTRACT.md || fail "required CLI parity section missing"
grep -q 'CS-CLI-001' docs/scenario-contracts/CLI_NATIVE_FIRST_CONTRACT.md || fail "CLI scenarios missing"

grep -q '^feature_area,feature_family,required_command_group' docs/scenario-contracts/CLI_FEATURE_PARITY_MATRIX.csv || fail "CLI feature parity matrix header missing"

row_count=$(awk -F, 'NR > 1 && $1 != "" { count++ } END { print count + 0 }' docs/scenario-contracts/CLI_FEATURE_PARITY_MATRIX.csv)
[ "$row_count" -ge 35 ] || fail "expected at least 35 CLI parity matrix rows, found $row_count"

not_required_count=$(awk -F, 'NR > 1 && $5 != "true" { count++ } END { print count + 0 }' docs/scenario-contracts/CLI_FEATURE_PARITY_MATRIX.csv)
[ "$not_required_count" = "0" ] || fail "all CLI parity rows must set cli_required=true"

not_blocking_count=$(awk -F, 'NR > 1 && $6 != "true" { count++ } END { print count + 0 }' docs/scenario-contracts/CLI_FEATURE_PARITY_MATRIX.csv)
[ "$not_blocking_count" = "0" ] || fail "all CLI parity rows must set release_block_if_missing=true"

missing_command_count=$(awk -F, 'NR > 1 && ($3 == "" || $4 == "") { count++ } END { print count + 0 }' docs/scenario-contracts/CLI_FEATURE_PARITY_MATRIX.csv)
[ "$missing_command_count" = "0" ] || fail "all CLI parity rows must include command group and example commands"

printf 'PASS: CornerStone CLI native-first docs verified (%s feature-family rows; all CLI-required and release-blocking).\n' "$row_count"
