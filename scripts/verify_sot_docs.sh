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

require_file "README.md"
require_file "AGENTS.md"
require_file "docs/handoff/AI_AGENT_HANDOFF_V2_FULL_WITH_MUST_PASS_EMBEDDED.md"
require_file "docs/sot/README.md"
require_file "docs/sot/01_PRODUCT_GOAL_AND_DIRECTION.md"
require_file "docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md"
require_file "docs/sot/03_TECHNICAL_ARCHITECTURE_DEFAULTS.md"
require_file "docs/sot/sot_manifest.yaml"
require_file "docs/scenario-contracts/SCENARIO_MATRIX_FULL.md"
require_file "docs/scenario-contracts/SCENARIO_MATRIX_FULL.csv"
require_file "docs/scenario-contracts/VS0_SCAFFOLD_CONTRACT.md"
require_file "docs/scenario-contracts/VS0_IMPLEMENTATION_CONTRACT.md"
require_file "docs/scenario-contracts/CLI_NATIVE_FIRST_CONTRACT.md"
require_file "docs/scenario-contracts/CLI_FEATURE_PARITY_MATRIX.csv"
require_file "docs/scenario-contracts/LOCAL_VERIFICATION_PLANE_V0.md"
require_file "docs/design/DESIGN_SYSTEM_CONTRACT_V0_3.md"
require_file "docs/design/DESIGN_CONCEPT_SYSTEM_V0_3.md"
require_file "docs/design/tokens/cornerstone_design_tokens_v0_3.json"
require_file "docs/verification-reports/VS0_SCAFFOLD_READINESS_REPORT_V0.md"
require_file "docs/verification-reports/template.md"
require_file "docs/adr/ADR-0002-framework-and-version-policy.md"
require_file "docs/adr/ADR-0003-monorepo-setup.md"
require_file "docs/adr/ADR-0004-cli-native-first-setup.md"
require_file "docs/adr/ADR-0005-domain-boundaries.md"
require_file "docs/adr/ADR-0006-agent-guide.md"
require_file "scripts/verify_cli_native_first_docs.sh"
require_file "scripts/verify_local_verification_plane_docs.sh"
require_file "scripts/verify_design_system_docs.sh"
require_file "scripts/verify_vs0_scaffold_readiness_docs.sh"
require_file "docs/agent/SCENARIO_FIRST_AGENT_INSTRUCTION.md"
require_file "docs/agent/PROJECT_OPERATING_CONSTITUTION.md"

scenario_count=$(grep -E '^## CS-[A-Z]+-[0-9]{3} ' docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md | wc -l | tr -d ' ')
[ "$scenario_count" = "206" ] || fail "expected 206 full scenarios, found $scenario_count"

matrix_md_count=$(grep -E '^\| CS-[A-Z]+-[0-9]{3} \|' docs/scenario-contracts/SCENARIO_MATRIX_FULL.md | wc -l | tr -d ' ')
[ "$matrix_md_count" = "206" ] || fail "expected 206 markdown matrix rows, found $matrix_md_count"

matrix_csv_count=$(awk -F, 'NR > 1 && $1 ~ /^CS-/ { count++ } END { print count + 0 }' docs/scenario-contracts/SCENARIO_MATRIX_FULL.csv)
[ "$matrix_csv_count" = "206" ] || fail "expected 206 csv matrix rows, found $matrix_csv_count"

vs0_count=$(grep -E '^\| CS-[A-Z]+-[0-9]{3} \|' docs/scenario-contracts/VS0_IMPLEMENTATION_CONTRACT.md | wc -l | tr -d ' ')
[ "$vs0_count" = "58" ] || fail "expected 58 VS-0 scenario rows, found $vs0_count"

grep -q 'Total parsed scenario IDs: \*\*206\*\*' docs/scenario-contracts/SCENARIO_MATRIX_FULL.md || fail "matrix markdown missing 206 total marker"
grep -q 'Total VS-0 scenario IDs: \*\*58\*\*' docs/scenario-contracts/VS0_IMPLEMENTATION_CONTRACT.md || fail "VS-0 contract missing 58 total marker"
grep -q 'scenario_count: 206' docs/sot/sot_manifest.yaml || fail "manifest missing full scenario count"
grep -q 'scenario_count: 58' docs/sot/sot_manifest.yaml || fail "manifest missing VS-0 scenario count"
grep -q 'No CLI, no feature PASS' docs/scenario-contracts/CLI_NATIVE_FIRST_CONTRACT.md || fail "CLI native-first hard gate missing"
grep -q 'cli_native_first:' docs/sot/sot_manifest.yaml || fail "manifest missing CLI native-first section"
grep -q 'CLI_FEATURE_PARITY_MATRIX.csv' docs/sot/README.md || fail "SoT README missing CLI feature parity matrix"
grep -q 'local_verification_plane:' docs/sot/sot_manifest.yaml || fail "manifest missing local verification plane section"
grep -q 'LOCAL_VERIFICATION_PLANE_V0.md' README.md || fail "README missing local verification plane"
grep -q 'LOCAL_VERIFICATION_PLANE_V0.md' docs/sot/README.md || fail "SoT README missing local verification plane"
grep -q 'design_system:' docs/sot/sot_manifest.yaml || fail "manifest missing design system section"
grep -q 'DESIGN_SYSTEM_CONTRACT_V0_3.md' README.md || fail "README missing design system contract"
grep -q 'DESIGN_SYSTEM_CONTRACT_V0_3.md' docs/sot/README.md || fail "SoT README missing design system contract"
grep -q 'DESIGN_SYSTEM_CONTRACT_V0_3.md' AGENTS.md || fail "AGENTS missing design system contract"
grep -q 'vs0_scaffold_readiness:' docs/sot/sot_manifest.yaml || fail "manifest missing VS-0 scaffold readiness section"
grep -q 'VS0_SCAFFOLD_READINESS_REPORT_V0.md' README.md || fail "README missing VS-0 scaffold readiness report"
grep -q 'VS0_SCAFFOLD_READINESS_REPORT_V0.md' docs/sot/README.md || fail "SoT README missing VS-0 scaffold readiness report"
grep -q 'VS0_SCAFFOLD_READINESS_REPORT_V0.md' AGENTS.md || fail "AGENTS missing VS-0 scaffold readiness report"
grep -q 'vs0_scaffold:' docs/sot/sot_manifest.yaml || fail "manifest missing VS-0 scaffold section"
grep -q 'Python target: 3.14.x' docs/scenario-contracts/VS0_SCAFFOLD_CONTRACT.md || fail "VS-0 scaffold contract missing Python 3.14 target"
grep -q 'Node target: 24.x LTS' docs/scenario-contracts/VS0_SCAFFOLD_CONTRACT.md || fail "VS-0 scaffold contract missing Node 24 LTS target"
grep -q 'PostgreSQL target: 18.x' docs/scenario-contracts/VS0_SCAFFOLD_CONTRACT.md || fail "VS-0 scaffold contract missing PostgreSQL 18 target"
grep -q 'VS0-SCAF-001' docs/scenario-contracts/VS0_SCAFFOLD_CONTRACT.md || fail "VS-0 scaffold scenarios missing"
grep -q 'ADR-0002-framework-and-version-policy.md' docs/scenario-contracts/VS0_SCAFFOLD_CONTRACT.md || fail "VS-0 scaffold contract missing version-policy ADR reference"
grep -q 'CLI Parity Summary' docs/verification-reports/template.md || fail "verification report template missing CLI parity section"
grep -q 'VS0_SCAFFOLD_CONTRACT.md' README.md || fail "README missing VS-0 scaffold contract"
grep -q 'VS0_SCAFFOLD_CONTRACT.md' AGENTS.md || fail "AGENTS missing VS-0 scaffold contract"

sh scripts/verify_cli_native_first_docs.sh
sh scripts/verify_local_verification_plane_docs.sh
sh scripts/verify_design_system_docs.sh
sh scripts/verify_vs0_scaffold_readiness_docs.sh

printf 'PASS: CornerStone SoT docs verified (206 full scenarios, design system, VS-0 scaffold readiness, VS-0 scaffold gate, 58 VS-0 scenarios, CLI native-first gate, local verification plane).\n'
