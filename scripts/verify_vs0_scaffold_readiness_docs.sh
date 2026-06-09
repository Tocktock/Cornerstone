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

REPORT="docs/verification-reports/VS0_SCAFFOLD_READINESS_REPORT_V0.md"

require_file "$REPORT"
require_file "docs/scenario-contracts/VS0_SCAFFOLD_CONTRACT.md"
require_file "docs/scenario-contracts/LOCAL_VERIFICATION_PLANE_V0.md"
require_file "docs/adr/ADR-0002-framework-and-version-policy.md"

grep -q 'Next implementation target: VS-0 Scaffold Foundation.' "$REPORT" || fail "readiness report missing scaffold target"
grep -q 'Do not proceed directly to VS-0 product features.' "$REPORT" || fail "readiness report missing feature-blocking rule"
grep -q 'Claim Local Verification Plane is implemented | No' "$REPORT" || fail "readiness report must not claim local verification implementation"
grep -q 'Current environment fully matches scaffold target tools.' "$REPORT" || fail "readiness report missing environment readiness row"
grep -q '| VS0-RDY-003 | MUST_PASS | Current environment fully matches scaffold target tools.' "$REPORT" || fail "readiness report missing VS0-RDY-003 gate"
grep -q '| H-SCAF-RDY-001 |' "$REPORT" || fail "readiness report missing dependency approval human gate"
grep -q 'clear only for VS-0 scaffold implementation after preflight and approval; blocked for product-feature implementation' "$REPORT" || fail "readiness verdict must block product features"

grep -q 'VS0_SCAFFOLD_READINESS_REPORT_V0.md' README.md || fail "README missing scaffold readiness report"
grep -q 'VS0_SCAFFOLD_READINESS_REPORT_V0.md' docs/sot/README.md || fail "SoT README missing scaffold readiness report"
grep -q 'vs0_scaffold_readiness:' docs/sot/sot_manifest.yaml || fail "manifest missing scaffold readiness section"
grep -q 'VS0_SCAFFOLD_READINESS_REPORT_V0.md' docs/implementation/ZERO_BASE_IMPLEMENTATION_ROADMAP.md || fail "roadmap missing scaffold readiness report"

printf 'PASS: CornerStone VS-0 scaffold readiness docs verified.\n'
