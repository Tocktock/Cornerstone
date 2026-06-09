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

require_file "docs/design/DESIGN_SYSTEM_CONTRACT_V0_3.md"
require_file "docs/design/DESIGN_CONCEPT_SYSTEM_V0_3.md"
require_file "docs/design/tokens/cornerstone_design_tokens_v0_3.json"

grep -q 'Calm Surface. Deep Evidence. Safe Action.' docs/design/DESIGN_SYSTEM_CONTRACT_V0_3.md || fail "design contract missing doctrine"
grep -q 'Reference Image Reading' docs/design/DESIGN_SYSTEM_CONTRACT_V0_3.md || fail "design contract missing reference image reading"
grep -q 'Home / Universal Workspace' docs/design/DESIGN_SYSTEM_CONTRACT_V0_3.md || fail "design contract missing Home surface"
grep -q 'Action Studio' docs/design/DESIGN_SYSTEM_CONTRACT_V0_3.md || fail "design contract missing Action Studio surface"
grep -q 'Admin Console' docs/design/DESIGN_SYSTEM_CONTRACT_V0_3.md || fail "design contract missing Admin Console surface"
grep -q 'DS-S01' docs/design/DESIGN_SYSTEM_CONTRACT_V0_3.md || fail "design contract missing design acceptance scenarios"
grep -q 'Do not make the first screen an admin dashboard' docs/design/DESIGN_CONCEPT_SYSTEM_V0_3.md || fail "source concept missing design redlines"

python3 - <<'PY'
import json
from pathlib import Path

path = Path("docs/design/tokens/cornerstone_design_tokens_v0_3.json")
data = json.loads(path.read_text())

required_top = ["meta", "color", "space", "radius", "layout", "shadow", "typography", "state"]
missing_top = [key for key in required_top if key not in data]
if missing_top:
    raise SystemExit(f"FAIL: missing top-level token groups: {missing_top}")

required_color = ["background", "border", "text", "primary", "evidence", "review", "danger", "neutral"]
missing_color = [key for key in required_color if key not in data["color"]]
if missing_color:
    raise SystemExit(f"FAIL: missing color token groups: {missing_color}")

required_states = [
    "saved",
    "searchable",
    "draft",
    "evidenceBacked",
    "corroborated",
    "underReview",
    "insufficientEvidence",
    "approved",
    "executed",
    "failed",
    "policyBlocked",
]
missing_states = [key for key in required_states if key not in data["state"]]
if missing_states:
    raise SystemExit(f"FAIL: missing state tokens: {missing_states}")

for state_name, state in data["state"].items():
    for field in ["bg", "fg", "border"]:
        if field not in state:
            raise SystemExit(f"FAIL: state {state_name} missing {field}")

if data.get("meta", {}).get("doctrine") != "Calm Surface. Deep Evidence. Safe Action.":
    raise SystemExit("FAIL: token doctrine mismatch")

print(f"PASS: design tokens verified ({len(required_states)} state tokens, {len(required_color)} color groups).")
PY

grep -q 'design_system:' docs/sot/sot_manifest.yaml || fail "manifest missing design system section"
grep -q 'DESIGN_SYSTEM_CONTRACT_V0_3.md' README.md || fail "README missing design system contract"
grep -q 'DESIGN_SYSTEM_CONTRACT_V0_3.md' docs/sot/README.md || fail "SoT README missing design system contract"
grep -q 'DESIGN_SYSTEM_CONTRACT_V0_3.md' AGENTS.md || fail "AGENTS missing design system contract"

printf 'PASS: CornerStone design system docs verified.\n'
