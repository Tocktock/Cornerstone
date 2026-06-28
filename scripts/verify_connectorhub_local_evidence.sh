#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

run() {
  printf '\n==> %s\n' "$*"
  "$@"
}

run git diff --check
run scripts/verify_sot_docs.sh
run python3 scripts/verify_connectorhub_engineering_trail.py
run python3 scripts/verify_connectorhub_review_split.py
run python3 -m unittest tests.scenario.test_connectorhub_cli
run python3 -m unittest tests.scenario.test_connectorhub_compact_reports
run python3 -m unittest tests.scenario.test_scaffold_cli
run python3 -m compileall packages/cornerstone_cli

if [[ "${1:-}" == "--strict" ]]; then
  run make verify-vs2-production-like
else
  cat <<'EOF'

Default local evidence gate complete.

Strict Docker rehearsal is intentionally local/manual because it depends on
current VS2 reusable proof state, Docker/network support, and a clean review
workspace:

  scripts/verify_connectorhub_local_evidence.sh --strict

EOF
fi
