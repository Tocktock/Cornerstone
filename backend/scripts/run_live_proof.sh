#!/usr/bin/env bash
set -euo pipefail

# Cross-platform Unix live proof helper for macOS and Linux.
# Requires RUN_POSTGRES_TESTS=1 and, for Notion, RUN_NOTION_E2E=1 plus Notion env vars.

cd "$(dirname "$0")/.."

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

cornerstone proof run --all --continue-on-failure --markdown --save reports/cornerstone-live-proof.json
