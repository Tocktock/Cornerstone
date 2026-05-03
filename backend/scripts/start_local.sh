#!/usr/bin/env bash
set -euo pipefail

# Cross-platform Unix local startup helper for macOS and Linux.
# Windows users should use scripts/windows_start_local.ps1.

cd "$(dirname "$0")/.."

if [[ -f .venv/bin/activate ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

cornerstone stack up --migrate
echo
echo "Starting API. Open another terminal for worker/proof commands."
cornerstone api --reload
