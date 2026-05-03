#!/usr/bin/env bash
set -euo pipefail

# Cross-platform Unix local setup helper for macOS and Linux.
# Windows users should use scripts/windows_setup.ps1.

cd "$(dirname "$0")/.."

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required."
  exit 1
fi

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev]'

cornerstone doctor --fix
echo
echo "Local setup complete."
echo "Next:"
echo "  cornerstone stack up --migrate"
echo "  cornerstone api --reload"
