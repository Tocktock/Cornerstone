#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "macos_setup.sh is kept as a compatibility wrapper."
echo "Using cross-platform Unix setup script: scripts/setup_local.sh"
exec bash scripts/setup_local.sh
