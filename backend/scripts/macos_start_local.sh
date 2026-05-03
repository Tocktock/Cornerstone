#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "macos_start_local.sh is kept as a compatibility wrapper."
echo "Using cross-platform Unix startup script: scripts/start_local.sh"
exec bash scripts/start_local.sh
