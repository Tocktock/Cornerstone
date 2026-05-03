#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "macos_run_live_proof.sh is kept as a compatibility wrapper."
echo "Using cross-platform Unix proof script: scripts/run_live_proof.sh"
exec bash scripts/run_live_proof.sh
