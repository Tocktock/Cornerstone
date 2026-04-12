#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export CORNERSTONE_LAUNCHER_PROFILE=dev
export CORNERSTONE_LAUNCHER_NAME=run-dev.sh
export CORNERSTONE_RUNTIME_MODE=mock
export CORNERSTONE_AUTO_SEED_DEMO=true
export CORNERSTONE_NOTION_DEMO_OAUTH_MODE=true

exec "$ROOT_DIR/run-all.sh" "$@"
