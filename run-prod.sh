#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export CORNERSTONE_LAUNCHER_PROFILE=prod
export CORNERSTONE_LAUNCHER_NAME=run-prod.sh
export CORNERSTONE_COMPOSE_PROJECT_NAME=cornerstone-prod
export CORNERSTONE_RUNTIME_MODE=production
export CORNERSTONE_AUTO_SEED_DEMO=false
export CORNERSTONE_NOTION_DEMO_OAUTH_MODE=false

exec "$ROOT_DIR/run-all.sh" "$@"
