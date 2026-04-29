#!/usr/bin/env bash
set -euo pipefail

mkdir -p reports
export PYTEST_DISABLE_PLUGIN_AUTOLOAD="${PYTEST_DISABLE_PLUGIN_AUTOLOAD:-1}"
export RUN_NOTION_E2E="${RUN_NOTION_E2E:-1}"

python scripts/run_live_notion_e2e.py > reports/live-notion-e2e-report.txt 2>&1
python -m pytest tests/live_notion -m live_notion --color=no > reports/live-notion-e2e-pytest-report.txt 2>&1
cat reports/live-notion-e2e-report.txt
cat reports/live-notion-e2e-pytest-report.txt
