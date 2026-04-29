#!/usr/bin/env bash
set -euo pipefail

export DATABASE_URL="${DATABASE_URL:-postgresql+psycopg://cornerstone:cornerstone@localhost:5432/cornerstone}"
export PERSISTENCE_BACKEND="${PERSISTENCE_BACKEND:-postgres}"
export RUN_POSTGRES_TESTS="${RUN_POSTGRES_TESTS:-1}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD="${PYTEST_DISABLE_PLUGIN_AUTOLOAD:-1}"

python scripts/run_live_postgres_tests.py
RUN_POSTGRES_TESTS=0 ./scripts/run_tests.sh --ignore=tests/postgres "$@"
