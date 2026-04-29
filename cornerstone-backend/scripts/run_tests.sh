#!/usr/bin/env bash
set -euo pipefail

mkdir -p reports
find src tests -type d -name '__pycache__' -prune -exec rm -rf {} +
find src tests -name '*.pyc' -delete
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD="${PYTEST_DISABLE_PLUGIN_AUTOLOAD:-1}"
export DATABASE_URL="${DATABASE_URL:-postgresql+psycopg://cornerstone:cornerstone@localhost:5432/cornerstone}"

PYTEST_ARGS=("$@")
if [[ "${RUN_POSTGRES_TESTS:-0}" != "1" ]]; then
  PYTEST_ARGS=(--ignore=tests/postgres "${PYTEST_ARGS[@]}")
fi
if [[ "${RUN_NOTION_E2E:-0}" != "1" && "${RUN_LIVE_NOTION_TESTS:-0}" != "1" ]]; then
  PYTEST_ARGS=(--ignore=tests/live_notion "${PYTEST_ARGS[@]}")
fi

python -m compileall -q src tests > reports/compile-report.txt 2>&1 || {
  cat reports/compile-report.txt
  exit 1
}
echo "compileall passed" > reports/compile-report.txt

alembic upgrade head --sql > reports/alembic-offline.sql 2> reports/alembic-offline.log
echo "Alembic offline SQL rendered" >> reports/alembic-offline.log

python -m pytest tests --color=no "${PYTEST_ARGS[@]}" > reports/test-report.txt 2>&1

coverage erase
coverage run -m pytest tests -q --color=no "${PYTEST_ARGS[@]}" > reports/coverage-pytest-report.txt 2>&1
coverage report > reports/coverage-summary.txt
coverage xml -o reports/coverage.xml > reports/coverage-xml-report.txt

ruff check src tests migrations scripts > reports/ruff-report.txt 2>&1

MYPY_TIMEOUT_SECONDS="${MYPY_TIMEOUT_SECONDS:-180}"
rm -rf reports/.mypy_cache
if command -v timeout >/dev/null 2>&1; then
  timeout "$MYPY_TIMEOUT_SECONDS" mypy src --show-error-codes --no-color-output --no-incremental --cache-dir reports/.mypy_cache > reports/mypy-report.txt 2>&1
else
  mypy src --show-error-codes --no-color-output --no-incremental --cache-dir reports/.mypy_cache > reports/mypy-report.txt 2>&1
fi

grep -E "=+ .*passed" reports/test-report.txt | tail -1
tail -1 reports/coverage-summary.txt
cat reports/ruff-report.txt
cat reports/mypy-report.txt
cat reports/compile-report.txt
printf 'Reports written to reports/\n'
