#!/usr/bin/env bash
set -euo pipefail

# Strict mode is intentionally explicit because the full dependency-complete
# verification applies migrations and live PostgreSQL tests to DATABASE_URL.
python scripts/run_dependency_complete_verification.py --strict --confirm-live-db "$@"
