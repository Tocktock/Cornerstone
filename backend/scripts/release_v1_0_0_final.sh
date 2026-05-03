#!/usr/bin/env bash
set -euo pipefail

echo "Cornerstone Backend v1.0.0 final release helper"
python scripts/check_release_candidate.py
python -m mypy src --show-error-codes --no-color-output --no-incremental
./scripts/run_tests.sh
echo "If checks pass and live gates/sign-off are complete, tag with:"
echo "  git tag -a v1.0.0 -m 'cornerstone-backend v1.0.0'"
