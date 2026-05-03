#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cornerstone.verification.postgres_ci import (  # noqa: E402
    PostgresCiVerificationError,
    assert_live_postgres_report_succeeded,
    validate_live_postgres_environment,
)

REPORT_DIR = ROOT / "reports"
LIVE_REPORT = REPORT_DIR / "live-postgres-report.txt"
LIVE_SUMMARY = REPORT_DIR / "live-postgres-summary.txt"


def _run(command: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run strict live PostgreSQL verification tests.")
    parser.add_argument("--min-passed", type=int, default=3, help="Minimum live PostgreSQL tests that must pass.")
    parser.add_argument(
        "--pytest-arg",
        action="append",
        default=[],
        help="Additional argument passed through to pytest. Repeat for multiple arguments.",
    )
    args = parser.parse_args()

    REPORT_DIR.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

    try:
        database_url = validate_live_postgres_environment(env)
    except PostgresCiVerificationError as exc:
        LIVE_REPORT.write_text(f"preflight failed: {exc}\n", encoding="utf-8")
        LIVE_SUMMARY.write_text(f"status=preflight_failed\nreason={exc}\n", encoding="utf-8")
        print(exc, file=sys.stderr)
        return 2

    env["DATABASE_URL"] = database_url
    env["RUN_POSTGRES_TESTS"] = "1"
    env.setdefault("PERSISTENCE_BACKEND", "postgres")

    output_parts: list[str] = []
    for command in (
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        [sys.executable, "scripts/check_postgres_extensions.py"],
        [sys.executable, "-m", "pytest", "tests/postgres", "-m", "postgres", "-vv", "--color=no", *args.pytest_arg],
    ):
        result = _run(command, env=env)
        output_parts.append(f"$ {' '.join(command)}\n{result.stdout}")
        if result.returncode != 0:
            LIVE_REPORT.write_text("\n\n".join(output_parts), encoding="utf-8")
            return result.returncode

    report_text = "\n\n".join(output_parts)
    LIVE_REPORT.write_text(report_text, encoding="utf-8")

    try:
        summary = assert_live_postgres_report_succeeded(report_text, min_passed=args.min_passed)
    except PostgresCiVerificationError as exc:
        LIVE_SUMMARY.write_text(f"failed: {exc}\n", encoding="utf-8")
        print(exc, file=sys.stderr)
        return 1

    LIVE_SUMMARY.write_text(
        f"passed={summary.passed}\nskipped={summary.skipped}\nfailed={summary.failed}\nerrors={summary.errors}\n",
        encoding="utf-8",
    )
    print(f"Live PostgreSQL verification passed: {summary.passed} tests, 0 skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
