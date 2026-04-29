from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlparse


class PostgresCiVerificationError(RuntimeError):
    """Raised when live PostgreSQL verification is misconfigured or incomplete."""


@dataclass(frozen=True)
class PytestSummary:
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    errors: int = 0

    @property
    def succeeded_without_skips(self) -> bool:
        return self.passed > 0 and self.failed == 0 and self.errors == 0 and self.skipped == 0


_SUMMARY_TOKEN_RE = re.compile(r"(?P<count>\d+)\s+(?P<kind>passed|failed|skipped|errors?)")


def parse_pytest_summary(report_text: str) -> PytestSummary:
    """Parse the final pytest terminal summary from a report.

    The parser intentionally looks at the last summary-looking line only. Pytest reports
    may contain historical examples or copied command output earlier in the text, and CI
    should judge the terminal summary produced by the current run.
    """
    summary_lines = [line for line in report_text.splitlines() if "==" in line and " in " in line]
    if not summary_lines:
        return PytestSummary()

    line = summary_lines[-1]
    counts = {"passed": 0, "failed": 0, "skipped": 0, "errors": 0}
    for match in _SUMMARY_TOKEN_RE.finditer(line):
        kind = match.group("kind")
        if kind == "error":
            kind = "errors"
        counts[kind] = int(match.group("count"))
    return PytestSummary(**counts)


def assert_live_postgres_report_succeeded(report_text: str, *, min_passed: int = 1) -> PytestSummary:
    """Fail closed if a live PostgreSQL run skipped, failed, errored, or ran no tests."""
    summary = parse_pytest_summary(report_text)
    if summary.passed < min_passed:
        raise PostgresCiVerificationError(
            f"Live PostgreSQL verification ran too few tests: passed={summary.passed}, min_passed={min_passed}."
        )
    if summary.skipped:
        raise PostgresCiVerificationError(
            f"Live PostgreSQL verification must not skip tests when explicitly requested: skipped={summary.skipped}."
        )
    if summary.failed or summary.errors:
        raise PostgresCiVerificationError(
            f"Live PostgreSQL verification failed: failed={summary.failed}, errors={summary.errors}."
        )
    return summary


def validate_live_postgres_environment(env: Mapping[str, str] | None = None) -> str:
    """Return DATABASE_URL only when live PostgreSQL verification was explicitly requested."""
    current_env = os.environ if env is None else env
    if current_env.get("RUN_POSTGRES_TESTS") != "1":
        raise PostgresCiVerificationError("Set RUN_POSTGRES_TESTS=1 to run live PostgreSQL verification.")

    database_url = current_env.get("DATABASE_URL", "").strip()
    if not database_url:
        raise PostgresCiVerificationError("DATABASE_URL is required for live PostgreSQL verification.")

    parsed = urlparse(database_url)
    if parsed.scheme not in {"postgresql", "postgresql+psycopg", "postgres"}:
        raise PostgresCiVerificationError(
            f"DATABASE_URL must use a PostgreSQL scheme for live verification; got {parsed.scheme!r}."
        )
    return database_url
