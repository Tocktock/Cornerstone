from __future__ import annotations

import pytest

from cornerstone.verification.postgres_ci import (
    PostgresCiVerificationError,
    assert_live_postgres_report_succeeded,
    parse_pytest_summary,
    validate_live_postgres_environment,
)


def test_parse_pytest_summary_uses_final_summary_line() -> None:
    report = """
    historical: ======================== 1 failed, 2 skipped in 0.10s ========================
    current: ======================== 4 passed in 1.23s ========================
    """

    summary = parse_pytest_summary(report)

    assert summary.passed == 4
    assert summary.failed == 0
    assert summary.skipped == 0
    assert summary.errors == 0
    assert summary.succeeded_without_skips is True


def test_live_postgres_report_rejects_skipped_tests_when_required() -> None:
    report = "======================== 3 passed, 1 skipped in 0.50s ========================"

    with pytest.raises(PostgresCiVerificationError, match="must not skip"):
        assert_live_postgres_report_succeeded(report, min_passed=3)


def test_live_postgres_report_rejects_no_tests() -> None:
    report = "======================== no tests ran in 0.01s ========================"

    with pytest.raises(PostgresCiVerificationError, match="too few tests"):
        assert_live_postgres_report_succeeded(report, min_passed=1)


def test_live_postgres_environment_requires_explicit_opt_in() -> None:
    env = {"DATABASE_URL": "postgresql+psycopg://cornerstone:cornerstone@localhost/cornerstone"}

    with pytest.raises(PostgresCiVerificationError, match="RUN_POSTGRES_TESTS=1"):
        validate_live_postgres_environment(env)


def test_live_postgres_environment_requires_postgres_url() -> None:
    env = {"RUN_POSTGRES_TESTS": "1", "DATABASE_URL": "sqlite+pysqlite:///:memory:"}

    with pytest.raises(PostgresCiVerificationError, match="PostgreSQL scheme"):
        validate_live_postgres_environment(env)


def test_live_postgres_environment_accepts_psycopg_url() -> None:
    url = "postgresql+psycopg://cornerstone:cornerstone@localhost:5432/cornerstone"
    env = {"RUN_POSTGRES_TESTS": "1", "DATABASE_URL": url}

    assert validate_live_postgres_environment(env) == url
