from __future__ import annotations

import argparse
import asyncio

import pytest

from cornerstone.config import RuntimeConfigError, Settings
from cornerstone.main import create_app
from cornerstone.store import InMemoryStore
from cornerstone.workers.sync_worker import _run_cli

pytestmark = pytest.mark.unit


def safe_production_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "app_env": "production",
        "production_mode": True,
        "persistence_backend": "postgres",
        "database_url": "postgresql+psycopg://svc_cornerstone:strong-password@postgres.internal:5432/cornerstone",
        "verify_postgres_extensions_on_startup": True,
        "postgres_required_extensions_raw": "pgcrypto,citext,vector",
        "connector_encryption_secret": "prod-secret-with-at-least-thirty-two-chars",
        "connector_oauth_callback_url": "https://cornerstone.example.com/v1/oauth/notion/callback",
        "notion_mock_external_api": False,
        "notion_client_id": "notion-prod-client-id",
        "notion_client_secret": "notion-prod-client-secret",
        "google_drive_mock_external_api": False,
        "google_drive_client_id": "google-drive-prod-client-id",
        "google_drive_client_secret": "google-drive-prod-client-secret",
        "authorized_reviewers_raw": "reviewer@company.internal,ops@company.internal",
    }
    defaults.update(overrides)
    return Settings(**defaults)


def test_local_default_runtime_config_is_allowed_for_development() -> None:
    settings = Settings()

    assert settings.production_mode is False
    assert settings.runtime_config_issues() == []


def test_production_mode_reports_all_unsafe_default_config_values() -> None:
    settings = Settings(production_mode=True)

    issue_codes = {issue.code for issue in settings.runtime_config_issues()}

    assert "production_mode_requires_production_app_env" in issue_codes
    assert "production_requires_postgres" in issue_codes
    assert "production_database_url_must_not_be_local_default" in issue_codes
    assert "production_requires_non_default_connector_secret" in issue_codes
    assert "production_connector_secret_too_short" in issue_codes
    assert "production_disallows_mock_notion_api" in issue_codes
    assert "production_requires_real_notion_client_id" in issue_codes
    assert "production_requires_real_notion_client_secret" in issue_codes
    assert "production_requires_real_google_drive_client_secret" in issue_codes
    assert "production_requires_real_google_drive_client_id" in issue_codes
    assert "production_rejects_google_drive_mock_provider" in issue_codes
    assert "production_oauth_callback_must_not_be_localhost" in issue_codes
    assert "production_oauth_callback_requires_https" in issue_codes
    assert "production_reviewers_must_not_be_placeholders" in issue_codes


def test_safe_production_runtime_config_has_no_issues() -> None:
    settings = safe_production_settings()

    assert settings.runtime_config_issues() == []


def test_create_app_fails_closed_when_production_config_is_unsafe() -> None:
    with pytest.raises(RuntimeConfigError) as exc_info:
        create_app(settings=Settings(production_mode=True), validate_runtime_config=True)

    assert "production_requires_postgres" in str(exc_info.value)
    assert "production_disallows_mock_notion_api" in str(exc_info.value)


def test_create_app_can_bypass_runtime_validation_for_explicit_test_store() -> None:
    app = create_app(
        store=InMemoryStore(),
        settings=Settings(production_mode=True),
        validate_runtime_config=False,
    )

    assert app.title == "Cornerstone Backend API"


def test_worker_cli_fails_closed_when_production_env_is_unsafe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PRODUCTION_MODE", "true")
    monkeypatch.setenv("APP_ENV", "production")
    monkeypatch.setenv("PERSISTENCE_BACKEND", "memory")
    args = argparse.Namespace(
        once=True,
        iterations=None,
        max_jobs=1,
        run_scheduler=False,
        include_not_ready=False,
        include_not_due=False,
        sleep_seconds=0.0,
    )

    with pytest.raises(RuntimeConfigError) as exc_info:
        asyncio.run(_run_cli(args))

    assert "production_requires_postgres" in str(exc_info.value)


def test_production_rejects_missing_required_postgres_extension() -> None:
    settings = safe_production_settings(postgres_required_extensions_raw="pgcrypto,citext")

    issue_codes = {issue.code for issue in settings.runtime_config_issues()}

    assert "production_requires_postgres_extensions" in issue_codes
