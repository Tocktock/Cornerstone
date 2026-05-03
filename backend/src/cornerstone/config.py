from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict

DEFAULT_DATABASE_URL = "postgresql+psycopg://cornerstone:cornerstone@localhost:5432/cornerstone"
DEFAULT_CONNECTOR_ENCRYPTION_SECRET = "local-dev-only-change-me-secret"
DEFAULT_AUTHORIZED_REVIEWERS_RAW = "system,reviewer@example.com"
DEFAULT_NOTION_CLIENT_ID = "local-notion-client-id"
DEFAULT_NOTION_CLIENT_SECRET = "local-notion-client-secret"
DEFAULT_GOOGLE_DRIVE_CLIENT_ID = "local-google-drive-client-id"
DEFAULT_GOOGLE_DRIVE_CLIENT_SECRET = "local-google-drive-client-secret"
DEFAULT_CONNECTOR_OAUTH_CALLBACK_URL = "http://localhost:8000/v1/oauth/notion/callback"


@dataclass(frozen=True)
class RuntimeConfigIssue:
    """One fail-closed runtime configuration problem."""

    code: str
    message: str


class RuntimeConfigError(ValueError):
    """Raised when production mode is enabled with unsafe runtime configuration."""

    def __init__(self, issues: list[RuntimeConfigIssue]) -> None:
        self.issues = tuple(issues)
        detail = "; ".join(f"{issue.code}: {issue.message}" for issue in self.issues)
        super().__init__(f"Unsafe Cornerstone runtime configuration: {detail}")


class Settings(BaseModel):
    app_env: str = "local"
    production_mode: bool = False
    api_prefix: str = "/v1"
    freshness_fresh_days: int = 7
    freshness_stale_days: int = 30
    log_level: str = "INFO"
    authorized_reviewers_raw: str = DEFAULT_AUTHORIZED_REVIEWERS_RAW

    # Manual upload ingestion
    manual_upload_max_file_count: int = 10
    manual_upload_max_file_bytes: int = 5_242_880

    # Persistence
    persistence_backend: Literal["memory", "postgres"] = "memory"
    database_url: str = DEFAULT_DATABASE_URL
    database_echo: bool = False
    postgres_required_extensions_raw: str = "pgcrypto,citext,vector"
    verify_postgres_extensions_on_startup: bool = True

    # Connector runtime
    connector_intent_ttl_minutes: int = 15
    connector_encryption_secret: str = DEFAULT_CONNECTOR_ENCRYPTION_SECRET
    connector_oauth_callback_url: str = DEFAULT_CONNECTOR_OAUTH_CALLBACK_URL
    connector_http_timeout_seconds: float = 10.0

    # Notion connector
    notion_client_id: str = DEFAULT_NOTION_CLIENT_ID
    notion_client_secret: str = DEFAULT_NOTION_CLIENT_SECRET
    google_drive_client_id: str = DEFAULT_GOOGLE_DRIVE_CLIENT_ID
    google_drive_client_secret: str = DEFAULT_GOOGLE_DRIVE_CLIENT_SECRET
    google_drive_oauth_authorize_url: str = "https://accounts.google.com/o/oauth2/v2/auth"
    google_drive_oauth_token_url: str = "https://oauth2.googleapis.com/token"
    google_drive_mock_external_api: bool = True
    google_drive_discovery_query: str = "trashed = false"
    google_drive_export_mime_type: str = "text/plain"
    google_drive_page_size: int = 25
    notion_oauth_authorize_url: str = "https://api.notion.com/v1/oauth/authorize"
    notion_oauth_token_url: str = "https://api.notion.com/v1/oauth/token"
    notion_api_base_url: str = "https://api.notion.com/v1"
    notion_version: str = "2026-03-11"
    notion_mock_external_api: bool = True
    notion_use_markdown_endpoint: bool = True
    notion_block_page_size: int = 100
    notion_max_block_depth: int = 3

    # Live ontology extraction provider. Disabled by default; live calls require
    # explicit operator configuration and never run from normal tests by accident.
    ontology_live_llm_enabled: bool = False
    ontology_live_llm_api_url: str = ""
    ontology_live_llm_api_key: str = ""
    ontology_live_llm_model_name: str = "cornerstone-live-ontology-model"
    ontology_live_llm_prompt_version: str = "ontology-extraction-v2.1.0"
    ontology_live_llm_fixture_response_json: str = ""
    ontology_live_llm_timeout_seconds: float = 20.0

    model_config = ConfigDict(extra="ignore")

    @classmethod
    def from_env(cls) -> Settings:
        values: dict[str, object] = {}
        bool_fields = {
            "production_mode",
            "database_echo",
            "verify_postgres_extensions_on_startup",
            "notion_mock_external_api",
            "notion_use_markdown_endpoint",
            "google_drive_mock_external_api",
            "ontology_live_llm_enabled",
        }
        int_fields = {
            "freshness_fresh_days",
            "freshness_stale_days",
            "connector_intent_ttl_minutes",
            "notion_block_page_size",
            "notion_max_block_depth",
            "google_drive_page_size",
            "manual_upload_max_file_count",
            "manual_upload_max_file_bytes",
        }
        float_fields = {"connector_http_timeout_seconds", "ontology_live_llm_timeout_seconds"}
        for field_name in cls.model_fields:
            env_name = field_name.upper()
            if env_name not in os.environ:
                continue
            raw_value = os.environ[env_name]
            if field_name in bool_fields:
                values[field_name] = _parse_bool(raw_value)
            elif field_name in int_fields:
                values[field_name] = int(raw_value)
            elif field_name in float_fields:
                values[field_name] = float(raw_value)
            else:
                values[field_name] = raw_value
        return cls(**cast(dict[str, Any], values))

    @property
    def authorized_reviewers(self) -> set[str]:
        return {
            reviewer.strip()
            for reviewer in self.authorized_reviewers_raw.split(",")
            if reviewer.strip()
        }

    @property
    def postgres_required_extensions(self) -> set[str]:
        return {
            extension.strip()
            for extension in self.postgres_required_extensions_raw.split(",")
            if extension.strip()
        }

    def runtime_config_issues(self) -> list[RuntimeConfigIssue]:
        """Return fail-closed production configuration issues.

        Local development remains convenient with production_mode=false. Once production mode
        is enabled, the backend refuses mock providers, in-memory persistence, default secrets,
        placeholder reviewers, and localhost/default database settings.
        """

        if not self.production_mode:
            return []

        issues: list[RuntimeConfigIssue] = []
        app_env = self.app_env.strip().lower()
        if app_env in {"local", "dev", "development", "test"}:
            issues.append(
                RuntimeConfigIssue(
                    code="production_mode_requires_production_app_env",
                    message="APP_ENV must be a production-like environment when PRODUCTION_MODE=true.",
                )
            )

        if self.persistence_backend != "postgres":
            issues.append(
                RuntimeConfigIssue(
                    code="production_requires_postgres",
                    message="PERSISTENCE_BACKEND must be 'postgres' when PRODUCTION_MODE=true.",
                )
            )

        if self.database_url == DEFAULT_DATABASE_URL or _looks_like_local_database_url(self.database_url):
            issues.append(
                RuntimeConfigIssue(
                    code="production_database_url_must_not_be_local_default",
                    message="DATABASE_URL must not use localhost, 127.0.0.1, or default cornerstone credentials in production mode.",
                )
            )

        required_extensions = {"pgcrypto", "citext", "vector"}
        missing_extensions = sorted(required_extensions - self.postgres_required_extensions)
        if missing_extensions:
            issues.append(
                RuntimeConfigIssue(
                    code="production_requires_postgres_extensions",
                    message=f"POSTGRES_REQUIRED_EXTENSIONS_RAW is missing: {', '.join(missing_extensions)}.",
                )
            )

        if not self.verify_postgres_extensions_on_startup:
            issues.append(
                RuntimeConfigIssue(
                    code="production_requires_extension_verification",
                    message="VERIFY_POSTGRES_EXTENSIONS_ON_STARTUP must remain true in production mode.",
                )
            )

        if self.connector_encryption_secret == DEFAULT_CONNECTOR_ENCRYPTION_SECRET:
            issues.append(
                RuntimeConfigIssue(
                    code="production_requires_non_default_connector_secret",
                    message="CONNECTOR_ENCRYPTION_SECRET must not use the local development placeholder.",
                )
            )
        if len(self.connector_encryption_secret) < 32:
            issues.append(
                RuntimeConfigIssue(
                    code="production_connector_secret_too_short",
                    message="CONNECTOR_ENCRYPTION_SECRET must be at least 32 characters in production mode.",
                )
            )

        if self.notion_mock_external_api:
            issues.append(
                RuntimeConfigIssue(
                    code="production_disallows_mock_notion_api",
                    message="NOTION_MOCK_EXTERNAL_API must be false in production mode.",
                )
            )

        if self.notion_client_id == DEFAULT_NOTION_CLIENT_ID or not self.notion_client_id.strip():
            issues.append(
                RuntimeConfigIssue(
                    code="production_requires_real_notion_client_id",
                    message="NOTION_CLIENT_ID must be configured for live provider access in production mode.",
                )
            )
        if self.notion_client_secret == DEFAULT_NOTION_CLIENT_SECRET or not self.notion_client_secret.strip():
            issues.append(
                RuntimeConfigIssue(
                    code="production_requires_real_notion_client_secret",
                    message="NOTION_CLIENT_SECRET must be configured for live provider access in production mode.",
                )
            )

        if self.google_drive_mock_external_api:
            issues.append(
                RuntimeConfigIssue(
                    code="production_rejects_google_drive_mock_provider",
                    message="GOOGLE_DRIVE_MOCK_EXTERNAL_API must be false in production mode.",
                )
            )

        if self.google_drive_client_id == DEFAULT_GOOGLE_DRIVE_CLIENT_ID or not self.google_drive_client_id.strip():
            issues.append(
                RuntimeConfigIssue(
                    code="production_requires_real_google_drive_client_id",
                    message="GOOGLE_DRIVE_CLIENT_ID must be configured for live provider access in production mode.",
                )
            )

        if self.google_drive_client_secret == DEFAULT_GOOGLE_DRIVE_CLIENT_SECRET or not self.google_drive_client_secret.strip():
            issues.append(
                RuntimeConfigIssue(
                    code="production_requires_real_google_drive_client_secret",
                    message="GOOGLE_DRIVE_CLIENT_SECRET must be configured for live provider access in production mode.",
                )
            )

        if _looks_like_local_url(self.connector_oauth_callback_url):
            issues.append(
                RuntimeConfigIssue(
                    code="production_oauth_callback_must_not_be_localhost",
                    message="CONNECTOR_OAUTH_CALLBACK_URL must use the deployed public HTTPS callback URL in production mode.",
                )
            )
        if not self.connector_oauth_callback_url.startswith("https://"):
            issues.append(
                RuntimeConfigIssue(
                    code="production_oauth_callback_requires_https",
                    message="CONNECTOR_OAUTH_CALLBACK_URL must use HTTPS in production mode.",
                )
            )

        reviewer_issues = _reviewer_config_issues(self.authorized_reviewers)
        issues.extend(reviewer_issues)
        return issues

    def assert_runtime_config_safe(self) -> None:
        issues = self.runtime_config_issues()
        if issues:
            raise RuntimeConfigError(issues)


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Invalid boolean environment value: {value!r}")


def _looks_like_local_database_url(database_url: str) -> bool:
    normalized = database_url.lower()
    local_markers = ("@localhost", "@127.0.0.1", "@0.0.0.0", "cornerstone:cornerstone@")
    return any(marker in normalized for marker in local_markers)


def _looks_like_local_url(url: str) -> bool:
    normalized = url.lower()
    return "localhost" in normalized or "127.0.0.1" in normalized or "0.0.0.0" in normalized


def _reviewer_config_issues(reviewers: set[str]) -> list[RuntimeConfigIssue]:
    issues: list[RuntimeConfigIssue] = []
    if not reviewers:
        return [
            RuntimeConfigIssue(
                code="production_requires_authorized_reviewers",
                message="AUTHORIZED_REVIEWERS_RAW must include at least one real reviewer identity in production mode.",
            )
        ]

    placeholder_reviewers = {"system", "reviewer@example.com", "admin@example.com"}
    configured_placeholders = sorted(reviewer for reviewer in reviewers if reviewer.lower() in placeholder_reviewers)
    example_reviewers = sorted(reviewer for reviewer in reviewers if reviewer.lower().endswith("@example.com"))
    if configured_placeholders or example_reviewers:
        issues.append(
            RuntimeConfigIssue(
                code="production_reviewers_must_not_be_placeholders",
                message="AUTHORIZED_REVIEWERS_RAW must not include system/example placeholder reviewers in production mode.",
            )
        )
    return issues


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()
