from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _discover_repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def discover_fixture_root() -> str:
    return str((_discover_repo_root() / "backend" / "fixtures").resolve())


def discover_workspace_source_root() -> str:
    return str(
        (Path(discover_fixture_root()) / "minimal" / "workspace" / "member-visible").resolve()
    )


def discover_personal_source_root() -> str:
    return str(
        (Path(discover_fixture_root()) / "minimal" / "personal" / "member-private").resolve()
    )


class Settings(BaseSettings):
    app_name: str = "Cornerstone"
    api_prefix: str = "/api/v1"
    contract_version: str = "2026-04-p0"

    database_url: str = "postgresql+psycopg://cornerstone:cornerstone@localhost:5432/cornerstone"
    auto_seed_demo: bool = True
    reset_database_on_start: bool = False

    fixture_root: str = Field(default_factory=discover_fixture_root)
    workspace_source_root: str = Field(default_factory=discover_workspace_source_root)
    personal_source_root: str = Field(default_factory=discover_personal_source_root)
    corpus_source_root: str = "sample-data/sendy-knowledge"

    fixed_now: str | None = None
    freshness_target_hours: int = 24
    source_stale_after_hours: int = 48
    source_drift_after_hours: int = 96

    default_workspace_name: str = "Cornerstone Workspace"
    default_workspace_slug: str = "cornerstone"
    default_personal_name: str = "Member Personal Context"
    default_personal_slug: str = "cornerstone-member-personal"
    default_sync_interval_seconds: int = 300

    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
        ]
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="CORNERSTONE_",
        case_sensitive=False,
        extra="ignore",
    )
