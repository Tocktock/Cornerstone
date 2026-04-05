from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def discover_source_root(start_cwd: Path | None = None, config_file: Path | None = None) -> str:
    cwd = (start_cwd or Path.cwd()).resolve()
    config_path = (config_file or Path(__file__)).resolve()

    search_roots: list[Path] = []
    seen: set[Path] = set()

    def add_root(path: Path) -> None:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            search_roots.append(resolved)

    add_root(cwd)
    add_root(cwd.parent)
    for parent in config_path.parents:
        add_root(parent)

    for root in search_roots:
        candidate = root / "demo_sources"
        if candidate.exists():
            return str(candidate)

    return str((cwd.parent / "demo_sources").resolve())


class Settings(BaseSettings):
    app_name: str = "Cornerstone"
    api_prefix: str = "/api/v1"
    database_url: str = "sqlite:///./data/cornerstone.db"
    source_root: str = Field(default_factory=discover_source_root)
    auto_seed_demo: bool = True
    ollama_enabled: bool = False
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_chat_model: str = "qwen3:0.6b"
    ollama_embedding_model: str = "qwen3-embedding:0.6b"
    ollama_timeout_seconds: int = 30
    answer_candidate_artifact_limit: int = 24
    answer_candidate_evidence_limit: int = 48
    answer_prompt_evidence_limit: int = 6
    answer_max_evidence: int = 12
    default_context_space_name: str = "Cornerstone"
    default_context_space_namespace: str = "cornerstone"
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
