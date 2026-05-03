from __future__ import annotations

from dataclasses import dataclass
from typing import Any

PROJECT_ROOT_MARKERS = ("pyproject.toml", "alembic.ini", "docker-compose.yml")
DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 10.0


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    ok: bool
    detail: str
    required: bool = True


@dataclass(frozen=True)
class Column:
    key: str
    label: str
    width: int = 18


@dataclass(frozen=True)
class ProofCheck:
    name: str
    status: str
    detail: str
    category: str = "general"
    command: str | None = None
    exit_code: int | None = None
    duration_seconds: float = 0.0
    payload: Any | None = None
