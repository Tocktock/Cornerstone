from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .models import DEFAULT_BASE_URL

CONFIG_DIR_ENV = "CORNERSTONE_CONFIG_DIR"
CONFIG_FILE_NAME = "config.json"
DEFAULT_CONFIG: dict[str, Any] = {
    "baseUrl": DEFAULT_BASE_URL,
    "defaultReviewer": None,
    "defaultReportsDir": "reports",
}


def config_dir() -> Path:
    override = os.environ.get(CONFIG_DIR_ENV)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cornerstone"


def config_path() -> Path:
    return config_dir() / CONFIG_FILE_NAME


def load_config() -> dict[str, Any]:
    path = config_path()
    if not path.exists():
        return dict(DEFAULT_CONFIG)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid CLI config JSON at {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise RuntimeError(f"Invalid CLI config at {path}: expected an object")
    config = dict(DEFAULT_CONFIG)
    config.update(loaded)
    return config


def save_config(config: dict[str, Any]) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_config = {key: value for key, value in config.items() if value is not None}
    path.write_text(json.dumps(safe_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def set_config_value(key: str, value: str) -> Path:
    allowed = {"baseUrl", "defaultReviewer", "defaultReportsDir"}
    if key not in allowed:
        raise RuntimeError(f"Unsupported config key: {key}. Allowed: {', '.join(sorted(allowed))}")
    if key == "baseUrl":
        value = value.rstrip("/")
    config = load_config()
    config[key] = value
    return save_config(config)


def unset_config_value(key: str) -> Path:
    allowed = {"baseUrl", "defaultReviewer", "defaultReportsDir"}
    if key not in allowed:
        raise RuntimeError(f"Unsupported config key: {key}. Allowed: {', '.join(sorted(allowed))}")
    config = load_config()
    if key in config:
        config.pop(key)
    return save_config(config)


def resolve_base_url(value: str | None = None) -> str:
    return (value or str(load_config().get("baseUrl") or DEFAULT_BASE_URL)).rstrip("/")


def resolve_default_reviewer(value: str | None = None) -> str | None:
    return value or load_config().get("defaultReviewer")


def config_payload() -> dict[str, Any]:
    return {"path": str(config_path()), "config": load_config()}
