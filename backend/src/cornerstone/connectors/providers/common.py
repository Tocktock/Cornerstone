from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import datetime
from typing import Any

from cornerstone.schemas import ConnectorError


def parse_datetime(value: object) -> datetime | None:
    """Parse provider ISO timestamps that may use a trailing Z UTC marker."""
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def stable_hash(value: dict[str, Any]) -> str:
    """Return a deterministic metadata hash shared by connector mappers."""
    payload = json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def connector_error_from_api_response(map_provider_error: Callable[..., ConnectorError], error: Any) -> ConnectorError:
    """Bridge provider-specific API error objects into shared connector error mapping."""
    return map_provider_error(
        status_code=error.status_code,
        provider_code=error.provider_code,
        technical_message=error.message,
        retry_after_seconds=error.retry_after_seconds,
    )
