from __future__ import annotations

import json
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from datetime import date, datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

LOGGER_NAME = "cornerstone"
logger = logging.getLogger(LOGGER_NAME)


class StructuredLogError(RuntimeError):
    """Raised when structured logging cannot serialize a payload."""


def configure_logging(level: str = "INFO") -> None:
    """Configure a minimal application logger.

    The emitted message is already JSON, so a plain formatter keeps runtime logs easy to parse
    while pytest can still capture the same events with caplog.
    """

    logger.setLevel(level.upper())
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    logger.propagate = True


def log_event(event: str, **fields: Any) -> None:
    """Emit one structured JSON log event.

    Log events intentionally use stable names because tests and dashboards can assert on them.
    """

    payload: dict[str, Any] = {
        "event": event,
        "schemaVersion": 1,
        **fields,
    }
    try:
        logger.info(json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":")))
    except TypeError as exc:  # pragma: no cover - defensive guard for future fields
        raise StructuredLogError(f"Could not serialize log event {event!r}.") from exc


def parse_log_message(message: str) -> dict[str, Any]:
    """Parse a Cornerstone JSON log message for tests and tooling."""

    parsed = json.loads(message)
    if not isinstance(parsed, dict):
        raise ValueError("Structured log message must decode to an object.")
    return parsed


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Add request IDs and emit request completion/failure logs."""

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = request.headers.get("X-Request-Id") or str(uuid4())
        started_at = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception as exc:  # pragma: no cover - defensive runtime behavior
            duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
            log_event(
                "http.request.failed",
                requestId=request_id,
                method=request.method,
                path=request.url.path,
                durationMs=duration_ms,
                errorType=type(exc).__name__,
            )
            raise

        duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
        response.headers["X-Request-Id"] = request_id
        log_event(
            "http.request.completed",
            requestId=request_id,
            method=request.method,
            path=request.url.path,
            statusCode=response.status_code,
            durationMs=duration_ms,
        )
        return response


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set):
        return [_json_safe(item) for item in value]
    return value
