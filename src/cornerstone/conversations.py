"""Conversation logging and analytics primitives."""

from __future__ import annotations

import json
import logging
import math
import re
import threading
from collections import Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence, TYPE_CHECKING, Union
from uuid import uuid4

from .projects import Project
from .personas import PersonaSnapshot

if TYPE_CHECKING:  # pragma: no cover - type hints only
    from .observability import MetricsRecorder

logger = logging.getLogger(__name__)

_EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_PHONE_PATTERN = re.compile(r"(?:(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\d{2,4})[\s.-]?){2,4}\d{2,4}")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_SENSITIVE_REPLACEMENTS = {
    _EMAIL_PATTERN: "[email]",
    _PHONE_PATTERN: "[phone]",
}


@dataclass(slots=True)
class ConversationRecord:
    """Persisted summary of a single chat interaction."""

    id: str
    project_id: str
    project_name: str
    persona_id: str | None
    persona_name: str | None
    created_at: str
    query: str
    history: list[str]
    response: str
    source_count: int
    sources: list[dict[str, str]]
    prompt_tokens: int
    completion_tokens: int
    backend: str | None
    duration_ms: float | None
    resolved: bool
    unanswered: bool
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationRecord":
        return cls(
            id=data.get("id", uuid4().hex),
            project_id=data.get("project_id", "unknown"),
            project_name=data.get("project_name", ""),
            persona_id=data.get("persona_id"),
            persona_name=data.get("persona_name"),
            created_at=data.get("created_at", _utc_now().isoformat()),
            query=data.get("query", ""),
            history=[str(item) for item in data.get("history", [])],
            response=data.get("response", ""),
            source_count=int(data.get("source_count", 0)),
            sources=[_coerce_source(item) for item in data.get("sources", [])],
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            backend=data.get("backend"),
            duration_ms=data.get("duration_ms"),
            resolved=bool(data.get("resolved", False)),
            unanswered=bool(data.get("unanswered", False)),
            metadata=dict(data.get("metadata", {})),
        )


class ConversationLogStore:
    """File-backed persistence for conversation records."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._lock = threading.Lock()
        self._root.mkdir(parents=True, exist_ok=True)

    def append(self, record: ConversationRecord) -> None:
        path = self._path_for(record.project_id)
        payload = record.to_dict()
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                json.dump(payload, handle)
                handle.write("\n")
        logger.debug(
            "conversation.log.appended project=%s persona=%s record_id=%s",
            record.project_id,
            record.persona_id,
            record.id,
        )

    def iter_records(self, project_id: str | None = None) -> Iterator[ConversationRecord]:
        paths: Iterable[Path]
        if project_id:
            paths = [self._path_for(project_id)]
        else:
            paths = sorted(self._root.glob("*.jsonl"))
        for path in paths:
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:  # pragma: no cover - defensive guard
                        logger.warning("conversation.log.decode_failed path=%s", path)
                        continue
                    yield ConversationRecord.from_dict(data)

    def rewrite_records(self, project_id: str, records: Sequence[ConversationRecord]) -> None:
        path = self._path_for(project_id)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                for record in records:
                    json.dump(record.to_dict(), handle)
                    handle.write("\n")

    def _path_for(self, project_id: str) -> Path:
        safe_id = project_id.replace("/", "-")
        return self._root / f"{safe_id}.jsonl"


class ConversationLogger:
    """High-level helper that anonymizes and stores chat interactions."""

    def __init__(
        self,
        store: ConversationLogStore,
        *,
        enabled: bool = True,
        retention_days: int = 30,
        metrics: "MetricsRecorder" | None = None,
    ) -> None:
        self._store = store
        self._enabled = enabled
        self._retention_days = max(retention_days, 0)
        self._metrics = metrics

    @property
    def enabled(self) -> bool:
        return self._enabled

    def log_chat(
        self,
        *,
        project: Project,
        persona: PersonaSnapshot | None,
        query: str,
        response: str,
        history: Sequence[Any] | None,
        sources: Sequence[dict[str, Any]] | None,
        definitions: Sequence[str] | None = None,
        backend: str | None = None,
        duration_ms: float | None = None,
        timestamp: datetime | None = None,
    ) -> None:
        if not self._enabled:
            return

        recorded_at = timestamp or _utc_now()
        sanitized_query = _sanitize_text(query)
        sanitized_history = [_sanitize_text(_extract_history_text(item)) for item in (history or []) if item is not None]
        sanitized_response = _sanitize_text(response)
        sanitized_sources = [_sanitize_source(item) for item in (sources or [])]
        sanitized_definitions = [_sanitize_text(item) for item in (definitions or [])]

        prompt_tokens = _estimate_prompt_tokens(sanitized_query, sanitized_history, sanitized_definitions)
        completion_tokens = _estimate_tokens(sanitized_response)
        resolved, unanswered = _resolution_flags(sanitized_response, sanitized_sources)

        record = ConversationRecord(
            id=uuid4().hex,
            project_id=project.id,
            project_name=project.name,
            persona_id=persona.id if persona else None,
            persona_name=_sanitize_text(persona.name) if persona and persona.name else None,
            created_at=recorded_at.isoformat(),
            query=sanitized_query,
            history=sanitized_history,
            response=sanitized_response,
            source_count=len(sanitized_sources),
            sources=sanitized_sources,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            backend=backend,
            duration_ms=round(duration_ms, 4) if duration_ms is not None else None,
            resolved=resolved,
            unanswered=unanswered,
            metadata={
                "definitions": sanitized_definitions,
            },
        )

        self._store.append(record)
        metrics = self._metrics
        if metrics:
            metrics.increment(
                "chat.logged_conversations",
                project_id=project.id,
                persona_id=persona.id if persona and persona.id else None,
            )
            if resolved:
                metrics.increment("chat.logged_resolved", project_id=project.id)
            if unanswered:
                metrics.increment("chat.logged_unanswered", project_id=project.id)
            total_tokens = record.prompt_tokens + record.completion_tokens
            if total_tokens > 0:
                metrics.increment("chat.logged_tokens", value=total_tokens, project_id=project.id)
            if duration_ms is not None:
                metrics.record_timing(
                    "chat.response_latency",
                    max(duration_ms, 0.0) / 1000.0,
                    project_id=project.id,
                )
        if self._retention_days > 0:
            cutoff = recorded_at - timedelta(days=self._retention_days)
            self._enforce_retention(project.id, cutoff)

    def list_conversations(
        self,
        *,
        project_id: str | None = None,
        limit: int | None = None,
    ) -> list[ConversationRecord]:
        records = list(self._store.iter_records(project_id))
        records.sort(key=lambda item: item.created_at, reverse=True)
        if limit is not None and limit >= 0:
            return records[:limit]
        return records

    def _enforce_retention(self, project_id: str, cutoff: datetime) -> None:
        retained: list[ConversationRecord] = []
        for record in self._store.iter_records(project_id):
            created = _coerce_datetime(record.created_at)
            if created is None or created >= cutoff:
                retained.append(record)
        if len(retained) == 0:
            path = self._store._path_for(project_id)
            with self._store._lock:
                if path.exists():
                    path.unlink()
            return
        self._store.rewrite_records(project_id, retained)


class AnalyticsService:
    """Compute aggregate analytics from stored conversation records."""

    def __init__(self, logger: ConversationLogger) -> None:
        self._logger = logger

    def build_summary(
        self,
        *,
        project_id: str | None = None,
        days: int = 30,
        top_limit: int = 10,
        unanswered_limit: int = 10,
    ) -> dict[str, Any]:
        records = self._logger.list_conversations(project_id=project_id)
        cutoff_date = None
        if days > 0:
            cutoff_date = (_utc_now() - timedelta(days=days - 1)).date()

        daily_stats: dict[str, dict[str, Any]] = {}
        top_queries: Counter[str] = Counter()
        unanswered_records: list[tuple[datetime, ConversationRecord]] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_conversations = 0
        resolved_conversations = 0
        durations: list[float] = []

        for record in records:
            created = _coerce_datetime(record.created_at)
            if created is None:
                continue
            created_date = created.date()
            if cutoff_date and created_date < cutoff_date:
                continue

            date_key = created_date.isoformat()
            stats = daily_stats.setdefault(
                date_key,
                {"date": date_key, "total": 0, "resolved": 0, "unanswered": 0},
            )
            stats["total"] += 1
            total_conversations += 1
            if record.resolved:
                stats["resolved"] += 1
                resolved_conversations += 1
            if record.unanswered:
                stats["unanswered"] += 1
                unanswered_records.append((created, record))

            top_queries[record.query] += 1
            total_prompt_tokens += max(record.prompt_tokens, 0)
            total_completion_tokens += max(record.completion_tokens, 0)
            if record.duration_ms is not None:
                durations.append(max(float(record.duration_ms), 0.0))

        resolution_rate = (
            resolved_conversations / total_conversations if total_conversations else 0.0
        )

        top_query_list = [
            {"query": query, "count": count}
            for query, count in top_queries.most_common(max(top_limit, 1))
        ]

        unanswered_records.sort(key=lambda item: item[0], reverse=True)
        unanswered_list = [
            {
                "query": record.query,
                "response": record.response,
                "created_at": created.isoformat(),
                "persona_name": record.persona_name,
            }
            for created, record in unanswered_records[: max(unanswered_limit, 1)]
        ]

        duration_avg = sum(durations) / len(durations) if durations else 0.0

        summary = {
            "generated_at": _utc_now().isoformat(),
            "project_id": project_id,
            "days": max(days, 0),
            "totals": {
                "conversations": total_conversations,
                "resolved": resolved_conversations,
                "resolution_rate": round(resolution_rate, 4),
                "avg_duration_ms": round(duration_avg, 3),
            },
            "token_usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_prompt_tokens + total_completion_tokens,
            },
            "daily_counts": sorted(daily_stats.values(), key=lambda item: item["date"]),
            "top_queries": top_query_list,
            "unanswered": unanswered_list,
        }
        return summary


def _sanitize_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value)
    if not text:
        return ""
    for pattern, replacement in _SENSITIVE_REPLACEMENTS.items():
        text = pattern.sub(replacement, text)
    text = _WHITESPACE_PATTERN.sub(" ", text)
    return text.strip()


def _sanitize_source(data: dict[str, Any]) -> dict[str, str]:
    sanitized: dict[str, str] = {}
    for key, value in (data or {}).items():
        if value is None:
            continue
        sanitized[str(key)] = _sanitize_text(value)
    return sanitized


def _coerce_source(data: Any) -> dict[str, str]:
    if isinstance(data, dict):
        return {str(key): _sanitize_text(value) for key, value in data.items()}
    return {"value": _sanitize_text(data)}


def _extract_history_text(item: Any) -> str:
    if isinstance(item, dict):
        content = item.get("content")
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False)
    if isinstance(item, Union[list, tuple]):
        joined = " ".join(str(part) for part in item if part is not None)
        return joined
    return str(item)


def _estimate_tokens(text: str) -> int:
    text = text.strip()
    if not text:
        return 0
    estimate = math.ceil(len(text) / 4)
    return max(1, estimate)


def _estimate_prompt_tokens(query: str, history: Sequence[str], definitions: Sequence[str]) -> int:
    total = _estimate_tokens(query)
    for item in history:
        total += _estimate_tokens(item)
    for definition in definitions:
        total += _estimate_tokens(definition)
    return total


def _resolution_flags(response: str, sources: Sequence[dict[str, str]]) -> tuple[bool, bool]:
    normalized = response.lower()
    fallback_markers = ["could not", "sorry", "no relevant", "unable to"]
    has_content = bool(response.strip())
    has_sources = any(bool(item) for item in sources)
    unanswered = (not has_content) or (not has_sources) or any(marker in normalized for marker in fallback_markers)
    resolved = has_content and not unanswered
    return resolved, unanswered


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _coerce_datetime(value: str | datetime | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:  # pragma: no cover - defensive guard
        return None
