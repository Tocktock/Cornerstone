"""Asynchronous helpers for keyword insight summarisation."""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Sequence
from uuid import uuid4

from .config import Settings
from .keywords import KeywordCandidate, KeywordLLMFilter


_DEFAULT_MAX_JOBS = 64


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class KeywordInsightJob:
    """Track the lifecycle for a queued insight summarisation request."""

    id: str
    project_id: str
    created_at: datetime = field(default_factory=_utcnow)
    status: str = "pending"
    started_at: datetime | None = None
    completed_at: datetime | None = None
    insights: list[dict[str, object]] | None = None
    debug: dict[str, object] | None = None
    error: str | None = None
    _event: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False)

    async def wait(self, timeout: float | None = None) -> bool:
        """Wait for the job to finish.

        Returns ``True`` if the job completed within the timeout, otherwise ``False``.
        """

        if self.status in {"success", "error"}:
            return True
        if timeout is not None and timeout <= 0:
            return False
        try:
            await asyncio.wait_for(self._event.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def mark_running(self) -> None:
        if self.started_at is None:
            self.started_at = _utcnow()
        self.status = "running"

    def mark_success(self, insights: list[dict[str, object]] | None, debug: dict[str, object] | None) -> None:
        self.status = "success"
        self.insights = insights or []
        self.debug = debug or {}
        self.completed_at = _utcnow()
        self._event.set()

    def mark_error(self, message: str, debug: dict[str, object] | None = None) -> None:
        self.status = "error"
        self.error = message
        if debug:
            self.debug = debug
        self.completed_at = _utcnow()
        self._event.set()

    def to_payload(self, *, include_result: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "id": self.id,
            "projectId": self.project_id,
            "status": self.status,
            "createdAt": self.created_at.isoformat(),
        }
        if self.started_at:
            payload["startedAt"] = self.started_at.isoformat()
        if self.completed_at:
            payload["completedAt"] = self.completed_at.isoformat()
        if self.error:
            payload["error"] = self.error
        if include_result and self.insights is not None:
            payload["insights"] = self.insights
        if self.debug:
            payload["debug"] = self.debug
        return payload


class KeywordInsightQueue:
    """Lightweight queue to offload Stage 7 insight generation."""

    def __init__(self, *, max_jobs: int = _DEFAULT_MAX_JOBS) -> None:
        self._jobs: dict[str, KeywordInsightJob] = {}
        self._order: deque[str] = deque()
        self._max_jobs = max(1, max_jobs)
        self._lock = asyncio.Lock()

    async def enqueue(
        self,
        *,
        project_id: str,
        settings: Settings,
        keywords: Sequence[KeywordCandidate],
        max_insights: int,
        max_concepts: int,
        context_snippets: Iterable[str],
    ) -> KeywordInsightJob:
        job = KeywordInsightJob(id=uuid4().hex, project_id=project_id)
        async with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            while len(self._order) > self._max_jobs:
                stale_id = self._order.popleft()
                if stale_id != job.id:
                    self._jobs.pop(stale_id, None)

        loop = asyncio.get_running_loop()
        loop.create_task(
            self._run_job(
                job,
                settings=settings,
                keywords=list(keywords),
                max_insights=max_insights,
                max_concepts=max_concepts,
                context_snippets=list(context_snippets),
            )
        )
        return job

    async def get(self, job_id: str) -> KeywordInsightJob | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def list_for_project(self, project_id: str) -> list[KeywordInsightJob]:
        async with self._lock:
            return [job for job in self._jobs.values() if job.project_id == project_id]

    async def _run_job(
        self,
        job: KeywordInsightJob,
        *,
        settings: Settings,
        keywords: list[KeywordCandidate],
        max_insights: int,
        max_concepts: int,
        context_snippets: list[str],
    ) -> None:
        job.mark_running()

        def _worker() -> tuple[list[dict[str, object]], dict[str, object]]:
            filter_instance = KeywordLLMFilter(settings)
            insights = filter_instance.summarize_keywords(
                keywords,
                max_insights=max_insights,
                max_concepts=max_concepts,
                context_snippets=context_snippets,
            )
            debug = filter_instance.insight_debug_payload()
            return insights, debug

        try:
            insights, debug = await asyncio.to_thread(_worker)
        except Exception as exc:  # pragma: no cover - defensive guard
            job.mark_error(str(exc))
            return

        job.mark_success(insights, debug)


__all__ = ["KeywordInsightJob", "KeywordInsightQueue"]

