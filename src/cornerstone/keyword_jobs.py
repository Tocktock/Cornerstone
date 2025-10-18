"""Asynchronous queue for executing keyword runs in background workers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable, Dict, Iterable, List

from .projects import KeywordRunRecord, ProjectStore


KeywordRunExecutor = Callable[["KeywordRunJob"], Awaitable[KeywordRunRecord]]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class KeywordRunJob:
    """In-memory representation of a keyword run queued for execution."""

    id: str
    project_id: str
    requested_by: str | None
    record: KeywordRunRecord
    status: str = "pending"
    error: str | None = None
    _event: asyncio.Event = field(default_factory=asyncio.Event, init=False, repr=False)

    async def wait(self, timeout: float | None = None) -> bool:
        if self.status in {"success", "error"}:
            return True
        if timeout is not None and timeout <= 0:
            return False
        try:
            await asyncio.wait_for(self._event.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def refresh_from_record(self, record: KeywordRunRecord) -> None:
        self.record = record
        self.status = record.status
        if record.status in {"success", "error"}:
            self.error = record.error
            self._event.set()

    def mark_error(self, message: str) -> None:
        self.status = "error"
        self.error = message
        self._event.set()


class KeywordRunQueue:
    """Manage asynchronous processing of keyword runs with worker concurrency."""

    def __init__(
        self,
        project_store: ProjectStore,
        *,
        max_queue: int = 8,
        max_concurrency: int = 1,
        executor: KeywordRunExecutor | None = None,
    ) -> None:
        self._store = project_store
        self._max_queue = max(1, max_queue)
        self._max_concurrency = max(1, max_concurrency)
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=self._max_queue)
        self._jobs: Dict[str, KeywordRunJob] = {}
        self._lock = asyncio.Lock()
        self._workers: List[asyncio.Task] = []
        self._shutdown = False
        self._executor: KeywordRunExecutor | None = executor

    def configure_executor(self, executor: KeywordRunExecutor) -> None:
        self._executor = executor

    def has_executor(self) -> bool:
        return self._executor is not None

    async def enqueue(self, project_id: str, *, requested_by: str | None = None) -> KeywordRunJob:
        if self._shutdown:
            raise RuntimeError("KeywordRunQueue is shut down")

        if self._queue.full():
            raise RuntimeError("Keyword run queue is full")

        record = self._store.create_keyword_run(project_id, requested_by=requested_by)
        job = KeywordRunJob(
            id=record.id,
            project_id=record.project_id,
            requested_by=requested_by,
            record=record,
            status=record.status,
        )

        async with self._lock:
            self._jobs[job.id] = job

        await self._queue.put(job.id)
        return job

    async def get(self, job_id: str) -> KeywordRunJob | None:
        async with self._lock:
            return self._jobs.get(job_id)

    async def list_for_project(self, project_id: str) -> list[KeywordRunJob]:
        async with self._lock:
            return [job for job in self._jobs.values() if job.project_id == project_id]

    def start(self) -> None:
        if self._workers:
            return
        for _ in range(self._max_concurrency):
            task = asyncio.create_task(self._worker_loop())
            self._workers.append(task)

    async def shutdown(self) -> None:
        self._shutdown = True
        for task in self._workers:
            task.cancel()
        for task in self._workers:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._workers.clear()

    async def _worker_loop(self) -> None:
        while not self._shutdown:
            try:
                job_id = await self._queue.get()
            except asyncio.CancelledError:
                break

            job = await self.get(job_id)
            if job is None:
                self._queue.task_done()
                continue

            try:
                record = self._store.update_keyword_run(
                    job.project_id,
                    job.id,
                    status="running",
                    started_at=_utcnow_iso(),
                )
                job.refresh_from_record(record)

                if self._executor is None:
                    raise RuntimeError("Keyword run executor not configured")

                result = await self._executor(job)
                job.refresh_from_record(result)
            except Exception as exc:  # pragma: no cover - defensive guard
                message = str(exc)
                self._store.update_keyword_run(
                    job.project_id,
                    job.id,
                    status="error",
                    error=message,
                    completed_at=_utcnow_iso(),
                )
                job.mark_error(message)
            finally:
                self._queue.task_done()

    async def update_job_record(self, job: KeywordRunJob, record: KeywordRunRecord) -> None:
        job.refresh_from_record(record)


__all__ = ["KeywordRunQueue", "KeywordRunJob", "KeywordRunExecutor"]
