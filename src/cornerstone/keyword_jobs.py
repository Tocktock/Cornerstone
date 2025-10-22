"""Asynchronous queue for executing keyword runs in background workers."""

from __future__ import annotations

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable, Dict, List, Optional

from .projects import KeywordRunRecord, ProjectStore
from .observability import MetricsRecorder


KeywordRunExecutor = Callable[["KeywordRunJob"], Awaitable[KeywordRunRecord]]


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


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
        max_concurrency_per_project: int | None = 1,
        executor: KeywordRunExecutor | None = None,
        metrics: MetricsRecorder | None = None,
    ) -> None:
        self._store = project_store
        self._max_queue = max(1, max_queue)
        self._max_concurrency = max(1, max_concurrency)
        if max_concurrency_per_project is None or max_concurrency_per_project <= 0:
            self._max_concurrency_per_project: int | None = None
        else:
            self._max_concurrency_per_project = max(1, max_concurrency_per_project)
        self._queue: asyncio.Queue[str] = asyncio.Queue(maxsize=self._max_queue)
        self._jobs: Dict[str, KeywordRunJob] = {}
        self._lock = asyncio.Lock()
        self._project_semaphore_lock = asyncio.Lock()
        self._project_semaphores: Dict[str, asyncio.Semaphore] = {}
        self._workers: List[asyncio.Task] = []
        self._shutdown = False
        self._executor: KeywordRunExecutor | None = executor
        self._metrics = metrics
        self._active_jobs = 0
        self._project_active_counts: defaultdict[str, int] = defaultdict(int)

    def configure_executor(self, executor: KeywordRunExecutor) -> None:
        self._executor = executor

    def configure_metrics(self, metrics: MetricsRecorder | None) -> None:
        self._metrics = metrics

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

        if self._metrics:
            self._metrics.increment("keyword.run.enqueued", project_id=project_id)

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

            project_semaphore: asyncio.Semaphore | None = None
            job_running = False
            try:
                if self._max_concurrency_per_project is not None:
                    project_semaphore = await self._acquire_project_slot(job.project_id)

                record = self._store.update_keyword_run(
                    job.project_id,
                    job.id,
                    status="running",
                    started_at=_utcnow_iso(),
                )
                job.refresh_from_record(record)
                job_running = True

                if self._metrics:
                    requested_at = _parse_iso(record.requested_at)
                    started_at = _parse_iso(record.started_at)
                    if requested_at and started_at:
                        queue_seconds = max((started_at - requested_at).total_seconds(), 0.0)
                        self._metrics.record_timing(
                            "keyword.run.queue_time",
                            queue_seconds,
                            project_id=job.project_id,
                        )
                    async with self._lock:
                        self._active_jobs += 1
                        project_active = self._project_active_counts[job.project_id] + 1
                        self._project_active_counts[job.project_id] = project_active
                    self._metrics.set_gauge(
                        "keyword.run.active",
                        float(self._active_jobs),
                        project_id=job.project_id,
                    )
                    self._metrics.set_gauge(
                        "keyword.run.active_per_project",
                        float(self._project_active_counts[job.project_id]),
                        project_id=job.project_id,
                    )

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
                if self._metrics and job_running:
                    async with self._lock:
                        self._active_jobs = max(self._active_jobs - 1, 0)
                        current_project_active = max(self._project_active_counts.get(job.project_id, 0) - 1, 0)
                        if current_project_active:
                            self._project_active_counts[job.project_id] = current_project_active
                        else:
                            self._project_active_counts.pop(job.project_id, None)
                        active_total = self._active_jobs
                    self._metrics.set_gauge(
                        "keyword.run.active",
                        float(active_total),
                        project_id=job.project_id,
                    )
                    self._metrics.set_gauge(
                        "keyword.run.active_per_project",
                        float(current_project_active),
                        project_id=job.project_id,
                    )
                self._queue.task_done()
                if project_semaphore is not None:
                    project_semaphore.release()

    async def update_job_record(self, job: KeywordRunJob, record: KeywordRunRecord) -> None:
        job.refresh_from_record(record)

    async def _acquire_project_slot(self, project_id: str) -> asyncio.Semaphore:
        async with self._project_semaphore_lock:
            semaphore = self._project_semaphores.get(project_id)
            if semaphore is None:
                semaphore = asyncio.Semaphore(self._max_concurrency_per_project)
                self._project_semaphores[project_id] = semaphore
        await semaphore.acquire()
        return semaphore


__all__ = ["KeywordRunQueue", "KeywordRunJob", "KeywordRunExecutor"]
