"""Helpers for automatically enqueuing keyword runs when projects change."""

from __future__ import annotations

import asyncio
import logging
import threading
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .config import Settings
from .keyword_jobs import KeywordRunQueue


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _AutoRefreshState:
    active: bool = False
    pending: bool = False


class KeywordRunAutoRefresher:
    """Schedule keyword run jobs after project ingestions mark them dirty."""

    def __init__(
        self,
        *,
        settings: Settings,
        queue: KeywordRunQueue,
        logger: logging.Logger | None = None,
    ) -> None:
        self._settings = settings
        self._queue = queue
        self._logger = logger or logging.getLogger("cornerstone.keyword_refresh")
        self._loop: asyncio.AbstractEventLoop | None = None
        self._states: dict[str, _AutoRefreshState] = {}
        self._state_lock = threading.Lock()
        self._pending_until_loop: List[Tuple[str, Optional[str]]] = []

    @property
    def enabled(self) -> bool:
        return self._settings.keyword_run_async_enabled and self._settings.keyword_run_auto_refresh

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        if not self.enabled:
            return
        with self._state_lock:
            self._loop = loop
            pending = list(self._pending_until_loop)
            self._pending_until_loop.clear()
        for project_id, requested_by in pending:
            self._schedule_enqueue(project_id, requested_by=requested_by)

    def mark_project_dirty(self, project_id: str, *, requested_by: str | None = None) -> None:
        if not self.enabled:
            return
        should_schedule = False
        with self._state_lock:
            state = self._states.get(project_id)
            if state is None:
                state = _AutoRefreshState(active=True, pending=False)
                self._states[project_id] = state
                should_schedule = True
            elif state.active:
                state.pending = True
            else:
                state.active = True
                state.pending = False
                should_schedule = True
        if should_schedule:
            self._schedule_enqueue(project_id, requested_by=requested_by)

    def _schedule_enqueue(self, project_id: str, *, requested_by: str | None = None) -> None:
        if not self.enabled:
            return

        async def runner() -> None:
            await self._enqueue_and_watch(project_id, requested_by=requested_by)

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = self._loop
            if loop is None:
                with self._state_lock:
                    self._pending_until_loop.append((project_id, requested_by))
                return
            loop.call_soon_threadsafe(
                lambda: asyncio.create_task(
                    runner(),
                    name=f"keyword-auto-refresh-{project_id}",
                )
            )
        else:
            loop.create_task(
                runner(),
                name=f"keyword-auto-refresh-{project_id}",
            )

    async def _enqueue_and_watch(self, project_id: str, *, requested_by: str | None) -> None:
        try:
            job = await self._queue.enqueue(project_id, requested_by=requested_by)
        except Exception as exc:  # pragma: no cover - defensive guard
            self._logger.warning(
                "keyword.auto_refresh.enqueue_failed project=%s error=%s",
                project_id,
                exc,
            )
            reschedule = False
            with self._state_lock:
                state = self._states.get(project_id)
                if state:
                    state.active = False
                    if state.pending:
                        state.pending = False
                        state.active = True
                        reschedule = True
                    else:
                        self._states.pop(project_id, None)
            if reschedule:
                self._schedule_enqueue(project_id)
            return

        self._logger.info(
            "keyword.auto_refresh.enqueued project=%s job_id=%s",
            project_id,
            job.id,
        )

        async def _monitor() -> None:
            await job.wait()
            reschedule = False
            with self._state_lock:
                state = self._states.get(project_id)
                if state:
                    state.active = False
                    if state.pending:
                        state.pending = False
                        state.active = True
                        reschedule = True
                    else:
                        self._states.pop(project_id, None)
            if reschedule:
                self._schedule_enqueue(project_id)

        asyncio.create_task(
            _monitor(),
            name=f"keyword-auto-refresh-monitor-{project_id}",
        )


__all__ = ["KeywordRunAutoRefresher"]
