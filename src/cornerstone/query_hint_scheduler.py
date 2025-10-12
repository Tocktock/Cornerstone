"""Query hint scheduling helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ScheduledHintJob:
    project_id: str
    schedule: str
    next_run: datetime


class QueryHintScheduler:
    """Track per-project query hint refresh schedules."""

    def __init__(self, default_cron: str) -> None:
        self._default_cron = default_cron or "0 3 * * *"
        self._jobs: Dict[str, ScheduledHintJob] = {}

    def update_job(self, project_id: str, schedule: str | None, *, start: datetime | None = None) -> None:
        schedule = (schedule or "").strip().lower()
        if not schedule:
            self._jobs.pop(project_id, None)
            return
        delta = self._delta_for(schedule)
        if delta is None:
            logger.warning("query_hints.scheduler.invalid schedule=%s project=%s", schedule, project_id)
            self._jobs.pop(project_id, None)
            return
        base = start or datetime.now(timezone.utc)
        next_run = base + delta
        self._jobs[project_id] = ScheduledHintJob(project_id=project_id, schedule=schedule, next_run=next_run)

    def due_projects(self, *, now: datetime | None = None) -> Iterable[str]:
        current = now or datetime.now(timezone.utc)
        due: list[str] = []
        for project_id, job in list(self._jobs.items()):
            if job.next_run <= current:
                due.append(project_id)
                delta = self._delta_for(job.schedule) or timedelta(days=1)
                self._jobs[project_id] = ScheduledHintJob(
                    project_id=project_id,
                    schedule=job.schedule,
                    next_run=current + delta,
                )
        return due

    def _delta_for(self, schedule: str) -> timedelta | None:
        if schedule == "daily":
            return timedelta(days=1)
        if schedule == "weekly":
            return timedelta(days=7)
        if not schedule:
            return None
        return timedelta(days=1)
