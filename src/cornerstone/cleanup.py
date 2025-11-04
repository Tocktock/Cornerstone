from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, TYPE_CHECKING
from uuid import uuid4

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from .fts import FTSIndex
    from .ingestion import ProjectVectorStoreManager
    from .projects import ProjectStore


logger = logging.getLogger(__name__)

_CLEANUP_STEPS: List[tuple[str, str]] = [
    ("purge_vectors", "Vector store"),
    ("delete_search_index", "Search index"),
    ("clear_documents", "Document records"),
    ("remove_manifest", "Manifest file"),
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class CleanupJobStep:
    name: str
    label: str
    status: str = "pending"
    error: str | None = None
    details: str | None = None
    started_at: str | None = None
    finished_at: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "status": self.status,
            "error": self.error,
            "details": self.details,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


@dataclass
class CleanupJob:
    id: str
    project_id: str
    status: str
    created_at: str
    updated_at: str
    error: str | None = None
    steps: list[CleanupJobStep] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "steps": [step.to_dict() for step in self.steps],
        }

    def get_step(self, name: str) -> CleanupJobStep | None:
        for step in self.steps:
            if step.name == name:
                return step
        return None


class CleanupJobManager:
    """Track asynchronous cleanup jobs per project."""

    def __init__(self, *, max_active_per_project: int = 1) -> None:
        self._jobs: Dict[str, CleanupJob] = {}
        self._project_jobs: Dict[str, list[str]] = {}
        self._project_active: Dict[str, int] = {}
        self._max_active_per_project = max(1, max_active_per_project)
        self._lock = threading.Lock()

    def create_job(self, project_id: str) -> CleanupJob:
        with self._lock:
            active = self._project_active.get(project_id, 0)
            if active >= self._max_active_per_project:
                raise RuntimeError("Cleanup job already running for project")
            job = CleanupJob(
                id=uuid4().hex,
                project_id=project_id,
                status="queued",
                created_at=_now(),
                updated_at=_now(),
                steps=[CleanupJobStep(name=name, label=label) for name, label in _CLEANUP_STEPS],
            )
            self._jobs[job.id] = job
            self._project_jobs.setdefault(project_id, []).insert(0, job.id)
            self._project_active[project_id] = active + 1
            return job

    def get(self, job_id: str) -> CleanupJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_for_project(self, project_id: str) -> list[CleanupJob]:
        with self._lock:
            ids = list(self._project_jobs.get(project_id, ()))
            return [self._jobs[job_id] for job_id in ids if job_id in self._jobs]

    def latest_for_project(self, project_id: str) -> CleanupJob | None:
        with self._lock:
            ids = self._project_jobs.get(project_id)
            if not ids:
                return None
            for job_id in ids:
                job = self._jobs.get(job_id)
                if job:
                    return job
            return None

    def get_active_job(self, project_id: str) -> CleanupJob | None:
        with self._lock:
            ids = self._project_jobs.get(project_id, ())
            for job_id in ids:
                job = self._jobs.get(job_id)
                if not job:
                    continue
                if job.status not in {"completed", "failed"}:
                    return job
            return None

    def mark_running(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.status = "running"
            job.updated_at = _now()

    def mark_completed(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.status = "completed"
            job.updated_at = _now()
            project_id = job.project_id
            active = self._project_active.get(project_id, 0)
            if active > 0:
                self._project_active[project_id] = active - 1

    def mark_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.status = "failed"
            job.error = error
            job.updated_at = _now()
            project_id = job.project_id
            active = self._project_active.get(project_id, 0)
            if active > 0:
                self._project_active[project_id] = active - 1

    def mark_step_running(self, job_id: str, step_name: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            step = job.get_step(step_name)
            if not step:
                return
            step.status = "running"
            if not step.started_at:
                step.started_at = _now()
            job.updated_at = _now()

    def mark_step_completed(self, job_id: str, step_name: str, *, details: str | None = None) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            step = job.get_step(step_name)
            if not step:
                return
            step.status = "completed"
            step.details = details
            step.finished_at = _now()
            job.updated_at = _now()

    def mark_step_failed(self, job_id: str, step_name: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            step = job.get_step(step_name)
            if not step:
                return
            step.status = "failed"
            step.error = error
            step.finished_at = _now()
            job.updated_at = _now()


def run_cleanup_job(
    manager: CleanupJobManager,
    job_id: str,
    project_id: str,
    store_manager: ProjectVectorStoreManager,
    fts_index: FTSIndex,
    project_store: ProjectStore,
    data_dir: Path,
) -> None:
    """Execute the cleanup steps, updating job status as progress is made."""

    try:
        manager.mark_running(job_id)
        logger.info("knowledge.cleanup.job.started project=%s job=%s", project_id, job_id)

        manager.mark_step_running(job_id, "purge_vectors")
        try:
            store_manager.purge_project(project_id)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("knowledge.cleanup.job.purge_failed project=%s", project_id)
            manager.mark_step_failed(job_id, "purge_vectors", str(exc))
            manager.mark_failed(job_id, f"Vector store purge failed: {exc}")
            return
        manager.mark_step_completed(job_id, "purge_vectors")

        manager.mark_step_running(job_id, "delete_search_index")
        try:
            fts_index.delete_project(project_id)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("knowledge.cleanup.job.fts_failed project=%s", project_id)
            manager.mark_step_failed(job_id, "delete_search_index", str(exc))
            manager.mark_failed(job_id, f"Search index cleanup failed: {exc}")
            return
        manager.mark_step_completed(job_id, "delete_search_index")

        manager.mark_step_running(job_id, "clear_documents")
        try:
            project_store.clear_documents(project_id)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("knowledge.cleanup.job.documents_failed project=%s", project_id)
            manager.mark_step_failed(job_id, "clear_documents", str(exc))
            manager.mark_failed(job_id, f"Document store cleanup failed: {exc}")
            return
        manager.mark_step_completed(job_id, "clear_documents")

        manager.mark_step_running(job_id, "remove_manifest")
        manifest_path = Path(data_dir) / "manifests" / f"{project_id}.json"
        try:
            if manifest_path.exists():
                manifest_path.unlink()
                manager.mark_step_completed(job_id, "remove_manifest", details="removed")
            else:
                manager.mark_step_completed(job_id, "remove_manifest", details="already absent")
        except OSError as exc:
            logger.warning(
                "knowledge.cleanup.job.manifest_remove_failed project=%s path=%s error=%s",
                project_id,
                manifest_path,
                exc,
            )
            manager.mark_step_failed(job_id, "remove_manifest", str(exc))
            manager.mark_failed(job_id, f"Manifest removal failed: {exc}")
            return

        manager.mark_completed(job_id)
        logger.info("knowledge.cleanup.job.completed project=%s job=%s", project_id, job_id)
    except Exception as exc:  # pragma: no cover - defensive guard
        logger.exception("knowledge.cleanup.job.unexpected_error project=%s job=%s", project_id, job_id)
        manager.mark_failed(job_id, f"Unexpected cleanup failure: {exc}")
