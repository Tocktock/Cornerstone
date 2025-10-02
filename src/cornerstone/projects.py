"""Project and document metadata management."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Iterable, List
from uuid import uuid4

import logging


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Project:
    """Metadata describing a knowledge base project."""

    id: str
    name: str
    description: str | None
    created_at: str


@dataclass(slots=True)
class DocumentMetadata:
    """Metadata about an ingested document."""

    id: str
    filename: str
    chunk_count: int
    created_at: str
    size_bytes: int | None = None


class ProjectStore:
    """File-based repository for projects and document metadata."""

    def __init__(self, root: Path, *, default_project_name: str) -> None:
        self._root = root
        self._projects_file = root / "projects.json"
        self._documents_dir = root / "documents"
        self._root.mkdir(parents=True, exist_ok=True)
        self._documents_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_default_project(default_project_name)

    def list_projects(self) -> List[Project]:
        data = self._read_projects()
        projects = [Project(**item) for item in data.get("projects", [])]
        return sorted(projects, key=lambda project: project.created_at)

    def get_project(self, project_id: str) -> Project | None:
        for project in self.list_projects():
            if project.id == project_id:
                return project
        return None

    def find_by_name(self, name: str) -> Project | None:
        target = name.strip().lower()
        for project in self.list_projects():
            if project.name.strip().lower() == target:
                return project
        return None

    def create_project(self, name: str, description: str | None = None) -> Project:
        projects = self._read_projects()
        existing_ids = {item["id"] for item in projects.get("projects", [])}
        project_id = self._generate_project_id(name, existing_ids)
        project = Project(
            id=project_id,
            name=name,
            description=description,
            created_at=self._now(),
        )
        projects.setdefault("projects", []).append(asdict(project))
        self._write_projects(projects)
        logger.info("project.created id=%s name=%s", project.id, project.name)
        return project

    def list_documents(self, project_id: str) -> List[DocumentMetadata]:
        path = self._documents_dir / f"{project_id}.json"
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as handle:
            items = json.load(handle)
        return [DocumentMetadata(**item) for item in items]

    def record_document(self, project_id: str, metadata: DocumentMetadata) -> None:
        path = self._documents_dir / f"{project_id}.json"
        documents = []
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                documents = json.load(handle)
        documents.append(asdict(metadata))
        with path.open("w", encoding="utf-8") as handle:
            json.dump(documents, handle, indent=2)
        logger.info(
            "project.document.recorded project=%s doc_id=%s chunks=%s",
            project_id,
            metadata.id,
            metadata.chunk_count,
        )

    def remove_document(self, project_id: str, doc_id: str) -> bool:
        path = self._documents_dir / f"{project_id}.json"
        if not path.exists():
            return False
        with path.open("r", encoding="utf-8") as handle:
            documents = json.load(handle)
        filtered = [item for item in documents if item.get("id") != doc_id]
        if len(filtered) == len(documents):
            return False
        with path.open("w", encoding="utf-8") as handle:
            json.dump(filtered, handle, indent=2)
        logger.info("project.document.removed project=%s doc_id=%s", project_id, doc_id)
        return True

    def clear_documents(self, project_id: str) -> None:
        """Remove all recorded documents for the given project."""

        path = self._documents_dir / f"{project_id}.json"
        if path.exists():
            path.unlink()
            logger.info("project.documents.cleared project=%s", project_id)

    def _ensure_default_project(self, default_project_name: str) -> None:
        if self._projects_file.exists():
            return
        default_project = Project(
            id=self._generate_project_id(default_project_name, set()),
            name=default_project_name,
            description="Initial project",
            created_at=self._now(),
        )
        self._write_projects({"projects": [asdict(default_project)]})

    def _read_projects(self) -> dict:
        if not self._projects_file.exists():
            return {"projects": []}
        with self._projects_file.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _write_projects(self, data: dict) -> None:
        with self._projects_file.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    def _generate_project_id(self, name: str, existing_ids: Iterable[str]) -> str:
        base = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()) or "project"
        base = base.strip("-") or "project"
        candidate = base
        counter = 1
        while candidate in existing_ids:
            counter += 1
            candidate = f"{base}-{counter}"
        if candidate in existing_ids:
            candidate = uuid4().hex[:8]
        return candidate

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


__all__ = ["Project", "DocumentMetadata", "ProjectStore"]
