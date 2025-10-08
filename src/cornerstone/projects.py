"""Project and document metadata management."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Iterable, List
from uuid import uuid4

import logging

from .personas import PersonaOverrides


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Project:
    """Metadata describing a knowledge base project."""

    id: str
    name: str
    description: str | None
    created_at: str
    persona_id: str | None = None
    persona_overrides: PersonaOverrides = field(default_factory=PersonaOverrides)


@dataclass(slots=True)
class DocumentMetadata:
    """Metadata about an ingested document."""

    id: str
    filename: str
    chunk_count: int
    created_at: str
    size_bytes: int | None = None
    title: str | None = None
    content_type: str | None = None


class ProjectStore:
    """File-based repository for projects and document metadata."""

    def __init__(self, root: Path, *, default_project_name: str) -> None:
        self._root = root
        self._projects_file = root / "projects.json"
        self._documents_dir = root / "documents"
        self._keywords_dir = root / "keywords"
        self._root.mkdir(parents=True, exist_ok=True)
        self._documents_dir.mkdir(parents=True, exist_ok=True)
        self._keywords_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_default_project(default_project_name)

    def list_projects(self) -> List[Project]:
        data = self._read_projects()
        projects = [self._deserialize_project(item) for item in data.get("projects", [])]
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
        record = asdict(project)
        if record.get("persona_id") is None:
            record.pop("persona_id", None)
        if not self._normalize_overrides(project.persona_overrides):
            record.pop("persona_overrides", None)
        projects.setdefault("projects", []).append(record)
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

    # Keyword insight persistence -------------------------------------------------

    def configure_persona(
        self,
        project_id: str,
        *,
        persona_id: str | None,
        overrides: PersonaOverrides | None = None,
    ) -> Project:
        projects = self._read_projects()
        project_items = projects.get("projects", [])
        persona_id = persona_id.strip() if persona_id and persona_id.strip() else None
        for item in project_items:
            if item.get("id") != project_id:
                continue
            if persona_id:
                item["persona_id"] = persona_id
            elif "persona_id" in item:
                item.pop("persona_id")
            normalized = self._normalize_overrides(overrides)
            if normalized:
                item["persona_overrides"] = normalized
            elif "persona_overrides" in item:
                item.pop("persona_overrides")
            self._write_projects(projects)
            updated = self._deserialize_project(item)
            logger.info(
                "project.persona.assigned project=%s persona=%s overrides=%s",
                updated.id,
                updated.persona_id,
                bool(normalized),
            )
            return updated
        raise ValueError(f"Project {project_id} not found")

    def save_keyword_insight(self, project_id: str, insight: dict) -> dict:
        path = self._keywords_dir / f"{project_id}.json"
        insights: list[dict] = []
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                insights = json.load(handle)

        insight_with_meta = {
            "id": uuid4().hex,
            "created_at": self._now(),
            **insight,
        }
        insights.append(insight_with_meta)

        with path.open("w", encoding="utf-8") as handle:
            json.dump(insights, handle, indent=2)

        logger.info(
            "project.keyword.saved project=%s term=%s insight_id=%s",
            project_id,
            insight.get("term"),
            insight_with_meta["id"],
        )
        return insight_with_meta

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

    def _deserialize_project(self, payload: dict) -> Project:
        overrides_payload = payload.get("persona_overrides") or {}
        persona_payload = payload.get("persona") or {}
        if not overrides_payload and persona_payload:
            overrides_payload = persona_payload  # legacy field support
        overrides = PersonaOverrides(
            name=overrides_payload.get("name"),
            tone=overrides_payload.get("tone"),
            system_prompt=overrides_payload.get("system_prompt"),
            avatar_url=overrides_payload.get("avatar_url"),
            glossary_top_k=overrides_payload.get("glossary_top_k"),
            retrieval_top_k=overrides_payload.get("retrieval_top_k"),
            chat_temperature=overrides_payload.get("chat_temperature"),
            chat_max_tokens=overrides_payload.get("chat_max_tokens"),
        )
        return Project(
            id=payload.get("id"),
            name=payload.get("name"),
            description=payload.get("description"),
            created_at=payload.get("created_at"),
            persona_id=payload.get("persona_id"),
            persona_overrides=overrides,
        )

    @staticmethod
    def _normalize(value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    def _normalize_overrides(self, overrides: PersonaOverrides | None) -> dict | None:
        if overrides is None:
            return None
        normalized = {
            "name": self._normalize(overrides.name),
            "tone": self._normalize(overrides.tone),
            "system_prompt": self._normalize(overrides.system_prompt),
            "avatar_url": self._normalize(overrides.avatar_url),
        }
        if overrides.glossary_top_k is not None:
            normalized["glossary_top_k"] = overrides.glossary_top_k
        if overrides.retrieval_top_k is not None:
            normalized["retrieval_top_k"] = overrides.retrieval_top_k
        if overrides.chat_temperature is not None:
            normalized["chat_temperature"] = overrides.chat_temperature
        if overrides.chat_max_tokens is not None:
            normalized["chat_max_tokens"] = overrides.chat_max_tokens
        filtered = {key: value for key, value in normalized.items() if value is not None}
        return filtered or None

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
