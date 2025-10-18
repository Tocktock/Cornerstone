"""Project and document metadata management."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import re
from pathlib import Path
from typing import Iterable, List, Sequence
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


@dataclass(slots=True)
class KeywordRunRecord:
    """Serialized representation of a keyword pipeline run."""

    id: str
    project_id: str
    status: str
    requested_at: str
    updated_at: str
    started_at: str | None = None
    completed_at: str | None = None
    requested_by: str | None = None
    stats: dict[str, object] = field(default_factory=dict)
    keywords: list[dict[str, object]] | None = None
    insights: list[dict[str, object]] | None = None
    debug: dict[str, object] | None = None
    error: str | None = None


class ProjectStore:
    """File-based repository for projects and document metadata."""

    def __init__(self, root: Path, *, default_project_name: str) -> None:
        self._root = root
        self._projects_file = root / "projects.json"
        self._documents_dir = root / "documents"
        self._keywords_dir = root / "keywords"
        self._keyword_runs_dir = root / "keyword_runs"
        self._glossary_dir = root / "glossary"
        self._query_hints_dir = root / "query_hints"
        self._root.mkdir(parents=True, exist_ok=True)
        self._documents_dir.mkdir(parents=True, exist_ok=True)
        self._keywords_dir.mkdir(parents=True, exist_ok=True)
        self._keyword_runs_dir.mkdir(parents=True, exist_ok=True)
        self._glossary_dir.mkdir(parents=True, exist_ok=True)
        self._query_hints_dir.mkdir(parents=True, exist_ok=True)
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

        timestamp = self._now()
        insight_with_meta = {
            "id": uuid4().hex,
            "created_at": timestamp,
            "updated_at": timestamp,
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

    def list_keyword_insights(self, project_id: str) -> list[dict]:
        path = self._keywords_dir / f"{project_id}.json"
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as handle:
            insights = json.load(handle)
        try:
            return sorted(
                insights,
                key=lambda item: item.get("created_at") or "",
                reverse=True,
            )
        except TypeError:
            return insights

    # Keyword run persistence ---------------------------------------------------

    def create_keyword_run(
        self,
        project_id: str,
        *,
        requested_by: str | None = None,
        status: str = "pending",
        stats: dict[str, object] | None = None,
    ) -> KeywordRunRecord:
        project = self.get_project(project_id)
        if project is None:
            raise ValueError(f"Project {project_id} not found")

        run_id = uuid4().hex
        now = self._now()
        record = KeywordRunRecord(
            id=run_id,
            project_id=project.id,
            status=status,
            requested_at=now,
            updated_at=now,
            requested_by=requested_by,
            stats=dict(stats or {}),
        )
        self._write_keyword_run(record)
        return record

    def update_keyword_run(
        self,
        project_id: str,
        run_id: str,
        *,
        status: str | None = None,
        started_at: str | None = None,
        completed_at: str | None = None,
        requested_by: str | None = None,
        stats: dict[str, object] | None = None,
        keywords: Sequence[dict[str, object]] | None = None,
        insights: Sequence[dict[str, object]] | None = None,
        debug: dict[str, object] | None = None,
        error: str | None = None,
    ) -> KeywordRunRecord:
        record = self.get_keyword_run(project_id, run_id)
        if record is None:
            raise ValueError("Keyword run not found")

        updated = asdict(record)
        if status is not None:
            updated["status"] = status
        if started_at is not None:
            updated["started_at"] = started_at
        if completed_at is not None:
            updated["completed_at"] = completed_at
        if requested_by is not None:
            updated["requested_by"] = requested_by
        if stats:
            merged_stats = dict(updated.get("stats") or {})
            merged_stats.update(stats)
            updated["stats"] = merged_stats
        if keywords is not None:
            updated["keywords"] = [dict(item) for item in keywords]
        if insights is not None:
            updated["insights"] = [dict(item) for item in insights]
        if debug is not None:
            updated["debug"] = dict(debug)
        if error is not None:
            updated["error"] = error

        updated["updated_at"] = self._now()

        new_record = self._deserialize_keyword_run(updated)
        self._write_keyword_run(new_record)

        if new_record.status == "success":
            self._write_latest_keyword_run_pointer(project_id, new_record.id)
        else:
            pointer = self._read_latest_keyword_run_pointer(project_id)
            if pointer == new_record.id and new_record.status != "success":
                self._remove_latest_keyword_run_pointer(project_id)

        return new_record

    def get_keyword_run(self, project_id: str, run_id: str) -> KeywordRunRecord | None:
        path = self._keyword_run_path(project_id, run_id)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except json.JSONDecodeError:
            logger.warning(
                "project.keyword_run.decode_failed project=%s run_id=%s path=%s",
                project_id,
                run_id,
                path,
            )
            return None
        return self._deserialize_keyword_run(data)

    def list_keyword_runs(
        self,
        project_id: str,
        *,
        limit: int | None = None,
        statuses: Sequence[str] | None = None,
    ) -> list[KeywordRunRecord]:
        run_dir = self._keyword_run_dir_for(project_id, ensure=False)
        if not run_dir.exists():
            return []

        records: list[KeywordRunRecord] = []
        for path in sorted(run_dir.glob("*.json")):
            if path.name.startswith("latest_"):
                continue
            try:
                with path.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
            except json.JSONDecodeError:
                logger.warning("project.keyword_run.decode_failed project=%s path=%s", project_id, path)
                continue
            record = self._deserialize_keyword_run(data)
            if statuses and record.status not in statuses:
                continue
            records.append(record)

        records.sort(key=lambda item: item.requested_at, reverse=True)
        if limit is not None and limit >= 0:
            records = records[:limit]
        return records

    def get_latest_keyword_run(
        self,
        project_id: str,
        *,
        status: str = "success",
    ) -> KeywordRunRecord | None:
        pointer = self._read_latest_keyword_run_pointer(project_id)
        if pointer:
            record = self.get_keyword_run(project_id, pointer)
            if record and record.status == status:
                return record

        runs = self.list_keyword_runs(project_id, statuses=[status])
        return runs[0] if runs else None

    def update_keyword_insight(self, project_id: str, insight_id: str, updates: dict) -> dict:
        path = self._keywords_dir / f"{project_id}.json"
        if not path.exists():
            raise ValueError("Insight not found")

        with path.open("r", encoding="utf-8") as handle:
            insights: list[dict] = json.load(handle)

        updated_item: dict | None = None
        timestamp = self._now()
        normalized_updates = {}
        if "term" in updates:
            term = str(updates.get("term", "")).strip()
            if term:
                normalized_updates["term"] = term
        if "candidates" in updates:
            normalized_updates["candidates"] = list(updates.get("candidates") or [])
        if "definitions" in updates:
            normalized_updates["definitions"] = list(updates.get("definitions") or [])
        if "filter" in updates:
            normal_filter = updates.get("filter") or {}
            if isinstance(normal_filter, dict):
                normalized_updates["filter"] = normal_filter

        new_items: list[dict] = []
        for item in insights:
            if item.get("id") != insight_id:
                new_items.append(item)
                continue
            updated = dict(item)
            updated.update(normalized_updates)
            updated["updated_at"] = timestamp
            updated_item = updated
            new_items.append(updated)

        if updated_item is None:
            raise ValueError("Insight not found")

        with path.open("w", encoding="utf-8") as handle:
            json.dump(new_items, handle, indent=2)

        logger.info(
            "project.keyword.updated project=%s insight_id=%s fields=%s",
            project_id,
            insight_id,
            list(normalized_updates.keys()),
        )
        return updated_item

    def delete_keyword_insight(self, project_id: str, insight_id: str) -> bool:
        path = self._keywords_dir / f"{project_id}.json"
        if not path.exists():
            return False

        with path.open("r", encoding="utf-8") as handle:
            insights: list[dict] = json.load(handle)

        remaining = [item for item in insights if item.get("id") != insight_id]
        if len(remaining) == len(insights):
            return False

        with path.open("w", encoding="utf-8") as handle:
            json.dump(remaining, handle, indent=2)

        logger.info("project.keyword.deleted project=%s insight_id=%s", project_id, insight_id)
        return True

    # Glossary entry persistence -------------------------------------------------

    def list_glossary_entries(self, project_id: str) -> list[dict]:
        return list(self._read_glossary_entries(project_id))

    def create_glossary_entry(
        self,
        project_id: str,
        *,
        term: str,
        definition: str,
        synonyms: Iterable[str] | None = None,
        keywords: Iterable[str] | None = None,
    ) -> dict:
        term_clean = term.strip()
        definition_clean = definition.strip()
        if not term_clean or not definition_clean:
            raise ValueError("term and definition are required")

        entries = self._read_glossary_entries(project_id)
        entry = {
            "id": uuid4().hex,
            "term": term_clean,
            "definition": definition_clean,
            "synonyms": self._normalize_string_list(synonyms),
            "keywords": self._normalize_string_list(keywords),
            "created_at": self._now(),
            "updated_at": self._now(),
        }
        entries.append(entry)
        self._write_glossary_entries(project_id, entries)
        logger.info("project.glossary.created project=%s entry_id=%s", project_id, entry["id"])
        return entry

    def update_glossary_entry(
        self,
        project_id: str,
        entry_id: str,
        *,
        term: str | None = None,
        definition: str | None = None,
        synonyms: Iterable[str] | None = None,
        keywords: Iterable[str] | None = None,
    ) -> dict:
        entries = self._read_glossary_entries(project_id)
        for item in entries:
            if item.get("id") != entry_id:
                continue
            if term is not None:
                term_clean = term.strip()
                if not term_clean:
                    raise ValueError("term cannot be empty")
                item["term"] = term_clean
            if definition is not None:
                definition_clean = definition.strip()
                if not definition_clean:
                    raise ValueError("definition cannot be empty")
                item["definition"] = definition_clean
            if synonyms is not None:
                item["synonyms"] = self._normalize_string_list(synonyms)
            if keywords is not None:
                item["keywords"] = self._normalize_string_list(keywords)
            item["updated_at"] = self._now()
            self._write_glossary_entries(project_id, entries)
            logger.info("project.glossary.updated project=%s entry_id=%s", project_id, entry_id)
            return item
        raise ValueError(f"Glossary entry {entry_id} not found for project {project_id}")

    def delete_glossary_entry(self, project_id: str, entry_id: str) -> bool:
        entries = self._read_glossary_entries(project_id)
        filtered = [item for item in entries if item.get("id") != entry_id]
        if len(filtered) == len(entries):
            return False
        self._write_glossary_entries(project_id, filtered)
        logger.info("project.glossary.deleted project=%s entry_id=%s", project_id, entry_id)
        return True

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

    def _keyword_run_dir_for(self, project_id: str, *, ensure: bool = True) -> Path:
        path = self._keyword_runs_dir / project_id
        if ensure:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def _keyword_run_path(self, project_id: str, run_id: str) -> Path:
        return self._keyword_run_dir_for(project_id) / f"{run_id}.json"

    def _serialize_keyword_run(self, record: KeywordRunRecord) -> dict:
        payload = asdict(record)
        # Ensure mutable defaults serialise as expected.
        payload.setdefault("stats", {})
        return payload

    def _deserialize_keyword_run(self, data: dict) -> KeywordRunRecord:
        project_id = data.get("project_id") or data.get("projectId")
        run_id = data.get("id")
        if not project_id or not run_id:
            raise ValueError("Invalid keyword run payload")
        return KeywordRunRecord(
            id=str(run_id),
            project_id=str(project_id),
            status=str(data.get("status", "pending")),
            requested_at=str(data.get("requested_at") or data.get("requestedAt") or self._now()),
            updated_at=str(data.get("updated_at") or data.get("updatedAt") or self._now()),
            started_at=data.get("started_at") or data.get("startedAt"),
            completed_at=data.get("completed_at") or data.get("completedAt"),
            requested_by=data.get("requested_by") or data.get("requestedBy"),
            stats=dict(data.get("stats") or {}),
            keywords=[dict(item) for item in data.get("keywords", [])] if data.get("keywords") is not None else None,
            insights=[dict(item) for item in data.get("insights", [])] if data.get("insights") is not None else None,
            debug=dict(data.get("debug") or {}),
            error=data.get("error"),
        )

    def _write_keyword_run(self, record: KeywordRunRecord) -> None:
        path = self._keyword_run_path(record.project_id, record.id)
        payload = self._serialize_keyword_run(record)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _latest_keyword_run_pointer_path(self, project_id: str) -> Path:
        return self._keyword_run_dir_for(project_id) / "latest_success.txt"

    def _write_latest_keyword_run_pointer(self, project_id: str, run_id: str) -> None:
        path = self._latest_keyword_run_pointer_path(project_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(run_id, encoding="utf-8")

    def _read_latest_keyword_run_pointer(self, project_id: str) -> str | None:
        path = self._latest_keyword_run_pointer_path(project_id)
        if not path.exists():
            return None
        try:
            value = path.read_text(encoding="utf-8").strip()
        except OSError:
            return None
        return value or None

    def _remove_latest_keyword_run_pointer(self, project_id: str) -> None:
        path = self._latest_keyword_run_pointer_path(project_id)
        if path.exists():
            try:
                path.unlink()
            except OSError:
                logger.warning("project.keyword_run.pointer_remove_failed project=%s path=%s", project_id, path)

    def _glossary_path(self, project_id: str) -> Path:
        return self._glossary_dir / f"{project_id}.json"

    def _read_glossary_entries(self, project_id: str) -> list[dict]:
        path = self._glossary_path(project_id)
        if not path.exists():
            return []
        with path.open("r", encoding="utf-8") as handle:
            try:
                data = json.load(handle)
            except json.JSONDecodeError:
                logger.warning("project.glossary.decode_failed project=%s path=%s", project_id, path)
                return []
        if not isinstance(data, list):
            return []
        return data

    def _write_glossary_entries(self, project_id: str, entries: list[dict]) -> None:
        path = self._glossary_path(project_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(entries, handle, indent=2)

    @staticmethod
    def _normalize_string_list(values: Iterable[str] | None) -> list[str]:
        if not values:
            return []
        normalized: list[str] = []
        for value in values:
            if value is None:
                continue
            cleaned = str(value).strip()
            if cleaned and cleaned not in normalized:
                normalized.append(cleaned)
        return normalized

    def get_query_hints(self, project_id: str) -> dict[str, list[str]]:
        path = self._query_hints_dir / f"{project_id}.json"
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            try:
                data = json.load(handle)
            except json.JSONDecodeError:
                logger.warning("project.query_hints.decode_failed project=%s path=%s", project_id, path)
                return {}
        if not isinstance(data, dict):
            return {}
        if "hints" in data and isinstance(data["hints"], dict):
            raw_hints = data["hints"]
        else:
            raw_hints = data
        hints: dict[str, list[str]] = {}
        for key, values in raw_hints.items():
            if not isinstance(key, str):
                continue
            cleaned_key = key.strip().lower()
            if not cleaned_key:
                continue
            bucket: list[str] = []
            if isinstance(values, list):
                for value in values:
                    if isinstance(value, str):
                        cleaned = value.strip()
                        if cleaned and cleaned not in bucket:
                            bucket.append(cleaned)
            elif isinstance(values, str):
                cleaned = values.strip()
                if cleaned:
                    bucket.append(cleaned)
            if bucket:
                hints[cleaned_key] = bucket
        return hints

    def get_query_hint_metadata(self, project_id: str) -> dict[str, object]:
        path = self._query_hints_dir / f"{project_id}.json"
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            try:
                data = json.load(handle)
            except json.JSONDecodeError:
                return {}
        if isinstance(data, dict) and "metadata" in data and isinstance(data["metadata"], dict):
            return data["metadata"]
        return {}

    def set_query_hints(
        self,
        project_id: str,
        hints: dict[str, list[str]],
        *,
        metadata: dict[str, object] | None = None,
    ) -> None:
        path = self._query_hints_dir / f"{project_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload_hints: dict[str, list[str]] = {}
        for key, values in hints.items():
            if key is None:
                continue
            cleaned_key = str(key).strip()
            if not cleaned_key:
                continue
            bucket: list[str] = []
            for value in values:
                if value is None:
                    continue
                cleaned = str(value).strip()
                if cleaned and cleaned not in bucket:
                    bucket.append(cleaned)
            payload_hints[cleaned_key.lower()] = bucket
        payload: dict[str, object] = {"hints": payload_hints}
        if metadata:
            payload["metadata"] = metadata
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        logger.info("project.query_hints.updated project=%s keys=%s", project_id, len(payload_hints))


__all__ = ["Project", "DocumentMetadata", "ProjectStore"]
