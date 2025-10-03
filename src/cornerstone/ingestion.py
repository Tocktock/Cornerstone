"""Document ingestion utilities for project-specific knowledge bases."""

from __future__ import annotations

import io
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence
from urllib.parse import urlparse
from uuid import uuid4

import httpx
from fastapi import UploadFile

try:  # pragma: no cover - optional dependency
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None

try:  # pragma: no cover - optional dependency
    from docx import Document as DocxDocument
except Exception:  # pragma: no cover
    DocxDocument = None

try:  # pragma: no cover - optional dependency
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover
    BeautifulSoup = None

import logging
import mimetypes
import threading

from .embeddings import EmbeddingService
from .projects import DocumentMetadata, ProjectStore
from .vector_store import QdrantVectorStore, VectorRecord
from qdrant_client import models


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestionResult:
    document: DocumentMetadata
    chunks_ingested: int


@dataclass(slots=True)
class ExtractedDocument:
    text: str
    title: str | None
    content_type: str | None


@dataclass(slots=True)
class IngestionJob:
    id: str
    project_id: str
    filename: str
    status: str
    created_at: str
    updated_at: str
    document: DocumentMetadata | None = None
    error: str | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "project_id": self.project_id,
            "filename": self.filename,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "document": asdict(self.document) if self.document else None,
        }


class IngestionJobManager:
    """Track ingestion job status for asynchronous uploads."""

    def __init__(self) -> None:
        self._jobs: dict[str, IngestionJob] = {}
        self._lock = threading.Lock()

    def create_job(self, project_id: str, filename: str) -> IngestionJob:
        now = self._now()
        job = IngestionJob(
            id=uuid4().hex,
            project_id=project_id,
            filename=filename,
            status="pending",
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._jobs[job.id] = job
        return job

    def mark_processing(self, job_id: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.status = "processing"
            job.updated_at = self._now()

    def mark_completed(self, job_id: str, document: DocumentMetadata) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.status = "completed"
            job.document = document
            job.error = None
            job.updated_at = self._now()

    def mark_failed(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            job.status = "failed"
            job.error = error
            job.updated_at = self._now()

    def list_for_project(self, project_id: str) -> list[IngestionJob]:
        with self._lock:
            return sorted(
                (job for job in self._jobs.values() if job.project_id == project_id),
                key=lambda job: job.created_at,
            )

    def get(self, job_id: str) -> IngestionJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


class ProjectVectorStoreManager:
    """Cache and manage vector stores per project."""

    def __init__(
        self,
        client_factory,
        *,
        vector_size: int,
        distance,
        collection_name_fn,
    ) -> None:
        self._client_factory = client_factory
        self._vector_size = vector_size
        self._distance = distance
        self._collection_name_fn = collection_name_fn
        self._stores: dict[str, QdrantVectorStore] = {}

    def get_store(self, project_id: str) -> QdrantVectorStore:
        if project_id in self._stores:
            return self._stores[project_id]
        collection_name = self._collection_name_fn(project_id)
        store = QdrantVectorStore(
            client=self._client_factory(),
            collection_name=collection_name,
            vector_size=self._vector_size,
            distance=self._distance,
        )
        store.ensure_collection()
        self._stores[project_id] = store
        return store

    def delete_document(self, project_id: str, doc_id: str) -> bool:
        store = self.get_store(project_id)
        flt = models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
        result = store.delete_by_filter(flt)
        return result.status == models.UpdateStatus.COMPLETED

    def purge_project(self, project_id: str) -> bool:
        """Remove all vectors associated with a project."""

        store = self.get_store(project_id)
        flt = models.Filter(
            must=[models.FieldCondition(key="project_id", match=models.MatchValue(value=project_id))]
        )
        result = store.delete_by_filter(flt)
        return result.status == models.UpdateStatus.COMPLETED

    def iter_project_payloads(self, project_id: str, *, batch_size: int = 256):
        """Yield payload dictionaries for all vectors stored for a project."""

        store = self.get_store(project_id)
        flt = models.Filter(
            must=[models.FieldCondition(key="project_id", match=models.MatchValue(value=project_id))]
        )
        yield from store.iter_payloads(scroll_filter=flt, batch_size=batch_size)


class DocumentIngestor:
    """Convert uploaded documents into embeddings stored per project."""

    def __init__(
        self,
        embedding_service: EmbeddingService,
        store_manager: ProjectVectorStoreManager,
        project_store: ProjectStore,
    ) -> None:
        self._embedding = embedding_service
        self._stores = store_manager
        self._projects = project_store

    async def ingest_upload(self, project_id: str, upload: UploadFile) -> IngestionResult:
        filename = upload.filename or "document"
        logger.info("ingest.start project=%s filename=%s", project_id, filename)
        raw_bytes = await upload.read()
        return self.ingest_bytes(
            project_id,
            filename=filename,
            data=raw_bytes,
            content_type=upload.content_type,
        )

    def ingest_bytes(
        self,
        project_id: str,
        *,
        filename: str,
        data: bytes,
        content_type: str | None = None,
    ) -> IngestionResult:
        if not data:
            raise ValueError("Uploaded file is empty")

        extracted = self._extract_document(filename, data, content_type)
        chunks = self._chunk_text(extracted.text)
        if not chunks:
            raise ValueError("No textual content could be extracted from the document")

        embeddings = self._embedding.embed(chunks)
        doc_id = uuid4().hex
        store = self._stores.get_store(project_id)
        records = self._build_records(
            doc_id,
            filename,
            chunks,
            embeddings,
            project_id,
            extracted.content_type,
            extracted.title,
        )
        store.upsert(records)
        logger.info(
            "ingest.upserted project=%s doc_id=%s chunks=%s",
            project_id,
            doc_id,
            len(records),
        )

        metadata = DocumentMetadata(
            id=doc_id,
            filename=filename,
            chunk_count=len(records),
            created_at=self._now(),
            size_bytes=len(data),
            title=extracted.title,
            content_type=extracted.content_type,
        )
        self._projects.record_document(project_id, metadata)
        logger.info("ingest.completed project=%s doc_id=%s", project_id, doc_id)
        return IngestionResult(document=metadata, chunks_ingested=len(records))

    def ingest_url(
        self,
        project_id: str,
        *,
        url: str,
        timeout: float = 10.0,
    ) -> IngestionResult:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError("Only http and https URLs are supported")
        try:
            response = httpx.get(url, timeout=timeout, follow_redirects=True)
        except httpx.HTTPError as exc:  # pragma: no cover - network failure
            raise ValueError(f"Failed to download URL: {exc}") from exc
        if response.status_code >= 400:
            raise ValueError(f"Failed to download URL (status {response.status_code})")
        content = response.content
        if not content:
            raise ValueError("URL returned no content")

        content_type = response.headers.get("content-type")
        if content_type:
            content_type = content_type.split(";", 1)[0].strip()

        filename = Path(parsed.path).name or "document"
        if not Path(filename).suffix and content_type:
            extension = mimetypes.guess_extension(content_type.split(";", 1)[0])
            if extension:
                filename = f"{filename}{extension}"

        return self.ingest_bytes(
            project_id,
            filename=filename,
            data=content,
            content_type=content_type,
        )

    def _build_records(
        self,
        doc_id: str,
        filename: str,
        chunks: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        project_id: str,
        content_type: str | None,
        title: str | None,
    ) -> list[VectorRecord]:
        records: list[VectorRecord] = []
        for index, (chunk, vector) in enumerate(zip(chunks, embeddings, strict=True)):
            payload = {
                "text": chunk,
                "project_id": project_id,
                "doc_id": doc_id,
                "chunk_index": index,
                "source": filename,
                "content_type": content_type,
                "title": title,
            }
            records.append(VectorRecord(id=uuid4().hex, vector=vector, payload=payload))
        return records

    @staticmethod
    def _extract_document(filename: str, data: bytes, content_type: str | None) -> ExtractedDocument:
        suffix = Path(filename).suffix.lower()
        guessed_type = content_type or mimetypes.guess_type(filename)[0]

        if suffix in {".md", ".markdown"}:
            text = data.decode("utf-8", errors="ignore")
            title = DocumentIngestor._derive_markdown_title(text)
            return ExtractedDocument(text=text, title=title, content_type=guessed_type or "text/markdown")

        if suffix in {".txt", ""} or (guessed_type and guessed_type.startswith("text/")):
            text = data.decode("utf-8", errors="ignore")
            title = DocumentIngestor._derive_plain_title(text)
            return ExtractedDocument(text=text, title=title, content_type=guessed_type or "text/plain")

        if suffix == ".pdf":
            if PdfReader is None:
                raise ValueError("PDF support requires the 'pypdf' package")
            try:
                reader = PdfReader(io.BytesIO(data))
                texts = [page.extract_text() or "" for page in reader.pages]
                metadata_title = None
                try:  # pragma: no cover - optional metadata
                    metadata_title = getattr(reader, "metadata", None)
                    if metadata_title and getattr(metadata_title, "title", None):
                        metadata_title = metadata_title.title
                    else:
                        metadata_title = None
                except Exception:
                    metadata_title = None
            except Exception as exc:  # pragma: no cover - parsing errors
                raise ValueError(f"Failed to extract text from PDF: {exc}") from exc
            combined = "\n\n".join(filter(None, texts))
            if not combined.strip():
                combined = data.decode("utf-8", errors="ignore")
            title = metadata_title or DocumentIngestor._derive_plain_title(combined)
            return ExtractedDocument(text=combined, title=title, content_type=guessed_type or "application/pdf")

        if suffix == ".docx":
            if DocxDocument is None:
                raise ValueError("DOCX support requires the 'python-docx' package")
            try:
                document = DocxDocument(io.BytesIO(data))
            except Exception as exc:  # pragma: no cover - parsing errors
                raise ValueError(f"Failed to extract text from DOCX: {exc}") from exc
            paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs if paragraph.text.strip()]
            text = "\n\n".join(paragraphs)
            core_title = None
            try:
                core_title = document.core_properties.title
            except Exception:  # pragma: no cover
                core_title = None
            base_title = core_title or (paragraphs[0] if paragraphs else None)
            if base_title:
                base_title = base_title.strip()
            return ExtractedDocument(
                text=text,
                title=base_title or DocumentIngestor._derive_plain_title(text),
                content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )

        if suffix in {".html", ".htm"} or (guessed_type and "html" in guessed_type):
            if BeautifulSoup is None:
                raise ValueError("HTML support requires the 'beautifulsoup4' package")
            soup = BeautifulSoup(data, "html.parser")
            for tag in soup(["script", "style"]):
                tag.extract()
            title = None
            if soup.title and soup.title.string:
                title = soup.title.string.strip()
            text = soup.get_text(separator="\n")
            return ExtractedDocument(
                text=text,
                title=title or DocumentIngestor._derive_plain_title(text),
                content_type="text/html",
            )

        # Fallback: treat as UTF-8 text
        text = data.decode("utf-8", errors="ignore")
        return ExtractedDocument(
            text=text,
            title=DocumentIngestor._derive_plain_title(text),
            content_type=guessed_type or "text/plain",
        )

    @staticmethod
    def _chunk_text(text: str, *, max_chars: int = 1200, overlap: int = 150) -> list[str]:
        cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
        paragraphs = [para.strip() for para in cleaned.split("\n\n") if para.strip()]
        chunks: list[str] = []
        buffer = ""
        for paragraph in paragraphs:
            if len(buffer) + len(paragraph) + 2 <= max_chars:
                buffer = f"{buffer}\n\n{paragraph}" if buffer else paragraph
            else:
                if buffer:
                    chunks.append(buffer.strip())
                buffer = paragraph
        if buffer:
            chunks.append(buffer.strip())

        if not chunks:
            return []

        merged: list[str] = []
        for chunk in chunks:
            if len(chunk) <= max_chars:
                merged.append(chunk)
                continue
            start = 0
            while start < len(chunk):
                end = min(start + max_chars, len(chunk))
                merged.append(chunk[start:end].strip())
                if end >= len(chunk):
                    break
                next_start = max(0, end - overlap)
                if next_start <= start:
                    next_start = end
                start = next_start
        return merged

    @staticmethod
    def _derive_markdown_title(text: str) -> str | None:
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()[:160] or None
            return stripped[:160]
        return None

    @staticmethod
    def _derive_plain_title(text: str) -> str | None:
        for line in text.splitlines():
            stripped = line.strip()
            if stripped:
                return stripped[:160]
        return None

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


__all__ = [
    "DocumentIngestor",
    "ProjectVectorStoreManager",
    "IngestionResult",
    "IngestionJob",
    "IngestionJobManager",
]
