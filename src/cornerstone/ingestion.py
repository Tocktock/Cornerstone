"""Document ingestion utilities for project-specific knowledge bases."""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from fastapi import UploadFile

try:  # pragma: no cover - optional dependency
    from pypdf import PdfReader
except Exception:  # pragma: no cover
    PdfReader = None

import logging

from .embeddings import EmbeddingService
from .projects import DocumentMetadata, ProjectStore
from .vector_store import QdrantVectorStore, VectorRecord
from qdrant_client import models


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestionResult:
    document: DocumentMetadata
    chunks_ingested: int


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
        logger.info("ingest.start project=%s filename=%s", project_id, upload.filename)
        raw_bytes = await upload.read()
        text = self._extract_text(upload.filename or upload.content_type or "", raw_bytes)
        chunks = self._chunk_text(text)
        if not chunks:
            raise ValueError("No textual content could be extracted from the document")

        embeddings = self._embedding.embed(chunks)
        doc_id = uuid4().hex
        store = self._stores.get_store(project_id)
        records = self._build_records(doc_id, upload.filename or "document", chunks, embeddings, project_id)
        store.upsert(records)
        logger.info(
            "ingest.upserted project=%s doc_id=%s chunks=%s",
            project_id,
            doc_id,
            len(records),
        )

        metadata = DocumentMetadata(
            id=doc_id,
            filename=upload.filename or "document",
            chunk_count=len(records),
            created_at=self._now(),
            size_bytes=len(raw_bytes),
        )
        self._projects.record_document(project_id, metadata)
        logger.info("ingest.completed project=%s doc_id=%s", project_id, doc_id)
        return IngestionResult(document=metadata, chunks_ingested=len(records))

    def _build_records(
        self,
        doc_id: str,
        filename: str,
        chunks: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        project_id: str,
    ) -> list[VectorRecord]:
        records: list[VectorRecord] = []
        for index, (chunk, vector) in enumerate(zip(chunks, embeddings, strict=True)):
            payload = {
                "text": chunk,
                "project_id": project_id,
                "doc_id": doc_id,
                "chunk_index": index,
                "source": filename,
            }
            records.append(VectorRecord(id=uuid4().hex, vector=vector, payload=payload))
        return records

    @staticmethod
    def _extract_text(filename: str, data: bytes) -> str:
        suffix = Path(filename).suffix.lower()
        if suffix in {".md", ".markdown", ".txt", ""}:
            return data.decode("utf-8", errors="ignore")
        if suffix == ".pdf":
            if PdfReader is None:
                raise ValueError("PDF support requires the 'pypdf' package")
            try:
                reader = PdfReader(io.BytesIO(data))
                texts = [page.extract_text() or "" for page in reader.pages]
            except Exception as exc:  # pragma: no cover - parsing errors
                raise ValueError(f"Failed to extract text from PDF: {exc}") from exc
            combined = "\n\n".join(filter(None, texts))
            if combined.strip():
                return combined
            # Fallback: best effort decode to avoid empty ingestion
            return data.decode("utf-8", errors="ignore")
        return data.decode("utf-8", errors="ignore")

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
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()


__all__ = ["DocumentIngestor", "ProjectVectorStoreManager", "IngestionResult"]
