"""Helpers for ingesting local filesystem directories."""

from __future__ import annotations

import json
import logging
import mimetypes
import re
import time
from pathlib import Path
from typing import Dict, Iterable
from uuid import uuid4

from .ingestion import DocumentIngestor
from .projects import DocumentMetadata
from .ingestion import IngestionJobManager

logger = logging.getLogger(__name__)

ALLOWED_SUFFIXES = {
    ".md",
    ".markdown",
    ".txt",
    ".html",
    ".htm",
    ".csv",
    ".pdf",
    ".docx",
}


def resolve_local_path(base_dir: Path, relative_path: str) -> Path:
    """Resolve a user-supplied relative path within the local data directory."""

    base_dir = base_dir.resolve()
    relative_path = relative_path.strip().lstrip("/\\")
    target = (base_dir / relative_path).resolve()
    try:
        target.relative_to(base_dir)
    except ValueError:
        raise ValueError("Path must reside inside the local data directory") from None
    return target


def _directory_stats(path: Path) -> dict[str, int]:
    file_count = 0
    total_bytes = 0
    for child in path.rglob("*"):
        if not child.is_file():
            continue
        if not is_supported_file(child):
            continue
        file_count += 1
        try:
            total_bytes += child.stat().st_size
        except OSError:
            continue
    return {"file_count": file_count, "total_bytes": total_bytes}


def _normalize_directory_query(relative_path: str | None) -> tuple[str, bool]:
    """Normalize a user-supplied path for directory suggestions."""

    if relative_path is None:
        return "", False
    cleaned = relative_path.strip()
    if not cleaned:
        return "", False
    cleaned = cleaned.replace("\\", "/")
    cleaned = re.sub(r"/{2,}", "/", cleaned)
    cleaned = cleaned.lstrip("/")
    if not cleaned:
        return "", False
    trailing_slash = cleaned.endswith("/")
    normalized = cleaned[:-1] if trailing_slash else cleaned
    return normalized, trailing_slash


def list_directories(base_dir: Path, relative_path: str | None = None) -> list[dict[str, str]]:
    """Return subdirectory suggestions with aggregate stats.

    The call supports lightweight auto-complete semantics. When *relative_path*
    points to an existing directory (or ends with a trailing slash), immediate
    children of that directory are returned. Otherwise we treat the final path
    segment as a prefix filter and return matching entries from the deepest
    existing parent.
    """

    normalized, trailing_slash = _normalize_directory_query(relative_path)
    target = resolve_local_path(base_dir, normalized)

    if normalized:
        if trailing_slash:
            directory_root = target
            if not directory_root.exists() or not directory_root.is_dir():
                return []
            prefix = None
        else:
            directory_root = target.parent
            prefix = target.name or None
            if not directory_root.exists() or not directory_root.is_dir():
                return []
    else:
        directory_root = target
        prefix = None

    directories: list[dict[str, str]] = []
    for entry in sorted(directory_root.iterdir()):
        if not entry.is_dir():
            continue
        if prefix and not entry.name.startswith(prefix):
            continue
        stats = _directory_stats(entry)
        directories.append(
            {
                "name": entry.name,
                "path": str(entry.relative_to(base_dir)),
                "file_count": stats["file_count"],
                "total_bytes": stats["total_bytes"],
            }
        )
    return directories


def is_supported_file(path: Path) -> bool:
    suffix = path.suffix.lower()
    return suffix in ALLOWED_SUFFIXES


def load_manifest(path: Path) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:  # pragma: no cover - corrupted manifest fallback
        logger.warning("Failed to read manifest at %s; starting fresh", path)
        return {}


def save_manifest(path: Path, manifest: dict[str, dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def ingest_directory(
    *,
    project_id: str,
    target_dir: Path,
    base_dir: Path,
    ingestion_service: DocumentIngestor,
    manifest_path: Path,
    job_manager: IngestionJobManager | None = None,
    job_id: str | None = None,
) -> DocumentMetadata:
    """Ingest all supported files under target_dir into the given project."""

    manifest = load_manifest(manifest_path)
    embedding_model_id = getattr(ingestion_service, "embedding_model_id", None)

    # Determine pending files before acquiring the processing slot so we can expose totals.
    all_supported_files: list[Path] = []
    for file_path in sorted(target_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if not is_supported_file(file_path):
            continue
        all_supported_files.append(file_path)

    pending_files: list[Path] = []
    total_target_bytes = 0
    for file_path in all_supported_files:
        stat = file_path.stat()
        rel_path = str(file_path.relative_to(base_dir))
        entry = manifest.get(rel_path)
        if entry and entry.get("status") == "completed" and entry.get("mtime_ns") == stat.st_mtime_ns:
            if embedding_model_id is None:
                continue
            recorded_model = entry.get("embedding_model")
            if recorded_model == embedding_model_id:
                continue
        pending_files.append(file_path)
        total_target_bytes += stat.st_size

    processed_files = 0
    processed_bytes = 0
    total_bytes = 0
    total_chunks = 0
    completed_files = 0

    if job_manager and job_id:
        job_manager.mark_processing(
            job_id,
            total_files=len(pending_files),
            processed_files=0,
            total_bytes=total_target_bytes,
            processed_bytes=0,
        )

    logger.info(
        "local.ingest.start project=%s dir=%s pending_files=%s pending_bytes=%s",
        project_id,
        target_dir,
        len(pending_files),
        total_target_bytes,
    )

    try:
        for file_path in pending_files:
            rel_path = str(file_path.relative_to(base_dir))
            stat = file_path.stat()

            content_type, _ = mimetypes.guess_type(file_path.name)
            try:
                if job_manager:
                    job_manager.wait_for_rate(project_id)
                data = file_path.read_bytes()
                result = ingestion_service.ingest_bytes(
                    project_id,
                    filename=rel_path,
                    data=data,
                    content_type=content_type,
                )
                entry: dict[str, object] = {
                    "mtime_ns": stat.st_mtime_ns,
                    "status": "completed",
                }
                if embedding_model_id is not None:
                    entry["embedding_model"] = embedding_model_id
                manifest[rel_path] = entry
                total_bytes += stat.st_size
                total_chunks += result.chunks_ingested
                completed_files += 1
                processed_bytes += stat.st_size
                logger.info(
                    "local.ingest.file.completed project=%s dir=%s file=%s size_bytes=%s processed_files=%s/%s processed_bytes=%s/%s",
                    project_id,
                    target_dir,
                    rel_path,
                    stat.st_size,
                    processed_files + 1,
                    len(pending_files),
                    processed_bytes,
                    total_target_bytes,
                )
            except Exception as exc:  # pragma: no cover - per-file failure path
                logger.exception("Failed to ingest %s: %s", rel_path, exc)
                manifest[rel_path] = {
                    "mtime_ns": stat.st_mtime_ns,
                    "status": "failed",
                    "error": str(exc),
                }
                if embedding_model_id is not None:
                    manifest[rel_path]["embedding_model"] = embedding_model_id
                logger.warning(
                    "local.ingest.file.failed project=%s dir=%s file=%s size_bytes=%s processed_files=%s/%s",
                    project_id,
                    target_dir,
                    rel_path,
                    stat.st_size,
                    processed_files,
                    len(pending_files),
                )
            finally:
                processed_files += 1
                if job_manager and job_id:
                    job_manager.update_progress(
                        job_id,
                        processed_files=processed_files,
                        total_files=len(pending_files),
                        processed_bytes=processed_bytes,
                        total_bytes=total_target_bytes,
                    )
                save_manifest(manifest_path, manifest)

        metadata = DocumentMetadata(
            id=uuid4().hex,
            filename=str(target_dir.relative_to(base_dir)),
            chunk_count=total_chunks,
            created_at=DocumentIngestor._now(),
            size_bytes=total_bytes if total_bytes else None,
            title=target_dir.name,
            content_type="application/x-directory",
        )

        if job_manager and job_id:
            job_manager.mark_completed(
                job_id,
                metadata,
                processed_files=processed_files,
                processed_bytes=processed_bytes,
            )

        logger.info(
            "local.ingest.completed project=%s dir=%s files=%s chunks=%s bytes=%s",
            project_id,
            target_dir,
            completed_files,
            total_chunks,
            processed_bytes,
        )
        return metadata
    except Exception as exc:
        if job_manager and job_id:
            job_manager.mark_failed(
                job_id,
                str(exc),
                processed_files=processed_files,
                total_files=len(pending_files) if pending_files else len(all_supported_files),
            )
        logger.exception("local.ingest.failed project=%s dir=%s error=%s", project_id, target_dir, exc)
        raise
