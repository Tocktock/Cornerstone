"""Helpers for ingesting local filesystem directories."""

from __future__ import annotations

import json
import logging
import mimetypes
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

    relative_path = relative_path.strip().lstrip("/\\")
    target = (base_dir / relative_path).resolve()
    if not str(target).startswith(str(base_dir.resolve())):
        raise ValueError("Path must reside inside the local data directory")
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


def list_directories(base_dir: Path, relative_path: str | None = None) -> list[dict[str, str]]:
    """Return immediate subdirectories with aggregate stats."""

    target = resolve_local_path(base_dir, relative_path or "")
    if not target.exists():
        return []
    directories: list[dict[str, str]] = []
    for entry in sorted(target.iterdir()):
        if entry.is_dir():
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

    if job_manager and job_id:
        job_manager.mark_processing(job_id)

    manifest = load_manifest(manifest_path)
    total_bytes = 0
    total_chunks = 0
    completed_files = 0

    try:
        files: Iterable[Path] = sorted(target_dir.rglob("*"))
        for file_path in files:
            if not file_path.is_file():
                continue
            if not is_supported_file(file_path):
                continue

            rel_path = str(file_path.relative_to(base_dir))
            stat = file_path.stat()
            entry = manifest.get(rel_path)
            if entry and entry.get("mtime_ns") == stat.st_mtime_ns and entry.get("status") == "completed":
                continue

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
                manifest[rel_path] = {
                    "mtime_ns": stat.st_mtime_ns,
                    "status": "completed",
                }
                total_bytes += stat.st_size
                total_chunks += result.chunks_ingested
                completed_files += 1
            except Exception as exc:  # pragma: no cover - per-file failure path
                logger.exception("Failed to ingest %s: %s", rel_path, exc)
                manifest[rel_path] = {
                    "mtime_ns": stat.st_mtime_ns,
                    "status": "failed",
                    "error": str(exc),
                }
            finally:
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
            job_manager.mark_completed(job_id, metadata)

        logger.info(
            "local.ingest.completed project=%s dir=%s files=%s chunks=%s",
            project_id,
            target_dir,
            completed_files,
            total_chunks,
        )
        return metadata
    except Exception as exc:
        if job_manager and job_id:
            job_manager.mark_failed(job_id, str(exc))
        logger.exception("local.ingest.failed project=%s dir=%s error=%s", project_id, target_dir, exc)
        raise
