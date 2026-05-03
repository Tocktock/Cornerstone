from __future__ import annotations

from pathlib import Path
from typing import Any

from cornerstone.schemas import SourceObject

_TEXT_MIME_PREFIX = "text/"
_SUPPORTED_MIME_TYPES = {
    "application/json",
    "application/x-json",
    "application/jsonl",
    "application/x-ndjson",
    "application/xml",
    "application/yaml",
    "application/x-yaml",
}
_BLOCKED_SUFFIXES = {
    ".doc",
    ".docx",
    ".gif",
    ".jpeg",
    ".jpg",
    ".pdf",
    ".png",
    ".ppt",
    ".pptx",
    ".webp",
    ".xls",
    ".xlsx",
    ".zip",
}
_SUPPORTED_SUFFIXES = {
    ".csv",
    ".htm",
    ".html",
    ".json",
    ".jsonl",
    ".log",
    ".markdown",
    ".md",
    ".rst",
    ".text",
    ".tsv",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}


class ManualUploadError(ValueError):
    """Base class for manual upload validation failures."""


class ManualUploadTooLargeError(ManualUploadError):
    """Raised when an uploaded file exceeds configured limits."""


class ManualUploadUnsupportedTypeError(ManualUploadError):
    """Raised when an uploaded file is not a supported text-like type."""


class ManualUploadEmptyFileError(ManualUploadError):
    """Raised when an uploaded file contains no useful text."""


def build_source_object_from_upload(
    *,
    filename: str | None,
    content_type: str | None,
    data: bytes,
    max_file_bytes: int,
) -> SourceObject:
    """Normalize one UTF-8 text-like upload into a provider-neutral SourceObject.

    v1.3.1 intentionally supports only text-like files. Binary files, PDFs, and Office
    documents are rejected before Artifact/Evidence creation so downstream extraction
    can remain deterministic and quote offsets stay explainable.
    """

    display_filename = _clean_filename(filename)
    if len(data) > max_file_bytes:
        raise ManualUploadTooLargeError(
            f"Uploaded file exceeds max size: {display_filename} ({len(data)} bytes > {max_file_bytes} bytes)"
        )
    if not data:
        raise ManualUploadEmptyFileError(f"Uploaded file is empty: {display_filename}")
    if _looks_like_binary(data):
        raise ManualUploadUnsupportedTypeError(f"Uploaded file appears to be binary: {display_filename}")
    normalized_content_type = _normalize_content_type(content_type)
    suffix = Path(display_filename).suffix.casefold()
    if not _is_supported_text_type(normalized_content_type, suffix):
        raise ManualUploadUnsupportedTypeError(
            f"Unsupported manual upload type for {display_filename}: "
            f"contentType={normalized_content_type or 'unknown'}, suffix={suffix or 'none'}"
        )
    try:
        decoded = data.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ManualUploadUnsupportedTypeError(
            f"Uploaded file must be UTF-8 text: {display_filename}"
        ) from exc
    normalized_content = decoded.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized_content.strip():
        raise ManualUploadEmptyFileError(f"Uploaded file has no text content: {display_filename}")

    return SourceObject(
        source_external_id=f"manual-upload:{display_filename}",
        title=display_filename,
        content=normalized_content,
        source_object_type="uploaded_file",
        provider_metadata={
            "uploadKind": "manual_file",
            "fileName": display_filename,
            "contentType": normalized_content_type or None,
            "sizeBytes": len(data),
            "encoding": "utf-8",
        },
    )


def build_source_object_from_text_upload(
    *,
    title: str,
    content: str,
    source_external_id: str | None = None,
    source_url: str | None = None,
    source_updated_at: Any = None,
    provider_metadata: dict[str, Any] | None = None,
) -> SourceObject:
    """Normalize pasted/manual text into a SourceObject.

    This is separate from the legacy `/sync` path so clients can model manual uploads
    without crafting provider-shaped SourceObjects themselves.
    """

    cleaned_title = " ".join(title.strip().split())
    if not cleaned_title:
        raise ManualUploadEmptyFileError("Manual text upload requires a title.")
    normalized_content = content.replace("\r\n", "\n").replace("\r", "\n")
    if not normalized_content.strip():
        raise ManualUploadEmptyFileError(f"Manual text upload has no text content: {cleaned_title}")
    external_id = source_external_id or f"manual-upload:text:{cleaned_title}"
    metadata = dict(provider_metadata or {})
    metadata.setdefault("uploadKind", "manual_text")
    metadata.setdefault("title", cleaned_title)
    return SourceObject(
        source_external_id=external_id,
        title=cleaned_title,
        content=normalized_content,
        source_url=source_url,
        source_updated_at=source_updated_at,
        source_object_type="uploaded_text",
        provider_metadata=metadata,
    )


def _clean_filename(filename: str | None) -> str:
    value = (filename or "uploaded-file.txt").replace("\\", "/").split("/")[-1]
    cleaned = " ".join(value.strip().split())
    return cleaned or "uploaded-file.txt"


def _normalize_content_type(content_type: str | None) -> str:
    return (content_type or "").split(";", 1)[0].strip().casefold()


def _is_supported_text_type(content_type: str, suffix: str) -> bool:
    if suffix in _BLOCKED_SUFFIXES:
        return False
    if content_type.startswith(_TEXT_MIME_PREFIX):
        return True
    if content_type in _SUPPORTED_MIME_TYPES:
        return True
    if suffix in _SUPPORTED_SUFFIXES and content_type in {"", "application/octet-stream"}:
        return True
    return suffix in _SUPPORTED_SUFFIXES and content_type in _SUPPORTED_MIME_TYPES


def _looks_like_binary(data: bytes) -> bool:
    if data.startswith(b"%PDF-"):
        return True
    sample = data[:1024]
    return b"\x00" in sample
