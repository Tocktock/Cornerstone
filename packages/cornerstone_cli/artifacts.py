from __future__ import annotations

import mimetypes
import re
from pathlib import Path
from typing import Any


MAX_BROWSER_UPLOAD_BYTES = 25 * 1024 * 1024


class ArtifactUploadError(ValueError):
    def __init__(self, code: str, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


def _safe_filename(value: str) -> str:
    filename = Path(value.replace("\x00", "")).name.strip()
    filename = re.sub(r"[\x00-\x1f\x7f]", "", filename)
    if not filename:
        filename = "upload.bin"
    return filename[:255]


_MEDIA_TYPE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9!#$&^_.+\-]*/[A-Za-z0-9][A-Za-z0-9!#$&^_.+\-]*$")


def normalize_media_type(value: str, filename: str = "") -> str:
    declared = value.split(";", 1)[0].strip().lower()
    if declared and declared != "application/octet-stream" and _MEDIA_TYPE_RE.fullmatch(declared):
        return declared
    if not declared or declared == "application/octet-stream" or not _MEDIA_TYPE_RE.fullmatch(declared):
        guessed, _ = mimetypes.guess_type(filename)
        if guessed and _MEDIA_TYPE_RE.fullmatch(guessed):
            return guessed
    return "application/octet-stream"


def artifact_presentation(
    artifact: dict[str, Any],
    *,
    text_available: bool | None = None,
    original_available: bool | None = None,
) -> dict[str, Any]:
    derived = artifact.get("derived") if isinstance(artifact.get("derived"), dict) else {}
    status = str(derived.get("status") or "missing").lower()
    has_text_ref = bool(str(derived.get("text_ref") or "").strip())
    has_text = has_text_ref and text_available is not False
    if status == "ready" and has_text:
        label, state = "Searchable", "searchable"
        explanation = "A readable derived representation is available for Search and evidence review."
        recovery = "Open the source text or use Search."
    elif status in {"processing", "pending", "queued", "not_started"}:
        label, state = "Processing", "underReview"
        explanation = "The original is saved while a searchable representation is still being prepared."
        recovery = "Keep the original and retry derived processing when a parser is available."
    elif status == "partial" or (status == "ready" and not has_text):
        label, state = "Partial", "underReview"
        explanation = "Only part of the source is available as a readable representation."
        recovery = "Inspect the original and retry extraction before relying on Search coverage."
    elif status == "failed":
        label, state = "Extraction failed", "failed"
        explanation = "The original is saved, but readable extraction did not complete."
        recovery = str(derived.get("message") or "Retry extraction; the original bytes remain unchanged.")
    elif status in {"deferred", "unsupported"}:
        label, state = "Unsupported preview", "underReview"
        explanation = "The original is saved, but this format does not have a readable preview yet."
        recovery = str(derived.get("message") or "Download the original or add a compatible parser.")
    else:
        label, state = "Saved", "saved"
        explanation = "The original is saved; searchable readiness has not been established."
        recovery = "Inspect the original before relying on Search coverage."
    saved_metadata = bool(artifact.get("original_storage_ref") and artifact.get("checksum_sha256"))
    saved = saved_metadata and original_available is not False
    if not saved:
        label, state = "Integrity issue", "failed"
        explanation = "The original source is unavailable or failed its integrity check."
        recovery = "Restore the verified original before using this record or its derived representation."
    return {
        "saved": saved,
        "searchable": saved and status in {"ready", "partial"} and has_text,
        "preview_supported": saved and status in {"ready", "partial"} and has_text,
        "derived_status": status,
        "label": label,
        "state": state,
        "explanation": explanation,
        "recovery": recovery,
    }


class ArtifactApplication:
    """Deep artifact module for byte preservation, paste intake, and presentation truth."""

    def __init__(self, store: Any) -> None:
        self.store = store

    def ingest_paste(
        self,
        text: str,
        scope: dict[str, str],
        *,
        source_ref: str = "home.drop_text",
    ) -> dict[str, Any]:
        if not text.strip():
            raise ArtifactUploadError("CS_ARTIFACT_TEXT_REQUIRED", "Paste text before saving.")
        return self.store.ingest_text_artifact(
            text,
            scope,
            source_type="user_paste",
            source_ref=source_ref,
            trust="untrusted",
        )

    def ingest_file(
        self,
        content: bytes,
        scope: dict[str, str],
        *,
        filename: str,
        media_type: str,
        source_ref: str = "home.file",
    ) -> dict[str, Any]:
        if len(content) > MAX_BROWSER_UPLOAD_BYTES:
            raise ArtifactUploadError(
                "CS_ARTIFACT_UPLOAD_TOO_LARGE",
                f"File exceeds the {MAX_BROWSER_UPLOAD_BYTES // (1024 * 1024)} MB local upload limit.",
                status_code=413,
            )
        safe_filename = _safe_filename(filename)
        normalized_media_type = normalize_media_type(media_type, safe_filename)
        return self.store.ingest_artifact_bytes(
            content,
            filename=safe_filename,
            **scope,
            source="browser_upload",
            source_ref=source_ref,
            media_type=normalized_media_type,
            derived_mode="auto",
            trust="untrusted",
            lineage_from=None,
        )

    def read_original(
        self,
        artifact_id: str,
        scope: dict[str, str],
        *,
        reason: str,
    ) -> dict[str, Any]:
        return self.store.read_artifact_original(artifact_id, scope, reason=reason)

    def presentation(self, artifact: dict[str, Any]) -> dict[str, Any]:
        return artifact_presentation(
            artifact,
            text_available=self.store.derived_text_available(artifact),
            original_available=self.store.original_available(artifact),
        )
