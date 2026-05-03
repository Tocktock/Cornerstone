from __future__ import annotations

from typing import Any

from cornerstone.connectors.providers.common import parse_datetime, stable_hash

from cornerstone.schemas import (
    DataSourceType,
    ProviderObjectAccessState,
    ProviderObjectSnapshot,
    ProviderObjectType,
    SourceObject,
)

GOOGLE_DOC_MIME_TYPE = "application/vnd.google-apps.document"
GOOGLE_SHEET_MIME_TYPE = "application/vnd.google-apps.spreadsheet"
GOOGLE_SLIDE_MIME_TYPE = "application/vnd.google-apps.presentation"
GOOGLE_FOLDER_MIME_TYPE = "application/vnd.google-apps.folder"
PDF_MIME_TYPE = "application/pdf"
TEXT_MIME_TYPES = {
    "text/plain",
    "text/markdown",
    "text/csv",
    "application/json",
    "application/xml",
}


def drive_file_to_snapshot(datasource_id: str, file_payload: dict[str, Any]) -> ProviderObjectSnapshot:
    mime_type = str(file_payload.get("mimeType") or "")
    object_type = drive_object_type(mime_type)
    access_state = _access_state(file_payload)
    ingestion_supported, unsupported_reason = drive_ingestion_support(object_type, access_state)
    provider_metadata = {
        "mimeType": mime_type,
        "driveId": file_payload.get("driveId"),
        "capabilities": file_payload.get("capabilities") or {},
    }
    return ProviderObjectSnapshot(
        datasource_id=datasource_id,
        provider=DataSourceType.GOOGLE_DRIVE,
        external_id=str(file_payload.get("id") or ""),
        external_url=file_payload.get("webViewLink"),
        object_type=object_type,
        title=file_payload.get("name"),
        parent_external_id=_first_parent(file_payload),
        last_edited_time=parse_datetime(file_payload.get("modifiedTime")),
        selected_for_sync=False,
        access_state=access_state,
        raw_metadata_hash=stable_hash(file_payload),
        provider_metadata=provider_metadata,
        ingestion_supported=ingestion_supported,
        ingestion_unsupported_reason=unsupported_reason,
    )


def drive_file_to_source_object(
    *,
    file_payload: dict[str, Any],
    snapshot: ProviderObjectSnapshot,
    content: str,
    content_format: str,
) -> SourceObject:
    source_updated_at = parse_datetime(file_payload.get("modifiedTime")) or snapshot.last_edited_time
    return SourceObject(
        source_external_id=snapshot.external_id,
        title=str(file_payload.get("name") or snapshot.title or "Untitled Google Drive file"),
        content=content,
        source_url=file_payload.get("webViewLink") or snapshot.external_url,
        source_updated_at=source_updated_at,
        source_object_type=str(snapshot.object_type),
        provider_metadata={
            "provider": "google_drive",
            "objectType": str(snapshot.object_type),
            "mimeType": file_payload.get("mimeType"),
            "snapshotId": snapshot.id,
            "parentExternalId": snapshot.parent_external_id,
            "modifiedTime": source_updated_at.isoformat() if source_updated_at is not None else None,
            "contentFormat": content_format,
        },
    )


def drive_object_type(mime_type: str) -> ProviderObjectType:
    if mime_type == GOOGLE_DOC_MIME_TYPE:
        return ProviderObjectType.DOCUMENT
    if mime_type == GOOGLE_SHEET_MIME_TYPE:
        return ProviderObjectType.SPREADSHEET
    if mime_type == GOOGLE_SLIDE_MIME_TYPE:
        return ProviderObjectType.PRESENTATION
    if mime_type == GOOGLE_FOLDER_MIME_TYPE:
        return ProviderObjectType.FOLDER
    if mime_type == PDF_MIME_TYPE:
        return ProviderObjectType.PDF
    if mime_type in TEXT_MIME_TYPES:
        return ProviderObjectType.TEXT_FILE
    return ProviderObjectType.FILE


def drive_ingestion_support(
    object_type: ProviderObjectType,
    access_state: ProviderObjectAccessState = ProviderObjectAccessState.ACCESSIBLE,
) -> tuple[bool, str | None]:
    if access_state != ProviderObjectAccessState.ACCESSIBLE:
        return False, "Google Drive file is not accessible or cannot be downloaded."
    if object_type == ProviderObjectType.DOCUMENT:
        return True, None
    if object_type == ProviderObjectType.TEXT_FILE:
        return True, None
    if object_type == ProviderObjectType.SPREADSHEET:
        return False, "Google Sheets are discoverable but not ingestible in this backend slice."
    if object_type == ProviderObjectType.PRESENTATION:
        return False, "Google Slides are discoverable but not ingestible in this backend slice."
    if object_type == ProviderObjectType.PDF:
        return False, "PDF ingestion requires extraction/OCR policy and is intentionally deferred."
    if object_type == ProviderObjectType.FOLDER:
        return False, "Google Drive folders can be discovered but cannot be ingested as source evidence."
    return False, "This Google Drive file type is discoverable but not supported for ingestion yet."


def _access_state(file_payload: dict[str, Any]) -> ProviderObjectAccessState:
    if bool(file_payload.get("trashed", False)):
        return ProviderObjectAccessState.DELETED
    capabilities = file_payload.get("capabilities") or {}
    can_download = capabilities.get("canDownload")
    if can_download is False:
        return ProviderObjectAccessState.INACCESSIBLE
    return ProviderObjectAccessState.ACCESSIBLE


def _first_parent(file_payload: dict[str, Any]) -> str | None:
    parents = file_payload.get("parents")
    if isinstance(parents, list) and parents:
        first = parents[0]
        if isinstance(first, str):
            return first
    return None
