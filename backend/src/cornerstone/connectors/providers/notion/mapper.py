from __future__ import annotations

from typing import Any

from cornerstone.connectors.providers.common import parse_datetime, stable_hash

from cornerstone.schemas import (
    DataSourceType,
    ProviderObjectAccessState,
    ProviderObjectSnapshot,
    ProviderObjectType,
    SourceObject,
    utc_now,
)


def notion_result_to_snapshot(datasource_id: str, result: dict[str, Any]) -> ProviderObjectSnapshot:
    external_id = str(result.get("id") or "")
    object_type = map_notion_object_type(str(result.get("object") or "unknown"))
    provider_metadata = dict(result.get("provider_metadata") or {})
    provider_metadata.update(
        {
            "providerObject": str(result.get("object") or "unknown"),
            "archived": bool(result.get("archived", False)),
            "inTrash": bool(result.get("in_trash", False)),
        }
    )
    ingestion_supported, ingestion_unsupported_reason = notion_ingestion_support(object_type)
    provider_metadata.update(
        {
            "ingestionSupported": ingestion_supported,
            "ingestionUnsupportedReason": ingestion_unsupported_reason,
        }
    )
    return ProviderObjectSnapshot(
        datasource_id=datasource_id,
        provider=DataSourceType.NOTION,
        external_id=external_id,
        external_url=result.get("url"),
        object_type=object_type,
        title=extract_title(result),
        parent_external_id=extract_parent_id(result.get("parent") or {}),
        last_edited_time=parse_datetime(result.get("last_edited_time")),
        discovered_at=utc_now(),
        access_state=(
            ProviderObjectAccessState.DELETED
            if bool(result.get("in_trash", False))
            else ProviderObjectAccessState.ACCESSIBLE
        ),
        ingestion_supported=ingestion_supported,
        ingestion_unsupported_reason=ingestion_unsupported_reason,
        raw_metadata_hash=stable_hash(result),
        provider_metadata=provider_metadata,
    )


def page_to_source_object(
    *,
    page: dict[str, Any],
    snapshot: ProviderObjectSnapshot,
    content: str,
    content_format: str,
) -> SourceObject:
    title = snapshot.title or extract_title(page) or snapshot.external_id
    source_updated_at = parse_datetime(page.get("last_edited_time")) or snapshot.last_edited_time
    return SourceObject(
        source_external_id=snapshot.external_id,
        title=title,
        content=content.strip(),
        source_url=page.get("url") or snapshot.external_url,
        source_updated_at=source_updated_at,
        source_object_type=str(snapshot.object_type),
        provider_metadata={
            "provider": "notion",
            "objectType": str(snapshot.object_type),
            "snapshotId": snapshot.id,
            "parentExternalId": snapshot.parent_external_id,
            "lastEditedTime": source_updated_at.isoformat() if source_updated_at is not None else None,
            "contentFormat": content_format,
        },
    )


def block_to_plain_text(block: dict[str, Any]) -> str:
    block_type = str(block.get("type") or "")
    payload = block.get(block_type)
    if not isinstance(payload, dict):
        return ""
    if block_type in {"paragraph", "heading_1", "heading_2", "heading_3", "quote", "callout", "toggle"}:
        return rich_text_plain_text(payload.get("rich_text") or []) or ""
    if block_type == "bulleted_list_item":
        text = rich_text_plain_text(payload.get("rich_text") or []) or ""
        return f"- {text}" if text else ""
    if block_type == "numbered_list_item":
        text = rich_text_plain_text(payload.get("rich_text") or []) or ""
        return f"1. {text}" if text else ""
    if block_type == "to_do":
        text = rich_text_plain_text(payload.get("rich_text") or []) or ""
        checked = "x" if payload.get("checked") else " "
        return f"[{checked}] {text}" if text else ""
    if block_type == "code":
        text = rich_text_plain_text(payload.get("rich_text") or []) or ""
        language = payload.get("language") or "text"
        return f"```{language}\n{text}\n```" if text else ""
    if block_type in {"child_page", "child_database"}:
        return str(payload.get("title") or "").strip()
    return rich_text_plain_text(payload.get("rich_text") or []) or ""


def notion_ingestion_support(object_type: ProviderObjectType) -> tuple[bool, str | None]:
    if object_type == ProviderObjectType.PAGE:
        return True, None
    if object_type == ProviderObjectType.DATABASE:
        return False, "Notion database ingestion is not implemented in this backend slice. Select Notion pages instead."
    if object_type == ProviderObjectType.DATA_SOURCE:
        return False, "Notion data_source ingestion is not implemented in this backend slice. Select Notion pages instead."
    if object_type == ProviderObjectType.BLOCK:
        return False, "Notion blocks are ingested through their parent page and cannot be selected as top-level sync objects."
    return False, "This Notion object type is discoverable but not supported for ingestion yet."


def map_notion_object_type(value: str) -> ProviderObjectType:
    if value == "page":
        return ProviderObjectType.PAGE
    if value == "database":
        return ProviderObjectType.DATABASE
    if value == "data_source":
        return ProviderObjectType.DATA_SOURCE
    if value == "block":
        return ProviderObjectType.BLOCK
    return ProviderObjectType.UNKNOWN


def extract_title(result: dict[str, Any]) -> str | None:
    if "title" in result and isinstance(result["title"], list):
        return rich_text_plain_text(result["title"])
    properties = result.get("properties") or {}
    for value in properties.values():
        if isinstance(value, dict) and value.get("type") == "title":
            return rich_text_plain_text(value.get("title") or [])
    return None


def rich_text_plain_text(items: list[dict[str, Any]]) -> str | None:
    text = "".join(str(item.get("plain_text") or "") for item in items).strip()
    return text or None


def extract_parent_id(parent: dict[str, Any]) -> str | None:
    for key in ("page_id", "database_id", "data_source_id", "block_id", "workspace"):
        value = parent.get(key)
        if isinstance(value, str):
            return value
    return None
