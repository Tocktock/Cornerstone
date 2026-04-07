from __future__ import annotations

from dataclasses import dataclass

from cornerstone.connectors.filesystem import FilesystemConnector
from cornerstone.connectors.notion import NotionConnector
from cornerstone.domain.enums import VisibilityClass


@dataclass(frozen=True, slots=True)
class ConnectorTemplate:
    template_key: str
    provider: str
    label: str
    description: str
    scope_kind: str
    default_visibility_class: VisibilityClass
    recommended_sync_interval_seconds: int
    preview_required: bool = True
    internal_only: bool = False


CONNECTOR_TEMPLATES: dict[str, ConnectorTemplate] = {
    "notion_shared_page_tree": ConnectorTemplate(
        template_key="notion_shared_page_tree",
        provider="notion",
        label="Notion shared page tree",
        description="Connect one root Notion page and sync descendant pages under that tree.",
        scope_kind="page_tree",
        default_visibility_class=VisibilityClass.MEMBER_VISIBLE,
        recommended_sync_interval_seconds=900,
    ),
    "notion_shared_database": ConnectorTemplate(
        template_key="notion_shared_database",
        provider="notion",
        label="Notion shared database",
        description="Connect one Notion database and sync database entry pages as source memory.",
        scope_kind="database",
        default_visibility_class=VisibilityClass.MEMBER_VISIBLE,
        recommended_sync_interval_seconds=900,
    ),
    "member-visible": ConnectorTemplate(
        template_key="member-visible",
        provider="filesystem",
        label="Filesystem member-visible fixture",
        description="Internal demo fixture connector.",
        scope_kind="filesystem_root",
        default_visibility_class=VisibilityClass.MEMBER_VISIBLE,
        recommended_sync_interval_seconds=300,
        internal_only=True,
    ),
    "evidence-only": ConnectorTemplate(
        template_key="evidence-only",
        provider="filesystem",
        label="Filesystem evidence-only fixture",
        description="Internal demo fixture connector.",
        scope_kind="filesystem_root",
        default_visibility_class=VisibilityClass.EVIDENCE_ONLY,
        recommended_sync_interval_seconds=300,
        internal_only=True,
    ),
    "personal-private": ConnectorTemplate(
        template_key="personal-private",
        provider="filesystem",
        label="Filesystem personal fixture",
        description="Internal demo fixture connector.",
        scope_kind="filesystem_root",
        default_visibility_class=VisibilityClass.EVIDENCE_ONLY,
        recommended_sync_interval_seconds=300,
        internal_only=True,
    ),
    "stale-snapshot": ConnectorTemplate(
        template_key="stale-snapshot",
        provider="filesystem",
        label="Filesystem stale snapshot fixture",
        description="Internal demo fixture connector.",
        scope_kind="filesystem_root",
        default_visibility_class=VisibilityClass.MEMBER_VISIBLE,
        recommended_sync_interval_seconds=300,
        internal_only=True,
    ),
    "degraded": ConnectorTemplate(
        template_key="degraded",
        provider="filesystem",
        label="Filesystem degraded fixture",
        description="Internal demo fixture connector.",
        scope_kind="filesystem_root",
        default_visibility_class=VisibilityClass.EVIDENCE_ONLY,
        recommended_sync_interval_seconds=300,
        internal_only=True,
    ),
    "paused": ConnectorTemplate(
        template_key="paused",
        provider="filesystem",
        label="Filesystem paused fixture",
        description="Internal demo fixture connector.",
        scope_kind="filesystem_root",
        default_visibility_class=VisibilityClass.MEMBER_VISIBLE,
        recommended_sync_interval_seconds=300,
        internal_only=True,
    ),
    "removed": ConnectorTemplate(
        template_key="removed",
        provider="filesystem",
        label="Filesystem removed fixture",
        description="Internal demo fixture connector.",
        scope_kind="filesystem_root",
        default_visibility_class=VisibilityClass.MEMBER_VISIBLE,
        recommended_sync_interval_seconds=300,
        internal_only=True,
    ),
}


def get_connector_template(template_key: str) -> ConnectorTemplate:
    try:
        return CONNECTOR_TEMPLATES[template_key]
    except KeyError as exc:  # pragma: no cover - simple guard
        raise ValueError(f"Unknown connector template: {template_key}") from exc


def list_manager_templates() -> list[ConnectorTemplate]:
    return [template for template in CONNECTOR_TEMPLATES.values() if not template.internal_only]


def get_connector_adapter(provider: str, locator: str | None = None):
    if provider == "filesystem":
        return FilesystemConnector(locator or "")
    if provider == "notion":
        return NotionConnector()
    raise ValueError(f"Unsupported connector provider: {provider}")
