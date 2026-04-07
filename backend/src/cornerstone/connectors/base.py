from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from cornerstone.domain.enums import SyncMode, VisibilityClass


@dataclass(slots=True)
class ParsedArtifact:
    external_id: str
    title: str
    artifact_type: str
    source_locator: str
    content_text: str
    content_hash: str
    source_updated_at: datetime
    metadata: dict[str, Any]
    support_fragments: list[tuple[str, str]]


@dataclass(slots=True)
class PreviewArtifact:
    upstream_id: str
    title: str
    artifact_type: str
    source_locator: str | None
    excerpt: str | None
    source_updated_at: datetime | None


@dataclass(slots=True)
class PreparedConnection:
    provider: str
    template_key: str
    source_boundary_locator: str
    selected_scope: dict[str, Any]
    visibility_class: VisibilityClass
    sync_mode: SyncMode
    sync_interval_seconds: int
    effective_sync_policy: dict[str, Any] = field(default_factory=dict)
    preview_items: list[PreviewArtifact] = field(default_factory=list)


@dataclass(slots=True)
class ProviderSyncResult:
    parsed_artifacts: list[ParsedArtifact]
    sync_checkpoint: dict[str, Any] = field(default_factory=dict)
    effective_sync_policy: dict[str, Any] = field(default_factory=dict)


class ConnectorAdapter(Protocol):
    provider: str

    def prepare_connection(
        self,
        *,
        template_key: str,
        source_label: str,
        selected_scope_input: str,
        visibility_class: VisibilityClass,
        settings,
        provider_credential=None,
    ) -> PreparedConnection: ...

    def sync(
        self,
        *,
        connection,
        settings,
        provider_credential=None,
    ) -> ProviderSyncResult: ...
