from __future__ import annotations

import asyncio

import pytest

from cornerstone.config import Settings
from cornerstone.connectors.registry import get_connector
from cornerstone.schemas import (
    Artifact,
    ConnectorAuthType,
    ConnectorCredential,
    CredentialStatus,
    DataSourceType,
    ProviderObjectAccessState,
    ProviderObjectSnapshot,
    ProviderObjectType,
    SourceSelection,
)

pytestmark = pytest.mark.unit


def test_core_artifact_schema_is_provider_agnostic() -> None:
    artifact = Artifact(
        datasource_id="source-1",
        source_type=DataSourceType.GITHUB,
        source_external_id="repo/path/file.py",
        source_object_type="file",
        source_url="https://github.com/acme/repo/blob/main/file.py",
        title="file.py",
        raw_content_hash="a" * 64,
        provider_metadata={"repository": "acme/repo", "branch": "main"},
    )

    payload = artifact.model_dump(mode="json", by_alias=True)

    assert "notionPageId" not in payload
    assert payload["sourceExternalId"] == "repo/path/file.py"
    assert payload["sourceObjectType"] == "file"
    assert payload["providerMetadata"]["repository"] == "acme/repo"


def test_provider_object_snapshot_supports_provider_metadata_without_notion_lock_in() -> None:
    snapshot = ProviderObjectSnapshot(
        datasource_id="source-1",
        provider=DataSourceType.SLACK,
        external_id="C123:thread:456",
        object_type=ProviderObjectType.UNKNOWN,
        title="Launch decision thread",
        access_state=ProviderObjectAccessState.ACCESSIBLE,
        raw_metadata_hash="slack-thread-metadata-hash",
        provider_metadata={"channelId": "C123", "threadTs": "456", "messageCount": 12},
    )

    assert snapshot.provider == DataSourceType.SLACK
    assert snapshot.provider_metadata["channelId"] == "C123"
    assert snapshot.external_id == "C123:thread:456"
    assert snapshot.ingestion_supported is True


def test_connector_registry_can_resolve_non_notion_manual_connector() -> None:
    connector = get_connector(DataSourceType.MANUAL, Settings())

    assert connector.provider == DataSourceType.MANUAL


def test_manual_connector_can_discover_select_and_normalize_source_objects() -> None:
    connector = get_connector(DataSourceType.MANUAL, Settings())
    credential = ConnectorCredential(
        datasource_id="manual-source-1",
        provider=DataSourceType.MANUAL,
        auth_type=ConnectorAuthType.MANUAL,
        encrypted_access_token="manual-token-placeholder",
        status=CredentialStatus.ACTIVE,
    )

    page = asyncio.run(
        connector.discover_objects(
            credential=credential,
            access_token="manual-token-placeholder",
            page_size=25,
        )
    )
    selection = SourceSelection(
        datasource_id="manual-source-1",
        selected_external_object_ids=[page.objects[0].external_id],
    )
    objects = asyncio.run(
        connector.list_objects(
            credential=credential,
            access_token="manual-token-placeholder",
            selection=selection,
            selected_objects=page.objects,
        )
    )

    assert len(objects) == 1
    assert objects[0].source_external_id == "manual-note-1"
    assert objects[0].source_object_type == "manual_note"
    assert objects[0].provider_metadata["provider"] == "manual"
    assert "Cornerstone is a shared organizational context layer" in objects[0].content
