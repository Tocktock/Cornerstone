from __future__ import annotations

import asyncio
from datetime import timedelta
from urllib.parse import parse_qs, urlparse

import pytest

from cornerstone.config import Settings
from cornerstone.connectors.catalog import list_connector_definitions
from cornerstone.connectors.notion import NotionConnector
from cornerstone.connectors.security import TokenCipher
from cornerstone.persistence.models import (
    Base,
    ConnectionIntentRow,
    ConnectorCredentialRow,
    ProviderObjectSnapshotRow,
    SourceSelectionRow,
    SyncJobRow,
)
from cornerstone.schemas import (
    ConnectionIntent,
    ConnectorAuthType,
    ConnectorCredential,
    ConnectorErrorCode,
    ConnectorNextAction,
    CredentialStatus,
    DataSourceType,
    ProviderObjectAccessState,
    ProviderObjectType,
    utc_now,
)

pytestmark = pytest.mark.unit


def test_connector_catalog_exposes_notion_first_and_future_connectors() -> None:
    definitions = list_connector_definitions(Settings())

    assert definitions[0].provider == DataSourceType.NOTION
    assert definitions[0].auth_type == "oauth2"
    assert definitions[0].availability == "available"
    assert definitions[0].required_scopes[0].key == "read_content"
    assert definitions[0].supported_objects == ["page"]
    assert definitions[0].discoverable_objects == ["page", "database", "data_source"]
    assert definitions[0].ingestible_objects == ["page"]
    assert {definition.provider for definition in definitions} == {
        DataSourceType.NOTION,
        DataSourceType.SLACK,
        DataSourceType.GOOGLE_DOCS,
        DataSourceType.GITHUB,
        DataSourceType.MANUAL,
    }


def test_token_cipher_round_trip_and_no_plaintext_leak() -> None:
    cipher = TokenCipher("connector-test-secret-that-is-long-enough")

    encrypted = cipher.encrypt("notion-secret-token")

    assert encrypted != "notion-secret-token"
    assert "notion-secret-token" not in encrypted
    assert cipher.decrypt(encrypted) == "notion-secret-token"


def test_token_cipher_rejects_wrong_secret() -> None:
    first = TokenCipher("connector-test-secret-that-is-long-enough")
    second = TokenCipher("another-connector-secret-that-is-long-enough")
    encrypted = first.encrypt("notion-secret-token")

    with pytest.raises(ValueError, match="could not be decrypted"):
        second.decrypt(encrypted)


def test_notion_authorization_url_contains_required_oauth_state() -> None:
    settings = Settings()
    connector = NotionConnector(settings)
    intent = ConnectionIntent(
        provider=DataSourceType.NOTION,
        source_name="Team Notion",
        created_by="admin@example.com",
        redirect_uri="http://localhost:8000/v1/oauth/notion/callback",
        state_nonce="state-nonce-with-good-length",
        expires_at=utc_now() + timedelta(minutes=15),
    )

    authorization_url = connector.build_authorization_url(intent)
    parsed = urlparse(authorization_url)
    query = parse_qs(parsed.query)

    assert parsed.path == "/v1/oauth/authorize"
    assert query["client_id"] == [settings.notion_client_id]
    assert query["redirect_uri"] == [intent.redirect_uri]
    assert query["response_type"] == ["code"]
    assert query["owner"] == ["user"]
    assert query["state"] == [intent.state_nonce]


@pytest.mark.parametrize(
    ("status_code", "expected_code", "expected_action"),
    [
        (401, ConnectorErrorCode.TOKEN_REVOKED, ConnectorNextAction.RECONNECT),
        (403, ConnectorErrorCode.PERMISSION_DENIED, ConnectorNextAction.GRANT_PERMISSION),
        (429, ConnectorErrorCode.RATE_LIMITED, ConnectorNextAction.WAIT_AND_RETRY),
        (503, ConnectorErrorCode.PROVIDER_UNAVAILABLE, ConnectorNextAction.RETRY),
        (418, ConnectorErrorCode.UNKNOWN, ConnectorNextAction.CONTACT_ADMIN),
    ],
)
def test_notion_provider_errors_map_to_actionable_connector_errors(
    status_code: int,
    expected_code: ConnectorErrorCode,
    expected_action: ConnectorNextAction,
) -> None:
    error = NotionConnector(Settings()).map_provider_error(
        status_code=status_code,
        technical_message="provider payload",
        retry_after_seconds=3,
    )

    assert error.code == expected_code
    assert error.next_action == expected_action
    assert error.user_message


def test_persistence_schema_contains_connector_tables() -> None:
    table_names = set(Base.metadata.tables)

    assert {
        "connection_intents",
        "connector_credentials",
        "source_selections",
        "sync_jobs",
        "sync_job_events",
        "provider_object_snapshots",
    }.issubset(table_names)
    assert ConnectionIntentRow.__tablename__ == "connection_intents"
    assert ConnectorCredentialRow.__tablename__ == "connector_credentials"
    assert SourceSelectionRow.__tablename__ == "source_selections"
    assert SyncJobRow.__tablename__ == "sync_jobs"
    assert ProviderObjectSnapshotRow.__tablename__ == "provider_object_snapshots"


def test_connector_credentials_only_expose_encrypted_token_columns_in_persistence() -> None:
    column_names = set(ConnectorCredentialRow.__table__.columns.keys())

    assert "encrypted_access_token" in column_names
    assert "encrypted_refresh_token" in column_names
    assert "access_token" not in column_names
    assert "refresh_token" not in column_names


def test_notion_mock_discovery_returns_snapshots_for_selection() -> None:
    connector = NotionConnector(Settings())
    credential = ConnectorCredential(
        provider=DataSourceType.NOTION,
        datasource_id="source-1",
        auth_type=ConnectorAuthType.OAUTH2,
        encrypted_access_token="encrypted",
        granted_scopes=["read_content"],
        status=CredentialStatus.ACTIVE,
    )

    page = asyncio.run(connector.discover_objects(credential=credential, access_token="mock-token", cursor=None, page_size=10))

    assert len(page.objects) == 2
    assert page.objects[0].external_id == "notion-page-1"
    assert page.objects[0].object_type == ProviderObjectType.PAGE
    assert page.objects[0].access_state == ProviderObjectAccessState.ACCESSIBLE
    assert page.objects[0].selected_for_sync is False
    assert page.objects[0].ingestion_supported is True
    assert page.objects[1].object_type == ProviderObjectType.DATABASE
    assert page.objects[1].ingestion_supported is False
    assert "database ingestion" in (page.objects[1].ingestion_unsupported_reason or "")
    assert page.next_cursor is None


def test_provider_object_snapshot_table_tracks_discovery_and_selection_state() -> None:
    column_names = set(ProviderObjectSnapshotRow.__table__.columns.keys())
    index_names = {index.name for index in ProviderObjectSnapshotRow.__table__.indexes}

    assert {
        "datasource_id",
        "provider",
        "external_id",
        "external_url",
        "object_type",
        "title",
        "last_edited_time",
        "discovered_at",
        "selected_for_sync",
        "access_state",
        "raw_metadata_hash",
    }.issubset(column_names)
    assert "ix_provider_object_snapshots_datasource" in index_names
    assert "ix_provider_object_snapshots_access" in index_names
    assert "ix_provider_object_snapshots_ingestion" in index_names
    assert "ix_provider_object_snapshots_selected" in index_names
