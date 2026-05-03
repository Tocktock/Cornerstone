from __future__ import annotations

from cornerstone.config import Settings
from cornerstone.connectors.base import DiscoveryPage, OAuthMaterial
from cornerstone.schemas import (
    ConnectionIntent,
    ConnectionTestResult,
    ConnectionTestStatus,
    ConnectorCredential,
    DataSourceType,
    ProviderObjectAccessState,
    ProviderObjectSnapshot,
    ProviderObjectType,
    SourceObject,
    SourceSelection,
    utc_now,
)


class ManualConnector:
    """Small non-Notion connector used as an architectural smoke test.

    Manual sources are ingested through /manual-sources/{id}/sync so provider connectors cannot bypass discovery and selection.
    This connector proves that discovery -> selection -> normalized SourceObject ingestion is not
    Notion-shaped.
    """

    provider = DataSourceType.MANUAL

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def build_authorization_url(self, intent: ConnectionIntent) -> str:
        return f"manual://connect/{intent.id}"

    async def complete_authorization(self, *, code: str, redirect_uri: str) -> OAuthMaterial:
        _ = (code, redirect_uri)
        return OAuthMaterial(access_token="manual-no-token", granted_scopes=[])

    async def test_connection(
        self,
        *,
        credential: ConnectorCredential,
        access_token: str,
    ) -> ConnectionTestResult:
        _ = access_token
        return ConnectionTestResult(
            status=ConnectionTestStatus.PASSED,
            datasource_id=credential.datasource_id,
            provider=DataSourceType.MANUAL,
            external_workspace_name=credential.external_workspace_name,
            can_read_objects=True,
            sample_object_count=1,
            limitations=["Manual connector smoke-test mode."],
        )

    async def discover_objects(
        self,
        *,
        credential: ConnectorCredential,
        access_token: str,
        page_size: int,
        cursor: str | None = None,
    ) -> DiscoveryPage:
        _ = (access_token, page_size, cursor)
        return DiscoveryPage(
            objects=[
                ProviderObjectSnapshot(
                    datasource_id=credential.datasource_id,
                    provider=DataSourceType.MANUAL,
                    external_id="manual-note-1",
                    external_url="manual://note/manual-note-1",
                    object_type=ProviderObjectType.UNKNOWN,
                    title="Manual Pilot Note",
                    discovered_at=utc_now(),
                    selected_for_sync=True,
                    access_state=ProviderObjectAccessState.ACCESSIBLE,
                    raw_metadata_hash="manual-note-1-hash-value",
                    provider_metadata={"connector": "manual", "sourceKind": "manual_note"},
                )
            ],
            next_cursor=None,
            has_more=False,
        )

    async def list_objects(
        self,
        *,
        credential: ConnectorCredential,
        access_token: str,
        selection: SourceSelection | None,
        selected_objects: list[ProviderObjectSnapshot] | None = None,
    ) -> list[SourceObject]:
        _ = (credential, access_token)
        external_ids = [snapshot.external_id for snapshot in selected_objects or []]
        if not external_ids and selection is not None:
            external_ids = list(selection.selected_external_object_ids)
        return [
            SourceObject(
                source_external_id=external_id,
                title="Manual Pilot Note",
                content=(
                    "Manual connector smoke test. Cornerstone is a shared organizational context layer. "
                    "Official context must preserve provenance."
                ),
                source_url=f"manual://note/{external_id}",
                source_object_type="manual_note",
                provider_metadata={"provider": "manual", "sourceKind": "manual_note"},
            )
            for external_id in external_ids
        ]
