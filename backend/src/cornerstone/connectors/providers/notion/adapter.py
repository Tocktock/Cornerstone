from __future__ import annotations

from typing import Any
from urllib.parse import urlencode

from cornerstone.config import Settings
from cornerstone.connectors.base import DiscoveryPage, OAuthMaterial
from cornerstone.connectors.providers.common import connector_error_from_api_response
from cornerstone.connectors.providers.notion.gateway import (
    MockNotionGateway,
    NotionAPIResponseError,
    NotionGateway,
    NotionHttpOAuthClient,
    NotionOAuthClient,
    NotionSdkGateway,
)
from cornerstone.connectors.providers.notion.mapper import (
    block_to_plain_text,
    notion_result_to_snapshot,
    page_to_source_object,
    stable_hash,
)
from cornerstone.schemas import (
    ConnectionIntent,
    ConnectionTestResult,
    ConnectionTestStatus,
    ConnectorCredential,
    ConnectorError,
    ConnectorErrorCode,
    ConnectorNextAction,
    DataSourceType,
    ProviderObjectAccessState,
    ProviderObjectSnapshot,
    ProviderObjectType,
    SourceObject,
    SourceSelection,
)


class NotionConnector:
    provider = DataSourceType.NOTION

    def __init__(
        self,
        settings: Settings,
        *,
        gateway: NotionGateway | None = None,
        oauth_client: NotionOAuthClient | None = None,
    ) -> None:
        self._settings = settings
        self._gateway = gateway or (
            MockNotionGateway() if settings.notion_mock_external_api else NotionSdkGateway(settings)
        )
        self._oauth_client = oauth_client or NotionHttpOAuthClient(settings)

    @property
    def gateway_name(self) -> str:
        return type(self._gateway).__name__

    def build_authorization_url(self, intent: ConnectionIntent) -> str:
        query = urlencode(
            {
                "client_id": self._settings.notion_client_id,
                "redirect_uri": intent.redirect_uri,
                "response_type": "code",
                "owner": "user",
                "state": intent.state_nonce,
            }
        )
        return f"{self._settings.notion_oauth_authorize_url}?{query}"

    async def complete_authorization(self, *, code: str, redirect_uri: str) -> OAuthMaterial:
        if self._settings.notion_mock_external_api:
            return OAuthMaterial(
                access_token=f"mock-notion-access-token-{code}",
                refresh_token=f"mock-notion-refresh-token-{code}",
                granted_scopes=["read_content"],
                external_account_id="mock-notion-user",
                external_workspace_id="mock-notion-workspace",
                external_workspace_name="Mock Notion Workspace",
                external_bot_id="mock-notion-bot",
            )
        try:
            return await self._oauth_client.exchange_code(code=code, redirect_uri=redirect_uri)
        except NotionAPIResponseError as exc:
            raise NotionProviderError(self.map_api_error(exc)) from exc

    async def test_connection(
        self,
        *,
        credential: ConnectorCredential,
        access_token: str,
    ) -> ConnectionTestResult:
        try:
            payload = await self._gateway.search(access_token=access_token, request_body={"page_size": 1})
        except NotionAPIResponseError as exc:
            error = self.map_api_error(exc)
            return ConnectionTestResult(
                status=ConnectionTestStatus.FAILED,
                datasource_id=credential.datasource_id,
                provider=DataSourceType.NOTION,
                external_workspace_name=credential.external_workspace_name,
                granted_scopes=credential.granted_scopes,
                can_read_objects=False,
                sample_object_count=0,
                limitations=[],
                error=error,
            )
        return ConnectionTestResult(
            status=ConnectionTestStatus.PASSED,
            datasource_id=credential.datasource_id,
            provider=DataSourceType.NOTION,
            external_workspace_name=credential.external_workspace_name,
            granted_scopes=credential.granted_scopes,
            can_read_objects=True,
            sample_object_count=len(payload.get("results", [])),
            limitations=(
                ["Notion connector is in mock external API mode; discovery returns fixture pages/databases."]
                if self._settings.notion_mock_external_api
                else []
            ),
        )

    async def discover_objects(
        self,
        *,
        credential: ConnectorCredential,
        access_token: str,
        page_size: int,
        cursor: str | None = None,
    ) -> DiscoveryPage:
        request_body: dict[str, Any] = {"page_size": page_size}
        if cursor is not None:
            request_body["start_cursor"] = cursor
        try:
            payload = await self._gateway.search(access_token=access_token, request_body=request_body)
        except NotionAPIResponseError as exc:
            raise NotionProviderError(self.map_api_error(exc)) from exc
        objects = [
            notion_result_to_snapshot(credential.datasource_id, result)
            for result in payload.get("results", [])
        ]
        return DiscoveryPage(
            objects=objects,
            next_cursor=payload.get("next_cursor"),
            has_more=bool(payload.get("has_more", False)),
        )

    def map_api_error(self, error: NotionAPIResponseError) -> ConnectorError:
        return connector_error_from_api_response(self.map_provider_error, error)

    def map_provider_error(
        self,
        *,
        status_code: int,
        provider_code: str | None = None,
        technical_message: str | None = None,
        retry_after_seconds: int | None = None,
    ) -> ConnectorError:
        """Map Notion API failures into product-actionable connector errors."""

        if status_code == 401 or provider_code in {"unauthorized", "invalid_grant"}:
            return ConnectorError(
                code=ConnectorErrorCode.TOKEN_REVOKED,
                user_message="The Notion connection is no longer authorized. Reconnect Notion.",
                technical_message=technical_message,
                retryable=False,
                next_action=ConnectorNextAction.RECONNECT,
            )
        if status_code == 403 or provider_code in {"restricted_resource", "insufficient_scope"}:
            return ConnectorError(
                code=ConnectorErrorCode.PERMISSION_DENIED,
                user_message=(
                    "Cornerstone cannot read the selected Notion content. Grant permission or "
                    "share the page/database with the integration."
                ),
                technical_message=technical_message,
                retryable=False,
                next_action=ConnectorNextAction.GRANT_PERMISSION,
            )
        if status_code == 404 or provider_code == "object_not_found":
            return ConnectorError(
                code=ConnectorErrorCode.OBJECT_NOT_FOUND,
                user_message="The selected Notion object was not found or is no longer shared with Cornerstone.",
                technical_message=technical_message,
                retryable=False,
                next_action=ConnectorNextAction.GRANT_PERMISSION,
            )
        if status_code == 429:
            suffix = (
                f" Retry after {retry_after_seconds} seconds."
                if retry_after_seconds is not None
                else " Retry later."
            )
            return ConnectorError(
                code=ConnectorErrorCode.RATE_LIMITED,
                user_message=f"Notion rate-limited this sync.{suffix}",
                technical_message=technical_message,
                retryable=True,
                next_action=ConnectorNextAction.WAIT_AND_RETRY,
                retry_after_seconds=retry_after_seconds,
            )
        if 500 <= status_code <= 599:
            return ConnectorError(
                code=ConnectorErrorCode.PROVIDER_UNAVAILABLE,
                user_message="Notion is temporarily unavailable. Retry the sync later.",
                technical_message=technical_message,
                retryable=True,
                next_action=ConnectorNextAction.RETRY,
            )
        return ConnectorError(
            code=ConnectorErrorCode.UNKNOWN,
            user_message="Notion returned an unexpected error. Review the connector logs.",
            technical_message=technical_message,
            retryable=False,
            next_action=ConnectorNextAction.CONTACT_ADMIN,
        )

    async def retrieve_page_snapshot(
        self,
        *,
        credential: ConnectorCredential,
        access_token: str,
        page_id: str,
    ) -> ProviderObjectSnapshot:
        """Retrieve one Notion page and map it to a selectable provider snapshot.

        Used by the live Notion E2E pilot when an operator supplies a specific
        test page ID. It keeps page lookup inside the Notion adapter and returns
        the same ProviderObjectSnapshot shape used by discovery/selection.
        """

        try:
            page = await self._gateway.retrieve_page(access_token=access_token, page_id=page_id)
        except NotionAPIResponseError as exc:
            raise NotionProviderError(self.map_api_error(exc)) from exc
        return notion_result_to_snapshot(credential.datasource_id, page)

    async def list_objects(
        self,
        *,
        credential: ConnectorCredential,
        access_token: str,
        selection: SourceSelection | None,
        selected_objects: list[ProviderObjectSnapshot] | None = None,
    ) -> list[SourceObject]:
        """Fetch selected Notion objects and normalize them for provider-agnostic ingestion.

        v0.8.2 supports selected Notion pages. Databases/data_sources stay discoverable
        but are rejected by source selection until their ingestion semantics are defined.
        """

        if selected_objects is None:
            selected_objects = _snapshots_from_selection(credential.datasource_id, selection)
        if not selected_objects:
            return []

        page_snapshots = [
            snapshot
            for snapshot in selected_objects
            if snapshot.object_type == ProviderObjectType.PAGE and snapshot.ingestion_supported
        ]
        objects: list[SourceObject] = []
        for snapshot in page_snapshots:
            objects.append(await self._fetch_page_as_source_object(access_token=access_token, snapshot=snapshot))
        return objects

    async def _fetch_page_as_source_object(
        self,
        *,
        access_token: str,
        snapshot: ProviderObjectSnapshot,
    ) -> SourceObject:
        try:
            page = await self._gateway.retrieve_page(access_token=access_token, page_id=snapshot.external_id)
            markdown = await self._gateway.retrieve_page_markdown(
                access_token=access_token,
                page_id=snapshot.external_id,
            )
            if markdown is not None:
                return page_to_source_object(
                    page=page,
                    snapshot=snapshot,
                    content=markdown,
                    content_format="markdown",
                )
            block_text = await self._fetch_block_text(
                access_token=access_token,
                block_id=snapshot.external_id,
                depth=0,
            )
        except NotionAPIResponseError as exc:
            raise NotionProviderError(self.map_api_error(exc)) from exc
        return page_to_source_object(
            page=page,
            snapshot=snapshot,
            content=block_text,
            content_format="notion_blocks_text",
        )

    async def _fetch_block_text(
        self,
        *,
        access_token: str,
        block_id: str,
        depth: int,
    ) -> str:
        if depth > self._settings.notion_max_block_depth:
            return ""
        cursor: str | None = None
        lines: list[str] = []
        while True:
            payload = await self._gateway.retrieve_block_children(
                access_token=access_token,
                block_id=block_id,
                page_size=self._settings.notion_block_page_size,
                start_cursor=cursor,
            )
            for block in payload.get("results", []):
                text = block_to_plain_text(block)
                if text:
                    lines.append(text)
                if bool(block.get("has_children", False)):
                    nested = await self._fetch_block_text(
                        access_token=access_token,
                        block_id=str(block.get("id") or ""),
                        depth=depth + 1,
                    )
                    if nested:
                        lines.append(nested)
            if not payload.get("has_more"):
                break
            cursor = payload.get("next_cursor")
            if cursor is None:
                break
        return "\n".join(line for line in lines if line).strip()


class NotionProviderError(RuntimeError):
    def __init__(self, connector_error: ConnectorError) -> None:
        super().__init__(connector_error.user_message)
        self.connector_error = connector_error


def _snapshots_from_selection(
    datasource_id: str,
    selection: SourceSelection | None,
) -> list[ProviderObjectSnapshot]:
    if selection is None:
        return []
    return [
        ProviderObjectSnapshot(
            datasource_id=datasource_id,
            provider=DataSourceType.NOTION,
            external_id=external_id,
            object_type=ProviderObjectType.PAGE,
            title=external_id,
            access_state=ProviderObjectAccessState.ACCESSIBLE,
            raw_metadata_hash=stable_hash({"external_id": external_id}),
            selected_for_sync=True,
            ingestion_supported=True,
            ingestion_unsupported_reason=None,
            provider_metadata={"selectionOnly": True},
        )
        for external_id in selection.selected_external_object_ids
    ]
