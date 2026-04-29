from __future__ import annotations

import asyncio
from typing import Any, ClassVar

import pytest

from cornerstone.config import Settings
from cornerstone.connectors.base import OAuthMaterial
from cornerstone.connectors.notion import NotionConnector, NotionProviderError
from cornerstone.connectors.providers.notion.gateway import (
    NotionAPIResponseError,
    NotionGateway,
    NotionSdkGateway,
)
from cornerstone.connectors.providers.notion.mapper import block_to_plain_text, rich_text_plain_text
from cornerstone.schemas import (
    ConnectionTestStatus,
    ConnectorAuthType,
    ConnectorCredential,
    ConnectorErrorCode,
    CredentialStatus,
    DataSourceType,
    ProviderObjectAccessState,
    ProviderObjectSnapshot,
    ProviderObjectType,
)

pytestmark = pytest.mark.unit


class FakeOAuthClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    async def exchange_code(self, *, code: str, redirect_uri: str) -> OAuthMaterial:
        self.calls.append({"code": code, "redirect_uri": redirect_uri})
        return OAuthMaterial(
            access_token="live-access-token",
            refresh_token="live-refresh-token",
            granted_scopes=["read_content"],
            external_account_id="user-1",
            external_workspace_id="workspace-1",
            external_workspace_name="Acme Workspace",
            external_bot_id="bot-1",
        )


class FakeNotionGateway:
    def __init__(self) -> None:
        self.search_responses: list[dict[str, Any] | NotionAPIResponseError] = []
        self.pages: dict[str, dict[str, Any] | NotionAPIResponseError] = {}
        self.markdown: dict[str, str | None | NotionAPIResponseError] = {}
        self.block_children: dict[str, list[dict[str, Any] | NotionAPIResponseError]] = {}
        self.search_calls: ClassVar[list[dict[str, Any]]] = []
        self.page_calls: list[str] = []
        self.markdown_calls: list[str] = []
        self.block_child_calls: ClassVar[list[dict[str, Any]]] = []

    async def search(self, *, access_token: str, request_body: dict[str, Any]) -> dict[str, Any]:
        self.search_calls.append({"access_token": access_token, "request_body": request_body})
        if not self.search_responses:
            raise AssertionError("No fake search response queued.")
        response = self.search_responses.pop(0)
        if isinstance(response, NotionAPIResponseError):
            raise response
        return response

    async def retrieve_page(self, *, access_token: str, page_id: str) -> dict[str, Any]:
        _ = access_token
        self.page_calls.append(page_id)
        response = self.pages[page_id]
        if isinstance(response, NotionAPIResponseError):
            raise response
        return response

    async def retrieve_page_markdown(self, *, access_token: str, page_id: str) -> str | None:
        _ = access_token
        self.markdown_calls.append(page_id)
        response = self.markdown.get(page_id)
        if isinstance(response, NotionAPIResponseError):
            raise response
        return response

    async def retrieve_block_children(
        self,
        *,
        access_token: str,
        block_id: str,
        page_size: int,
        start_cursor: str | None = None,
    ) -> dict[str, Any]:
        _ = access_token
        self.block_child_calls.append(
            {"block_id": block_id, "page_size": page_size, "start_cursor": start_cursor}
        )
        responses = self.block_children.get(block_id)
        if not responses:
            return {"has_more": False, "results": []}
        response = responses.pop(0)
        if isinstance(response, NotionAPIResponseError):
            raise response
        return response


def _settings() -> Settings:
    return Settings(notion_mock_external_api=False, notion_client_id="client-id", notion_client_secret="client-secret")


def _credential() -> ConnectorCredential:
    return ConnectorCredential(
        provider=DataSourceType.NOTION,
        datasource_id="source-1",
        auth_type=ConnectorAuthType.OAUTH2,
        encrypted_access_token="encrypted",
        granted_scopes=["read_content"],
        status=CredentialStatus.ACTIVE,
        external_workspace_name="Workspace",
    )


def _page_snapshot() -> ProviderObjectSnapshot:
    return ProviderObjectSnapshot(
        datasource_id="source-1",
        provider=DataSourceType.NOTION,
        external_id="page-1",
        object_type=ProviderObjectType.PAGE,
        title="Product Requirements",
        access_state=ProviderObjectAccessState.ACCESSIBLE,
        raw_metadata_hash="page-1-metadata-hash",
    )


def test_notion_live_default_gateway_is_sdk_backed() -> None:
    connector = NotionConnector(_settings())

    assert connector.gateway_name == "NotionSdkGateway"


def test_notion_sdk_gateway_is_runtime_checkable_protocol() -> None:
    gateway: NotionGateway = NotionSdkGateway(_settings())

    assert isinstance(gateway, NotionGateway)


def test_notion_live_oauth_token_exchange_uses_oauth_boundary() -> None:
    oauth_client = FakeOAuthClient()
    connector = NotionConnector(_settings(), oauth_client=oauth_client)

    material = asyncio.run(connector.complete_authorization(code="oauth-code", redirect_uri="http://callback"))

    assert material.access_token == "live-access-token"
    assert material.refresh_token == "live-refresh-token"
    assert material.external_workspace_id == "workspace-1"
    assert material.external_workspace_name == "Acme Workspace"
    assert oauth_client.calls == [{"code": "oauth-code", "redirect_uri": "http://callback"}]


def test_notion_live_oauth_failure_maps_provider_error() -> None:
    class FailingOAuthClient:
        async def exchange_code(self, *, code: str, redirect_uri: str) -> OAuthMaterial:
            _ = (code, redirect_uri)
            raise NotionAPIResponseError(
                status_code=401,
                provider_code="invalid_grant",
                message="Code expired.",
            )

    connector = NotionConnector(_settings(), oauth_client=FailingOAuthClient())

    with pytest.raises(NotionProviderError) as exc_info:
        asyncio.run(connector.complete_authorization(code="bad", redirect_uri="http://callback"))

    assert exc_info.value.connector_error.code == ConnectorErrorCode.TOKEN_REVOKED
    assert exc_info.value.connector_error.next_action == "reconnect"


def test_notion_live_connection_test_success_uses_gateway_search() -> None:
    gateway = FakeNotionGateway()
    gateway.search_responses.append({"results": [{"id": "page-1"}]})
    connector = NotionConnector(_settings(), gateway=gateway)

    result = asyncio.run(connector.test_connection(credential=_credential(), access_token="access-token"))

    assert result.status == ConnectionTestStatus.PASSED
    assert result.can_read_objects is True
    assert result.sample_object_count == 1
    assert gateway.search_calls == [
        {"access_token": "access-token", "request_body": {"page_size": 1}}
    ]


def test_notion_live_connection_test_failure_maps_provider_error() -> None:
    gateway = FakeNotionGateway()
    gateway.search_responses.append(
        NotionAPIResponseError(
            status_code=403,
            provider_code="restricted_resource",
            message="Page not shared with integration.",
        )
    )
    connector = NotionConnector(_settings(), gateway=gateway)

    result = asyncio.run(connector.test_connection(credential=_credential(), access_token="access-token"))

    assert result.status == ConnectionTestStatus.FAILED
    assert result.error is not None
    assert result.error.code == ConnectorErrorCode.PERMISSION_DENIED
    assert result.error.next_action == "grant_permission"


def test_notion_live_discovery_maps_pages_databases_and_data_sources() -> None:
    gateway = FakeNotionGateway()
    gateway.search_responses.append(
        {
            "has_more": True,
            "next_cursor": "cursor-2",
            "results": [
                {
                    "object": "page",
                    "id": "page-1",
                    "url": "https://notion.so/page-1",
                    "last_edited_time": "2026-04-20T12:00:00.000Z",
                    "parent": {"workspace": True},
                    "properties": {
                        "Name": {
                            "type": "title",
                            "title": [{"plain_text": "Product Requirements"}],
                        }
                    },
                },
                {
                    "object": "database",
                    "id": "db-1",
                    "url": "https://notion.so/db-1",
                    "parent": {"page_id": "page-1"},
                    "title": [{"plain_text": "Decision Log"}],
                },
                {
                    "object": "data_source",
                    "id": "ds-1",
                    "url": "https://notion.so/ds-1",
                    "in_trash": True,
                    "parent": {"database_id": "db-1"},
                    "title": [{"plain_text": "Data Source"}],
                },
            ],
        }
    )
    connector = NotionConnector(_settings(), gateway=gateway)

    page = asyncio.run(
        connector.discover_objects(
            credential=_credential(),
            access_token="access-token",
            page_size=50,
            cursor="cursor-1",
        )
    )

    assert page.has_more is True
    assert page.next_cursor == "cursor-2"
    assert [item.external_id for item in page.objects] == ["page-1", "db-1", "ds-1"]
    assert [item.object_type for item in page.objects] == [
        ProviderObjectType.PAGE,
        ProviderObjectType.DATABASE,
        ProviderObjectType.DATA_SOURCE,
    ]
    assert page.objects[0].title == "Product Requirements"
    assert page.objects[0].ingestion_supported is True
    assert page.objects[1].parent_external_id == "page-1"
    assert page.objects[1].ingestion_supported is False
    assert "database ingestion" in (page.objects[1].ingestion_unsupported_reason or "")
    assert page.objects[2].access_state == ProviderObjectAccessState.DELETED
    assert page.objects[2].ingestion_supported is False
    assert gateway.search_calls[0]["request_body"] == {"page_size": 50, "start_cursor": "cursor-1"}


def test_notion_live_discovery_raises_actionable_error_on_rate_limit() -> None:
    gateway = FakeNotionGateway()
    gateway.search_responses.append(
        NotionAPIResponseError(
            status_code=429,
            provider_code="rate_limited",
            message="Slow down.",
            retry_after_seconds=7,
        )
    )
    connector = NotionConnector(_settings(), gateway=gateway)

    with pytest.raises(NotionProviderError) as error_info:
        asyncio.run(
            connector.discover_objects(
                credential=_credential(),
                access_token="access-token",
                page_size=25,
            )
        )

    assert error_info.value.connector_error.code == ConnectorErrorCode.RATE_LIMITED
    assert error_info.value.connector_error.retryable is True
    assert "7 seconds" in error_info.value.connector_error.user_message


def test_notion_live_list_objects_fetches_selected_page_markdown() -> None:
    gateway = FakeNotionGateway()
    gateway.pages["page-1"] = {
        "id": "page-1",
        "url": "https://notion.so/page-1",
        "last_edited_time": "2026-04-20T12:00:00.000Z",
        "properties": {
            "Name": {"type": "title", "title": [{"plain_text": "Product Requirements"}]}
        },
    }
    gateway.markdown["page-1"] = "# Product Requirements\nCornerstone is a shared organizational context layer."
    connector = NotionConnector(_settings(), gateway=gateway)

    objects = asyncio.run(
        connector.list_objects(
            credential=_credential(),
            access_token="access-token",
            selection=None,
            selected_objects=[_page_snapshot()],
        )
    )

    assert len(objects) == 1
    assert objects[0].source_external_id == "page-1"
    assert objects[0].source_object_type == "page"
    assert objects[0].content.startswith("# Product Requirements")
    assert objects[0].provider_metadata["provider"] == "notion"
    assert gateway.page_calls == ["page-1"]
    assert gateway.markdown_calls == ["page-1"]


def test_notion_live_list_objects_falls_back_to_block_children() -> None:
    gateway = FakeNotionGateway()
    gateway.pages["page-1"] = {
        "id": "page-1",
        "url": "https://notion.so/page-1",
        "last_edited_time": "2026-04-20T12:00:00.000Z",
    }
    gateway.markdown["page-1"] = None
    gateway.block_children["page-1"] = [
        {
            "has_more": False,
            "results": [
                {
                    "id": "block-1",
                    "type": "paragraph",
                    "has_children": False,
                    "paragraph": {"rich_text": [{"plain_text": "Official context requires evidence."}]},
                }
            ],
        }
    ]
    connector = NotionConnector(_settings(), gateway=gateway)

    objects = asyncio.run(
        connector.list_objects(
            credential=_credential(),
            access_token="access-token",
            selection=None,
            selected_objects=[_page_snapshot()],
        )
    )

    assert objects[0].content == "Official context requires evidence."
    assert objects[0].provider_metadata["contentFormat"] == "notion_blocks_text"


def test_notion_map_api_error_handles_non_json_payload_equivalent() -> None:
    connector = NotionConnector(_settings(), gateway=FakeNotionGateway())

    error = connector.map_api_error(NotionAPIResponseError(status_code=500, message="plain failure"))

    assert error.code == ConnectorErrorCode.PROVIDER_UNAVAILABLE
    assert error.retryable is True
    assert error.technical_message == "plain failure"


def test_notion_map_provider_error_handles_missing_object_and_rate_limit_without_retry_after() -> None:
    connector = NotionConnector(_settings(), gateway=FakeNotionGateway())

    missing = connector.map_provider_error(status_code=404, technical_message="missing")
    limited = connector.map_provider_error(status_code=429, technical_message="slow")

    assert missing.code == ConnectorErrorCode.OBJECT_NOT_FOUND
    assert missing.next_action == "grant_permission"
    assert limited.code == ConnectorErrorCode.RATE_LIMITED
    assert "Retry later" in limited.user_message


def test_notion_list_objects_without_selection_returns_empty_list() -> None:
    connector = NotionConnector(_settings(), gateway=FakeNotionGateway())

    objects = asyncio.run(
        connector.list_objects(
            credential=_credential(),
            access_token="access-token",
            selection=None,
            selected_objects=None,
        )
    )

    assert objects == []


def test_notion_list_objects_ignores_non_page_snapshots_as_a_defense_in_depth() -> None:
    connector = NotionConnector(Settings(notion_mock_external_api=True))
    snapshot = ProviderObjectSnapshot(
        datasource_id="source-1",
        provider=DataSourceType.NOTION,
        external_id="database-1",
        object_type=ProviderObjectType.DATABASE,
        title="Decision Database",
        access_state=ProviderObjectAccessState.ACCESSIBLE,
        raw_metadata_hash="database-metadata-hash",
        ingestion_supported=False,
        ingestion_unsupported_reason="Notion database ingestion is not implemented.",
    )

    objects = asyncio.run(
        connector.list_objects(
            credential=_credential(),
            access_token="access-token",
            selection=None,
            selected_objects=[snapshot],
        )
    )

    assert objects == []


def test_notion_block_text_respects_max_depth_and_cursor_without_next_cursor() -> None:
    gateway = FakeNotionGateway()
    gateway.block_children["block-root"] = [
        {
            "has_more": True,
            "next_cursor": None,
            "results": [
                {
                    "id": "block-1",
                    "type": "bulleted_list_item",
                    "has_children": True,
                    "bulleted_list_item": {"rich_text": [{"plain_text": "Evidence"}]},
                }
            ],
        }
    ]
    gateway.block_children["block-1"] = [{"has_more": False, "results": []}]
    connector = NotionConnector(_settings(), gateway=gateway)

    async def run() -> tuple[str, str]:
        over_depth = await connector._fetch_block_text(
            access_token="access-token",
            block_id="block-root",
            depth=999,
        )
        text = await connector._fetch_block_text(
            access_token="access-token",
            block_id="block-root",
            depth=0,
        )
        return over_depth, text

    assert asyncio.run(run()) == ("", "- Evidence")


def test_block_mapping_and_rich_text_helpers_are_provider_local() -> None:
    assert rich_text_plain_text([{"plain_text": "A"}, {"plain_text": "B"}]) == "AB"
    assert block_to_plain_text(
        {"type": "to_do", "to_do": {"checked": True, "rich_text": [{"plain_text": "Reviewed"}]}}
    ) == "[x] Reviewed"


def test_notion_http_oauth_client_exchange_maps_workspace_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    from cornerstone.connectors.providers.notion import gateway as gateway_module
    from cornerstone.connectors.providers.notion.gateway import NotionHttpOAuthClient

    class FakeAsyncClient:
        calls: ClassVar[list[dict[str, Any]]] = []

        def __init__(self, *, timeout: float) -> None:
            self.timeout = timeout

        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        async def post(self, url: str, **kwargs: Any) -> httpx.Response:
            self.calls.append({"url": url, **kwargs})
            response = httpx.Response(
                200,
                json={
                    "access_token": "sdk-access-token",
                    "refresh_token": "sdk-refresh-token",
                    "workspace_id": "workspace-1",
                    "workspace_name": "Acme Workspace",
                    "bot_id": "bot-1",
                    "owner": {"user": {"id": "user-1"}},
                },
            )
            response.request = httpx.Request("POST", url)
            return response

    monkeypatch.setattr(gateway_module.httpx, "AsyncClient", FakeAsyncClient)
    client = NotionHttpOAuthClient(_settings())

    material = asyncio.run(client.exchange_code(code="oauth-code", redirect_uri="http://callback"))

    assert material.access_token == "sdk-access-token"
    assert material.refresh_token == "sdk-refresh-token"
    assert material.external_account_id == "user-1"
    assert FakeAsyncClient.calls[0]["json"]["grant_type"] == "authorization_code"
    assert FakeAsyncClient.calls[0]["headers"]["Authorization"].startswith("Basic ")


def test_notion_sdk_gateway_uses_notion_client_for_standard_api_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from types import SimpleNamespace

    calls: ClassVar[list[dict[str, Any]]] = []

    class FakePages:
        async def retrieve(self, **kwargs: Any) -> dict[str, Any]:
            calls.append({"method": "pages.retrieve", "kwargs": kwargs})
            return {"id": kwargs["page_id"]}

    class FakeChildren:
        async def list(self, **kwargs: Any) -> dict[str, Any]:
            calls.append({"method": "blocks.children.list", "kwargs": kwargs})
            return {"has_more": False, "results": []}

    class FakeBlocks:
        children = FakeChildren()

    class FakeAsyncClient:
        def __init__(self, *, auth: str, notion_version: str) -> None:
            calls.append({"method": "init", "auth": auth, "notion_version": notion_version})
            self.pages = FakePages()
            self.blocks = FakeBlocks()

        async def search(self, **kwargs: Any) -> dict[str, Any]:
            calls.append({"method": "search", "kwargs": kwargs})
            return {"results": [{"id": "page-1"}]}

    monkeypatch.setitem(sys.modules, "notion_client", SimpleNamespace(AsyncClient=FakeAsyncClient))
    gateway = NotionSdkGateway(_settings())

    search = asyncio.run(gateway.search(access_token="token", request_body={"page_size": 1}))
    page = asyncio.run(gateway.retrieve_page(access_token="token", page_id="page-1"))
    children = asyncio.run(
        gateway.retrieve_block_children(
            access_token="token",
            block_id="page-1",
            page_size=25,
            start_cursor="cursor-1",
        )
    )

    assert search["results"][0]["id"] == "page-1"
    assert page == {"id": "page-1"}
    assert children == {"has_more": False, "results": []}
    assert {call["method"] for call in calls} == {
        "init",
        "search",
        "pages.retrieve",
        "blocks.children.list",
    }
    assert calls[0]["notion_version"] == "2026-03-11"


def test_notion_sdk_gateway_markdown_fallback_maps_http_responses(monkeypatch: pytest.MonkeyPatch) -> None:
    import httpx

    from cornerstone.connectors.providers.notion import gateway as gateway_module

    class FakeAsyncClient:
        responses: ClassVar[list[httpx.Response]] = []
        calls: ClassVar[list[dict[str, Any]]] = []

        def __init__(self, *, timeout: float) -> None:
            self.timeout = timeout

        async def __aenter__(self) -> FakeAsyncClient:
            return self

        async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
            return None

        async def get(self, url: str, **kwargs: Any) -> httpx.Response:
            self.calls.append({"url": url, **kwargs})
            response = self.responses.pop(0)
            response.request = httpx.Request("GET", url)
            return response

    FakeAsyncClient.responses = [
        httpx.Response(200, json={"markdown": "# Page"}),
        httpx.Response(404, json={}),
        httpx.Response(429, headers={"Retry-After": "9"}, json={"code": "rate_limited", "message": "Slow"}),
    ]
    monkeypatch.setattr(gateway_module.httpx, "AsyncClient", FakeAsyncClient)
    gateway = NotionSdkGateway(_settings())

    markdown = asyncio.run(gateway.retrieve_page_markdown(access_token="token", page_id="page-1"))
    missing = asyncio.run(gateway.retrieve_page_markdown(access_token="token", page_id="page-1"))

    assert markdown == "# Page"
    assert missing is None
    with pytest.raises(NotionAPIResponseError) as exc_info:
        asyncio.run(gateway.retrieve_page_markdown(access_token="token", page_id="page-1"))
    assert exc_info.value.status_code == 429
    assert exc_info.value.retry_after_seconds == 9
    assert FakeAsyncClient.calls[0]["headers"]["Notion-Version"] == "2026-03-11"


def test_notion_sdk_gateway_converts_sdk_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from types import SimpleNamespace

    class FakeSdkError(Exception):
        status = 403
        code = "restricted_resource"
        message = "No access"
        headers: ClassVar[dict[str, str]] = {"Retry-After": "4"}

    class FakeAsyncClient:
        def __init__(self, *, auth: str, notion_version: str) -> None:
            _ = (auth, notion_version)

        async def search(self, **kwargs: Any) -> dict[str, Any]:
            _ = kwargs
            raise FakeSdkError()

    monkeypatch.setitem(sys.modules, "notion_client", SimpleNamespace(AsyncClient=FakeAsyncClient))
    gateway = NotionSdkGateway(_settings())

    with pytest.raises(NotionAPIResponseError) as exc_info:
        asyncio.run(gateway.search(access_token="token", request_body={"page_size": 1}))

    assert exc_info.value.status_code == 403
    assert exc_info.value.provider_code == "restricted_resource"
    assert exc_info.value.retry_after_seconds == 4


def test_notion_connector_retrieves_specific_page_snapshot_for_live_e2e() -> None:
    gateway = FakeNotionGateway()
    gateway.pages["page-1"] = {
        "object": "page",
        "id": "page-1",
        "url": "https://notion.so/page-1",
        "last_edited_time": "2026-04-20T12:00:00.000Z",
        "properties": {
            "Name": {"type": "title", "title": [{"plain_text": "Live Pilot Page"}]}
        },
    }
    connector = NotionConnector(_settings(), gateway=gateway)

    snapshot = asyncio.run(
        connector.retrieve_page_snapshot(
            credential=_credential(),
            access_token="access-token",
            page_id="page-1",
        )
    )

    assert snapshot.external_id == "page-1"
    assert snapshot.object_type == ProviderObjectType.PAGE
    assert snapshot.title == "Live Pilot Page"
    assert snapshot.ingestion_supported is True
    assert gateway.page_calls == ["page-1"]
