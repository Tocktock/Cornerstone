from __future__ import annotations

import base64
from datetime import UTC, datetime
from typing import Any, Protocol, runtime_checkable

import httpx

from cornerstone.config import Settings
from cornerstone.connectors.base import OAuthMaterial


class NotionAPIResponseError(RuntimeError):
    """Provider-level API failure normalized before Cornerstone error mapping."""

    def __init__(
        self,
        *,
        status_code: int,
        provider_code: str | None = None,
        message: str | None = None,
        retry_after_seconds: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message or provider_code or f"Notion API returned HTTP {status_code}.")
        self.status_code = status_code
        self.provider_code = provider_code
        self.message = message
        self.retry_after_seconds = retry_after_seconds
        self.payload = payload or {}


@runtime_checkable
class NotionOAuthClient(Protocol):
    """OAuth exchange boundary.

    notion-client focuses on authenticated API access; OAuth token exchange stays isolated here.
    """

    async def exchange_code(self, *, code: str, redirect_uri: str) -> OAuthMaterial:
        ...


@runtime_checkable
class NotionGateway(Protocol):
    """Gateway used by the Notion adapter for provider API access.

    The adapter owns Cornerstone semantics; the gateway owns provider API mechanics.
    """

    async def search(self, *, access_token: str, request_body: dict[str, Any]) -> dict[str, Any]:
        ...

    async def retrieve_page(self, *, access_token: str, page_id: str) -> dict[str, Any]:
        ...

    async def retrieve_page_markdown(self, *, access_token: str, page_id: str) -> str | None:
        ...

    async def retrieve_block_children(
        self,
        *,
        access_token: str,
        block_id: str,
        page_size: int,
        start_cursor: str | None = None,
    ) -> dict[str, Any]:
        ...


class NotionHttpOAuthClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def exchange_code(self, *, code: str, redirect_uri: str) -> OAuthMaterial:
        credentials = f"{self._settings.notion_client_id}:{self._settings.notion_client_secret}"
        basic_token = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")
        async with httpx.AsyncClient(timeout=self._settings.connector_http_timeout_seconds) as client:
            response = await client.post(
                self._settings.notion_oauth_token_url,
                headers={
                    "Authorization": f"Basic {basic_token}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                },
                json={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                },
            )
            if response.status_code >= 400:
                raise _response_to_error(response)
            payload = response.json()
        owner = payload.get("owner") or {}
        user = owner.get("user") or {}
        return OAuthMaterial(
            access_token=str(payload["access_token"]),
            refresh_token=payload.get("refresh_token"),
            granted_scopes=["read_content"],
            external_account_id=user.get("id"),
            external_workspace_id=payload.get("workspace_id"),
            external_workspace_name=payload.get("workspace_name"),
            external_bot_id=payload.get("bot_id"),
        )


class MockNotionGateway:
    """Fixture gateway for local/dev tests with no external Notion dependency."""

    async def search(self, *, access_token: str, request_body: dict[str, Any]) -> dict[str, Any]:
        _ = (access_token, request_body)
        return {
            "has_more": False,
            "next_cursor": None,
            "results": [
                {
                    "object": "page",
                    "id": "notion-page-1",
                    "url": "https://www.notion.so/mock/notion-page-1",
                    "last_edited_time": _mock_last_edited_time(),
                    "parent": {"workspace": True},
                    "properties": {
                        "Name": {
                            "type": "title",
                            "title": [{"plain_text": "Cornerstone Product Requirements"}],
                        }
                    },
                    "provider_metadata": {"mock": True},
                },
                {
                    "object": "database",
                    "id": "notion-database-1",
                    "url": "https://www.notion.so/mock/notion-database-1",
                    "parent": {"workspace": True},
                    "title": [{"plain_text": "Team Decisions"}],
                    "provider_metadata": {"mock": True},
                },
            ],
        }

    async def retrieve_page(self, *, access_token: str, page_id: str) -> dict[str, Any]:
        _ = access_token
        return {
            "object": "page",
            "id": page_id,
            "url": f"https://www.notion.so/mock/{page_id}",
            "last_edited_time": _mock_last_edited_time(),
            "properties": {
                "Name": {"type": "title", "title": [{"plain_text": "Cornerstone Product Requirements"}]}
            },
        }

    async def retrieve_page_markdown(self, *, access_token: str, page_id: str) -> str | None:
        _ = access_token
        return (
            f"# {page_id}\n"
            "Cornerstone is a shared organizational context layer. "
            "Official context must preserve provenance and freshness. "
            "A Concept cannot become official without reviewed evidence or a DecisionRecord."
        )

    async def retrieve_block_children(
        self,
        *,
        access_token: str,
        block_id: str,
        page_size: int,
        start_cursor: str | None = None,
    ) -> dict[str, Any]:
        _ = (access_token, block_id, page_size, start_cursor)
        return {"has_more": False, "next_cursor": None, "results": []}


def _mock_last_edited_time() -> str:
    """Return a current Notion-style timestamp for deterministic freshness tests.

    The previous fixed fixture timestamp made local connector tests time-dependent once
    the fixture aged beyond the freshness threshold.
    """

    now = datetime.now(UTC).replace(microsecond=0)
    return now.isoformat().replace("+00:00", ".000Z")


class NotionSdkGateway:
    """notion-client-backed gateway for authenticated Notion API access.

    The official Python client handles standard Notion API calls. A small raw HTTP fallback is kept
    only for endpoints that may not be exposed by the SDK yet, such as the page markdown endpoint.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def search(self, *, access_token: str, request_body: dict[str, Any]) -> dict[str, Any]:
        client = self._client(access_token)
        try:
            payload = await client.search(**request_body)
        except Exception as exc:  # pragma: no cover - exercised through adapter contract tests.
            raise _exception_to_error(exc) from exc
        return _ensure_dict(payload)

    async def retrieve_page(self, *, access_token: str, page_id: str) -> dict[str, Any]:
        client = self._client(access_token)
        try:
            payload = await client.pages.retrieve(page_id=page_id)
        except Exception as exc:  # pragma: no cover - exercised through adapter contract tests.
            raise _exception_to_error(exc) from exc
        return _ensure_dict(payload)

    async def retrieve_page_markdown(self, *, access_token: str, page_id: str) -> str | None:
        if not self._settings.notion_use_markdown_endpoint:
            return None
        async with httpx.AsyncClient(timeout=self._settings.connector_http_timeout_seconds) as client:
            response = await client.get(
                f"{self._settings.notion_api_base_url}/pages/{page_id}/markdown",
                headers=self._headers(access_token),
            )
            if response.status_code == 404:
                return None
            if response.status_code >= 400:
                raise _response_to_error(response)
            payload = response.json()
        markdown = payload.get("markdown")
        return markdown if isinstance(markdown, str) else None

    async def retrieve_block_children(
        self,
        *,
        access_token: str,
        block_id: str,
        page_size: int,
        start_cursor: str | None = None,
    ) -> dict[str, Any]:
        client = self._client(access_token)
        kwargs: dict[str, Any] = {"block_id": block_id, "page_size": page_size}
        if start_cursor is not None:
            kwargs["start_cursor"] = start_cursor
        try:
            payload = await client.blocks.children.list(**kwargs)
        except Exception as exc:  # pragma: no cover - exercised through adapter contract tests.
            raise _exception_to_error(exc) from exc
        return _ensure_dict(payload)

    def _client(self, access_token: str) -> Any:
        try:
            from notion_client import AsyncClient
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency installed in packaged env.
            raise RuntimeError(
                "notion-client is required for live Notion API access. "
                "Install project dependencies with `pip install -e '.[dev]'`."
            ) from exc
        return AsyncClient(auth=access_token, notion_version=self._settings.notion_version)

    def _headers(self, access_token: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {access_token}",
            "Notion-Version": self._settings.notion_version,
            "Content-Type": "application/json",
        }


def _ensure_dict(payload: Any) -> dict[str, Any]:
    return payload if isinstance(payload, dict) else {}


def _response_to_error(response: httpx.Response) -> NotionAPIResponseError:
    retry_after_header = response.headers.get("Retry-After")
    retry_after_seconds = int(retry_after_header) if retry_after_header and retry_after_header.isdigit() else None
    try:
        payload = response.json()
        provider_code = payload.get("code")
        message = payload.get("message") or response.text
    except ValueError:
        payload = None
        provider_code = None
        message = response.text
    return NotionAPIResponseError(
        status_code=response.status_code,
        provider_code=provider_code,
        message=message,
        retry_after_seconds=retry_after_seconds,
        payload=payload,
    )


def _exception_to_error(exc: Exception) -> NotionAPIResponseError:
    status_code = int(getattr(exc, "status", 0) or getattr(exc, "status_code", 0) or 0)
    code = getattr(exc, "code", None)
    message = str(getattr(exc, "message", None) or exc)
    headers = getattr(exc, "headers", {}) or {}
    retry_after = headers.get("Retry-After") if isinstance(headers, dict) else None
    retry_after_seconds = int(retry_after) if isinstance(retry_after, str) and retry_after.isdigit() else None
    return NotionAPIResponseError(
        status_code=status_code or 500,
        provider_code=str(code) if code is not None else None,
        message=message,
        retry_after_seconds=retry_after_seconds,
    )
