from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from cornerstone.schemas import (
    ConnectionIntent,
    ConnectionTestResult,
    ConnectorCredential,
    DataSourceType,
    ProviderObjectSnapshot,
    SourceObject,
    SourceSelection,
)


@dataclass(frozen=True)
class OAuthMaterial:
    access_token: str
    refresh_token: str | None = None
    granted_scopes: list[str] | None = None
    external_account_id: str | None = None
    external_workspace_id: str | None = None
    external_workspace_name: str | None = None
    external_bot_id: str | None = None


@dataclass(frozen=True)
class DiscoveryPage:
    objects: list[ProviderObjectSnapshot]
    next_cursor: str | None = None
    has_more: bool = False


class Connector(Protocol):
    """Provider adapter contract.

    Runtime routes and sync jobs should call this interface rather than provider-specific code.
    """

    provider: DataSourceType

    def build_authorization_url(self, intent: ConnectionIntent) -> str:
        ...

    async def complete_authorization(self, *, code: str, redirect_uri: str) -> OAuthMaterial:
        ...

    async def test_connection(
        self,
        *,
        credential: ConnectorCredential,
        access_token: str,
    ) -> ConnectionTestResult:
        ...

    async def discover_objects(
        self,
        *,
        credential: ConnectorCredential,
        access_token: str,
        page_size: int,
        cursor: str | None = None,
    ) -> DiscoveryPage:
        ...

    async def list_objects(
        self,
        *,
        credential: ConnectorCredential,
        access_token: str,
        selection: SourceSelection | None,
        selected_objects: list[ProviderObjectSnapshot] | None = None,
    ) -> list[SourceObject]:
        ...
