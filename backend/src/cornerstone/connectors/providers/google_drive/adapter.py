from __future__ import annotations

from urllib.parse import urlencode

from cornerstone.config import Settings
from cornerstone.connectors.base import DiscoveryPage, OAuthMaterial
from cornerstone.connectors.providers.common import connector_error_from_api_response
from cornerstone.connectors.providers.google_drive.gateway import (
    GoogleDriveAPIResponseError,
    GoogleDriveGateway,
    GoogleDriveHttpOAuthClient,
    GoogleDriveOAuthClient,
    GoogleDriveSdkGateway,
    MockGoogleDriveGateway,
)
from cornerstone.connectors.providers.google_drive.mapper import (
    GOOGLE_DOC_MIME_TYPE,
    TEXT_MIME_TYPES,
    drive_file_to_snapshot,
    drive_file_to_source_object,
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


class GoogleDriveConnector:
    provider = DataSourceType.GOOGLE_DRIVE

    def __init__(
        self,
        settings: Settings,
        *,
        gateway: GoogleDriveGateway | None = None,
        oauth_client: GoogleDriveOAuthClient | None = None,
    ) -> None:
        self._settings = settings
        self._gateway = gateway or (
            MockGoogleDriveGateway()
            if settings.google_drive_mock_external_api
            else GoogleDriveSdkGateway(settings)
        )
        self._oauth_client = oauth_client or GoogleDriveHttpOAuthClient(settings)

    @property
    def gateway_name(self) -> str:
        return type(self._gateway).__name__

    def build_authorization_url(self, intent: ConnectionIntent) -> str:
        scopes = intent.requested_scopes or ["https://www.googleapis.com/auth/drive.readonly"]
        query = urlencode(
            {
                "client_id": self._settings.google_drive_client_id,
                "redirect_uri": intent.redirect_uri,
                "response_type": "code",
                "scope": " ".join(scopes),
                "state": intent.state_nonce,
                "access_type": "offline",
                "prompt": "consent",
                "include_granted_scopes": "true",
            }
        )
        return f"{self._settings.google_drive_oauth_authorize_url}?{query}"

    async def complete_authorization(self, *, code: str, redirect_uri: str) -> OAuthMaterial:
        if self._settings.google_drive_mock_external_api:
            return OAuthMaterial(
                access_token=f"mock-google-drive-access-token-{code}",
                refresh_token=f"mock-google-drive-refresh-token-{code}",
                granted_scopes=["https://www.googleapis.com/auth/drive.readonly"],
                external_account_id="mock-google-user",
                external_workspace_id="mock-google-drive",
                external_workspace_name="Mock Google Drive",
            )
        try:
            return await self._oauth_client.exchange_code(code=code, redirect_uri=redirect_uri)
        except GoogleDriveAPIResponseError as exc:
            raise GoogleDriveProviderError(self.map_api_error(exc)) from exc

    async def test_connection(
        self,
        *,
        credential: ConnectorCredential,
        access_token: str,
    ) -> ConnectionTestResult:
        try:
            payload = await self._gateway.list_files(
                access_token=access_token,
                page_size=1,
                query=self._settings.google_drive_discovery_query,
            )
        except GoogleDriveAPIResponseError as exc:
            error = self.map_api_error(exc)
            return ConnectionTestResult(
                status=ConnectionTestStatus.FAILED,
                datasource_id=credential.datasource_id,
                provider=DataSourceType.GOOGLE_DRIVE,
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
            provider=DataSourceType.GOOGLE_DRIVE,
            external_workspace_name=credential.external_workspace_name,
            granted_scopes=credential.granted_scopes,
            can_read_objects=True,
            sample_object_count=len(payload.get("files", [])),
            limitations=(
                ["Google Drive connector is in mock external API mode; discovery returns fixture files."]
                if self._settings.google_drive_mock_external_api
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
        try:
            payload = await self._gateway.list_files(
                access_token=access_token,
                page_size=page_size,
                page_token=cursor,
                query=self._settings.google_drive_discovery_query,
            )
        except GoogleDriveAPIResponseError as exc:
            raise GoogleDriveProviderError(self.map_api_error(exc)) from exc
        objects = [
            drive_file_to_snapshot(credential.datasource_id, file_payload)
            for file_payload in payload.get("files", [])
            if isinstance(file_payload, dict)
        ]
        return DiscoveryPage(
            objects=objects,
            next_cursor=payload.get("nextPageToken"),
            has_more=bool(payload.get("nextPageToken")),
        )

    async def retrieve_file_snapshot(
        self,
        *,
        credential: ConnectorCredential,
        access_token: str,
        file_id: str,
    ) -> ProviderObjectSnapshot:
        """Retrieve one Google Drive file and map it to a selectable provider snapshot."""

        try:
            file_payload = await self._gateway.get_file(access_token=access_token, file_id=file_id)
        except GoogleDriveAPIResponseError as exc:
            raise GoogleDriveProviderError(self.map_api_error(exc)) from exc
        return drive_file_to_snapshot(credential.datasource_id, file_payload)

    async def list_objects(
        self,
        *,
        credential: ConnectorCredential,
        access_token: str,
        selection: SourceSelection | None,
        selected_objects: list[ProviderObjectSnapshot] | None = None,
    ) -> list[SourceObject]:
        if selected_objects is None:
            selected_objects = _snapshots_from_selection(credential.datasource_id, selection)
        if not selected_objects:
            return []

        objects: list[SourceObject] = []
        for snapshot in selected_objects:
            if not snapshot.ingestion_supported:
                continue
            if snapshot.object_type not in {ProviderObjectType.DOCUMENT, ProviderObjectType.TEXT_FILE}:
                continue
            objects.append(await self._fetch_file_as_source_object(access_token=access_token, snapshot=snapshot))
        return objects

    async def _fetch_file_as_source_object(
        self,
        *,
        access_token: str,
        snapshot: ProviderObjectSnapshot,
    ) -> SourceObject:
        try:
            file_payload = await self._gateway.get_file(access_token=access_token, file_id=snapshot.external_id)
            mime_type = str(file_payload.get("mimeType") or snapshot.provider_metadata.get("mimeType") or "")
            if mime_type == GOOGLE_DOC_MIME_TYPE:
                content = await self._gateway.export_file_text(
                    access_token=access_token,
                    file_id=snapshot.external_id,
                    mime_type=self._settings.google_drive_export_mime_type,
                )
                content_format = "google_drive_export_text"
            elif mime_type in TEXT_MIME_TYPES:
                content = await self._gateway.download_file_text(
                    access_token=access_token,
                    file_id=snapshot.external_id,
                )
                content_format = "google_drive_download_text"
            else:
                raise GoogleDriveProviderError(
                    ConnectorError(
                        code=ConnectorErrorCode.UNSUPPORTED_OBJECT_TYPE,
                        user_message="Selected Google Drive file type is not supported for ingestion.",
                        retryable=False,
                        next_action=ConnectorNextAction.SELECT_SOURCES,
                    )
                )
        except GoogleDriveAPIResponseError as exc:
            raise GoogleDriveProviderError(self.map_api_error(exc)) from exc
        return drive_file_to_source_object(
            file_payload=file_payload,
            snapshot=snapshot,
            content=content,
            content_format=content_format,
        )

    def map_api_error(self, error: GoogleDriveAPIResponseError) -> ConnectorError:
        return connector_error_from_api_response(self.map_provider_error, error)

    def map_provider_error(
        self,
        *,
        status_code: int,
        provider_code: str | None = None,
        technical_message: str | None = None,
        retry_after_seconds: int | None = None,
    ) -> ConnectorError:
        code_text = (provider_code or "").lower()
        if status_code == 401 or code_text in {"unauthorized", "invalid_grant"}:
            return ConnectorError(
                code=ConnectorErrorCode.TOKEN_REVOKED,
                user_message="The Google Drive connection is no longer authorized. Reconnect Google Drive.",
                technical_message=technical_message,
                retryable=False,
                next_action=ConnectorNextAction.RECONNECT,
            )
        if status_code == 403 or code_text in {"forbidden", "insufficientpermissions", "insufficient_scope"}:
            return ConnectorError(
                code=ConnectorErrorCode.PERMISSION_DENIED,
                user_message=(
                    "Cornerstone cannot read the selected Google Drive file. "
                    "Grant Drive read access or share the file with the authorized account."
                ),
                technical_message=technical_message,
                retryable=False,
                next_action=ConnectorNextAction.GRANT_PERMISSION,
            )
        if status_code == 404:
            return ConnectorError(
                code=ConnectorErrorCode.OBJECT_NOT_FOUND,
                user_message="The selected Google Drive file was not found or is no longer accessible.",
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
                user_message=f"Google Drive rate-limited this sync.{suffix}",
                technical_message=technical_message,
                retryable=True,
                next_action=ConnectorNextAction.WAIT_AND_RETRY,
                retry_after_seconds=retry_after_seconds,
            )
        if 500 <= status_code <= 599:
            return ConnectorError(
                code=ConnectorErrorCode.PROVIDER_UNAVAILABLE,
                user_message="Google Drive is temporarily unavailable. Retry the sync later.",
                technical_message=technical_message,
                retryable=True,
                next_action=ConnectorNextAction.RETRY,
            )
        return ConnectorError(
            code=ConnectorErrorCode.UNKNOWN,
            user_message="Google Drive returned an unexpected error. Review the connector logs.",
            technical_message=technical_message,
            retryable=False,
            next_action=ConnectorNextAction.CONTACT_ADMIN,
        )


class GoogleDriveProviderError(RuntimeError):
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
            provider=DataSourceType.GOOGLE_DRIVE,
            external_id=external_id,
            object_type=ProviderObjectType.DOCUMENT,
            title=external_id,
            access_state=ProviderObjectAccessState.ACCESSIBLE,
            raw_metadata_hash=external_id * 4 if len(external_id) < 16 else external_id,
            selected_for_sync=True,
            ingestion_supported=True,
            ingestion_unsupported_reason=None,
            provider_metadata={"selectionOnly": True, "mimeType": GOOGLE_DOC_MIME_TYPE},
        )
        for external_id in selection.selected_external_object_ids
    ]
