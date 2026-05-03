from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, runtime_checkable

import httpx

from cornerstone.config import Settings
from cornerstone.connectors.base import OAuthMaterial
from cornerstone.connectors.providers.google_drive.mapper import GOOGLE_DOC_MIME_TYPE


class GoogleDriveAPIResponseError(RuntimeError):
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
        super().__init__(message or provider_code or f"Google Drive API returned HTTP {status_code}.")
        self.status_code = status_code
        self.provider_code = provider_code
        self.message = message
        self.retry_after_seconds = retry_after_seconds
        self.payload = payload or {}


@runtime_checkable
class GoogleDriveOAuthClient(Protocol):
    async def exchange_code(self, *, code: str, redirect_uri: str) -> OAuthMaterial:
        ...


@runtime_checkable
class GoogleDriveGateway(Protocol):
    async def list_files(
        self,
        *,
        access_token: str,
        page_size: int,
        page_token: str | None = None,
        query: str | None = None,
    ) -> dict[str, Any]:
        ...

    async def get_file(self, *, access_token: str, file_id: str) -> dict[str, Any]:
        ...

    async def export_file_text(self, *, access_token: str, file_id: str, mime_type: str) -> str:
        ...

    async def download_file_text(self, *, access_token: str, file_id: str) -> str:
        ...


class GoogleDriveHttpOAuthClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def exchange_code(self, *, code: str, redirect_uri: str) -> OAuthMaterial:
        async with httpx.AsyncClient(timeout=self._settings.connector_http_timeout_seconds) as client:
            response = await client.post(
                self._settings.google_drive_oauth_token_url,
                data={
                    "client_id": self._settings.google_drive_client_id,
                    "client_secret": self._settings.google_drive_client_secret,
                    "code": code,
                    "grant_type": "authorization_code",
                    "redirect_uri": redirect_uri,
                },
                headers={"Accept": "application/json"},
            )
            if response.status_code >= 400:
                raise _response_to_error(response)
            payload = response.json()
        scopes = str(payload.get("scope") or "").split()
        expires_in = payload.get("expires_in")
        expires_at = None
        if isinstance(expires_in, int):
            expires_at = datetime.now(UTC) + timedelta(seconds=expires_in)
        return OAuthMaterial(
            access_token=str(payload["access_token"]),
            refresh_token=payload.get("refresh_token"),
            granted_scopes=scopes or ["https://www.googleapis.com/auth/drive.readonly"],
            external_account_id=None,
            external_workspace_id=None,
            external_workspace_name="Google Drive",
            external_bot_id=None,
        )


class MockGoogleDriveGateway:
    """Fixture gateway for local/dev tests with no external Google dependency."""

    async def list_files(
        self,
        *,
        access_token: str,
        page_size: int,
        page_token: str | None = None,
        query: str | None = None,
    ) -> dict[str, Any]:
        _ = (access_token, page_size, page_token, query)
        modified = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        return {
            "nextPageToken": None,
            "files": [
                {
                    "id": "google-doc-1",
                    "name": "Cornerstone Google Drive Overview",
                    "mimeType": GOOGLE_DOC_MIME_TYPE,
                    "webViewLink": "https://docs.google.com/document/d/google-doc-1/edit",
                    "modifiedTime": modified,
                    "parents": ["folder-1"],
                    "capabilities": {"canDownload": True},
                },
                {
                    "id": "google-sheet-1",
                    "name": "Cornerstone Metrics Sheet",
                    "mimeType": "application/vnd.google-apps.spreadsheet",
                    "webViewLink": "https://docs.google.com/spreadsheets/d/google-sheet-1/edit",
                    "modifiedTime": modified,
                    "parents": ["folder-1"],
                    "capabilities": {"canDownload": True},
                },
            ],
        }

    async def get_file(self, *, access_token: str, file_id: str) -> dict[str, Any]:
        _ = access_token
        modified = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        mime_type = "text/plain" if file_id == "google-text-1" else GOOGLE_DOC_MIME_TYPE
        return {
            "id": file_id,
            "name": "Cornerstone Google Drive Overview",
            "mimeType": mime_type,
            "webViewLink": f"https://docs.google.com/document/d/{file_id}/edit",
            "modifiedTime": modified,
            "parents": ["folder-1"],
            "capabilities": {"canDownload": True},
        }

    async def export_file_text(self, *, access_token: str, file_id: str, mime_type: str) -> str:
        _ = (access_token, file_id, mime_type)
        return (
            "Cornerstone is a shared organizational context layer.\n"
            "Cornerstone preserves provenance, freshness, trust labels, and review state."
        )

    async def download_file_text(self, *, access_token: str, file_id: str) -> str:
        _ = (access_token, file_id)
        return (
            "Cornerstone is a shared organizational context layer.\n"
            "This text file was downloaded from Google Drive."
        )


class GoogleDriveSdkGateway:
    """google-api-python-client-backed gateway for authenticated Drive access."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    async def list_files(
        self,
        *,
        access_token: str,
        page_size: int,
        page_token: str | None = None,
        query: str | None = None,
    ) -> dict[str, Any]:
        return await asyncio.to_thread(
            self._list_files_sync,
            access_token,
            page_size,
            page_token,
            query,
        )

    async def get_file(self, *, access_token: str, file_id: str) -> dict[str, Any]:
        return await asyncio.to_thread(self._get_file_sync, access_token, file_id)

    async def export_file_text(self, *, access_token: str, file_id: str, mime_type: str) -> str:
        return await asyncio.to_thread(self._export_file_text_sync, access_token, file_id, mime_type)

    async def download_file_text(self, *, access_token: str, file_id: str) -> str:
        return await asyncio.to_thread(self._download_file_text_sync, access_token, file_id)

    def _service(self, access_token: str) -> Any:
        try:
            from google.oauth2.credentials import Credentials
            from googleapiclient.discovery import build
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency installed in packaged env.
            raise RuntimeError(
                "google-api-python-client and google-auth are required for live Google Drive access. "
                "Install project dependencies with `pip install -e '.[dev]'`."
            ) from exc
        credentials = Credentials(token=access_token)
        return build("drive", "v3", credentials=credentials, cache_discovery=False)

    def _list_files_sync(
        self,
        access_token: str,
        page_size: int,
        page_token: str | None,
        query: str | None,
    ) -> dict[str, Any]:
        try:
            request = self._service(access_token).files().list(
                pageSize=page_size,
                pageToken=page_token,
                q=query,
                fields=(
                    "nextPageToken, files(id,name,mimeType,webViewLink,modifiedTime,"
                    "trashed,parents,driveId,capabilities)"
                ),
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
            )
            return _ensure_dict(request.execute())
        except Exception as exc:  # pragma: no cover - exercised with adapter contract tests.
            raise _exception_to_error(exc) from exc

    def _get_file_sync(self, access_token: str, file_id: str) -> dict[str, Any]:
        try:
            request = self._service(access_token).files().get(
                fileId=file_id,
                fields="id,name,mimeType,webViewLink,modifiedTime,trashed,parents,driveId,capabilities",
                supportsAllDrives=True,
            )
            return _ensure_dict(request.execute())
        except Exception as exc:  # pragma: no cover
            raise _exception_to_error(exc) from exc

    def _export_file_text_sync(self, access_token: str, file_id: str, mime_type: str) -> str:
        try:
            request = self._service(access_token).files().export(fileId=file_id, mimeType=mime_type)
            payload = request.execute()
        except Exception as exc:  # pragma: no cover
            raise _exception_to_error(exc) from exc
        if isinstance(payload, bytes):
            return payload.decode("utf-8", errors="replace")
        return str(payload)

    def _download_file_text_sync(self, access_token: str, file_id: str) -> str:
        try:
            request = self._service(access_token).files().get_media(fileId=file_id, supportsAllDrives=True)
            payload = request.execute()
        except Exception as exc:  # pragma: no cover
            raise _exception_to_error(exc) from exc
        if isinstance(payload, bytes):
            return payload.decode("utf-8", errors="replace")
        return str(payload)


def _response_to_error(response: httpx.Response) -> GoogleDriveAPIResponseError:
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    error_payload = payload.get("error") if isinstance(payload, dict) else {}
    if isinstance(error_payload, dict):
        provider_code = error_payload.get("status") or error_payload.get("code")
        message = error_payload.get("message")
    else:
        provider_code = None
        message = None
    retry_after = response.headers.get("Retry-After")
    retry_after_seconds = int(retry_after) if retry_after and retry_after.isdigit() else None
    return GoogleDriveAPIResponseError(
        status_code=response.status_code,
        provider_code=str(provider_code) if provider_code is not None else None,
        message=str(message) if message is not None else None,
        retry_after_seconds=retry_after_seconds,
        payload=payload if isinstance(payload, dict) else {},
    )


def _exception_to_error(exc: Exception) -> GoogleDriveAPIResponseError:
    status_code = getattr(exc, "status_code", None) or getattr(getattr(exc, "resp", None), "status", None)
    content = getattr(exc, "content", None)
    message = str(exc)
    if isinstance(content, bytes):
        message = content.decode("utf-8", errors="replace")
    return GoogleDriveAPIResponseError(
        status_code=int(status_code or 500),
        provider_code=None,
        message=message,
    )


def _ensure_dict(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}
