from __future__ import annotations

import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import RedirectResponse

from cornerstone.api.dependencies import get_store
from cornerstone.config import Settings, get_settings
from cornerstone.connectors.catalog import get_connector_definition
from cornerstone.connectors.google_drive import GoogleDriveProviderError
from cornerstone.connectors.notion import NotionProviderError
from cornerstone.connectors.registry import get_connector, get_token_cipher
from cornerstone.observability import log_event
from cornerstone.schemas import (
    AuditEvent,
    ConnectionIntent,
    ConnectionIntentStatus,
    ConnectionTestResult,
    ConnectionTestStatus,
    ConnectorAuthType,
    ConnectorCredential,
    ConnectorCredentialPublic,
    ConnectorError,
    ConnectorErrorCode,
    ConnectorNextAction,
    CreateConnectionIntentRequest,
    CredentialStatus,
    DataSource,
    DataSourceAuthStatus,
    DataSourceConnectionStatus,
    DataSourceNextAction,
    DataSourceStatus,
    DataSourceSyncStatus,
    DataSourceType,
    DiscoverProviderObjectsRequest,
    DiscoverProviderObjectsResponse,
    ErrorInfo,
    FreshnessState,
    OAuthCallbackResponse,
    ProviderObjectAccessState,
    ProviderObjectSnapshot,
    ProviderObjectSnapshotListResponse,
    SourceSelection,
    SourceSelectionMode,
    SyncJob,
    UpsertSourceSelectionRequest,
    utc_now,
)
from cornerstone.store import NotFoundError

router = APIRouter(tags=["connectors"])

_RUNTIME_PROVIDERS = {DataSourceType.NOTION, DataSourceType.GOOGLE_DRIVE}
_CONNECTOR_SYNCABLE_STATUSES = {
    DataSourceStatus.CONNECTED,
    DataSourceStatus.SYNC_PENDING,
    DataSourceStatus.DEGRADED,
    DataSourceStatus.STALE,
}


@router.post(
    "/connections/intents",
    response_model=ConnectionIntent,
    status_code=status.HTTP_201_CREATED,
)
def create_connection_intent(
    request: CreateConnectionIntentRequest,
    store: Any = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> ConnectionIntent:
    _ensure_runtime_provider(request.provider)
    definition = get_connector_definition(request.provider, settings)
    redirect_uri = _oauth_callback_url(request.provider, settings)
    state_nonce = secrets.token_urlsafe(32)
    requested_scopes = request.requested_scopes or [scope.key for scope in definition.required_scopes]
    expires_at = utc_now() + timedelta(minutes=settings.connector_intent_ttl_minutes)
    intent = ConnectionIntent(
        provider=request.provider,
        status=ConnectionIntentStatus.CREATED,
        auth_type=definition.auth_type,
        source_name=request.source_name,
        created_by=request.created_by,
        requested_scopes=requested_scopes,
        redirect_uri=redirect_uri,
        return_url=request.return_url,
        state_nonce=state_nonce,
        expires_at=expires_at,
    )
    connector = get_connector(request.provider, settings)
    intent = intent.model_copy(update={"authorization_url": connector.build_authorization_url(intent)}, deep=True)
    saved = cast(ConnectionIntent, store.add_connection_intent(intent))
    log_event(
        "connection.intent_created",
        intentId=saved.id,
        provider=saved.provider,
        createdBy=saved.created_by,
        expiresAt=saved.expires_at,
    )
    return saved


@router.get("/connections/intents/{intent_id}", response_model=ConnectionIntent)
def get_connection_intent(intent_id: str, store: Any = Depends(get_store)) -> ConnectionIntent:
    intent = _get_intent_or_404(intent_id, store)
    log_event("connection.intent_read", intentId=intent.id, provider=intent.provider, status=intent.status)
    return intent


@router.get("/oauth/{provider}/authorize", response_class=RedirectResponse)
def authorize_connector_oauth(
    provider: DataSourceType,
    intent_id: str = Query(alias="intentId"),
    store: Any = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> RedirectResponse:
    _ensure_runtime_provider(provider)
    intent = _get_intent_or_404(intent_id, store)
    if intent.provider != provider:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Intent provider mismatch.")
    _ensure_intent_active(intent, store)
    authorization_url = intent.authorization_url
    if not authorization_url:
        connector = get_connector(provider, settings)
        authorization_url = connector.build_authorization_url(intent)
    updated = intent.model_copy(
        update={
            "status": ConnectionIntentStatus.OAUTH_REDIRECTED,
            "authorization_url": authorization_url,
        },
        deep=True,
    )
    saved = store.update_connection_intent(updated)
    log_event(
        "oauth.redirect_started",
        intentId=saved.id,
        provider=saved.provider,
        status=saved.status,
    )
    return RedirectResponse(url=authorization_url, status_code=status.HTTP_307_TEMPORARY_REDIRECT)


@router.get("/oauth/{provider}/callback", response_model=OAuthCallbackResponse)
async def complete_connector_oauth(
    provider: DataSourceType,
    state: str,
    code: str | None = None,
    error: str | None = None,
    store: Any = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> OAuthCallbackResponse:
    _ensure_runtime_provider(provider)
    intent = _get_intent_by_state_or_404(state, store)
    if intent.provider != provider:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OAuth state provider mismatch.")
    _ensure_intent_active(intent, store)
    log_event("oauth.callback_received", intentId=intent.id, provider=provider, hasCode=code is not None, error=error)

    if error is not None:
        connector_error = ConnectorError(
            code=ConnectorErrorCode.OAUTH_FAILED,
            user_message="The provider rejected the OAuth authorization request.",
            technical_message=error,
            retryable=False,
            next_action=ConnectorNextAction.RECONNECT,
        )
        failed = intent.model_copy(update={"status": ConnectionIntentStatus.FAILED, "failure_error": connector_error}, deep=True)
        saved = store.update_connection_intent(failed)
        log_event("oauth.failed", intentId=saved.id, provider=provider, errorCode=connector_error.code)
        return OAuthCallbackResponse(intent=saved, next_action=ConnectorNextAction.RECONNECT)

    if code is None:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="OAuth callback requires code or error.")

    connector = get_connector(provider, settings)
    try:
        material = await connector.complete_authorization(code=code, redirect_uri=intent.redirect_uri)
    except Exception as exc:  # pragma: no cover - live provider failure path
        connector_error = ConnectorError(
            code=ConnectorErrorCode.OAUTH_FAILED,
            user_message="OAuth authorization could not be completed.",
            technical_message=str(exc),
            retryable=True,
            next_action=ConnectorNextAction.RECONNECT,
        )
        failed = intent.model_copy(update={"status": ConnectionIntentStatus.FAILED, "failure_error": connector_error}, deep=True)
        saved = store.update_connection_intent(failed)
        log_event("oauth.failed", intentId=saved.id, provider=provider, errorType=type(exc).__name__)
        return OAuthCallbackResponse(intent=saved, next_action=ConnectorNextAction.RECONNECT)

    cipher = get_token_cipher(settings)
    now = utc_now()
    data_source = DataSource(
        type=provider,
        name=intent.source_name,
        status=DataSourceStatus.SYNC_PENDING,
        production_enabled=True,
        auth_status=DataSourceAuthStatus.AUTHORIZED,
        connection_status=DataSourceConnectionStatus.UNTESTED,
        sync_status=DataSourceSyncStatus.NEVER_SYNCED,
        next_action=DataSourceNextAction.TEST_CONNECTION,
        freshness_state=FreshnessState.UNKNOWN,
        sync_freshness_state=FreshnessState.UNKNOWN,
        content_freshness_state=FreshnessState.UNKNOWN,
        created_at=now,
    )
    credential = ConnectorCredential(
        datasource_id=data_source.id,
        provider=provider,
        auth_type=ConnectorAuthType.OAUTH2,
        encrypted_access_token=cipher.encrypt(material.access_token),
        encrypted_refresh_token=(
            cipher.encrypt(material.refresh_token) if material.refresh_token is not None else None
        ),
        granted_scopes=material.granted_scopes or [],
        external_account_id=material.external_account_id,
        external_workspace_id=material.external_workspace_id,
        external_workspace_name=material.external_workspace_name,
        external_bot_id=material.external_bot_id,
        status=CredentialStatus.ACTIVE,
        created_at=now,
        updated_at=now,
    )
    with store.transaction():
        saved_source = store.add_data_source(data_source)
        saved_credential = store.add_connector_credential(credential)
        completed = intent.model_copy(
            update={
                "status": ConnectionIntentStatus.COMPLETED,
                "completed_at": now,
                "datasource_id": saved_source.id,
                "failure_error": None,
            },
            deep=True,
        )
        saved_intent = store.update_connection_intent(completed)
        store.add_audit_event(
            AuditEvent(
                event_type="oauth.completed",
                actor=intent.created_by,
                entity_type="data_source",
                entity_id=saved_source.id,
                metadata={"provider": str(provider), "intentId": intent.id},
            )
        )

    log_event(
        "oauth.completed",
        intentId=saved_intent.id,
        sourceId=saved_source.id,
        provider=provider,
        externalWorkspaceId=saved_credential.external_workspace_id,
    )
    return OAuthCallbackResponse(
        intent=saved_intent,
        data_source=saved_source,
        credential=_public_credential(saved_credential),
        next_action=ConnectorNextAction.TEST_CONNECTION,
    )


@router.get("/sources/{source_id}", response_model=DataSource)
def get_source(source_id: str, store: Any = Depends(get_store)) -> DataSource:
    return _get_source_or_404(source_id, store)


@router.post("/sources/{source_id}/test", response_model=ConnectionTestResult)
async def test_source_connection(
    source_id: str,
    store: Any = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> ConnectionTestResult:
    data_source = _get_source_or_404(source_id, store)
    _ensure_runtime_provider(data_source.type)
    credential = _get_active_credential_or_409(source_id, store)
    access_token = get_token_cipher(settings).decrypt(credential.encrypted_access_token)
    connector = get_connector(data_source.type, settings)
    log_event("source.test_started", sourceId=source_id, provider=data_source.type)
    result = await connector.test_connection(credential=credential, access_token=access_token)
    if result.status == ConnectionTestStatus.PASSED:
        next_action = (
            DataSourceNextAction.DISCOVER_SOURCES
            if result.can_read_objects
            else DataSourceNextAction.GRANT_PERMISSION
        )
        connection_status = (
            DataSourceConnectionStatus.TEST_PASSED
            if result.can_read_objects
            else DataSourceConnectionStatus.PERMISSION_LIMITED
        )
        updated = data_source.model_copy(
            update={
                "status": DataSourceStatus.CONNECTED,
                "connection_status": connection_status,
                "next_action": next_action,
                "last_connection_test_at": result.tested_at,
                "last_error": None,
            },
            deep=True,
        )
        store.update_data_source(updated)
        log_event("source.test_succeeded", sourceId=source_id, provider=data_source.type)
    else:
        connector_error = result.error or ConnectorError(
            code=ConnectorErrorCode.CONNECTION_TEST_FAILED,
            user_message="The connector test failed.",
            retryable=True,
            next_action=ConnectorNextAction.RETRY,
        )
        failed = data_source.model_copy(
            update={
                "status": DataSourceStatus.FAILED,
                "connection_status": DataSourceConnectionStatus.TEST_FAILED,
                "next_action": (
                    DataSourceNextAction.GRANT_PERMISSION
                    if connector_error.next_action == ConnectorNextAction.GRANT_PERMISSION
                    else DataSourceNextAction.RECONNECT
                ),
                "last_connection_test_at": result.tested_at,
                "last_error": ErrorInfo(code=str(connector_error.code), message=connector_error.user_message),
            },
            deep=True,
        )
        store.update_data_source(failed)
        log_event("source.test_failed", sourceId=source_id, provider=data_source.type, errorCode=connector_error.code)
    return result


@router.post("/sources/{source_id}/discover", response_model=DiscoverProviderObjectsResponse)
async def discover_source_objects(
    source_id: str,
    request: DiscoverProviderObjectsRequest | None = None,
    store: Any = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> DiscoverProviderObjectsResponse:
    data_source = _get_source_or_404(source_id, store)
    _ensure_runtime_provider(data_source.type)
    if data_source.connection_status != DataSourceConnectionStatus.TEST_PASSED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Test the source connection before discovery.",
        )
    credential = _get_active_credential_or_409(source_id, store)
    access_token = get_token_cipher(settings).decrypt(credential.encrypted_access_token)
    connector = get_connector(data_source.type, settings)
    options = request or DiscoverProviderObjectsRequest()
    log_event(
        "source.discovery_started",
        sourceId=source_id,
        provider=data_source.type,
        pageSize=options.page_size,
        cursor=options.cursor,
    )
    try:
        page = await connector.discover_objects(
            credential=credential,
            access_token=access_token,
            page_size=options.page_size,
            cursor=options.cursor,
        )
    except (NotionProviderError, GoogleDriveProviderError) as exc:
        error = exc.connector_error
        failed = data_source.model_copy(
            update={
                "connection_status": (
                    DataSourceConnectionStatus.PERMISSION_LIMITED
                    if error.next_action == ConnectorNextAction.GRANT_PERMISSION
                    else data_source.connection_status
                ),
                "next_action": _source_next_action_from_connector_action(error.next_action),
                "last_error": ErrorInfo(code=str(error.code), message=error.user_message),
            },
            deep=True,
        )
        store.update_data_source(failed)
        log_event("source.discovery_failed", sourceId=source_id, provider=data_source.type, errorCode=error.code)
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=error.model_dump(mode="json", by_alias=True)) from exc

    saved_objects = store.upsert_provider_object_snapshots(source_id, page.objects)
    all_objects = store.list_provider_object_snapshots(datasource_id=source_id)
    accessible_count = sum(1 for item in all_objects if item.access_state == ProviderObjectAccessState.ACCESSIBLE)
    now = utc_now()
    next_action = DataSourceNextAction.SELECT_SOURCES if accessible_count else DataSourceNextAction.GRANT_PERMISSION
    connection_status = (
        DataSourceConnectionStatus.TEST_PASSED
        if accessible_count
        else DataSourceConnectionStatus.PERMISSION_LIMITED
    )
    updated_source = data_source.model_copy(
        update={
            "status": DataSourceStatus.CONNECTED if accessible_count else DataSourceStatus.DEGRADED,
            "connection_status": connection_status,
            "next_action": next_action,
            "last_discovery_at": now,
            "discovered_object_count": len(all_objects),
            "last_error": None if accessible_count else ErrorInfo(
                code="no_accessible_provider_objects",
                message="No readable provider objects were found. Share content with the connector account and retry discovery.",
            ),
        },
        deep=True,
    )
    updated_source = store.update_data_source(updated_source)
    log_event(
        "source.discovery_succeeded",
        sourceId=source_id,
        provider=data_source.type,
        discoveredObjectCount=len(saved_objects),
        totalDiscoveredObjectCount=len(all_objects),
        accessibleObjectCount=accessible_count,
        nextAction=updated_source.next_action,
    )
    return DiscoverProviderObjectsResponse(
        data_source=updated_source,
        objects=saved_objects,
        next_cursor=page.next_cursor,
        has_more=page.has_more,
    )


@router.get("/sources/{source_id}/objects", response_model=ProviderObjectSnapshotListResponse)
def list_source_provider_objects(source_id: str, store: Any = Depends(get_store)) -> ProviderObjectSnapshotListResponse:
    _get_source_or_404(source_id, store)
    objects = store.list_provider_object_snapshots(datasource_id=source_id)
    return ProviderObjectSnapshotListResponse(
        datasource_id=source_id,
        objects=objects,
        total_count=len(objects),
        accessible_count=sum(1 for item in objects if item.access_state == ProviderObjectAccessState.ACCESSIBLE),
        syncable_count=sum(1 for item in objects if _is_provider_object_syncable(item)),
        selected_count=sum(1 for item in objects if item.selected_for_sync),
    )


@router.get("/sources/{source_id}/selections", response_model=SourceSelection)
def get_source_selection(source_id: str, store: Any = Depends(get_store)) -> SourceSelection:
    _get_source_or_404(source_id, store)
    try:
        return cast(SourceSelection, store.get_source_selection(source_id))
    except NotFoundError:
        return SourceSelection(datasource_id=source_id)


@router.put("/sources/{source_id}/selections", response_model=SourceSelection)
def upsert_source_selection(
    source_id: str,
    request: UpsertSourceSelectionRequest,
    store: Any = Depends(get_store),
) -> SourceSelection:
    data_source = _get_source_or_404(source_id, store)
    discovered = store.list_provider_object_snapshots(datasource_id=source_id)
    discovered_by_id = {item.external_id: item for item in discovered}
    syncable_by_id = {
        item.external_id: item
        for item in discovered
        if _is_provider_object_syncable(item)
    }
    selected_ids = list(request.selected_external_object_ids)
    if request.sync_mode == SourceSelectionMode.WORKSPACE_LIMITED:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": ConnectorErrorCode.UNSUPPORTED_OBJECT_TYPE,
                "userMessage": "workspace_limited selection is not implemented yet. Use selected_only or all_accessible.",
                "nextAction": ConnectorNextAction.SELECT_SOURCES,
            },
        )
    if request.sync_mode == SourceSelectionMode.ALL_ACCESSIBLE:
        selected_ids = list(syncable_by_id)
    if not discovered:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Discover provider objects before saving a source selection.",
        )
    if request.sync_mode == SourceSelectionMode.SELECTED_ONLY and not selected_ids:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Select at least one discovered provider object.",
        )
    if request.sync_mode == SourceSelectionMode.ALL_ACCESSIBLE and not selected_ids:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": ConnectorErrorCode.UNSUPPORTED_OBJECT_TYPE,
                "userMessage": "No discovered provider objects are currently supported for ingestion. Select supported pages after discovery.",
                "nextAction": ConnectorNextAction.SELECT_SOURCES,
            },
        )
    unknown_ids = sorted(set(selected_ids) - set(discovered_by_id))
    inaccessible_ids = sorted(
        item.external_id
        for item in discovered
        if item.external_id in selected_ids and item.access_state != ProviderObjectAccessState.ACCESSIBLE
    )
    unsupported_ids = sorted(
        item.external_id
        for item in discovered
        if item.external_id in selected_ids
        and item.access_state == ProviderObjectAccessState.ACCESSIBLE
        and not item.ingestion_supported
    )
    if unknown_ids or inaccessible_ids or unsupported_ids:
        user_message = "Only accessible provider objects that are supported for ingestion can be selected for sync."
        if unsupported_ids and not unknown_ids and not inaccessible_ids:
            user_message = "The selected provider objects are discoverable but not supported for ingestion in this backend version."
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": (
                    ConnectorErrorCode.UNSUPPORTED_OBJECT_TYPE
                    if unsupported_ids and not unknown_ids and not inaccessible_ids
                    else ConnectorErrorCode.PERMISSION_DENIED
                ),
                "userMessage": user_message,
                "nextAction": ConnectorNextAction.SELECT_SOURCES,
                "unknownExternalObjectIds": unknown_ids,
                "inaccessibleExternalObjectIds": inaccessible_ids,
                "unsupportedExternalObjectIds": unsupported_ids,
                "unsupportedReasons": {
                    item.external_id: item.ingestion_unsupported_reason
                    for item in discovered
                    if item.external_id in unsupported_ids
                },
            },
        )
    now = utc_now()
    selection = SourceSelection(
        datasource_id=source_id,
        sync_mode=request.sync_mode,
        include_rules=request.include_rules,
        exclude_rules=request.exclude_rules,
        selected_external_object_ids=selected_ids,
        updated_at=now,
    )
    with store.transaction():
        saved = cast(SourceSelection, store.upsert_source_selection(selection))
        store.mark_provider_object_selection(source_id, selected_ids)
        store.update_data_source(
            data_source.model_copy(
                update={
                    "selected_object_count": len(selected_ids),
                    "next_action": DataSourceNextAction.RUN_FIRST_SYNC if selected_ids else DataSourceNextAction.SELECT_SOURCES,
                    "last_error": None,
                },
                deep=True,
            )
        )
    log_event(
        "source.selection_saved",
        sourceId=source_id,
        selectionId=saved.id,
        syncMode=saved.sync_mode,
        selectedObjectCount=len(saved.selected_external_object_ids),
    )
    return saved


@router.post("/sources/{source_id}/disconnect", response_model=DataSource)
def disconnect_source(source_id: str, store: Any = Depends(get_store)) -> DataSource:
    data_source = _get_source_or_404(source_id, store)
    try:
        credential = store.get_active_credential_for_source(source_id)
    except NotFoundError:
        credential = None
    now = utc_now()
    if credential is not None:
        revoked = credential.model_copy(
            update={"status": CredentialStatus.REVOKED, "revoked_at": now, "updated_at": now},
            deep=True,
        )
        store.update_connector_credential(revoked)
    disconnected = data_source.model_copy(
        update={
            "status": DataSourceStatus.DISCONNECTED,
            "auth_status": DataSourceAuthStatus.REVOKED,
            "connection_status": DataSourceConnectionStatus.UNTESTED,
            "next_action": DataSourceNextAction.RECONNECT,
            "last_error": None,
        },
        deep=True,
    )
    saved = cast(DataSource, store.update_data_source(disconnected))
    store.add_audit_event(
        AuditEvent(
            event_type="credential.revoked",
            actor="system",
            entity_type="data_source",
            entity_id=source_id,
            metadata={"provider": str(data_source.type)},
        )
    )
    log_event("credential.revoked", sourceId=source_id, provider=data_source.type)
    return saved



def _is_provider_object_syncable(snapshot: ProviderObjectSnapshot) -> bool:
    return snapshot.access_state == ProviderObjectAccessState.ACCESSIBLE and snapshot.ingestion_supported


def _get_selected_provider_snapshots_for_sync(source_id: str, store: Any) -> list[ProviderObjectSnapshot]:
    snapshots = cast(list[ProviderObjectSnapshot], store.list_provider_object_snapshots(datasource_id=source_id))
    return [snapshot for snapshot in snapshots if snapshot.selected_for_sync and _is_provider_object_syncable(snapshot)]


def _source_next_action_from_connector_action(action: ConnectorNextAction) -> DataSourceNextAction:
    if action == ConnectorNextAction.GRANT_PERMISSION:
        return DataSourceNextAction.GRANT_PERMISSION
    if action == ConnectorNextAction.RECONNECT:
        return DataSourceNextAction.RECONNECT
    if action in {ConnectorNextAction.RETRY, ConnectorNextAction.WAIT_AND_RETRY}:
        return DataSourceNextAction.RETRY_SYNC
    if action == ConnectorNextAction.TEST_CONNECTION:
        return DataSourceNextAction.TEST_CONNECTION
    if action == ConnectorNextAction.SELECT_SOURCES:
        return DataSourceNextAction.SELECT_SOURCES
    return DataSourceNextAction.NONE


def _public_credential(credential: ConnectorCredential) -> ConnectorCredentialPublic:
    return ConnectorCredentialPublic(
        id=credential.id,
        datasource_id=credential.datasource_id,
        provider=credential.provider,
        auth_type=credential.auth_type,
        status=credential.status,
        granted_scopes=credential.granted_scopes,
        external_account_id=credential.external_account_id,
        external_workspace_id=credential.external_workspace_id,
        external_workspace_name=credential.external_workspace_name,
        token_expires_at=credential.token_expires_at,
        created_at=credential.created_at,
        updated_at=credential.updated_at,
        revoked_at=credential.revoked_at,
    )


def _ensure_runtime_provider(provider: DataSourceType) -> None:
    if provider not in _RUNTIME_PROVIDERS:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"Connector provider '{provider}' is cataloged but not implemented in this backend slice.",
        )


def _get_intent_or_404(intent_id: str, store: Any) -> ConnectionIntent:
    try:
        return cast(ConnectionIntent, store.get_connection_intent(intent_id))
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


def _get_intent_by_state_or_404(state_nonce: str, store: Any) -> ConnectionIntent:
    try:
        return cast(ConnectionIntent, store.get_connection_intent_by_state_nonce(state_nonce))
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="OAuth state not found.") from exc


def _get_source_or_404(source_id: str, store: Any) -> DataSource:
    try:
        return cast(DataSource, store.get_data_source(source_id))
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


def _get_sync_job_or_404(sync_job_id: str, store: Any) -> SyncJob:
    try:
        return cast(SyncJob, store.get_sync_job(sync_job_id))
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


def _get_active_credential_or_409(source_id: str, store: Any) -> ConnectorCredential:
    try:
        return cast(ConnectorCredential, store.get_active_credential_for_source(source_id))
    except NotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "code": ConnectorErrorCode.NO_CREDENTIAL,
                "userMessage": "No active credential exists for this source. Reconnect the source.",
                "nextAction": ConnectorNextAction.RECONNECT,
            },
        ) from exc


def _ensure_intent_active(intent: ConnectionIntent, store: Any) -> None:
    now = utc_now()
    expires_at = _ensure_aware(intent.expires_at)
    if intent.status == ConnectionIntentStatus.COMPLETED:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Connection intent is already completed.")
    if intent.status == ConnectionIntentStatus.FAILED:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Connection intent already failed.")
    if expires_at <= now:
        expired = intent.model_copy(update={"status": ConnectionIntentStatus.EXPIRED}, deep=True)
        store.update_connection_intent(expired)
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="Connection intent expired.")


def _ensure_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _oauth_callback_url(provider: DataSourceType, settings: Settings) -> str:
    if provider == DataSourceType.NOTION:
        return settings.connector_oauth_callback_url
    return settings.connector_oauth_callback_url.replace("/notion/", f"/{provider}/")
