from __future__ import annotations

import secrets
from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from cornerstone.clock import utcnow
from cornerstone.config import Settings
from cornerstone.connectors.registry import get_connector_adapter, get_connector_template
from cornerstone.domain.enums import (
    BaseRole,
    Capability,
    ConnectorAction,
    SourceConnectionState,
)
from cornerstone.domain.models import (
    Actor,
    ConnectorScopeGrant,
    ProviderCredential,
    SourceConnection,
    SyncRun,
)
from cornerstone.domain.schemas import (
    ConnectorTemplateSummary,
    ProviderBindingStartResponse,
    ProviderBindingSummary,
    SourceConnectionCreate,
    SourceConnectionDetail,
    SourceConnectionPreviewRequest,
    SourceConnectionPreviewResponse,
    SourceConnectionStatus,
    SourceConnectionUpdate,
    SourcePreviewItem,
    SyncRunSummary,
)
from cornerstone.services.normalization import stable_id
from cornerstone.services.sync import run_sync


def actor_has_workspace_capability(actor: Actor, capability: Capability) -> bool:
    return any(
        scoped.get("capability") == capability.value and scoped.get("scope") == "workspace"
        for scoped in actor.scoped_capabilities
    )


def actor_can_manage_connectors(actor: Actor) -> bool:
    return actor.base_role in {BaseRole.OWNER, BaseRole.ADMIN} or actor_has_workspace_capability(
        actor, Capability.MANAGE_CONNECTORS
    )


def actor_can_connector_action(
    session: Session, actor: Actor, action: ConnectorAction, *, context_space_id: str
) -> bool:
    if actor.context_space_id != context_space_id:
        return False
    if actor_can_manage_connectors(actor):
        return True
    if action in {
        ConnectorAction.SYNC,
        ConnectorAction.PAUSE,
        ConnectorAction.RESUME,
    }:
        grants = session.scalars(
            select(ConnectorScopeGrant).where(
                ConnectorScopeGrant.actor_id == actor.id,
                ConnectorScopeGrant.context_space_id == context_space_id,
            )
        ).all()
        return any(action.value in grant.allowed_connector_actions for grant in grants)
    return False


def list_connector_templates() -> list[ConnectorTemplateSummary]:
    from cornerstone.connectors.registry import list_manager_templates

    return [
        ConnectorTemplateSummary(
            template_key=template.template_key,
            provider=template.provider,
            label=template.label,
            description=template.description,
            scope_kind=template.scope_kind,
            default_visibility_class=template.default_visibility_class,
            recommended_sync_interval_seconds=template.recommended_sync_interval_seconds,
            preview_required=template.preview_required,
        )
        for template in list_manager_templates()
    ]


def resolve_provider_credential(
    session: Session,
    *,
    actor: Actor,
    provider_credential_ref: str | None,
    require_manager: bool = False,
) -> ProviderCredential | None:
    if provider_credential_ref is None:
        return None
    credential = session.scalar(
        select(ProviderCredential).where(
            ProviderCredential.credential_reference == provider_credential_ref
        )
    )
    if credential is None:
        raise ValueError("Unknown provider credential reference.")
    if credential.context_space_id != actor.context_space_id:
        raise ValueError("Provider credential is outside the actor workspace.")
    if credential.revoked_at is not None:
        raise ValueError("Provider credential has been revoked.")
    if require_manager and not actor_can_manage_connectors(actor):
        raise PermissionError("Actor lacks connector manager privileges.")
    return credential


def start_provider_binding(
    session: Session,
    *,
    actor: Actor,
    settings: Settings,
    provider: str,
) -> ProviderBindingStartResponse:
    if not actor_can_manage_connectors(actor):
        raise PermissionError("Actor lacks connector manager privileges.")
    if provider != "notion":
        raise ValueError("Unsupported provider binding.")

    adapter = get_connector_adapter("notion")
    now = utcnow(settings)
    credential_reference = stable_id(
        "credref",
        actor.context_space_id,
        provider,
        secrets.token_hex(6),
    )

    if settings.notion_demo_binding_enabled:
        credential = ProviderCredential(
            id=stable_id("cred", credential_reference),
            context_space_id=actor.context_space_id,
            provider=provider,
            credential_reference=credential_reference,
            account_label="Demo Notion workspace",
            auth_payload=adapter.build_demo_binding_payload(settings),
            created_by_actor_id=actor.id,
            last_validated_at=now,
        )
        session.add(credential)
        session.flush()
        return ProviderBindingStartResponse(
            provider=provider,
            provider_credential_ref=credential_reference,
            account_label=credential.account_label,
            demo_mode=True,
        )

    if not settings.notion_client_id or not settings.notion_client_secret:
        raise ValueError(
            "Notion OAuth is not configured for production runtime mode. "
            "Set CORNERSTONE_NOTION_CLIENT_ID and CORNERSTONE_NOTION_CLIENT_SECRET."
        )

    binding_state = secrets.token_urlsafe(24)
    credential = ProviderCredential(
        id=stable_id("cred", credential_reference),
        context_space_id=actor.context_space_id,
        provider=provider,
        credential_reference=credential_reference,
        binding_state=binding_state,
        account_label=None,
        auth_payload={"mode": "pending_oauth"},
        created_by_actor_id=actor.id,
    )
    session.add(credential)
    session.flush()
    return ProviderBindingStartResponse(
        provider=provider,
        provider_credential_ref=credential_reference,
        authorization_url=adapter.build_authorization_url(
            binding_state=binding_state, settings=settings
        ),
        binding_state=binding_state,
        demo_mode=False,
    )


def complete_provider_binding(
    session: Session,
    *,
    actor: Actor,
    settings: Settings,
    provider: str,
    binding_state: str,
    code: str,
) -> ProviderBindingSummary:
    if not actor_can_manage_connectors(actor):
        raise PermissionError("Actor lacks connector manager privileges.")
    credential = session.scalar(
        select(ProviderCredential).where(
            ProviderCredential.binding_state == binding_state,
            ProviderCredential.provider == provider,
            ProviderCredential.context_space_id == actor.context_space_id,
        )
    )
    if credential is None:
        raise ValueError("Unknown provider binding state.")
    adapter = get_connector_adapter(provider)
    auth_payload, account_label = adapter.exchange_code_for_token(code=code, settings=settings)
    credential.auth_payload = auth_payload
    credential.account_label = account_label or "Notion workspace"
    credential.binding_state = None
    credential.last_validated_at = utcnow(settings)
    session.add(credential)
    session.flush()
    return ProviderBindingSummary(
        provider=provider,
        provider_credential_ref=credential.credential_reference,
        account_label=credential.account_label,
    )


def preview_source_connection(
    session: Session,
    *,
    actor: Actor,
    settings: Settings,
    payload: SourceConnectionPreviewRequest,
) -> SourceConnectionPreviewResponse:
    if not actor_can_manage_connectors(actor):
        raise PermissionError("Actor lacks connector manager privileges.")
    template = get_connector_template(payload.template_key)
    provider_credential = resolve_provider_credential(
        session,
        actor=actor,
        provider_credential_ref=payload.provider_credential_ref,
    )
    adapter = get_connector_adapter(template.provider, payload.selected_scope_input)
    prepared = adapter.prepare_connection(
        template_key=template.template_key,
        source_label=payload.source_label,
        selected_scope_input=payload.selected_scope_input,
        visibility_class=payload.visibility_class,
        settings=settings,
        provider_credential=provider_credential,
    )
    return SourceConnectionPreviewResponse(
        provider=prepared.provider,
        template_key=prepared.template_key,
        resolved_source_boundary_locator=prepared.source_boundary_locator,
        selected_scope_json=prepared.selected_scope,
        suggested_sync_mode=prepared.sync_mode,
        suggested_sync_interval_seconds=prepared.sync_interval_seconds,
        preview_items=[
            SourcePreviewItem(
                upstream_id=item.upstream_id,
                title=item.title,
                artifact_type=item.artifact_type,
                source_locator=item.source_locator,
                excerpt=item.excerpt,
                source_updated_at=item.source_updated_at,
            )
            for item in prepared.preview_items
        ],
        visibility_class=prepared.visibility_class,
        effective_sync_policy=prepared.effective_sync_policy,
    )


def create_source_connection(
    session: Session,
    *,
    actor: Actor,
    settings: Settings,
    payload: SourceConnectionCreate,
) -> SourceConnection:
    preview = preview_source_connection(
        session,
        actor=actor,
        settings=settings,
        payload=SourceConnectionPreviewRequest(
            template_key=payload.template_key,
            provider_credential_ref=payload.provider_credential_ref,
            source_label=payload.source_label,
            selected_scope_input=payload.selected_scope_input,
            visibility_class=payload.visibility_class,
        ),
    )
    existing = session.scalar(
        select(SourceConnection).where(
            SourceConnection.context_space_id == actor.context_space_id,
            SourceConnection.provider == preview.provider,
            SourceConnection.source_boundary_locator == preview.resolved_source_boundary_locator,
            SourceConnection.removed_at.is_(None),
        )
    )
    if existing is not None:
        raise ValueError("A source connection for that boundary already exists in this workspace.")
    connection = SourceConnection(
        id=stable_id(
            "source",
            actor.context_space_id,
            preview.provider,
            preview.resolved_source_boundary_locator,
        ),
        context_space_id=actor.context_space_id,
        provider=preview.provider,
        source_label=payload.source_label,
        source_boundary_locator=preview.resolved_source_boundary_locator,
        template_key=preview.template_key,
        provider_credential_ref=payload.provider_credential_ref,
        selected_scope_json=preview.selected_scope_json,
        sync_checkpoint_json={},
        visibility_class=payload.visibility_class,
        sync_mode=preview.suggested_sync_mode,
        sync_interval_seconds=payload.sync_interval_seconds
        or preview.suggested_sync_interval_seconds,
        effective_sync_policy=preview.effective_sync_policy,
        next_scheduled_sync_at=utcnow(settings)
        + timedelta(
            seconds=payload.sync_interval_seconds
            or preview.suggested_sync_interval_seconds
        ),
    )
    session.add(connection)
    session.flush()
    run_sync(session, connection, settings, trigger_kind="initial")
    session.flush()
    return connection


def update_source_connection(
    session: Session,
    *,
    actor: Actor,
    settings: Settings,
    connection: SourceConnection,
    payload: SourceConnectionUpdate,
) -> SourceConnection:
    if not actor_can_manage_connectors(actor):
        raise PermissionError("Actor lacks connector manager privileges.")
    if connection.context_space_id != actor.context_space_id:
        raise PermissionError("Source connection is outside the actor workspace.")

    if payload.source_label is not None:
        connection.source_label = payload.source_label
    if payload.visibility_class is not None:
        connection.visibility_class = payload.visibility_class
    if payload.sync_interval_seconds is not None:
        connection.sync_interval_seconds = payload.sync_interval_seconds

    should_resync = False
    provider_credential_ref = connection.provider_credential_ref
    if payload.provider_credential_ref is not None:
        provider_credential = resolve_provider_credential(
            session,
            actor=actor,
            provider_credential_ref=payload.provider_credential_ref,
            require_manager=True,
        )
        provider_credential_ref = (
            provider_credential.credential_reference if provider_credential is not None else None
        )
        connection.provider_credential_ref = provider_credential_ref
        should_resync = True

    if payload.selected_scope_input is not None:
        preview = preview_source_connection(
            session,
            actor=actor,
            settings=settings,
            payload=SourceConnectionPreviewRequest(
                template_key=connection.template_key,
                provider_credential_ref=provider_credential_ref,
                source_label=connection.source_label,
                selected_scope_input=payload.selected_scope_input,
                visibility_class=connection.visibility_class,
            ),
        )
        connection.source_boundary_locator = preview.resolved_source_boundary_locator
        connection.selected_scope_json = preview.selected_scope_json
        connection.effective_sync_policy = preview.effective_sync_policy
        connection.sync_checkpoint_json = {}
        should_resync = True

    if connection.source_connection_state.value != "paused":
        connection.next_scheduled_sync_at = utcnow(settings) + timedelta(
            seconds=connection.sync_interval_seconds
        )

    session.add(connection)
    session.flush()
    if should_resync:
        run_sync(session, connection, settings, trigger_kind="recovery")
        session.flush()
    return connection


def pause_source_connection(
    session: Session, *, actor: Actor, settings: Settings, connection: SourceConnection
) -> SourceConnection:
    if not actor_can_connector_action(
        session, actor, ConnectorAction.PAUSE, context_space_id=connection.context_space_id
    ):
        raise PermissionError("Actor lacks connector pause privileges.")
    connection.source_connection_state = SourceConnectionState.PAUSED
    connection.next_scheduled_sync_at = None
    connection.last_error = "Paused by connector manager."
    session.add(connection)
    session.flush()
    return connection


def resume_source_connection(
    session: Session, *, actor: Actor, settings: Settings, connection: SourceConnection
) -> SourceConnection:
    if not actor_can_connector_action(
        session, actor, ConnectorAction.RESUME, context_space_id=connection.context_space_id
    ):
        raise PermissionError("Actor lacks connector resume privileges.")
    connection.source_connection_state = SourceConnectionState.ACTIVE
    connection.next_scheduled_sync_at = utcnow(settings) + timedelta(
        seconds=connection.sync_interval_seconds
    )
    connection.last_error = None
    session.add(connection)
    session.flush()
    return connection


def remove_source_connection(
    session: Session, *, actor: Actor, settings: Settings, connection: SourceConnection
) -> SourceConnection:
    if not actor_can_manage_connectors(actor):
        raise PermissionError("Actor lacks connector manager privileges.")
    connection.source_connection_state = SourceConnectionState.REMOVED
    connection.removed_at = utcnow(settings)
    connection.next_scheduled_sync_at = None
    connection.last_error = "Removed from workspace scope."
    session.add(connection)
    session.flush()
    return connection


def latest_sync_run(session: Session, connection_id: str) -> SyncRun | None:
    return session.scalar(
        select(SyncRun)
        .where(SyncRun.source_connection_id == connection_id)
        .order_by(SyncRun.started_at.desc())
        .limit(1)
    )


def list_sync_runs(session: Session, connection_id: str) -> list[SyncRunSummary]:
    runs = session.scalars(
        select(SyncRun)
        .where(SyncRun.source_connection_id == connection_id)
        .order_by(SyncRun.started_at.desc())
    ).all()
    return [
        SyncRunSummary(
            id=run.id,
            source_connection_id=run.source_connection_id,
            trigger_kind=run.trigger_kind,
            run_status=run.run_status,
            started_at=run.started_at,
            finished_at=run.finished_at,
            artifact_count=run.artifact_count,
            support_item_count=run.support_item_count,
            error_summary=run.error_summary,
        )
        for run in runs
    ]


def source_connection_status_for_actor(
    session: Session, *, actor: Actor, connection: SourceConnection
) -> SourceConnectionStatus:
    last_run = latest_sync_run(session, connection.id)
    return SourceConnectionStatus(
        id=connection.id,
        context_space_id=connection.context_space_id,
        provider=connection.provider,
        source_label=connection.source_label,
        source_boundary_locator=connection.source_boundary_locator,
        template_key=connection.template_key,
        visibility_class=connection.visibility_class,
        sync_mode=connection.sync_mode,
        sync_interval_seconds=connection.sync_interval_seconds,
        source_connection_state=connection.source_connection_state,
        freshness_state=connection.freshness_state,
        last_attempted_sync_at=connection.last_attempted_sync_at,
        last_successful_sync_at=connection.last_successful_sync_at,
        last_error=connection.last_error,
        effective_sync_policy=connection.effective_sync_policy,
        next_scheduled_sync_at=connection.next_scheduled_sync_at,
        last_run_at=last_run.started_at if last_run else None,
        last_run_status=last_run.run_status if last_run else None,
        can_manage=actor_can_manage_connectors(actor)
        or actor_can_connector_action(
            session, actor, ConnectorAction.SYNC, context_space_id=connection.context_space_id
        ),
        removed_at=connection.removed_at,
    )


def source_connection_detail_for_actor(
    session: Session, *, actor: Actor, connection: SourceConnection
) -> SourceConnectionDetail:
    status = source_connection_status_for_actor(session, actor=actor, connection=connection)
    return SourceConnectionDetail(
        **status.model_dump(),
        provider_credential_ref=connection.provider_credential_ref
        if actor_can_manage_connectors(actor)
        else None,
        selected_scope_json=connection.selected_scope_json if status.can_manage else {},
        sync_checkpoint_json=connection.sync_checkpoint_json if status.can_manage else {},
    )
