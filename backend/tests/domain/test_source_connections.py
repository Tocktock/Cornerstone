from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from cornerstone.config import Settings
from cornerstone.domain.enums import (
    ActorKind,
    BaseRole,
    Capability,
    ConnectorAction,
    ConsumerScope,
    ContextSpaceKind,
    FreshnessState,
    SourceConnectionState,
    SyncMode,
    VisibilityClass,
)
from cornerstone.domain.models import (
    Actor,
    ConnectorScopeGrant,
    ContextSpace,
    SourceConnection,
)
from cornerstone.domain.schemas import SourceConnectionPreviewRequest
from cornerstone.services.normalization import stable_id
from cornerstone.services.source_connections import (
    actor_can_connector_action,
    actor_can_manage_connectors,
    list_connector_templates,
    preview_source_connection,
)
from cornerstone.services.sync import run_sync


def test_list_connector_templates_excludes_internal_templates():
    templates = list_connector_templates()

    template_keys = {template.template_key for template in templates}

    assert "notion_shared_page_tree" in template_keys
    assert "notion_shared_database" in template_keys
    assert "member-visible" not in template_keys


def test_connector_permission_checks_cover_manager_and_action_grants(db_session: Session):
    workspace = _workspace()
    db_session.add(workspace)
    manager = _actor(
        workspace.id,
        "manager",
        scoped_capabilities=[
            {
                "capability": Capability.MANAGE_CONNECTORS.value,
                "scope": "workspace",
            }
        ],
    )
    delegated = _actor(workspace.id, "delegated")
    admin = _actor(workspace.id, "admin", base_role=BaseRole.ADMIN)
    db_session.add_all([manager, delegated, admin])
    db_session.add(
        ConnectorScopeGrant(
            id=stable_id("connector-grant", workspace.id, delegated.id),
            context_space_id=workspace.id,
            actor_id=delegated.id,
            allowed_connector_actions=[
                ConnectorAction.SYNC.value,
                ConnectorAction.PAUSE.value,
                ConnectorAction.RESUME.value,
            ],
        )
    )
    db_session.flush()

    assert actor_can_manage_connectors(manager) is True
    assert actor_can_manage_connectors(admin) is True
    assert actor_can_manage_connectors(delegated) is False
    assert (
        actor_can_connector_action(
            db_session,
            delegated,
            ConnectorAction.SYNC,
            context_space_id=workspace.id,
        )
        is True
    )
    assert (
        actor_can_connector_action(
            db_session,
            delegated,
            ConnectorAction.REMOVE,
            context_space_id=workspace.id,
        )
        is False
    )


def test_preview_notion_templates_resolve_demo_fixture_scopes(
    db_session: Session, test_database_url: str
):
    settings = Settings(
        database_url=test_database_url,
        notion_demo_oauth_mode=True,
        notion_fixture_root=str(
            (Path(__file__).resolve().parents[2] / "fixtures" / "notion").resolve()
        ),
    )
    workspace = _workspace()
    actor = _actor(
        workspace.id,
        "manager",
        scoped_capabilities=[
            {
                "capability": Capability.MANAGE_CONNECTORS.value,
                "scope": "workspace",
            }
        ],
    )
    db_session.add_all([workspace, actor])
    db_session.flush()

    page_tree_preview = preview_source_connection(
        db_session,
        actor=actor,
        settings=settings,
        payload=SourceConnectionPreviewRequest(
            template_key="notion_shared_page_tree",
            source_label="Operations knowledge",
            selected_scope_input="11111111-1111-1111-1111-111111111111",
            visibility_class=VisibilityClass.MEMBER_VISIBLE,
        ),
    )
    database_preview = preview_source_connection(
        db_session,
        actor=actor,
        settings=settings,
        payload=SourceConnectionPreviewRequest(
            template_key="notion_shared_database",
            source_label="Incident tracker",
            selected_scope_input="22222222-2222-2222-2222-222222222222",
            visibility_class=VisibilityClass.MEMBER_VISIBLE,
        ),
    )

    assert page_tree_preview.resolved_source_boundary_locator.startswith("notion://page-tree/")
    assert database_preview.resolved_source_boundary_locator.startswith("notion://database/")
    assert len(page_tree_preview.preview_items) >= 1
    assert len(database_preview.preview_items) >= 1


def test_run_sync_marks_deleted_notion_entries_monitoring_without_degrading(
    tmp_path: Path, db_session: Session, test_database_url: str
):
    fixture_root = tmp_path / "notion"
    fixture_root.mkdir()
    scope_id = "33333333-3333-3333-3333-333333333333"
    scopes = {
        "page_trees": {},
        "databases": {
            scope_id: {
                "title": "Incident tracker",
                "url": f"https://www.notion.so/{scope_id.replace('-', '')}",
                "entries": [
                    {
                        "id": "44444444-4444-4444-4444-444444444444",
                        "title": "Incident A",
                        "url": "https://www.notion.so/incident-a",
                        "last_edited_time": "2026-04-05T00:00:00Z",
                        "properties": {"Status": "Open"},
                    },
                    {
                        "id": "55555555-5555-5555-5555-555555555555",
                        "title": "Incident B",
                        "url": "https://www.notion.so/incident-b",
                        "last_edited_time": "2026-04-05T00:10:00Z",
                        "properties": {"Status": "Closed"},
                    },
                ],
            }
        },
    }
    (fixture_root / "scopes.json").write_text(json.dumps(scopes), encoding="utf-8")

    settings = Settings(
        database_url=test_database_url,
        notion_demo_oauth_mode=True,
        notion_fixture_root=str(fixture_root),
        fixed_now="2026-04-06T09:00:00+09:00",
    )

    workspace = _workspace()
    db_session.add(workspace)
    db_session.flush()
    connection = SourceConnection(
        id=stable_id("source", workspace.id, "notion-demo"),
        context_space_id=workspace.id,
        provider="notion",
        source_label="Incident tracker",
        source_boundary_locator=f"notion://database/{scope_id}",
        template_key="notion_shared_database",
        selected_scope_json={
            "scope_kind": "database",
            "object_id": scope_id,
            "title": "Incident tracker",
            "url": f"https://www.notion.so/{scope_id.replace('-', '')}",
            "input": scope_id,
        },
        sync_checkpoint_json={},
        visibility_class=VisibilityClass.MEMBER_VISIBLE,
        sync_mode=SyncMode.SCHEDULED_SYNC,
        sync_interval_seconds=900,
        source_connection_state=SourceConnectionState.PENDING_SETUP,
        freshness_state=FreshnessState.UNKNOWN,
        effective_sync_policy={},
    )
    db_session.add(connection)
    db_session.flush()

    first_result = run_sync(db_session, connection, settings, trigger_kind="initial")
    assert first_result.artifact_count == 2

    scopes["databases"][scope_id]["entries"] = scopes["databases"][scope_id]["entries"][:1]
    (fixture_root / "scopes.json").write_text(json.dumps(scopes), encoding="utf-8")

    second_result = run_sync(db_session, connection, settings, trigger_kind="scheduled")
    db_session.flush()

    artifacts = {
        artifact.external_id: artifact
        for artifact in db_session.scalars(
            select(SourceConnection).where(SourceConnection.id == connection.id).limit(1)
        )
        .one()
        .artifacts
    }

    assert second_result.source_connection_state is SourceConnectionState.ACTIVE
    assert second_result.freshness_state is FreshnessState.MONITORING
    assert (
        artifacts["55555555-5555-5555-5555-555555555555"].freshness_state
        is FreshnessState.MONITORING
    )


def _workspace() -> ContextSpace:
    return ContextSpace(
        id=stable_id("ctx", "workspace"),
        kind=ContextSpaceKind.WORKSPACE,
        name="Workspace",
        slug="workspace",
        membership_boundary="workspace:test",
        default_visibility_class=VisibilityClass.MEMBER_VISIBLE,
        visibility_defaults={"member": VisibilityClass.MEMBER_VISIBLE.value},
        is_default=True,
    )


def _actor(
    context_space_id: str,
    slug: str,
    *,
    base_role: BaseRole = BaseRole.MEMBER,
    scoped_capabilities: list[dict[str, str]] | None = None,
) -> Actor:
    return Actor(
        id=stable_id("actor", context_space_id, slug),
        context_space_id=context_space_id,
        principal_key=f"principal:{slug}",
        actor_kind=ActorKind.HUMAN,
        display_name=slug.title(),
        base_role=base_role,
        auth_token=f"token-{slug}",
        scoped_capabilities=scoped_capabilities or [],
        preferred_consumer_scope=ConsumerScope.ADMIN,
    )
