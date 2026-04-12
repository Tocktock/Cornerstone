from __future__ import annotations

from sqlalchemy.orm import Session

from cornerstone.config import Settings
from cornerstone.domain.enums import (
    FreshnessState,
    RuntimeMode,
    SourceConnectionState,
    SyncMode,
    VisibilityClass,
    WorkspaceDataState,
)
from cornerstone.domain.models import Artifact, SourceConnection
from cornerstone.services.bootstrap import derive_workspace_data_state, ensure_minimal_bootstrap
from cornerstone.services.normalization import stable_id


def test_workspace_data_state_is_awaiting_sources_for_production_bootstrap(
    db_session: Session, test_database_url: str
):
    settings = Settings(database_url=test_database_url, runtime_mode=RuntimeMode.PRODUCTION)
    bootstrap = ensure_minimal_bootstrap(db_session, settings)

    state, linked_source_count, active_source_count, degraded_source_count = (
        derive_workspace_data_state(db_session, settings, bootstrap["workspace"])
    )

    assert state is WorkspaceDataState.AWAITING_SOURCES
    assert linked_source_count == 0
    assert active_source_count == 0
    assert degraded_source_count == 0


def test_workspace_data_state_is_syncing_when_sources_are_linked_without_artifacts(
    db_session: Session, test_database_url: str
):
    settings = Settings(database_url=test_database_url, runtime_mode=RuntimeMode.PRODUCTION)
    bootstrap = ensure_minimal_bootstrap(db_session, settings)
    workspace = bootstrap["workspace"]
    db_session.add(_source_connection(workspace.id, "syncing"))
    db_session.flush()

    state, linked_source_count, active_source_count, degraded_source_count = (
        derive_workspace_data_state(db_session, settings, workspace)
    )

    assert state is WorkspaceDataState.SYNCING_SOURCES
    assert linked_source_count == 1
    assert active_source_count == 1
    assert degraded_source_count == 0


def test_workspace_data_state_is_ready_when_artifacts_exist(
    db_session: Session, test_database_url: str
):
    settings = Settings(database_url=test_database_url, runtime_mode=RuntimeMode.PRODUCTION)
    bootstrap = ensure_minimal_bootstrap(db_session, settings)
    workspace = bootstrap["workspace"]
    connection = _source_connection(workspace.id, "ready")
    db_session.add(connection)
    db_session.flush()
    db_session.add(_artifact(workspace.id, connection.id, "ready"))
    db_session.flush()

    state, linked_source_count, active_source_count, degraded_source_count = (
        derive_workspace_data_state(db_session, settings, workspace)
    )

    assert state is WorkspaceDataState.READY
    assert linked_source_count == 1
    assert active_source_count == 1
    assert degraded_source_count == 0


def test_workspace_data_state_is_degraded_when_source_health_requires_recovery(
    db_session: Session, test_database_url: str
):
    settings = Settings(database_url=test_database_url, runtime_mode=RuntimeMode.PRODUCTION)
    bootstrap = ensure_minimal_bootstrap(db_session, settings)
    workspace = bootstrap["workspace"]
    db_session.add(
        _source_connection(
            workspace.id,
            "degraded",
            state=SourceConnectionState.DEGRADED,
            freshness=FreshnessState.STALE,
        )
    )
    db_session.flush()

    state, linked_source_count, active_source_count, degraded_source_count = (
        derive_workspace_data_state(db_session, settings, workspace)
    )

    assert state is WorkspaceDataState.DEGRADED
    assert linked_source_count == 1
    assert active_source_count == 0
    assert degraded_source_count == 1
def _source_connection(
    workspace_id: str,
    suffix: str,
    *,
    state: SourceConnectionState = SourceConnectionState.ACTIVE,
    freshness: FreshnessState = FreshnessState.CURRENT,
) -> SourceConnection:
    return SourceConnection(
        id=stable_id("source", workspace_id, suffix),
        context_space_id=workspace_id,
        provider="notion",
        source_label=f"Source {suffix}",
        source_boundary_locator=f"notion://page-tree/{suffix}",
        template_key="notion_shared_page_tree",
        selected_scope_json={
            "scope_kind": "page_tree",
            "object_id": suffix,
            "title": f"Source {suffix}",
            "url": f"https://www.notion.so/{suffix}",
            "input": suffix,
        },
        sync_checkpoint_json={},
        visibility_class=VisibilityClass.MEMBER_VISIBLE,
        sync_mode=SyncMode.SCHEDULED_SYNC,
        sync_interval_seconds=900,
        source_connection_state=state,
        freshness_state=freshness,
        effective_sync_policy={},
    )


def _artifact(workspace_id: str, connection_id: str, suffix: str) -> Artifact:
    return Artifact(
        id=stable_id("artifact", workspace_id, suffix),
        context_space_id=workspace_id,
        source_connection_id=connection_id,
        external_id=f"artifact-{suffix}",
        artifact_type="notion_page",
        title=f"Artifact {suffix}",
        source_locator=f"https://www.notion.so/{suffix}",
        content_hash=f"hash-{suffix}",
        content_text=f"Artifact content {suffix}",
        freshness_state=FreshnessState.CURRENT,
        visibility_class=VisibilityClass.MEMBER_VISIBLE,
        metadata_json={},
    )
