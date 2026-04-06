from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from cornerstone.clock import utcnow
from cornerstone.config import Settings
from cornerstone.connectors.filesystem import FilesystemConnector, ParsedArtifact
from cornerstone.domain.enums import (
    FreshnessState,
    SourceConnectionState,
    SupportItemKind,
    VisibilityClass,
)
from cornerstone.domain.models import Artifact, SourceConnection, SupportItem
from cornerstone.domain.schemas import SyncRunResult
from cornerstone.services.normalization import stable_id
from cornerstone.services.policies import stamp_sync_freshness


def run_sync(
    session: Session,
    connection: SourceConnection,
    settings: Settings | None = None,
) -> SyncRunResult:
    resolved_settings = settings or Settings()
    now = utcnow(resolved_settings)
    connection.last_attempted_sync_at = now

    if connection.source_connection_state in {
        SourceConnectionState.PAUSED,
        SourceConnectionState.REMOVED,
    }:
        return SyncRunResult(
            source_connection_id=connection.id,
            artifact_count=0,
            support_item_count=0,
            source_connection_state=connection.source_connection_state,
            freshness_state=connection.freshness_state,
        )

    connector = FilesystemConnector(connection.source_boundary_locator)
    if not connector.root.exists():
        connection.source_connection_state = SourceConnectionState.DEGRADED
        connection.freshness_state = FreshnessState.UNKNOWN
        connection.last_error = f"Source root does not exist: {connector.root}"
        session.add(connection)
        session.flush()
        return SyncRunResult(
            source_connection_id=connection.id,
            artifact_count=0,
            support_item_count=0,
            source_connection_state=connection.source_connection_state,
            freshness_state=connection.freshness_state,
        )

    connection.source_connection_state = SourceConnectionState.SYNCING
    session.add(connection)
    session.flush()

    parsed_artifacts = connector.list_artifacts()
    artifact_count = 0
    support_item_count = 0
    seen_external_ids: set[str] = set()
    artifact_freshness_states: list[FreshnessState] = []

    for parsed_artifact in parsed_artifacts:
        artifact = upsert_artifact(session, connection, parsed_artifact, resolved_settings, now=now)
        artifact_count += 1
        support_item_count += len(artifact.support_items)
        seen_external_ids.add(parsed_artifact.external_id)
        artifact_freshness_states.append(artifact.freshness_state)

    for artifact in session.scalars(
        select(Artifact).where(Artifact.source_connection_id == connection.id)
    ).all():
        if artifact.external_id in seen_external_ids:
            continue
        artifact.freshness_state = FreshnessState.MONITORING
        for support_item in artifact.support_items:
            support_item.freshness_state = FreshnessState.MONITORING

    connection.last_successful_sync_at = now
    connection.last_error = None
    connection.source_connection_state = SourceConnectionState.ACTIVE
    connection.freshness_state = _aggregate_source_freshness(artifact_freshness_states)
    session.add(connection)
    session.flush()

    return SyncRunResult(
        source_connection_id=connection.id,
        artifact_count=artifact_count,
        support_item_count=support_item_count,
        source_connection_state=connection.source_connection_state,
        freshness_state=connection.freshness_state,
    )


def upsert_artifact(
    session: Session,
    connection: SourceConnection,
    parsed_artifact: ParsedArtifact,
    settings: Settings,
    *,
    now,
) -> Artifact:
    artifact = session.scalar(
        select(Artifact).where(
            Artifact.source_connection_id == connection.id,
            Artifact.external_id == parsed_artifact.external_id,
        )
    )
    frontmatter = parsed_artifact.metadata.get("frontmatter", {})
    visibility_class = VisibilityClass(
        frontmatter.get("visibility_class", connection.visibility_class.value)
    )
    freshness_state = stamp_sync_freshness(
        parsed_artifact.source_updated_at,
        now,
        stale_after_hours=settings.source_stale_after_hours,
        drift_after_hours=settings.source_drift_after_hours,
    )
    if artifact is None:
        artifact = Artifact(
            id=stable_id("art", connection.id, parsed_artifact.external_id),
            context_space_id=connection.context_space_id,
            source_connection_id=connection.id,
            external_id=parsed_artifact.external_id,
            artifact_type=parsed_artifact.artifact_type,
            title=parsed_artifact.title,
            source_locator=parsed_artifact.source_locator,
            source_updated_at=parsed_artifact.source_updated_at,
            last_refreshed_at=now,
            content_hash=parsed_artifact.content_hash,
            content_text=parsed_artifact.content_text,
            freshness_state=freshness_state,
            visibility_class=visibility_class,
            metadata_json=parsed_artifact.metadata,
        )
        session.add(artifact)
        session.flush()
    else:
        artifact.artifact_type = parsed_artifact.artifact_type
        artifact.title = parsed_artifact.title
        artifact.source_locator = parsed_artifact.source_locator
        artifact.source_updated_at = parsed_artifact.source_updated_at
        artifact.last_refreshed_at = now
        artifact.content_text = parsed_artifact.content_text
        artifact.freshness_state = freshness_state
        artifact.visibility_class = visibility_class
        artifact.metadata_json = parsed_artifact.metadata

    if artifact.content_hash != parsed_artifact.content_hash or not artifact.support_items:
        artifact.content_hash = parsed_artifact.content_hash
        artifact.support_items.clear()
        session.flush()
        for selector, excerpt in parsed_artifact.support_fragments:
            artifact.support_items.append(
                SupportItem(
                    id=stable_id("supp", connection.id, parsed_artifact.external_id, selector),
                    context_space_id=connection.context_space_id,
                    support_item_kind=SupportItemKind.EVIDENCE_FRAGMENT,
                    visibility_class=visibility_class,
                    source_label=connection.source_label,
                    excerpt_or_summary=excerpt,
                    source_locator=parsed_artifact.source_locator,
                    freshness_state=freshness_state,
                    selector=selector,
                    normalized_claim=excerpt,
                )
            )
    else:
        for support_item in artifact.support_items:
            support_item.visibility_class = visibility_class
            support_item.source_label = connection.source_label
            support_item.source_locator = parsed_artifact.source_locator
            support_item.freshness_state = freshness_state
    session.flush()
    return artifact


def _aggregate_source_freshness(states: list[FreshnessState]) -> FreshnessState:
    if not states:
        return FreshnessState.UNKNOWN
    if FreshnessState.DRIFT_DETECTED in states:
        return FreshnessState.DRIFT_DETECTED
    if FreshnessState.STALE in states:
        return FreshnessState.STALE
    if FreshnessState.MONITORING in states:
        return FreshnessState.MONITORING
    return FreshnessState.CURRENT
