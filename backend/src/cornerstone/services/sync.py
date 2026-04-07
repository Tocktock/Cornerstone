from __future__ import annotations

from datetime import timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from cornerstone.clock import utcnow
from cornerstone.config import Settings
from cornerstone.connectors.base import ParsedArtifact
from cornerstone.connectors.registry import get_connector_adapter
from cornerstone.domain.enums import (
    FreshnessState,
    SourceConnectionState,
    SupportItemKind,
    SyncRunStatus,
    SyncTriggerKind,
    VisibilityClass,
)
from cornerstone.domain.models import Artifact, SourceConnection, SupportItem, SyncRun
from cornerstone.domain.schemas import SyncRunResult
from cornerstone.services.normalization import stable_id
from cornerstone.services.policies import stamp_sync_freshness


def run_sync(
    session: Session,
    connection: SourceConnection,
    settings: Settings | None = None,
    *,
    trigger_kind: str | SyncTriggerKind = SyncTriggerKind.MANUAL,
) -> SyncRunResult:
    resolved_settings = settings or Settings()
    now = utcnow(resolved_settings)
    trigger = (
        trigger_kind if isinstance(trigger_kind, SyncTriggerKind) else SyncTriggerKind(trigger_kind)
    )
    connection.last_attempted_sync_at = now
    sync_run = SyncRun(
        id=stable_id("run", connection.id, trigger.value, now.isoformat()),
        context_space_id=connection.context_space_id,
        source_connection_id=connection.id,
        trigger_kind=trigger,
        run_status=SyncRunStatus.RUNNING,
        started_at=now,
    )
    session.add(sync_run)
    session.flush()

    if connection.source_connection_state in {
        SourceConnectionState.PAUSED,
        SourceConnectionState.REMOVED,
    }:
        sync_run.run_status = SyncRunStatus.SKIPPED
        sync_run.finished_at = now
        sync_run.error_summary = (
            f"Sync skipped because source is {connection.source_connection_state.value}."
        )
        session.add(sync_run)
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

    try:
        connector = get_connector_adapter(connection.provider, connection.source_boundary_locator)
        provider_result = connector.sync(
            connection=connection,
            settings=resolved_settings,
            provider_credential=connection.provider_credential,
        )
        parsed_artifacts = provider_result.parsed_artifacts
        artifact_count = 0
        support_item_count = 0
        seen_external_ids: set[str] = set()
        artifact_freshness_states: list[FreshnessState] = []

        for parsed_artifact in parsed_artifacts:
            artifact = upsert_artifact(
                session, connection, parsed_artifact, resolved_settings, now=now
            )
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
            artifact_freshness_states.append(FreshnessState.MONITORING)

        connection.sync_checkpoint_json = provider_result.sync_checkpoint or {}
        connection.effective_sync_policy = {
            **(connection.effective_sync_policy or {}),
            **(provider_result.effective_sync_policy or {}),
        }
        connection.last_successful_sync_at = now
        connection.last_error = None
        connection.source_connection_state = SourceConnectionState.ACTIVE
        connection.freshness_state = _aggregate_source_freshness(artifact_freshness_states)
        if (
            connection.sync_mode is not None
            and connection.source_connection_state is SourceConnectionState.ACTIVE
        ):
            connection.next_scheduled_sync_at = now.replace(microsecond=0) + _sync_interval_delta(
                connection.sync_interval_seconds
            )

        sync_run.run_status = SyncRunStatus.SUCCEEDED
        sync_run.finished_at = now
        sync_run.artifact_count = artifact_count
        sync_run.support_item_count = support_item_count
        sync_run.error_summary = None
    except Exception as exc:
        connection.source_connection_state = SourceConnectionState.DEGRADED
        connection.freshness_state = FreshnessState.UNKNOWN
        connection.last_error = str(exc)
        connection.next_scheduled_sync_at = now.replace(microsecond=0) + _sync_interval_delta(
            connection.sync_interval_seconds
        )
        sync_run.run_status = SyncRunStatus.FAILED
        sync_run.finished_at = now
        sync_run.error_summary = str(exc)
        artifact_count = 0
        support_item_count = 0

    session.add(sync_run)
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


def _sync_interval_delta(sync_interval_seconds: int) -> timedelta:
    return timedelta(seconds=sync_interval_seconds)
