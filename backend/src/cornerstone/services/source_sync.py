from __future__ import annotations

from collections import Counter
from typing import Any

from cornerstone.config import Settings
from cornerstone.observability import log_event
from cornerstone.schemas import (
    Artifact,
    DataSource,
    DataSourceNextAction,
    DataSourceStatus,
    DataSourceSyncStatus,
    ErrorInfo,
    ExtractionStatus,
    FreshnessState,
    SourceObject,
    SyncSourceResponse,
    utc_now,
)
from cornerstone.services.extraction import content_hash, extract_evidence_fragments
from cornerstone.services.freshness import FreshnessPolicy

_SYNCABLE_STATUSES = {
    DataSourceStatus.CONNECTED,
    DataSourceStatus.SYNC_PENDING,
    DataSourceStatus.DEGRADED,
    DataSourceStatus.STALE,
    DataSourceStatus.SYNCING,
}


class SourceNotSyncableError(RuntimeError):
    """Raised when the current source state does not allow content sync."""


def sync_source_objects(
    *,
    data_source: DataSource,
    objects: list[SourceObject],
    store: Any,
    settings: Settings,
    emit_logs: bool = True,
) -> SyncSourceResponse:
    """Persist normalized connector objects as Artifacts and EvidenceFragments.

    This service is intentionally provider-agnostic. Connectors only produce normalized
    SourceObject instances; provenance, idempotency, freshness, extraction, and source counters
    remain centralized here.
    """

    if data_source.status not in _SYNCABLE_STATUSES:
        raise SourceNotSyncableError(
            f"DataSource cannot sync while status is '{data_source.status}'."
        )

    policy = FreshnessPolicy(
        fresh_days=settings.freshness_fresh_days,
        stale_days=settings.freshness_stale_days,
    )
    sync_started_at = utc_now()
    syncing_source = data_source.model_copy(
        update={
            "status": DataSourceStatus.SYNCING,
            "sync_status": DataSourceSyncStatus.SYNCING,
            "next_action": DataSourceNextAction.NONE,
            "last_sync_at": sync_started_at,
        },
        deep=True,
    )
    store.update_data_source(syncing_source)
    if emit_logs:
        log_event(
            "source.sync_started",
            sourceId=data_source.id,
            sourceType=data_source.type,
            objectCount=len(objects),
            startedAt=sync_started_at,
        )

    artifacts: list[Artifact] = []
    evidence_fragments = []
    artifact_created_count = 0
    artifact_reused_count = 0
    artifact_changed_count = 0
    created_artifact_ids: list[str] = []
    reused_artifact_ids: list[str] = []
    changed_artifact_ids: list[str] = []
    try:
        with store.transaction():
            for source_object in objects:
                raw_hash = content_hash(source_object.content)
                previous_artifacts_for_object = [
                    artifact
                    for artifact in store.list_artifacts(datasource_id=data_source.id)
                    if artifact.source_external_id == source_object.source_external_id
                ]
                existing_artifact = store.find_artifact_by_source_identity(
                    datasource_id=data_source.id,
                    source_external_id=source_object.source_external_id,
                    raw_content_hash=raw_hash,
                )
                if existing_artifact is not None:
                    artifact_reused_count += 1
                    artifacts.append(existing_artifact)
                    reused_artifact_ids.append(existing_artifact.id)
                    existing_evidence = store.list_evidence_fragments(artifact_id=existing_artifact.id)
                    evidence_fragments.extend(existing_evidence)
                    if emit_logs:
                        log_event(
                            "artifact.reused",
                            sourceId=data_source.id,
                            artifactId=existing_artifact.id,
                            sourceExternalId=existing_artifact.source_external_id,
                            evidenceFragmentCount=len(existing_evidence),
                        )
                    continue

                freshness_reference = source_object.source_updated_at or sync_started_at
                freshness_state = policy.classify(freshness_reference, now=sync_started_at)
                artifact = Artifact(
                    datasource_id=data_source.id,
                    source_type=data_source.type,
                    source_external_id=source_object.source_external_id,
                    source_url=None if source_object.source_url is None else str(source_object.source_url),
                    source_object_type=source_object.source_object_type,
                    title=source_object.title,
                    raw_content_hash=raw_hash,
                    captured_at=sync_started_at,
                    source_updated_at=source_object.source_updated_at,
                    freshness_state=freshness_state,
                    extraction_status=ExtractionStatus.COMPLETE,
                    provider_metadata=source_object.provider_metadata,
                )
                saved_artifact = store.add_artifact(artifact)
                artifact_created_count += 1
                created_artifact_ids.append(saved_artifact.id)
                if previous_artifacts_for_object:
                    artifact_changed_count += 1
                    changed_artifact_ids.append(saved_artifact.id)
                artifacts.append(saved_artifact)
                extracted_fragments = extract_evidence_fragments(saved_artifact, source_object.content)
                if emit_logs:
                    log_event(
                        "artifact.extracted",
                        sourceId=data_source.id,
                        artifactId=saved_artifact.id,
                        sourceExternalId=saved_artifact.source_external_id,
                        freshnessState=saved_artifact.freshness_state,
                        evidenceFragmentCount=len(extracted_fragments),
                    )
                for fragment in extracted_fragments:
                    evidence_fragments.append(store.add_evidence_fragment(fragment))

            total_artifacts = store.list_artifacts(datasource_id=data_source.id)
            content_freshness_state = _aggregate_freshness(
                [artifact.freshness_state for artifact in total_artifacts]
            )
            sync_freshness_state = policy.classify(sync_started_at, now=sync_started_at)
            final_source = syncing_source.model_copy(
                update={
                    "status": DataSourceStatus.CONNECTED,
                    "sync_status": DataSourceSyncStatus.SUCCEEDED,
                    "next_action": (
                        DataSourceNextAction.REVIEW_EVIDENCE
                        if len(evidence_fragments) > 0
                        else DataSourceNextAction.NONE
                    ),
                    "last_successful_sync_at": sync_started_at,
                    "last_error": None,
                    "freshness_state": content_freshness_state,
                    "sync_freshness_state": sync_freshness_state,
                    "content_freshness_state": content_freshness_state,
                    "artifact_count": len(total_artifacts),
                    "evidence_fragment_count": _evidence_count_for_source(store, data_source.id),
                },
                deep=True,
            )
            final_source = store.update_data_source(final_source)
    except Exception as exc:
        failed_status = (
            DataSourceStatus.DEGRADED
            if data_source.last_successful_sync_at is not None
            else DataSourceStatus.FAILED
        )
        failed_source = syncing_source.model_copy(
            update={
                "status": failed_status,
                "sync_status": (
                    DataSourceSyncStatus.DEGRADED
                    if data_source.last_successful_sync_at is not None
                    else DataSourceSyncStatus.FAILED
                ),
                "next_action": DataSourceNextAction.RETRY_SYNC,
                "freshness_state": FreshnessState.UNKNOWN,
                "sync_freshness_state": FreshnessState.UNKNOWN,
                "content_freshness_state": FreshnessState.UNKNOWN,
                "last_error": ErrorInfo(code="sync_failed", message=str(exc)),
            },
            deep=True,
        )
        store.update_data_source(failed_source)
        if emit_logs:
            log_event(
                "source.sync_failed",
                sourceId=data_source.id,
                sourceType=data_source.type,
                status=failed_status,
                errorType=type(exc).__name__,
                errorMessage=str(exc),
            )
        raise

    if emit_logs:
        log_event(
            "source.sync_succeeded",
            sourceId=final_source.id,
            sourceType=final_source.type,
            status=final_source.status,
            artifactCount=len(artifacts),
            artifactCreatedCount=artifact_created_count,
            artifactReusedCount=artifact_reused_count,
            artifactChangedCount=artifact_changed_count,
            evidenceFragmentCount=len(evidence_fragments),
            totalArtifactCount=final_source.artifact_count,
            totalEvidenceFragmentCount=final_source.evidence_fragment_count,
            freshnessState=final_source.freshness_state,
            syncFreshnessState=final_source.sync_freshness_state,
            contentFreshnessState=final_source.content_freshness_state,
        )
    return SyncSourceResponse(
        data_source=final_source,
        artifacts=artifacts,
        evidence_fragments=evidence_fragments,
        artifact_created_count=artifact_created_count,
        artifact_reused_count=artifact_reused_count,
        artifact_changed_count=artifact_changed_count,
        evidence_created_count=len(evidence_fragments),
        created_artifact_ids=created_artifact_ids,
        reused_artifact_ids=reused_artifact_ids,
        changed_artifact_ids=changed_artifact_ids,
    )


def _evidence_count_for_source(store: Any, data_source_id: str) -> int:
    artifact_ids = {artifact.id for artifact in store.list_artifacts(datasource_id=data_source_id)}
    return sum(
        1
        for fragment in store.list_evidence_fragments()
        if fragment.artifact_id in artifact_ids
    )


def _aggregate_freshness(states: list[FreshnessState]) -> FreshnessState:
    if not states:
        return FreshnessState.UNKNOWN
    counts = Counter(states)
    if counts[FreshnessState.STALE]:
        return FreshnessState.STALE
    if counts[FreshnessState.UNKNOWN]:
        return FreshnessState.UNKNOWN
    if len(counts) > 1:
        return FreshnessState.MIXED
    return states[0]
