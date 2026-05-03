from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from cornerstone.api.dependencies import get_store
from cornerstone.config import Settings, get_settings
from cornerstone.observability import log_event
from cornerstone.schemas import (
    DataSource,
    DataSourceType,
    ManualTextUploadRequest,
    OntologyReExtractionTrigger,
    RunOntologyReExtractionRunRequest,
    SourceObject,
    SyncSourceRequest,
    SyncSourceResponse,
)
from cornerstone.services.manual_uploads import (
    ManualUploadEmptyFileError,
    ManualUploadTooLargeError,
    ManualUploadUnsupportedTypeError,
    build_source_object_from_text_upload,
    build_source_object_from_upload,
)
from cornerstone.services.ontology_reextraction import OntologyReExtractionService
from cornerstone.services.source_sync import SourceNotSyncableError, sync_source_objects
from cornerstone.store import InMemoryStore, NotFoundError

router = APIRouter(prefix="/manual-sources", tags=["manual sources"])


@router.post("/{source_id}/sync", response_model=SyncSourceResponse)
def sync_manual_source(
    source_id: str,
    request: SyncSourceRequest,
    store: InMemoryStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> SyncSourceResponse:
    """Sync explicitly provided objects into a manual source only.

    Provider-backed sources such as Notion must use their connector sync-job paths so they
    cannot bypass credential, discovery, selection, and provider provenance checks.
    """

    data_source = _get_manual_source_or_404(source_id, store)
    return _sync_manual_objects(
        data_source=data_source,
        objects=request.objects,
        store=store,
        settings=settings,
        queue_ontology_reextraction=request.queue_ontology_reextraction,
        run_ontology_reextraction_inline=request.run_ontology_reextraction_inline,
        ontology_focus_concept=request.ontology_focus_concept,
        created_by="manual-sync",
        trigger=OntologyReExtractionTrigger.MANUAL_SYNC,
    )


@router.post("/{source_id}/uploads/text", response_model=SyncSourceResponse)
def upload_manual_text(
    source_id: str,
    request: ManualTextUploadRequest,
    store: InMemoryStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> SyncSourceResponse:
    """Ingest one or more pasted/manual text objects into a manual source.

    This v1.3.1 path gives API clients a manual-upload-shaped contract while still
    reusing the same Artifact/EvidenceFragment write path as connector sync.
    """

    data_source = _get_manual_source_or_404(source_id, store)
    objects: list[SourceObject] = []
    for upload in request.objects:
        try:
            objects.append(
                build_source_object_from_text_upload(
                    title=upload.title,
                    content=upload.content,
                    source_external_id=upload.source_external_id,
                    source_url=None if upload.source_url is None else str(upload.source_url),
                    source_updated_at=upload.source_updated_at,
                    provider_metadata=upload.provider_metadata,
                )
            )
        except ManualUploadEmptyFileError as exc:
            log_event(
                "manual_source.text_upload_rejected",
                sourceId=data_source.id,
                reason="empty_text_upload",
                detail=str(exc),
            )
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    log_event(
        "manual_source.text_upload_requested",
        sourceId=data_source.id,
        objectCount=len(objects),
    )
    return _sync_manual_objects(
        data_source=data_source,
        objects=objects,
        store=store,
        settings=settings,
        queue_ontology_reextraction=request.queue_ontology_reextraction,
        run_ontology_reextraction_inline=request.run_ontology_reextraction_inline,
        ontology_focus_concept=request.ontology_focus_concept,
        created_by="manual-text-upload",
        trigger=OntologyReExtractionTrigger.MANUAL_UPLOAD,
    )


@router.post("/{source_id}/uploads", response_model=SyncSourceResponse)
async def upload_manual_files(
    source_id: str,
    files: list[UploadFile] = File(..., description="One or more UTF-8 text-like files."),
    queue_ontology_reextraction: bool = Query(default=True, alias="queueOntologyReExtraction"),
    run_ontology_reextraction_inline: bool = Query(default=False, alias="runOntologyReExtractionInline"),
    ontology_focus_concept: str | None = Query(default=None, min_length=1, alias="ontologyFocusConcept"),
    store: InMemoryStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> SyncSourceResponse:
    """Ingest uploaded UTF-8 text-like files into a manual source.

    Binary files, PDFs, Office documents, and unsupported encodings are intentionally
    rejected in v1.3.1. Uploaded content becomes SourceObject records before the existing
    sync service creates Artifacts and EvidenceFragments with provenance.
    """

    data_source = _get_manual_source_or_404(source_id, store)
    if not files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one file is required.",
        )
    if len(files) > settings.manual_upload_max_file_count:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                "Too many files for one manual upload request: "
                f"{len(files)} > {settings.manual_upload_max_file_count}"
            ),
        )

    objects: list[SourceObject] = []
    for upload in files:
        data = await upload.read()
        try:
            objects.append(
                build_source_object_from_upload(
                    filename=upload.filename,
                    content_type=upload.content_type,
                    data=data,
                    max_file_bytes=settings.manual_upload_max_file_bytes,
                )
            )
        except ManualUploadTooLargeError as exc:
            log_event(
                "manual_source.file_upload_rejected",
                sourceId=data_source.id,
                filename=upload.filename,
                reason="too_large",
                detail=str(exc),
            )
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(exc)) from exc
        except ManualUploadUnsupportedTypeError as exc:
            log_event(
                "manual_source.file_upload_rejected",
                sourceId=data_source.id,
                filename=upload.filename,
                reason="unsupported_type",
                detail=str(exc),
            )
            raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)) from exc
        except ManualUploadEmptyFileError as exc:
            log_event(
                "manual_source.file_upload_rejected",
                sourceId=data_source.id,
                filename=upload.filename,
                reason="empty_file",
                detail=str(exc),
            )
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    log_event(
        "manual_source.file_upload_requested",
        sourceId=data_source.id,
        fileCount=len(objects),
    )
    return _sync_manual_objects(
        data_source=data_source,
        objects=objects,
        store=store,
        settings=settings,
        queue_ontology_reextraction=queue_ontology_reextraction,
        run_ontology_reextraction_inline=run_ontology_reextraction_inline,
        ontology_focus_concept=ontology_focus_concept,
        created_by="manual-file-upload",
        trigger=OntologyReExtractionTrigger.MANUAL_UPLOAD,
    )


def _get_manual_source_or_404(source_id: str, store: InMemoryStore) -> DataSource:
    try:
        data_source = store.get_data_source(source_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    if data_source.type != DataSourceType.MANUAL:
        log_event(
            "manual_source.sync_rejected",
            sourceId=data_source.id,
            sourceType=data_source.type,
            reason="not_manual_source",
        )
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "Manual sync is only available for manual sources. "
                "Manual upload is only available for manual sources. "
                "Provider-backed sources must use connector discovery, selection, and sync jobs."
            ),
        )
    return data_source


def _sync_manual_objects(
    *,
    data_source: DataSource,
    objects: list[SourceObject],
    store: InMemoryStore,
    settings: Settings,
    queue_ontology_reextraction: bool,
    run_ontology_reextraction_inline: bool,
    ontology_focus_concept: str | None,
    created_by: str,
    trigger: OntologyReExtractionTrigger,
) -> SyncSourceResponse:
    try:
        response = sync_source_objects(
            data_source=data_source,
            objects=objects,
            store=store,
            settings=settings,
        )
        if not queue_ontology_reextraction:
            return response
        reextraction_run = OntologyReExtractionService(store).queue_from_sync_response(
            response=response,
            trigger=trigger,
            created_by=created_by,
            focus_concept=ontology_focus_concept,
        )
        if reextraction_run is None:
            return response
        if run_ontology_reextraction_inline:
            reextraction_response = OntologyReExtractionService(store).run(
                reextraction_run.id,
                RunOntologyReExtractionRunRequest(requested_by=created_by),
            )
            reextraction_run = reextraction_response.run
        return response.model_copy(
            update={
                "ontology_reextraction_run_id": reextraction_run.id,
                "ontology_reextraction_status": str(reextraction_run.status),
            },
            deep=True,
        )
    except SourceNotSyncableError as exc:
        log_event(
            "manual_source.sync_blocked",
            sourceId=data_source.id,
            sourceType=data_source.type,
            status=data_source.status,
            reason="source_status_not_syncable",
        )
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
