from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from cornerstone.api.dependencies import get_store
from cornerstone.config import Settings, get_settings
from cornerstone.observability import log_event
from cornerstone.schemas import (
    ApproveConceptCandidateRequest,
    ApproveRelationCandidateRequest,
    CandidateReviewPreview,
    CandidateReviewQueueSummary,
    ConceptCandidateListResponse,
    ConceptCandidateReviewResponse,
    CreateOntologyExtractionRunRequest,
    CreateOntologyProofRunRequest,
    CreateOntologyReExtractionRunRequest,
    EditConceptCandidateRequest,
    EditRelationCandidateRequest,
    MergeConceptCandidateRequest,
    MergeRelationCandidateRequest,
    OntologyCandidateStatus,
    OntologyExtractionRunListResponse,
    OntologyExtractionRunResponse,
    OntologyProofRunResponse,
    OntologySsotReadinessResponse,
    OntologyReExtractionRunListResponse,
    OntologyReExtractionRunResponse,
    OntologyReExtractionRunStatus,
    OntologyGraphMode,
    OntologyGraphResponse,
    OntologySearchResponse,
    RejectOntologyCandidateRequest,
    RelationCandidateListResponse,
    RunOntologyReExtractionRunRequest,
    RelationCandidateReviewResponse,
)
from cornerstone.services.officialization import OfficializationError, ReviewerAuthorizationError
from cornerstone.services.ontology_extraction import OntologyExtractionService
from cornerstone.services.ontology_graph import OntologyGraphService
from cornerstone.services.ontology_proof import OntologyProofError, OntologyProofService
from cornerstone.services.ontology_reextraction import OntologyReExtractionError, OntologyReExtractionService
from cornerstone.services.ontology_review import OntologyCandidateReviewService, OntologyReviewError
from cornerstone.services.ontology_ssot_readiness import OntologySsotReadinessService
from cornerstone.store import InMemoryStore, NotFoundError

router = APIRouter(prefix="/ontology", tags=["ontology"])


@router.get("/search", response_model=OntologySearchResponse)
def search_ontology(
    q: str = Query(min_length=1),
    mode: OntologyGraphMode = Query(default=OntologyGraphMode.OFFICIAL),
    limit: int = Query(default=10, ge=1, le=50),
    store: InMemoryStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> OntologySearchResponse:
    service = OntologyGraphService(store, production_mode=settings.production_mode)
    response = service.search(q, mode=mode, limit=limit)
    log_event(
        "ontology.search",
        query=q,
        mode=mode,
        resultCount=len(response.results),
    )
    return response


@router.get("/graph", response_model=OntologyGraphResponse)
def get_ontology_graph(
    concept: str = Query(min_length=1),
    depth: int = Query(default=1, ge=0, le=1),
    mode: OntologyGraphMode = Query(default=OntologyGraphMode.OFFICIAL),
    store: InMemoryStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> OntologyGraphResponse:
    service = OntologyGraphService(store, production_mode=settings.production_mode)
    try:
        response = service.graph(concept, depth=depth, mode=mode)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    log_event(
        "ontology.graph_served",
        query=concept,
        mode=mode,
        depth=depth,
        nodeCount=len(response.nodes),
        edgeCount=len(response.edges),
        trustLabel=response.trust_label,
    )
    return response


@router.get("/explain", response_model=OntologyGraphResponse)
def explain_ontology_graph(
    concept: str = Query(min_length=1),
    depth: int = Query(default=1, ge=0, le=1),
    mode: OntologyGraphMode = Query(default=OntologyGraphMode.OFFICIAL),
    store: InMemoryStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> OntologyGraphResponse:
    service = OntologyGraphService(store, production_mode=settings.production_mode)
    try:
        response = service.graph(concept, depth=depth, mode=mode)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    log_event(
        "ontology.graph_explained",
        query=concept,
        mode=mode,
        depth=depth,
        nodeCount=len(response.nodes),
        edgeCount=len(response.edges),
        trustLabel=response.trust_label,
    )
    return response


@router.get("/ssot/readiness", response_model=OntologySsotReadinessResponse)
def get_ontology_ssot_readiness(
    focus_concept: str = Query(default="settlement", min_length=1, alias="focusConcept"),
    depth: int = Query(default=1, ge=0, le=1),
    mode: OntologyGraphMode = Query(default=OntologyGraphMode.OFFICIAL),
    include_graph: bool = Query(default=False, alias="includeGraph"),
    store: InMemoryStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> OntologySsotReadinessResponse:
    service = OntologySsotReadinessService(store, production_mode=settings.production_mode)
    try:
        response = service.readiness(
            focus_concept=focus_concept,
            depth=depth,
            mode=mode,
            include_graph=include_graph,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    log_event(
        "ontology.ssot_readiness_served",
        focusConcept=focus_concept,
        mode=mode,
        depth=depth,
        status=response.status,
        officialGraphSafe=response.official_graph_safe,
        requiredFailed=sum(1 for check in response.checks if check.required and check.status != "passed"),
    )
    return response


@router.post("/proof-runs", response_model=OntologyProofRunResponse, status_code=status.HTTP_201_CREATED)
def run_ontology_proof(
    request: CreateOntologyProofRunRequest,
    store: InMemoryStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> OntologyProofRunResponse:
    service = OntologyProofService(store, settings=settings)
    try:
        response = service.run(request)
    except OntologyProofError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    log_event(
        "ontology.proof_run_completed",
        runId=response.run_id,
        status=response.status,
        focusConcept=response.focus_concept,
        dryRun=response.dry_run,
        requiredPassed=response.summary.required_passed,
        requiredFailed=response.summary.required_failed,
        officialGraphAvailable=response.summary.official_graph_available,
    )
    return response


@router.post("/extraction-runs", response_model=OntologyExtractionRunResponse, status_code=status.HTTP_201_CREATED)
def create_ontology_extraction_run(
    request: CreateOntologyExtractionRunRequest,
    store: InMemoryStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> OntologyExtractionRunResponse:
    service = OntologyExtractionService(store, settings=settings)
    try:
        response = service.create_run(request)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    log_event(
        "ontology.extraction_run_created",
        runId=response.run.id,
        provider=response.run.provider,
        conceptCandidateCount=response.run.concept_candidate_count,
        relationCandidateCount=response.run.relation_candidate_count,
    )
    return response


@router.get("/extraction-runs", response_model=OntologyExtractionRunListResponse)
def list_ontology_extraction_runs(
    store: InMemoryStore = Depends(get_store),
) -> OntologyExtractionRunListResponse:
    return OntologyExtractionRunListResponse(runs=store.list_ontology_extraction_runs())


@router.get("/extraction-runs/{run_id}", response_model=OntologyExtractionRunResponse)
def get_ontology_extraction_run(
    run_id: str,
    store: InMemoryStore = Depends(get_store),
) -> OntologyExtractionRunResponse:
    service = OntologyExtractionService(store)
    try:
        return service.get_run_response(run_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/re-extraction-runs", response_model=OntologyReExtractionRunResponse, status_code=status.HTTP_201_CREATED)
def create_ontology_reextraction_run(
    request: CreateOntologyReExtractionRunRequest,
    store: InMemoryStore = Depends(get_store),
) -> OntologyReExtractionRunResponse:
    service = OntologyReExtractionService(store)
    try:
        response = service.create_run(request)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (ValueError, OntologyReExtractionError) as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    log_event(
        "ontology.reextraction_run_created",
        runId=response.run.id,
        sourceId=response.run.datasource_id,
        trigger=response.run.trigger,
        status=response.run.status,
    )
    return response


@router.get("/re-extraction-runs", response_model=OntologyReExtractionRunListResponse)
def list_ontology_reextraction_runs(
    datasource_id: str | None = Query(default=None, alias="datasourceId"),
    run_status: OntologyReExtractionRunStatus | None = Query(default=None, alias="status"),
    store: InMemoryStore = Depends(get_store),
) -> OntologyReExtractionRunListResponse:
    service = OntologyReExtractionService(store)
    return service.list_runs(datasource_id=datasource_id, status=run_status)


@router.get("/re-extraction-runs/{run_id}", response_model=OntologyReExtractionRunResponse)
def get_ontology_reextraction_run(
    run_id: str,
    store: InMemoryStore = Depends(get_store),
) -> OntologyReExtractionRunResponse:
    service = OntologyReExtractionService(store)
    try:
        return service.get_response(run_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/re-extraction-runs/{run_id}/run", response_model=OntologyReExtractionRunResponse)
def run_ontology_reextraction_run(
    run_id: str,
    request: RunOntologyReExtractionRunRequest | None = None,
    store: InMemoryStore = Depends(get_store),
) -> OntologyReExtractionRunResponse:
    service = OntologyReExtractionService(store)
    try:
        response = service.run(run_id, request)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except OntologyReExtractionError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    log_event(
        "ontology.reextraction_run_executed",
        runId=response.run.id,
        status=response.run.status,
        extractionRunIds=response.run.extraction_run_ids,
        officialGraphMutated=response.run.official_graph_mutated,
    )
    return response


@router.get("/concept-candidates", response_model=ConceptCandidateListResponse)
def list_concept_candidates(
    extraction_run_id: str | None = Query(default=None, alias="runId"),
    candidate_status: OntologyCandidateStatus | None = Query(default=None, alias="status"),
    store: InMemoryStore = Depends(get_store),
) -> ConceptCandidateListResponse:
    return ConceptCandidateListResponse(
        candidates=store.list_concept_candidates(
            extraction_run_id=extraction_run_id,
            status=candidate_status,
        )
    )


@router.get("/relation-candidates", response_model=RelationCandidateListResponse)
def list_relation_candidates(
    extraction_run_id: str | None = Query(default=None, alias="runId"),
    candidate_status: OntologyCandidateStatus | None = Query(default=None, alias="status"),
    store: InMemoryStore = Depends(get_store),
) -> RelationCandidateListResponse:
    return RelationCandidateListResponse(
        candidates=store.list_relation_candidates(
            extraction_run_id=extraction_run_id,
            status=candidate_status,
        )
    )


@router.get("/review-queue/summary", response_model=CandidateReviewQueueSummary)
def get_candidate_review_queue_summary(
    extraction_run_id: str | None = Query(default=None, alias="runId"),
    source_id: str | None = Query(default=None, alias="sourceId"),
    candidate_status: OntologyCandidateStatus | None = Query(default=OntologyCandidateStatus.PENDING, alias="status"),
    min_confidence: float | None = Query(default=None, ge=0, le=1, alias="minConfidence"),
    max_confidence: float | None = Query(default=None, ge=0, le=1, alias="maxConfidence"),
    store: InMemoryStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> CandidateReviewQueueSummary:
    service = OntologyCandidateReviewService(
        store,
        production_mode=settings.production_mode,
        authorized_reviewers=settings.authorized_reviewers,
    )
    return service.queue_summary(
        status=candidate_status,
        extraction_run_id=extraction_run_id,
        source_id=source_id,
        min_confidence=min_confidence,
        max_confidence=max_confidence,
    )


@router.get("/concept-candidates/{candidate_id}/preview", response_model=CandidateReviewPreview)
def preview_concept_candidate(
    candidate_id: str,
    action: str = Query(default="approve", pattern="^(approve|merge|reject)$"),
    target_concept_id: str | None = Query(default=None, alias="targetConceptId"),
    store: InMemoryStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> CandidateReviewPreview:
    service = OntologyCandidateReviewService(
        store,
        production_mode=settings.production_mode,
        authorized_reviewers=settings.authorized_reviewers,
    )
    try:
        return service.preview_concept_candidate(
            candidate_id,
            action=action,
            target_concept_id=target_concept_id,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/relation-candidates/{candidate_id}/preview", response_model=CandidateReviewPreview)
def preview_relation_candidate(
    candidate_id: str,
    action: str = Query(default="approve", pattern="^(approve|merge|reject)$"),
    target_relation_id: str | None = Query(default=None, alias="targetRelationId"),
    store: InMemoryStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> CandidateReviewPreview:
    service = OntologyCandidateReviewService(
        store,
        production_mode=settings.production_mode,
        authorized_reviewers=settings.authorized_reviewers,
    )
    try:
        return service.preview_relation_candidate(
            candidate_id,
            action=action,
            target_relation_id=target_relation_id,
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.patch("/concept-candidates/{candidate_id}", response_model=ConceptCandidateReviewResponse)
def edit_concept_candidate(
    candidate_id: str,
    request: EditConceptCandidateRequest,
    store: InMemoryStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> ConceptCandidateReviewResponse:
    service = OntologyCandidateReviewService(
        store,
        production_mode=settings.production_mode,
        authorized_reviewers=settings.authorized_reviewers,
    )
    try:
        response = service.edit_concept_candidate(candidate_id, request)
    except ReviewerAuthorizationError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except OntologyReviewError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    log_event("ontology.concept_candidate_edited", candidateId=response.candidate.id, actor=request.edited_by)
    return response


@router.post("/concept-candidates/{candidate_id}/approve", response_model=ConceptCandidateReviewResponse)
def approve_concept_candidate(
    candidate_id: str,
    request: ApproveConceptCandidateRequest,
    store: InMemoryStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> ConceptCandidateReviewResponse:
    service = OntologyCandidateReviewService(
        store,
        production_mode=settings.production_mode,
        authorized_reviewers=settings.authorized_reviewers,
    )
    try:
        response = service.approve_concept_candidate(candidate_id, request)
    except ReviewerAuthorizationError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (OfficializationError, OntologyReviewError) as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    log_event(
        "ontology.concept_candidate_approved",
        candidateId=response.candidate.id,
        conceptId=None if response.concept is None else response.concept.id,
        actor=request.reviewed_by,
    )
    return response


@router.post("/concept-candidates/{candidate_id}/reject", response_model=ConceptCandidateReviewResponse)
def reject_concept_candidate(
    candidate_id: str,
    request: RejectOntologyCandidateRequest,
    store: InMemoryStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> ConceptCandidateReviewResponse:
    service = OntologyCandidateReviewService(
        store,
        production_mode=settings.production_mode,
        authorized_reviewers=settings.authorized_reviewers,
    )
    try:
        response = service.reject_concept_candidate(candidate_id, request)
    except ReviewerAuthorizationError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except OntologyReviewError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    log_event("ontology.concept_candidate_rejected", candidateId=response.candidate.id, actor=request.reviewed_by)
    return response


@router.post("/concept-candidates/{candidate_id}/merge", response_model=ConceptCandidateReviewResponse)
def merge_concept_candidate(
    candidate_id: str,
    request: MergeConceptCandidateRequest,
    store: InMemoryStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> ConceptCandidateReviewResponse:
    service = OntologyCandidateReviewService(
        store,
        production_mode=settings.production_mode,
        authorized_reviewers=settings.authorized_reviewers,
    )
    try:
        response = service.merge_concept_candidate(candidate_id, request)
    except ReviewerAuthorizationError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (OfficializationError, OntologyReviewError) as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    log_event(
        "ontology.concept_candidate_merged",
        candidateId=response.candidate.id,
        conceptId=None if response.concept is None else response.concept.id,
        actor=request.reviewed_by,
    )
    return response


@router.patch("/relation-candidates/{candidate_id}", response_model=RelationCandidateReviewResponse)
def edit_relation_candidate(
    candidate_id: str,
    request: EditRelationCandidateRequest,
    store: InMemoryStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> RelationCandidateReviewResponse:
    service = OntologyCandidateReviewService(
        store,
        production_mode=settings.production_mode,
        authorized_reviewers=settings.authorized_reviewers,
    )
    try:
        response = service.edit_relation_candidate(candidate_id, request)
    except ReviewerAuthorizationError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except OntologyReviewError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    log_event("ontology.relation_candidate_edited", candidateId=response.candidate.id, actor=request.edited_by)
    return response


@router.post("/relation-candidates/{candidate_id}/approve", response_model=RelationCandidateReviewResponse)
def approve_relation_candidate(
    candidate_id: str,
    request: ApproveRelationCandidateRequest,
    store: InMemoryStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> RelationCandidateReviewResponse:
    service = OntologyCandidateReviewService(
        store,
        production_mode=settings.production_mode,
        authorized_reviewers=settings.authorized_reviewers,
    )
    try:
        response = service.approve_relation_candidate(candidate_id, request)
    except ReviewerAuthorizationError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (OfficializationError, OntologyReviewError) as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    log_event(
        "ontology.relation_candidate_approved",
        candidateId=response.candidate.id,
        relationId=None if response.relation is None else response.relation.id,
        actor=request.reviewed_by,
    )
    return response


@router.post("/relation-candidates/{candidate_id}/reject", response_model=RelationCandidateReviewResponse)
def reject_relation_candidate(
    candidate_id: str,
    request: RejectOntologyCandidateRequest,
    store: InMemoryStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> RelationCandidateReviewResponse:
    service = OntologyCandidateReviewService(
        store,
        production_mode=settings.production_mode,
        authorized_reviewers=settings.authorized_reviewers,
    )
    try:
        response = service.reject_relation_candidate(candidate_id, request)
    except ReviewerAuthorizationError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except OntologyReviewError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    log_event("ontology.relation_candidate_rejected", candidateId=response.candidate.id, actor=request.reviewed_by)
    return response


@router.post("/relation-candidates/{candidate_id}/merge", response_model=RelationCandidateReviewResponse)
def merge_relation_candidate(
    candidate_id: str,
    request: MergeRelationCandidateRequest,
    store: InMemoryStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> RelationCandidateReviewResponse:
    service = OntologyCandidateReviewService(
        store,
        production_mode=settings.production_mode,
        authorized_reviewers=settings.authorized_reviewers,
    )
    try:
        response = service.merge_relation_candidate(candidate_id, request)
    except ReviewerAuthorizationError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (OfficializationError, OntologyReviewError) as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    log_event(
        "ontology.relation_candidate_merged",
        candidateId=response.candidate.id,
        relationId=None if response.relation is None else response.relation.id,
        actor=request.reviewed_by,
    )
    return response
