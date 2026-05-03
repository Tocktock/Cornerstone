from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, Depends, HTTPException, Query, status

from cornerstone.api.dependencies import get_store
from cornerstone.config import Settings, get_settings
from cornerstone.observability import log_event
from cornerstone.schemas import (
    CreateGroundedContextEvalTaskRequest,
    CreateOntologyGraphEvalTaskRequest,
    GroundedContextEvalMetricSummary,
    GroundedContextEvalResult,
    GroundedContextEvalRunResponse,
    GroundedContextEvalTask,
    OntologyGraphEvalMetricSummary,
    OntologyGraphEvalResult,
    OntologyGraphEvalRunResponse,
    OntologyGraphEvalTask,
    RunGroundedContextEvalRequest,
    RunGroundedContextEvalTaskRequest,
    RunOntologyGraphEvalRequest,
    RunOntologyGraphEvalTaskRequest,
)
from cornerstone.services.evaluation import GroundedContextEvaluationService, OntologyGraphEvaluationService
from cornerstone.store import NotFoundError

router = APIRouter(prefix="/evaluations", tags=["evaluations"])


@router.post("/tasks", response_model=GroundedContextEvalTask, status_code=status.HTTP_201_CREATED)
def create_eval_task(
    request: CreateGroundedContextEvalTaskRequest,
    store: Any = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> GroundedContextEvalTask:
    task = GroundedContextEvaluationService(store, production_mode=settings.production_mode).create_task(request)
    log_event(
        "evaluation.task_created",
        taskId=task.id,
        query=task.query,
        expectedTrustLabel=task.expected_trust_label,
    )
    return task


@router.get("/tasks", response_model=list[GroundedContextEvalTask])
def list_eval_tasks(store: Any = Depends(get_store)) -> list[GroundedContextEvalTask]:
    return cast(list[GroundedContextEvalTask], store.list_grounded_context_eval_tasks())


@router.get("/tasks/{task_id}", response_model=GroundedContextEvalTask)
def get_eval_task(task_id: str, store: Any = Depends(get_store)) -> GroundedContextEvalTask:
    try:
        return cast(GroundedContextEvalTask, store.get_grounded_context_eval_task(task_id))
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/tasks/{task_id}/run", response_model=GroundedContextEvalResult)
def run_eval_task(
    task_id: str,
    request: RunGroundedContextEvalTaskRequest | None = None,
    store: Any = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> GroundedContextEvalResult:
    try:
        result = GroundedContextEvaluationService(store, production_mode=settings.production_mode).run_task(
            task_id,
            request=request or RunGroundedContextEvalTaskRequest(),
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    log_event(
        "evaluation.task_completed",
        taskId=result.task_id,
        resultId=result.id,
        success=result.success,
        trustLabel=result.trust_label,
    )
    return result


@router.post("/run", response_model=GroundedContextEvalRunResponse)
def run_eval_tasks(
    request: RunGroundedContextEvalRequest | None = None,
    store: Any = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> GroundedContextEvalRunResponse:
    try:
        response = GroundedContextEvaluationService(store, production_mode=settings.production_mode).run_tasks(request)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    log_event(
        "evaluation.run_completed",
        totalCount=response.total_count,
        successCount=response.success_count,
        groundedContextTaskSuccessRate=response.grounded_context_task_success_rate,
    )
    return response


@router.get("/results", response_model=list[GroundedContextEvalResult])
def list_eval_results(
    task_id: str | None = Query(default=None, alias="taskId"),
    store: Any = Depends(get_store),
) -> list[GroundedContextEvalResult]:
    return cast(list[GroundedContextEvalResult], store.list_grounded_context_eval_results(task_id=task_id))


@router.get("/results/{result_id}", response_model=GroundedContextEvalResult)
def get_eval_result(result_id: str, store: Any = Depends(get_store)) -> GroundedContextEvalResult:
    try:
        return cast(GroundedContextEvalResult, store.get_grounded_context_eval_result(result_id))
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/summary", response_model=GroundedContextEvalMetricSummary)
def get_eval_summary(
    store: Any = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> GroundedContextEvalMetricSummary:
    return GroundedContextEvaluationService(store, production_mode=settings.production_mode).summarize()

@router.post("/ontology/tasks", response_model=OntologyGraphEvalTask, status_code=status.HTTP_201_CREATED)
def create_ontology_graph_eval_task(
    request: CreateOntologyGraphEvalTaskRequest,
    store: Any = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> OntologyGraphEvalTask:
    task = OntologyGraphEvaluationService(store, production_mode=settings.production_mode).create_task(request)
    log_event(
        "evaluation.ontology_graph_task_created",
        taskId=task.id,
        conceptQuery=task.concept_query,
        expectedTrustLabel=task.expected_trust_label,
    )
    return task


@router.get("/ontology/tasks", response_model=list[OntologyGraphEvalTask])
def list_ontology_graph_eval_tasks(store: Any = Depends(get_store)) -> list[OntologyGraphEvalTask]:
    return cast(list[OntologyGraphEvalTask], store.list_ontology_graph_eval_tasks())


@router.get("/ontology/tasks/{task_id}", response_model=OntologyGraphEvalTask)
def get_ontology_graph_eval_task(task_id: str, store: Any = Depends(get_store)) -> OntologyGraphEvalTask:
    try:
        return cast(OntologyGraphEvalTask, store.get_ontology_graph_eval_task(task_id))
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.post("/ontology/tasks/{task_id}/run", response_model=OntologyGraphEvalResult)
def run_ontology_graph_eval_task(
    task_id: str,
    request: RunOntologyGraphEvalTaskRequest | None = None,
    store: Any = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> OntologyGraphEvalResult:
    try:
        result = OntologyGraphEvaluationService(store, production_mode=settings.production_mode).run_task(
            task_id,
            request=request or RunOntologyGraphEvalTaskRequest(),
        )
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    log_event(
        "evaluation.ontology_graph_task_completed",
        taskId=result.task_id,
        resultId=result.id,
        success=result.success,
        trustLabel=result.trust_label,
    )
    return result


@router.post("/ontology/run", response_model=OntologyGraphEvalRunResponse)
def run_ontology_graph_eval_tasks(
    request: RunOntologyGraphEvalRequest | None = None,
    store: Any = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> OntologyGraphEvalRunResponse:
    try:
        response = OntologyGraphEvaluationService(store, production_mode=settings.production_mode).run_tasks(request)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    log_event(
        "evaluation.ontology_graph_run_completed",
        totalCount=response.total_count,
        successCount=response.success_count,
        ontologyGraphTaskSuccessRate=response.ontology_graph_task_success_rate,
    )
    return response


@router.get("/ontology/results", response_model=list[OntologyGraphEvalResult])
def list_ontology_graph_eval_results(
    task_id: str | None = Query(default=None, alias="taskId"),
    store: Any = Depends(get_store),
) -> list[OntologyGraphEvalResult]:
    return cast(list[OntologyGraphEvalResult], store.list_ontology_graph_eval_results(task_id=task_id))


@router.get("/ontology/results/{result_id}", response_model=OntologyGraphEvalResult)
def get_ontology_graph_eval_result(result_id: str, store: Any = Depends(get_store)) -> OntologyGraphEvalResult:
    try:
        return cast(OntologyGraphEvalResult, store.get_ontology_graph_eval_result(result_id))
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


@router.get("/ontology/summary", response_model=OntologyGraphEvalMetricSummary)
def get_ontology_graph_eval_summary(
    store: Any = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> OntologyGraphEvalMetricSummary:
    return OntologyGraphEvaluationService(store, production_mode=settings.production_mode).summarize()

