from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from cornerstone.api.dependencies import get_store
from cornerstone.config import Settings, get_settings
from cornerstone.observability import log_event
from cornerstone.schemas import (
    AuditEvent,
    Concept,
    ConceptStatus,
    CreateConceptRequest,
    OfficializeConceptRequest,
)
from cornerstone.services.officialization import (
    OfficializationError,
    ReviewerAuthorizationError,
    officialize_concept,
)
from cornerstone.store import InMemoryStore, NotFoundError

router = APIRouter(prefix="/concepts", tags=["concepts"])


@router.get("", response_model=list[Concept])
def list_concepts(store: InMemoryStore = Depends(get_store)) -> list[Concept]:
    concepts = store.list_concepts()
    log_event("concept.listed", conceptCount=len(concepts))
    return concepts


@router.get("/{concept_id}", response_model=Concept)
def get_concept(concept_id: str, store: InMemoryStore = Depends(get_store)) -> Concept:
    concept = _get_concept_or_404(concept_id, store)
    log_event("concept.read", conceptId=concept.id, status=concept.status)
    return concept


@router.post("", response_model=Concept, status_code=status.HTTP_201_CREATED)
def create_concept(
    request: CreateConceptRequest,
    store: InMemoryStore = Depends(get_store),
) -> Concept:
    _validate_evidence_exists(request.evidence_fragment_ids, store)
    _validate_decision_records_exist(request.decision_record_ids, store)
    concept = Concept(
        name=request.name,
        short_definition=request.short_definition,
        body=request.body,
        owner=request.owner,
        status=ConceptStatus.CANDIDATE,
        evidence_fragment_ids=request.evidence_fragment_ids,
        decision_record_ids=request.decision_record_ids,
        created_by=request.created_by,
    )
    event = AuditEvent(
        event_type="concept.created",
        actor=concept.created_by,
        entity_type="Concept",
        entity_id=concept.id,
        metadata={
            "conceptName": concept.name,
            "evidenceFragmentCount": len(concept.evidence_fragment_ids),
            "decisionRecordCount": len(concept.decision_record_ids),
        },
    )
    with store.transaction():
        saved = store.add_concept(concept)
        store.add_audit_event(event)
    log_event(
        "concept.created",
        conceptId=saved.id,
        name=saved.name,
        status=saved.status,
        evidenceFragmentCount=len(saved.evidence_fragment_ids),
        decisionRecordCount=len(saved.decision_record_ids),
        actor=saved.created_by,
        auditEventId=event.id,
    )
    return saved


@router.post("/{concept_id}/officialize", response_model=Concept)
def officialize(
    concept_id: str,
    request: OfficializeConceptRequest,
    store: InMemoryStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> Concept:
    concept = _get_concept_or_404(concept_id, store)
    log_event(
        "concept.officialize_attempted",
        conceptId=concept.id,
        status=concept.status,
        actor=request.reviewed_by,
        evidenceFragmentCount=len(concept.evidence_fragment_ids),
        decisionRecordCount=len(concept.decision_record_ids),
    )
    try:
        updated, event = officialize_concept(
            concept,
            reviewed_by=request.reviewed_by,
            evidence=store.list_evidence_fragments(),
            decision_records=store.list_decision_records(),
            artifacts=store.list_artifacts(),
            data_sources=store.list_data_sources(),
            production_mode=settings.production_mode,
            authorized_reviewers=settings.authorized_reviewers,
        )
    except ReviewerAuthorizationError as exc:
        _record_officialization_blocked(concept, request.reviewed_by, str(exc), store)
        log_event(
            "concept.officialize_blocked",
            conceptId=concept.id,
            status=concept.status,
            actor=request.reviewed_by,
            reason=str(exc),
        )
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except OfficializationError as exc:
        _record_officialization_blocked(concept, request.reviewed_by, str(exc), store)
        log_event(
            "concept.officialize_blocked",
            conceptId=concept.id,
            status=concept.status,
            actor=request.reviewed_by,
            reason=str(exc),
        )
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    with store.transaction():
        saved = store.update_concept(updated)
        store.add_audit_event(event)
    log_event(
        "concept.officialized",
        conceptId=saved.id,
        status=saved.status,
        actor=request.reviewed_by,
        auditEventId=event.id,
    )
    return saved


def _get_concept_or_404(concept_id: str, store: InMemoryStore) -> Concept:
    try:
        return store.get_concept(concept_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


def _validate_evidence_exists(evidence_fragment_ids: list[str], store: InMemoryStore) -> None:
    for evidence_fragment_id in evidence_fragment_ids:
        try:
            store.get_evidence_fragment(evidence_fragment_id)
        except NotFoundError as exc:
            log_event(
                "concept.create_blocked",
                reason="evidence_not_found",
                evidenceFragmentId=evidence_fragment_id,
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"EvidenceFragment not found: {evidence_fragment_id}",
            ) from exc


def _validate_decision_records_exist(decision_record_ids: list[str], store: InMemoryStore) -> None:
    for decision_record_id in decision_record_ids:
        try:
            store.get_decision_record(decision_record_id)
        except NotFoundError as exc:
            log_event(
                "concept.create_blocked",
                reason="decision_record_not_found",
                decisionRecordId=decision_record_id,
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"DecisionRecord not found: {decision_record_id}",
            ) from exc


def _record_officialization_blocked(
    concept: Concept,
    actor: str,
    reason: str,
    store: InMemoryStore,
) -> None:
    event = AuditEvent(
        event_type="concept.officialization_blocked",
        actor=actor,
        entity_type="Concept",
        entity_id=concept.id,
        metadata={"conceptName": concept.name, "reason": reason},
    )
    store.add_audit_event(event)
