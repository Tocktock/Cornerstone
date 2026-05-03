from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from cornerstone.api.dependencies import get_store
from cornerstone.config import Settings, get_settings
from cornerstone.observability import log_event
from cornerstone.schemas import (
    AuditEvent,
    Concept,
    ConceptRelation,
    CreateConceptRelationRequest,
    DecisionRecord,
    EvidenceFragment,
    OfficializeConceptRelationRequest,
    RelationStatus,
)
from cornerstone.services.officialization import (
    OfficializationError,
    ReviewerAuthorizationError,
    ensure_reviewer_authorized,
    officialize_concept_relation,
)
from cornerstone.store import InMemoryStore, NotFoundError

router = APIRouter(prefix="/concept-relations", tags=["concept-relations"])


@router.get("", response_model=list[ConceptRelation])
def list_concept_relations(
    concept_id: str | None = Query(default=None, alias="conceptId"),
    store: InMemoryStore = Depends(get_store),
) -> list[ConceptRelation]:
    relations = store.list_concept_relations(concept_id=concept_id)
    log_event("concept_relation.listed", conceptId=concept_id, conceptRelationCount=len(relations))
    return relations


@router.get("/{relation_id}", response_model=ConceptRelation)
def get_concept_relation(
    relation_id: str,
    store: InMemoryStore = Depends(get_store),
) -> ConceptRelation:
    relation = _get_relation_or_404(relation_id, store)
    log_event("concept_relation.read", conceptRelationId=relation.id, status=relation.status)
    return relation


@router.post("", response_model=ConceptRelation, status_code=status.HTTP_201_CREATED)
def create_concept_relation(
    request: CreateConceptRelationRequest,
    store: InMemoryStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> ConceptRelation:
    try:
        ensure_reviewer_authorized(request.created_by, settings.authorized_reviewers)
    except ReviewerAuthorizationError as exc:
        log_event(
            "concept_relation.create_blocked",
            actor=request.created_by,
            reason="reviewer_not_authorized",
        )
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc

    _get_concept_or_404(request.source_concept_id, store)
    _get_concept_or_404(request.target_concept_id, store)
    _validate_evidence_exists(request.evidence_fragment_ids, store)
    if request.decision_record_id is not None:
        _get_decision_record_or_404(request.decision_record_id, store)

    relation = ConceptRelation(
        source_concept_id=request.source_concept_id,
        target_concept_id=request.target_concept_id,
        relation_type=request.relation_type,
        status=RelationStatus(request.status),
        evidence_fragment_ids=request.evidence_fragment_ids,
        decision_record_id=request.decision_record_id,
        created_by=request.created_by,
    )
    event = AuditEvent(
        event_type="concept_relation.created",
        actor=request.created_by,
        entity_type="ConceptRelation",
        entity_id=relation.id,
        metadata={
            "sourceConceptId": relation.source_concept_id,
            "targetConceptId": relation.target_concept_id,
            "relationType": relation.relation_type,
            "evidenceFragmentCount": len(relation.evidence_fragment_ids),
            "decisionRecordId": relation.decision_record_id,
        },
    )
    with store.transaction():
        saved = store.add_concept_relation(relation)
        store.add_audit_event(event)
    log_event(
        "concept_relation.created",
        conceptRelationId=saved.id,
        relationType=saved.relation_type,
        sourceConceptId=saved.source_concept_id,
        targetConceptId=saved.target_concept_id,
        evidenceFragmentCount=len(saved.evidence_fragment_ids),
        actor=saved.created_by,
        auditEventId=event.id,
    )
    return saved


@router.post("/{relation_id}/officialize", response_model=ConceptRelation)
def officialize_relation(
    relation_id: str,
    request: OfficializeConceptRelationRequest,
    store: InMemoryStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> ConceptRelation:
    relation = _get_relation_or_404(relation_id, store)
    log_event(
        "concept_relation.officialize_attempted",
        conceptRelationId=relation.id,
        status=relation.status,
        actor=request.reviewed_by,
        evidenceFragmentCount=len(relation.evidence_fragment_ids),
        decisionRecordId=relation.decision_record_id,
    )
    try:
        updated, event = officialize_concept_relation(
            relation,
            reviewed_by=request.reviewed_by,
            concepts=store.list_concepts(),
            evidence=store.list_evidence_fragments(),
            decision_records=store.list_decision_records(),
            artifacts=store.list_artifacts(),
            data_sources=store.list_data_sources(),
            production_mode=settings.production_mode,
            authorized_reviewers=settings.authorized_reviewers,
        )
    except ReviewerAuthorizationError as exc:
        _record_relation_officialization_blocked(relation, request.reviewed_by, str(exc), store)
        log_event(
            "concept_relation.officialize_blocked",
            conceptRelationId=relation.id,
            actor=request.reviewed_by,
            reason=str(exc),
        )
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    except OfficializationError as exc:
        _record_relation_officialization_blocked(relation, request.reviewed_by, str(exc), store)
        log_event(
            "concept_relation.officialize_blocked",
            conceptRelationId=relation.id,
            actor=request.reviewed_by,
            reason=str(exc),
        )
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    with store.transaction():
        saved = store.update_concept_relation(updated)
        store.add_audit_event(event)
    log_event(
        "concept_relation.officialized",
        conceptRelationId=saved.id,
        status=saved.status,
        actor=request.reviewed_by,
        auditEventId=event.id,
    )
    return saved


def _get_relation_or_404(relation_id: str, store: InMemoryStore) -> ConceptRelation:
    try:
        return store.get_concept_relation(relation_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc


def _get_concept_or_404(concept_id: str, store: InMemoryStore) -> Concept:
    try:
        return store.get_concept(concept_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Concept not found: {concept_id}") from exc


def _get_decision_record_or_404(decision_record_id: str, store: InMemoryStore) -> DecisionRecord:
    try:
        return store.get_decision_record(decision_record_id)
    except NotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"DecisionRecord not found: {decision_record_id}",
        ) from exc


def _validate_evidence_exists(evidence_fragment_ids: list[str], store: InMemoryStore) -> list[EvidenceFragment]:
    evidence = []
    for evidence_fragment_id in evidence_fragment_ids:
        try:
            evidence.append(store.get_evidence_fragment(evidence_fragment_id))
        except NotFoundError as exc:
            log_event(
                "concept_relation.create_blocked",
                reason="evidence_not_found",
                evidenceFragmentId=evidence_fragment_id,
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"EvidenceFragment not found: {evidence_fragment_id}",
            ) from exc
    return evidence


def _record_relation_officialization_blocked(
    relation: ConceptRelation,
    actor: str,
    reason: str,
    store: InMemoryStore,
) -> None:
    event = AuditEvent(
        event_type="concept_relation.officialization_blocked",
        actor=actor,
        entity_type="ConceptRelation",
        entity_id=relation.id,
        metadata={
            "sourceConceptId": relation.source_concept_id,
            "targetConceptId": relation.target_concept_id,
            "relationType": relation.relation_type,
            "reason": reason,
        },
    )
    store.add_audit_event(event)
