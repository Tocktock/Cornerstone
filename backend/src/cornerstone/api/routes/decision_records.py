from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from cornerstone.api.dependencies import get_store
from cornerstone.config import Settings, get_settings
from cornerstone.observability import log_event
from cornerstone.schemas import (
    Artifact,
    AuditEvent,
    CreateDecisionRecordRequest,
    DataSource,
    DecisionRecord,
    EvidenceFragment,
)
from cornerstone.services.officialization import (
    OfficializationError,
    ReviewerAuthorizationError,
    ensure_decision_record_is_officialization_ready,
    ensure_reviewer_authorized,
)
from cornerstone.store import InMemoryStore, NotFoundError

router = APIRouter(prefix="/decision-records", tags=["decision-records"])


@router.get("", response_model=list[DecisionRecord])
def list_decision_records(store: InMemoryStore = Depends(get_store)) -> list[DecisionRecord]:
    records = store.list_decision_records()
    log_event("decision_record.listed", decisionRecordCount=len(records))
    return records


@router.get("/{decision_record_id}", response_model=DecisionRecord)
def get_decision_record(
    decision_record_id: str,
    store: InMemoryStore = Depends(get_store),
) -> DecisionRecord:
    try:
        record = store.get_decision_record(decision_record_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    log_event("decision_record.read", decisionRecordId=record.id)
    return record


@router.post("", response_model=DecisionRecord, status_code=status.HTTP_201_CREATED)
def create_decision_record(
    request: CreateDecisionRecordRequest,
    store: InMemoryStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> DecisionRecord:
    try:
        ensure_reviewer_authorized(request.decided_by, settings.authorized_reviewers)
    except ReviewerAuthorizationError as exc:
        log_event(
            "decision_record.create_blocked",
            actor=request.decided_by,
            reason="reviewer_not_authorized",
        )
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc

    _validate_affected_concepts_exist(request.affected_concept_ids, store)

    record = DecisionRecord(
        title=request.title,
        decision=request.decision,
        reason=request.reason,
        alternatives_considered=request.alternatives_considered,
        decided_by=request.decided_by,
        evidence_fragment_ids=request.evidence_fragment_ids,
        affected_concept_ids=request.affected_concept_ids,
    )
    evidence = _load_evidence(request.evidence_fragment_ids, store)
    artifacts = _load_artifacts_for_evidence(evidence, store)
    data_sources = _load_data_sources_for_artifacts(artifacts, store)
    try:
        ensure_decision_record_is_officialization_ready(
            record,
            evidence_by_id={item.id: item for item in evidence},
            artifact_by_id={item.id: item for item in artifacts},
            data_source_by_id={item.id: item for item in data_sources},
            production_mode=settings.production_mode,
        )
    except OfficializationError as exc:
        log_event(
            "decision_record.create_blocked",
            actor=request.decided_by,
            reason=str(exc),
            evidenceFragmentCount=len(request.evidence_fragment_ids),
        )
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc

    event = AuditEvent(
        event_type="decision_record.created",
        actor=record.decided_by,
        entity_type="DecisionRecord",
        entity_id=record.id,
        metadata={
            "title": record.title,
            "evidenceFragmentCount": len(record.evidence_fragment_ids),
            "affectedConceptCount": len(record.affected_concept_ids),
        },
    )
    with store.transaction():
        saved = store.add_decision_record(record)
        store.add_audit_event(event)
    log_event(
        "decision_record.created",
        decisionRecordId=saved.id,
        actor=saved.decided_by,
        evidenceFragmentCount=len(saved.evidence_fragment_ids),
        auditEventId=event.id,
    )
    return saved


def _load_evidence(evidence_fragment_ids: list[str], store: InMemoryStore) -> list[EvidenceFragment]:
    evidence = []
    for evidence_id in evidence_fragment_ids:
        try:
            evidence.append(store.get_evidence_fragment(evidence_id))
        except NotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"EvidenceFragment not found: {evidence_id}",
            ) from exc
    return evidence


def _load_artifacts_for_evidence(evidence: list[EvidenceFragment], store: InMemoryStore) -> list[Artifact]:
    artifacts = []
    for fragment in evidence:
        try:
            artifacts.append(store.get_artifact(fragment.artifact_id))
        except NotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Artifact not found: {fragment.artifact_id}",
            ) from exc
    return artifacts


def _load_data_sources_for_artifacts(artifacts: list[Artifact], store: InMemoryStore) -> list[DataSource]:
    data_sources = []
    for artifact in artifacts:
        try:
            data_sources.append(store.get_data_source(artifact.datasource_id))
        except NotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"DataSource not found: {artifact.datasource_id}",
            ) from exc
    return data_sources


def _validate_affected_concepts_exist(concept_ids: list[str], store: InMemoryStore) -> None:
    for concept_id in concept_ids:
        try:
            store.get_concept(concept_id)
        except NotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Concept not found: {concept_id}",
            ) from exc
