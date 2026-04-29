from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status

from cornerstone.api.dependencies import get_store
from cornerstone.config import Settings, get_settings
from cornerstone.observability import log_event
from cornerstone.schemas import (
    AuditEvent,
    Concept,
    ConceptStatus,
    CreateConceptFromEvidenceRequest,
    EvidenceFragment,
    EvidenceFragmentType,
    EvidenceReviewQueueItem,
    EvidenceReviewQueueResponse,
    FreshnessState,
    ReviewEvidenceRequest,
    TrustState,
    utc_now,
)
from cornerstone.services.officialization import (
    ReviewerAuthorizationError,
    ensure_reviewer_authorized,
)
from cornerstone.store import InMemoryStore, NotFoundError

router = APIRouter(prefix="/evidence", tags=["evidence"])


@router.get("", response_model=list[EvidenceFragment])
def list_evidence(
    artifact_id: str | None = Query(default=None, alias="artifactId"),
    store: InMemoryStore = Depends(get_store),
) -> list[EvidenceFragment]:
    fragments = store.list_evidence_fragments(artifact_id=artifact_id)
    log_event("evidence.listed", artifactId=artifact_id, evidenceFragmentCount=len(fragments))
    return fragments


@router.get("/review-queue", response_model=EvidenceReviewQueueResponse)
def get_evidence_review_queue(
    trust_state: TrustState | None = Query(default=TrustState.UNREVIEWED, alias="trustState"),
    datasource_id: str | None = Query(default=None, alias="dataSourceId"),
    freshness_state: FreshnessState | None = Query(default=None, alias="freshnessState"),
    fragment_type: EvidenceFragmentType | None = Query(default=None, alias="fragmentType"),
    limit: int = Query(default=50, ge=1, le=250),
    store: InMemoryStore = Depends(get_store),
) -> EvidenceReviewQueueResponse:
    fragments = store.list_evidence_fragments()
    if trust_state is not None:
        fragments = [item for item in fragments if item.trust_state == trust_state]
    if freshness_state is not None:
        fragments = [item for item in fragments if item.freshness_state == freshness_state]
    if fragment_type is not None:
        fragments = [item for item in fragments if item.fragment_type == fragment_type]

    concepts = store.list_concepts()
    decision_records = store.list_decision_records()
    items: list[EvidenceReviewQueueItem] = []
    for fragment in fragments:
        try:
            artifact = store.get_artifact(fragment.artifact_id)
            data_source = store.get_data_source(artifact.datasource_id)
        except NotFoundError:
            continue
        if datasource_id is not None and data_source.id != datasource_id:
            continue
        linked_concept_ids = [
            concept.id for concept in concepts if fragment.id in concept.evidence_fragment_ids
        ]
        linked_decision_record_ids = [
            record.id for record in decision_records if fragment.id in record.evidence_fragment_ids
        ]
        suggested_next_actions = _suggest_review_actions(fragment, linked_concept_ids)
        items.append(
            EvidenceReviewQueueItem(
                evidence_fragment=fragment,
                artifact=artifact,
                data_source=data_source,
                linked_concept_ids=linked_concept_ids,
                linked_decision_record_ids=linked_decision_record_ids,
                suggested_next_actions=suggested_next_actions,
            )
        )
    items.sort(key=lambda item: (item.evidence_fragment.reviewed_at or item.evidence_fragment.provenance.captured_at, item.evidence_fragment.id))
    limited_items = items[:limit]
    all_fragments = store.list_evidence_fragments()
    response = EvidenceReviewQueueResponse(
        items=limited_items,
        total_count=len(items),
        unreviewed_count=sum(1 for item in all_fragments if item.trust_state == TrustState.UNREVIEWED),
        reviewed_count=sum(1 for item in all_fragments if item.trust_state == TrustState.REVIEWED),
        rejected_count=sum(1 for item in all_fragments if item.trust_state == TrustState.REJECTED),
        conflicted_count=sum(1 for item in all_fragments if item.trust_state == TrustState.CONFLICTED),
    )
    log_event(
        "evidence.review_queue_listed",
        trustState=trust_state,
        dataSourceId=datasource_id,
        itemCount=len(limited_items),
        totalCount=response.total_count,
    )
    return response


@router.post("/{evidence_fragment_id}/concept-candidates", response_model=Concept, status_code=status.HTTP_201_CREATED)
def create_concept_candidate_from_evidence(
    evidence_fragment_id: str,
    request: CreateConceptFromEvidenceRequest,
    store: InMemoryStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> Concept:
    try:
        ensure_reviewer_authorized(request.created_by, settings.authorized_reviewers)
    except ReviewerAuthorizationError as exc:
        log_event(
            "concept.create_from_evidence_blocked",
            evidenceFragmentId=evidence_fragment_id,
            actor=request.created_by,
            reason="reviewer_not_authorized",
        )
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
    try:
        fragment = store.get_evidence_fragment(evidence_fragment_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    concept = Concept(
        name=request.name,
        short_definition=request.short_definition,
        body=request.body,
        owner=request.owner,
        status=ConceptStatus(request.status),
        evidence_fragment_ids=[fragment.id],
        created_by=request.created_by,
    )
    event = AuditEvent(
        event_type="concept.candidate_created_from_evidence",
        actor=request.created_by,
        entity_type="Concept",
        entity_id=concept.id,
        metadata={"evidenceFragmentId": fragment.id, "conceptName": concept.name},
    )
    with store.transaction():
        saved = store.add_concept(concept)
        store.add_audit_event(event)
    log_event(
        "concept.candidate_created_from_evidence",
        conceptId=saved.id,
        evidenceFragmentId=fragment.id,
        actor=request.created_by,
        auditEventId=event.id,
    )
    return saved


@router.post("/{evidence_fragment_id}/review", response_model=EvidenceFragment)
def review_evidence(
    evidence_fragment_id: str,
    request: ReviewEvidenceRequest,
    store: InMemoryStore = Depends(get_store),
    settings: Settings = Depends(get_settings),
) -> EvidenceFragment:
    try:
        ensure_reviewer_authorized(request.reviewed_by, settings.authorized_reviewers)
    except ReviewerAuthorizationError as exc:
        log_event(
            "evidence.review_blocked",
            evidenceFragmentId=evidence_fragment_id,
            actor=request.reviewed_by,
            reason="reviewer_not_authorized",
        )
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc

    try:
        fragment = store.get_evidence_fragment(evidence_fragment_id)
    except NotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    now = utc_now()
    updated = fragment.model_copy(
        update={
            "trust_state": request.trust_state,
            "reviewed_by": request.reviewed_by,
            "reviewed_at": now,
        },
        deep=True,
    )
    event = AuditEvent(
        event_type="evidence.reviewed",
        actor=request.reviewed_by,
        entity_type="EvidenceFragment",
        entity_id=updated.id,
        metadata={
            "trustState": updated.trust_state,
            "artifactId": updated.artifact_id,
            "reviewNote": request.review_note,
        },
    )
    with store.transaction():
        saved = store.update_evidence_fragment(updated)
        store.add_audit_event(event)
    log_event(
        "evidence.reviewed",
        evidenceFragmentId=saved.id,
        trustState=saved.trust_state,
        actor=request.reviewed_by,
        auditEventId=event.id,
    )
    return saved


def _suggest_review_actions(fragment: EvidenceFragment, linked_concept_ids: list[str]) -> list[str]:
    if fragment.trust_state == TrustState.UNREVIEWED:
        actions = ["review_evidence", "reject_evidence"]
        if not linked_concept_ids:
            actions.append("create_concept_candidate")
        else:
            actions.append("attach_to_existing_concept")
        return actions
    if fragment.trust_state == TrustState.REVIEWED and not linked_concept_ids:
        return ["create_concept_candidate"]
    if fragment.trust_state == TrustState.CONFLICTED:
        return ["resolve_conflict", "create_decision_record"]
    return []
