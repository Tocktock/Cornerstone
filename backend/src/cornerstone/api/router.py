from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from cornerstone.api.deps import get_db
from cornerstone.domain.models import (
    Actor,
    Artifact,
    Concept,
    ConceptRelation,
    ContextSpace,
    DecisionRecord,
    EvidenceFragment,
    SourceConnection,
)
from cornerstone.domain.schemas import (
    ActorRead,
    ArtifactRead,
    ConceptCreate,
    ConceptRead,
    ContextSpaceRead,
    DecisionCreate,
    DecisionRead,
    GraphEdge,
    GraphNode,
    GraphResponse,
    RelationCreate,
    RelationRead,
    ReviewActionRequest,
    SourceConnectionRead,
    StructuredAnswerResponse,
    SyncRunResult,
)
from cornerstone.services.answering import answer_query
from cornerstone.services.creation import create_concept, create_decision, create_relation
from cornerstone.services.review import ReviewInvariantError, review_concept, review_decision, review_relation
from cornerstone.services.serialization import concept_read, decision_read, relation_read
from cornerstone.services.sync import run_sync

router = APIRouter()


@router.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/context-spaces", response_model=list[ContextSpaceRead])
def list_context_spaces(db: Session = Depends(get_db)):
    return list(db.scalars(select(ContextSpace).order_by(ContextSpace.name)))


@router.get("/actors", response_model=list[ActorRead])
def list_actors(context_space_id: str | None = None, db: Session = Depends(get_db)):
    stmt = select(Actor).order_by(Actor.display_name)
    if context_space_id:
        stmt = stmt.where(Actor.context_space_id == context_space_id)
    return list(db.scalars(stmt))


@router.get("/source-connections", response_model=list[SourceConnectionRead])
def list_source_connections(context_space_id: str | None = None, db: Session = Depends(get_db)):
    stmt = select(SourceConnection).order_by(SourceConnection.created_at.desc())
    if context_space_id:
        stmt = stmt.where(SourceConnection.context_space_id == context_space_id)
    return list(db.scalars(stmt))


@router.post("/source-connections/{connection_id}/sync", response_model=SyncRunResult)
def sync_source_connection(connection_id: str, db: Session = Depends(get_db)):
    connection = db.get(SourceConnection, connection_id)
    if connection is None:
        raise HTTPException(status_code=404, detail="Source connection not found.")
    result = run_sync(db, connection)
    db.commit()
    return result


@router.get("/artifacts", response_model=list[ArtifactRead])
def list_artifacts(context_space_id: str | None = None, db: Session = Depends(get_db)):
    stmt = select(Artifact).order_by(Artifact.created_at.desc())
    if context_space_id:
        stmt = stmt.where(Artifact.context_space_id == context_space_id)
    artifacts = list(db.scalars(stmt))
    return [
        ArtifactRead(
            id=artifact.id,
            context_space_id=artifact.context_space_id,
            source_connection_id=artifact.source_connection_id,
            external_id=artifact.external_id,
            artifact_type=artifact.artifact_type,
            title=artifact.title,
            canonical_url=artifact.canonical_url,
            status=artifact.status.value,
            evidence_count=len(artifact.evidence_fragments),
            metadata_json=artifact.metadata_json,
        )
        for artifact in artifacts
    ]


@router.get("/concepts", response_model=list[ConceptRead])
def list_concepts(
    context_space_id: str | None = None,
    status: str | None = None,
    q: str | None = None,
    db: Session = Depends(get_db),
):
    stmt = select(Concept).order_by(Concept.canonical_name)
    if context_space_id:
        stmt = stmt.where(Concept.context_space_id == context_space_id)
    if status:
        stmt = stmt.where(Concept.status == status.upper())
    if q:
        pattern = f"%{q}%"
        stmt = stmt.where(or_(Concept.canonical_name.ilike(pattern), Concept.definition.ilike(pattern)))
    concepts = list(db.scalars(stmt).unique())
    return [concept_read(db, concept) for concept in concepts]


@router.post("/concepts", response_model=ConceptRead)
def post_concept(payload: ConceptCreate, db: Session = Depends(get_db)):
    concept = create_concept(db, payload)
    db.commit()
    db.refresh(concept)
    return concept_read(db, concept)


@router.post("/concepts/{concept_id}/review", response_model=ConceptRead)
def review_concept_route(concept_id: str, payload: ReviewActionRequest, db: Session = Depends(get_db)):
    concept = db.get(Concept, concept_id)
    actor = db.get(Actor, payload.actor_id)
    if concept is None or actor is None:
        raise HTTPException(status_code=404, detail="Concept or actor not found.")
    try:
        review_concept(db, concept, actor, payload.action)
        db.commit()
        db.refresh(concept)
    except ReviewInvariantError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return concept_read(db, concept)


@router.get("/relations", response_model=list[RelationRead])
def list_relations(context_space_id: str | None = None, status: str | None = None, db: Session = Depends(get_db)):
    stmt = select(ConceptRelation).order_by(ConceptRelation.created_at.desc())
    if context_space_id:
        stmt = stmt.where(ConceptRelation.context_space_id == context_space_id)
    if status:
        stmt = stmt.where(ConceptRelation.status == status.upper())
    relations = list(db.scalars(stmt))
    return [relation_read(db, relation) for relation in relations]


@router.post("/relations", response_model=RelationRead)
def post_relation(payload: RelationCreate, db: Session = Depends(get_db)):
    relation = create_relation(db, payload)
    db.commit()
    db.refresh(relation)
    return relation_read(db, relation)


@router.post("/relations/{relation_id}/review", response_model=RelationRead)
def review_relation_route(relation_id: str, payload: ReviewActionRequest, db: Session = Depends(get_db)):
    relation = db.get(ConceptRelation, relation_id)
    actor = db.get(Actor, payload.actor_id)
    if relation is None or actor is None:
        raise HTTPException(status_code=404, detail="Relation or actor not found.")
    try:
        review_relation(db, relation, actor, payload.action)
        db.commit()
        db.refresh(relation)
    except ReviewInvariantError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return relation_read(db, relation)


@router.get("/decisions", response_model=list[DecisionRead])
def list_decisions(context_space_id: str | None = None, status: str | None = None, db: Session = Depends(get_db)):
    stmt = select(DecisionRecord).order_by(DecisionRecord.created_at.desc())
    if context_space_id:
        stmt = stmt.where(DecisionRecord.context_space_id == context_space_id)
    if status:
        stmt = stmt.where(DecisionRecord.status == status.upper())
    decisions = list(db.scalars(stmt))
    return [decision_read(db, decision) for decision in decisions]


@router.post("/decisions", response_model=DecisionRead)
def post_decision(payload: DecisionCreate, db: Session = Depends(get_db)):
    decision = create_decision(db, payload)
    db.commit()
    db.refresh(decision)
    return decision_read(db, decision)


@router.post("/decisions/{decision_id}/review", response_model=DecisionRead)
def review_decision_route(decision_id: str, payload: ReviewActionRequest, db: Session = Depends(get_db)):
    decision = db.get(DecisionRecord, decision_id)
    actor = db.get(Actor, payload.actor_id)
    if decision is None or actor is None:
        raise HTTPException(status_code=404, detail="Decision or actor not found.")
    try:
        review_decision(db, decision, actor, payload.action)
        db.commit()
        db.refresh(decision)
    except ReviewInvariantError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return decision_read(db, decision)


@router.get("/graph", response_model=GraphResponse)
def get_graph(context_space_id: str = Query(...), db: Session = Depends(get_db)):
    concepts = list(db.scalars(select(Concept).where(Concept.context_space_id == context_space_id).order_by(Concept.canonical_name)))
    relations = list(
        db.scalars(
            select(ConceptRelation)
            .where(ConceptRelation.context_space_id == context_space_id)
            .order_by(ConceptRelation.created_at.desc())
        )
    )
    nodes = [
        GraphNode(id=concept.id, label=concept.canonical_name, type=concept.concept_type.value, status=concept.status.value)
        for concept in concepts
    ]
    edges = [
        GraphEdge(
            id=relation.id,
            source=relation.subject_concept_id,
            target=relation.object_concept_id,
            label=relation.predicate,
            status=relation.status.value,
        )
        for relation in relations
    ]
    return GraphResponse(nodes=nodes, edges=edges)


@router.get("/answers", response_model=StructuredAnswerResponse)
def get_answer(
    request: Request,
    q: str = Query(..., min_length=2),
    context_space_id: str | None = None,
    db: Session = Depends(get_db),
):
    return answer_query(db, q, context_space_id, request.app.state.settings)


@router.get("/stats")
def get_stats(context_space_id: str = Query(...), db: Session = Depends(get_db)):
    counts = {
        "concept_count": db.scalar(
            select(func.count()).select_from(Concept).where(Concept.context_space_id == context_space_id)
        )
        or 0,
        "relation_count": db.scalar(
            select(func.count()).select_from(ConceptRelation).where(ConceptRelation.context_space_id == context_space_id)
        )
        or 0,
        "decision_count": db.scalar(
            select(func.count()).select_from(DecisionRecord).where(DecisionRecord.context_space_id == context_space_id)
        )
        or 0,
        "artifact_count": db.scalar(
            select(func.count()).select_from(Artifact).where(Artifact.context_space_id == context_space_id)
        )
        or 0,
        "evidence_count": db.scalar(
            select(func.count())
            .select_from(EvidenceFragment)
            .join(Artifact, Artifact.id == EvidenceFragment.artifact_id)
            .where(Artifact.context_space_id == context_space_id)
        )
        or 0,
    }
    return counts
