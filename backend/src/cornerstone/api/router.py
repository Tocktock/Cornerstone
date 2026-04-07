from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from cornerstone.api.deps import get_consumer_scope, get_current_actor, get_db, get_settings
from cornerstone.config import Settings
from cornerstone.domain.enums import (
    BaseRole,
    ConnectorAction,
    ConsumerScope,
    ContextSpaceKind,
    RequestIntent,
    ResourceKind,
    ReviewAction,
)
from cornerstone.domain.models import (
    Actor,
    Concept,
    ConceptRelation,
    ContextSpace,
    DecisionRecord,
    SourceConnection,
)
from cornerstone.domain.schemas import (
    ActorSession,
    ConnectorTemplateSummary,
    DraftConceptCreate,
    DraftDecisionCreate,
    DraftRelationCreate,
    McpReadRequest,
    PromotionRequest,
    ProviderBindingCompleteRequest,
    ProviderBindingStartResponse,
    ProviderBindingSummary,
    ReviewActionRequest,
    ReviewQueueItem,
    SourceConnectionCreate,
    SourceConnectionDetail,
    SourceConnectionPreviewRequest,
    SourceConnectionPreviewResponse,
    SourceConnectionStatus,
    SourceConnectionUpdate,
    SyncRunResult,
    SyncRunSummary,
    ViewerBootstrap,
)
from cornerstone.services.answering import answer_query, search_context
from cornerstone.services.catalog import (
    filter_visible_resources,
    get_context_space,
    get_policy,
    list_concepts,
    list_decisions,
    list_relations,
    resource_visible_to_consumer,
)
from cornerstone.services.creation import (
    create_concept,
    create_decision,
    create_promoted_support,
    create_relation,
)
from cornerstone.services.review import (
    ReviewInvariantError,
    review_concept,
    review_decision,
    review_relation,
)
from cornerstone.services.serialization import (
    concept_envelope,
    context_space_ref,
    decision_envelope,
    graph_slice_envelope,
    provenance_envelope,
    relation_envelope,
    resource_ref,
    support_items_for_resource,
)
from cornerstone.services.source_connections import (
    actor_can_connector_action,
    complete_provider_binding,
    create_source_connection,
    list_connector_templates,
    list_sync_runs,
    pause_source_connection,
    preview_source_connection,
    remove_source_connection,
    resume_source_connection,
    source_connection_detail_for_actor,
    source_connection_status_for_actor,
    start_provider_binding,
    update_source_connection,
)
from cornerstone.services.sync import run_sync

router = APIRouter()


@router.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/bootstrap", response_model=ViewerBootstrap)
def bootstrap(db: Session = Depends(get_db)):
    workspace = db.scalar(
        select(ContextSpace).where(ContextSpace.kind == ContextSpaceKind.WORKSPACE).limit(1)
    )
    personal_context = db.scalar(
        select(ContextSpace).where(ContextSpace.kind == ContextSpaceKind.PERSONAL).limit(1)
    )
    if workspace is None or personal_context is None:
        raise HTTPException(status_code=503, detail="Bootstrap data not seeded.")
    actors = list(
        db.scalars(
            select(Actor).where(Actor.context_space_id == workspace.id).order_by(Actor.display_name)
        )
    )
    return ViewerBootstrap(
        workspace=context_space_ref(workspace),
        personal_context=context_space_ref(personal_context),
        actors=[
            ActorSession(
                actor_id=actor.id,
                display_name=actor.display_name,
                base_role=actor.base_role.value,
                token=actor.auth_token,
                scoped_capabilities=actor.scoped_capabilities,
                preferred_consumer_scope=actor.preferred_consumer_scope,
            )
            for actor in actors
        ],
    )


@router.get("/connector-templates", response_model=list[ConnectorTemplateSummary])
def get_connector_templates(actor: Actor = Depends(get_current_actor)):
    _ensure_connector_manager(actor)
    return list_connector_templates()


@router.post("/provider-bindings/notion/start", response_model=ProviderBindingStartResponse)
def start_notion_provider_binding(
    settings: Settings = Depends(get_settings),
    actor: Actor = Depends(get_current_actor),
    db: Session = Depends(get_db),
):
    try:
        result = start_provider_binding(db, actor=actor, settings=settings, provider="notion")
        db.commit()
        return result
    except PermissionError as exc:
        db.rollback()
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/provider-bindings/notion/complete", response_model=ProviderBindingSummary)
def complete_notion_provider_binding(
    payload: ProviderBindingCompleteRequest,
    settings: Settings = Depends(get_settings),
    actor: Actor = Depends(get_current_actor),
    db: Session = Depends(get_db),
):
    if payload.provider != "notion":
        raise HTTPException(status_code=400, detail="Route only supports the Notion provider.")
    try:
        result = complete_provider_binding(
            db,
            actor=actor,
            settings=settings,
            provider="notion",
            binding_state=payload.binding_state,
            code=payload.code,
        )
        db.commit()
        return result
    except PermissionError as exc:
        db.rollback()
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/source-connections", response_model=list[SourceConnectionStatus])
def get_source_connections(
    actor: Actor = Depends(get_current_actor),
    db: Session = Depends(get_db),
):
    connections = list(
        db.scalars(
            select(SourceConnection)
            .where(SourceConnection.context_space_id == actor.context_space_id)
            .order_by(SourceConnection.source_label.asc())
        )
    )
    return [
        source_connection_status_for_actor(db, actor=actor, connection=connection)
        for connection in connections
    ]


@router.post("/source-connections/preview", response_model=SourceConnectionPreviewResponse)
def post_source_connection_preview(
    payload: SourceConnectionPreviewRequest,
    settings: Settings = Depends(get_settings),
    actor: Actor = Depends(get_current_actor),
    db: Session = Depends(get_db),
):
    try:
        return preview_source_connection(db, actor=actor, settings=settings, payload=payload)
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post(
    "/source-connections",
    response_model=SourceConnectionDetail,
    status_code=status.HTTP_201_CREATED,
)
def post_source_connection(
    payload: SourceConnectionCreate,
    settings: Settings = Depends(get_settings),
    actor: Actor = Depends(get_current_actor),
    db: Session = Depends(get_db),
):
    try:
        connection = create_source_connection(db, actor=actor, settings=settings, payload=payload)
        db.commit()
        return source_connection_detail_for_actor(db, actor=actor, connection=connection)
    except PermissionError as exc:
        db.rollback()
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/source-connections/{connection_id}", response_model=SourceConnectionDetail)
def get_source_connection_detail(
    connection_id: str,
    actor: Actor = Depends(get_current_actor),
    db: Session = Depends(get_db),
):
    connection = _source_connection_for_actor_or_404(db, actor, connection_id)
    return source_connection_detail_for_actor(db, actor=actor, connection=connection)


@router.patch("/source-connections/{connection_id}", response_model=SourceConnectionDetail)
def patch_source_connection(
    connection_id: str,
    payload: SourceConnectionUpdate,
    settings: Settings = Depends(get_settings),
    actor: Actor = Depends(get_current_actor),
    db: Session = Depends(get_db),
):
    connection = _source_connection_for_actor_or_404(db, actor, connection_id)
    try:
        updated = update_source_connection(
            db,
            actor=actor,
            settings=settings,
            connection=connection,
            payload=payload,
        )
        db.commit()
        return source_connection_detail_for_actor(db, actor=actor, connection=updated)
    except PermissionError as exc:
        db.rollback()
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/source-connections/{connection_id}/pause", response_model=SourceConnectionDetail)
def post_pause_source_connection(
    connection_id: str,
    settings: Settings = Depends(get_settings),
    actor: Actor = Depends(get_current_actor),
    db: Session = Depends(get_db),
):
    connection = _source_connection_for_actor_or_404(db, actor, connection_id)
    try:
        paused = pause_source_connection(db, actor=actor, settings=settings, connection=connection)
        db.commit()
        return source_connection_detail_for_actor(db, actor=actor, connection=paused)
    except PermissionError as exc:
        db.rollback()
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.post("/source-connections/{connection_id}/resume", response_model=SourceConnectionDetail)
def post_resume_source_connection(
    connection_id: str,
    settings: Settings = Depends(get_settings),
    actor: Actor = Depends(get_current_actor),
    db: Session = Depends(get_db),
):
    connection = _source_connection_for_actor_or_404(db, actor, connection_id)
    try:
        resumed = resume_source_connection(
            db,
            actor=actor,
            settings=settings,
            connection=connection,
        )
        db.commit()
        return source_connection_detail_for_actor(db, actor=actor, connection=resumed)
    except PermissionError as exc:
        db.rollback()
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.get("/source-connections/{connection_id}/runs", response_model=list[SyncRunSummary])
def get_source_connection_runs(
    connection_id: str,
    actor: Actor = Depends(get_current_actor),
    db: Session = Depends(get_db),
):
    _source_connection_for_actor_or_404(db, actor, connection_id)
    return list_sync_runs(db, connection_id)


@router.post("/source-connections/{connection_id}/sync", response_model=SyncRunResult)
def sync_source_connection(
    connection_id: str,
    settings: Settings = Depends(get_settings),
    actor: Actor = Depends(get_current_actor),
    db: Session = Depends(get_db),
):
    connection = _source_connection_for_actor_or_404(db, actor, connection_id)
    if not actor_can_connector_action(
        db, actor, ConnectorAction.SYNC, context_space_id=connection.context_space_id
    ):
        raise HTTPException(status_code=403, detail="Actor lacks connector sync privileges.")
    result = run_sync(db, connection, settings, trigger_kind="manual")
    db.commit()
    return result


@router.delete("/source-connections/{connection_id}", response_model=SourceConnectionDetail)
def delete_source_connection(
    connection_id: str,
    settings: Settings = Depends(get_settings),
    actor: Actor = Depends(get_current_actor),
    db: Session = Depends(get_db),
):
    connection = _source_connection_for_actor_or_404(db, actor, connection_id)
    try:
        removed = remove_source_connection(
            db,
            actor=actor,
            settings=settings,
            connection=connection,
        )
        db.commit()
        return source_connection_detail_for_actor(db, actor=actor, connection=removed)
    except PermissionError as exc:
        db.rollback()
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@router.get("/concepts")
def get_concepts(
    q: str | None = Query(default=None),
    settings: Settings = Depends(get_settings),
    consumer_scope: ConsumerScope = Depends(get_consumer_scope),
    actor: Actor = Depends(get_current_actor),
    db: Session = Depends(get_db),
):
    context_space = _workspace_context_or_404(db, actor.context_space_id)
    policy = get_policy(db, context_space.id)
    concepts = filter_visible_resources(
        list_concepts(db, context_space.id, query=q), consumer_scope, policy
    )
    return [
        concept_envelope(settings, context_space, policy, concept, consumer_scope)
        for concept in concepts
    ]


@router.get("/concepts/{public_slug}")
def get_concept(
    public_slug: str,
    settings: Settings = Depends(get_settings),
    consumer_scope: ConsumerScope = Depends(get_consumer_scope),
    actor: Actor = Depends(get_current_actor),
    db: Session = Depends(get_db),
):
    context_space = _workspace_context_or_404(db, actor.context_space_id)
    policy = get_policy(db, context_space.id)
    concept = db.scalar(
        select(Concept)
        .where(Concept.context_space_id == context_space.id, Concept.public_slug == public_slug)
        .limit(1)
    )
    if concept is None or not resource_visible_to_consumer(concept, consumer_scope, policy):
        raise HTTPException(status_code=404, detail="Concept not found.")
    db.refresh(concept)
    return concept_envelope(settings, context_space, policy, concept, consumer_scope)


@router.post("/concepts")
def post_concept(
    payload: DraftConceptCreate,
    settings: Settings = Depends(get_settings),
    consumer_scope: ConsumerScope = Depends(get_consumer_scope),
    actor: Actor = Depends(get_current_actor),
    db: Session = Depends(get_db),
):
    _ensure_operator(actor)
    concept = create_concept(db, payload)
    db.commit()
    db.refresh(concept)
    context_space = _workspace_context_or_404(db, concept.context_space_id)
    return concept_envelope(
        settings, context_space, get_policy(db, context_space.id), concept, consumer_scope
    )


@router.post("/concepts/{concept_id}/review")
def review_concept_route(
    concept_id: str,
    payload: ReviewActionRequest,
    settings: Settings = Depends(get_settings),
    consumer_scope: ConsumerScope = Depends(get_consumer_scope),
    actor: Actor = Depends(get_current_actor),
    db: Session = Depends(get_db),
):
    concept = db.get(Concept, concept_id)
    if concept is None:
        raise HTTPException(status_code=404, detail="Concept not found.")
    context_space = _workspace_context_or_404(db, concept.context_space_id)
    policy = get_policy(db, context_space.id)
    try:
        review_concept(db, concept, actor, payload.action, policy=policy)
        db.commit()
        db.refresh(concept)
    except ReviewInvariantError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return concept_envelope(settings, context_space, policy, concept, consumer_scope)


@router.get("/relations")
def get_relations(
    settings: Settings = Depends(get_settings),
    consumer_scope: ConsumerScope = Depends(get_consumer_scope),
    actor: Actor = Depends(get_current_actor),
    db: Session = Depends(get_db),
):
    context_space = _workspace_context_or_404(db, actor.context_space_id)
    policy = get_policy(db, context_space.id)
    relations = filter_visible_resources(
        list_relations(db, context_space.id), consumer_scope, policy
    )
    return [
        relation_envelope(settings, context_space, policy, relation, consumer_scope)
        for relation in relations
    ]


@router.get("/relations/{relation_id}")
def get_relation(
    relation_id: str,
    settings: Settings = Depends(get_settings),
    consumer_scope: ConsumerScope = Depends(get_consumer_scope),
    actor: Actor = Depends(get_current_actor),
    db: Session = Depends(get_db),
):
    relation = db.get(ConceptRelation, relation_id)
    if relation is None:
        raise HTTPException(status_code=404, detail="Relation not found.")
    context_space = _workspace_context_or_404(db, relation.context_space_id)
    policy = get_policy(db, context_space.id)
    if not resource_visible_to_consumer(relation, consumer_scope, policy):
        raise HTTPException(status_code=404, detail="Relation not found.")
    return relation_envelope(settings, context_space, policy, relation, consumer_scope)


@router.post("/relations")
def post_relation(
    payload: DraftRelationCreate,
    settings: Settings = Depends(get_settings),
    consumer_scope: ConsumerScope = Depends(get_consumer_scope),
    actor: Actor = Depends(get_current_actor),
    db: Session = Depends(get_db),
):
    _ensure_operator(actor)
    relation = create_relation(db, payload)
    db.commit()
    db.refresh(relation)
    context_space = _workspace_context_or_404(db, relation.context_space_id)
    return relation_envelope(
        settings, context_space, get_policy(db, context_space.id), relation, consumer_scope
    )


@router.post("/relations/{relation_id}/review")
def review_relation_route(
    relation_id: str,
    payload: ReviewActionRequest,
    settings: Settings = Depends(get_settings),
    consumer_scope: ConsumerScope = Depends(get_consumer_scope),
    actor: Actor = Depends(get_current_actor),
    db: Session = Depends(get_db),
):
    relation = db.get(ConceptRelation, relation_id)
    if relation is None:
        raise HTTPException(status_code=404, detail="Relation not found.")
    context_space = _workspace_context_or_404(db, relation.context_space_id)
    policy = get_policy(db, context_space.id)
    try:
        review_relation(db, relation, actor, payload.action, policy=policy)
        db.commit()
        db.refresh(relation)
    except ReviewInvariantError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return relation_envelope(settings, context_space, policy, relation, consumer_scope)


@router.get("/decisions")
def get_decisions(
    q: str | None = Query(default=None),
    settings: Settings = Depends(get_settings),
    consumer_scope: ConsumerScope = Depends(get_consumer_scope),
    actor: Actor = Depends(get_current_actor),
    db: Session = Depends(get_db),
):
    context_space = _workspace_context_or_404(db, actor.context_space_id)
    policy = get_policy(db, context_space.id)
    decisions = filter_visible_resources(
        list_decisions(db, context_space.id, query=q), consumer_scope, policy
    )
    return [
        decision_envelope(db, settings, context_space, policy, decision, consumer_scope)
        for decision in decisions
    ]


@router.get("/decisions/{decision_id}")
def get_decision(
    decision_id: str,
    settings: Settings = Depends(get_settings),
    consumer_scope: ConsumerScope = Depends(get_consumer_scope),
    actor: Actor = Depends(get_current_actor),
    db: Session = Depends(get_db),
):
    decision = db.get(DecisionRecord, decision_id)
    if decision is None:
        raise HTTPException(status_code=404, detail="Decision not found.")
    context_space = _workspace_context_or_404(db, decision.context_space_id)
    policy = get_policy(db, context_space.id)
    if not resource_visible_to_consumer(decision, consumer_scope, policy):
        raise HTTPException(status_code=404, detail="Decision not found.")
    return decision_envelope(db, settings, context_space, policy, decision, consumer_scope)


@router.post("/decisions")
def post_decision(
    payload: DraftDecisionCreate,
    settings: Settings = Depends(get_settings),
    consumer_scope: ConsumerScope = Depends(get_consumer_scope),
    actor: Actor = Depends(get_current_actor),
    db: Session = Depends(get_db),
):
    _ensure_operator(actor)
    decision = create_decision(db, payload)
    db.commit()
    db.refresh(decision)
    context_space = _workspace_context_or_404(db, decision.context_space_id)
    return decision_envelope(
        db, settings, context_space, get_policy(db, context_space.id), decision, consumer_scope
    )


@router.post("/decisions/{decision_id}/review")
def review_decision_route(
    decision_id: str,
    payload: ReviewActionRequest,
    settings: Settings = Depends(get_settings),
    consumer_scope: ConsumerScope = Depends(get_consumer_scope),
    actor: Actor = Depends(get_current_actor),
    db: Session = Depends(get_db),
):
    decision = db.get(DecisionRecord, decision_id)
    if decision is None:
        raise HTTPException(status_code=404, detail="Decision not found.")
    context_space = _workspace_context_or_404(db, decision.context_space_id)
    policy = get_policy(db, context_space.id)
    try:
        review_decision(
            db,
            decision,
            actor,
            payload.action,
            policy=policy,
            supersedes_decision_id=payload.supersedes_decision_id,
        )
        db.commit()
        db.refresh(decision)
    except ReviewInvariantError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return decision_envelope(db, settings, context_space, policy, decision, consumer_scope)


@router.get("/review-queue", response_model=list[ReviewQueueItem])
def get_review_queue(
    actor: Actor = Depends(get_current_actor),
    db: Session = Depends(get_db),
):
    if (
        actor.base_role is not BaseRole.ADMIN
        and actor.preferred_consumer_scope is not ConsumerScope.REVIEW
    ):
        has_review_grant = db.scalar(
            select(Actor).where(Actor.id == actor.id, Actor.base_role == BaseRole.ADMIN)
        )
        if has_review_grant is None:
            raise HTTPException(status_code=403, detail="Review queue requires review scope.")
    context_space = _workspace_context_or_404(db, actor.context_space_id)
    concepts = [
        concept
        for concept in list_concepts(db, context_space.id)
        if not _is_published_or_verified(concept)
    ]
    relations = [
        relation
        for relation in list_relations(db, context_space.id)
        if not _is_published_or_verified(relation)
    ]
    decisions = [
        decision
        for decision in list_decisions(db, context_space.id)
        if not _is_published_or_verified(decision)
    ]
    return [
        ReviewQueueItem(
            resource_ref=resource_ref(resource),
            review_domain=resource.review_domain,
            lifecycle_state=resource.lifecycle_state,
            verification_state=resource.verification_state,
            support_visibility=resource.support_visibility,
            suggested_actions=[
                ReviewAction.OFFICIALIZE,
                ReviewAction.REJECT,
                ReviewAction.MARK_FOR_REVALIDATION,
            ],
        )
        for resource in [*concepts, *relations, *decisions]
    ]


@router.get("/search")
def search_route(
    q: str = Query(..., min_length=2),
    settings: Settings = Depends(get_settings),
    consumer_scope: ConsumerScope = Depends(get_consumer_scope),
    actor: Actor = Depends(get_current_actor),
    db: Session = Depends(get_db),
):
    context_space = _workspace_context_or_404(db, actor.context_space_id)
    return search_context(db, settings, context_space, consumer_scope, q)


@router.get("/answers")
def answer_route(
    q: str = Query(..., min_length=2),
    settings: Settings = Depends(get_settings),
    consumer_scope: ConsumerScope = Depends(get_consumer_scope),
    actor: Actor = Depends(get_current_actor),
    db: Session = Depends(get_db),
):
    context_space = _workspace_context_or_404(db, actor.context_space_id)
    return answer_query(db, settings, context_space, consumer_scope, q)


@router.get("/graph")
def graph_route(
    root: str | None = Query(default=None),
    settings: Settings = Depends(get_settings),
    consumer_scope: ConsumerScope = Depends(get_consumer_scope),
    actor: Actor = Depends(get_current_actor),
    db: Session = Depends(get_db),
):
    context_space = _workspace_context_or_404(db, actor.context_space_id)
    policy = get_policy(db, context_space.id)
    concepts = filter_visible_resources(list_concepts(db, context_space.id), consumer_scope, policy)
    relations = filter_visible_resources(
        list_relations(db, context_space.id), consumer_scope, policy
    )
    root_concepts = concepts
    if root:
        root_concepts = [
            concept for concept in concepts if concept.public_slug == root or concept.id == root
        ]
        if not root_concepts:
            raise HTTPException(status_code=404, detail="Graph root concept not found.")
        root_ids = {concept.id for concept in root_concepts}
        concepts = [
            concept
            for concept in concepts
            if concept.id in root_ids
            or any(
                relation.subject_concept_id == concept.id
                or relation.object_concept_id == concept.id
                for relation in relations
                if relation.subject_concept_id in root_ids or relation.object_concept_id in root_ids
            )
        ]
        relations = [
            relation
            for relation in relations
            if relation.subject_concept_id in root_ids or relation.object_concept_id in root_ids
        ]
    return graph_slice_envelope(
        settings,
        context_space,
        consumer_scope,
        root_concepts=root_concepts[:3] or concepts[:3],
        nodes=concepts[:8],
        edges=relations[:12],
        policy=policy,
    )


@router.get("/provenance/{resource_kind}/{resource_id}")
def provenance_route(
    resource_kind: ResourceKind,
    resource_id: str,
    settings: Settings = Depends(get_settings),
    consumer_scope: ConsumerScope = Depends(get_consumer_scope),
    actor: Actor = Depends(get_current_actor),
    db: Session = Depends(get_db),
):
    resource = _resource_by_kind_or_404(db, resource_kind, resource_id)
    context_space = _workspace_context_or_404(db, resource.context_space_id)
    policy = get_policy(db, context_space.id)
    if (
        not resource_visible_to_consumer(resource, consumer_scope, policy)
        and consumer_scope is ConsumerScope.MEMBER
    ):
        raise HTTPException(status_code=404, detail="Resource not found.")
    return provenance_envelope(
        settings=settings,
        context_space=context_space,
        consumer_scope=consumer_scope,
        subject=resource,
        support_items=support_items_for_resource(resource),
        policy=policy,
        verification_state=resource.verification_state,
    )


@router.post("/promotions")
def promote_support_route(
    payload: PromotionRequest,
    actor: Actor = Depends(get_current_actor),
    db: Session = Depends(get_db),
):
    promoted_support = create_promoted_support(db, payload, actor)
    db.commit()
    return {"promoted_support_id": promoted_support.id}


@router.post("/mcp/read")
def mcp_read(
    payload: McpReadRequest,
    settings: Settings = Depends(get_settings),
    consumer_scope: ConsumerScope = Depends(get_consumer_scope),
    actor: Actor = Depends(get_current_actor),
    db: Session = Depends(get_db),
):
    context_space = _workspace_context_or_404(db, actor.context_space_id)
    if payload.request_intent is RequestIntent.SEARCH_CONTEXT:
        if not payload.query:
            raise HTTPException(status_code=400, detail="MCP search requires query.")
        return search_context(db, settings, context_space, consumer_scope, payload.query)
    if payload.request_intent is RequestIntent.GET_ANSWER:
        if not payload.query:
            raise HTTPException(status_code=400, detail="MCP answer requires query.")
        return answer_query(db, settings, context_space, consumer_scope, payload.query)
    if payload.request_intent is RequestIntent.GET_GRAPH_SLICE:
        return graph_route(
            root=payload.root_concept_id,
            settings=settings,
            consumer_scope=consumer_scope,
            actor=actor,
            db=db,
        )
    if payload.request_intent is RequestIntent.FOLLOW_PROVENANCE:
        if payload.resource_kind is None or payload.resource_id is None:
            raise HTTPException(
                status_code=400, detail="MCP provenance requires resource reference."
            )
        return provenance_route(
            payload.resource_kind,
            payload.resource_id,
            settings=settings,
            consumer_scope=consumer_scope,
            actor=actor,
            db=db,
        )
    if payload.resource_kind is None or payload.resource_id is None:
        raise HTTPException(status_code=400, detail="MCP read requires a resource reference.")
    if payload.request_intent is RequestIntent.GET_CONCEPT:
        concept = _resource_by_kind_or_404(db, payload.resource_kind, payload.resource_id)
        return concept_envelope(
            settings, context_space, get_policy(db, context_space.id), concept, consumer_scope
        )
    if payload.request_intent is RequestIntent.GET_RELATION:
        relation = _resource_by_kind_or_404(db, payload.resource_kind, payload.resource_id)
        return relation_envelope(
            settings, context_space, get_policy(db, context_space.id), relation, consumer_scope
        )
    if payload.request_intent is RequestIntent.GET_DECISION:
        decision = _resource_by_kind_or_404(db, payload.resource_kind, payload.resource_id)
        return decision_envelope(
            db, settings, context_space, get_policy(db, context_space.id), decision, consumer_scope
        )
    raise HTTPException(status_code=400, detail="Unsupported MCP read request.")


def _workspace_context_or_404(db: Session, context_space_id: str) -> ContextSpace:
    context_space = get_context_space(db, context_space_id)
    if context_space is None:
        raise HTTPException(status_code=404, detail="Context space not found.")
    return context_space


def _ensure_operator(actor: Actor) -> None:
    if actor.base_role is BaseRole.ADMIN:
        return
    if any(entry.get("capability") == "operate" for entry in actor.scoped_capabilities):
        return
    raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Actor lacks operate scope.")


def _ensure_connector_manager(actor: Actor) -> None:
    if actor.base_role is BaseRole.ADMIN:
        return
    if any(
        entry.get("capability") == "manage_connectors" and entry.get("scope") == "workspace"
        for entry in actor.scoped_capabilities
    ):
        return
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Actor lacks connector manager scope.",
    )


def _source_connection_for_actor_or_404(
    db: Session, actor: Actor, connection_id: str
) -> SourceConnection:
    connection = db.get(SourceConnection, connection_id)
    if connection is None:
        raise HTTPException(status_code=404, detail="Source connection not found.")
    if connection.context_space_id != actor.context_space_id:
        raise HTTPException(
            status_code=403, detail="Source connection is outside the actor workspace."
        )
    return connection


def _resource_by_kind_or_404(db: Session, resource_kind: ResourceKind, resource_id: str):
    if resource_kind is ResourceKind.CONCEPT:
        resource = db.get(Concept, resource_id)
    elif resource_kind is ResourceKind.RELATION:
        resource = db.get(ConceptRelation, resource_id)
    elif resource_kind is ResourceKind.DECISION:
        resource = db.get(DecisionRecord, resource_id)
    else:
        raise HTTPException(status_code=400, detail="Unsupported resource kind for this route.")
    if resource is None:
        raise HTTPException(status_code=404, detail="Resource not found.")
    return resource


def _is_published_or_verified(resource) -> bool:
    if resource.verification_state.value == "review_required":
        return False
    if hasattr(resource, "lifecycle_state"):
        return resource.lifecycle_state.value in {"official", "accepted", "superseded"}
    return False
