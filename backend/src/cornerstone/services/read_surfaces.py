from __future__ import annotations

from fastapi import HTTPException
from sqlalchemy.orm import Session

from cornerstone.config import Settings
from cornerstone.domain.enums import ConsumerScope, ResourceKind, VerificationState, VisibilityClass
from cornerstone.domain.models import (
    Artifact,
    Concept,
    ConceptRelation,
    ContextSpace,
    DecisionRecord,
    SupportItem,
)
from cornerstone.services.catalog import (
    filter_visible_resources,
    get_context_space,
    get_policy,
    list_concepts,
    list_relations,
    resource_visible_to_consumer,
)
from cornerstone.services.serialization import (
    graph_slice_envelope,
    provenance_envelope,
    support_items_for_resource,
)

ReadSubject = Concept | ConceptRelation | DecisionRecord | Artifact | SupportItem


def context_space_or_404(db: Session, context_space_id: str) -> ContextSpace:
    context_space = get_context_space(db, context_space_id)
    if context_space is None:
        raise HTTPException(status_code=404, detail="Context space not found.")
    return context_space


def resource_by_kind_or_404(
    db: Session, resource_kind: ResourceKind, resource_id: str
) -> ReadSubject:
    if resource_kind is ResourceKind.CONCEPT:
        resource = db.get(Concept, resource_id)
    elif resource_kind is ResourceKind.RELATION:
        resource = db.get(ConceptRelation, resource_id)
    elif resource_kind is ResourceKind.DECISION:
        resource = db.get(DecisionRecord, resource_id)
    elif resource_kind is ResourceKind.ARTIFACT:
        resource = db.get(Artifact, resource_id)
    elif resource_kind is ResourceKind.SUPPORT_ITEM:
        resource = db.get(SupportItem, resource_id)
    else:  # pragma: no cover - ResourceKind is exhaustive
        resource = None
    if resource is None:
        raise HTTPException(status_code=404, detail="Resource not found.")
    return resource


def graph_slice_response(
    db: Session,
    settings: Settings,
    context_space: ContextSpace,
    consumer_scope: ConsumerScope,
    *,
    root: str | None = None,
):
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


def provenance_response(
    db: Session,
    settings: Settings,
    consumer_scope: ConsumerScope,
    *,
    resource_kind: ResourceKind,
    resource_id: str,
):
    resource = resource_by_kind_or_404(db, resource_kind, resource_id)
    context_space = context_space_or_404(db, resource.context_space_id)
    policy = get_policy(db, context_space.id)
    if not provenance_subject_visible_to_consumer(resource, consumer_scope, policy):
        raise HTTPException(status_code=404, detail="Resource not found.")
    return provenance_envelope(
        settings=settings,
        context_space=context_space,
        consumer_scope=consumer_scope,
        subject=resource,
        support_items=support_items_for_subject(resource),
        policy=policy,
        verification_state=verification_state_for_subject(resource),
    )


def provenance_subject_visible_to_consumer(
    resource: ReadSubject,
    consumer_scope: ConsumerScope,
    policy,
) -> bool:
    if consumer_scope is not ConsumerScope.MEMBER:
        return True
    if isinstance(resource, (Concept, ConceptRelation, DecisionRecord)):
        return resource_visible_to_consumer(resource, consumer_scope, policy)
    return resource.visibility_class is VisibilityClass.MEMBER_VISIBLE


def support_items_for_subject(resource: ReadSubject) -> list[SupportItem]:
    if isinstance(resource, Artifact):
        return list(resource.support_items)
    if isinstance(resource, SupportItem):
        return [resource]
    return support_items_for_resource(resource)


def verification_state_for_subject(resource: ReadSubject) -> VerificationState | None:
    if isinstance(resource, (Concept, ConceptRelation, DecisionRecord)):
        return resource.verification_state
    return None
