from __future__ import annotations

from collections.abc import Iterable

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from cornerstone.domain.enums import (
    ConsumerScope,
    ContextSpaceKind,
    CuratedLifecycleState,
    DecisionLifecycleState,
    SupportVisibility,
)
from cornerstone.domain.models import (
    Concept,
    ConceptRelation,
    ContextSpace,
    DecisionRecord,
    VerificationPolicy,
)
from cornerstone.services.policies import (
    member_can_see_restricted_outputs,
    support_visibility_for_consumer,
)
from cornerstone.services.serialization import support_items_for_resource, visible_support_items


def get_context_space(session: Session, context_space_id: str | None = None) -> ContextSpace | None:
    if context_space_id:
        return session.get(ContextSpace, context_space_id)
    return session.scalar(
        select(ContextSpace)
        .where(ContextSpace.kind == ContextSpaceKind.WORKSPACE)
        .order_by(ContextSpace.created_at.asc())
    )


def get_policy(session: Session, context_space_id: str) -> VerificationPolicy | None:
    return session.scalar(
        select(VerificationPolicy)
        .where(VerificationPolicy.context_space_id == context_space_id)
        .limit(1)
    )


def is_published(resource: Concept | ConceptRelation | DecisionRecord) -> bool:
    if isinstance(resource, DecisionRecord):
        return resource.lifecycle_state in {
            DecisionLifecycleState.ACCEPTED,
            DecisionLifecycleState.SUPERSEDED,
        }
    return resource.lifecycle_state in {
        CuratedLifecycleState.OFFICIAL,
        CuratedLifecycleState.DEPRECATED,
    }


def resource_visible_to_consumer(
    resource: Concept | ConceptRelation | DecisionRecord,
    consumer_scope: ConsumerScope,
    policy: VerificationPolicy | None,
) -> bool:
    if consumer_scope is not ConsumerScope.MEMBER:
        return True
    if not is_published(resource):
        return False
    support_items = support_items_for_resource(resource)
    support_visibility = support_visibility_for_consumer(
        policy,
        support_items,
        visible_support_items(support_items, consumer_scope),
    )
    if support_visibility is SupportVisibility.RESTRICTED_SUPPORT:
        return member_can_see_restricted_outputs(policy)
    return True


def list_concepts(
    session: Session,
    context_space_id: str,
    *,
    query: str | None = None,
) -> list[Concept]:
    stmt = (
        select(Concept)
        .where(Concept.context_space_id == context_space_id)
        .order_by(Concept.canonical_name.asc())
    )
    if query:
        pattern = f"%{query}%"
        stmt = stmt.where(
            or_(Concept.canonical_name.ilike(pattern), Concept.definition.ilike(pattern))
        )
    return list(session.scalars(stmt).unique())


def list_relations(
    session: Session,
    context_space_id: str,
) -> list[ConceptRelation]:
    stmt = (
        select(ConceptRelation)
        .where(ConceptRelation.context_space_id == context_space_id)
        .order_by(ConceptRelation.created_at.desc())
    )
    return list(session.scalars(stmt).unique())


def list_decisions(
    session: Session,
    context_space_id: str,
    *,
    query: str | None = None,
) -> list[DecisionRecord]:
    stmt = (
        select(DecisionRecord)
        .where(DecisionRecord.context_space_id == context_space_id)
        .order_by(DecisionRecord.created_at.desc())
    )
    if query:
        pattern = f"%{query}%"
        stmt = stmt.where(
            or_(
                DecisionRecord.title.ilike(pattern),
                DecisionRecord.decision_statement.ilike(pattern),
            )
        )
    return list(session.scalars(stmt).unique())


def search_resources(
    session: Session,
    context_space_id: str,
    *,
    query: str,
) -> tuple[list[Concept], list[ConceptRelation], list[DecisionRecord]]:
    concepts = list_concepts(session, context_space_id, query=query)
    decisions = list_decisions(session, context_space_id, query=query)
    lowered = query.lower()
    relations = [
        relation
        for relation in list_relations(session, context_space_id)
        if lowered in relation.predicate.lower()
        or lowered in (relation.description or "").lower()
        or lowered in relation.subject_concept.canonical_name.lower()
        or lowered in relation.object_concept.canonical_name.lower()
    ]
    if concepts:
        concept_ids = {concept.id for concept in concepts}
        related_relations = [
            relation
            for relation in list_relations(session, context_space_id)
            if relation.subject_concept_id in concept_ids
            or relation.object_concept_id in concept_ids
        ]
        relation_ids = {relation.id for relation in relations}
        relations.extend(
            relation for relation in related_relations if relation.id not in relation_ids
        )
    return concepts, relations, decisions


def filter_visible_resources[T: Concept | ConceptRelation | DecisionRecord](
    resources: Iterable[T],
    consumer_scope: ConsumerScope,
    policy: VerificationPolicy | None,
) -> list[T]:
    return [
        resource
        for resource in resources
        if resource_visible_to_consumer(resource, consumer_scope, policy)
    ]
