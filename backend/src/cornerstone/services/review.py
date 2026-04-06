from __future__ import annotations

from sqlalchemy.orm import Session

from cornerstone.domain.enums import ConsumerScope, ReviewAction
from cornerstone.domain.models import (
    Actor,
    Concept,
    ConceptRelation,
    DecisionRecord,
    VerificationPolicy,
)
from cornerstone.services.policies import (
    PolicyError,
    require_review_permission,
    support_visibility_for_consumer,
    transition_after_review,
)
from cornerstone.services.serialization import support_items_for_resource, visible_support_items


class ReviewInvariantError(PolicyError):
    pass


def _apply_review(
    session: Session,
    resource: Concept | ConceptRelation | DecisionRecord,
    actor: Actor,
    action: ReviewAction,
    *,
    policy: VerificationPolicy | None,
    supersedes_decision_id: str | None = None,
) -> None:
    try:
        require_review_permission(session, actor, resource, action)
        if isinstance(resource, DecisionRecord) and action is ReviewAction.SUPERSEDE:
            if supersedes_decision_id is None:
                raise ReviewInvariantError("Supersede requires a target decision to supersede.")
            resource.supersedes_decision_id = supersedes_decision_id
        transition_after_review(resource, action, policy=policy)
        support_items = support_items_for_resource(resource)
        resource.support_visibility = support_visibility_for_consumer(
            policy,
            support_items,
            visible_support_items(support_items, ConsumerScope.REVIEW),
        )
    except PolicyError as exc:
        raise ReviewInvariantError(str(exc)) from exc


def review_concept(
    session: Session,
    concept: Concept,
    actor: Actor,
    action: ReviewAction,
    *,
    policy: VerificationPolicy | None,
) -> Concept:
    _apply_review(session, concept, actor, action, policy=policy)
    return concept


def review_relation(
    session: Session,
    relation: ConceptRelation,
    actor: Actor,
    action: ReviewAction,
    *,
    policy: VerificationPolicy | None,
) -> ConceptRelation:
    _apply_review(session, relation, actor, action, policy=policy)
    return relation


def review_decision(
    session: Session,
    decision: DecisionRecord,
    actor: Actor,
    action: ReviewAction,
    *,
    policy: VerificationPolicy | None,
    supersedes_decision_id: str | None = None,
) -> DecisionRecord:
    _apply_review(
        session,
        decision,
        actor,
        action,
        policy=policy,
        supersedes_decision_id=supersedes_decision_id,
    )
    return decision
