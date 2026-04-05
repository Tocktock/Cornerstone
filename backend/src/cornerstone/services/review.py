from __future__ import annotations

from sqlalchemy.orm import Session

from cornerstone.domain.enums import ConceptStatus, DecisionStatus, RelationStatus
from cornerstone.domain.models import Actor, Concept, ConceptRelation, DecisionRecord


class ReviewInvariantError(ValueError):
    pass


REVIEWER_ROLES = {"admin", "reviewer"}


def _ensure_reviewer(actor: Actor) -> None:
    if not REVIEWER_ROLES.intersection({role.lower() for role in actor.roles}):
        raise ReviewInvariantError("Actor lacks reviewer privileges.")


def _has_accepted_decision_link(entity: Concept | ConceptRelation) -> bool:
    if isinstance(entity, Concept):
        return any(link.decision_record.status == DecisionStatus.ACCEPTED for link in entity.decision_links)
    direct_links = any(link.decision_record.status == DecisionStatus.ACCEPTED for link in entity.decision_links)
    introduced_by = entity.introduced_by_decision and entity.introduced_by_decision.status == DecisionStatus.ACCEPTED
    return direct_links or introduced_by


def _has_grounding(entity: Concept | ConceptRelation | DecisionRecord) -> bool:
    if isinstance(entity, Concept):
        return bool(entity.evidence_links) or _has_accepted_decision_link(entity)
    if isinstance(entity, ConceptRelation):
        return bool(entity.evidence_links) or _has_accepted_decision_link(entity)
    return bool(entity.evidence_links)


def review_concept(_: Session, concept: Concept, actor: Actor, action: str) -> Concept:
    _ensure_reviewer(actor)
    normalized = action.lower()
    if normalized == "approve":
        if not _has_grounding(concept):
            raise ReviewInvariantError("Concept cannot become official without evidence or decision lineage.")
        concept.status = ConceptStatus.OFFICIAL
    elif normalized == "reject":
        concept.status = ConceptStatus.REJECTED
    elif normalized == "deprecate":
        concept.status = ConceptStatus.DEPRECATED
    else:
        raise ReviewInvariantError(f"Unsupported concept action: {action}")
    return concept


def review_relation(_: Session, relation: ConceptRelation, actor: Actor, action: str) -> ConceptRelation:
    _ensure_reviewer(actor)
    normalized = action.lower()
    if normalized == "approve":
        if not _has_grounding(relation):
            raise ReviewInvariantError("Relation cannot become official without evidence or accepted decision lineage.")
        relation.status = RelationStatus.OFFICIAL
    elif normalized == "reject":
        relation.status = RelationStatus.REJECTED
    elif normalized == "deprecate":
        relation.status = RelationStatus.DEPRECATED
    else:
        raise ReviewInvariantError(f"Unsupported relation action: {action}")
    return relation


def review_decision(_: Session, decision: DecisionRecord, actor: Actor, action: str) -> DecisionRecord:
    _ensure_reviewer(actor)
    normalized = action.lower()
    if normalized == "approve":
        if not _has_grounding(decision):
            raise ReviewInvariantError("Decision cannot be accepted without evidence.")
        decision.status = DecisionStatus.ACCEPTED
    elif normalized == "reject":
        decision.status = DecisionStatus.REJECTED
    elif normalized == "supersede":
        decision.status = DecisionStatus.SUPERSEDED
    else:
        raise ReviewInvariantError(f"Unsupported decision action: {action}")
    return decision
