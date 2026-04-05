from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from cornerstone.domain.enums import DecisionConceptRole, DecisionRelationRole
from cornerstone.domain.models import (
    Concept,
    ConceptEvidenceLink,
    ConceptRelation,
    DecisionConceptLink,
    DecisionEvidenceLink,
    DecisionRecord,
    DecisionRelationLink,
    RelationEvidenceLink,
)
from cornerstone.domain.schemas import ConceptCreate, DecisionCreate, RelationCreate
from cornerstone.services.normalization import normalize_key


def _link_evidence(entity, evidence_ids: list[str], link_cls, relation_attr: str) -> None:
    existing = {getattr(link, "evidence_fragment_id") for link in getattr(entity, relation_attr)}
    for evidence_id in evidence_ids:
        if evidence_id not in existing:
            getattr(entity, relation_attr).append(link_cls(evidence_fragment_id=evidence_id))


def create_concept(db: Session, payload: ConceptCreate) -> Concept:
    concept = Concept(
        context_space_id=payload.context_space_id,
        concept_type=payload.concept_type,
        canonical_name=payload.canonical_name,
        canonical_key=normalize_key(payload.canonical_name),
        aliases=payload.aliases,
        definition=payload.definition,
        owner_actor_id=payload.owner_actor_id,
    )
    db.add(concept)
    db.flush()
    _link_evidence(concept, payload.evidence_fragment_ids, ConceptEvidenceLink, "evidence_links")
    existing_decisions = {
        link.decision_id for link in concept.decision_links
    }
    for decision_id in payload.linked_decision_ids:
        if decision_id not in existing_decisions:
            concept.decision_links.append(
                DecisionConceptLink(decision_id=decision_id, relationship_type=DecisionConceptRole.ABOUT)
            )
    return concept


def create_relation(db: Session, payload: RelationCreate) -> ConceptRelation:
    relation = ConceptRelation(
        context_space_id=payload.context_space_id,
        subject_concept_id=payload.subject_concept_id,
        predicate=payload.predicate,
        object_concept_id=payload.object_concept_id,
        description=payload.description,
    )
    db.add(relation)
    db.flush()
    _link_evidence(relation, payload.evidence_fragment_ids, RelationEvidenceLink, "evidence_links")
    existing_decisions = {link.decision_id for link in relation.decision_links}
    for decision_id in payload.linked_decision_ids:
        if decision_id not in existing_decisions:
            relation.decision_links.append(
                DecisionRelationLink(decision_id=decision_id, relationship_type=DecisionRelationRole.ABOUT)
            )
    return relation


def create_decision(db: Session, payload: DecisionCreate) -> DecisionRecord:
    decision = DecisionRecord(
        context_space_id=payload.context_space_id,
        title=payload.title,
        title_key=normalize_key(payload.title),
        problem=payload.problem,
        decision=payload.decision,
        rationale=payload.rationale,
        constraints=payload.constraints,
        impact=payload.impact,
    )
    db.add(decision)
    db.flush()
    for evidence_id in payload.evidence_fragment_ids:
        decision.evidence_links.append(DecisionEvidenceLink(evidence_fragment_id=evidence_id))
    for concept_id in payload.linked_concept_ids:
        decision.concept_links.append(
            DecisionConceptLink(concept_id=concept_id, relationship_type=DecisionConceptRole.ABOUT)
        )
    for relation_id in payload.linked_relation_ids:
        decision.relation_links.append(
            DecisionRelationLink(relation_id=relation_id, relationship_type=DecisionRelationRole.ABOUT)
        )
    return decision


def get_concept_by_name(db: Session, context_space_id: str, canonical_name: str) -> Concept | None:
    key = normalize_key(canonical_name)
    return db.scalar(
        select(Concept).where(Concept.context_space_id == context_space_id, Concept.canonical_key == key)
    )
