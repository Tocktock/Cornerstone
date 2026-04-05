from __future__ import annotations

from sqlalchemy.orm import Session

from cornerstone.domain.models import Concept, ConceptRelation, DecisionRecord, EvidenceFragment
from cornerstone.domain.schemas import ConceptRead, DecisionRead, EvidenceRead, RelationRead


def evidence_read(evidence: EvidenceFragment) -> EvidenceRead:
    return EvidenceRead(
        id=evidence.id,
        selector=evidence.selector,
        excerpt=evidence.excerpt,
        normalized_claim=evidence.normalized_claim,
        verification_status=evidence.verification_status.value,
        artifact_id=evidence.artifact.id,
        artifact_title=evidence.artifact.title,
        artifact_url=evidence.artifact.canonical_url,
    )


def concept_read(_: Session, concept: Concept) -> ConceptRead:
    return ConceptRead(
        id=concept.id,
        context_space_id=concept.context_space_id,
        concept_type=concept.concept_type,
        canonical_name=concept.canonical_name,
        aliases=list(concept.aliases),
        definition=concept.definition,
        status=concept.status,
        evidence=[evidence_read(link.evidence_fragment) for link in concept.evidence_links],
        linked_decisions=[link.decision_record.title for link in concept.decision_links],
    )


def relation_read(_: Session, relation: ConceptRelation) -> RelationRead:
    return RelationRead(
        id=relation.id,
        context_space_id=relation.context_space_id,
        subject_concept_id=relation.subject_concept_id,
        subject_name=relation.subject_concept.canonical_name,
        predicate=relation.predicate,
        object_concept_id=relation.object_concept_id,
        object_name=relation.object_concept.canonical_name,
        description=relation.description,
        status=relation.status,
        evidence=[evidence_read(link.evidence_fragment) for link in relation.evidence_links],
        linked_decisions=[link.decision_record.title for link in relation.decision_links],
    )


def decision_read(_: Session, decision: DecisionRecord) -> DecisionRead:
    return DecisionRead(
        id=decision.id,
        context_space_id=decision.context_space_id,
        title=decision.title,
        problem=decision.problem,
        decision=decision.decision,
        rationale=decision.rationale,
        constraints=list(decision.constraints),
        impact=list(decision.impact),
        status=decision.status,
        evidence=[evidence_read(link.evidence_fragment) for link in decision.evidence_links],
        concepts=[link.concept.canonical_name for link in decision.concept_links],
        relations=[
            f"{link.concept_relation.subject_concept.canonical_name} {link.concept_relation.predicate} {link.concept_relation.object_concept.canonical_name}"
            for link in decision.relation_links
        ],
    )
