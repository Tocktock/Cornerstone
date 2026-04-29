from __future__ import annotations

from collections.abc import Iterable

from cornerstone.schemas import (
    Artifact,
    AuditEvent,
    Concept,
    ConceptRelation,
    ConceptStatus,
    DataSource,
    DecisionRecord,
    EvidenceFragment,
    FreshnessState,
    RelationStatus,
    TrustState,
    utc_now,
)
from cornerstone.services.source_eligibility import source_can_officialize


class OfficializationError(ValueError):
    """Raised when a Concept cannot become official."""


class ReviewerAuthorizationError(PermissionError):
    """Raised when an actor is not allowed to perform reviewer actions."""


_OFFICIAL_ELIGIBLE_FRESHNESS = {FreshnessState.FRESH, FreshnessState.AGING}


def ensure_reviewer_authorized(reviewed_by: str, authorized_reviewers: set[str]) -> None:
    if reviewed_by not in authorized_reviewers:
        raise ReviewerAuthorizationError(f"Reviewer is not authorized: {reviewed_by}")


def ensure_evidence_is_officialization_ready(
    evidence: EvidenceFragment,
    *,
    artifact: Artifact,
    data_source: DataSource,
    production_mode: bool,
) -> None:
    if evidence.trust_state != TrustState.REVIEWED:
        raise OfficializationError(
            f"EvidenceFragment must be reviewed before officialization: {evidence.id}"
        )
    if evidence.freshness_state not in _OFFICIAL_ELIGIBLE_FRESHNESS:
        raise OfficializationError(
            f"EvidenceFragment freshness must be fresh or aging before officialization: {evidence.id}"
        )
    if artifact.id != evidence.artifact_id:
        raise OfficializationError(f"EvidenceFragment artifact mismatch: {evidence.id}")
    if artifact.datasource_id != data_source.id:
        raise OfficializationError(f"Artifact source mismatch: {artifact.id}")
    if production_mode:
        if not data_source.production_enabled:
            raise OfficializationError(
                f"EvidenceFragment comes from a non-production source: {evidence.id}"
            )
        if not source_can_officialize(data_source, production_mode=production_mode):
            raise OfficializationError(
                "EvidenceFragment source must be connected and healthy before officialization: "
                f"{evidence.id}"
            )


def ensure_decision_record_is_officialization_ready(
    decision_record: DecisionRecord,
    *,
    evidence_by_id: dict[str, EvidenceFragment],
    artifact_by_id: dict[str, Artifact],
    data_source_by_id: dict[str, DataSource],
    production_mode: bool,
) -> None:
    if not decision_record.evidence_fragment_ids:
        raise OfficializationError(
            f"DecisionRecord requires at least one reviewed EvidenceFragment: {decision_record.id}"
        )
    for evidence_id in decision_record.evidence_fragment_ids:
        evidence = evidence_by_id.get(evidence_id)
        if evidence is None:
            raise OfficializationError(
                f"DecisionRecord references missing EvidenceFragment: {evidence_id}"
            )
        artifact = artifact_by_id.get(evidence.artifact_id)
        if artifact is None:
            raise OfficializationError(f"EvidenceFragment references missing Artifact: {evidence.id}")
        data_source = data_source_by_id.get(artifact.datasource_id)
        if data_source is None:
            raise OfficializationError(f"Artifact references missing DataSource: {artifact.id}")
        ensure_evidence_is_officialization_ready(
            evidence,
            artifact=artifact,
            data_source=data_source,
            production_mode=production_mode,
        )


def ensure_concept_can_be_official(
    concept: Concept,
    *,
    evidence_by_id: dict[str, EvidenceFragment],
    decision_record_by_id: dict[str, DecisionRecord],
    artifact_by_id: dict[str, Artifact],
    data_source_by_id: dict[str, DataSource],
    production_mode: bool,
) -> None:
    if concept.status not in {ConceptStatus.CANDIDATE, ConceptStatus.REVIEWING, ConceptStatus.OFFICIAL}:
        raise OfficializationError(
            f"Only candidate, reviewing, or already official Concepts can be officialized: {concept.id}"
        )
    if not concept.evidence_fragment_ids and not concept.decision_record_ids:
        raise OfficializationError(
            "Official Concepts require at least one reviewed EvidenceFragment or valid DecisionRecord."
        )

    for evidence_id in concept.evidence_fragment_ids:
        evidence = evidence_by_id.get(evidence_id)
        if evidence is None:
            raise OfficializationError(f"Concept references missing EvidenceFragment: {evidence_id}")
        artifact = artifact_by_id.get(evidence.artifact_id)
        if artifact is None:
            raise OfficializationError(f"EvidenceFragment references missing Artifact: {evidence.id}")
        data_source = data_source_by_id.get(artifact.datasource_id)
        if data_source is None:
            raise OfficializationError(f"Artifact references missing DataSource: {artifact.id}")
        ensure_evidence_is_officialization_ready(
            evidence,
            artifact=artifact,
            data_source=data_source,
            production_mode=production_mode,
        )

    for decision_record_id in concept.decision_record_ids:
        decision_record = decision_record_by_id.get(decision_record_id)
        if decision_record is None:
            raise OfficializationError(f"Concept references missing DecisionRecord: {decision_record_id}")
        ensure_decision_record_is_officialization_ready(
            decision_record,
            evidence_by_id=evidence_by_id,
            artifact_by_id=artifact_by_id,
            data_source_by_id=data_source_by_id,
            production_mode=production_mode,
        )


def officialize_concept(
    concept: Concept,
    *,
    reviewed_by: str,
    evidence: Iterable[EvidenceFragment],
    decision_records: Iterable[DecisionRecord],
    artifacts: Iterable[Artifact],
    data_sources: Iterable[DataSource],
    production_mode: bool,
    authorized_reviewers: set[str],
) -> tuple[Concept, AuditEvent]:
    ensure_reviewer_authorized(reviewed_by, authorized_reviewers)
    evidence_by_id = {item.id: item for item in evidence}
    decision_record_by_id = {item.id: item for item in decision_records}
    artifact_by_id = {item.id: item for item in artifacts}
    data_source_by_id = {item.id: item for item in data_sources}
    ensure_concept_can_be_official(
        concept,
        evidence_by_id=evidence_by_id,
        decision_record_by_id=decision_record_by_id,
        artifact_by_id=artifact_by_id,
        data_source_by_id=data_source_by_id,
        production_mode=production_mode,
    )
    now = utc_now()
    updated = concept.model_copy(
        update={
            "status": ConceptStatus.OFFICIAL,
            "officialized_by": reviewed_by,
            "updated_at": now,
            "last_reviewed_at": now,
        },
        deep=True,
    )
    event = AuditEvent(
        event_type="concept.officialized",
        actor=reviewed_by,
        entity_type="Concept",
        entity_id=concept.id,
        metadata={
            "conceptName": concept.name,
            "evidenceFragmentCount": len(concept.evidence_fragment_ids),
            "decisionRecordCount": len(concept.decision_record_ids),
        },
    )
    return updated, event


def ensure_concept_relation_can_be_official(
    relation: ConceptRelation,
    *,
    concept_by_id: dict[str, Concept],
    evidence_by_id: dict[str, EvidenceFragment],
    decision_record_by_id: dict[str, DecisionRecord],
    artifact_by_id: dict[str, Artifact],
    data_source_by_id: dict[str, DataSource],
    production_mode: bool,
) -> None:
    if relation.status not in {RelationStatus.CANDIDATE, RelationStatus.REVIEWING, RelationStatus.OFFICIAL}:
        raise OfficializationError(
            f"Only candidate, reviewing, or already official ConceptRelations can be officialized: {relation.id}"
        )
    source_concept = concept_by_id.get(relation.source_concept_id)
    target_concept = concept_by_id.get(relation.target_concept_id)
    if source_concept is None:
        raise OfficializationError(f"ConceptRelation references missing source Concept: {relation.source_concept_id}")
    if target_concept is None:
        raise OfficializationError(f"ConceptRelation references missing target Concept: {relation.target_concept_id}")
    if source_concept.status != ConceptStatus.OFFICIAL or target_concept.status != ConceptStatus.OFFICIAL:
        raise OfficializationError("Official ConceptRelations require both Concepts to be official.")
    if not relation.evidence_fragment_ids and relation.decision_record_id is None:
        raise OfficializationError(
            "Official ConceptRelations require at least one reviewed EvidenceFragment or valid DecisionRecord."
        )

    for evidence_id in relation.evidence_fragment_ids:
        evidence = evidence_by_id.get(evidence_id)
        if evidence is None:
            raise OfficializationError(f"ConceptRelation references missing EvidenceFragment: {evidence_id}")
        artifact = artifact_by_id.get(evidence.artifact_id)
        if artifact is None:
            raise OfficializationError(f"EvidenceFragment references missing Artifact: {evidence.id}")
        data_source = data_source_by_id.get(artifact.datasource_id)
        if data_source is None:
            raise OfficializationError(f"Artifact references missing DataSource: {artifact.id}")
        ensure_evidence_is_officialization_ready(
            evidence,
            artifact=artifact,
            data_source=data_source,
            production_mode=production_mode,
        )

    if relation.decision_record_id is not None:
        decision_record = decision_record_by_id.get(relation.decision_record_id)
        if decision_record is None:
            raise OfficializationError(f"ConceptRelation references missing DecisionRecord: {relation.decision_record_id}")
        ensure_decision_record_is_officialization_ready(
            decision_record,
            evidence_by_id=evidence_by_id,
            artifact_by_id=artifact_by_id,
            data_source_by_id=data_source_by_id,
            production_mode=production_mode,
        )


def officialize_concept_relation(
    relation: ConceptRelation,
    *,
    reviewed_by: str,
    concepts: Iterable[Concept],
    evidence: Iterable[EvidenceFragment],
    decision_records: Iterable[DecisionRecord],
    artifacts: Iterable[Artifact],
    data_sources: Iterable[DataSource],
    production_mode: bool,
    authorized_reviewers: set[str],
) -> tuple[ConceptRelation, AuditEvent]:
    ensure_reviewer_authorized(reviewed_by, authorized_reviewers)
    concept_by_id = {item.id: item for item in concepts}
    evidence_by_id = {item.id: item for item in evidence}
    decision_record_by_id = {item.id: item for item in decision_records}
    artifact_by_id = {item.id: item for item in artifacts}
    data_source_by_id = {item.id: item for item in data_sources}
    ensure_concept_relation_can_be_official(
        relation,
        concept_by_id=concept_by_id,
        evidence_by_id=evidence_by_id,
        decision_record_by_id=decision_record_by_id,
        artifact_by_id=artifact_by_id,
        data_source_by_id=data_source_by_id,
        production_mode=production_mode,
    )
    now = utc_now()
    updated = relation.model_copy(
        update={
            "status": RelationStatus.OFFICIAL,
            "officialized_by": reviewed_by,
            "updated_at": now,
            "last_reviewed_at": now,
        },
        deep=True,
    )
    event = AuditEvent(
        event_type="concept_relation.officialized",
        actor=reviewed_by,
        entity_type="ConceptRelation",
        entity_id=relation.id,
        metadata={
            "sourceConceptId": relation.source_concept_id,
            "targetConceptId": relation.target_concept_id,
            "relationType": relation.relation_type,
            "evidenceFragmentCount": len(relation.evidence_fragment_ids),
            "decisionRecordId": relation.decision_record_id,
        },
    )
    return updated, event
