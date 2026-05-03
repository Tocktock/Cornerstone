from __future__ import annotations

from datetime import datetime, timezone

import pytest

from cornerstone.schemas import (
    Artifact,
    Concept,
    ConceptStatus,
    DataSource,
    DataSourceStatus,
    DataSourceType,
    EvidenceFragment,
    EvidenceFragmentType,
    ExtractionStatus,
    FreshnessState,
    Provenance,
    TrustState,
)
from cornerstone.services.officialization import (
    OfficializationError,
    ReviewerAuthorizationError,
    officialize_concept,
)

pytestmark = pytest.mark.unit


def test_officialization_rejects_unsupported_concept() -> None:
    concept = Concept(
        name="Cornerstone",
        short_definition="Shared organizational context layer.",
        created_by="reviewer@example.com",
    )

    with pytest.raises(OfficializationError, match="reviewed EvidenceFragment"):
        officialize_concept(
            concept,
            reviewed_by="reviewer@example.com",
            evidence=[],
            decision_records=[],
            artifacts=[],
            data_sources=[],
            production_mode=True,
            authorized_reviewers={"reviewer@example.com"},
        )


def test_officialization_rejects_unreviewed_evidence() -> None:
    source, artifact, evidence = _supporting_entities(trust_state=TrustState.UNREVIEWED)
    concept = Concept(
        name="Cornerstone",
        short_definition="Shared organizational context layer.",
        evidence_fragment_ids=[evidence.id],
        created_by="reviewer@example.com",
    )

    with pytest.raises(OfficializationError, match="must be reviewed"):
        officialize_concept(
            concept,
            reviewed_by="reviewer@example.com",
            evidence=[evidence],
            decision_records=[],
            artifacts=[artifact],
            data_sources=[source],
            production_mode=True,
            authorized_reviewers={"reviewer@example.com"},
        )


def test_officialization_rejects_non_production_source_evidence() -> None:
    source, artifact, evidence = _supporting_entities(production_enabled=False)
    concept = Concept(
        name="Cornerstone",
        short_definition="Shared organizational context layer.",
        evidence_fragment_ids=[evidence.id],
        created_by="reviewer@example.com",
    )

    with pytest.raises(OfficializationError, match="non-production source"):
        officialize_concept(
            concept,
            reviewed_by="reviewer@example.com",
            evidence=[evidence],
            decision_records=[],
            artifacts=[artifact],
            data_sources=[source],
            production_mode=True,
            authorized_reviewers={"reviewer@example.com"},
        )


def test_officialization_rejects_unauthorized_reviewer() -> None:
    source, artifact, evidence = _supporting_entities()
    concept = Concept(
        name="Cornerstone",
        short_definition="Shared organizational context layer.",
        evidence_fragment_ids=[evidence.id],
        created_by="reviewer@example.com",
    )

    with pytest.raises(ReviewerAuthorizationError):
        officialize_concept(
            concept,
            reviewed_by="random@example.com",
            evidence=[evidence],
            decision_records=[],
            artifacts=[artifact],
            data_sources=[source],
            production_mode=True,
            authorized_reviewers={"reviewer@example.com"},
        )


def test_officialization_accepts_reviewed_evidence_supported_concept() -> None:
    source, artifact, evidence = _supporting_entities()
    concept = Concept(
        name="Cornerstone",
        short_definition="Shared organizational context layer.",
        evidence_fragment_ids=[evidence.id],
        created_by="reviewer@example.com",
    )

    updated, event = officialize_concept(
        concept,
        reviewed_by="reviewer@example.com",
        evidence=[evidence],
        decision_records=[],
        artifacts=[artifact],
        data_sources=[source],
        production_mode=True,
        authorized_reviewers={"reviewer@example.com"},
    )

    assert updated.status == ConceptStatus.OFFICIAL
    assert updated.officialized_by == "reviewer@example.com"
    assert updated.last_reviewed_at is not None
    assert event.event_type == "concept.officialized"


def _supporting_entities(
    *,
    trust_state: TrustState = TrustState.REVIEWED,
    production_enabled: bool = True,
    source_status: DataSourceStatus = DataSourceStatus.CONNECTED,
) -> tuple[DataSource, Artifact, EvidenceFragment]:
    captured_at = datetime(2026, 4, 25, tzinfo=timezone.utc)
    source = DataSource(
        id="source-1",
        type=DataSourceType.MANUAL,
        name="Manual",
        status=source_status,
        production_enabled=production_enabled,
        last_successful_sync_at=captured_at,
    )
    artifact = Artifact(
        id="artifact-1",
        datasource_id=source.id,
        source_type=DataSourceType.MANUAL,
        source_external_id="doc-1",
        source_url="https://example.internal/doc-1",
        title="Cornerstone Overview",
        raw_content_hash="hash",
        captured_at=captured_at,
        freshness_state=FreshnessState.FRESH,
        extraction_status=ExtractionStatus.COMPLETE,
    )
    evidence = EvidenceFragment(
        id="evidence-1",
        artifact_id=artifact.id,
        text="Cornerstone is a shared organizational context layer.",
        fragment_type=EvidenceFragmentType.DEFINITION,
        provenance=Provenance(
            data_source_id=source.id,
            source_type=artifact.source_type,
            source_external_id=artifact.source_external_id,
            source_url=artifact.source_url,
            artifact_title=artifact.title,
            captured_at=artifact.captured_at,
        ),
        trust_state=trust_state,
        freshness_state=FreshnessState.FRESH,
        reviewed_by="reviewer@example.com" if trust_state == TrustState.REVIEWED else None,
        reviewed_at=captured_at if trust_state == TrustState.REVIEWED else None,
    )
    return source, artifact, evidence


def test_relation_officialization_rejects_candidate_concepts() -> None:
    source, artifact, evidence = _supporting_entities()
    source_concept = Concept(
        id="concept-source",
        name="Source",
        short_definition="Source concept.",
        status=ConceptStatus.CANDIDATE,
        evidence_fragment_ids=[evidence.id],
        created_by="reviewer@example.com",
    )
    target_concept = Concept(
        id="concept-target",
        name="Target",
        short_definition="Target concept.",
        status=ConceptStatus.OFFICIAL,
        evidence_fragment_ids=[evidence.id],
        created_by="reviewer@example.com",
    )
    from cornerstone.schemas import ConceptRelation, RelationType
    from cornerstone.services.officialization import officialize_concept_relation

    relation = ConceptRelation(
        source_concept_id=source_concept.id,
        target_concept_id=target_concept.id,
        relation_type=RelationType.DEPENDS_ON,
        evidence_fragment_ids=[evidence.id],
        created_by="reviewer@example.com",
    )

    with pytest.raises(OfficializationError, match="both Concepts to be official"):
        officialize_concept_relation(
            relation,
            reviewed_by="reviewer@example.com",
            concepts=[source_concept, target_concept],
            evidence=[evidence],
            decision_records=[],
            artifacts=[artifact],
            data_sources=[source],
            production_mode=True,
            authorized_reviewers={"reviewer@example.com"},
        )


def test_relation_officialization_accepts_reviewed_evidence_supported_relation() -> None:
    source, artifact, evidence = _supporting_entities()
    source_concept = Concept(
        id="concept-source",
        name="Source",
        short_definition="Source concept.",
        status=ConceptStatus.OFFICIAL,
        evidence_fragment_ids=[evidence.id],
        created_by="reviewer@example.com",
    )
    target_concept = Concept(
        id="concept-target",
        name="Target",
        short_definition="Target concept.",
        status=ConceptStatus.OFFICIAL,
        evidence_fragment_ids=[evidence.id],
        created_by="reviewer@example.com",
    )
    from cornerstone.schemas import ConceptRelation, RelationStatus, RelationType
    from cornerstone.services.officialization import officialize_concept_relation

    relation = ConceptRelation(
        source_concept_id=source_concept.id,
        target_concept_id=target_concept.id,
        relation_type=RelationType.DEPENDS_ON,
        evidence_fragment_ids=[evidence.id],
        created_by="reviewer@example.com",
    )

    updated, event = officialize_concept_relation(
        relation,
        reviewed_by="reviewer@example.com",
        concepts=[source_concept, target_concept],
        evidence=[evidence],
        decision_records=[],
        artifacts=[artifact],
        data_sources=[source],
        production_mode=True,
        authorized_reviewers={"reviewer@example.com"},
    )

    assert updated.status == RelationStatus.OFFICIAL
    assert updated.officialized_by == "reviewer@example.com"
    assert event.event_type == "concept_relation.officialized"
