from __future__ import annotations

from datetime import datetime, timezone

import pytest

from cornerstone.schemas import (
    Artifact,
    CitationSupportType,
    Concept,
    ConceptRelation,
    ConceptStatus,
    DataSource,
    DataSourceStatus,
    DataSourceType,
    DecisionRecord,
    EvidenceFragment,
    EvidenceFragmentType,
    ExtractionStatus,
    FreshnessState,
    Provenance,
    RelationStatus,
    RelationType,
    TrustState,
)
from cornerstone.services.grounded_context import GroundedContextService
from cornerstone.store import InMemoryStore

pytestmark = pytest.mark.unit


def test_candidate_concept_with_reviewed_evidence_is_evidence_supported() -> None:
    store = InMemoryStore()
    evidence = _add_evidence(store, freshness_state=FreshnessState.FRESH)
    store.add_concept(
        Concept(
            name="Cornerstone",
            short_definition="Shared organizational context layer.",
            evidence_fragment_ids=[evidence.id],
            created_by="reviewer@example.com",
        )
    )

    response = GroundedContextService(store).query("What is Cornerstone?")

    assert response.trust_label == "evidence_supported"
    assert response.freshness.state == "fresh"
    assert response.evidence[0].evidence_fragment_id == evidence.id
    assert "Matching Concept is not official yet." in response.limitations


def test_candidate_concept_with_unreviewed_evidence_is_partially_supported() -> None:
    store = InMemoryStore()
    evidence = _add_evidence(
        store,
        freshness_state=FreshnessState.FRESH,
        trust_state=TrustState.UNREVIEWED,
    )
    store.add_concept(
        Concept(
            name="Cornerstone",
            short_definition="Shared organizational context layer.",
            evidence_fragment_ids=[evidence.id],
            created_by="reviewer@example.com",
        )
    )

    response = GroundedContextService(store).query("What is Cornerstone?")

    assert response.trust_label == "partially_supported"
    assert "At least one supporting EvidenceFragment has not been reviewed." in response.limitations


def test_official_concept_with_stale_evidence_is_stale() -> None:
    store = InMemoryStore()
    evidence = _add_evidence(store, freshness_state=FreshnessState.STALE)
    store.add_concept(
        Concept(
            name="Cornerstone",
            short_definition="Shared organizational context layer.",
            status=ConceptStatus.OFFICIAL,
            evidence_fragment_ids=[evidence.id],
            created_by="reviewer@example.com",
        )
    )

    response = GroundedContextService(store).query("Cornerstone")

    assert response.trust_label == "stale"
    assert response.freshness.state == "stale"
    assert response.freshness.stale_evidence_count == 1
    assert "At least one supporting EvidenceFragment is stale." in response.limitations


def test_official_concept_with_unknown_freshness_is_not_labeled_official() -> None:
    store = InMemoryStore()
    evidence = _add_evidence(store, freshness_state=FreshnessState.UNKNOWN)
    store.add_concept(
        Concept(
            name="Cornerstone",
            short_definition="Shared organizational context layer.",
            status=ConceptStatus.OFFICIAL,
            evidence_fragment_ids=[evidence.id],
            created_by="reviewer@example.com",
        )
    )

    response = GroundedContextService(store).query("Cornerstone")

    assert response.trust_label == "partially_supported"
    assert response.freshness.state == "unknown"


def test_conflicted_concept_returns_conflicted() -> None:
    store = InMemoryStore()
    evidence = _add_evidence(store, freshness_state=FreshnessState.FRESH)
    store.add_concept(
        Concept(
            name="Cornerstone",
            short_definition="Shared organizational context layer.",
            status=ConceptStatus.CONFLICTED,
            evidence_fragment_ids=[evidence.id],
            created_by="reviewer@example.com",
        )
    )

    response = GroundedContextService(store).query("Cornerstone")

    assert response.trust_label == "conflicted"
    assert "The matching Concept is marked conflicted." in response.limitations


def test_official_concept_with_missing_evidence_is_unsupported() -> None:
    store = InMemoryStore()
    store.add_concept(
        Concept(
            name="Cornerstone",
            short_definition="Shared organizational context layer.",
            status=ConceptStatus.OFFICIAL,
            evidence_fragment_ids=["missing-evidence"],
            created_by="reviewer@example.com",
        )
    )

    response = GroundedContextService(store).query("Cornerstone")

    assert response.trust_label == "unsupported"
    assert response.evidence == []
    assert "No serving-eligible supporting EvidenceFragments are attached." in response.limitations


def test_production_mode_excludes_non_production_source_evidence() -> None:
    store = InMemoryStore()
    evidence = _add_evidence(
        store,
        freshness_state=FreshnessState.FRESH,
        production_enabled=False,
    )
    store.add_concept(
        Concept(
            name="Demo Concept",
            short_definition="Demo-only content.",
            evidence_fragment_ids=[evidence.id],
            created_by="system",
        )
    )

    response = GroundedContextService(store, production_mode=True).query("Demo Concept")

    assert response.trust_label == "unsupported"
    assert response.evidence == []



def test_degraded_source_still_serves_reviewed_captured_evidence_with_limitation() -> None:
    store = InMemoryStore()
    evidence = _add_evidence(
        store,
        freshness_state=FreshnessState.FRESH,
        source_status=DataSourceStatus.DEGRADED,
    )
    store.add_concept(
        Concept(
            name="Cornerstone",
            short_definition="Shared organizational context layer.",
            status=ConceptStatus.OFFICIAL,
            evidence_fragment_ids=[evidence.id],
            created_by="reviewer@example.com",
        )
    )

    response = GroundedContextService(store, production_mode=True).query("Cornerstone")

    assert response.trust_label == "official"
    assert response.evidence[0].evidence_fragment_id == evidence.id
    assert any("currently degraded" in limitation for limitation in response.limitations)

def _add_evidence(
    store: InMemoryStore,
    freshness_state: FreshnessState,
    *,
    trust_state: TrustState = TrustState.REVIEWED,
    production_enabled: bool = True,
    source_status: DataSourceStatus = DataSourceStatus.CONNECTED,
) -> EvidenceFragment:
    captured_at = datetime(2026, 4, 25, tzinfo=timezone.utc)
    source = store.add_data_source(
        DataSource(
            id="source-1",
            type=DataSourceType.MANUAL,
            name="Manual",
            status=source_status,
            production_enabled=production_enabled,
            last_successful_sync_at=captured_at,
        )
    )
    artifact = store.add_artifact(
        Artifact(
            datasource_id=source.id,
            source_type=DataSourceType.MANUAL,
            source_external_id="doc-1",
            source_url="https://example.internal/doc-1",
            title="Cornerstone Overview",
            raw_content_hash="hash",
            captured_at=captured_at,
            freshness_state=freshness_state,
            extraction_status=ExtractionStatus.COMPLETE,
        )
    )
    return store.add_evidence_fragment(
        EvidenceFragment(
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
            freshness_state=freshness_state,
            reviewed_by="reviewer@example.com" if trust_state == TrustState.REVIEWED else None,
            reviewed_at=captured_at if trust_state == TrustState.REVIEWED else None,
        )
    )



def test_related_reviewed_evidence_without_concept_is_evidence_supported() -> None:
    store = InMemoryStore()
    evidence = _add_evidence(store, freshness_state=FreshnessState.FRESH)

    response = GroundedContextService(store).query("organizational context")

    assert response.trust_label == "evidence_supported"
    assert response.official_answer_available is False
    assert response.concepts == []
    assert response.evidence[0].evidence_fragment_id == evidence.id
    assert response.evidence[0].supports[0].entity_type == CitationSupportType.EVIDENCE_FRAGMENT
    assert "No matching official Concept was found" in response.limitations[0]


def test_related_rejected_evidence_is_not_served_as_support() -> None:
    store = InMemoryStore()
    _add_evidence(
        store,
        freshness_state=FreshnessState.FRESH,
        trust_state=TrustState.REJECTED,
    )

    response = GroundedContextService(store).query("organizational context")

    assert response.trust_label == "unsupported"
    assert response.evidence == []
    assert "No matching Concept or EvidenceFragment was found." in response.limitations


def test_related_conflicted_evidence_returns_conflicted() -> None:
    store = InMemoryStore()
    evidence = _add_evidence(
        store,
        freshness_state=FreshnessState.FRESH,
        trust_state=TrustState.CONFLICTED,
    )

    response = GroundedContextService(store).query("organizational context")

    assert response.trust_label == "conflicted"
    assert response.evidence[0].evidence_fragment_id == evidence.id
    assert "At least one related EvidenceFragment is conflicted." in response.limitations


def test_official_concept_response_includes_official_relations_and_relation_citations() -> None:
    store = InMemoryStore()
    evidence = _add_evidence(store, freshness_state=FreshnessState.FRESH)
    source_concept = store.add_concept(
        Concept(
            name="Cornerstone",
            short_definition="Shared organizational context layer.",
            status=ConceptStatus.OFFICIAL,
            evidence_fragment_ids=[evidence.id],
            created_by="reviewer@example.com",
        )
    )
    target_concept = store.add_concept(
        Concept(
            name="Source Studio",
            short_definition="Connector administration surface.",
            status=ConceptStatus.OFFICIAL,
            evidence_fragment_ids=[evidence.id],
            created_by="reviewer@example.com",
        )
    )
    relation = store.add_concept_relation(
        ConceptRelation(
            source_concept_id=source_concept.id,
            target_concept_id=target_concept.id,
            relation_type=RelationType.DEPENDS_ON,
            status=RelationStatus.OFFICIAL,
            evidence_fragment_ids=[evidence.id],
            created_by="reviewer@example.com",
        )
    )

    response = GroundedContextService(store).query("Cornerstone")

    assert response.trust_label == "official"
    assert response.relations[0].id == relation.id
    support_types = {support.entity_type for support in response.evidence[0].supports}
    assert CitationSupportType.CONCEPT in support_types
    assert CitationSupportType.CONCEPT_RELATION in support_types


def test_official_concept_response_includes_decisions_and_decision_citations() -> None:
    store = InMemoryStore()
    evidence = _add_evidence(store, freshness_state=FreshnessState.FRESH)
    decision = store.add_decision_record(
        DecisionRecord(
            title="Adopt Cornerstone",
            decision="Use Cornerstone as the shared context layer.",
            reason="It preserves provenance.",
            decided_by="reviewer@example.com",
            evidence_fragment_ids=[evidence.id],
        )
    )
    store.add_concept(
        Concept(
            name="Cornerstone",
            short_definition="Shared organizational context layer.",
            status=ConceptStatus.OFFICIAL,
            decision_record_ids=[decision.id],
            created_by="reviewer@example.com",
        )
    )

    response = GroundedContextService(store).query("Cornerstone")

    assert response.trust_label == "official"
    assert response.decisions[0].id == decision.id
    assert response.evidence[0].supports[0].entity_type == CitationSupportType.DECISION_RECORD
