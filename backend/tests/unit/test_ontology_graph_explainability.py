from __future__ import annotations

import pytest

from cornerstone.schemas import (
    Artifact,
    Concept,
    ConceptCandidate,
    ConceptRelation,
    ConceptStatus,
    DataSource,
    DataSourceConnectionStatus,
    DataSourceNextAction,
    DataSourceStatus,
    DataSourceSyncStatus,
    DataSourceType,
    EvidenceFragment,
    EvidenceFragmentType,
    ExtractionStatus,
    FreshnessState,
    OntologyExtractionRun,
    OntologyExtractionRunStatus,
    Provenance,
    RelationStatus,
    RelationType,
    TrustLabel,
    TrustState,
    utc_now,
)
from cornerstone.services.ontology_graph import OntologyGraphService
from cornerstone.store import InMemoryStore

pytestmark = pytest.mark.unit


def _seed_reviewed_evidence(store: InMemoryStore) -> str:
    source = DataSource(
        type=DataSourceType.MANUAL,
        name="Settlement Explainability",
        status=DataSourceStatus.CONNECTED,
        production_enabled=True,
        connection_status=DataSourceConnectionStatus.TEST_PASSED,
        sync_status=DataSourceSyncStatus.SUCCEEDED,
        next_action=DataSourceNextAction.NONE,
    )
    store.add_data_source(source)
    artifact = Artifact(
        datasource_id=source.id,
        source_type=DataSourceType.MANUAL,
        source_external_id="settlement-doc",
        source_url="https://example.internal/settlement-doc",
        title="Settlement Guide",
        raw_content_hash="settlement-hash",
        freshness_state=FreshnessState.FRESH,
        extraction_status=ExtractionStatus.COMPLETE,
    )
    store.add_artifact(artifact)
    evidence = EvidenceFragment(
        artifact_id=artifact.id,
        text="Settlement updates the ledger after clearing is complete.",
        fragment_type=EvidenceFragmentType.CLAIM,
        provenance=Provenance(
            data_source_id=source.id,
            source_type=DataSourceType.MANUAL,
            source_external_id="settlement-doc",
            source_url="https://example.internal/settlement-doc",
            artifact_title="Settlement Guide",
            captured_at=utc_now(),
        ),
        trust_state=TrustState.REVIEWED,
        freshness_state=FreshnessState.FRESH,
        reviewed_by="reviewer@example.com",
        reviewed_at=utc_now(),
    )
    store.add_evidence_fragment(evidence)
    return evidence.id


def _seed_official_settlement_graph(store: InMemoryStore) -> dict[str, str]:
    evidence_id = _seed_reviewed_evidence(store)
    settlement = Concept(
        name="Settlement",
        aliases=["payment settlement"],
        short_definition="The process of finalizing obligations.",
        status=ConceptStatus.OFFICIAL,
        evidence_fragment_ids=[evidence_id],
        created_by="reviewer@example.com",
        officialized_by="reviewer@example.com",
        last_reviewed_at=utc_now(),
    )
    ledger = Concept(
        name="Ledger",
        short_definition="The system of record for transaction state.",
        status=ConceptStatus.OFFICIAL,
        evidence_fragment_ids=[evidence_id],
        created_by="reviewer@example.com",
        officialized_by="reviewer@example.com",
        last_reviewed_at=utc_now(),
    )
    store.add_concept(settlement)
    store.add_concept(ledger)
    relation = ConceptRelation(
        source_concept_id=settlement.id,
        target_concept_id=ledger.id,
        relation_type=RelationType.UPDATES,
        status=RelationStatus.OFFICIAL,
        evidence_fragment_ids=[evidence_id],
        created_by="reviewer@example.com",
        officialized_by="reviewer@example.com",
        last_reviewed_at=utc_now(),
    )
    store.add_concept_relation(relation)
    return {"evidence_id": evidence_id, "settlement_id": settlement.id, "ledger_id": ledger.id, "relation_id": relation.id}


def test_explainable_graph_includes_review_provenance_support_summary_and_explanation() -> None:
    store = InMemoryStore()
    ids = _seed_official_settlement_graph(store)

    response = OntologyGraphService(store, production_mode=True).graph("payment settlement")

    assert response.trust_label == TrustLabel.OFFICIAL
    assert response.official_graph_available is True
    assert response.support_summary.node_count == 2
    assert response.support_summary.edge_count == 1
    assert response.support_summary.official_node_count == 2
    assert response.support_summary.official_edge_count == 1
    assert response.support_summary.reviewed_evidence_count == 1
    assert response.explanation is not None
    assert response.explanation.ssot_status == "official_ssot"
    assert "Depth 1" in response.explanation.graph_scope
    assert response.focus_concept is not None
    assert response.focus_concept.review_provenance is not None
    assert response.focus_concept.review_provenance.officialized_by == "reviewer@example.com"
    edge = response.edges[0]
    assert edge.id == ids["relation_id"]
    assert edge.source_concept_name == "Settlement"
    assert edge.target_concept_name == "Ledger"
    assert edge.focus_direction == "outgoing"
    assert edge.support_summary.reviewed_evidence_count == 1
    citation = response.evidence[0]
    assert citation.evidence_fragment_id == ids["evidence_id"]
    assert citation.data_source_id is not None
    assert citation.reviewed_by == "reviewer@example.com"
    support_types = {support.entity_type for support in citation.supports}
    assert {"concept", "concept_relation"} <= support_types
    assert response.visualization.focus_concept_id == ids["settlement_id"]
    assert response.visualization.state_legend["official"] == "Reviewed official graph object."
    assert response.visualization.layout_hints["depth"] == 1
    assert response.visualization.citation_panel["citationCount"] == 1
    visual_nodes = {node.id: node for node in response.visualization.nodes}
    assert visual_nodes[ids["settlement_id"]].display_state == "official"
    assert visual_nodes[ids["settlement_id"]].group == "focus"
    assert visual_nodes[ids["settlement_id"]].citation_panel["evidenceFragmentIds"] == [ids["evidence_id"]]
    assert response.visualization.edges[0].id == ids["relation_id"]
    assert response.visualization.edges[0].direction == "outgoing"
    assert response.visualization.edges[0].display_state == "official"


def test_explainable_graph_summarizes_pending_candidates_without_promoting_them() -> None:
    store = InMemoryStore()
    evidence_id = _seed_reviewed_evidence(store)
    run = OntologyExtractionRun(
        model_name="local-rule-based-ontology-extractor-v1.4.0",
        prompt_version="ontology-extraction-v1.4.0",
        status=OntologyExtractionRunStatus.COMPLETED,
        requested_by="reviewer@example.com",
        evidence_fragment_ids=[evidence_id],
    )
    store.add_ontology_extraction_run(run)
    store.add_concept_candidate(
        ConceptCandidate(
            extraction_run_id=run.id,
            name="Settlement",
            normalized_name="settlement",
            proposed_definition="Settlement is the process of finalizing obligations.",
            evidence_fragment_ids=[evidence_id],
            confidence=0.86,
        )
    )

    response = OntologyGraphService(store, production_mode=True).graph("settlement")

    assert response.trust_label == TrustLabel.UNSUPPORTED
    assert response.official_graph_available is False
    assert response.nodes == []
    assert response.candidate_summary.has_pending_candidates is True
    assert response.candidate_summary.pending_concept_candidate_count == 1
    assert response.explanation is not None
    assert "not official truth" in response.explanation.candidate_boundary
    assert response.visualization.empty_state == "No graph is available for this concept and mode."
    assert response.visualization.state_legend["unsupported"] == "Missing serving-eligible evidence."
