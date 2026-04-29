from __future__ import annotations

from datetime import datetime, timezone

from cornerstone.schemas import Artifact, DataSourceType, ExtractionStatus, FreshnessState
from cornerstone.services.extraction import (
    classify_fragment,
    content_hash,
    extract_evidence_fragments,
)


def test_content_hash_is_deterministic() -> None:
    assert content_hash("same") == content_hash("same")
    assert content_hash("same") != content_hash("different")


def test_extracted_evidence_contains_required_provenance() -> None:
    artifact = Artifact(
        datasource_id="source-1",
        source_type=DataSourceType.MANUAL,
        source_external_id="doc-1",
        source_url="https://example.internal/doc-1",
        title="Doc 1",
        raw_content_hash=content_hash("Cornerstone is a context layer."),
        captured_at=datetime(2026, 4, 24, tzinfo=timezone.utc),
        freshness_state=FreshnessState.FRESH,
        extraction_status=ExtractionStatus.COMPLETE,
    )

    fragments = extract_evidence_fragments(artifact, "Cornerstone is a context layer.")

    assert len(fragments) == 1
    fragment = fragments[0]
    assert fragment.artifact_id == artifact.id
    assert fragment.provenance.data_source_id == artifact.datasource_id
    assert fragment.provenance.source_type == artifact.source_type
    assert fragment.provenance.source_external_id == artifact.source_external_id
    assert fragment.provenance.source_url == "https://example.internal/doc-1"
    assert fragment.provenance.artifact_title == "Doc 1"
    assert fragment.provenance.captured_at == artifact.captured_at
    assert fragment.provenance.quote_range is not None
    assert fragment.freshness_state == FreshnessState.FRESH


def test_fragment_classifier_detects_requirements_decisions_and_questions() -> None:
    assert classify_fragment("The system must preserve provenance.") == "requirement"
    assert classify_fragment("We decided to use FastAPI.") == "decision"
    assert classify_fragment("Who owns this concept?") == "open_question"
