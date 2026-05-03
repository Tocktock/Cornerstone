from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from cornerstone.config import Settings
from cornerstone.schemas import (
    DataSource,
    DataSourceAuthStatus,
    DataSourceConnectionStatus,
    DataSourceNextAction,
    DataSourceStatus,
    DataSourceSyncStatus,
    DataSourceType,
    EvidenceFragmentType,
    FreshnessState,
)
from cornerstone.services.manual_uploads import build_source_object_from_text_upload
from cornerstone.services.source_sync import sync_source_objects
from cornerstone.store import InMemoryStore


CORPUS_ROOT = Path(__file__).resolve().parents[2] / "test-data" / "shared-synthetic-corpus"
SECRET_PATTERNS = (
    re.compile(r"ntn_[A-Za-z0-9]+"),
    re.compile(r"secret_[A-Za-z0-9]{16,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{32,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]+"),
    re.compile(r"ghp_[A-Za-z0-9_]+"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
)


def test_shared_synthetic_corpus_manifest_is_large_domain_specific_and_safe() -> None:
    manifest = _load_json("manifest.json")

    assert manifest["datasetId"] == "cornerstone-shared-synthetic-corpus-v1"
    assert manifest["dataPolicy"]["syntheticOnly"] is True
    assert manifest["dataPolicy"]["publicDatasetDependency"] is False
    assert manifest["dataPolicy"]["containsRealPeople"] is False
    assert manifest["dataPolicy"]["containsSecrets"] is False
    assert manifest["domain"] == "temperature-controlled specialty pharmacy logistics"

    coverage = manifest["coverage"]
    assert coverage["documentCount"] >= 40
    assert coverage["sourceObjectCount"] == coverage["documentCount"]
    assert coverage["conceptCount"] >= 15
    assert coverage["expectedRelationCount"] >= 15
    assert coverage["totalWords"] >= 15_000
    assert coverage["totalSentences"] >= 1_000
    assert set(coverage["artifactTypes"]) >= {"sop", "decision_record", "field_report", "glossary"}
    assert set(coverage["visibilityModes"]) == {"evidence_only", "member_visible"}

    for document in manifest["documents"]:
        path = CORPUS_ROOT / document["path"]
        assert path.exists(), document["path"]
        content = path.read_text(encoding="utf-8")
        assert "synthetic" in content.lower()
        assert "public benchmark dataset" in content.lower() or "not-public-data" in document["tags"]
        for pattern in SECRET_PATTERNS:
            assert pattern.search(content) is None, f"{pattern.pattern} matched {document['path']}"


def test_shared_synthetic_corpus_source_objects_and_expected_ontology_are_coherent() -> None:
    manifest = _load_json("manifest.json")
    source_objects = _load_source_objects()
    expected_ontology = _load_json("expected_ontology.json")
    evaluation_tasks = _load_json("evaluation_tasks.json")

    document_ids = {document["documentId"] for document in manifest["documents"]}
    source_ids = {source["sourceExternalId"] for source in source_objects}
    assert source_ids == document_ids

    for source in source_objects:
        content_path = CORPUS_ROOT / source["contentPath"]
        assert content_path.exists(), source["contentPath"]
        assert source["providerMetadata"]["provider"] == "shared_synthetic_corpus"
        assert source["providerMetadata"]["datasetId"] == manifest["datasetId"]

    concepts = {concept["canonicalName"] for concept in expected_ontology["concepts"]}
    assert {
        "Lane Qualification",
        "Stability Budget",
        "Temperature Excursion",
        "Quality Hold",
        "Release Evidence Packet",
        "GDP Audit Readiness",
    } <= concepts

    relations = {(relation["from"], relation["type"], relation["to"]) for relation in expected_ontology["relations"]}
    assert ("Temperature Excursion", "triggers", "Quality Hold") in relations
    assert ("Quality Hold", "requires", "Release Evidence Packet") in relations
    assert ("GDP Audit Readiness", "requires", "Chain of Custody") in relations

    assert len(evaluation_tasks["tasks"]) >= 4
    assert any(task["expectedTrustLabel"] == "unsupported" for task in evaluation_tasks["tasks"])
    assert any("Quality Hold" in task["requiredConcepts"] for task in evaluation_tasks["tasks"])


def test_shared_synthetic_corpus_ingests_into_artifacts_and_evidence_fragments() -> None:
    store = InMemoryStore()
    source = store.add_data_source(
        DataSource(
            type=DataSourceType.MANUAL,
            name="Shared Synthetic Corpus",
            status=DataSourceStatus.CONNECTED,
            production_enabled=True,
            auth_status=DataSourceAuthStatus.AUTHORIZED,
            connection_status=DataSourceConnectionStatus.TEST_PASSED,
            sync_status=DataSourceSyncStatus.NEVER_SYNCED,
            next_action=DataSourceNextAction.RUN_FIRST_SYNC,
            freshness_state=FreshnessState.UNKNOWN,
            sync_freshness_state=FreshnessState.UNKNOWN,
            content_freshness_state=FreshnessState.UNKNOWN,
        )
    )

    source_objects = []
    for item in _load_source_objects():
        content = (CORPUS_ROOT / item["contentPath"]).read_text(encoding="utf-8")
        source_objects.append(
            build_source_object_from_text_upload(
                title=item["title"],
                content=content,
                source_external_id=item["sourceExternalId"],
                source_url=item["sourceUrl"],
                provider_metadata=item["providerMetadata"],
            )
        )

    response = sync_source_objects(
        data_source=source,
        objects=source_objects,
        store=store,
        settings=Settings(),
        emit_logs=False,
    )

    assert response.artifact_created_count == len(source_objects)
    assert response.artifact_reused_count == 0
    assert response.evidence_created_count >= 1_000
    assert len(store.list_artifacts(datasource_id=source.id)) == len(source_objects)
    assert len(store.list_evidence_fragments()) == response.evidence_created_count

    fragment_counts = Counter(fragment.fragment_type for fragment in store.list_evidence_fragments())
    assert fragment_counts[EvidenceFragmentType.REQUIREMENT] >= 250
    assert fragment_counts[EvidenceFragmentType.POLICY] >= 80
    assert fragment_counts[EvidenceFragmentType.DECISION] >= 90
    assert fragment_counts[EvidenceFragmentType.EXAMPLE] >= 90
    assert fragment_counts[EvidenceFragmentType.OPEN_QUESTION] >= 90

    saved_source = store.get_data_source(source.id)
    assert saved_source.sync_status == DataSourceSyncStatus.SUCCEEDED
    assert saved_source.next_action == DataSourceNextAction.REVIEW_EVIDENCE


def _load_json(relative_path: str) -> dict:
    return json.loads((CORPUS_ROOT / relative_path).read_text(encoding="utf-8"))


def _load_source_objects() -> list[dict]:
    return [
        json.loads(line)
        for line in (CORPUS_ROOT / "source_objects.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
