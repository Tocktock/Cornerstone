from __future__ import annotations

from fastapi.testclient import TestClient


def test_sync_creates_artifact_and_evidence_with_provenance(client: TestClient) -> None:
    source_response = client.post(
        "/v1/sources",
        json={"type": "manual", "name": "Manual Pilot", "productionEnabled": True},
    )
    assert source_response.status_code == 201
    source_id = source_response.json()["id"]

    sync_response = client.post(
        f"/v1/manual-sources/{source_id}/sync",
        json={
            "objects": [
                {
                    "sourceExternalId": "doc-1",
                    "title": "Cornerstone Overview",
                    "content": "Cornerstone is a shared organizational context layer. The system must preserve provenance.",
                    "sourceUrl": "https://example.internal/doc-1",
                }
            ]
        },
    )

    assert sync_response.status_code == 200
    body = sync_response.json()
    assert body["dataSource"]["artifactCount"] == 1
    assert body["dataSource"]["evidenceFragmentCount"] == 2
    assert body["dataSource"]["syncFreshnessState"] == "fresh"
    assert body["dataSource"]["contentFreshnessState"] == "fresh"

    artifact = body["artifacts"][0]
    assert artifact["sourceExternalId"] == "doc-1"
    assert artifact["rawContentHash"]
    assert artifact["freshnessState"] == "fresh"
    assert artifact["extractionStatus"] == "complete"

    evidence = body["evidenceFragments"][0]
    assert evidence["artifactId"] == artifact["id"]
    assert evidence["provenance"]["dataSourceId"] == source_id
    assert evidence["provenance"]["sourceType"] == "manual"
    assert evidence["provenance"]["sourceExternalId"] == "doc-1"
    assert evidence["provenance"]["sourceUrl"] == "https://example.internal/doc-1"
    assert evidence["provenance"]["artifactTitle"] == "Cornerstone Overview"
    assert evidence["provenance"]["capturedAt"]
    assert evidence["provenance"]["quoteRange"]["startOffset"] >= 0

    evidence_list_response = client.get(f"/v1/evidence?artifactId={artifact['id']}")
    assert evidence_list_response.status_code == 200
    assert len(evidence_list_response.json()) == 2
