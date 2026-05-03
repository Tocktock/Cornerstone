from __future__ import annotations

from fastapi.testclient import TestClient


def _create_manual_source(client: TestClient) -> str:
    response = client.post(
        "/v1/sources",
        json={"type": "manual", "name": "Manual Uploads", "productionEnabled": True},
    )
    assert response.status_code == 201
    return str(response.json()["id"])


def test_manual_file_upload_creates_artifacts_and_evidence(client: TestClient) -> None:
    source_id = _create_manual_source(client)

    response = client.post(
        f"/v1/manual-sources/{source_id}/uploads",
        files=[
            (
                "files",
                (
                    "settlement.md",
                    b"Settlement is the process of finalizing obligations. Clearing precedes settlement.",
                    "text/markdown",
                ),
            )
        ],
    )

    assert response.status_code == 200
    body = response.json()
    assert body["dataSource"]["artifactCount"] == 1
    assert body["dataSource"]["evidenceFragmentCount"] == 2
    assert body["artifactCreatedCount"] == 1
    assert body["artifactReusedCount"] == 0

    artifact = body["artifacts"][0]
    assert artifact["sourceExternalId"] == "manual-upload:settlement.md"
    assert artifact["sourceObjectType"] == "uploaded_file"
    assert artifact["providerMetadata"]["uploadKind"] == "manual_file"
    assert artifact["providerMetadata"]["fileName"] == "settlement.md"
    assert artifact["providerMetadata"]["contentType"] == "text/markdown"
    assert artifact["providerMetadata"]["encoding"] == "utf-8"

    evidence = body["evidenceFragments"][0]
    assert evidence["artifactId"] == artifact["id"]
    assert evidence["provenance"]["dataSourceId"] == source_id
    assert evidence["provenance"]["sourceExternalId"] == "manual-upload:settlement.md"
    assert evidence["provenance"]["artifactTitle"] == "settlement.md"
    assert evidence["provenance"]["quoteRange"]["startOffset"] >= 0


def test_manual_file_upload_is_idempotent_for_same_filename_and_content(client: TestClient) -> None:
    source_id = _create_manual_source(client)
    files = [
        (
            "files",
            (
                "settlement.txt",
                b"Settlement is final. Reconciliation validates settlement.",
                "text/plain",
            ),
        )
    ]

    first = client.post(f"/v1/manual-sources/{source_id}/uploads", files=files)
    second = client.post(f"/v1/manual-sources/{source_id}/uploads", files=files)

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["artifactCreatedCount"] == 0
    assert second.json()["artifactReusedCount"] == 1
    assert second.json()["artifacts"][0]["id"] == first.json()["artifacts"][0]["id"]
    assert second.json()["dataSource"]["artifactCount"] == 1


def test_manual_file_upload_rejects_pdf(client: TestClient) -> None:
    source_id = _create_manual_source(client)

    response = client.post(
        f"/v1/manual-sources/{source_id}/uploads",
        files=[("files", ("settlement.pdf", b"%PDF-1.7", "application/pdf"))],
    )

    assert response.status_code == 415
    assert "settlement.pdf" in response.json()["detail"]


def test_manual_file_upload_rejects_invalid_utf8_text(client: TestClient) -> None:
    source_id = _create_manual_source(client)

    response = client.post(
        f"/v1/manual-sources/{source_id}/uploads",
        files=[("files", ("settlement.txt", b"\xff\xfe\xfd", "text/plain"))],
    )

    assert response.status_code == 415
    assert "UTF-8 text" in response.json()["detail"]


def test_manual_text_upload_creates_uploaded_text_artifact(client: TestClient) -> None:
    source_id = _create_manual_source(client)

    response = client.post(
        f"/v1/manual-sources/{source_id}/uploads/text",
        json={
            "objects": [
                {
                    "title": "Settlement Notes",
                    "content": "Settlement is official after review. Ledger updates follow settlement.",
                    "providerMetadata": {"importBatch": "pilot"},
                }
            ]
        },
    )

    assert response.status_code == 200
    body = response.json()
    artifact = body["artifacts"][0]
    assert artifact["sourceExternalId"] == "manual-upload:text:Settlement Notes"
    assert artifact["sourceObjectType"] == "uploaded_text"
    assert artifact["providerMetadata"]["uploadKind"] == "manual_text"
    assert artifact["providerMetadata"]["importBatch"] == "pilot"
    assert body["dataSource"]["nextAction"] == "review_evidence"
