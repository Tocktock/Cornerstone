from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

GROUNDED_SNAPSHOT_PATH = Path(__file__).resolve().parents[1] / "snapshots" / "openapi_v0_12_1_snapshot.json"
EVALUATION_SNAPSHOT_PATH = Path(__file__).resolve().parents[1] / "snapshots" / "openapi_v0_12_1_evaluations_snapshot.json"


def test_grounded_serving_and_evaluation_openapi_contract_matches_snapshot(client: TestClient) -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    snapshot = _grounded_serving_snapshot(response.json())
    expected = json.loads(GROUNDED_SNAPSHOT_PATH.read_text())

    assert snapshot == expected


def test_evaluation_openapi_contract_matches_snapshot(client: TestClient) -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200
    snapshot = _evaluation_snapshot(response.json())
    expected = json.loads(EVALUATION_SNAPSHOT_PATH.read_text())

    assert snapshot == expected


def _grounded_serving_snapshot(openapi: dict[str, Any]) -> dict[str, Any]:
    schemas = openapi["components"]["schemas"]
    schema_names = [
        "GroundedContextResponse",
        "EvidenceCitation",
        "CitationSupportRef",
        "CitationSupportType",
        "TrustLabel",
        "FreshnessSummary",
        "ConceptRef",
        "ConceptRelationRef",
        "DecisionRecordRef",
        "CreateGroundedContextEvalTaskRequest",
        "GroundedContextEvalTask",
        "GroundedContextEvalResult",
        "GroundedContextEvalRunResponse",
        "GroundedContextEvalMetricSummary",
        "RunGroundedContextEvalTaskRequest",
        "RunGroundedContextEvalRequest",
    ]
    evaluation_paths = {
        path: openapi["paths"][path]
        for path in sorted(openapi["paths"])
        if path.startswith("/v1/evaluations")
    }
    return {
        "paths": {
            "/v1/context/query": openapi["paths"]["/v1/context/query"],
            **evaluation_paths,
        },
        "schemas": {name: schemas[name] for name in schema_names},
    }


def _evaluation_snapshot(openapi: dict[str, Any]) -> dict[str, Any]:
    schemas = openapi["components"]["schemas"]
    schema_names = [
        "CreateGroundedContextEvalTaskRequest",
        "GroundedContextEvalTask",
        "GroundedContextEvalResult",
        "GroundedContextEvalRunResponse",
        "GroundedContextEvalMetricSummary",
        "RunGroundedContextEvalRequest",
        "RunGroundedContextEvalTaskRequest",
    ]
    path_names = [
        "/v1/evaluations/tasks",
        "/v1/evaluations/tasks/{task_id}",
        "/v1/evaluations/tasks/{task_id}/run",
        "/v1/evaluations/run",
        "/v1/evaluations/results",
        "/v1/evaluations/results/{result_id}",
        "/v1/evaluations/summary",
    ]
    return {
        "paths": {name: sorted(openapi["paths"][name].keys()) for name in path_names},
        "schemas": {name: sorted(schemas[name].get("properties", {}).keys()) for name in schema_names},
    }
