from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from cornerstone.app import create_app
from cornerstone.config import Settings


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    project_root = Path(__file__).resolve().parents[3]
    source_src = project_root / "demo_sources"
    source_dst = tmp_path / "demo_sources"
    shutil.copytree(source_src, source_dst)
    db_path = tmp_path / "cornerstone-test.db"

    settings = Settings(
        database_url=f"sqlite:///{db_path}",
        source_root=str(source_dst),
        auto_seed_demo=True,
        default_context_space_name="Cornerstone Test",
        default_context_space_namespace="cornerstone-test",
        cors_origins=["http://localhost:5173"],
    )
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def unstructured_client(tmp_path: Path) -> TestClient:
    source_root = tmp_path / "sample-data"
    source_root.mkdir()
    (source_root / "cargo-translation-backend.md").write_text(
        "# Cargo Translation Backend\n\n"
        "The cargo translation backend converts file and chat inputs into structured JSON.\n\n"
        "It preserves revision history so operators can review, correct, and finalize results.\n",
        encoding="utf-8",
    )
    db_path = tmp_path / "cornerstone-unstructured.db"
    settings = Settings(
        database_url=f"sqlite:///{db_path}",
        source_root=str(source_root),
        auto_seed_demo=True,
        default_context_space_name="Cornerstone Unstructured Test",
        default_context_space_namespace="cornerstone-unstructured-test",
        cors_origins=["http://localhost:5173"],
    )
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client
