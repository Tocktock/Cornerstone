from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_release_candidate_docs_exist_and_are_actionable() -> None:
    required = [
        "docs/27-backend-release-candidate-v0.13.0.md",
        "docs/28-backend-v1.0.0-rc.1.md",
        "docs/29-backend-v1.0.0.md",
        "docs/release/backend-operator-runbook.md",
        "docs/release/backend-release-checklist.md",
        "docs/release/known-limitations.md",
        "docs/release/production-deployment-checklist.md",
        "docs/release/secrets-and-credential-handling.md",
        "docs/release/api-freeze-review.md",
        "docs/release/live-proof-artifact-template.md",
        "docs/release/v1.0.0-readiness.md",
        "docs/release/v1.0.0-release-notes.md",
        "docs/release/v1.0.0-rc.1-verification-checklist.md",
        "docs/release/v1.0.0-rc.1-human-signoff.md",
        "docs/live-proof-records/2026-04-28-v0.13.1-blocker-fix.md",
    ]
    for rel_path in required:
        content = (ROOT / rel_path).read_text(encoding="utf-8")
        assert len(content) > 200
        assert "Cornerstone" in content or "grounded" in content or "Release" in content


def test_release_candidate_api_freeze_documents_removed_unsafe_routes() -> None:
    content = (ROOT / "docs/release/api-freeze-review.md").read_text(encoding="utf-8")
    assert "POST /v1/sources/{sourceId}/oauth/complete" in content
    assert "POST /v1/sources/{sourceId}/sync" in content
    assert "POST /v1/manual-sources/{notionSourceId}/sync" in content
    assert "removed; fake OAuth path" in content
    assert "removed; legacy bypass" in content
    assert "rejected; manual sync is manual-only" in content


def test_release_candidate_check_script_exists_and_targets_v100() -> None:
    script = ROOT / "scripts/check_release_candidate.py"
    assert script.exists()
    content = script.read_text(encoding="utf-8")
    assert 'EXPECTED_VERSION = "1.0.0"' in content
    assert 'RC_TAG = "v1.0.0"' in content


def test_release_candidate_checker_allows_local_runtime_artifacts() -> None:
    script = ROOT / "scripts/check_release_candidate.py"
    spec = importlib.util.spec_from_file_location("check_release_candidate", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    errors: list[str] = []
    module.check_package_hygiene(errors)
    assert not [error for error in errors if "__pycache__" in error]
    assert not [error for error in errors if ".pytest_cache" in error]
    assert not [error for error in errors if ".ruff_cache" in error]
    assert not [error for error in errors if ".mypy_cache" in error]
    assert not [error for error in errors if ".coverage" in error]
