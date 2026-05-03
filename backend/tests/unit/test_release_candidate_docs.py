from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_release_checker():
    script = ROOT / "scripts/check_release_candidate.py"
    spec = importlib.util.spec_from_file_location("check_release_candidate", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_release_candidate_docs_exist_and_are_actionable() -> None:
    module = _load_release_checker()
    for rel_path in module.REQUIRED_DOCS:
        content = (ROOT / rel_path).read_text(encoding="utf-8")
        assert len(content) > 200
        assert "Cornerstone" in content or "grounded" in content or "Release" in content or "Product" in content


def test_release_candidate_api_freeze_documents_removed_unsafe_routes() -> None:
    content = (ROOT / "docs/release/api-freeze-review.md").read_text(encoding="utf-8")
    for phrase in [
        "POST /v1/sources/{sourceId}/oauth/complete",
        "POST /v1/sources/{sourceId}/sync",
        "POST /v1/manual-sources/{notionSourceId}/sync",
        "removed; fake OAuth path",
        "removed; legacy bypass",
        "rejected; manual sync is manual-only",
        "POST /v1/manual-sources/{sourceId}/uploads",
        "POST /v1/manual-sources/{sourceId}/uploads/text",
        "POST /v1/ontology/extraction-runs",
        "GET  /v1/ontology/extraction-runs/{runId}",
        "GET  /v1/ontology/concept-candidates",
        "GET  /v1/ontology/relation-candidates",
        "PATCH /v1/ontology/concept-candidates/{candidateId}",
        "POST  /v1/ontology/concept-candidates/{candidateId}/approve",
        "POST  /v1/ontology/concept-candidates/{candidateId}/reject",
        "POST  /v1/ontology/concept-candidates/{candidateId}/merge",
        "PATCH /v1/ontology/relation-candidates/{candidateId}",
        "POST  /v1/ontology/relation-candidates/{candidateId}/approve",
        "POST  /v1/ontology/relation-candidates/{candidateId}/reject",
        "POST  /v1/ontology/relation-candidates/{candidateId}/merge",
        "GET  /v1/ontology/explain",
        "POST /v1/evaluations/ontology/tasks",
        "POST /v1/evaluations/ontology/tasks/{taskId}/run",
        "POST /v1/evaluations/ontology/run",
        "GET  /v1/evaluations/ontology/summary",
        "POST /v1/ontology/re-extraction-runs",
        "GET  /v1/ontology/re-extraction-runs",
        "GET  /v1/ontology/re-extraction-runs/{runId}",
        "POST /v1/ontology/re-extraction-runs/{runId}/run",
        "POST /v1/ontology/proof-runs",
        "GET /v1/ontology/ssot/readiness",
        "POST /v1/ontology/extraction-runs accepts provider=live_llm",
        "GET  /v1/ontology/review-queue/summary",
        "GET  /v1/ontology/concept-candidates/{candidateId}/preview",
        "GET  /v1/ontology/relation-candidates/{candidateId}/preview",
        "GET  /v1/connectors/support-matrix",
        "GET  /v1/integration/package/manifest",
        "GET  /v1/integration/ontology/{concept}",
        "v2.0.2 Product Documentation API Freeze Note",
        "v2.0.3 Dependency-Complete Verification API Freeze Note",
        "v2.0.4 Forward Roadmap API Freeze Note",
        "v2.5.0 External Integration Package API Freeze Note",
    ]:
        assert phrase in content


def test_release_candidate_check_script_exists_and_targets_current_version() -> None:
    script = ROOT / "scripts/check_release_candidate.py"
    assert script.exists()
    content = script.read_text(encoding="utf-8")
    assert 'EXPECTED_VERSION = "2.5.0"' in content
    assert 'RC_TAG = "v2.5.0"' in content


def test_release_candidate_checker_allows_local_runtime_artifacts() -> None:
    module = _load_release_checker()
    errors: list[str] = []
    module.check_package_hygiene(errors)
    assert not [error for error in errors if "__pycache__" in error]
    assert not [error for error in errors if ".pytest_cache" in error]
    assert not [error for error in errors if ".ruff_cache" in error]
    assert not [error for error in errors if ".mypy_cache" in error]
    assert not [error for error in errors if ".coverage" in error]


def test_v2_0_0_docs_describe_ssot_readiness_endpoint() -> None:
    release_doc = (ROOT / "docs/47-ontology-ssot-release-v2.0.0.md").read_text(encoding="utf-8")
    readiness_doc = (ROOT / "docs/release/v2.0.0-ontology-ssot-readiness.md").read_text(encoding="utf-8")
    operator_doc = (ROOT / "docs/release/v2.0.0-operator-checklist.md").read_text(encoding="utf-8")
    api_contract = (ROOT / "docs/01-api-contract.md").read_text(encoding="utf-8")

    assert "GET /v1/ontology/ssot/readiness" in release_doc
    assert "source_ingestion_available" in readiness_doc
    assert "cornerstone proof run" in operator_doc
    assert "includeGraph" in api_contract
    assert "read-only" in release_doc


def test_chronicle_docs_are_measurable_through_v2_0_4() -> None:
    chronicle = (ROOT / "docs/48-version-chronicle-and-measurable-checklists-v2.0.0.md").read_text(encoding="utf-8")
    readiness = (ROOT / "docs/release/v2.0.0-documentation-chronicle-readiness.md").read_text(encoding="utf-8")
    template = (ROOT / "docs/templates/release-doc-template.md").read_text(encoding="utf-8")

    for phrase in [
        "Complete version chronicle",
        "Measurable acceptance condition",
        "Planned-but-not-released item",
        "v1.4.1",
        "docs/27-backend-release-candidate-v0.13.0.md",
        "Documentation completeness checklist",
        "v2.0.1",
        "v2.0.2",
        "V201-01",
        "V202-01",
        "Layer 1 — Product documentation",
        "Dependency-Complete Verification",
        "Forward Roadmap Goals",
        "v2.1.0 — Live LLM ontology provider",
        "v2.5.0 — Frontend MVP or external integration package",
    ]:
        assert phrase in chronicle
    assert "DOC-RDY-01" in readiness
    assert "Measurable acceptance checklist" in template

    for version in [
        "v1.2.1",
        "v1.3.0",
        "v1.3.1",
        "v1.4.0",
        "v1.5.0",
        "v1.6.0",
        "v1.7.0",
        "v1.8.0",
        "v1.9.0",
        "v2.0.0",
        "v2.0.1",
        "v2.0.2",
        "v2.0.3",
        "v2.0.4",
    ]:
        assert version in chronicle

    for rel_path in [
        "docs/36-ontology-ssot-product-contract-v1.2.1.md",
        "docs/37-ontology-domain-model-proposal-v1.2.1.md",
        "docs/38-ontology-versioned-implementation-plan-v1.2.1.md",
        "docs/39-ontology-graph-runtime-v1.3.0.md",
        "docs/40-manual-upload-ingestion-v1.3.1.md",
        "docs/41-llm-ontology-extraction-v1.4.0.md",
        "docs/42-ontology-candidate-review-workflow-v1.5.0.md",
        "docs/43-explainable-graph-serving-v1.6.0.md",
        "docs/44-ontology-evaluation-v1.7.0.md",
        "docs/45-connector-driven-reextraction-v1.8.0.md",
        "docs/46-end-to-end-proof-operator-ux-v1.9.0.md",
        "docs/47-ontology-ssot-release-v2.0.0.md",
        "docs/49-refactor-domain-boundary-v2.0.1.md",
        "docs/50-product-documentation-layer-v2.0.2.md",
        "docs/51-dependency-complete-verification-v2.0.3.md",
        "docs/52-forward-roadmap-goals-checklists-v2.0.4.md",
    ]:
        content = (ROOT / rel_path).read_text(encoding="utf-8")
        assert "Measurable acceptance condition" in content


def test_v2_0_1_refactor_docs_remain_measurable() -> None:
    release_doc = (ROOT / "docs/49-refactor-domain-boundary-v2.0.1.md").read_text(encoding="utf-8")
    readiness_doc = (ROOT / "docs/release/v2.0.1-refactor-readiness.md").read_text(encoding="utf-8")
    release_notes = (ROOT / "docs/release/v2.0.1-release-notes.md").read_text(encoding="utf-8")

    assert "Version goal" in release_doc
    assert "Confirmed non-goal" in release_doc
    assert "Measurable acceptance checklist" in release_doc
    assert "V201-01" in release_doc
    assert "V201-RDY-01" in readiness_doc
    assert "behavior-preserving" in release_notes


def test_v2_0_2_product_documentation_layer_is_clear_and_separate() -> None:
    product_docs = [
        "docs/product/README.md",
        "docs/product/00-product-overview.md",
        "docs/product/01-user-problem-and-value.md",
        "docs/product/02-how-cornerstone-works.md",
        "docs/product/03-settlement-walkthrough.md",
        "docs/product/04-ontology-graph-explained.md",
        "docs/product/05-user-roles-and-workflows.md",
        "docs/product/06-trust-model.md",
        "docs/product/07-product-vs-chatbot-rag-wiki.md",
        "docs/product/08-operator-quickstart.md",
        "docs/product/09-product-glossary.md",
    ]
    for rel_path in product_docs:
        content = (ROOT / rel_path).read_text(encoding="utf-8")
        assert "acceptance checklist" in content.lower()
        assert "Cornerstone" in content

    overview = (ROOT / "docs/product/00-product-overview.md").read_text(encoding="utf-8")
    settlement = (ROOT / "docs/product/03-settlement-walkthrough.md").read_text(encoding="utf-8")
    trust = (ROOT / "docs/product/06-trust-model.md").read_text(encoding="utf-8")
    quickstart = (ROOT / "docs/product/08-operator-quickstart.md").read_text(encoding="utf-8")
    release_doc = (ROOT / "docs/50-product-documentation-layer-v2.0.2.md").read_text(encoding="utf-8")
    readiness = (ROOT / "docs/release/v2.0.2-product-documentation-readiness.md").read_text(encoding="utf-8")
    schemas = (ROOT / "src/cornerstone/schemas.py").read_text(encoding="utf-8")

    assert "Cornerstone turns scattered company knowledge" in overview
    assert "source text" in settlement
    assert "The reviewed official ontology graph is the Single Source of Truth" in trust
    assert "manual source" in quickstart
    assert "V202-01" in release_doc
    assert "V202-RDY-01" in readiness
    assert 'release_version: str = "2.5.0"' in schemas


def test_readme_is_product_first_before_release_chronicle() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert readme.index("## Why it exists") < readme.index("## Current version")
    assert readme.index("## Product documentation") < readme.index("## Release chronicle")
    assert "docs/product/00-product-overview.md" in readme
    assert "docs/50-product-documentation-layer-v2.0.2.md" in readme
    assert "docs/51-dependency-complete-verification-v2.0.3.md" in readme
    assert "docs/52-forward-roadmap-goals-checklists-v2.0.4.md" in readme
    assert "docs/57-external-integration-package-v2.5.0.md" in readme
    assert "docs/roadmap/v2.1.0-live-llm-ontology-provider.md" in readme
    assert "scripts/run_dependency_complete_verification.py --plan-only" in readme


def test_v2_0_3_dependency_complete_verification_docs_are_measurable() -> None:
    release_doc = (ROOT / "docs/51-dependency-complete-verification-v2.0.3.md").read_text(encoding="utf-8")
    readiness_doc = (ROOT / "docs/release/v2.0.3-verification-readiness.md").read_text(encoding="utf-8")
    release_notes = (ROOT / "docs/release/v2.0.3-release-notes.md").read_text(encoding="utf-8")
    runner = (ROOT / "scripts/run_dependency_complete_verification.py").read_text(encoding="utf-8")
    plan_module = (ROOT / "src/cornerstone/verification/dependency_complete.py").read_text(encoding="utf-8")
    workflow = (ROOT / ".github/workflows/dependency-complete-verification.yml").read_text(encoding="utf-8")

    for phrase in [
        "Version goal",
        "Confirmed non-goal",
        "Measurable acceptance checklist",
        "V203-01",
        "V203-10",
        "V203-CMD-13",
    ]:
        assert phrase in release_doc
    assert "V203-RDY-01" in readiness_doc
    assert "V203-RDY-10" in readiness_doc
    assert "No runtime behavior change" in release_notes
    assert "--strict" in runner
    assert "--confirm-live-db" in runner
    assert "--plan-only" in runner
    assert "dependency_complete_command_plan" in plan_module
    assert "V203-CMD-01-python-version" in plan_module
    assert "V203-CMD-13-duplicate-audit" in plan_module
    assert "pgvector/pgvector:pg17" in workflow
    assert "run_dependency_complete_verification.py --strict --confirm-live-db" in workflow



def test_v2_0_4_forward_roadmap_docs_are_measurable() -> None:
    release_doc = (ROOT / "docs/52-forward-roadmap-goals-checklists-v2.0.4.md").read_text(encoding="utf-8")
    readiness_doc = (ROOT / "docs/release/v2.0.4-forward-roadmap-readiness.md").read_text(encoding="utf-8")
    release_notes = (ROOT / "docs/release/v2.0.4-release-notes.md").read_text(encoding="utf-8")
    roadmap_docs = {
        "docs/roadmap/v2.1.0-live-llm-ontology-provider.md": "V210-15",
        "docs/roadmap/v2.2.0-review-operator-experience.md": "V220-14",
        "docs/roadmap/v2.3.0-graph-visualization-contract.md": "V230-13",
        "docs/roadmap/v2.4.0-connector-expansion-live-proof-hardening.md": "V240-13",
        "docs/roadmap/v2.5.0-frontend-mvp-external-integration-package.md": "V250-13",
    }

    for phrase in [
        "Version goal",
        "Confirmed non-goal",
        "Measurable acceptance checklist",
        "V204-01",
        "V204-12",
        "v2.1.0 — Live LLM ontology provider",
        "v2.5.0 — Frontend MVP or external integration package",
    ]:
        assert phrase in release_doc
    assert "V204-RDY-01" in readiness_doc
    assert "V204-RDY-10" in readiness_doc
    assert "No runtime behavior change" in release_notes

    for rel_path, terminal_check in roadmap_docs.items():
        content = (ROOT / rel_path).read_text(encoding="utf-8")
        for phrase in [
            "Version goal",
            "Confirmed non-goal",
            "Domain boundary",
            "Measurable acceptance checklist",
            "Verification checklist",
            "Exit criteria",
            "Next-version handoff",
            terminal_check,
        ]:
            assert phrase in content

    roadmap_index = (ROOT / "docs/roadmap/README.md").read_text(encoding="utf-8")
    assert "ROADMAP-README-01" in roadmap_index
    assert "v2.1.0 — Live LLM ontology provider" in roadmap_index
    assert "v2.5.0 — Frontend MVP or external integration package" in roadmap_index


def test_v2_5_0_implemented_roadmap_docs_are_measurable() -> None:
    implemented_docs = {
        "docs/53-live-llm-ontology-provider-v2.1.0.md": "V210-15",
        "docs/54-review-operator-experience-v2.2.0.md": "V220-14",
        "docs/55-graph-visualization-contract-v2.3.0.md": "V230-13",
        "docs/56-connector-expansion-live-proof-hardening-v2.4.0.md": "V240-13",
        "docs/57-external-integration-package-v2.5.0.md": "V250-13",
    }
    readiness_docs = {
        "docs/release/v2.1.0-live-llm-ontology-provider-readiness.md": "V210-RDY-01",
        "docs/release/v2.2.0-review-operator-experience-readiness.md": "V220-RDY-01",
        "docs/release/v2.3.0-graph-visualization-contract-readiness.md": "V230-RDY-01",
        "docs/release/v2.4.0-connector-expansion-live-proof-readiness.md": "V240-RDY-01",
        "docs/release/v2.5.0-external-integration-package-readiness.md": "V250-RDY-01",
    }
    chronicle = (ROOT / "docs/48-version-chronicle-and-measurable-checklists-v2.0.0.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    for rel_path, terminal_check in implemented_docs.items():
        content = (ROOT / rel_path).read_text(encoding="utf-8")
        for phrase in [
            "Version goal",
            "Confirmed non-goal",
            "Measurable acceptance checklist",
            "Verification checklist",
            "Exit criteria",
            "Next-version handoff",
            terminal_check,
        ]:
            assert phrase in content

    for rel_path, readiness_check in readiness_docs.items():
        assert readiness_check in (ROOT / rel_path).read_text(encoding="utf-8")

    assert "v2.5.0 — External Integration Package" in readme
    assert "2.5.0" in readme
    assert "docs/57-external-integration-package-v2.5.0.md" in chronicle
    assert "external_integration_package" in (ROOT / "docs/release/v2.5.0-release-notes.md").read_text(encoding="utf-8")
