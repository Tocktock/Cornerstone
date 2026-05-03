#!/usr/bin/env python3
"""Static release-candidate readiness checks.

This script intentionally avoids importing FastAPI, SQLAlchemy, Pydantic, or other
runtime dependencies so it can run even in minimal packaging environments.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSION = "2.5.0"
RC_TAG = "v2.5.0"

REQUIRED_DOCS = ['docs/27-backend-release-candidate-v0.13.0.md',
 'docs/28-backend-v1.0.0-rc.1.md',
 'docs/29-backend-v1.0.0.md',
 'docs/release/backend-operator-runbook.md',
 'docs/release/backend-release-checklist.md',
 'docs/release/known-limitations.md',
 'docs/integration-starter-kit/macos-quickstart.md',
 'docs/release/production-deployment-checklist.md',
 'docs/release/secrets-and-credential-handling.md',
 'docs/release/api-freeze-review.md',
 'docs/release/live-proof-artifact-template.md',
 'docs/release/v1.0.0-readiness.md',
 'docs/release/v1.0.0-release-notes.md',
 'docs/30-cli-macos-starter-v1.1.0.md',
 'docs/31-cli-product-loop-v1.1.1.md',
 'docs/32-one-command-proof-runner-v1.1.2.md',
 'docs/33-cross-platform-starter-v1.1.3.md',
 'docs/34-cli-maintainability-v1.1.4.md',
 'docs/35-google-drive-connector-v1.2.0.md',
 'docs/36-ontology-ssot-product-contract-v1.2.1.md',
 'docs/37-ontology-domain-model-proposal-v1.2.1.md',
 'docs/38-ontology-versioned-implementation-plan-v1.2.1.md',
 'docs/adr/0001-ontology-official-trust-boundary.md',
 'docs/templates/release-doc-template.md',
 'docs/release/v1.2.1-ontology-contract-readiness.md',
 'docs/release/v1.2.1-release-notes.md',
 'docs/39-ontology-graph-runtime-v1.3.0.md',
 'docs/release/v1.3.0-ontology-graph-readiness.md',
 'docs/release/v1.3.0-release-notes.md',
 'docs/40-manual-upload-ingestion-v1.3.1.md',
 'docs/release/v1.3.1-manual-upload-readiness.md',
 'docs/release/v1.3.1-release-notes.md',
 'docs/41-llm-ontology-extraction-v1.4.0.md',
 'docs/42-ontology-candidate-review-workflow-v1.5.0.md',
 'docs/release/v1.4.0-llm-ontology-extraction-readiness.md',
 'docs/release/v1.4.0-release-notes.md',
 'docs/release/v1.5.0-ontology-candidate-review-readiness.md',
 'docs/release/v1.5.0-release-notes.md',
 'docs/43-explainable-graph-serving-v1.6.0.md',
 'docs/release/v1.6.0-explainable-graph-readiness.md',
 'docs/release/v1.6.0-release-notes.md',
 'docs/44-ontology-evaluation-v1.7.0.md',
 'docs/45-connector-driven-reextraction-v1.8.0.md',
 'docs/46-end-to-end-proof-operator-ux-v1.9.0.md',
 'docs/release/v1.7.0-ontology-evaluation-readiness.md',
 'docs/release/v1.8.0-connector-driven-reextraction-readiness.md',
 'docs/release/v1.9.0-end-to-end-proof-readiness.md',
 'docs/release/v1.7.0-release-notes.md',
 'docs/release/v1.8.0-release-notes.md',
 'docs/release/v1.9.0-release-notes.md',
 'docs/release/v2.0.0-operator-checklist.md',
 'docs/release/v2.0.0-release-notes.md',
 'docs/release/v2.0.0-ontology-ssot-readiness.md',
 'docs/47-ontology-ssot-release-v2.0.0.md',
 'docs/48-version-chronicle-and-measurable-checklists-v2.0.0.md',
 'docs/release/v2.0.0-documentation-chronicle-readiness.md',
 'docs/49-refactor-domain-boundary-v2.0.1.md',
 'docs/release/v2.0.1-refactor-readiness.md',
 'docs/release/v2.0.1-release-notes.md',
 'docs/integration-starter-kit/cli-guide.md',
 'docs/integration-starter-kit/local-quickstart.md',
 'docs/integration-starter-kit/linux-quickstart.md',
 'docs/integration-starter-kit/windows-quickstart.md',
 'docs/integration-starter-kit/notion-live-proof.md',
 'docs/integration-starter-kit/google-drive-quickstart.md',
 'docs/release/v1.0.0-rc.1-verification-checklist.md',
 'docs/release/v1.0.0-rc.1-human-signoff.md',
 'docs/live-proof-records/2026-04-27-change-log.md',
 'docs/live-proof-records/2026-04-28-v0.13.1-blocker-fix.md',
 'docs/product/README.md',
 'docs/product/00-product-overview.md',
 'docs/product/01-user-problem-and-value.md',
 'docs/product/02-how-cornerstone-works.md',
 'docs/product/03-settlement-walkthrough.md',
 'docs/product/04-ontology-graph-explained.md',
 'docs/product/05-user-roles-and-workflows.md',
 'docs/product/06-trust-model.md',
 'docs/product/07-product-vs-chatbot-rag-wiki.md',
 'docs/product/08-operator-quickstart.md',
 'docs/product/09-product-glossary.md',
 'docs/50-product-documentation-layer-v2.0.2.md',
 'docs/release/v2.0.2-product-documentation-readiness.md',
 'docs/release/v2.0.2-release-notes.md',
 'docs/51-dependency-complete-verification-v2.0.3.md',
 'docs/release/v2.0.3-verification-readiness.md',
 'docs/release/v2.0.3-release-notes.md',
 'docs/52-forward-roadmap-goals-checklists-v2.0.4.md',
 'docs/roadmap/README.md',
 'docs/roadmap/v2.1.0-live-llm-ontology-provider.md',
 'docs/roadmap/v2.2.0-review-operator-experience.md',
 'docs/roadmap/v2.3.0-graph-visualization-contract.md',
 'docs/roadmap/v2.4.0-connector-expansion-live-proof-hardening.md',
 'docs/roadmap/v2.5.0-frontend-mvp-external-integration-package.md',
 'docs/release/v2.0.4-forward-roadmap-readiness.md',
 'docs/release/v2.0.4-release-notes.md',
 'docs/53-live-llm-ontology-provider-v2.1.0.md',
 'docs/54-review-operator-experience-v2.2.0.md',
 'docs/55-graph-visualization-contract-v2.3.0.md',
 'docs/56-connector-expansion-live-proof-hardening-v2.4.0.md',
 'docs/57-external-integration-package-v2.5.0.md',
 'docs/release/v2.1.0-live-llm-ontology-provider-readiness.md',
 'docs/release/v2.1.0-release-notes.md',
 'docs/release/v2.2.0-review-operator-experience-readiness.md',
 'docs/release/v2.2.0-release-notes.md',
 'docs/release/v2.3.0-graph-visualization-contract-readiness.md',
 'docs/release/v2.3.0-release-notes.md',
 'docs/release/v2.4.0-connector-expansion-live-proof-readiness.md',
 'docs/release/v2.4.0-release-notes.md',
 'docs/release/v2.5.0-external-integration-package-readiness.md',
 'docs/release/v2.5.0-release-notes.md']

LOCAL_RUNTIME_ALLOWLIST_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".venv",
}
LOCAL_RUNTIME_ALLOWLIST_FILES = {".coverage"}

FORBIDDEN_FILE_NAMES = {".env"}
FORBIDDEN_FILE_SUFFIXES_OUTSIDE_LOCAL_RUNTIME = {".pyc", ".pyo"}

REQUIRED_README_PHRASES = [
    "Cornerstone turns scattered organizational knowledge into a reviewed, explainable ontology graph",
    "Scattered Sources",
    "EvidenceFragments",
    "ConceptCandidates / RelationCandidates",
    "reviewed official ontology graph is the Single Source of Truth",
    "Example: Settlement",
    "docs/product/00-product-overview.md",
    "docs/product/03-settlement-walkthrough.md",
    "docs/product/06-trust-model.md",
    "docs/product/08-operator-quickstart.md",
    "GET  /v1/ontology/graph?concept=Settlement&depth=1&mode=official",
    "GET  /v1/ontology/ssot/readiness?focusConcept=Settlement&depth=1&mode=official",
    "v2.0.4 — Forward Roadmap Goals and Measurable Checklists",
    "docs/50-product-documentation-layer-v2.0.2.md",
    "docs/51-dependency-complete-verification-v2.0.3.md",
    "docs/52-forward-roadmap-goals-checklists-v2.0.4.md",
    "docs/57-external-integration-package-v2.5.0.md",
    "docs/roadmap/v2.1.0-live-llm-ontology-provider.md",
    "docs/roadmap/v2.5.0-frontend-mvp-external-integration-package.md",
    "scripts/run_dependency_complete_verification.py --plan-only",
    "docs/48-version-chronicle-and-measurable-checklists-v2.0.0.md",
]

IMPLEMENTED_ROADMAP_DOC_REQUIRED_PHRASES = {
    "docs/53-live-llm-ontology-provider-v2.1.0.md": ["Version goal", "Confirmed non-goal", "V210-15", "candidate-only"],
    "docs/54-review-operator-experience-v2.2.0.md": ["Version goal", "Confirmed non-goal", "V220-14", "review queue"],
    "docs/55-graph-visualization-contract-v2.3.0.md": ["Version goal", "Confirmed non-goal", "V230-13", "visualization"],
    "docs/56-connector-expansion-live-proof-hardening-v2.4.0.md": ["Version goal", "Confirmed non-goal", "V240-13", "mutatesOfficialGraph=false"],
    "docs/57-external-integration-package-v2.5.0.md": ["Version goal", "Confirmed non-goal", "V250-13", "external_integration_package"],
    "docs/release/v2.1.0-live-llm-ontology-provider-readiness.md": ["V210-RDY-01", "candidate"],
    "docs/release/v2.2.0-review-operator-experience-readiness.md": ["V220-RDY-01", "Review gates"],
    "docs/release/v2.3.0-graph-visualization-contract-readiness.md": ["V230-RDY-01", "visualization"],
    "docs/release/v2.4.0-connector-expansion-live-proof-readiness.md": ["V240-RDY-01", "Official graph mutation"],
    "docs/release/v2.5.0-external-integration-package-readiness.md": ["V250-RDY-01", "reviewGateBypassAllowed=false"],
    "docs/release/v2.5.0-release-notes.md": ["Integration Package", "candidate bypass"],
}

PRODUCT_DOC_REQUIRED_PHRASES = {
    "docs/product/README.md": ["Core product sentence", "Core trust sentence", "PROD-DOC-README-01"],
    "docs/product/00-product-overview.md": ["Cornerstone turns scattered", "Single Source of Truth", "PROD-OVERVIEW-01"],
    "docs/product/01-user-problem-and-value.md": ["chatbots", "reviewed truth", "PROD-VALUE-01"],
    "docs/product/02-how-cornerstone-works.md": ["Source ingestion", "Human review", "PROD-HOW-01"],
    "docs/product/03-settlement-walkthrough.md": ["Settlement", "official graph", "PROD-SETTLEMENT-01"],
    "docs/product/04-ontology-graph-explained.md": ["Concept", "Relation", "PROD-GRAPH-01"],
    "docs/product/05-user-roles-and-workflows.md": ["Source Admin", "Reviewer", "PROD-ROLES-01"],
    "docs/product/06-trust-model.md": ["Extractor output is not", "Official graph", "PROD-TRUST-01"],
    "docs/product/07-product-vs-chatbot-rag-wiki.md": ["RAG chatbot", "Cornerstone difference", "PROD-POS-01"],
    "docs/product/08-operator-quickstart.md": ["SSOT readiness", "manual source", "PROD-OPS-01"],
    "docs/product/09-product-glossary.md": ["ConceptCandidate", "RelationCandidate", "PROD-GLOSSARY-01"],
}

ROADMAP_DOC_REQUIRED_PHRASES = {
    "docs/roadmap/README.md": ["v2.1.0 — Live LLM ontology provider", "ROADMAP-README-01"],
    "docs/roadmap/v2.1.0-live-llm-ontology-provider.md": ["Version goal", "Confirmed non-goal", "V210-15"],
    "docs/roadmap/v2.2.0-review-operator-experience.md": ["Version goal", "Confirmed non-goal", "V220-14"],
    "docs/roadmap/v2.3.0-graph-visualization-contract.md": ["Version goal", "Confirmed non-goal", "V230-13"],
    "docs/roadmap/v2.4.0-connector-expansion-live-proof-hardening.md": ["Version goal", "Confirmed non-goal", "V240-13"],
    "docs/roadmap/v2.5.0-frontend-mvp-external-integration-package.md": ["Version goal", "Confirmed non-goal", "V250-13"],
}


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _has_allowed_runtime_part(path: Path) -> bool:
    return any(part in LOCAL_RUNTIME_ALLOWLIST_PARTS for part in path.parts)


def check_versions(errors: list[str]) -> None:
    pyproject = _read("pyproject.toml")
    init_py = _read("src/cornerstone/__init__.py")
    schemas = _read("src/cornerstone/schemas.py")
    chronicle = _read("docs/48-version-chronicle-and-measurable-checklists-v2.0.0.md")
    roadmap_doc = _read("docs/52-forward-roadmap-goals-checklists-v2.0.4.md")
    integration_doc = _read("docs/57-external-integration-package-v2.5.0.md")
    if f'version = "{EXPECTED_VERSION}"' not in pyproject:
        errors.append("pyproject.toml version does not match expected release-candidate base version")
    if f'__version__ = "{EXPECTED_VERSION}"' not in init_py:
        errors.append("src/cornerstone/__init__.py version does not match expected release-candidate base version")
    if f'release_version: str = "{EXPECTED_VERSION}"' not in schemas:
        errors.append("OntologySsotReadinessResponse release_version does not match expected version")
    for phrase in [f"v{EXPECTED_VERSION}", "V250-01", "V250-13", "External Integration Package"]:
        if phrase not in chronicle and phrase not in integration_doc:
            errors.append(f"v2.5.0 chronicle/version documentation missing: {phrase}")


def check_required_docs(errors: list[str]) -> None:
    for doc in REQUIRED_DOCS:
        path = ROOT / doc
        if not path.exists():
            errors.append(f"missing required release doc: {doc}")
        elif path.stat().st_size < 200:
            errors.append(f"release doc is unexpectedly small: {doc}")


def check_product_docs(errors: list[str]) -> None:
    for rel_path, phrases in PRODUCT_DOC_REQUIRED_PHRASES.items():
        content = _read(rel_path)
        if "acceptance checklist" not in content.lower():
            errors.append(f"product doc missing acceptance checklist: {rel_path}")
        for phrase in phrases:
            if phrase not in content:
                errors.append(f"product doc {rel_path} missing required phrase: {phrase}")


def check_forward_roadmap_docs(errors: list[str]) -> None:
    roadmap_release = _read("docs/52-forward-roadmap-goals-checklists-v2.0.4.md")
    readiness_doc = _read("docs/release/v2.0.4-forward-roadmap-readiness.md")
    release_notes = _read("docs/release/v2.0.4-release-notes.md")
    for phrase in ["Version goal", "Confirmed non-goal", "Measurable acceptance checklist", "V204-01", "V204-12"]:
        if phrase not in roadmap_release:
            errors.append(f"v2.0.4 roadmap release doc missing: {phrase}")
    for phrase in ["V204-RDY-01", "V204-RDY-10"]:
        if phrase not in readiness_doc:
            errors.append(f"v2.0.4 roadmap readiness doc missing: {phrase}")
    for phrase in ["No runtime behavior change", "v2.1.0 — Live LLM ontology provider"]:
        if phrase not in release_notes:
            errors.append(f"v2.0.4 roadmap release notes missing: {phrase}")
    for rel_path, phrases in ROADMAP_DOC_REQUIRED_PHRASES.items():
        content = _read(rel_path)
        if rel_path == "docs/roadmap/README.md":
            if "Roadmap acceptance checklist" not in content:
                errors.append(f"roadmap README missing acceptance checklist: {rel_path}")
        else:
            for required_section in ["Version goal", "Confirmed non-goal", "Measurable acceptance checklist", "Verification checklist", "Exit criteria", "Next-version handoff"]:
                if required_section not in content:
                    errors.append(f"roadmap doc {rel_path} missing section: {required_section}")
        for phrase in phrases:
            if phrase not in content:
                errors.append(f"roadmap doc {rel_path} missing required phrase: {phrase}")


def check_implemented_roadmap_docs(errors: list[str]) -> None:
    for rel_path, phrases in IMPLEMENTED_ROADMAP_DOC_REQUIRED_PHRASES.items():
        content = _read(rel_path)
        for phrase in phrases:
            if phrase not in content:
                errors.append(f"implemented roadmap doc {rel_path} missing required phrase: {phrase}")


def check_dependency_complete_verification(errors: list[str]) -> None:
    verification_doc = _read("docs/51-dependency-complete-verification-v2.0.3.md")
    readiness_doc = _read("docs/release/v2.0.3-verification-readiness.md")
    runner = _read("scripts/run_dependency_complete_verification.py")
    plan_module = _read("src/cornerstone/verification/dependency_complete.py")
    workflow = _read(".github/workflows/dependency-complete-verification.yml")
    for phrase in [
        "Version goal",
        "Confirmed non-goal",
        "Measurable acceptance checklist",
        "V203-01",
        "V203-CMD-13",
        "--strict --confirm-live-db",
    ]:
        if phrase not in verification_doc:
            errors.append(f"v2.0.3 verification doc missing: {phrase}")
    for phrase in ["V203-RDY-01", "V203-RDY-10"]:
        if phrase not in readiness_doc:
            errors.append(f"v2.0.3 readiness doc missing: {phrase}")
    for phrase in ["dependency_complete_command_plan", "V203-CMD-01-python-version", "V203-CMD-13-duplicate-audit"]:
        if phrase not in plan_module:
            errors.append(f"v2.0.3 command plan missing: {phrase}")
    for phrase in ["--strict", "--confirm-live-db", "--plan-only"]:
        if phrase not in runner:
            errors.append(f"v2.0.3 runner missing: {phrase}")
    for phrase in ["pgvector/pgvector:pg17", "run_dependency_complete_verification.py --strict --confirm-live-db"]:
        if phrase not in workflow:
            errors.append(f"v2.0.3 workflow missing: {phrase}")


def check_readme(errors: list[str]) -> None:
    readme = _read("README.md")
    for phrase in REQUIRED_README_PHRASES:
        if phrase not in readme:
            errors.append(f"README.md missing required phrase: {phrase}")


def check_api_freeze(errors: list[str]) -> None:
    api_freeze = _read("docs/release/api-freeze-review.md")
    required = [
        "POST /v1/sources/{sourceId}/oauth/complete",
        "POST /v1/sources/{sourceId}/sync",
        "POST /v1/manual-sources/{notionSourceId}/sync",
        "GET  /v1/context/query",
        "POST /v1/evaluations/tasks/{taskId}/run",
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
    ]
    for phrase in required:
        if phrase not in api_freeze:
            errors.append(f"API freeze review missing: {phrase}")


def check_package_hygiene(errors: list[str]) -> None:
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT).as_posix()
        is_allowed_runtime = _has_allowed_runtime_part(path) or path.name in LOCAL_RUNTIME_ALLOWLIST_FILES
        if path.is_file():
            if path.name in FORBIDDEN_FILE_NAMES:
                errors.append(f"forbidden local secret/config file present: {rel}")
            if path.suffix in FORBIDDEN_FILE_SUFFIXES_OUTSIDE_LOCAL_RUNTIME and not is_allowed_runtime:
                errors.append(f"forbidden compiled Python file outside local runtime cache: {rel}")


def check_live_proof_record(errors: list[str]) -> None:
    proof = _read("docs/live-proof-records/2026-04-28-v0.13.1-blocker-fix.md")
    required = [
        "230 passed",
        "Live PostgreSQL verification passed: 5 tests, 0 skipped.",
        "runnerArtifactCount: 1",
        "runnerEvidenceFragmentCount: 5",
        "groundedContextTaskSuccessRate: 1.0",
        "direct_notion_source_409",
        "fake_oauth_completion_404",
        "legacy_source_sync_404",
        "manual_sync_on_notion_409",
        "weak_evaluation_task_422",
    ]
    for phrase in required:
        if phrase not in proof:
            errors.append(f"v0.13.1 proof record missing: {phrase}")


def check_rc_docs(errors: list[str]) -> None:
    rc_doc = _read("docs/28-backend-v1.0.0-rc.1.md")
    required = ["v1.0.0-rc.1", "same verified commit as v0.13.1", "No new backend feature work"]
    for phrase in required:
        if phrase not in rc_doc:
            errors.append(f"RC-1 doc missing: {phrase}")


def main() -> int:
    errors: list[str] = []
    check_versions(errors)
    check_required_docs(errors)
    check_product_docs(errors)
    check_forward_roadmap_docs(errors)
    check_implemented_roadmap_docs(errors)
    check_dependency_complete_verification(errors)
    check_readme(errors)
    check_api_freeze(errors)
    check_live_proof_record(errors)
    check_rc_docs(errors)
    check_package_hygiene(errors)

    if errors:
        print("release-candidate check: failed")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"release-candidate check: passed for v{EXPECTED_VERSION}")
    print(f"required_docs={len(REQUIRED_DOCS)}")
    print("product_docs=present")
    print("dependency_complete_verification=present")
    print("forward_roadmap_docs=present")
    print("package_hygiene=passed")
    print("api_freeze_review=present")
    print("live_proof_record=present")
    print("rc_tag_plan=present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
