#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

FEEDBACK_REPORT = ROOT / "docs/verification-reports/CONNECTOR_HUB_PR20_FEEDBACK_RESOLUTION_2026-06-28.md"
REVIEWER_GUIDE = ROOT / "docs/verification-reports/CONNECTOR_HUB_REVIEWER_GUIDE.md"
CONTRACT = ROOT / "docs/scenario-contracts/CONNECTOR_HUB_APPLICATION_CONTRACT.md"
README = ROOT / "README.md"
LOCAL_GATE = ROOT / "scripts/verify_connectorhub_local_evidence.sh"
SPLIT_MANIFEST = ROOT / "docs/verification-reports/CONNECTOR_HUB_REVIEW_SPLIT_MANIFEST_2026-06-28.json"
COMPACT_MANIFEST = ROOT / "reports/scenario/connector-contract-adapter/manifest-2026-06-23.json"
VS2_REHEARSAL_REPORT = ROOT / "reports/security/vs2-production-like-integration-2026-06-27.json"


def _relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def _read(path: Path, errors: list[str]) -> str:
    if not path.exists():
        errors.append(f"missing required file: {_relative(path)}")
        return ""
    return path.read_text()


def _load_json(path: Path, errors: list[str]) -> dict[str, Any]:
    if not path.exists():
        errors.append(f"missing required JSON file: {_relative(path)}")
        return {}
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        errors.append(f"invalid JSON in {_relative(path)}: {error}")
        return {}
    if not isinstance(payload, dict):
        errors.append(f"{_relative(path)} must contain a JSON object")
        return {}
    return payload


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json_sha256(path: Path) -> str:
    payload = json.loads(path.read_text())
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_text(errors: list[str], label: str, text: str, token: str) -> None:
    if token not in text:
        errors.append(f"{label} missing required text: {token}")


def _reject_text(errors: list[str], label: str, text: str, pattern: str) -> None:
    if re.search(pattern, text, flags=re.IGNORECASE):
        errors.append(f"{label} still contains prohibited pattern: {pattern}")


def _verify_feedback_report(errors: list[str], text: str) -> None:
    expected_findings = [
        "B1: draft/no visible CI",
        "B2: contract mixed status/PASS",
        "B3: VS2 status inconsistent",
        "B4: non-portable evidence paths",
        "B5: PR too large",
        "M1: `connector.py`/test monolith",
        "M2: Action lane boundary",
        "M3: weak simulated egress checks",
        "M4: milestone wording",
        "M5: generated evidence hard to review",
        "N1: README as status source",
        "N2: downloaded source docs",
        "N3: H04 substitution ADR",
        "N4: trimmed stdout proof risk",
    ]
    for finding in expected_findings:
        _require_text(errors, "feedback report", text, finding)
    for token in [
        "Keep PR #20 draft",
        "No GitHub Actions workflow is added",
        "CORNERSTONE_SKIP_VS2_REGRESSION_TESTS=1",
        "PR20 feedback-response verifier",
        "PR #20 should remain draft and `needs-follow-up`",
        "`connector.py` and `tests/scenario/test_connectorhub_cli.py` still need the ADR-0008 module/test split",
    ]:
        _require_text(errors, "feedback report", text, token)
    _reject_text(errors, "feedback report", text, r"b343393|full ConnectorHub CLI suite, compact report unittest, scaffold suite")


def _verify_local_gate(errors: list[str], text: str) -> None:
    for token in [
        "run env CORNERSTONE_SKIP_VS2_REGRESSION_TESTS=1 python3 -m unittest tests.scenario.test_connectorhub_cli",
        "run env CORNERSTONE_SKIP_VS2_REGRESSION_TESTS=1 python3 -m unittest tests.scenario.test_scaffold_cli",
        "run make verify-vs2-production-like",
    ]:
        _require_text(errors, "local evidence gate", text, token)
    _reject_text(errors, "local evidence gate", text, r"\.github/workflows|gh workflow|github actions")


def _verify_contract_and_readme(errors: list[str], contract: str, readme: str) -> None:
    header = "\n".join(contract.splitlines()[:80])
    _require_text(errors, "ConnectorHub contract header", header, "Status:** Frozen task-scoped ConnectorHub adoption contract.")
    _require_text(errors, "ConnectorHub contract header", header, "Implementation status:")
    _reject_text(errors, "ConnectorHub contract header", header, r"current matrix state")
    for token in [
        "CS-CH-031 -> Execute a declared local fixture Action and re-ingest its fixture outcome",
        "CS-CH-031` covers only a deterministic local fixture Action and fixture outcome re-ingest",
    ]:
        _require_text(errors, "ConnectorHub contract", contract, token)
    _require_text(errors, "README", readme, "current generated status is recorded in")
    _reject_text(errors, "README", readme, r"current generated status is\s+\d+\s+PASS")
    _reject_text(errors, "README", readme, r"ConnectorHub milestone complete")


def _verify_compact_evidence(errors: list[str]) -> None:
    old_reports = sorted((ROOT / "reports/scenario").glob("connector-contract-adapter*.json"))
    if old_reports:
        sample = ", ".join(_relative(path) for path in old_reports[:5])
        errors.append(f"old full connector-contract-adapter reports remain at reports/scenario root: {sample}")
    manifest = _load_json(COMPACT_MANIFEST, errors)
    if not manifest:
        return
    summary = manifest.get("summary", {})
    expected_summary = {
        "source_full_report_count": 41,
        "compact_report_count": 41,
        "focused_scenario_count": 40,
        "shared_object_count": 676,
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            errors.append(f"compact manifest summary {key} expected {expected}, found {summary.get(key)!r}")
    if manifest.get("layout") != "content_addressed_objects_v1":
        errors.append("compact manifest layout must be content_addressed_objects_v1")
    shared = manifest.get("shared_evidence", {})
    shared_path = ROOT / str(shared.get("path", ""))
    if not shared_path.exists():
        errors.append("compact shared evidence index is missing")
    elif shared.get("sha256") != _file_sha256(shared_path):
        errors.append("compact shared evidence index sha256 mismatch")
    scenario_reports = manifest.get("scenario_reports", [])
    if not isinstance(scenario_reports, list) or len(scenario_reports) != 40:
        errors.append("compact manifest must list 40 scenario reports")
        return
    for entry in scenario_reports:
        if not isinstance(entry, dict):
            errors.append("compact scenario report entry must be an object")
            continue
        path = ROOT / str(entry.get("path", ""))
        if not path.exists():
            errors.append(f"missing compact scenario report: {entry.get('path')}")
            continue
        if entry.get("sha256") != _file_sha256(path):
            errors.append(f"compact scenario report sha256 mismatch: {entry.get('path')}")
    aggregate = manifest.get("aggregate_report", {})
    aggregate_path = ROOT / str(aggregate.get("path", ""))
    if not aggregate_path.exists():
        errors.append("compact aggregate report is missing")
    elif aggregate.get("sha256") != _file_sha256(aggregate_path):
        errors.append("compact aggregate report sha256 mismatch")

    if shared_path.exists():
        shared_index = _load_json(shared_path, errors)
        refs = []
        for section in (shared_index.get("sections") or {}).values():
            if isinstance(section, dict):
                refs.extend(section.get("items", []))
                refs.extend(section.get("entries", []))
        for ref in refs:
            if not isinstance(ref, dict):
                errors.append("shared evidence object ref must be an object")
                continue
            object_path = ROOT / str(ref.get("path", ""))
            if not object_path.exists():
                errors.append(f"missing shared evidence object: {ref.get('path')}")
            elif ref.get("sha256") != _canonical_json_sha256(object_path):
                errors.append(f"shared evidence object canonical sha mismatch: {ref.get('path')}")


def _verify_vs2_rehearsal(errors: list[str]) -> None:
    compose = _read(ROOT / "compose.vs2.yml", errors)
    for token in ["provider-sink:", "forbidden-sink:", "cornerstone_external_test"]:
        _require_text(errors, "VS2 compose", compose, token)
    report = _load_json(VS2_REHEARSAL_REPORT, errors)
    if not report:
        return
    if report.get("status") != "passed":
        errors.append("VS2 production-like rehearsal report must be passed")
    scenario_results = report.get("scenario_results")
    if not isinstance(scenario_results, list) or len(scenario_results) != 8:
        errors.append("VS2 production-like rehearsal must report 8 scenario rows")
    elif any(row.get("status") != "PASS" for row in scenario_results if isinstance(row, dict)):
        errors.append("VS2 production-like rehearsal scenario rows must all be PASS")
    controlled = report.get("controlled_sink_egress", {})
    if controlled.get("status") != "passed":
        errors.append("controlled sink egress proof must be passed")
    checks = controlled.get("checks", {})
    for key in [
        "provider_sink_service_reached_once",
        "forbidden_sink_denied_by_policy",
        "forbidden_sink_not_contacted",
        "opa_allow_decision_recorded",
    ]:
        if checks.get(key) is not True:
            errors.append(f"controlled sink egress check must be true: {key}")
    evidence = controlled.get("log_evidence", {})
    if evidence.get("provider_sink_token_count") != 1:
        errors.append("controlled sink provider_sink_token_count must be 1")
    if evidence.get("forbidden_sink_token_count") != 0:
        errors.append("controlled sink forbidden_sink_token_count must be 0")


def _verify_reviewer_surfaces(errors: list[str], guide: str, split_manifest: dict[str, Any]) -> None:
    for token in [
        "scripts/verify_connectorhub_local_evidence.sh",
        "python3 scripts/verify_connectorhub_review_split.py",
        "scripts/verify_connectorhub_local_evidence.sh --strict",
        "It is planning evidence only; it does not claim the PR has already been split.",
        "The current branch does not claim the `connector.py` or scenario-test split is already complete.",
    ]:
        _require_text(errors, "reviewer guide", guide, token)
    notes = split_manifest.get("notes", [])
    if not isinstance(notes, list):
        errors.append("split manifest notes must be a list")
    else:
        for token in [
            "It does not claim the branch has already been split.",
            "It does not claim connector.py or test_connectorhub_cli.py refactoring is complete.",
            "It does not require or add GitHub Actions workflows.",
        ]:
            if token not in notes:
                errors.append(f"split manifest missing note: {token}")


def _verify_trimmed_stdout_boundary(errors: list[str]) -> None:
    main_py = _read(ROOT / "packages/cornerstone_cli/main.py", errors)
    test_py = _read(ROOT / "tests/scenario/test_connectorhub_cli.py", errors)
    for token in ["full_report_path", "full_report_sha256"]:
        _require_text(errors, "main.py command evidence filter", main_py, token)
        _require_text(errors, "ConnectorHub CLI tests", test_py, token)


def _verify_path_portability(errors: list[str]) -> None:
    # The committed compact/review surfaces may keep absolute paths only as
    # explicit historical transcript metadata. This verifier focuses on the
    # current ConnectorHub review artifacts rather than older VS0/VS1/VS2
    # generated transcripts outside PR20's feedback scope.
    scoped_files = [
        CONTRACT,
        FEEDBACK_REPORT,
        REVIEWER_GUIDE,
        ROOT / "docs/verification-reports/CONNECTOR_HUB_ENGINEERING_TRAIL_INDEX_2026-06-24.md",
        ROOT / "reports/scenario/connectorhub-scenario-delivery-unit-manifest-2026-06-24.json",
    ]
    for path in scoped_files:
        text = _read(path, errors)
        if "/Users/" in text and "historical" not in text and "regenerable" not in text:
            errors.append(f"{_relative(path)} contains /Users/ without historical/regenerable boundary text")
    source_archive = ROOT / "docs/archive/research"
    for name in [
        "CornerStone_ConnectorHub_Application_Guide_2026-06-22.md",
        "CornerStone_ConnectorHub_Test_Scenario_Implementation_Document_2026-06-22.md",
    ]:
        if not (source_archive / name).exists():
            errors.append(f"missing archived source document: docs/archive/research/{name}")


def main() -> int:
    errors: list[str] = []
    feedback = _read(FEEDBACK_REPORT, errors)
    guide = _read(REVIEWER_GUIDE, errors)
    contract = _read(CONTRACT, errors)
    readme = _read(README, errors)
    local_gate = _read(LOCAL_GATE, errors)
    split_manifest = _load_json(SPLIT_MANIFEST, errors)

    _verify_feedback_report(errors, feedback)
    _verify_local_gate(errors, local_gate)
    _verify_contract_and_readme(errors, contract, readme)
    _verify_compact_evidence(errors)
    _verify_vs2_rehearsal(errors)
    _verify_reviewer_surfaces(errors, guide, split_manifest)
    _verify_trimmed_stdout_boundary(errors)
    _verify_path_portability(errors)

    if errors:
        print("FAIL: ConnectorHub PR20 feedback verification failed")
        for error in errors:
            print(f"- {error}")
        return 1

    print(
        "PASS: ConnectorHub PR20 feedback response verified "
        "(14 findings covered, local gate guarded, compact evidence hashed, "
        "controlled sink proof present, unresolved split/monolith boundaries explicit)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
