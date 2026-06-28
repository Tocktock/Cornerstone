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
VS2_CURRENT_REPORT = ROOT / "docs/verification-reports/VS2_POLICY_TENANCY_EGRESS_CURRENT_VERIFICATION_REPORT_2026-06-28.md"
VS2_SCENARIO_REPORT = ROOT / "reports/scenario/vs2-policy-tenancy-egress-2026-06-19.json"
VS2_SCENARIO_SPECIFIC_EVIDENCE = ROOT / "reports/security/vs2-scenario-specific-evidence.json"
VS2_EVIDENCE_MANIFEST = ROOT / "reports/security/vs2/evidence-manifest.json"


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
        "shared_object_count": 5,
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
                if isinstance(section.get("object"), dict):
                    refs.append(section["object"])
                refs.extend(section.get("items", []))
                refs.extend(section.get("entries", []))
        unique_ref_paths = {
            ref.get("path")
            for ref in refs
            if isinstance(ref, dict) and isinstance(ref.get("path"), str)
        }
        if len(unique_ref_paths) != summary.get("shared_object_count"):
            errors.append(
                "shared evidence object ref count expected "
                f"{summary.get('shared_object_count')}, found {len(unique_ref_paths)}"
            )
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


def _verify_vs2_status_reconciliation(errors: list[str], readme: str) -> None:
    report_text = _read(VS2_CURRENT_REPORT, errors)
    for token in [
        "Current local deterministic VS2 verification report.",
        "reports/scenario/vs2-policy-tenancy-egress-2026-06-19.json",
        "VS2_POLICY_TENANCY_EGRESS_SCENARIO_SPECIFIC_REMEDIATION_REPORT_2026-06-19.md",
        "prior remediation/baseline document, not the current generated-status authority",
        "The earlier remediation status represented a gap-oriented state",
        "Human-owned H rows remain unchanged as `HUMAN_REQUIRED`",
        "This command is the generated-status authority.",
    ]:
        _require_text(errors, "VS2 current verification report", report_text, token)
    for token in [
        "| Scenario rows | 93 |",
        "| PASS | 86 |",
        "| HUMAN_REQUIRED | 7 |",
        "| NOT_VERIFIED | 0 |",
        "| FAIL | 0 |",
        "| `MUST_PASS` | 70 | 0 | 0 | 0 |",
        "| `REGRESSION` | 16 | 0 | 0 | 0 |",
    ]:
        _require_text(errors, "VS2 current verification report", report_text, token)
    _require_text(errors, "README", readme, _relative(VS2_CURRENT_REPORT))

    scenario_report = _load_json(VS2_SCENARIO_REPORT, errors)
    scenario_specific = _load_json(VS2_SCENARIO_SPECIFIC_EVIDENCE, errors)
    evidence_manifest = _load_json(VS2_EVIDENCE_MANIFEST, errors)
    if not scenario_report or not scenario_specific or not evidence_manifest:
        return

    rows = scenario_report.get("scenario_results")
    if not isinstance(rows, list):
        errors.append("VS2 scenario report scenario_results must be a list")
        return
    if len(rows) != 93:
        errors.append(f"VS2 scenario report expected 93 rows, found {len(rows)}")

    status_counts: dict[str, int] = {}
    type_status_counts: dict[tuple[str, str], int] = {}
    owner_counts: dict[str, int] = {}
    seen_ids: set[str] = set()
    ai_ids: set[str] = set()
    human_ids: set[str] = set()
    for row in rows:
        if not isinstance(row, dict):
            errors.append("VS2 scenario row must be an object")
            continue
        scenario_id = str(row.get("scenario_id") or row.get("id") or "")
        status = str(row.get("status") or "")
        scenario_type = str(row.get("type") or "")
        owner = str(row.get("owner") or "")
        if not scenario_id:
            errors.append("VS2 scenario row missing scenario_id")
            continue
        if scenario_id in seen_ids:
            errors.append(f"duplicate VS2 scenario row: {scenario_id}")
        seen_ids.add(scenario_id)
        status_counts[status] = status_counts.get(status, 0) + 1
        type_status_counts[(scenario_type, status)] = type_status_counts.get((scenario_type, status), 0) + 1
        owner_counts[owner] = owner_counts.get(owner, 0) + 1

        evidence_paths = row.get("evidence_paths", [])
        evidence_hashes = row.get("evidence_hashes", [])
        if owner == "AI":
            ai_ids.add(scenario_id)
            if status != "PASS":
                errors.append(f"AI-owned VS2 row must be PASS: {scenario_id} -> {status}")
            if not isinstance(evidence_paths, list) or not evidence_paths:
                errors.append(f"AI-owned VS2 row missing evidence_paths: {scenario_id}")
                continue
            if not isinstance(evidence_hashes, list) or len(evidence_hashes) != len(evidence_paths):
                errors.append(f"AI-owned VS2 row evidence_hashes length mismatch: {scenario_id}")
            expected_paths = {
                f"reports/security/vs2/evidence/{scenario_id}.json",
                "reports/security/vs2/evidence-manifest.json",
                "reports/security/vs2/post-commit-rollup.json",
                "reports/security/vs2-scenario-specific-evidence.json",
            }
            missing_paths = expected_paths.difference(str(path) for path in evidence_paths)
            if missing_paths:
                errors.append(f"AI-owned VS2 row missing required evidence paths {sorted(missing_paths)}: {scenario_id}")
            for path_value, hash_value in zip(evidence_paths, evidence_hashes):
                evidence_path = ROOT / str(path_value)
                if not evidence_path.exists():
                    errors.append(f"VS2 evidence path missing for {scenario_id}: {path_value}")
                elif _file_sha256(evidence_path) != hash_value:
                    errors.append(f"VS2 evidence hash mismatch for {scenario_id}: {path_value}")
        elif owner == "Human":
            human_ids.add(scenario_id)
            if status != "HUMAN_REQUIRED":
                errors.append(f"human-owned VS2 row must remain HUMAN_REQUIRED: {scenario_id} -> {status}")
            if scenario_type != "HUMAN_REQUIRED":
                errors.append(f"human-owned VS2 row type must be HUMAN_REQUIRED: {scenario_id} -> {scenario_type}")
        else:
            errors.append(f"VS2 row owner must be AI or Human: {scenario_id} -> {owner}")

    expected_status_counts = {"PASS": 86, "HUMAN_REQUIRED": 7}
    if status_counts != expected_status_counts:
        errors.append(f"VS2 status counts expected {expected_status_counts}, found {status_counts}")
    expected_owner_counts = {"AI": 86, "Human": 7}
    if owner_counts != expected_owner_counts:
        errors.append(f"VS2 owner counts expected {expected_owner_counts}, found {owner_counts}")
    expected_type_status_counts = {
        ("MUST_PASS", "PASS"): 70,
        ("REGRESSION", "PASS"): 16,
        ("HUMAN_REQUIRED", "HUMAN_REQUIRED"): 7,
    }
    if type_status_counts != expected_type_status_counts:
        errors.append(f"VS2 type/status counts expected {expected_type_status_counts}, found {type_status_counts}")

    registry = scenario_specific.get("scenario_check_registry")
    if not isinstance(registry, list):
        errors.append("VS2 scenario-specific evidence registry must be a list")
    elif set(registry) != ai_ids:
        errors.append("VS2 scenario-specific evidence registry must exactly match the 86 AI-owned scenario ids")
    scenario_evidence = scenario_specific.get("scenario_evidence")
    if not isinstance(scenario_evidence, dict):
        errors.append("VS2 scenario-specific evidence must contain a scenario_evidence object")
    elif set(scenario_evidence) != ai_ids:
        errors.append("VS2 scenario-specific evidence keys must exactly match the 86 AI-owned scenario ids")
    if scenario_specific.get("evidence_manifest") != _relative(VS2_EVIDENCE_MANIFEST):
        errors.append("VS2 scenario-specific evidence must point to the committed evidence manifest")
    elif scenario_specific.get("evidence_manifest_sha256") != _file_sha256(VS2_EVIDENCE_MANIFEST):
        errors.append("VS2 scenario-specific evidence manifest sha256 mismatch")

    if evidence_manifest.get("artifact_count") != 86:
        errors.append(f"VS2 evidence manifest artifact_count expected 86, found {evidence_manifest.get('artifact_count')!r}")
    raw_artifacts = evidence_manifest.get("raw_scenario_artifacts")
    if not isinstance(raw_artifacts, list):
        errors.append("VS2 evidence manifest raw_scenario_artifacts must be a list")
    else:
        raw_paths = {str(entry.get("path")) for entry in raw_artifacts if isinstance(entry, dict)}
        expected_raw_paths = {f"reports/security/vs2/evidence/{scenario_id}.json" for scenario_id in ai_ids}
        if raw_paths != expected_raw_paths:
            errors.append("VS2 evidence manifest raw_scenario_artifacts must exactly match the 86 AI evidence files")
        for entry in raw_artifacts:
            if not isinstance(entry, dict):
                errors.append("VS2 evidence manifest raw_scenario_artifact entries must be objects")
                continue
            artifact_path = ROOT / str(entry.get("path", ""))
            if not artifact_path.exists():
                errors.append(f"missing VS2 raw scenario artifact: {entry.get('path')}")
            elif entry.get("sha256") != _file_sha256(artifact_path):
                errors.append(f"VS2 raw scenario artifact sha256 mismatch: {entry.get('path')}")


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
    _verify_vs2_status_reconciliation(errors, readme)
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
        "VS2 status reconciled, controlled sink proof present, "
        "unresolved split/monolith boundaries explicit)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
