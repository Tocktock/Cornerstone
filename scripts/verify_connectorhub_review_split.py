#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPLIT_MANIFEST = ROOT / "docs/verification-reports/CONNECTOR_HUB_REVIEW_SPLIT_MANIFEST_2026-06-28.json"
EXPECTED_AI_SCENARIOS = {f"CS-CH-{number:03d}" for number in range(1, 41)}
EXPECTED_HUMAN_SCENARIOS = {f"CS-CH-H{number:02d}" for number in range(1, 8)}
EXPECTED_SLICE_IDS = [
    "PR20-A",
    "PR20-B",
    "PR20-C",
    "PR20-D",
    "PR20-E",
    "PR20-F",
    "PR20-G",
    "PR20-H",
]
EXPECTED_RUNTIME_SLICE_COVERAGE = {
    "PR20-C": {f"CS-CH-{number:03d}" for number in range(1, 15)},
    "PR20-D": {f"CS-CH-{number:03d}" for number in range(15, 21)},
    "PR20-E": {f"CS-CH-{number:03d}" for number in range(21, 29)},
    "PR20-F": {f"CS-CH-{number:03d}" for number in range(29, 34)},
    "PR20-G": {f"CS-CH-{number:03d}" for number in range(34, 41)},
}
REQUIRED_TARGET_MODULES = {
    "packages/cornerstone_cli/connector/contracts.py",
    "packages/cornerstone_cli/connector/raw_access.py",
    "packages/cornerstone_cli/connector/github_readonly.py",
    "packages/cornerstone_cli/connector/capture_macos.py",
    "packages/cornerstone_cli/connector/capture_chrome.py",
    "packages/cornerstone_cli/connector/watch_rules.py",
    "packages/cornerstone_cli/connector/watch_results.py",
    "packages/cornerstone_cli/connector/action_preflight.py",
    "packages/cornerstone_cli/connector/audit_bridge.py",
    "packages/cornerstone_cli/connector/human_gates.py",
    "packages/cornerstone_cli/connector/validation.py",
}
REQUIRED_TARGET_TESTS = {
    "tests/scenario/connectorhub/test_setup_delivery.py",
    "tests/scenario/connectorhub/test_github_readonly.py",
    "tests/scenario/connectorhub/test_capture_watch.py",
    "tests/scenario/connectorhub/test_action_lane.py",
    "tests/scenario/connectorhub/test_scope_credential_egress_audit.py",
    "tests/scenario/connectorhub/test_human_gates.py",
}


def _fail(errors: list[str]) -> int:
    print("FAIL: ConnectorHub review split verification failed")
    for error in errors:
        print(f"- {error}")
    return 1


def _load_manifest(errors: list[str]) -> dict[str, object] | None:
    if not SPLIT_MANIFEST.exists():
        errors.append(f"missing split manifest: {SPLIT_MANIFEST.relative_to(ROOT)}")
        return None
    try:
        payload = json.loads(SPLIT_MANIFEST.read_text())
    except json.JSONDecodeError as error:
        errors.append(f"split manifest is invalid JSON: {error}")
        return None
    if not isinstance(payload, dict):
        errors.append("split manifest must be a JSON object")
        return None
    return payload


def _validate_paths(errors: list[str], paths: object, context: str) -> None:
    if not isinstance(paths, list) or not paths:
        errors.append(f"{context} must include required_paths")
        return
    for path in paths:
        if not isinstance(path, str) or not path:
            errors.append(f"{context} has invalid required path: {path!r}")
            continue
        if path.startswith(".github/workflows/") or "/.github/workflows/" in path:
            errors.append(f"{context} must not require GitHub Actions workflow path: {path}")
        if path.startswith("/") or ".." in Path(path).parts:
            errors.append(f"{context} path must be repo-relative: {path}")
            continue
        if not (ROOT / path).exists():
            errors.append(f"{context} required path missing: {path}")


def _validate_commands(errors: list[str], commands: object, context: str) -> None:
    if not isinstance(commands, list) or not commands:
        errors.append(f"{context} must include local_commands")
        return
    for command in commands:
        if not isinstance(command, str) or not command:
            errors.append(f"{context} has invalid local command: {command!r}")
            continue
        lowered = command.lower()
        if "github actions" in lowered or ".github/workflows" in lowered or "gh workflow" in lowered:
            errors.append(f"{context} command must stay local, found workflow-dependent command: {command}")


def main() -> int:
    errors: list[str] = []
    payload = _load_manifest(errors)
    if payload is None:
        return _fail(errors)

    if payload.get("schema_version") != "cs.connectorhub.review_split_manifest.v1":
        errors.append("split manifest schema_version mismatch")
    if payload.get("status") != "draft_review_split":
        errors.append("split manifest status must be draft_review_split")
    if payload.get("verdict") != "needs_follow_up":
        errors.append("split manifest verdict must remain needs_follow_up")
    if payload.get("generated_for_pr") != 20:
        errors.append("split manifest generated_for_pr must be 20")
    if payload.get("claim_boundary") != "split_manifest_is_review_planning_not_merge_or_implementation_completion":
        errors.append("split manifest claim_boundary mismatch")
    if payload.get("local_verification_command") != "python3 scripts/verify_connectorhub_review_split.py":
        errors.append("split manifest local_verification_command mismatch")

    notes = payload.get("notes")
    if not isinstance(notes, list) or not all(isinstance(note, str) for note in notes):
        errors.append("split manifest notes must be a list of strings")
    else:
        for token in [
            "It does not claim the branch has already been split.",
            "It does not claim connector.py or test_connectorhub_cli.py refactoring is complete.",
            "It does not require or add GitHub Actions workflows.",
        ]:
            if token not in notes:
                errors.append(f"split manifest notes missing boundary: {token}")

    slices = payload.get("slices")
    if not isinstance(slices, list):
        errors.append("split manifest slices must be a list")
        slices = []
    observed_slice_ids = [item.get("slice_id") for item in slices if isinstance(item, dict)]
    if observed_slice_ids != EXPECTED_SLICE_IDS:
        errors.append(f"split manifest slice order mismatch: {observed_slice_ids}")

    runtime_coverage: set[str] = set()
    for item in slices:
        if not isinstance(item, dict):
            errors.append(f"split manifest slice must be object: {item!r}")
            continue
        slice_id = item.get("slice_id")
        context = f"slice {slice_id}"
        if not isinstance(slice_id, str) or slice_id not in EXPECTED_SLICE_IDS:
            errors.append(f"unexpected slice_id: {slice_id!r}")
            continue
        if not isinstance(item.get("title"), str) or not item.get("title"):
            errors.append(f"{context} missing title")
        boundary = item.get("claim_boundary")
        if not isinstance(boundary, str) or not boundary:
            errors.append(f"{context} missing claim_boundary")
        elif "production_readiness" in boundary and slice_id != "PR20-H":
            errors.append(f"{context} boundary must not imply production readiness")
        _validate_paths(errors, item.get("required_paths"), context)
        _validate_commands(errors, item.get("local_commands"), context)

        scenario_ids = item.get("scenario_ids")
        if not isinstance(scenario_ids, list) or not all(isinstance(value, str) for value in scenario_ids):
            errors.append(f"{context} scenario_ids must be a list of strings")
            continue
        scenario_set = set(scenario_ids)
        unknown = scenario_set - EXPECTED_AI_SCENARIOS - EXPECTED_HUMAN_SCENARIOS
        if unknown:
            errors.append(f"{context} has unknown scenario IDs: {sorted(unknown)}")
        expected_coverage = EXPECTED_RUNTIME_SLICE_COVERAGE.get(slice_id)
        if expected_coverage is not None:
            if scenario_set != expected_coverage:
                errors.append(
                    f"{context} scenario coverage expected {sorted(expected_coverage)}, "
                    f"found {sorted(scenario_set)}"
                )
            runtime_coverage.update(scenario_set)
        if slice_id == "PR20-A" and scenario_set != EXPECTED_AI_SCENARIOS | EXPECTED_HUMAN_SCENARIOS:
            errors.append("PR20-A contract slice must cover all ConnectorHub scenarios")
        if slice_id == "PR20-B" and scenario_set != EXPECTED_AI_SCENARIOS:
            errors.append("PR20-B compact evidence slice must cover all AI-owned ConnectorHub scenarios")
        if slice_id == "PR20-H" and scenario_set:
            errors.append("PR20-H VS2 rehearsal slice must not claim ConnectorHub scenario IDs")

    if runtime_coverage != EXPECTED_AI_SCENARIOS:
        errors.append(
            "runtime slices PR20-C..PR20-G must cover CS-CH-001..CS-CH-040 exactly, "
            f"found {sorted(runtime_coverage)}"
        )

    target_modules = payload.get("target_module_map")
    if not isinstance(target_modules, list) or not REQUIRED_TARGET_MODULES.issubset(set(target_modules)):
        errors.append("target_module_map missing required ConnectorHub modules")
    target_tests = payload.get("target_test_map")
    if not isinstance(target_tests, list) or not REQUIRED_TARGET_TESTS.issubset(set(target_tests)):
        errors.append("target_test_map missing required ConnectorHub test split files")

    if errors:
        return _fail(errors)

    print(
        "PASS: ConnectorHub review split manifest verified "
        f"({len(EXPECTED_SLICE_IDS)} slices, {len(runtime_coverage)} runtime scenarios, "
        f"{len(EXPECTED_HUMAN_SCENARIOS)} human gates kept in contract slice)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
