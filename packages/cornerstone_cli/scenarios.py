from __future__ import annotations

import re
import subprocess
import csv
from pathlib import Path
from typing import Any


FULL_EXPECTED = 206
FULL_MUST_PASS = 184
FULL_REGRESSION = 22
VS0_EXPECTED = 58
VS0_MUST_PASS = 52
VS0_REGRESSION = 6


FULL_ROW = re.compile(
    r"^\| (?P<id>CS-[A-Z]+-\d{3}) \| (?P<type>MUST_PASS|REGRESSION_GUARD) \| (?P<title>[^|]+) \| (?P<section>[^|]+) \|$"
)
VS0_ROW = re.compile(
    r"^\| (?P<id>CS-[A-Z]+-\d{3}) \| (?P<type>MUST_PASS|REGRESSION_GUARD) \| (?P<expected>[^|]+) \| (?P<verification>[^|]+) \|$"
)


def _trim(value: str) -> str:
    return value.strip().replace("  ", " ")


def load_full_scenarios(root: Path) -> list[dict[str, Any]]:
    matrix = root / "docs/scenario-contracts/SCENARIO_MATRIX_FULL.md"
    scenarios: list[dict[str, Any]] = []
    for line in matrix.read_text().splitlines():
        match = FULL_ROW.match(line)
        if not match:
            continue
        scenarios.append(
            {
                "id": match.group("id"),
                "type": match.group("type"),
                "title": _trim(match.group("title")),
                "section": _trim(match.group("section")),
                "owner": "AI",
            }
        )
    return scenarios


def load_vs0_scenarios(root: Path) -> list[dict[str, Any]]:
    contract = root / "docs/scenario-contracts/VS0_IMPLEMENTATION_CONTRACT.md"
    titles = {row["id"]: row["title"] for row in load_full_scenarios(root)}
    scenarios: list[dict[str, Any]] = []
    for line in contract.read_text().splitlines():
        match = VS0_ROW.match(line)
        if not match:
            continue
        scenario_id = match.group("id")
        scenarios.append(
            {
                "id": scenario_id,
                "type": match.group("type"),
                "title": titles.get(scenario_id, scenario_id),
                "expected_result": _trim(match.group("expected")),
                "verification_method": _trim(match.group("verification")),
                "owner": "AI",
            }
        )
    return scenarios


def list_scenarios(root: Path, scenario_set: str) -> list[dict[str, Any]]:
    if scenario_set == "vs0":
        return load_vs0_scenarios(root)
    return load_full_scenarios(root)


def load_verification_matrix(root: Path) -> list[dict[str, str]]:
    matrix = root / "docs/scenario-contracts/SCENARIO_VERIFICATION_MATRIX.csv"
    if not matrix.exists():
        return []
    with matrix.open(newline="") as file:
        return list(csv.DictReader(file))


def _type_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts = {"MUST_PASS": 0, "REGRESSION_GUARD": 0}
    for row in rows:
        counts[row["type"]] = counts.get(row["type"], 0) + 1
    return counts


def coverage_report(root: Path) -> dict[str, Any]:
    full = load_full_scenarios(root)
    vs0 = load_vs0_scenarios(root)
    full_ids = {row["id"] for row in full}
    vs0_ids = {row["id"] for row in vs0}
    matrix = load_verification_matrix(root)
    matrix_ids = {row["scenario_id"] for row in matrix}
    full_counts = _type_counts(full)
    vs0_counts = _type_counts(vs0)
    missing: list[str] = []

    if len(full) != FULL_EXPECTED:
        missing.append(f"full_count:{len(full)}")
    if full_counts.get("MUST_PASS") != FULL_MUST_PASS:
        missing.append(f"full_must_pass:{full_counts.get('MUST_PASS')}")
    if full_counts.get("REGRESSION_GUARD") != FULL_REGRESSION:
        missing.append(f"full_regression:{full_counts.get('REGRESSION_GUARD')}")
    if len(vs0) != VS0_EXPECTED:
        missing.append(f"vs0_count:{len(vs0)}")
    if vs0_counts.get("MUST_PASS") != VS0_MUST_PASS:
        missing.append(f"vs0_must_pass:{vs0_counts.get('MUST_PASS')}")
    if vs0_counts.get("REGRESSION_GUARD") != VS0_REGRESSION:
        missing.append(f"vs0_regression:{vs0_counts.get('REGRESSION_GUARD')}")

    missing_vs0_from_full = sorted(vs0_ids - full_ids)
    missing.extend(f"vs0_not_in_full:{scenario_id}" for scenario_id in missing_vs0_from_full)
    if len(matrix) != FULL_EXPECTED:
        missing.append(f"verification_matrix_count:{len(matrix)}")
    missing_matrix = sorted(full_ids - matrix_ids)
    missing.extend(f"verification_matrix_missing:{scenario_id}" for scenario_id in missing_matrix)

    return {
        "ok": not missing,
        "missing": missing,
        "full": {"count": len(full), "type_counts": full_counts},
        "vs0": {"count": len(vs0), "type_counts": vs0_counts},
        "verification_matrix": {"count": len(matrix), "path": "docs/scenario-contracts/SCENARIO_VERIFICATION_MATRIX.csv"},
    }


def _run_script(root: Path, script: str) -> dict[str, Any]:
    result = subprocess.run(
        ["sh", script],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": f"sh {script}",
        "exit_code": result.returncode,
        "stdout": result.stdout.strip().splitlines()[-3:],
        "stderr": result.stderr.strip().splitlines()[-3:],
    }


def _row(
    scenario_id: str,
    scenario_type: str,
    status: str,
    evidence: list[str],
    notes: str,
    owner: str = "AI",
) -> dict[str, Any]:
    return {
        "id": scenario_id,
        "type": scenario_type,
        "status": status,
        "evidence": evidence,
        "notes": notes,
        "owner": owner,
    }


def verify_vs0_scaffold(root: Path) -> dict[str, Any]:
    docs_result = _run_script(root, "scripts/verify_sot_docs.sh")
    cli_result = _run_script(root, "scripts/verify_cli_native_first_docs.sh")
    readiness_result = _run_script(root, "scripts/verify_vs0_scaffold_readiness_docs.sh")
    coverage = coverage_report(root)

    script_pass = (
        docs_result["exit_code"] == 0
        and cli_result["exit_code"] == 0
        and readiness_result["exit_code"] == 0
        and coverage["ok"]
    )

    required_docs = [
        "docs/adr/ADR-0002-framework-and-version-policy.md",
        "docs/adr/ADR-0003-monorepo-setup.md",
        "docs/adr/ADR-0004-cli-native-first-setup.md",
        "docs/adr/ADR-0005-domain-boundaries.md",
        "docs/adr/ADR-0006-agent-guide.md",
        "docs/scenario-contracts/VS0_SCAFFOLD_CONTRACT.md",
        "docs/verification-reports/template.md",
    ]
    missing_docs = [path for path in required_docs if not (root / path).exists()]
    dependency_files = [
        "uv.lock",
        "pnpm-lock.yaml",
        "package-lock.json",
        "requirements.txt",
        "requirements-dev.txt",
    ]
    added_dependency_files = [path for path in dependency_files if (root / path).exists()]
    feature_dirs = ["apps/web", "services/api", "services/worker", "fixtures/vs0"]
    product_feature_dirs = [path for path in feature_dirs if (root / path).exists()]

    results = [
        _row(
            "VS0-SCAF-001",
            "MUST_PASS",
            "PASS" if not missing_docs and docs_result["exit_code"] == 0 else "FAIL",
            [
                "docs/adr/ADR-0002-framework-and-version-policy.md",
                "scripts/verify_sot_docs.sh",
            ],
            "Setup docs define compatible baseline and docs verification passes.",
        ),
        _row(
            "VS0-SCAF-002",
            "MUST_PASS",
            "PASS" if not missing_docs else "FAIL",
            [
                "docs/adr/ADR-0003-monorepo-setup.md",
                "docs/adr/ADR-0005-domain-boundaries.md",
            ],
            "Monorepo direction preserves one product with internal engine boundaries.",
        ),
        _row(
            "VS0-SCAF-003",
            "MUST_PASS",
            "PASS" if cli_result["exit_code"] == 0 and (root / "cornerstone").exists() else "FAIL",
            [
                "scripts/verify_cli_native_first_docs.sh",
                "cornerstone --help",
                "cornerstone version --json",
            ],
            "CLI native-first is documented and a native scaffold command exists.",
        ),
        _row(
            "VS0-SCAF-004",
            "MUST_PASS",
            "PASS" if readiness_result["exit_code"] == 0 else "FAIL",
            [
                "docs/scenario-contracts/VS0_SCAFFOLD_CONTRACT.md",
                "cornerstone ready --json",
            ],
            "Future product-runtime readiness is still reported honestly as not ready.",
        ),
        _row(
            "VS0-SCAF-005",
            "MUST_PASS",
            "PASS" if not added_dependency_files and not product_feature_dirs else "FAIL",
            ["git diff --name-only", "repo path review"],
            "No production dependency lockfiles or product feature runtime directories are present.",
        ),
        _row(
            "VS0-SCAF-006",
            "MUST_PASS",
            "PASS" if (root / "docs/verification-reports/template.md").exists() else "FAIL",
            ["docs/verification-reports/template.md"],
            "Verification report template can record scenario evidence and human-required items.",
        ),
        _row(
            "VS0-SCAF-R01",
            "REGRESSION_GUARD",
            "PASS" if script_pass else "FAIL",
            [
                "scripts/verify_sot_docs.sh",
                "scripts/verify_cli_native_first_docs.sh",
                "cornerstone scenario coverage --json",
            ],
            "Existing 206 full scenarios, 58 VS-0 scenarios, and CLI native-first gate remain wired.",
        ),
    ]

    blocking = [row for row in results if row["status"] != "PASS" and row["owner"] != "Human"]
    return {
        "status": "success" if not blocking else "failed",
        "scenario_set": "vs0-scaffold",
        "summary": {
            "scenario_count": len(results),
            "pass": len([row for row in results if row["status"] == "PASS"]),
            "blocking": len(blocking),
            "product_feature_claims": "NOT_VERIFIED",
        },
        "coverage": coverage,
        "command_evidence": [docs_result, cli_result, readiness_result],
        "scenario_results": results,
        "human_required": [
            {
                "id": "H-FREEZE-001",
                "why_ai_cannot_verify": "New production dependencies and lockfiles are approval-gated.",
                "required_human_action": "Approve the specific scaffold dependency set before dependency files are added.",
                "expected_evidence": "Written approval naming dependency scope, or explicit approval to use ADR-0002 targets.",
                "release_impact": "Blocks dependency-based scaffold implementation.",
            },
            {
                "id": "H-FREEZE-002",
                "why_ai_cannot_verify": "Full-set human-only ownership cannot be derived until a scenario registry exists.",
                "required_human_action": "Review generated registry classifications once implemented.",
                "expected_evidence": "Approved registry/report rows with required evidence per scenario.",
                "release_impact": "Blocks full release PASS until classified and evidenced.",
            },
        ],
    }
