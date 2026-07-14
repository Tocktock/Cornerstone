#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))

from cornerstone_cli.scenarios import load_full_scenarios, load_vs0_scenarios


OUTPUT = ROOT / "docs/scenario-contracts/SCENARIO_VERIFICATION_MATRIX.csv"
FIELDS = [
    "scenario_id",
    "type",
    "local_required",
    "verification_class",
    "verification_owner",
    "verification_command",
    "evidence_artifact",
    "human_required_reason",
    "required_human_action",
    "expected_human_evidence",
    "release_impact",
    "status",
]
PRESERVED_FIELDS = [
    "verification_command",
    "evidence_artifact",
    "human_required_reason",
    "required_human_action",
    "expected_human_evidence",
    "release_impact",
    "status",
]

CLASS_BY_PREFIX = {
    "CS-PROD": "U+D",
    "CS-ARCH": "D",
    "CS-UND": "D+M",
    "CS-CLAIM": "M+D+P",
    "CS-MEM": "D+P+U",
    "CS-NS": "D+P",
    "CS-AUTO": "D+P+U",
    "CS-AGENT": "D+P",
    "CS-BRAIN": "M+D+P",
    "CS-LEARN": "M+D",
    "CS-EXT": "D+P+S",
    "CS-SEC": "S+P+D",
    "CS-REG": "D+P+S",
    "CS-VAL": "M+D+H",
}

SECURITY_CLASS_OVERRIDES = {
    "CS-ARCH-006": "S+P+D",
    "CS-ARCH-007": "S+P+D",
    "CS-SEC-002": "S+P+D",
    "CS-SEC-003": "S+P+D",
    "CS-SEC-007": "S+P+D",
    "CS-SEC-008": "S+P+D",
    "CS-SEC-010": "P+D",
    "CS-SEC-011": "D",
    "CS-SEC-015": "S+P+D",
    "CS-SEC-016": "S+P+D",
    "CS-REG-006": "S+P+D",
    "CS-REG-007": "S+P+D",
    "CS-REG-013": "S+P+D",
    "CS-REG-014": "S+P+D",
    "CS-REG-018": "S+P+D",
}


def class_for(scenario_id: str) -> str:
    if scenario_id in SECURITY_CLASS_OVERRIDES:
        return SECURITY_CLASS_OVERRIDES[scenario_id]
    for prefix, verification_class in CLASS_BY_PREFIX.items():
        if scenario_id.startswith(prefix):
            return verification_class
    return "D"


def command_for(scenario_id: str, vs0_ids: set[str]) -> str:
    if scenario_id.startswith("CS-VAL-"):
        return "cornerstone scenario verify vs5-citation-grounded-brief --model-provider ollama --json"
    if scenario_id in vs0_ids:
        return (
            "cornerstone scenario verify vs0 "
            f"--scenario {scenario_id} --corpus fixtures/vs0 "
            "--model-provider local_test --json"
        )
    return f"cornerstone scenario verify full --scenario {scenario_id} --json"


def owner_for(scenario_id: str) -> str:
    if scenario_id in {"CS-VAL-008", "CS-VAL-009"}:
        return "Human"
    if scenario_id in {"CS-VAL-003", "CS-VAL-004", "CS-VAL-005", "CS-VAL-007"}:
        return "AI+Human"
    return "AI"


def load_existing_rows() -> dict[str, dict[str, str]]:
    if not OUTPUT.exists():
        return {}
    with OUTPUT.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != FIELDS:
            return {}
        return {row["scenario_id"]: row for row in reader}


def build_rows(existing_rows: dict[str, dict[str, str]] | None = None) -> list[dict[str, str]]:
    full = load_full_scenarios(ROOT)
    vs0_ids = {row["id"] for row in load_vs0_scenarios(ROOT)}
    rows: list[dict[str, str]] = []
    for scenario in full:
        scenario_id = scenario["id"]
        row = {
            "scenario_id": scenario_id,
            "type": scenario["type"],
            "local_required": "true",
            "verification_class": class_for(scenario_id),
            "verification_owner": owner_for(scenario_id),
            "verification_command": command_for(scenario_id, vs0_ids),
            "evidence_artifact": (
                "reports/scenario/vs5-citation-grounded-brief-2026-07-12.json"
                if scenario_id.startswith("CS-VAL-")
                else f"reports/scenario/{scenario_id.lower()}.json"
            ),
            "human_required_reason": "",
            "required_human_action": "",
            "expected_human_evidence": "",
            "release_impact": "",
            "status": "NOT_VERIFIED",
        }
        if row["verification_owner"] in {"Human", "AI+Human"}:
            row.update(
                {
                    "human_required_reason": "Subjective product value or external-user behavior cannot be established by automation alone.",
                    "required_human_action": "Complete the dated VS5 rubric or external-session record required by the active contract.",
                    "expected_human_evidence": "A dated reviewer or participant record bound to the frozen corpus and current model stack.",
                    "release_impact": "Blocks VALUE_VERIFIED_LOCAL or VALUE_VERIFIED_EXTERNAL until accepted.",
                    "status": "HUMAN_REQUIRED",
                }
            )
        existing = (existing_rows or {}).get(scenario_id, {})
        for field in PRESERVED_FIELDS:
            if existing.get(field):
                row[field] = existing[field]
        rows.append(row)
    return rows


def render_csv(rows: list[dict[str, str]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Fail if the committed matrix is stale")
    args = parser.parse_args()

    content = render_csv(build_rows(load_existing_rows()))
    if args.check:
        if not OUTPUT.exists():
            print(f"FAIL: missing {OUTPUT.relative_to(ROOT)}", file=sys.stderr)
            return 1
        current = OUTPUT.read_text()
        if current != content:
            print(f"FAIL: stale {OUTPUT.relative_to(ROOT)}; regenerate it", file=sys.stderr)
            return 1
        print("PASS: scenario verification matrix is current.")
        return 0

    OUTPUT.write_text(content)
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
