#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages"))

from cornerstone_cli.scenarios import load_full_scenarios


MATRIX = ROOT / "docs/scenario-contracts/SCENARIO_VERIFICATION_MATRIX.csv"
REQUIRED_FIELDS = [
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
ALLOWED_STATUS = {"PASS", "FAIL", "NOT_VERIFIED", "NOT_RUN", "HUMAN_REQUIRED", "OUT_OF_SCOPE"}
ALLOWED_OWNER = {"AI", "Human", "AI+Human"}


def fail(message: str) -> int:
    print(f"FAIL: {message}", file=sys.stderr)
    return 1


def main() -> int:
    if not MATRIX.exists():
        return fail(f"missing {MATRIX.relative_to(ROOT)}")

    with MATRIX.open(newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != REQUIRED_FIELDS:
            return fail("scenario verification matrix header mismatch")
        rows = list(reader)

    full = load_full_scenarios(ROOT)
    expected = {row["id"]: row["type"] for row in full}
    seen = {row["scenario_id"]: row for row in rows}

    missing = sorted(set(expected) - set(seen))
    extra = sorted(set(seen) - set(expected))
    if missing:
        return fail(f"matrix missing scenarios: {', '.join(missing[:10])}")
    if extra:
        return fail(f"matrix has unknown scenarios: {', '.join(extra[:10])}")

    for row in rows:
        scenario_id = row["scenario_id"]
        if row["type"] != expected[scenario_id]:
            return fail(f"{scenario_id} type mismatch")
        if row["local_required"] != "true":
            return fail(f"{scenario_id} must set local_required=true")
        if not row["verification_class"]:
            return fail(f"{scenario_id} missing verification_class")
        if row["verification_owner"] not in ALLOWED_OWNER:
            return fail(f"{scenario_id} has invalid owner {row['verification_owner']}")
        if row["verification_owner"] in {"AI", "AI+Human"} and not row["verification_command"]:
            return fail(f"{scenario_id} missing AI verification command")
        if not row["evidence_artifact"]:
            return fail(f"{scenario_id} missing evidence artifact")
        if row["status"] not in ALLOWED_STATUS:
            return fail(f"{scenario_id} invalid status {row['status']}")
        if row["status"] == "PASS" and not row["evidence_artifact"]:
            return fail(f"{scenario_id} PASS lacks evidence artifact")
        if row["status"] == "HUMAN_REQUIRED":
            for field in [
                "human_required_reason",
                "required_human_action",
                "expected_human_evidence",
                "release_impact",
            ]:
                if not row[field]:
                    return fail(f"{scenario_id} HUMAN_REQUIRED missing {field}")

    print(
        "PASS: scenario verification matrix verified "
        f"({len(rows)} scenarios; no missing rows; no unevidenced PASS claims)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
