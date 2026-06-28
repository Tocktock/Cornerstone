#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone


# The generator imports the verifier so the manifest file set has one source of
# truth. Avoid leaving __pycache__ behind because VS2 source fingerprints include
# scripts/**/*.py.
sys.dont_write_bytecode = True

import verify_connectorhub_engineering_trail as verifier  # noqa: E402


def main() -> int:
    rows = verifier._read_matrix()
    delivery_unit_manifest = verifier._build_scenario_delivery_unit_manifest(rows)
    delivery_out = verifier.SCENARIO_DELIVERY_UNIT_MANIFEST
    delivery_out.parent.mkdir(parents=True, exist_ok=True)
    delivery_out.write_text(json.dumps(delivery_unit_manifest, indent=2, sort_keys=True) + "\n")
    paths = verifier._required_manifest_paths(rows)
    root = verifier.ROOT.resolve()

    files = []
    for path in paths:
        resolved = path.resolve()
        relative_path = resolved.relative_to(root).as_posix()
        files.append(
            {
                "path": relative_path,
                "sha256": verifier._sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )

    source_inputs = []
    for source in verifier.SOURCE_INPUTS:
        path = source["path"]
        source_inputs.append(
            {
                "label": source["label"],
                "path": path.resolve().relative_to(root).as_posix(),
                "sha256": verifier._sha256(path),
                "line_count": verifier._line_count(path),
                "size_bytes": path.stat().st_size,
                "role": "source_input_not_implementation_proof",
                "claim_boundary": "source_reconciliation_only",
            }
        )

    manifest = {
        "manifest_schema": "cornerstone.connectorhub.engineering_trail_manifest.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_matrix_rows": len(rows),
            "ai_pass_rows": sum(
                1
                for row in rows
                if row.get("owner") == "AI" and row.get("status") == "PASS"
            ),
            "human_required_rows": sum(
                1
                for row in rows
                if row.get("owner") == "Human"
                and row.get("status") == "HUMAN_REQUIRED"
            ),
            "source_requirements_covered": len(verifier.SOURCE_REQUIREMENTS),
            "file_count": len(files),
            "source_input_count": len(source_inputs),
            "claim_boundary": "local_fixture_and_local_vs2_topology_only",
            "verdict": "needs-follow-up",
        },
        "files": files,
        "source_inputs": source_inputs,
    }

    out = verifier.ENGINEERING_TRAIL_MANIFEST
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        f"wrote {out.relative_to(verifier.ROOT)} "
        f"({len(files)} repo files, {len(source_inputs)} repo source inputs)"
    )
    print(
        f"wrote {delivery_out.relative_to(verifier.ROOT)} "
        f"({delivery_unit_manifest['summary']['delivery_unit_closed_count']} closed AI delivery units)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
