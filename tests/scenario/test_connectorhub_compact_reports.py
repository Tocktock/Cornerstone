from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from compact_connectorhub_reports import compact_reports  # noqa: E402


def sample_report(path: Path, scenario_id: str | None = None) -> dict[str, object]:
    rows = [
        {
            "id": scenario_id or "CS-CH-001",
            "status": "PASS",
            "type": "MUST_PASS",
            "owner": "AI",
            "evidence": ["cornerstone connector contract validate --json"],
            "notes": "sample",
        }
    ]
    return {
        "schema_version": "cs.cli.v0",
        "cli_schema_version": "cs.cli.v0",
        "product": "CornerStone",
        "version": "0.0.0-scaffold",
        "mode": "local_scaffold",
        "command": "cornerstone scenario verify connector-contract-adapter",
        "owner_id": "local-user",
        "tenant_id": "local-dev",
        "workspace_id": "default",
        "namespace_id": "personal",
        "scenario_set": "connector-contract-adapter",
        "status": "success",
        "ids": {"git_commit": "test"},
        "errors": [],
        "human_required": [],
        "summary": {
            "scenario_count": len(rows),
            "pass": len(rows),
            "fail": 0,
            "not_verified": 0,
            "human_required": 0,
            "blocking": 0,
            "product_feature_claims": "LOCAL_FIXTURE_CONNECTOR_CONTRACT_ADAPTER_40_AI_ROWS_HUMAN_GATES_PENDING",
        },
        "scenario_filter": [scenario_id] if scenario_id else None,
        "scenario_results": rows,
        "command_evidence": [
            {
                "schema_version": "cs.cli_transcript.v0",
                "command": ["cornerstone", "connector", "contract", "validate", "--json"],
                "exit_code": 0,
            }
        ],
        "connector_contract_evidence": {"sample_state_dir": f"tmp/{scenario_id or 'aggregate'}"},
        "negative_evidence": {"production_readiness_overclaims": 0},
        "readiness_dimensions": {
            "contract_schema_verified": "LOCAL_FIXTURE_VERIFIED",
            "local_fixture_behavior_verified": "LOCAL_FIXTURE_VERIFIED",
            "local_physical_device_behavior_verified": "HUMAN_REQUIRED",
            "live_provider_read_verified": "NOT_VERIFIED",
            "live_provider_write_verified": "OUT_OF_SCOPE_READ_ONLY",
            "production_tenancy_policy_egress_verified": "NOT_VERIFIED",
            "human_ux_privacy_accepted": "HUMAN_REQUIRED",
            "release_publishing_approved": "NOT_VERIFIED",
        },
        "audit_refs": [],
        "evidence_refs": [],
        "policy_decision_refs": [],
        "self_command_transcript": {
            "schema_version": "cs.command_transcript.v0",
            "command": ["cornerstone", "scenario", "verify", "connector-contract-adapter", "--json", "--output", str(path)],
            "exit_code": 0,
        },
        "output_path": str(path),
    }


class ConnectorHubCompactReportsTest(unittest.TestCase):
    def test_compactor_writes_shared_evidence_and_hash_addressed_envelopes(self) -> None:
        tmp_root = ROOT / "tmp"
        tmp_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=tmp_root) as tmp:
            root = Path(tmp)
            source_dir = root / "source"
            output_dir = root / "compact"
            source_dir.mkdir()
            aggregate_path = source_dir / "connector-contract-adapter-2026-06-23.json"
            focused_path = source_dir / "connector-contract-adapter-cs-ch-001-2026-06-23.json"
            aggregate_path.write_text(json.dumps(sample_report(aggregate_path), sort_keys=True) + "\n")
            focused_path.write_text(json.dumps(sample_report(focused_path, "CS-CH-001"), sort_keys=True) + "\n")

            manifest = compact_reports(
                source_dir=source_dir,
                output_dir=output_dir,
                expected_focused_count=1,
                delete_sources=True,
            )

            self.assertFalse(aggregate_path.exists())
            self.assertFalse(focused_path.exists())
            self.assertEqual(manifest["summary"]["source_full_report_count"], 2)
            self.assertEqual(manifest["summary"]["compact_report_count"], 2)

            shared_path = output_dir / "shared-evidence-2026-06-23.json"
            shared_index_path = output_dir / "shared-evidence-index-2026-06-23.json"
            aggregate_compact_path = output_dir / "aggregate-2026-06-23.json"
            focused_compact_path = output_dir / "scenarios/CS-CH-001.json"
            self.assertFalse(shared_path.exists())
            self.assertTrue(shared_index_path.exists())
            self.assertTrue(aggregate_compact_path.exists())
            self.assertTrue(focused_compact_path.exists())

            focused_compact = json.loads(focused_compact_path.read_text())
            shared_index = json.loads(shared_index_path.read_text())
            self.assertEqual(focused_compact["compact_evidence_layout"], "content_addressed_objects_v1")
            self.assertEqual(focused_compact["shared_evidence_ref"]["path"], str(shared_index_path.relative_to(ROOT)))
            self.assertEqual(shared_index["layout"], "content_addressed_objects_v1")
            self.assertEqual(shared_index["summary"]["section_count"], 7)
            self.assertEqual(shared_index["summary"]["object_count"], 5)
            self.assertEqual(shared_index["summary"]["deduplicated_object_ref_count"], 5)
            self.assertEqual(len(shared_index["sections"]), 7)
            self.assertEqual(focused_compact["scenario_results"][0]["id"], "CS-CH-001")
            self.assertEqual(
                focused_compact["path_portability"]["claim_boundary"],
                "absolute_paths_are_historical_transcript_metadata_not_portable_evidence",
            )
            self.assertEqual(
                focused_compact["path_portability"]["portable_report_path"],
                str(focused_compact_path.relative_to(ROOT)),
            )
            self.assertEqual(
                focused_compact["path_portability"]["historical_absolute_path_fields"],
                ["output_path", "source_report.output_path"],
            )
            self.assertEqual(
                shared_index["path_portability"]["claim_boundary"],
                "absolute_paths_are_historical_transcript_metadata_not_portable_evidence",
            )
            self.assertIn(
                "objects referenced by sections.command_evidence may include historical transcript source paths",
                shared_index["path_portability"]["historical_absolute_path_fields"],
            )
            self.assertEqual(
                focused_compact["source_report"]["omitted_duplicate_section_refs"]["command_evidence"]["count"],
                1,
            )
            command_section_ref = shared_index["sections"]["command_evidence"]
            self.assertNotIn("items", command_section_ref)
            command_object_ref = command_section_ref["object"]
            command_object_path = ROOT / command_object_ref["path"]
            self.assertTrue(command_object_path.exists())
            self.assertEqual(command_object_ref["sha256"], command_section_ref["sha256"])
            self.assertEqual(
                command_object_path.read_text().strip(),
                json.dumps(sample_report(Path("x"))["command_evidence"], sort_keys=True, separators=(",", ":")),
            )


if __name__ == "__main__":
    unittest.main()
