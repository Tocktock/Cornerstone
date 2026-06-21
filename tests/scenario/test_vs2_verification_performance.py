from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))

from cornerstone_cli import scenarios, vs2_security
from cornerstone_cli.vs2_verification_metadata import build_source_fingerprint, proof_hash


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


class Vs2VerificationPerformanceTests(unittest.TestCase):
    def test_local_proof_rejects_stale_local_range_reuse_without_rerun(self) -> None:
        stale_range_path = ROOT / "tmp/test-vs2-stale-local-range.json"
        proof_path = Path("tmp/test-vs2-proof-reuse-rejected.json")
        fingerprint = build_source_fingerprint(ROOT, family="vs2_local_range")
        stale_fingerprint = dict(fingerprint)
        stale_fingerprint["input_digest"] = "stale"
        write_json(
            stale_range_path,
            {
                "schema_version": "cs.vs2_local_range.v1",
                "status": "passed",
                "source_fingerprint": stale_fingerprint,
            },
        )

        with mock.patch.object(vs2_security, "VS2_PROOF_REPORT", proof_path):
            report = vs2_security.run_vs2_local_security_proof(
                ROOT,
                local_range_report=stale_range_path.relative_to(ROOT),
            )

        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["local_range_reuse"]["status"], "rejected")
        self.assertIn("source_fingerprint_input_digest_mismatch", report["local_range_reuse"]["errors"])
        self.assertEqual(report["negative_evidence"]["stale_or_invalid_local_range_reuse_blocked"], 1)

    def test_scenario_verify_reuses_current_proof_without_fresh_proof_rerun(self) -> None:
        proof_path = ROOT / "tmp/test-vs2-current-local-proof.json"
        proof = {
            "schema_version": "cs.vs2_local_security_proof.v0",
            "status": "failed",
            "scenario_set": "vs2-policy-tenancy-egress",
            "source_fingerprint": build_source_fingerprint(ROOT, family="vs2_local_proof"),
            "summary": {
                "product_feature_claims": "LOCAL_VS2_READINESS_REJECTED_REMEDIATION_REQUIRED",
            },
            "negative_evidence": {
                "ai_rows_marked_pass_without_evidence": 0,
                "ai_rows_marked_pass_without_scenario_validator": 0,
                "blanket_dependencies_ok_pass_used": 0,
            },
            "scenario_results": [
                {
                    "id": "VS2-SEC-001",
                    "scenario_id": "VS2-SEC-001",
                    "status": "PASS",
                    "owner": "AI",
                    "validator": "fixture_validator",
                    "evidence": ["tmp/test-evidence.json"],
                    "evidence_paths": ["tmp/test-evidence.json"],
                }
            ],
        }
        proof["proof_hash"] = proof_hash(proof)
        write_json(proof_path, proof)

        original_glob = Path.glob

        def fast_observation_glob(path: Path, pattern: str):
            if pattern in {"**/*.rego", "**/*.sql", "compose*.yml"}:
                return []
            return original_glob(path, pattern)

        with (
            mock.patch.object(Path, "glob", fast_observation_glob),
            mock.patch.object(scenarios, "run_vs2_local_security_proof", side_effect=AssertionError("fresh proof rerun")),
        ):
            report = scenarios.verify_vs2_policy_tenancy_egress(
                ROOT,
                local_proof_report=proof_path.relative_to(ROOT),
            )

        self.assertEqual(report["local_security_proof"]["reuse"]["status"], "reused")
        self.assertEqual(report["scenario_results"][0]["status"], "PASS")
        self.assertEqual(report["summary"]["blocking"], 85)


if __name__ == "__main__":
    unittest.main()
