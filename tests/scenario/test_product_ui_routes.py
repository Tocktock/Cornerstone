from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))

from cornerstone_cli.product_runtime import DEFAULT_SCOPE, make_server

CAPTURE_SCRIPT = ROOT / "scripts" / "capture_vs4_h01_ui_recovery_screenshots.py"
_capture_spec = importlib.util.spec_from_file_location("capture_vs4_h01_ui_recovery_screenshots", CAPTURE_SCRIPT)
assert _capture_spec is not None and _capture_spec.loader is not None
capture_vs4 = importlib.util.module_from_spec(_capture_spec)
_capture_spec.loader.exec_module(capture_vs4)


PRODUCT_ROUTES = ["/", "/search", "/artifacts", "/briefs", "/claims", "/actions", "/inbox", "/audit"]

FORBIDDEN_PRODUCT_PATTERNS = [
    r"local_scenario_ready=",
    r"vs0_runtime_ready=",
    r"production_release_ready=",
    r"real_external_http_calls=",
    r"\bVS[0-9]",
    r"scenario",
    r"verifier",
    r"human gate",
    r"acceptance",
    r"walkthrough",
    r"package path",
    r"readiness",
    r"browser proof",
    r"review packet",
]


def _request(
    base_url: str,
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, str, str]:
    data = None
    request_headers = dict(headers or {})
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        request_headers.setdefault("content-type", "application/json")
    request = urllib.request.Request(base_url + path, data=data, headers=request_headers, method=method)
    with urllib.request.urlopen(request, timeout=10) as response:
        body = response.read().decode("utf-8")
        return response.status, response.headers.get("content-type", ""), body


class ProductUiRoutesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_root = Path(tempfile.mkdtemp(prefix="cornerstone-product-ui-"))
        self.state_dir = self.temp_root / "state"
        self.server = make_server(ROOT, self.state_dir)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        shutil.rmtree(self.temp_root)

    def test_product_home_moves_internal_markers_to_review(self) -> None:
        status, content_type, home = _request(self.base_url, "/", headers={"accept": "text/html"})

        self.assertEqual(status, 200)
        self.assertIn("text/html", content_type)
        self.assertIn('data-product-surface="home"', home)
        self.assertIn("Drop anything, or ask what we know", home)
        self.assertIn("Save a source", home)
        self.assertIn("Ask the workspace", home)
        self.assertIn("Recent items", home)
        self.assertIn("Knowledge states", home)
        self.assertIn("Suggested next steps", home)
        self.assertIn("No brief has been drafted from saved sources yet.", home)
        self.assertIn("Saved sources will appear here.", home)
        self.assert_product_surface_is_clean(home)

        review_status, review_content_type, review = _request(self.base_url, "/review", headers={"accept": "text/html"})
        self.assertEqual(review_status, 200)
        self.assertIn("text/html", review_content_type)
        self.assertIn('data-product-surface="owner-review"', review)
        self.assertIn("Connector governance", review)
        self.assertIn("Connector sources", review)
        self.assertIn("Namespace settings", review)
        self.assertIn("Admin containment", review)
        self.assertIn("Recent connector activity", review)
        self.assertIn("local_scenario_ready=", review)

        legacy_status, legacy_content_type, legacy = _request(self.base_url, "/?scenario=vs0-evux", headers={"accept": "text/html"})
        self.assertEqual(legacy_status, 200)
        self.assertIn("text/html", legacy_content_type)
        self.assertIn('id="run-evux"', legacy)

    def test_product_routes_share_shell_and_hide_internal_language(self) -> None:
        for route in PRODUCT_ROUTES:
            with self.subTest(route=route):
                status, content_type, html = _request(self.base_url, route, headers={"accept": "text/html"})
                self.assertEqual(status, 200)
                self.assertIn("text/html", content_type)
                self.assertIn('data-product-shell="cornerstone"', html)
                self.assertIn("CornerStone", html)
                for label in ["Home", "Search", "Artifacts", "Claims", "Actions", "Inbox", "Audit", "Owner"]:
                    self.assertIn(f">{label}<", html)
                self.assertIn("--cs-color-background-app:", html)
                self.assertIn("--cs-state-saved-bg:", html)
                self.assertIn("--cs-radius-pill:", html)
                self.assert_product_surface_is_clean(html)

    def test_screenshot_matrix_covers_primary_mobile_routes(self) -> None:
        ids = {
            "artifact_id": "art_mobile_matrix",
            "brief_id": "brief_mobile_matrix",
            "claim_id": "claim_mobile_matrix",
            "action_id": "action_mobile_matrix",
        }
        specs = capture_vs4.route_specs(ids)
        mobile_routes = {spec["route"] for spec in specs if spec.get("mobile")}

        for route in [
            "/",
            "/search?q=vendor%20renewal",
            "/artifacts",
            "/briefs",
            "/claims",
            "/actions",
            "/inbox",
            "/audit",
            "/review",
            "/briefs/brief_mobile_matrix?view=html",
            "/actions/action_mobile_matrix?view=html",
        ]:
            with self.subTest(route=route):
                self.assertIn(route, mobile_routes)

    def test_record_detail_routes_preserve_json_default_and_offer_html(self) -> None:
        artifact_id, _, _ = self.create_source_stack()

        json_status, json_content_type, shown_raw = _request(self.base_url, f"/artifacts/{artifact_id}")
        self.assertEqual(json_status, 200)
        self.assertIn("application/json", json_content_type)
        shown = json.loads(shown_raw)
        self.assertEqual(shown["artifact"]["artifact_id"], artifact_id)

        html_status, html_content_type, html = _request(
            self.base_url,
            f"/artifacts/{artifact_id}?view=html",
            headers={"accept": "text/html"},
        )
        self.assertEqual(html_status, 200)
        self.assertIn("text/html", html_content_type)
        self.assertIn('data-product-surface="artifact-detail"', html)
        self.assertIn("Original source", html)
        self.assertIn("Original artifact preview", html)
        self.assertIn("Source text", html)
        self.assertIn("Source metadata", html)
        self.assertIn("Summary", html)
        self.assertIn("Extracted keywords", html)
        self.assertIn("Linked work", html)
        self.assertIn("View linked evidence", html)
        self.assertIn("Provenance", html)
        self.assert_product_surface_is_clean(html)

        search_status, _, search_html = _request(self.base_url, "/search?q=renewal", headers={"accept": "text/html"})
        self.assertEqual(search_status, 200)
        self.assertIn("Vendor renewal", search_html)
        self.assertIn("What we found", search_html)
        self.assertIn("Suggested follow-ups", search_html)
        self.assertIn("Receipt coverage", search_html)
        self.assertIn("Sort: keyword match", search_html)
        self.assert_product_surface_is_clean(search_html)

        audit_html = self.fetch_product_html("/audit")
        self.assertIn('data-product-surface="audit"', audit_html)
        self.assertIn("Activity trail", audit_html)
        self.assertIn("Event stream", audit_html)
        self.assertIn("Raw event detail", audit_html)
        self.assertIn("Audit posture", audit_html)
        self.assertIn("Source saved", audit_html)
        self.assertIn("Evidence bundle prepared", audit_html)
        self.assertIn("Hash chained", audit_html)
        self.assert_product_surface_is_clean(audit_html)

    def test_brief_claim_and_action_detail_pages_use_plain_disclosure(self) -> None:
        _, bundle_id, _ = self.create_source_stack()

        brief_status, _, brief_raw = _request(
            self.base_url,
            "/briefs",
            method="POST",
            payload={**DEFAULT_SCOPE, "evidence_bundle_id": bundle_id},
        )
        self.assertEqual(brief_status, 200)
        brief_id = json.loads(brief_raw)["brief"]["brief_id"]

        claim_status, _, claim_raw = _request(
            self.base_url,
            "/claims",
            method="POST",
            payload={
                **DEFAULT_SCOPE,
                "evidence_bundle_id": bundle_id,
                "statement": "Vendor renewal needs a decision before the August renewal date.",
            },
        )
        self.assertEqual(claim_status, 200)
        claim_id = json.loads(claim_raw)["claim"]["claim_id"]

        action_status, _, action_raw = _request(
            self.base_url,
            "/actions",
            method="POST",
            payload={
                **DEFAULT_SCOPE,
                "claim_id": claim_id,
                "goal": "Draft a vendor renewal follow-up",
                "action_kind": "external_writeback",
                "risk": "medium",
                "target": "planner task for vendor renewal follow-up",
            },
        )
        self.assertEqual(action_status, 200)
        action_id = json.loads(action_raw)["action_card"]["action_id"]

        briefs_html = self.fetch_product_html("/briefs")
        self.assertIn('data-product-surface="briefs"', briefs_html)
        self.assertIn("Brief workspace", briefs_html)
        self.assertIn("Brief queue", briefs_html)
        self.assertIn("Source coverage", briefs_html)
        self.assertIn("Use next", briefs_html)
        self.assertIn("Brief posture", briefs_html)
        self.assertIn("Keyword summary", briefs_html)
        self.assertIn("Open brief", briefs_html)
        self.assert_product_surface_is_clean(briefs_html)

        brief_html = self.fetch_product_html(f"/briefs/{brief_id}?view=html")
        self.assertIn('data-product-surface="brief-detail"', brief_html)
        self.assertIn("What we found", brief_html)
        self.assertIn("Brief status", brief_html)
        self.assertIn("Source coverage", brief_html)
        self.assertIn("Findings with citations", brief_html)
        self.assertIn("Limits and gaps", brief_html)
        self.assertIn("Use this brief", brief_html)
        self.assertIn("Citation trail", brief_html)
        self.assertIn("Source 1", brief_html)
        self.assertIn("Provenance", brief_html)
        self.assertIn("Keyword summary", brief_html)
        self.assertNotIn("extractive_fallback", brief_html)
        self.assert_product_surface_is_clean(brief_html)

        claim_html = self.fetch_product_html(f"/claims/{claim_id}?view=html")
        self.assertIn('data-product-surface="claim-detail"', claim_html)
        self.assertIn("Trust ladder", claim_html)
        self.assertIn("Evidence-backed", claim_html)
        self.assertIn("Supporting evidence", claim_html)
        self.assertIn("Review controls", claim_html)
        self.assertIn("Claim statement", claim_html)
        self.assertIn("Decision gate", claim_html)
        self.assertIn("Source support", claim_html)
        self.assertIn("Promote to decision locked", claim_html)
        self.assertIn("Review required before approval", claim_html)
        self.assert_product_surface_is_clean(claim_html)

        action_html = self.fetch_product_html(f"/actions/{action_id}?view=html")
        self.assertIn('data-product-surface="action-detail"', action_html)
        self.assertIn("Action preview", action_html)
        self.assertIn("Action review status", action_html)
        self.assertIn("Impacted objects", action_html)
        self.assertIn("Proposed changes", action_html)
        self.assertIn("External calls", action_html)
        self.assertIn("Policy decision", action_html)
        self.assertIn("Risk and approval", action_html)
        self.assertIn("Required reason", action_html)
        self.assertIn("Approval history", action_html)
        self.assertIn("Simulated in local mode", action_html)
        self.assertIn("Approval required", action_html)
        self.assertNotIn("external_writeback", action_html)
        self.assert_product_surface_is_clean(action_html)

        artifacts_html = self.fetch_product_html("/artifacts")
        self.assertIn("Collection summary", artifacts_html)
        self.assertIn("Source register", artifacts_html)
        self.assertIn("Source posture", artifacts_html)
        self.assert_product_surface_is_clean(artifacts_html)

        claims_html = self.fetch_product_html("/claims")
        self.assertIn("Claim review queue", claims_html)
        self.assertIn("Review posture", claims_html)
        self.assertIn("Trust ladder", claims_html)
        self.assert_product_surface_is_clean(claims_html)

        actions_html = self.fetch_product_html("/actions")
        self.assertIn("Action preview queue", actions_html)
        self.assertIn("Dry-run posture", actions_html)
        self.assertIn("Action safeguards", actions_html)
        self.assert_product_surface_is_clean(actions_html)

        inbox_html = self.fetch_product_html("/inbox")
        self.assertIn('data-product-surface="inbox"', inbox_html)
        self.assertIn("Needs review", inbox_html)
        self.assertIn("Approval requests", inbox_html)
        self.assertIn("Selected item", inbox_html)
        self.assertIn("Next actions", inbox_html)
        self.assertIn("Review sources", inbox_html)
        self.assert_product_surface_is_clean(inbox_html)

    def create_source_stack(self) -> tuple[str, str, str]:
        body = dict(DEFAULT_SCOPE)
        body.update({"text": "Vendor renewal: auto-renewal is August 1 and finance says the annual price increased."})
        status, content_type, created_raw = _request(self.base_url, "/artifacts", method="POST", payload=body)
        self.assertEqual(status, 200)
        self.assertIn("application/json", content_type)
        created = json.loads(created_raw)
        artifact_id = created["artifact"]["artifact_id"]

        search_status, _, search_raw = _request(
            self.base_url,
            "/search",
            method="POST",
            payload={**DEFAULT_SCOPE, "query": "vendor renewal"},
        )
        self.assertEqual(search_status, 200)
        snapshot_id = json.loads(search_raw)["search_snapshot"]["search_snapshot_id"]

        bundle_status, _, bundle_raw = _request(
            self.base_url,
            "/evidence-bundles",
            method="POST",
            payload={**DEFAULT_SCOPE, "search_snapshot_id": snapshot_id},
        )
        self.assertEqual(bundle_status, 200)
        bundle_id = json.loads(bundle_raw)["evidence_bundle"]["evidence_bundle_id"]
        return artifact_id, bundle_id, snapshot_id

    def fetch_product_html(self, path: str) -> str:
        status, content_type, html = _request(self.base_url, path, headers={"accept": "text/html"})
        self.assertEqual(status, 200)
        self.assertIn("text/html", content_type)
        return html

    def assert_product_surface_is_clean(self, html: str) -> None:
        for pattern in FORBIDDEN_PRODUCT_PATTERNS:
            self.assertIsNone(re.search(pattern, html, re.IGNORECASE), pattern)


if __name__ == "__main__":
    unittest.main()
