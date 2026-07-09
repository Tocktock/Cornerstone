from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlencode


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "packages"))

from cornerstone_cli import product_ui
from cornerstone_cli.acceptance import VS4_PRODUCT_FORBIDDEN_RE
from cornerstone_cli.product_runtime import DEFAULT_SCOPE, make_server
from cornerstone_cli.runtime import LocalRuntimeStore

CAPTURE_SCRIPT = ROOT / "scripts" / "capture_vs4_h01_ui_recovery_screenshots.py"
_capture_spec = importlib.util.spec_from_file_location("capture_vs4_h01_ui_recovery_screenshots", CAPTURE_SCRIPT)
assert _capture_spec is not None and _capture_spec.loader is not None
capture_vs4 = importlib.util.module_from_spec(_capture_spec)
_capture_spec.loader.exec_module(capture_vs4)


PRODUCT_ROUTES = ["/", "/search", "/artifacts", "/briefs", "/claims", "/actions", "/inbox", "/audit"]

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
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            body = response.read().decode("utf-8")
            return response.status, response.headers.get("content-type", ""), body
    except urllib.error.HTTPError as response:
        try:
            body = response.read().decode("utf-8")
            return response.status, response.headers.get("content-type", ""), body
        finally:
            response.close()


def _request_bytes(base_url: str, path: str, *, headers: dict[str, str] | None = None) -> tuple[int, str, bytes]:
    request = urllib.request.Request(base_url + path, headers=dict(headers or {}), method="GET")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            body = response.read()
            return response.status, response.headers.get("content-type", ""), body
    except urllib.error.HTTPError as response:
        try:
            body = response.read()
            return response.status, response.headers.get("content-type", ""), body
        finally:
            response.close()


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
        self.assertIn("Drag and drop files or paste notes here", home)
        self.assertIn("Paste text source", home)
        self.assertIn("Dropped files are read locally by the browser before saving.", home)
        self.assertIn("Browse files", home)
        self.assertIn("Save source", home)
        self.assertIn("Ask the workspace", home)
        self.assertIn("Daily loop handoff", home)
        self.assertIn("Original source kept", home)
        self.assertIn("Draft from saved sources", home)
        self.assertIn("Receipts before decisions", home)
        self.assertIn("Work leaves a trail", home)
        self.assertIn('id="cs-save-source-button"', home)
        self.assertIn('id="cs-file-button"', home)
        self.assertIn('id="cs-ask-submit-button"', home)
        self.assertIn('id="cs-drop-status" data-state="idle"', home)
        self.assertIn('id="cs-ask-status" data-state="idle"', home)
        self.assertIn("cs-status is-idle", home)
        self.assertIn("Paste text before saving.", home)
        self.assertIn("Enter a question first.", home)
        self.assertIn("Recent items", home)
        self.assertIn("Knowledge states", home)
        self.assertIn("Suggested next steps", home)
        self.assertIn("Recent activity", home)
        self.assertIn("Activity appears after you save, search, draft, or review work.", home)
        self.assertIn("No brief has been drafted from saved sources yet.", home)
        self.assertIn("Saved sources will appear here.", home)
        self.assert_product_surface_is_clean(home)

        review_status, review_content_type, review = _request(self.base_url, "/review", headers={"accept": "text/html"})
        self.assertEqual(review_status, 200)
        self.assertIn("text/html", review_content_type)
        self.assertIn('data-product-surface="owner-review"', review)
        self.assertIn("Connector governance", review)
        self.assertIn("Connector governance console", review)
        self.assertIn("Connected source posture", review)
        self.assertIn("Activity / scope", review)
        self.assertIn("Policy controls", review)
        self.assertIn("Access roles", review)
        self.assertIn("Namespace settings", review)
        self.assertIn("Logical isolation, workspace scoped", review)
        self.assertIn("Admin containment", review)
        self.assertIn("Recent connector activity", review)
        self.assertIn("Open reference gallery", review)
        self.assertIn("local_scenario_ready=", review)

        reference_status, reference_content_type, reference_page = _request(self.base_url, "/review/reference-images", headers={"accept": "text/html"})
        self.assertEqual(reference_status, 200)
        self.assertIn("text/html", reference_content_type)
        self.assertIn('data-product-surface="owner-review"', reference_page)
        self.assertIn("Reference image gallery", reference_page)
        self.assertIn("Vendor object detail", reference_page)
        self.assertIn("Operations inbox", reference_page)
        self.assertIn("Home workspace", reference_page)
        self.assertIn("Action dry-run", reference_page)
        self.assertIn('data-vs4-reference-images-pass-evidence="false"', reference_page)
        self.assertIn('data-vs4-reference-images-acceptance-evidence="false"', reference_page)

        image_status, image_content_type, image_body = _request_bytes(
            self.base_url,
            "/review/reference-images/cornerstone-reference-07-home-upload-ask.png",
            headers={"accept": "image/png"},
        )
        self.assertEqual(image_status, 200)
        self.assertIn("image/png", image_content_type)
        self.assertGreater(len(image_body), 1000)

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
                self.assertIn("Evidence-first workspace", html)
                self.assertIn("Drop, ask, decide, and audit with visible receipts.", html)
                self.assertIn("Global search", html)
                self.assertIn("Search across saved sources, claims, briefs, and action drafts", html)
                self.assertIn("Workspace: default", html)
                self.assertIn("Receipts required", html)
                self.assertIn('aria-label="Workspace posture"', html)
                self.assertIn("Review queue", html)
                self.assertIn("cs-nav-count", html)
                for label in ["Home", "Search", "Artifacts", "Claims", "Actions"]:
                    self.assertIn(f"<span>{label}</span>", html)
                self.assertIn('href="/review" aria-label="Open owner area"', html)
                self.assertIn("--cs-color-background-app:", html)
                self.assertIn("--cs-state-saved-bg:", html)
                self.assertIn("--cs-radius-pill:", html)
                self.assert_product_surface_is_clean(html)

        nav = product_ui._nav("/briefs", {"artifacts": [], "briefs": [], "claims": [], "actions": [], "inbox": [], "audit": []})
        for label in ["Briefs", "Inbox", "Audit", "Owner"]:
            self.assertNotIn(f"<span>{label}</span>", nav)
        self.assertNotIn('aria-current="page"', nav)

    def test_day_zero_product_routes_offer_composed_empty_states(self) -> None:
        expected = {
            "/search": ["Search", "Search starts with saved work", "Save a source", "Open artifacts", "Startup path", "First receipts"],
            "/artifacts": ["Day zero", "Start with a source", "Go to Home", "Search workspace", "Startup path", "First receipts"],
            "/briefs": ["Day zero", "Create the first brief", "Save a source", "Open artifacts", "Startup path", "First receipts"],
            "/claims": ["Day zero", "No claims need review", "Open briefs", "Check sources", "Startup path", "First receipts"],
            "/actions": ["Day zero", "No action previews yet", "Open claims", "Open briefs", "Startup path", "First receipts"],
            "/inbox": ["Day zero", "No work waiting", "No selected work", "Start from Home", "Startup path", "First receipts"],
            "/audit": ["Audit ready", "No activity recorded yet", "Start from Home", "Open artifacts", "Startup path", "First receipts"],
        }

        for route, phrases in expected.items():
            with self.subTest(route=route):
                html = self.fetch_product_html(route)
                self.assertIn("cs-empty-state", html)
                self.assertIn("cs-empty-actions", html)
                for phrase in phrases:
                    self.assertIn(phrase, html)
                self.assert_product_surface_is_clean(html)

    def test_product_not_found_routes_keep_html_and_json_boundaries(self) -> None:
        html_status, html_content_type, html = _request(self.base_url, "/missing-product-route", headers={"accept": "text/html"})
        self.assertEqual(html_status, 404)
        self.assertIn("text/html", html_content_type)
        self.assertIn('data-product-surface="not-found"', html)
        self.assertIn("We could not find that page", html)
        self.assertIn("Search workspace", html)
        self.assertIn("Useful places", html)
        self.assertIn("Saved sources", html)
        self.assert_product_surface_is_clean(html)

        detail_status, detail_content_type, detail_html = _request(
            self.base_url,
            "/artifacts/missing-source?view=html",
            headers={"accept": "text/html"},
        )
        self.assertEqual(detail_status, 404)
        self.assertIn("text/html", detail_content_type)
        self.assertIn('data-product-surface="not-found"', detail_html)
        self.assertIn("This source is not available", detail_html)
        self.assertIn("Brief workspace", detail_html)
        self.assert_product_surface_is_clean(detail_html)

        json_status, json_content_type, json_body = _request(self.base_url, "/missing-product-route")
        self.assertEqual(json_status, 404)
        self.assertIn("application/json", json_content_type)
        self.assertEqual(json.loads(json_body)["errors"][0]["code"], "CS_API_NOT_FOUND")

        artifact_status, artifact_content_type, artifact_body = _request(self.base_url, "/artifacts/missing-source")
        self.assertEqual(artifact_status, 404)
        self.assertIn("application/json", artifact_content_type)
        self.assertEqual(json.loads(artifact_body)["errors"][0]["code"], "CS_ARTIFACT_NOT_FOUND")

    def test_user_content_with_review_vocabulary_remains_visible(self) -> None:
        text = "Business continuity readiness scenario for the annual acceptance review."
        status, _, created_raw = _request(
            self.base_url,
            "/artifacts",
            method="POST",
            payload={**DEFAULT_SCOPE, "text": text},
        )
        self.assertEqual(status, 200)
        artifact_id = json.loads(created_raw)["artifact"]["artifact_id"]

        artifacts_html = self.fetch_product_html("/artifacts")
        detail_html = self.fetch_product_html(f"/artifacts/{artifact_id}?view=html")

        self.assertIn("Business continuity readiness scenario", artifacts_html)
        self.assertIn(text, detail_html)
        self.assertNotIn('data-product-surface="owner-record"', detail_html)
        self.assert_product_surface_is_clean(detail_html)
        self.assertFalse(product_ui._internal_product_record({"_preview": text}))
        self.assertTrue(product_ui._internal_product_record({"visibility": "owner_only", "_preview": text}))

    def test_internal_fixture_lineage_stays_out_of_normal_product_routes(self) -> None:
        store = LocalRuntimeStore(self.state_dir)
        source_text = "VS4-H01 scenario verifier review packet."
        artifact = store.ingest_text_artifact(
            source_text,
            DEFAULT_SCOPE,
            source_type="local_fixture",
            source_ref="fixtures/vs4/internal.txt",
            trust="trusted",
        )["artifact"]
        search = store.search("scenario verifier", **DEFAULT_SCOPE)["snapshot"]
        bundle = store.create_evidence_bundle(search["search_snapshot_id"], DEFAULT_SCOPE)["bundle"]
        brief = store.create_brief_from_evidence_bundle(bundle["evidence_bundle_id"], DEFAULT_SCOPE)["brief"]
        brief_title = str(brief.get("title") or "")
        self.assertTrue(brief_title)

        artifacts_html = self.fetch_product_html("/artifacts")
        briefs_html = self.fetch_product_html("/briefs")
        artifact_detail = self.fetch_product_html(f"/artifacts/{artifact['artifact_id']}?view=html")
        brief_detail = self.fetch_product_html(f"/briefs/{brief['brief_id']}?view=html")

        self.assertNotIn(source_text, artifacts_html)
        self.assertNotIn(brief_title, briefs_html)
        self.assertIn('data-product-surface="owner-record"', artifact_detail)
        self.assertIn('data-product-surface="owner-record"', brief_detail)
        self.assertNotIn(source_text, artifact_detail)
        self.assertNotIn(source_text, brief_detail)
        self.assert_product_surface_is_clean(artifacts_html)
        self.assert_product_surface_is_clean(briefs_html)

    def test_nondefault_workspace_is_visible_and_propagated(self) -> None:
        scope = {
            **DEFAULT_SCOPE,
            "namespace_id": "project",
            "workspace_id": "project-x",
        }
        query = urlencode(scope)
        status, _, html = _request(self.base_url, f"/?{query}", headers={"accept": "text/html"})

        self.assertEqual(status, 200)
        self.assertIn('data-namespace-id="project"', html)
        self.assertIn('data-workspace-id="project-x"', html)
        self.assertIn("project / project-x", html)
        self.assertIn("Workspace: project-x", html)
        self.assertIn(
            'const scope = {"namespace_id":"project","owner_id":"local-user","tenant_id":"local-dev","workspace_id":"project-x"};',
            html,
        )
        self.assertIn("preserveScope();", html)
        self.assertIn('window.location.href = scopedUrl("/artifacts/"', html)
        self.assertNotIn('workspace_id: "default"', html)

        source_text = "Project X scope isolation receipt."
        create_status, _, created_raw = _request(
            self.base_url,
            "/artifacts",
            method="POST",
            payload={**scope, "text": source_text},
        )
        self.assertEqual(create_status, 200)
        artifact_id = json.loads(created_raw)["artifact"]["artifact_id"]
        scoped_status, _, scoped_detail = _request(
            self.base_url,
            f"/artifacts/{artifact_id}?view=html&{query}",
            headers={"accept": "text/html"},
        )
        default_status, _, default_detail = _request(
            self.base_url,
            f"/artifacts/{artifact_id}?view=html",
            headers={"accept": "text/html"},
        )
        self.assertEqual(scoped_status, 200)
        self.assertIn(source_text, scoped_detail)
        self.assertIn("project / project-x", scoped_detail)
        self.assertEqual(default_status, 404)
        self.assertNotIn(source_text, default_detail)

        hostile_scope = {**scope, "workspace_id": "</script><script>alert(1)</script>"}
        _, _, hostile_html = _request(
            self.base_url,
            f"/?{urlencode(hostile_scope)}",
            headers={"accept": "text/html"},
        )
        self.assertNotIn("</script><script>alert(1)</script>", hostile_html)
        self.assertIn("\\u003c/script\\u003e", hostile_html)

    def test_workspace_load_failures_render_a_degraded_recovery_state(self) -> None:
        class BrokenStore:
            def _artifact_records(self, scope: dict[str, str]) -> list[dict[str, Any]]:
                raise OSError("artifact store unavailable")

            def _brief_records(self, scope: dict[str, str]) -> list[dict[str, Any]]:
                raise OSError("brief store unavailable")

            def _claim_records(self, scope: dict[str, str]) -> list[dict[str, Any]]:
                raise OSError("claim store unavailable")

            def _action_records(self, scope: dict[str, str]) -> list[dict[str, Any]]:
                raise OSError("action store unavailable")

            def _memory_records(self, scope: dict[str, str]) -> list[dict[str, Any]]:
                raise OSError("memory store unavailable")

            def _all_audit_events(self) -> list[dict[str, Any]]:
                raise OSError("audit store unavailable")

            def verify_audit(self) -> dict[str, Any]:
                raise OSError("audit verification unavailable")

        html = product_ui.render_product_page(ROOT, BrokenStore(), DEFAULT_SCOPE, "/", {})

        self.assertIn('data-product-state="degraded"', html)
        self.assertIn("Some workspace data could not be loaded", html)
        self.assertIn("saved sources", html)
        self.assertIn("audit integrity", html)
        self.assertIn("No records were changed", html)
        self.assertIn("Retry this page", html)

    def test_audit_page_does_not_claim_integrity_after_tampering(self) -> None:
        _request(
            self.base_url,
            "/artifacts",
            method="POST",
            payload={**DEFAULT_SCOPE, "text": "Audit integrity source."},
        )
        audit_path = self.state_dir / "audit" / "events.jsonl"
        events = audit_path.read_text().splitlines()
        event = json.loads(events[0])
        event["event_type"] = "artifact.tampered"
        events[0] = json.dumps(event, sort_keys=True)
        audit_path.write_text("\n".join(events) + "\n")

        audit_html = self.fetch_product_html("/audit")

        self.assertIn('data-audit-integrity-status="failed"', audit_html)
        self.assertIn("Integrity failed", audit_html)
        self.assertNotIn("Hash chain verified", audit_html)

    def test_screenshot_matrix_covers_primary_mobile_routes(self) -> None:
        ids = {
            "artifact_id": "art_mobile_matrix",
            "brief_id": "brief_mobile_matrix",
            "claim_id": "claim_mobile_matrix",
            "action_id": "action_mobile_matrix",
        }
        specs = capture_vs4.route_specs(ids)
        mobile_routes = {spec["route"] for spec in specs if spec.get("mobile")}
        day_zero_specs = capture_vs4.day_zero_route_specs()
        day_zero_mobile_names = {spec["name"] for spec in day_zero_specs if spec.get("mobile")}
        not_found_specs = capture_vs4.not_found_route_specs()
        not_found_names = {spec["name"] for spec in not_found_specs}
        interaction_specs = capture_vs4.interaction_route_specs()
        interaction_names = {spec["name"] for spec in interaction_specs}

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
        for name in [
            "day-zero-artifacts-mobile",
            "day-zero-briefs-mobile",
            "day-zero-claims-mobile",
            "day-zero-actions-mobile",
            "day-zero-inbox-mobile",
            "day-zero-audit-mobile",
        ]:
            with self.subTest(day_zero=name):
                self.assertIn(name, day_zero_mobile_names)
        for name in [
            "not-found-page-desktop",
            "not-found-page-mobile",
            "not-found-source-desktop",
            "not-found-source-mobile",
        ]:
            with self.subTest(not_found=name):
                self.assertIn(name, not_found_names)
        for name in [
            "home-validation-desktop",
            "home-validation-mobile",
        ]:
            with self.subTest(interaction=name):
                self.assertIn(name, interaction_names)

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
        self.assertIn("Source inspection workspace", html)
        self.assertIn("Detail path", html)
        self.assertIn("Saved sources", html)
        self.assertIn("Back to saved sources", html)
        self.assertIn("Original source", html)
        self.assertIn("Artifact evidence workspace", html)
        self.assertIn("Original source document viewer", html)
        self.assertIn("Original artifact preview", html)
        self.assertIn("Text preview controls", html)
        self.assertIn("1 / 1", html)
        self.assertIn("Source pages", html)
        self.assertIn("Preview rail", html)
        self.assertIn("Source text", html)
        self.assertIn("Source metadata", html)
        self.assertIn("Artifact inspection summary", html)
        self.assertIn("Source reading summary", html)
        self.assertIn("Original source summary", html)
        self.assertIn("Evidence links", html)
        self.assertIn("Preview mode", html)
        self.assertIn("Plain text preview", html)
        self.assertIn("Original content primary", html)
        self.assertIn("Details", html)
        self.assertIn("Tags", html)
        self.assertIn("Keyword summary", html)
        self.assertIn("Extracted keywords", html)
        self.assertIn("Linked work", html)
        self.assertIn("View linked evidence", html)
        self.assertIn("Provenance", html)
        self.assertIn("Open audit trail", html)
        self.assert_product_surface_is_clean(html)

        search_status, _, search_html = _request(self.base_url, "/search?q=renewal", headers={"accept": "text/html"})
        self.assertEqual(search_status, 200)
        self.assertIn("Vendor renewal", search_html)
        self.assertIn("What we found", search_html)
        self.assertIn("Suggested follow-ups", search_html)
        self.assertIn("Receipt coverage", search_html)
        self.assertIn("Search command center", search_html)
        self.assertIn("Type filters", search_html)
        self.assertIn("Current search controls", search_html)
        self.assertIn("Search mode: local keyword", search_html)
        self.assertIn("Keyword match", search_html)
        self.assertIn("Receipt-first results", search_html)
        self.assertIn("Local record receipt", search_html)
        self.assertIn("Open receipt", search_html)
        self.assertIn("Sort by: keyword match", search_html)
        self.assert_product_surface_is_clean(search_html)

        audit_html = self.fetch_product_html("/audit")
        self.assertIn('data-product-surface="audit"', audit_html)
        self.assertIn("Audit receipt workspace", audit_html)
        self.assertIn("Activity trail", audit_html)
        self.assertIn("Audit status", audit_html)
        self.assertIn("Latest readable receipt", audit_html)
        self.assertIn("Read activity receipts", audit_html)
        self.assertIn("Receipt summary", audit_html)
        self.assertIn("Audit lifecycle", audit_html)
        self.assertIn("Activity receipts", audit_html)
        self.assertIn("Readable receipts", audit_html)
        self.assertIn("Event stream", audit_html)
        self.assertIn("Raw event detail", audit_html)
        self.assertIn("Audit posture", audit_html)
        self.assertIn("Audit integrity checks", audit_html)
        self.assertIn("Integrity chain", audit_html)
        self.assertIn("Scope and recovery", audit_html)
        self.assertIn("Source saved", audit_html)
        self.assertIn("Evidence bundle prepared", audit_html)
        self.assertIn("Decision recorded", audit_html)
        self.assertIn("Action proposed", audit_html)
        self.assertIn("Hash chain verified", audit_html)
        self.assert_product_surface_is_clean(audit_html)

    def test_brief_chunk_citation_receipts_do_not_fallback_to_first_source(self) -> None:
        source_items = [
            {
                "ref": "artifact:primary",
                "label": "Source 1",
                "title": "Primary source",
                "snippet": "General source preview.",
                "href": "/artifacts/primary?view=html",
                "date": "2026-07-05",
                "fingerprint": "primary-fingerprint",
            },
            {
                "ref": "artifact:other",
                "label": "Source 2",
                "title": "Second unrelated source",
                "snippet": "This should not be used as a fallback.",
                "href": "/artifacts/other?view=html",
                "date": "2026-07-05",
                "fingerprint": "other-fingerprint",
            },
        ]
        brief = {
            "presented_as_fact": False,
            "citation_check_refs": ["citation_check:local"],
            "evidence_links": [
                {
                    "artifact_ref": "artifact:primary",
                    "evidence_chunk_ref": "evidence_chunk:chunk-1",
                    "snippet": "Exact source span for the first statement.",
                    "span": {"start": 10, "end": 52},
                }
            ],
            "key_point_citations": [
                {"statement": "First statement.", "citation_refs": ["evidence_chunk:chunk-1"]},
                {"statement": "Second statement.", "citation_refs": ["evidence_chunk:missing"]},
            ],
        }

        html = product_ui._statement_rows(brief, ["First statement.", "Second statement."], source_items)
        receipt = product_ui._brief_citation_receipt(brief, source_items)

        self.assertEqual(receipt["citation_refs_count"], 2)
        self.assertEqual(receipt["resolved_citation_count"], 1)
        self.assertEqual(receipt["unresolved_citation_count"], 1)
        self.assertIn("Evidence chunk", html)
        self.assertIn("10-52", html)
        self.assertIn("Exact source span for the first statement.", html)
        self.assertIn("Unresolved citation ref", html)
        self.assertIn("Unsupported or unresolved", html)
        self.assertNotIn("Second unrelated source", html)

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
        self.assertIn("Decision queue", briefs_html)
        self.assertIn("Brief reading queue", briefs_html)
        self.assertIn("Review lanes", briefs_html)
        self.assertIn("visible queue item", briefs_html)
        self.assertIn("Next review step", briefs_html)
        self.assertIn("Brief queue", briefs_html)
        self.assertIn("Source coverage", briefs_html)
        self.assertIn("Use next", briefs_html)
        self.assertIn("Brief posture", briefs_html)
        self.assertIn("Keyword summary", briefs_html)
        self.assertIn("Open brief", briefs_html)
        self.assert_product_surface_is_clean(briefs_html)

        brief_html = self.fetch_product_html(f"/briefs/{brief_id}?view=html")
        self.assertIn('data-product-surface="brief-detail"', brief_html)
        self.assertIn("Detail path", brief_html)
        self.assertIn("Back to briefs", brief_html)
        self.assertIn("Open audit trail", brief_html)
        self.assertIn("Brief reading workspace", brief_html)
        self.assertIn("What we found", brief_html)
        self.assertIn("Brief status", brief_html)
        self.assertIn("Receipt summary", brief_html)
        self.assertIn("Brief answer and receipt", brief_html)
        self.assertIn("Decision snapshot", brief_html)
        self.assertIn("Citation receipt", brief_html)
        self.assertIn("Label state", brief_html)
        self.assertIn("Drafted findings", brief_html)
        self.assertIn('data-presented-as-fact="false"', brief_html)
        self.assertIn('data-citation-check-refs-count="0"', brief_html)
        self.assertIn('data-resolved-citation-count="0"', brief_html)
        self.assertIn('data-unresolved-citation-count="0"', brief_html)
        self.assertIn("Source coverage", brief_html)
        self.assertIn("Findings with citations", brief_html)
        self.assertIn("What this brief cannot confirm", brief_html)
        self.assertIn("Suggested next steps", brief_html)
        self.assertIn("Use this brief", brief_html)
        self.assertIn("Citation trail", brief_html)
        self.assertIn("Sources used", brief_html)
        self.assertIn("Citation disclosure for finding", brief_html)
        self.assertIn("Inspect source", brief_html)
        self.assertIn("Source snippet", brief_html)
        self.assertIn("Full provenance", brief_html)
        self.assertIn("Audit trail", brief_html)
        self.assertIn("Source 1", brief_html)
        self.assertIn("Provenance", brief_html)
        self.assertIn("Keyword summary", brief_html)
        self.assertNotIn("extractive_fallback", brief_html)
        self.assertNotIn("Evidence-backed", brief_html)
        self.assertNotIn("decision-ready", brief_html)
        self.assert_product_surface_is_clean(brief_html)

        claim_html = self.fetch_product_html(f"/claims/{claim_id}?view=html")
        self.assertIn('data-product-surface="claim-detail"', claim_html)
        self.assertIn("Detail path", claim_html)
        self.assertIn("Back to claims", claim_html)
        self.assertIn("Open inbox", claim_html)
        self.assertIn("Trust ladder", claim_html)
        self.assertIn("Claim draft workspace", claim_html)
        self.assertIn("Evidence-to-decision path", claim_html)
        self.assertIn("Claim review summary", claim_html)
        self.assertIn("Claim state", claim_html)
        self.assertIn('data-source-support-attached="true"', claim_html)
        self.assertIn('data-evidence-backed-earned="false"', claim_html)
        self.assertIn("Source support", claim_html)
        self.assertIn("Evidence-backed locked", claim_html)
        self.assertIn("Citation checks", claim_html)
        self.assertIn("Citation checks required", claim_html)
        self.assertIn("Supporting evidence", claim_html)
        self.assertIn("Evidence picker controls", claim_html)
        self.assertIn("Sort: source order", claim_html)
        self.assertIn("Impacted objects", claim_html)
        self.assertIn("Related frameworks", claim_html)
        self.assertIn("Saved locally", claim_html)
        self.assertIn("Local draft", claim_html)
        self.assertIn("Review controls", claim_html)
        self.assertIn("Claim statement", claim_html)
        self.assertIn("Decision gate", claim_html)
        self.assertIn("Promote to decision locked", claim_html)
        self.assertIn("Review required before approval", claim_html)
        self.assert_product_surface_is_clean(claim_html)

        action_html = self.fetch_product_html(f"/actions/{action_id}?view=html")
        self.assertIn('data-product-surface="action-detail"', action_html)
        self.assertIn("Detail path", action_html)
        self.assertIn("Back to actions", action_html)
        self.assertIn("Open audit trail", action_html)
        self.assertNotIn('href="/claims">Back to claims', action_html)
        self.assertIn("Action preview", action_html)
        self.assertIn("Preview impact, policy, and approval history before any external step.", action_html)
        self.assertIn('data-approval-required="true"', action_html)
        self.assertIn('data-real-external-http-calls="0"', action_html)
        self.assertIn("Dry-run approval receipt", action_html)
        self.assertIn("Preview only", action_html)
        self.assertIn("Proposed change preview", action_html)
        self.assertIn("External call plan", action_html)
        self.assertIn("Approval gate", action_html)
        self.assertIn("No provider send has run", action_html)
        self.assertIn("Action review status", action_html)
        self.assertIn("Summary", action_html)
        self.assertIn("Dry-run sequence", action_html)
        self.assertIn("Impacted objects", action_html)
        self.assertIn("Proposed changes", action_html)
        self.assertIn("Before approval: preview only", action_html)
        self.assertIn("External calls", action_html)
        self.assertIn("Call preview", action_html)
        self.assertIn("Policy decision", action_html)
        self.assertIn("Policy checkpoints", action_html)
        self.assertIn("Risk and approval", action_html)
        self.assertIn("Request approval", action_html)
        self.assertIn("Required reason", action_html)
        self.assertIn("Approval history", action_html)
        self.assertIn("Sources", action_html)
        self.assertIn("Simulated in local mode", action_html)
        self.assertIn("Approval required", action_html)
        self.assertNotIn("external_writeback", action_html)
        self.assertNotIn("external writeback", action_html.lower())
        self.assert_product_surface_is_clean(action_html)

        artifacts_html = self.fetch_product_html("/artifacts")
        self.assertIn("Collection summary", artifacts_html)
        self.assertIn("Source register", artifacts_html)
        self.assertIn("Source posture", artifacts_html)
        self.assert_product_surface_is_clean(artifacts_html)

        claims_html = self.fetch_product_html("/claims")
        self.assertIn("Decision queue", claims_html)
        self.assertIn("Claim review lanes", claims_html)
        self.assertIn("visible queue item", claims_html)
        self.assertIn("Source-support lane", claims_html)
        self.assertIn("Evidence-backed locked", claims_html)
        self.assertIn("Citation checks required", claims_html)
        self.assertIn("Next review step", claims_html)
        self.assertIn("Claim review queue", claims_html)
        self.assertIn("Review posture", claims_html)
        self.assertIn("Trust ladder", claims_html)
        self.assert_product_surface_is_clean(claims_html)

        actions_html = self.fetch_product_html("/actions")
        self.assertIn("Decision queue", actions_html)
        self.assertIn("Action approval lanes", actions_html)
        self.assertIn("visible queue item", actions_html)
        self.assertIn("Approval lane", actions_html)
        self.assertIn("Next review step", actions_html)
        self.assertIn("Action preview queue", actions_html)
        self.assertIn("Dry-run posture", actions_html)
        self.assertIn("Action safeguards", actions_html)
        self.assert_product_surface_is_clean(actions_html)

        inbox_html = self.fetch_product_html("/inbox")
        self.assertIn('data-product-surface="inbox"', inbox_html)
        self.assertIn("Needs review", inbox_html)
        self.assertIn("Approval requests", inbox_html)
        self.assertIn("Triage summary", inbox_html)
        self.assertIn("open review items across one queue", inbox_html)
        self.assertIn("Filters", inbox_html)
        self.assertIn("Showing 3 open items", inbox_html)
        self.assertIn("1-3 of 3 items", inbox_html)
        self.assertIn("cs-inbox-select", inbox_html)
        self.assertIn("<span>Owner</span>", inbox_html)
        self.assertIn("Selected item", inbox_html)
        self.assertIn("Linked sources", inbox_html)
        self.assertIn("Why this is here", inbox_html)
        self.assertIn("Safety state", inbox_html)
        self.assertIn("Inbox receipt", inbox_html)
        self.assertIn("Next actions", inbox_html)
        self.assertIn("Review item", inbox_html)
        self.assertIn("Review sources", inbox_html)
        self.assertIn("Open audit trail", inbox_html)
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
        self.assertIsNone(VS4_PRODUCT_FORBIDDEN_RE.search(html))


if __name__ == "__main__":
    unittest.main()
