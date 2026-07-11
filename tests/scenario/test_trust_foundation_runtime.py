from __future__ import annotations

import hashlib
import io
import json
import os
import re
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from cornerstone_cli.main import main as cli_main
from cornerstone_cli.runtime import LocalRuntimeStore


SCOPE = {
    "tenant_id": "tenant-trust",
    "owner_id": "owner-trust",
    "namespace_id": "namespace-trust",
    "workspace_id": "workspace-trust",
}


class TrustFoundationRuntimeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.state_dir = Path(self.temporary.name)
        self.store = LocalRuntimeStore(self.state_dir)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def ingest(self, text: str, *, source_ref: str = "trust-probe") -> dict:
        return self.store.ingest_text_artifact(
            text,
            dict(SCOPE),
            source_type="user_paste",
            source_ref=source_ref,
        )["artifact"]

    def bundle_for(self, query: str) -> dict:
        snapshot = self.store.search(query, **SCOPE)["snapshot"]
        result = self.store.create_evidence_bundle(snapshot["search_snapshot_id"], dict(SCOPE))
        self.assertNotIn("status", result)
        return result["bundle"]

    def test_original_store_is_atomic_fresh_and_repairs_corrupt_existing_content(self) -> None:
        data = b"expected original bytes"
        corrupt = b"corrupt! original bytes"
        self.assertEqual(len(data), len(corrupt))
        artifact = self.store.ingest_artifact_bytes(
            data,
            filename="source.txt",
            source="upload",
            media_type="text/plain",
            derived_mode="auto",
            trust="untrusted",
            lineage_from=None,
            **SCOPE,
        )["artifact"]
        original_path = self.store.original_dir / artifact["checksum_sha256"]
        original_stat = original_path.stat()
        self.assertTrue(self.store.original_available(artifact))

        original_path.write_bytes(corrupt)
        os.utime(original_path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        self.assertFalse(self.store.original_available(artifact))

        repaired = self.store.ingest_artifact_bytes(
            data,
            filename="source.txt",
            source="upload",
            media_type="text/plain",
            derived_mode="auto",
            trust="untrusted",
            lineage_from=None,
            **SCOPE,
        )
        self.assertTrue(repaired["deduplicated"])
        self.assertEqual(repaired["audit_event"]["details"]["original_storage_status"], "repaired")
        self.assertEqual(original_path.read_bytes(), data)
        self.assertTrue(self.store.original_available(repaired["artifact"]))

    def test_interrupted_atomic_write_creates_no_artifact_audit_or_temporary_file(self) -> None:
        with patch("cornerstone_cli.archive_integrity.os.replace", side_effect=OSError("interrupted")):
            with self.assertRaises(OSError):
                self.store.ingest_text_artifact(
                    "interrupted archive write",
                    dict(SCOPE),
                    source_type="user_paste",
                    source_ref="interrupted",
                )
        self.assertEqual(list(self.store.record_dir.rglob("*.json")), [])
        self.assertFalse(self.store.audit_path.exists())
        self.assertEqual(list(self.store.original_dir.glob("*.tmp")), [])

    def test_atomic_repair_replaces_symlink_without_mutating_its_target(self) -> None:
        data = b"content addressed original"
        checksum = hashlib.sha256(data).hexdigest()
        self.store.original_dir.mkdir(parents=True, exist_ok=True)
        target = self.state_dir / "outside-target"
        target.write_bytes(b"outside remains unchanged")
        original_path = self.store.original_dir / checksum
        original_path.symlink_to(target)

        artifact = self.store.ingest_artifact_bytes(
            data,
            filename="source.txt",
            source="upload",
            media_type="text/plain",
            derived_mode="auto",
            trust="untrusted",
            lineage_from=None,
            **SCOPE,
        )["artifact"]
        self.assertFalse(original_path.is_symlink())
        self.assertEqual(original_path.read_bytes(), data)
        self.assertEqual(target.read_bytes(), b"outside remains unchanged")
        self.assertTrue(self.store.original_available(artifact))

    def test_artifact_ids_are_full_sha256_with_verified_legacy_compatibility(self) -> None:
        data = "legacy-compatible artifact"
        canonical = self.ingest(data)
        checksum = canonical["checksum_sha256"]
        self.assertEqual(canonical["artifact_id"], f"art_{checksum}")

        canonical_path = self.store.artifact_path(canonical["artifact_id"], SCOPE)
        legacy_id = f"art_{checksum[:16]}"
        legacy = json.loads(canonical_path.read_text())
        legacy["artifact_id"] = legacy_id
        legacy_path = self.store.artifact_path(legacy_id, SCOPE)
        legacy_path.parent.mkdir(parents=True, exist_ok=True)
        legacy_path.write_text(json.dumps(legacy, indent=2, sort_keys=True) + "\n")
        canonical_path.unlink()

        reused = self.store.ingest_text_artifact(
            data,
            dict(SCOPE),
            source_type="user_paste",
            source_ref="legacy-retry",
        )
        self.assertTrue(reused["deduplicated"])
        self.assertEqual(reused["artifact"]["artifact_id"], legacy_id)
        self.assertRegex(reused["artifact"]["derived"]["derived_representation_id"], r"^drv_[0-9a-f]{64}$")

    def test_short_prefix_conflict_does_not_deduplicate_different_content(self) -> None:
        incoming = b"incoming collision-safe content"
        checksum = hashlib.sha256(incoming).hexdigest()
        legacy_id = f"art_{checksum[:16]}"
        other = b"different stored content"
        other_checksum = hashlib.sha256(other).hexdigest()
        conflict = {
            "schema_version": "cs.artifact.v0",
            "artifact_id": legacy_id,
            "checksum_sha256": other_checksum,
            "content_identity": {"algorithm": "sha256", "value": other_checksum},
            "original_storage_ref": f"sha256:{other_checksum}",
            "original_size_bytes": len(other),
            "media_type": "text/plain",
            "scope": dict(SCOPE),
            "derived": {"status": "deferred"},
        }
        conflict_path = self.store.artifact_path(legacy_id, SCOPE)
        conflict_path.parent.mkdir(parents=True, exist_ok=True)
        conflict_path.write_text(json.dumps(conflict, indent=2, sort_keys=True) + "\n")

        created = self.store.ingest_artifact_bytes(
            incoming,
            filename="incoming.txt",
            source="upload",
            media_type="text/plain",
            derived_mode="auto",
            trust="untrusted",
            lineage_from=None,
            **SCOPE,
        )
        self.assertFalse(created["deduplicated"])
        self.assertEqual(created["artifact"]["artifact_id"], f"art_{checksum}")
        self.assertNotEqual(created["artifact"]["artifact_id"], legacy_id)

    def test_derived_representation_revision_survives_artifact_reprocessing(self) -> None:
        source_text = "Alice approved the vendor renewal."
        artifact = self.ingest(source_text)
        bundle = self.bundle_for("Alice approved vendor renewal")
        item = bundle["evidence_items"][0]
        original_representation_id = item["derived_representation_id"]

        replacement = self.store._create_derived_representation(
            artifact_id=artifact["artifact_id"],
            artifact_checksum_sha256=artifact["checksum_sha256"],
            text="Bob rejected the vendor renewal.",
            redacted=False,
        )
        stored_artifact = json.loads(self.store.artifact_path(artifact["artifact_id"], SCOPE).read_text())
        stored_artifact["derived"] = replacement
        self.store.artifact_path(artifact["artifact_id"], SCOPE).write_text(
            json.dumps(stored_artifact, indent=2, sort_keys=True) + "\n"
        )

        viewer = self.store.view_evidence_bundle(bundle["evidence_bundle_id"], dict(SCOPE))["viewer"]
        self.assertIn("Alice approved", viewer["viewer_items"][0]["derived"]["text_preview"])
        self.assertNotIn("Bob rejected", viewer["viewer_items"][0]["derived"]["text_preview"])
        self.assertEqual(
            viewer["viewer_items"][0]["derived"]["metadata"]["derived_representation_id"],
            original_representation_id,
        )

        chunks = self.store._build_evidence_chunks(
            bundle["evidence_items"],
            dict(SCOPE),
            query="Alice approved",
            evidence_bundle_id=bundle["evidence_bundle_id"],
            evidence_revision_sha256=bundle["evidence_revision_sha256"],
        )["chunks"]
        citation_ref = f"evidence_chunk:{chunks[0]['evidence_chunk_id']}"
        citation = self.store._citation_check(
            output_kind="trust_regression",
            citation_refs=[citation_ref],
            scope=dict(SCOPE),
        )
        self.assertEqual(citation["status"], "passed")
        self.assertEqual(citation["resolved_citations"][0]["derived_representation_ref"], item["derived_representation_ref"])

    def test_snapshot_bundle_and_derived_tampering_fail_closed(self) -> None:
        artifact = self.ingest("Alpha evidence remains immutable.")
        snapshot = self.store.search("Alpha evidence", **SCOPE)["snapshot"]
        snapshot_path = self.store.search_snapshot_path(snapshot["search_snapshot_id"])
        changed_snapshot = json.loads(snapshot_path.read_text())
        changed_snapshot["results"][0]["snippet"] = "forged snapshot snippet"
        snapshot_path.write_text(json.dumps(changed_snapshot, indent=2, sort_keys=True) + "\n")
        denied = self.store.create_evidence_bundle(snapshot["search_snapshot_id"], dict(SCOPE))
        self.assertEqual(denied["status"], "integrity_failed")
        self.assertEqual(denied["reason"], "search_snapshot_record_changed")

        clean_snapshot = self.store.search("Alpha evidence", **SCOPE)["snapshot"]
        bundle = self.store.create_evidence_bundle(clean_snapshot["search_snapshot_id"], dict(SCOPE))["bundle"]
        chunks = self.store._build_evidence_chunks(
            bundle["evidence_items"],
            dict(SCOPE),
            query="Alpha evidence",
            evidence_bundle_id=bundle["evidence_bundle_id"],
            evidence_revision_sha256=bundle["evidence_revision_sha256"],
        )["chunks"]
        citation_ref = f"evidence_chunk:{chunks[0]['evidence_chunk_id']}"

        bundle_path = self.store.evidence_bundle_path(bundle["evidence_bundle_id"])
        changed_bundle = json.loads(bundle_path.read_text())
        changed_bundle["evidence_items"][0]["snippet"] = "forged bundle snippet"
        bundle_path.write_text(json.dumps(changed_bundle, indent=2, sort_keys=True) + "\n")
        bundle_check = self.store._citation_check(
            output_kind="trust_regression",
            citation_refs=[citation_ref],
            scope=dict(SCOPE),
        )
        self.assertEqual(bundle_check["status"], "failed")
        self.assertEqual(bundle_check["evidence_revision_mismatch_count"], 1)

        bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n")
        derived_path = self.store.artifact_dir / bundle["evidence_items"][0]["derived_storage_ref"]
        original_derived = derived_path.read_bytes()
        derived_path.write_bytes(b"X" * len(original_derived))
        derived_check = self.store._citation_check(
            output_kind="trust_regression",
            citation_refs=[citation_ref],
            scope=dict(SCOPE),
        )
        self.assertEqual(derived_check["status"], "failed")
        self.assertEqual(derived_check["derived_representation_mismatch_count"], 1)
        self.assertTrue(self.store.original_available(artifact))

    def test_evidence_chunks_are_immutable_across_retrieval_queries(self) -> None:
        self.ingest("Alpha evidence and Gamma evidence share one source.")
        bundle = self.bundle_for("Alpha Gamma evidence")
        first = self.store._build_evidence_chunks(
            bundle["evidence_items"],
            dict(SCOPE),
            query="Alpha",
            evidence_bundle_id=bundle["evidence_bundle_id"],
            evidence_revision_sha256=bundle["evidence_revision_sha256"],
        )["chunks"][0]
        first_path = self.store.evidence_chunk_path(first["evidence_chunk_id"])
        first_bytes = first_path.read_bytes()
        second = self.store._build_evidence_chunks(
            bundle["evidence_items"],
            dict(SCOPE),
            query="MissingTerm",
            evidence_bundle_id=bundle["evidence_bundle_id"],
            evidence_revision_sha256=bundle["evidence_revision_sha256"],
        )["chunks"][0]
        self.assertRegex(first["evidence_chunk_id"], r"^chunk_[0-9a-f]{64}$")
        self.assertNotEqual(first["evidence_chunk_id"], second["evidence_chunk_id"])
        self.assertEqual(first_path.read_bytes(), first_bytes)

    def test_view_export_and_effective_reads_require_bound_provenance_and_verified_originals(self) -> None:
        source_text = "Vendor renewal auto-renewal is August 1."
        artifact = self.ingest(source_text, source_ref="source-a")
        bundle = self.bundle_for("Vendor renewal August 1")
        brief = self.store.create_brief_from_evidence_bundle(
            bundle["evidence_bundle_id"],
            dict(SCOPE),
        )["brief"]
        claim = self.store.create_claim_from_evidence_bundle(
            bundle["evidence_bundle_id"],
            source_text,
            dict(SCOPE),
        )["claim"]
        bound_source = bundle["evidence_items"][0]["source"]
        bound_provenance = bundle["evidence_items"][0]["provenance"]

        rebound = self.store.ingest_text_artifact(
            source_text,
            dict(SCOPE),
            source_type="user_paste",
            source_ref="source-b",
        )["artifact"]
        self.assertEqual(rebound["source"]["ref"], "source-b")

        viewer = self.store.view_evidence_bundle(bundle["evidence_bundle_id"], dict(SCOPE))["viewer"]
        self.assertEqual(viewer["viewer_items"][0]["original"]["source"], bound_source)
        self.assertEqual(viewer["viewer_items"][0]["provenance"], bound_provenance)
        exported = self.store.export_claim_basis(claim["claim_id"], dict(SCOPE))["claim_basis_export"]
        self.assertEqual(exported["source_artifacts"][0]["source"], bound_source)
        self.assertEqual(exported["source_artifacts"][0]["provenance"], bound_provenance)
        self.assertTrue(exported["freshness"]["reproducible_from_archive"])

        original_path = self.store.original_dir / artifact["checksum_sha256"]
        original_path.write_bytes(b"X" * artifact["original_size_bytes"])
        denied_view = self.store.view_evidence_bundle(bundle["evidence_bundle_id"], dict(SCOPE))
        denied_export = self.store.export_claim_basis(claim["claim_id"], dict(SCOPE))
        self.assertEqual(denied_view["status"], "integrity_failed")
        self.assertEqual(denied_view["reason"], "evidence_original_missing_or_changed")
        self.assertEqual(denied_export["status"], "integrity_failed")

        effective_claim = self.store.get_claim(claim["claim_id"])
        self.assertEqual(effective_claim["evidence_integrity"]["status"], "failed")
        self.assertEqual(effective_claim["statement_support"]["status"], "integrity_failed")
        self.assertFalse(effective_claim["authority"]["can_be_approved"])
        effective_brief = self.store.get_brief(brief["brief_id"])
        self.assertEqual(effective_brief["status"], "integrity_failed")
        self.assertEqual(effective_brief["trust_label"], "draft")
        self.assertFalse(effective_brief["presented_as_fact"])

    def test_chunk_identity_and_span_schema_tampering_fail_closed_without_exception(self) -> None:
        self.ingest("Alpha evidence has a valid source span.")
        bundle = self.bundle_for("Alpha evidence")
        chunk = self.store._build_evidence_chunks(
            bundle["evidence_items"],
            dict(SCOPE),
            query="Alpha evidence",
            evidence_bundle_id=bundle["evidence_bundle_id"],
            evidence_revision_sha256=bundle["evidence_revision_sha256"],
        )["chunks"][0]
        full_ref = f"evidence_chunk:{chunk['evidence_chunk_id']}"
        full_path = self.store.evidence_chunk_path(chunk["evidence_chunk_id"])
        tampered = json.loads(full_path.read_text())
        tampered["span"]["char_start"] = "oops"
        full_path.write_text(json.dumps(tampered, indent=2, sort_keys=True) + "\n")
        full_check = self.store._citation_check(
            output_kind="trust_regression",
            citation_refs=[full_ref],
            scope=dict(SCOPE),
        )
        self.assertEqual(full_check["status"], "failed")
        self.assertEqual(full_check["errors"][0]["code"], "UNRESOLVED_CITATION_REF")

        legacy_id = f"chunk_{'a' * 16}"
        legacy = dict(chunk)
        legacy["evidence_chunk_id"] = legacy_id
        legacy["evidence_refs"] = [
            f"evidence_chunk:{legacy_id}" if ref.startswith("evidence_chunk:") else ref
            for ref in legacy["evidence_refs"]
        ]
        legacy["span"] = {"char_start": "oops", "char_end": 5}
        self.store.evidence_chunk_path(legacy_id).write_text(json.dumps(legacy, indent=2, sort_keys=True) + "\n")
        legacy_check = self.store._citation_check(
            output_kind="trust_regression",
            citation_refs=[f"evidence_chunk:{legacy_id}"],
            scope=dict(SCOPE),
        )
        self.assertEqual(legacy_check["status"], "failed")
        self.assertEqual(legacy_check["errors"][0]["code"], "UNRESOLVED_CITATION_REF")

    def test_artifact_identity_mismatches_and_legacy_revisions_fail_closed_structurally(self) -> None:
        source_text = "Canonical identity must remain exact."
        artifact = self.ingest(source_text)
        artifact_path = self.store.artifact_path(artifact["artifact_id"], SCOPE)
        changed = json.loads(artifact_path.read_text())
        changed["original_size_bytes"] += 1
        artifact_path.write_text(json.dumps(changed, indent=2, sort_keys=True) + "\n")
        audit_count = len(self.store._all_audit_events())
        canonical_retry = self.store.ingest_text_artifact(
            source_text,
            dict(SCOPE),
            source_type="user_paste",
            source_ref="canonical-retry",
        )
        self.assertEqual(canonical_retry["status"], "integrity_failed")
        self.assertEqual(canonical_retry["reason"], "canonical_artifact_identity_mismatch")
        self.assertEqual(len(self.store._all_audit_events()), audit_count)
        ingest_output = io.StringIO()
        with patch("sys.stdout", ingest_output):
            ingest_exit = cli_main(
                [
                    "artifact",
                    "ingest",
                    "--text",
                    source_text,
                    "--state-dir",
                    str(self.state_dir),
                    "--tenant-id",
                    SCOPE["tenant_id"],
                    "--owner-id",
                    SCOPE["owner_id"],
                    "--namespace-id",
                    SCOPE["namespace_id"],
                    "--workspace-id",
                    SCOPE["workspace_id"],
                    "--json",
                ]
            )
        self.assertEqual(ingest_exit, 5)
        self.assertEqual(
            json.loads(ingest_output.getvalue())["errors"][0]["code"],
            "CS_ARTIFACT_IDENTITY_INTEGRITY_FAILED",
        )

        legacy_text = "Legacy Artifact identity must remain exact."
        legacy_artifact = self.ingest(legacy_text)
        legacy_checksum = legacy_artifact["checksum_sha256"]
        canonical_legacy_path = self.store.artifact_path(legacy_artifact["artifact_id"], SCOPE)
        legacy_id = f"art_{legacy_checksum[:16]}"
        legacy_record = json.loads(canonical_legacy_path.read_text())
        legacy_record["artifact_id"] = legacy_id
        legacy_record["original_size_bytes"] += 1
        legacy_path = self.store.artifact_path(legacy_id, SCOPE)
        legacy_path.write_text(json.dumps(legacy_record, indent=2, sort_keys=True) + "\n")
        canonical_legacy_path.unlink()
        legacy_retry = self.store.ingest_text_artifact(
            legacy_text,
            dict(SCOPE),
            source_type="user_paste",
            source_ref="legacy-integrity-retry",
        )
        self.assertEqual(legacy_retry["status"], "integrity_failed")
        self.assertEqual(legacy_retry["reason"], "legacy_artifact_identity_mismatch")

        clean_snapshot = self.store.search("Canonical identity", **SCOPE)["snapshot"]
        snapshot_path = self.store.search_snapshot_path(clean_snapshot["search_snapshot_id"])
        legacy_snapshot = json.loads(snapshot_path.read_text())
        legacy_snapshot.pop("result_revision_sha256")
        snapshot_path.write_text(json.dumps(legacy_snapshot, indent=2, sort_keys=True) + "\n")
        denied_snapshot = self.store.read_search_snapshot(
            clean_snapshot["search_snapshot_id"],
            dict(SCOPE),
            reason="legacy_revision_probe",
        )
        self.assertEqual(denied_snapshot["status"], "integrity_failed")

        output = io.StringIO()
        with patch("sys.stdout", output):
            exit_code = cli_main(
                [
                    "search",
                    "snapshot",
                    "show",
                    clean_snapshot["search_snapshot_id"],
                    "--state-dir",
                    str(self.state_dir),
                    "--tenant-id",
                    SCOPE["tenant_id"],
                    "--owner-id",
                    SCOPE["owner_id"],
                    "--namespace-id",
                    SCOPE["namespace_id"],
                    "--workspace-id",
                    SCOPE["workspace_id"],
                    "--json",
                ]
            )
        self.assertEqual(exit_code, 5)
        cli_payload = json.loads(output.getvalue())
        self.assertEqual(cli_payload["errors"][0]["code"], "CS_SEARCH_SNAPSHOT_INTEGRITY_FAILED")

        bundle_source = self.ingest("Legacy Bundle revisions are retired fail-closed.")
        self.assertTrue(bundle_source["artifact_id"].startswith("art_"))
        bundle = self.bundle_for("Legacy Bundle revisions")
        bundle_path = self.store.evidence_bundle_path(bundle["evidence_bundle_id"])
        legacy_bundle = json.loads(bundle_path.read_text())
        legacy_bundle.pop("evidence_revision_sha256")
        bundle_path.write_text(json.dumps(legacy_bundle, indent=2, sort_keys=True) + "\n")
        denied_bundle = self.store.show_evidence_bundle(bundle["evidence_bundle_id"], dict(SCOPE))
        self.assertEqual(denied_bundle["status"], "integrity_failed")
        self.assertEqual(denied_bundle["reason"], "evidence_bundle_record_changed")

    def test_semantic_inversions_remain_source_supported_drafts_and_cannot_be_approved(self) -> None:
        self.ingest(
            "Alice sold the asset to Bob. Flooding caused equipment failure. "
            "Alice scored greater than Bob. Revenue rose from 10 to 20. "
            "The vendor may terminate the agreement. Payment occurs before delivery. "
            "Alice did not approve the plan, and Bob did sign the plan."
        )
        bundle = self.bundle_for("Alice Bob asset flooding failure revenue vendor payment delivery plan")
        inverted = [
            "Bob sold the asset to Alice.",
            "Equipment failure caused flooding.",
            "Bob scored greater than Alice.",
            "Revenue fell from 20 to 10.",
            "The vendor must terminate the agreement.",
            "Payment occurs after delivery.",
            "Alice did approve the plan, and Bob did not sign the plan.",
        ]
        for statement in inverted:
            with self.subTest(statement=statement):
                claim = self.store.create_claim_from_evidence_bundle(
                    bundle["evidence_bundle_id"], statement, dict(SCOPE)
                )["claim"]
                self.assertEqual(claim["status"], "draft")
                self.assertEqual(claim["trust_state"], "draft")
                self.assertEqual(claim["statement_support"]["status"], "source_supported")
                self.assertFalse(claim["statement_support"]["semantic_support_verified"])
                self.assertFalse(claim["statement_support"]["approval_eligible"])
                self.assertFalse(claim["authority"]["can_be_approved"])
                approval = self.store.approve_claim(claim["claim_id"], dict(SCOPE))
                self.assertEqual(approval["status"], "semantic_support_required")
                self.assertEqual(approval["claim"]["trust_state"], "draft")
                self.assertFalse(approval["claim"]["authority"]["can_publish_shared_truth"])

    def test_legacy_claim_authority_is_effectively_downgraded_without_mutating_history(self) -> None:
        legacy = {
            "schema_version": "cs.claim.v0",
            "claim_id": "claim_legacy_authority",
            "status": "approved",
            "trust_state": "approved",
            "statement": "A lexical Claim was previously approved.",
            "scope": dict(SCOPE),
            "statement_support": {
                "status": "passed",
                "statement_support_state": "deterministic_anchor_passed",
                "semantic_faithfulness_state": "human_required",
                "semantic_support_verified": True,
                "approval_eligible": True,
            },
            "authority": {
                "can_be_approved": True,
                "can_publish_shared_truth": True,
                "can_drive_autonomous_action": True,
            },
            "evidence_bundle": {"artifact_refs": ["artifact:legacy"]},
        }
        path = self.store.claim_path(legacy["claim_id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(legacy, indent=2, sort_keys=True) + "\n")

        effective = self.store.get_claim(legacy["claim_id"])
        self.assertEqual(effective["status"], "draft")
        self.assertEqual(effective["trust_state"], "draft")
        self.assertEqual(effective["statement_support"]["status"], "not_verified")
        self.assertFalse(effective["statement_support"]["semantic_support_verified"])
        self.assertFalse(effective["authority"]["can_publish_shared_truth"])
        self.assertFalse(effective["authority"]["can_drive_autonomous_action"])
        self.assertEqual(effective["effective_authority_migration"]["recorded_status"], "approved")
        self.assertEqual(json.loads(path.read_text())["status"], "approved")

    def test_brief_and_claim_payload_identity_are_revalidated_on_read(self) -> None:
        self.ingest("Acme renewal is due on July 31.")
        bundle = self.bundle_for("Acme renewal")

        brief = self.store.create_brief_from_evidence_bundle(
            bundle["evidence_bundle_id"],
            dict(SCOPE),
        )["brief"]
        brief_path = self.store.brief_path(brief["brief_id"])
        forged_brief = json.loads(brief_path.read_text())
        forged_brief["trust_label"] = "evidence_backed"
        forged_brief["status"] = "evidence_backed"
        forged_brief["presented_as_fact"] = True
        forged_brief["key_points"] = ["Forged conclusion with no citation."]
        brief_path.write_text(json.dumps(forged_brief, indent=2, sort_keys=True) + "\n")

        effective_brief = self.store.get_brief(brief["brief_id"])
        self.assertEqual(effective_brief["status"], "integrity_failed")
        self.assertEqual(effective_brief["trust_label"], "draft")
        self.assertFalse(effective_brief["presented_as_fact"])
        self.assertEqual(effective_brief["evidence_integrity"]["reason"], "brief_record_changed")

        claim = self.store.create_claim_from_evidence_bundle(
            bundle["evidence_bundle_id"],
            "Acme renewal is due on July 31.",
            dict(SCOPE),
        )["claim"]
        claim_path = self.store.claim_path(claim["claim_id"])
        forged_claim = json.loads(claim_path.read_text())
        forged_claim["statement"] = "Acme renewal is not due on July 31."
        claim_path.write_text(json.dumps(forged_claim, indent=2, sort_keys=True) + "\n")

        effective_claim = self.store.get_claim(claim["claim_id"])
        self.assertEqual(effective_claim["statement_support"]["status"], "integrity_failed")
        self.assertFalse(effective_claim["authority"]["can_be_approved"])
        self.assertEqual(effective_claim["evidence_integrity"]["reason"], "claim_record_changed")

    def test_evidence_backed_brief_requires_current_immutable_citations(self) -> None:
        self.ingest("Acme renewal is due on July 31.")
        bundle = self.bundle_for("Acme renewal")

        def generated(*_: object, **kwargs: object) -> dict:
            match = re.search(r"evidence_chunk:[a-zA-Z0-9_-]+", str(kwargs.get("prompt") or ""))
            self.assertIsNotNone(match)
            return {
                "title": "Acme renewal",
                "key_points": [
                    {
                        "statement": "Acme renewal is due on July 31.",
                        "citation_refs": [match.group(0)],
                    }
                ],
                "uncertainty": [],
                "recommended_next_steps": [],
                "contradictions": [],
            }

        with patch("cornerstone_cli.runtime._ollama_embedding", return_value=[1.0, 0.0]), patch(
            "cornerstone_cli.runtime._ollama_generate_json",
            side_effect=generated,
        ):
            brief = self.store.create_brief_from_evidence_bundle(
                bundle["evidence_bundle_id"],
                dict(SCOPE),
                model_provider="ollama",
            )["brief"]

        self.assertEqual(self.store.get_brief(brief["brief_id"])["trust_label"], "evidence_backed")
        chunk_id = brief["citation_refs"][0].split(":", 1)[1]
        self.store.evidence_chunk_path(chunk_id).unlink()
        effective = self.store.get_brief(brief["brief_id"])
        self.assertEqual(effective["status"], "integrity_failed")
        self.assertEqual(effective["trust_label"], "draft")
        self.assertFalse(effective["presented_as_fact"])
        self.assertIn("UNRESOLVED_CITATION_REF", effective["evidence_integrity"]["reason"])

    def test_artifact_read_rejects_cross_identity_substitution(self) -> None:
        first = self.ingest("Artifact A content", source_ref="artifact-a")
        second = self.ingest("Artifact B replacement content", source_ref="artifact-b")
        first_path = self.store.artifact_path(first["artifact_id"], SCOPE)
        substituted = json.loads(first_path.read_text())
        for key in (
            "checksum_sha256",
            "content_identity",
            "original_storage_ref",
            "original_size_bytes",
        ):
            substituted[key] = second[key]
        first_path.write_text(json.dumps(substituted, indent=2, sort_keys=True) + "\n")

        shown = self.store.read_product_record(
            "artifact",
            first["artifact_id"],
            dict(SCOPE),
            reason="identity-substitution-probe",
        )
        downloaded = self.store.read_artifact_original(
            first["artifact_id"],
            dict(SCOPE),
            reason="identity-substitution-probe",
        )
        self.assertEqual(shown["status"], "integrity_failed")
        self.assertEqual(downloaded["status"], "integrity_failed")
        self.assertEqual(shown["reason"], "canonical_artifact_checksum_mismatch")

    def test_full_record_identity_and_malformed_active_records_fail_structurally(self) -> None:
        self.ingest("The renewal owner is Mina.")
        snapshot = self.store.search("renewal owner", **SCOPE)["snapshot"]
        snapshot_path = self.store.search_snapshot_path(snapshot["search_snapshot_id"])
        tampered_snapshot = json.loads(snapshot_path.read_text())
        tampered_snapshot["query"] = "forged display query"
        snapshot_path.write_text(json.dumps(tampered_snapshot, indent=2, sort_keys=True) + "\n")
        snapshot_read = self.store.read_search_snapshot(
            snapshot["search_snapshot_id"],
            dict(SCOPE),
            reason="full-record-probe",
        )
        self.assertEqual(snapshot_read["status"], "integrity_failed")
        self.assertEqual(snapshot_read["reason"], "search_snapshot_record_changed")

        fresh_snapshot = self.store.search("renewal owner Mina", **SCOPE)["snapshot"]
        bundle = self.store.create_evidence_bundle(fresh_snapshot["search_snapshot_id"], dict(SCOPE))["bundle"]
        bundle_path = self.store.evidence_bundle_path(bundle["evidence_bundle_id"])
        tampered_bundle = json.loads(bundle_path.read_text())
        tampered_bundle["result_snapshot"]["query"] = "forged embedded query"
        bundle_path.write_text(json.dumps(tampered_bundle, indent=2, sort_keys=True) + "\n")
        bundle_read = self.store.show_evidence_bundle(bundle["evidence_bundle_id"], dict(SCOPE))
        self.assertEqual(bundle_read["status"], "integrity_failed")
        self.assertEqual(bundle_read["reason"], "evidence_bundle_record_changed")

        malformed = self.store.create_unsupported_claim("Malformed record probe", dict(SCOPE))["claim"]
        self.store.claim_path(malformed["claim_id"]).write_text("{not-json")
        claim_read = self.store.read_product_record(
            "claim",
            malformed["claim_id"],
            dict(SCOPE),
            reason="malformed-record-probe",
        )
        self.assertEqual(claim_read["status"], "integrity_failed")
        self.assertEqual(claim_read["reason"], "claim_record_unreadable_or_changed")

        substituted = self.store.create_unsupported_claim(
            "Substituted record identity probe",
            dict(SCOPE),
        )["claim"]
        substituted_path = self.store.claim_path(substituted["claim_id"])
        substituted_record = json.loads(substituted_path.read_text())
        substituted_record["claim_id"] = f"claim_{'f' * 16}"
        substituted_path.write_text(json.dumps(substituted_record, indent=2, sort_keys=True) + "\n")
        substituted_read = self.store.read_product_record(
            "claim",
            substituted["claim_id"],
            dict(SCOPE),
            reason="substituted-record-probe",
        )
        self.assertEqual(substituted_read["status"], "integrity_failed")
        self.assertEqual(substituted_read["reason"], "claim_record_unreadable_or_changed")

        memory_id = f"memory_{'a' * 16}"
        memory_path = self.store.memory_path(memory_id)
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        memory_path.write_text(
            json.dumps(
                {
                    "schema_version": "cs.memory.v0",
                    "memory_id": f"memory_{'b' * 16}",
                    "scope": dict(SCOPE),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        memory_read = self.store.read_product_record(
            "memory",
            memory_id,
            dict(SCOPE),
            reason="substituted-memory-probe",
        )
        self.assertEqual(memory_read["status"], "integrity_failed")
        self.assertEqual(memory_read["reason"], "memory_record_unreadable_or_changed")

    def test_concurrent_same_content_ingest_preserves_both_observations(self) -> None:
        def ingest(source_ref: str) -> dict:
            return self.store.ingest_text_artifact(
                "One immutable source observed concurrently.",
                dict(SCOPE),
                source_type="user_paste",
                source_ref=source_ref,
            )

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(ingest, ["source-a", "source-b"]))

        self.assertEqual(sorted(result["deduplicated"] for result in results), [False, True])
        artifact_id = results[0]["artifact"]["artifact_id"]
        stored = self.store.get_artifact(artifact_id, dict(SCOPE))
        observed_refs = {
            str(source.get("ref"))
            for source in stored.get("source_history", [])
            if isinstance(source, dict)
        }
        self.assertEqual(observed_refs, {"source-a", "source-b"})
        events = self.store._all_audit_events()
        self.assertEqual(sum(event["event_type"] == "artifact.ingested" for event in events), 1)
        self.assertEqual(sum(event["event_type"] == "artifact.deduplicated" for event in events), 1)

    def test_draft_claim_cannot_approve_or_execute_action_before_replay(self) -> None:
        self.ingest("The renewal needs an internal status update.")
        bundle = self.bundle_for("renewal internal status update")
        claim = self.store.create_claim_from_evidence_bundle(
            bundle["evidence_bundle_id"],
            "The renewal needs an internal status update.",
            dict(SCOPE),
        )["claim"]
        mission = self.store.create_mission_contract(
            "Track the renewal",
            dict(SCOPE),
            claim_id=claim["claim_id"],
        )["mission"]
        self.store.activate_mission(mission["mission_id"], dict(SCOPE), mode="autopilot")
        action = self.store.propose_action(
            mission["mission_id"],
            claim["claim_id"],
            "internal_status_update",
            "low",
            dict(SCOPE),
            goal="Record renewal status",
        )["action_card"]
        self.assertEqual(action["policy_decision"]["reason_code"], "CS_ACTION_CLAIM_AUTHORITY_REQUIRED")
        self.assertFalse(action["execution"]["can_execute_now"])
        self.assertEqual(action["execution"]["status"], "blocked_by_claim_authority")

        approval = self.store.approve_action(action["action_id"], dict(SCOPE), approver=SCOPE["owner_id"])
        self.assertEqual(approval["status"], "policy_denied")
        self.assertEqual(approval["reason_code"], "CS_ACTION_CLAIM_AUTHORITY_REQUIRED")
        execution = self.store.execute_action(action["action_id"], dict(SCOPE))
        self.assertEqual(execution["status"], "policy_denied")
        self.assertEqual(execution["reason_code"], "CS_ACTION_CLAIM_AUTHORITY_REQUIRED")
        self.assertFalse(self.store.workflow_run_dir.exists())
        self.assertFalse(self.store.action_result_dir.exists())


if __name__ == "__main__":
    unittest.main()
