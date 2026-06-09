from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from time import perf_counter
from pathlib import Path
from typing import Any

from cornerstone_cli.validators import redact_text


UNSAFE_INSTRUCTION_PATTERNS = [
    re.compile(r"ignore (all )?(previous|prior) instructions", re.IGNORECASE),
    re.compile(r"\b(call|invoke|use|run)\b.*\b(tool|api|http|url|webhook)\b", re.IGNORECASE),
    re.compile(r"\b(authority|permission|approval)\b.*\b(now|granted|expanded|bypass)\b", re.IGNORECASE),
]

SEMANTIC_ALIASES = {
    "retain": ["keep", "preserve", "stored"],
    "retained": ["keep", "preserve", "stored"],
    "raw": ["original", "source"],
    "evidence": ["source", "material", "artifact"],
    "proof": ["evidence", "source", "material"],
    "source": ["original", "material"],
    "preserve": ["keep", "stored", "original"],
    "preserved": ["keep", "stored", "original"],
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(encoded)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def detect_unsafe_instructions(text: str) -> list[str]:
    blocked: list[str] = []
    for pattern in UNSAFE_INSTRUCTION_PATTERNS:
        if pattern.search(text):
            blocked.append(pattern.pattern)
    return blocked


def search_terms(text: str) -> list[str]:
    return [term for term in re.findall(r"[a-z0-9-]+", text.lower()) if term]


def scope_key(scope: dict[str, str]) -> str:
    return _json_hash(
        {
            "tenant_id": scope["tenant_id"],
            "owner_id": scope["owner_id"],
            "namespace_id": scope["namespace_id"],
            "workspace_id": scope["workspace_id"],
        }
    )[:16]


class LocalRuntimeStore:
    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.artifact_dir = state_dir / "artifacts"
        self.original_dir = self.artifact_dir / "originals"
        self.record_dir = self.artifact_dir / "records"
        self.audit_path = state_dir / "audit" / "events.jsonl"

    def reset(self) -> None:
        if self.state_dir.exists():
            shutil.rmtree(self.state_dir)

    def _last_audit_hash(self) -> str:
        if not self.audit_path.exists():
            return "0" * 64
        last = ""
        for line in self.audit_path.read_text().splitlines():
            if line.strip():
                last = line
        if not last:
            return "0" * 64
        return json.loads(last)["event_hash"]

    def append_audit(self, event_type: str, scope: dict[str, str], subject: dict[str, str], details: dict[str, Any]) -> dict[str, Any]:
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        event_without_hash = {
            "schema_version": "cs.audit_event.v0",
            "event_type": event_type,
            "occurred_at": utc_now(),
            "tenant_id": scope["tenant_id"],
            "owner_id": scope["owner_id"],
            "namespace_id": scope["namespace_id"],
            "workspace_id": scope["workspace_id"],
            "subject": subject,
            "details": details,
            "previous_hash": self._last_audit_hash(),
        }
        event_hash = _json_hash(event_without_hash)
        event = dict(event_without_hash)
        event["event_id"] = f"audit_{event_hash[:16]}"
        event["event_hash"] = event_hash
        with self.audit_path.open("a") as file:
            file.write(json.dumps(event, sort_keys=True) + "\n")
        return event

    def verify_audit(self) -> dict[str, Any]:
        previous_hash = "0" * 64
        event_count = 0
        errors: list[dict[str, str]] = []
        if not self.audit_path.exists():
            return {"status": "success", "event_count": 0, "errors": []}

        for line_no, line in enumerate(self.audit_path.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            event_count += 1
            try:
                event = json.loads(line)
            except ValueError as error:
                errors.append({"line": str(line_no), "code": "AUDIT_JSON_INVALID", "message": str(error)})
                continue
            event_hash = event.get("event_hash")
            event_without_hash = dict(event)
            event_without_hash.pop("event_id", None)
            event_without_hash.pop("event_hash", None)
            calculated = _json_hash(event_without_hash)
            if event.get("previous_hash") != previous_hash:
                errors.append({"line": str(line_no), "code": "AUDIT_PREVIOUS_HASH_MISMATCH", "message": "Previous hash does not match prior event."})
            if event_hash != calculated:
                errors.append({"line": str(line_no), "code": "AUDIT_EVENT_HASH_MISMATCH", "message": "Event hash does not match event body."})
            previous_hash = event_hash or ""
        return {"status": "success" if not errors else "failed", "event_count": event_count, "errors": errors}

    def artifact_path(self, artifact_id: str, scope: dict[str, str] | None = None) -> Path:
        if scope is None:
            return self.record_dir / f"{artifact_id}.json"
        return self.record_dir / scope_key(scope) / f"{artifact_id}.json"

    def search_snapshot_path(self, snapshot_id: str) -> Path:
        return self.state_dir / "search" / "snapshots" / f"{snapshot_id}.json"

    def evidence_bundle_path(self, bundle_id: str) -> Path:
        return self.state_dir / "evidence" / "bundles" / f"{bundle_id}.json"

    def claim_path(self, claim_id: str) -> Path:
        return self.state_dir / "claims" / f"{claim_id}.json"

    def get_artifact(self, artifact_id: str, scope: dict[str, str] | None = None) -> dict[str, Any] | None:
        if scope is not None:
            scoped_path = self.artifact_path(artifact_id, scope)
            if scoped_path.exists():
                return _read_json(scoped_path)
            legacy_path = self.artifact_path(artifact_id)
            if legacy_path.exists():
                legacy = _read_json(legacy_path)
                if legacy.get("scope") == scope:
                    return legacy
            return None

        candidate_paths = sorted(self.record_dir.glob(f"*/{artifact_id}.json")) if self.record_dir.exists() else []
        legacy_path = self.artifact_path(artifact_id)
        if legacy_path.exists():
            candidate_paths.append(legacy_path)
        if not candidate_paths:
            return None
        return _read_json(candidate_paths[0])

    def get_search_snapshot(self, snapshot_id: str) -> dict[str, Any] | None:
        path = self.search_snapshot_path(snapshot_id)
        if not path.exists():
            return None
        return _read_json(path)

    def get_evidence_bundle(self, bundle_id: str) -> dict[str, Any] | None:
        path = self.evidence_bundle_path(bundle_id)
        if not path.exists():
            return None
        return _read_json(path)

    def get_claim(self, claim_id: str) -> dict[str, Any] | None:
        path = self.claim_path(claim_id)
        if not path.exists():
            return None
        return _read_json(path)

    def _artifact_records(self, scope: dict[str, str]) -> list[dict[str, Any]]:
        if not self.record_dir.exists():
            return []
        records = []
        for path in sorted(self.record_dir.glob("**/*.json")):
            record = _read_json(path)
            if record.get("scope") == scope:
                records.append(record)
        return records

    def _claim_records(self, scope: dict[str, str]) -> list[dict[str, Any]]:
        claim_dir = self.state_dir / "claims"
        if not claim_dir.exists():
            return []
        records = []
        for path in sorted(claim_dir.glob("*.json")):
            record = _read_json(path)
            if record.get("scope") == scope:
                records.append(record)
        return records

    def related_claims_for_artifact(self, artifact_id: str, scope: dict[str, str]) -> list[dict[str, Any]]:
        related = []
        artifact_ref = f"artifact:{artifact_id}"
        for claim in self._claim_records(scope):
            evidence = claim.get("evidence_bundle", {})
            if artifact_ref in evidence.get("artifact_refs", []):
                related.append(
                    {
                        "claim_id": claim["claim_id"],
                        "status": claim.get("status"),
                        "statement": claim.get("statement"),
                        "evidence_bundle_id": evidence.get("evidence_bundle_id"),
                    }
                )
        return related

    def _derived_text(self, artifact: dict[str, Any]) -> str:
        text_ref = artifact.get("derived", {}).get("text_ref")
        if not text_ref:
            return ""
        path = self.artifact_dir / text_ref
        if not path.exists():
            return ""
        return path.read_text()

    def derived_text_preview(self, artifact: dict[str, Any], limit: int = 500) -> str:
        return self._derived_text(artifact)[:limit].replace("\n", " ").strip()

    def search(self, query: str, *, tenant_id: str, owner_id: str, namespace_id: str, workspace_id: str) -> dict[str, Any]:
        started = perf_counter()
        scope = {
            "tenant_id": tenant_id,
            "owner_id": owner_id,
            "namespace_id": namespace_id,
            "workspace_id": workspace_id,
        }
        query_terms = search_terms(query)
        results: list[dict[str, Any]] = []
        for artifact in self._artifact_records(scope):
            text = self._derived_text(artifact)
            haystack = text.lower()
            score = 0
            match_reasons: list[dict[str, str]] = []
            retrieval_modes: set[str] = set()
            if query.lower() in haystack:
                score += 10
                retrieval_modes.add("exact")
                match_reasons.append({"type": "exact", "query": query})
            for term in query_terms:
                if term in haystack:
                    score += 1
                    retrieval_modes.add("keyword")
                    match_reasons.append({"type": "keyword", "query_term": term, "matched_term": term})
                for alias in SEMANTIC_ALIASES.get(term, []):
                    if alias in haystack and alias != term:
                        score += 2
                        retrieval_modes.add("semantic")
                        match_reasons.append({"type": "semantic_alias", "query_term": term, "matched_term": alias})
            if score <= 0:
                continue
            first_term = ""
            for reason in match_reasons:
                candidate = reason.get("matched_term") or reason.get("query_term") or reason.get("query")
                if isinstance(candidate, str) and candidate.lower() in haystack:
                    first_term = candidate.lower()
                    break
            start = haystack.find(first_term) if first_term else 0
            start = max(start, 0)
            snippet = text[start : start + 180].replace("\n", " ").strip()
            results.append(
                {
                    "artifact_id": artifact["artifact_id"],
                    "score": score,
                    "snippet": snippet,
                    "derived_text_ref": artifact.get("derived", {}).get("text_ref"),
                    "original_storage_ref": artifact.get("original_storage_ref"),
                    "scope": artifact.get("scope"),
                    "retrieval_modes": sorted(retrieval_modes),
                    "match_reasons": match_reasons,
                    "evidence_refs": [
                        f"artifact:{artifact['artifact_id']}",
                        f"storage:{artifact['original_storage_ref']}",
                    ],
                }
            )
        results.sort(key=lambda row: (-int(row["score"]), row["artifact_id"]))
        duration_ms = round((perf_counter() - started) * 1000, 3)
        snapshot_base = {
            "schema_version": "cs.search_snapshot.v0",
            "query": query,
            "filters": scope,
            "result_count": len(results),
            "results": results,
            "created_at": utc_now(),
            "duration_ms": duration_ms,
        }
        snapshot_id = f"search_{_json_hash(snapshot_base)[:16]}"
        snapshot = dict(snapshot_base)
        snapshot["search_snapshot_id"] = snapshot_id
        _write_json(self.search_snapshot_path(snapshot_id), snapshot)
        event = self.append_audit(
            "search.snapshot.created",
            scope,
            {"type": "search_snapshot", "id": snapshot_id},
            {"query": query, "result_count": len(results), "duration_ms": duration_ms},
        )
        return {"snapshot": snapshot, "audit_event": event}

    def create_evidence_bundle(self, snapshot_id: str, scope: dict[str, str]) -> dict[str, Any]:
        snapshot = self.get_search_snapshot(snapshot_id)
        if snapshot is None:
            return {"status": "not_found"}
        if snapshot.get("filters") != scope:
            return {"status": "scope_denied", "resource_scope": snapshot.get("filters")}
        evidence_items = []
        for result in snapshot.get("results", []):
            artifact = self.get_artifact(result["artifact_id"], scope)
            if artifact is None:
                return {"status": "not_found", "artifact_id": result["artifact_id"]}
            if artifact.get("scope") != scope:
                return {"status": "scope_denied", "resource_scope": artifact.get("scope")}
            derived_text = self._derived_text(artifact)
            snippet = result.get("snippet") or derived_text[:180].replace("\n", " ").strip()
            if not snippet:
                continue
            evidence_items.append(
                {
                    "artifact_id": artifact["artifact_id"],
                    "search_snapshot_id": snapshot_id,
                    "snippet": snippet,
                    "original_storage_ref": artifact.get("original_storage_ref"),
                    "derived_text_ref": artifact.get("derived", {}).get("text_ref"),
                    "source": artifact.get("source"),
                    "provenance": artifact.get("provenance"),
                }
            )
        bundle_base = {
            "schema_version": "cs.evidence_bundle.v0",
            "search_snapshot_id": snapshot_id,
            "query": snapshot["query"],
            "filters": scope,
            "result_snapshot": {
                "search_snapshot_id": snapshot_id,
                "query": snapshot["query"],
                "filters": scope,
                "result_count": snapshot.get("result_count", 0),
                "results": snapshot.get("results", []),
            },
            "evidence_items": evidence_items,
            "created_at": utc_now(),
        }
        bundle_id = f"evb_{_json_hash(bundle_base)[:16]}"
        bundle = dict(bundle_base)
        bundle["evidence_bundle_id"] = bundle_id
        _write_json(self.evidence_bundle_path(bundle_id), bundle)
        event = self.append_audit(
            "evidence_bundle.created",
            scope,
            {"type": "evidence_bundle", "id": bundle_id},
            {"search_snapshot_id": snapshot_id, "evidence_item_count": len(evidence_items)},
        )
        return {"bundle": bundle, "audit_event": event}

    def show_evidence_bundle(self, bundle_id: str, scope: dict[str, str]) -> dict[str, Any]:
        bundle = self.get_evidence_bundle(bundle_id)
        if bundle is None:
            return {"status": "not_found"}
        if bundle.get("filters") != scope:
            return {"status": "scope_denied", "resource_scope": bundle.get("filters")}
        event = self.append_audit(
            "evidence_bundle.read",
            scope,
            {"type": "evidence_bundle", "id": bundle_id},
            {"reason": "cli_evidence_bundle_show"},
        )
        return {"bundle": bundle, "audit_event": event}

    def view_evidence_bundle(self, bundle_id: str, scope: dict[str, str]) -> dict[str, Any]:
        bundle = self.get_evidence_bundle(bundle_id)
        if bundle is None:
            return {"status": "not_found"}
        if bundle.get("filters") != scope:
            return {"status": "scope_denied", "resource_scope": bundle.get("filters")}

        viewer_items = []
        for item in bundle.get("evidence_items", []):
            artifact = self.get_artifact(item["artifact_id"], scope)
            if artifact is None:
                return {"status": "not_found", "artifact_id": item["artifact_id"]}
            if artifact.get("scope") != scope:
                return {"status": "scope_denied", "resource_scope": artifact.get("scope")}
            derived_text = self._derived_text(artifact)
            viewer_items.append(
                {
                    "artifact_id": artifact["artifact_id"],
                    "original": {
                        "storage_ref": artifact.get("original_storage_ref"),
                        "media_type": artifact.get("media_type"),
                        "source": artifact.get("source"),
                        "raw_original_access": artifact.get("raw_original_access"),
                        "size_bytes": artifact.get("original_size_bytes"),
                    },
                    "derived": {
                        "text_ref": artifact.get("derived", {}).get("text_ref"),
                        "status": artifact.get("derived", {}).get("status"),
                        "text_preview": derived_text[:240].replace("\n", " ").strip(),
                        "metadata": artifact.get("derived"),
                    },
                    "provenance": artifact.get("provenance"),
                    "snippet": item.get("snippet"),
                }
            )

        viewer_base = {
            "schema_version": "cs.evidence_viewer.v0",
            "evidence_bundle_id": bundle_id,
            "search_snapshot_id": bundle.get("search_snapshot_id"),
            "query": bundle.get("query"),
            "filters": scope,
            "viewer_items": viewer_items,
            "opened_at": utc_now(),
        }
        viewer_id = f"viewer_{_json_hash(viewer_base)[:16]}"
        viewer = dict(viewer_base)
        viewer["evidence_viewer_id"] = viewer_id
        event = self.append_audit(
            "evidence_bundle.viewed",
            scope,
            {"type": "evidence_viewer", "id": viewer_id},
            {"evidence_bundle_id": bundle_id, "viewer_item_count": len(viewer_items)},
        )
        return {"viewer": viewer, "audit_event": event}

    def create_unsupported_claim(self, statement: str, scope: dict[str, str]) -> dict[str, Any]:
        claim_base = {
            "schema_version": "cs.claim.v0",
            "status": "draft",
            "trust_state": "draft",
            "statement": statement,
            "scope": scope,
            "evidence_bundle": {
                "evidence_bundle_id": None,
                "search_snapshot_id": None,
                "query": None,
                "filters": scope,
                "evidence_item_count": 0,
                "artifact_refs": [],
                "result_refs": [],
            },
            "authority": {
                "can_be_approved": False,
                "can_publish_shared_truth": False,
                "can_drive_autonomous_action": False,
                "blocked_reason": "Evidence Bundle is required before approval or authority use.",
            },
            "created_at": utc_now(),
        }
        claim_id = f"claim_{_json_hash(claim_base)[:16]}"
        claim = dict(claim_base)
        claim["claim_id"] = claim_id
        _write_json(self.claim_path(claim_id), claim)
        event = self.append_audit(
            "claim.draft.created",
            scope,
            {"type": "claim", "id": claim_id},
            {"evidence_bundle_id": None, "evidence_item_count": 0},
        )
        return {"claim": claim, "audit_event": event}

    def create_claim_from_evidence_bundle(self, bundle_id: str, statement: str, scope: dict[str, str]) -> dict[str, Any]:
        bundle = self.get_evidence_bundle(bundle_id)
        if bundle is None:
            return {"status": "not_found"}
        if bundle.get("filters") != scope:
            return {"status": "scope_denied", "resource_scope": bundle.get("filters")}

        claim_base = {
            "schema_version": "cs.claim.v0",
            "status": "draft",
            "trust_state": "evidence_backed",
            "statement": statement,
            "scope": scope,
            "evidence_bundle": {
                "evidence_bundle_id": bundle_id,
                "search_snapshot_id": bundle.get("search_snapshot_id"),
                "query": bundle.get("query"),
                "filters": scope,
                "evidence_item_count": len(bundle.get("evidence_items", [])),
                "artifact_refs": [f"artifact:{item['artifact_id']}" for item in bundle.get("evidence_items", [])],
                "result_refs": [
                    f"search_snapshot:{bundle.get('search_snapshot_id')}",
                    f"evidence_bundle:{bundle_id}",
                ],
            },
            "authority": {
                "can_be_approved": True,
                "can_publish_shared_truth": False,
                "can_drive_autonomous_action": False,
                "blocked_reason": "Owner approval is required before shared truth or autonomous action use.",
            },
            "created_at": utc_now(),
        }
        claim_id = f"claim_{_json_hash(claim_base)[:16]}"
        claim = dict(claim_base)
        claim["claim_id"] = claim_id
        _write_json(self.claim_path(claim_id), claim)
        event = self.append_audit(
            "claim.draft.created",
            scope,
            {"type": "claim", "id": claim_id},
            {"evidence_bundle_id": bundle_id, "search_snapshot_id": bundle.get("search_snapshot_id")},
        )
        return {"claim": claim, "audit_event": event}

    def approve_claim(self, claim_id: str, scope: dict[str, str]) -> dict[str, Any]:
        claim = self.get_claim(claim_id)
        if claim is None:
            return {"status": "not_found"}
        if claim.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": claim.get("scope")}

        evidence = claim.get("evidence_bundle", {})
        evidence_item_count = int(evidence.get("evidence_item_count", 0) or 0)
        artifact_refs = evidence.get("artifact_refs", [])
        if evidence_item_count <= 0 or not artifact_refs:
            event = self.append_audit(
                "claim.approval.denied",
                scope,
                {"type": "claim", "id": claim_id},
                {
                    "reason": "missing_evidence_bundle",
                    "required": "Attach an Evidence Bundle with at least one artifact reference before approval.",
                },
            )
            return {"status": "evidence_required", "claim": claim, "audit_event": event}

        approved = dict(claim)
        approved["status"] = "approved"
        approved["trust_state"] = "approved"
        approved["approved_at"] = utc_now()
        approved["authority"] = {
            "can_be_approved": True,
            "can_publish_shared_truth": True,
            "can_drive_autonomous_action": False,
            "blocked_reason": "Autonomous action still requires a governed action or mission path.",
        }
        _write_json(self.claim_path(claim_id), approved)
        event = self.append_audit(
            "claim.approved",
            scope,
            {"type": "claim", "id": claim_id},
            {
                "evidence_bundle_id": evidence.get("evidence_bundle_id"),
                "evidence_item_count": evidence_item_count,
                "artifact_refs": artifact_refs,
            },
        )
        return {"status": "approved", "claim": approved, "audit_event": event}

    def show_claim(self, claim_id: str, scope: dict[str, str]) -> dict[str, Any]:
        claim = self.get_claim(claim_id)
        if claim is None:
            return {"status": "not_found"}
        if claim.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": claim.get("scope")}
        event = self.append_audit(
            "claim.read",
            scope,
            {"type": "claim", "id": claim_id},
            {"reason": "cli_claim_show"},
        )
        return {"claim": claim, "audit_event": event}

    def ingest_artifact(
        self,
        input_path: Path,
        *,
        tenant_id: str,
        owner_id: str,
        namespace_id: str,
        workspace_id: str,
        source: str,
        media_type: str,
        derived_mode: str,
        trust: str,
        lineage_from: str | None,
    ) -> dict[str, Any]:
        data = input_path.read_bytes()
        checksum = sha256_bytes(data)
        artifact_id = f"art_{checksum[:16]}"
        original_storage_ref = f"sha256:{checksum}"
        original_path = self.original_dir / checksum
        self.original_dir.mkdir(parents=True, exist_ok=True)
        if not original_path.exists():
            original_path.write_bytes(data)

        scope = {
            "tenant_id": tenant_id,
            "owner_id": owner_id,
            "namespace_id": namespace_id,
            "workspace_id": workspace_id,
        }
        existing = self.get_artifact(artifact_id, scope)
        if existing and existing.get("scope") == scope:
            event = self.append_audit(
                "artifact.deduplicated",
                scope,
                {"type": "artifact", "id": artifact_id},
                {"checksum_sha256": checksum, "original_storage_ref": original_storage_ref},
            )
            return {
                "artifact": existing,
                "deduplicated": True,
                "audit_event": event,
            }

        now = utc_now()
        transformations = ["hash_calculated", "original_preserved"]
        derived: dict[str, Any]
        if derived_mode == "fail":
            derived = {
                "status": "failed",
                "reason": "fixture_derived_processor_failure",
                "message": "Original artifact is preserved and derived processing can be retried.",
            }
            transformations.append("derived_failed")
        elif derived_mode == "unsupported" or not media_type.startswith("text/"):
            derived = {
                "status": "deferred",
                "reason": "unsupported_format",
                "message": "Original artifact is preserved for future parser support.",
            }
            transformations.append("derived_deferred")
        else:
            raw_text = data.decode("utf-8", errors="replace")
            redacted_text = redact_text(raw_text)
            derived_text_ref = f"derived/{artifact_id}.txt"
            derived_path = self.artifact_dir / derived_text_ref
            derived_path.parent.mkdir(parents=True, exist_ok=True)
            derived_path.write_text(redacted_text)
            derived = {
                "status": "ready",
                "media_type": "text/plain",
                "text_ref": derived_text_ref,
                "redacted": redacted_text != raw_text,
            }
            transformations.append("derived_text_created")

        raw_text_for_safety = data.decode("utf-8", errors="replace") if media_type.startswith("text/") else ""
        blocked_attempts = detect_unsafe_instructions(raw_text_for_safety) if trust == "untrusted" else []
        safety = {
            "untrusted_evidence": trust == "untrusted",
            "unsafe_instruction_detected": bool(blocked_attempts),
            "blocked_attempt_count": len(blocked_attempts),
            "blocked_attempts": blocked_attempts,
            "tool_calls_created": 0,
            "action_cards_created_from_untrusted_artifact": 0,
            "external_http_calls": 0,
            "authority_expanded": False,
        }
        record = {
            "schema_version": "cs.artifact.v0",
            "artifact_id": artifact_id,
            "checksum_sha256": checksum,
            "content_identity": {"algorithm": "sha256", "value": checksum},
            "original_storage_ref": original_storage_ref,
            "raw_original_access": {"policy": "owner_scope_required", "display": "controlled"},
            "original_size_bytes": len(data),
            "media_type": media_type,
            "trust_state": trust,
            "safety": safety,
            "scope": scope,
            "source": {
                "type": source,
                "path": str(input_path),
                "ingested_at": now,
            },
            "provenance": {
                "created_at": now,
                "lineage_from": lineage_from,
                "transformations": transformations,
            },
            "derived": derived,
        }
        _write_json(self.artifact_path(artifact_id, scope), record)
        event = self.append_audit(
            "artifact.ingested",
            scope,
            {"type": "artifact", "id": artifact_id},
            {
                "checksum_sha256": checksum,
                "original_storage_ref": original_storage_ref,
                "derived_status": derived["status"],
                "trust_state": trust,
            },
        )
        audit_events = [event]
        policy_decisions: list[dict[str, str]] = []
        if blocked_attempts:
            unsafe_event = self.append_audit(
                "unsafe_instruction.detected",
                scope,
                {"type": "artifact", "id": artifact_id},
                {
                    "blocked_attempt_count": len(blocked_attempts),
                    "tool_calls_created": 0,
                    "action_cards_created_from_untrusted_artifact": 0,
                    "external_http_calls": 0,
                    "authority_expanded": False,
                },
            )
            audit_events.append(unsafe_event)
            policy_decisions.append(
                {
                    "id": f"policy_{unsafe_event['event_hash'][:16]}",
                    "decision": "deny",
                    "reason": "prompt_injection_blocked",
                    "scope": "artifact_ingest",
                }
            )
        return {
            "artifact": record,
            "deduplicated": False,
            "audit_event": event,
            "audit_events": audit_events,
            "policy_decisions": policy_decisions,
        }
