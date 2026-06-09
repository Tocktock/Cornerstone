from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cornerstone_cli.validators import redact_text


UNSAFE_INSTRUCTION_PATTERNS = [
    re.compile(r"ignore (all )?(previous|prior) instructions", re.IGNORECASE),
    re.compile(r"\b(call|invoke|use|run)\b.*\b(tool|api|http|url|webhook)\b", re.IGNORECASE),
    re.compile(r"\b(authority|permission|approval)\b.*\b(now|granted|expanded|bypass)\b", re.IGNORECASE),
]


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

    def artifact_path(self, artifact_id: str) -> Path:
        return self.record_dir / f"{artifact_id}.json"

    def get_artifact(self, artifact_id: str) -> dict[str, Any] | None:
        path = self.artifact_path(artifact_id)
        if not path.exists():
            return None
        return _read_json(path)

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

        existing = self.get_artifact(artifact_id)
        scope = {
            "tenant_id": tenant_id,
            "owner_id": owner_id,
            "namespace_id": namespace_id,
            "workspace_id": workspace_id,
        }
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
        _write_json(self.artifact_path(artifact_id), record)
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
