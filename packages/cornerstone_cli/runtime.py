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

ANSWER_STOP_TERMS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "in",
    "is",
    "of",
    "or",
    "the",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
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
        self.workspace_dir = state_dir / "workspaces"
        self.conversation_dir = state_dir / "conversations"
        self.mission_dir = state_dir / "missions"
        self.action_dir = state_dir / "actions"
        self.answer_dir = state_dir / "answers"
        self.memory_dir = state_dir / "memories"
        self.memory_conflict_dir = state_dir / "memory_conflicts"
        self.learning_dir = state_dir / "learning"
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

    def brief_path(self, brief_id: str) -> Path:
        return self.state_dir / "briefs" / f"{brief_id}.json"

    def claim_path(self, claim_id: str) -> Path:
        return self.state_dir / "claims" / f"{claim_id}.json"

    def conversation_path(self, conversation_id: str) -> Path:
        return self.conversation_dir / f"{conversation_id}.json"

    def workspace_path(self, scope: dict[str, str]) -> Path:
        return self.workspace_dir / f"{scope_key(scope)}.json"

    def mission_path(self, mission_id: str) -> Path:
        return self.mission_dir / f"{mission_id}.json"

    def action_path(self, action_id: str) -> Path:
        return self.action_dir / f"{action_id}.json"

    def answer_path(self, answer_id: str) -> Path:
        return self.answer_dir / f"{answer_id}.json"

    def memory_path(self, memory_id: str) -> Path:
        return self.memory_dir / f"{memory_id}.json"

    def memory_conflict_path(self, conflict_id: str) -> Path:
        return self.memory_conflict_dir / f"{conflict_id}.json"

    def learning_path(self, learning_id: str) -> Path:
        return self.learning_dir / f"{learning_id}.json"

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

    def get_brief(self, brief_id: str) -> dict[str, Any] | None:
        path = self.brief_path(brief_id)
        if not path.exists():
            return None
        return _read_json(path)

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        path = self.conversation_path(conversation_id)
        if not path.exists():
            return None
        return _read_json(path)

    def get_mission(self, mission_id: str) -> dict[str, Any] | None:
        path = self.mission_path(mission_id)
        if not path.exists():
            return None
        return _read_json(path)

    def get_action(self, action_id: str) -> dict[str, Any] | None:
        path = self.action_path(action_id)
        if not path.exists():
            return None
        return _read_json(path)

    def get_memory(self, memory_id: str) -> dict[str, Any] | None:
        path = self.memory_path(memory_id)
        if not path.exists():
            return None
        return _read_json(path)

    def get_memory_conflict(self, conflict_id: str) -> dict[str, Any] | None:
        path = self.memory_conflict_path(conflict_id)
        if not path.exists():
            return None
        return _read_json(path)

    def get_learning(self, learning_id: str) -> dict[str, Any] | None:
        path = self.learning_path(learning_id)
        if not path.exists():
            return None
        return _read_json(path)

    def get_workspace_mode(self, scope: dict[str, str]) -> dict[str, Any]:
        path = self.workspace_path(scope)
        if path.exists():
            return _read_json(path)
        return self._workspace_mode_record("assist", scope)

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

    def _brief_records(self, scope: dict[str, str]) -> list[dict[str, Any]]:
        brief_dir = self.state_dir / "briefs"
        if not brief_dir.exists():
            return []
        records = []
        for path in sorted(brief_dir.glob("*.json")):
            record = _read_json(path)
            if record.get("scope") == scope:
                records.append(record)
        return records

    def _conversation_records(self, scope: dict[str, str]) -> list[dict[str, Any]]:
        if not self.conversation_dir.exists():
            return []
        records = []
        for path in sorted(self.conversation_dir.glob("*.json")):
            record = _read_json(path)
            if record.get("scope") == scope:
                records.append(record)
        return records

    def _mission_records(self, scope: dict[str, str]) -> list[dict[str, Any]]:
        if not self.mission_dir.exists():
            return []
        records = []
        for path in sorted(self.mission_dir.glob("*.json")):
            record = _read_json(path)
            if record.get("scope") == scope:
                records.append(record)
        return records

    def _action_records(self, scope: dict[str, str]) -> list[dict[str, Any]]:
        if not self.action_dir.exists():
            return []
        records = []
        for path in sorted(self.action_dir.glob("*.json")):
            record = _read_json(path)
            if record.get("scope") == scope:
                records.append(record)
        return records

    def _memory_records(self, scope: dict[str, str]) -> list[dict[str, Any]]:
        if not self.memory_dir.exists():
            return []
        records = []
        for path in sorted(self.memory_dir.glob("*.json")):
            record = _read_json(path)
            if record.get("scope") == scope:
                records.append(record)
        return records

    def _learning_records(self, scope: dict[str, str]) -> list[dict[str, Any]]:
        if not self.learning_dir.exists():
            return []
        records = []
        for path in sorted(self.learning_dir.glob("*.json")):
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

    def related_missions_for_artifact(self, artifact_id: str, scope: dict[str, str]) -> list[dict[str, Any]]:
        related = []
        artifact_ref = f"artifact:{artifact_id}"
        for mission in self._mission_records(scope):
            evidence = mission.get("evidence", {})
            if artifact_ref in evidence.get("artifact_refs", []):
                related.append(
                    {
                        "mission_id": mission["mission_id"],
                        "status": mission.get("status"),
                        "goal": mission.get("goal"),
                        "evidence_bundle_id": evidence.get("evidence_bundle_id"),
                    }
                )
        return related

    def workspace_detail(self, scope: dict[str, str]) -> dict[str, Any]:
        mode = self.get_workspace_mode(scope)
        return {
            "schema_version": "cs.workspace_detail.v0",
            "active_scope": scope,
            "active_workspace_label": (
                f"{scope['namespace_id']} / {scope['workspace_id']} "
                f"({scope['owner_id']} in {scope['tenant_id']})"
            ),
            "workspace_mode": mode,
            "context_boundary": {
                "default_context": "active_workspace_only",
                "implicit_cross_namespace_context": False,
                "implicit_cross_owner_context": False,
                "promotion_required_for_cross_namespace_use": True,
                "allowed_inherited_context_refs": [],
            },
            "visible_navigation": [
                {"id": "home", "label": "Home"},
                {"id": "search", "label": "Search"},
                {"id": "artifacts", "label": "Artifacts"},
                {"id": "claims", "label": "Claims"},
                {"id": "actions", "label": "Actions"},
            ],
            "explanation": (
                "CornerStone is using only the active tenant, owner, namespace, "
                "and workspace unless a governed promotion or reference is recorded."
            ),
            "shown_at": utc_now(),
        }

    def autopilot_readiness(self, scope: dict[str, str]) -> dict[str, Any]:
        briefs = self._brief_records(scope)
        conversations = self._conversation_records(scope)
        missions = self._mission_records(scope)
        actions = self._action_records(scope)
        evidence_backed_brief_count = sum(1 for brief in briefs if brief.get("status") == "evidence_backed")
        optional_suggestion_count = sum(
            1
            for conversation in conversations
            for suggestion in conversation.get("suggested_outputs", [])
            if isinstance(suggestion, dict) and suggestion.get("mode") == "optional_promotion" and suggestion.get("forced") is False
        )
        active_or_draft_mission_count = sum(1 for mission in missions if mission.get("status") in {"draft", "active"})
        successful_internal_action_count = sum(
            1
            for action in actions
            if action.get("execution", {}).get("status") == "executed"
            and action.get("execution", {}).get("result", {}).get("side_effect_boundary") == "local_internal_state"
        )
        successful_playbook_count = successful_internal_action_count
        ready = (
            evidence_backed_brief_count >= 1
            and optional_suggestion_count >= 1
            and active_or_draft_mission_count >= 1
            and successful_internal_action_count >= 1
            and successful_playbook_count >= 1
        )
        current_mode = self.get_workspace_mode(scope)
        return {
            "schema_version": "cs.autopilot_readiness.v0",
            "scope": scope,
            "current_workspace_mode": current_mode["mode"],
            "starts_conservatively": True,
            "progression": [
                "assist_by_default",
                "evidence_backed_briefs",
                "optional_structure_suggestions",
                "governed_internal_tasks",
                "successful_playbook_history",
                "autopilot_recommendation_with_mission_contract",
            ],
            "signals": {
                "evidence_backed_brief_count": evidence_backed_brief_count,
                "optional_suggestion_count": optional_suggestion_count,
                "mission_contract_count": active_or_draft_mission_count,
                "successful_internal_task_count": successful_internal_action_count,
                "successful_playbook_count": successful_playbook_count,
            },
            "recommendation": "recommend_autopilot" if ready else "stay_assist",
            "recommended_mode": "autopilot" if ready else "assist",
            "ready": ready,
            "reason": (
                "Fixture history includes an evidence-backed brief, optional durable-output suggestions, "
                "a Mission Goal Contract, and a successful low-risk internal playbook/action run."
                if ready
                else "More evidence-backed briefs, suggestions, mission contracts, and successful internal runs are required before recommending Autopilot."
            ),
            "mission_contract_required": True,
            "approval_boundary": "Recommendation does not grant authority; a Mission Goal Contract and policy still control execution.",
            "generated_at": utc_now(),
        }

    def create_memory_from_evidence_bundle(self, bundle_id: str, statement: str, scope: dict[str, str]) -> dict[str, Any]:
        bundle = self.get_evidence_bundle(bundle_id)
        if bundle is None:
            return {"status": "not_found", "resource": "evidence_bundle"}
        if bundle.get("filters") != scope:
            return {"status": "scope_denied", "resource_scope": bundle.get("filters")}

        evidence_items = bundle.get("evidence_items", [])
        artifact_refs = [f"artifact:{item['artifact_id']}" for item in evidence_items]
        if not evidence_items or not artifact_refs:
            return {"status": "evidence_required"}

        memory_base = {
            "schema_version": "cs.memory.v0",
            "status": "owner_approved",
            "trust_state": "evidence_backed",
            "memory_type": "durable_fact",
            "statement": redact_text(statement),
            "scope": scope,
            "source": {
                "created_from": "memory.create",
                "source_type": "evidence_bundle",
                "evidence_bundle_id": bundle_id,
                "search_snapshot_id": bundle.get("search_snapshot_id"),
                "artifact_refs": artifact_refs,
            },
            "provenance": {
                "source_evidence_bundle_id": bundle_id,
                "source_search_snapshot_id": bundle.get("search_snapshot_id"),
                "source_artifact_refs": artifact_refs,
                "created_from": "owner_approved_memory_create",
            },
            "canonicality": {
                "canonical_truth_foundation": "archive_evidence",
                "raw_agent_memory_canonical": False,
                "owner_approved": True,
                "requires_evidence_for_truth_claims": True,
            },
            "evidence_refs": [
                f"evidence_bundle:{bundle_id}",
                f"search_snapshot:{bundle.get('search_snapshot_id')}",
                *artifact_refs,
            ],
            "created_at": utc_now(),
        }
        memory_id = f"memory_{_json_hash(memory_base)[:16]}"
        memory = dict(memory_base)
        memory["memory_id"] = memory_id
        _write_json(self.memory_path(memory_id), memory)
        event = self.append_audit(
            "memory.owner_approved.created",
            scope,
            {"type": "memory", "id": memory_id},
            {
                "evidence_bundle_id": bundle_id,
                "artifact_refs": artifact_refs,
                "canonical_truth_foundation": memory["canonicality"]["canonical_truth_foundation"],
            },
        )
        return {"memory": memory, "audit_event": event}

    def create_raw_agent_memory(self, statement: str, scope: dict[str, str]) -> dict[str, Any]:
        memory_base = {
            "schema_version": "cs.memory.v0",
            "status": "raw_agent_memory",
            "trust_state": "unverified",
            "memory_type": "agent_recall_candidate",
            "statement": redact_text(statement),
            "scope": scope,
            "source": {
                "created_from": "agent.raw_memory",
                "source_type": "agent_inference",
                "evidence_bundle_id": None,
                "search_snapshot_id": None,
                "artifact_refs": [],
            },
            "provenance": {
                "created_from": "agent_memory_candidate",
                "verified_against_archive": False,
                "source_artifact_refs": [],
            },
            "canonicality": {
                "canonical_truth_foundation": "none",
                "raw_agent_memory_canonical": False,
                "owner_approved": False,
                "requires_evidence_for_truth_claims": True,
            },
            "evidence_refs": [],
            "created_at": utc_now(),
        }
        memory_id = f"memory_{_json_hash(memory_base)[:16]}"
        memory = dict(memory_base)
        memory["memory_id"] = memory_id
        _write_json(self.memory_path(memory_id), memory)
        event = self.append_audit(
            "memory.raw_agent.created",
            scope,
            {"type": "memory", "id": memory_id},
            {
                "raw_agent_memory_canonical": False,
                "owner_approved": False,
                "evidence_ref_count": 0,
            },
        )
        return {"memory": memory, "audit_event": event}

    def resolve_memory_conflict(
        self,
        raw_memory_id: str,
        evidence_bundle_id: str,
        question: str,
        scope: dict[str, str],
    ) -> dict[str, Any]:
        raw_memory = self.get_memory(raw_memory_id)
        if raw_memory is None:
            return {"status": "not_found", "resource": "raw_memory"}
        if raw_memory.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": raw_memory.get("scope")}
        bundle = self.get_evidence_bundle(evidence_bundle_id)
        if bundle is None:
            return {"status": "not_found", "resource": "evidence_bundle"}
        if bundle.get("filters") != scope:
            return {"status": "scope_denied", "resource_scope": bundle.get("filters")}
        evidence_items = bundle.get("evidence_items", [])
        artifact_refs = [f"artifact:{item['artifact_id']}" for item in evidence_items]
        if not evidence_items or not artifact_refs:
            return {"status": "evidence_required"}

        owner_memory_refs = [
            f"memory:{memory['memory_id']}"
            for memory in self._memory_records(scope)
            if memory.get("status") == "owner_approved"
            and memory.get("source", {}).get("evidence_bundle_id") == evidence_bundle_id
        ]
        evidence_refs = [
            f"evidence_bundle:{evidence_bundle_id}",
            f"search_snapshot:{bundle.get('search_snapshot_id')}",
            *artifact_refs,
            *owner_memory_refs,
        ]
        conflict_base = {
            "schema_version": "cs.memory_conflict_resolution.v0",
            "status": "resolved",
            "scope": scope,
            "question": redact_text(question),
            "raw_memory": {
                "memory_id": raw_memory_id,
                "status": raw_memory.get("status"),
                "trust_state": raw_memory.get("trust_state"),
                "statement": raw_memory.get("statement"),
                "raw_agent_memory_canonical": raw_memory.get("canonicality", {}).get("raw_agent_memory_canonical"),
                "owner_approved": raw_memory.get("canonicality", {}).get("owner_approved"),
            },
            "evidence_bundle": {
                "evidence_bundle_id": evidence_bundle_id,
                "search_snapshot_id": bundle.get("search_snapshot_id"),
                "evidence_item_count": len(evidence_items),
                "artifact_refs": artifact_refs,
                "snippets": [str(item.get("snippet", "")).strip() for item in evidence_items if str(item.get("snippet", "")).strip()],
            },
            "owner_approved_memory_refs": owner_memory_refs,
            "decision": {
                "selected_truth_foundation": "archive_evidence",
                "raw_agent_memory_used_as_truth": False,
                "owner_approved_memory_requires_evidence": True,
                "answer_label": "evidence_backed",
                "reason": "Archive evidence and owner-approved evidence-backed memory outrank raw agent recall.",
            },
            "answer": {
                "presented_as_fact": True,
                "based_on": "archive_evidence",
                "evidence_refs": evidence_refs,
                "raw_memory_ref": f"memory:{raw_memory_id}",
            },
            "created_at": utc_now(),
        }
        conflict_id = f"memconf_{_json_hash(conflict_base)[:16]}"
        conflict = dict(conflict_base)
        conflict["conflict_id"] = conflict_id
        _write_json(self.memory_conflict_path(conflict_id), conflict)
        event = self.append_audit(
            "memory.conflict.resolved",
            scope,
            {"type": "memory_conflict_resolution", "id": conflict_id},
            {
                "raw_memory_id": raw_memory_id,
                "evidence_bundle_id": evidence_bundle_id,
                "selected_truth_foundation": conflict["decision"]["selected_truth_foundation"],
                "raw_agent_memory_used_as_truth": False,
            },
        )
        return {"conflict": conflict, "audit_event": event}

    def show_memory(self, memory_id: str, scope: dict[str, str]) -> dict[str, Any]:
        memory = self.get_memory(memory_id)
        if memory is None:
            return {"status": "not_found"}
        if memory.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": memory.get("scope")}
        event = self.append_audit(
            "memory.read",
            scope,
            {"type": "memory", "id": memory_id},
            {"reason": "cli_memory_show"},
        )
        return {"memory": memory, "audit_event": event}

    def record_learning_from_action(self, action_id: str, lesson: str, scope: dict[str, str]) -> dict[str, Any]:
        action = self.get_action(action_id)
        if action is None:
            return {"status": "not_found", "resource": "action"}
        if action.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": action.get("scope")}
        execution = action.get("execution", {})
        result = execution.get("result") or {}
        if execution.get("status") != "executed" or result.get("status") != "success":
            return {"status": "evidence_required", "resource": "executed_action"}

        evidence = action.get("evidence", {})
        artifact_refs = evidence.get("artifact_refs", [])
        evidence_bundle_id = evidence.get("evidence_bundle_id")
        learning_base = {
            "schema_version": "cs.learning_record.v0",
            "status": "recorded",
            "scope": scope,
            "lesson": redact_text(lesson),
            "source_action": {
                "action_id": action_id,
                "mission_id": action.get("mission_id"),
                "action_kind": action.get("action_kind"),
                "risk": action.get("risk"),
                "execution_status": execution.get("status"),
                "result_status": result.get("status"),
                "side_effect_boundary": result.get("side_effect_boundary"),
                "external_http_calls": result.get("external_http_calls"),
            },
            "source_policy": action.get("policy_decision"),
            "source_dry_run": action.get("dry_run"),
            "evidence_refs": [
                f"action:{action_id}",
                f"mission:{action.get('mission_id')}",
                f"claim:{action.get('source_claim_id')}",
                *([f"evidence_bundle:{evidence_bundle_id}"] if evidence_bundle_id else []),
                *artifact_refs,
            ],
            "learning_boundary": {
                "updates_product_behavior": False,
                "changes_user_or_org_truth": False,
                "requires_review_before_memory_update": True,
                "source_of_truth": "audit_and_evidence",
            },
            "created_at": utc_now(),
        }
        learning_id = f"learn_{_json_hash(learning_base)[:16]}"
        learning = dict(learning_base)
        learning["learning_id"] = learning_id
        _write_json(self.learning_path(learning_id), learning)
        event = self.append_audit(
            "learning.recorded",
            scope,
            {"type": "learning_record", "id": learning_id},
            {
                "action_id": action_id,
                "mission_id": action.get("mission_id"),
                "changes_user_or_org_truth": False,
            },
        )
        return {"learning": learning, "audit_event": event}

    def show_learning(self, learning_id: str, scope: dict[str, str]) -> dict[str, Any]:
        learning = self.get_learning(learning_id)
        if learning is None:
            return {"status": "not_found"}
        if learning.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": learning.get("scope")}
        event = self.append_audit(
            "learning.read",
            scope,
            {"type": "learning_record", "id": learning_id},
            {"reason": "cli_learning_show"},
        )
        return {"learning": learning, "audit_event": event}

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

    def create_brief_from_evidence_bundle(self, bundle_id: str, scope: dict[str, str]) -> dict[str, Any]:
        bundle = self.get_evidence_bundle(bundle_id)
        if bundle is None:
            return {"status": "not_found"}
        if bundle.get("filters") != scope:
            return {"status": "scope_denied", "resource_scope": bundle.get("filters")}

        evidence_items = bundle.get("evidence_items", [])
        if not evidence_items:
            return {"status": "evidence_required"}
        key_points = []
        evidence_links = []
        for item in evidence_items[:5]:
            snippet = str(item.get("snippet", "")).strip()
            if snippet:
                key_points.append(snippet)
            evidence_links.append(
                {
                    "artifact_ref": f"artifact:{item.get('artifact_id')}",
                    "evidence_bundle_ref": f"evidence_bundle:{bundle_id}",
                    "search_snapshot_ref": f"search_snapshot:{bundle.get('search_snapshot_id')}",
                    "snippet": snippet,
                    "original_storage_ref": item.get("original_storage_ref"),
                    "derived_text_ref": item.get("derived_text_ref"),
                }
            )

        brief_base = {
            "schema_version": "cs.brief.v0",
            "status": "evidence_backed",
            "scope": scope,
            "title": f"Brief for {bundle.get('query')}",
            "evidence_bundle": {
                "evidence_bundle_id": bundle_id,
                "search_snapshot_id": bundle.get("search_snapshot_id"),
                "query": bundle.get("query"),
                "filters": scope,
                "evidence_item_count": len(evidence_items),
                "artifact_refs": [link["artifact_ref"] for link in evidence_links],
            },
            "key_points": key_points,
            "evidence_links": evidence_links,
            "uncertainty": [
                "This brief is grounded only in the attached Evidence Bundle.",
                "Add more sources before using it as broad organizational truth.",
            ],
            "contradictions": [],
            "recommended_next_steps": [
                "Create a draft claim from this Evidence Bundle if the user wants durable truth work.",
                "Collect more evidence if the decision requires higher confidence.",
            ],
            "suggested_outputs": [
                {"type": "Claim", "mode": "optional_promotion"},
                {"type": "Knowledge Capsule", "mode": "optional_promotion"},
                {"type": "Mission Card", "mode": "optional_promotion"},
            ],
            "ontology": {
                "preconfigured_ontology_required": False,
                "ontology_suggestions_required_before_brief": False,
                "suggestions": [],
            },
            "created_at": utc_now(),
        }
        brief_id = f"brief_{_json_hash(brief_base)[:16]}"
        brief = dict(brief_base)
        brief["brief_id"] = brief_id
        _write_json(self.brief_path(brief_id), brief)
        event = self.append_audit(
            "brief.created",
            scope,
            {"type": "brief", "id": brief_id},
            {
                "evidence_bundle_id": bundle_id,
                "search_snapshot_id": bundle.get("search_snapshot_id"),
                "evidence_item_count": len(evidence_items),
            },
        )
        return {"brief": brief, "audit_event": event}

    def show_brief(self, brief_id: str, scope: dict[str, str]) -> dict[str, Any]:
        brief = self.get_brief(brief_id)
        if brief is None:
            return {"status": "not_found"}
        if brief.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": brief.get("scope")}
        event = self.append_audit(
            "brief.read",
            scope,
            {"type": "brief", "id": brief_id},
            {"reason": "cli_brief_show"},
        )
        return {"brief": brief, "audit_event": event}

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

    def deny_egress_attempt(self, target_url: str, scope: dict[str, str]) -> dict[str, Any]:
        decision_base = {
            "schema_version": "cs.policy_decision.v0",
            "decision": "deny",
            "policy": "default_egress_deny",
            "reason": "External network access is denied unless an explicit scoped policy allows it.",
            "target": {"type": "url", "value": target_url},
            "scope": scope,
            "external_http_calls": 0,
            "resolution_path": [
                "Attach the requested action to a governed workflow or mission.",
                "Request owner approval for a scoped connector capability.",
                "Retry only after policy grants egress for this target and purpose.",
            ],
            "decided_at": utc_now(),
        }
        decision = dict(decision_base)
        decision["id"] = f"policy_{_json_hash(decision_base)[:16]}"
        event = self.append_audit(
            "policy.egress.denied",
            scope,
            {"type": "policy_decision", "id": decision["id"]},
            {
                "policy": decision["policy"],
                "target_url": target_url,
                "external_http_calls": 0,
                "reason": decision["reason"],
            },
        )
        return {"policy_decision": decision, "audit_event": event}

    def deny_sandbox_access(self, capability: str, target: str, scope: dict[str, str]) -> dict[str, Any]:
        decision_base = {
            "schema_version": "cs.policy_decision.v0",
            "decision": "deny",
            "policy": "declared_sandbox_capability_required",
            "reason": "Host, shell, filesystem, and environment access require an explicit safe runtime capability.",
            "target": {"type": capability, "value": target},
            "scope": scope,
            "host_operations_executed": 0,
            "shell_commands_executed": 0,
            "filesystem_reads": 0,
            "environment_reads": 0,
            "resolution_path": [
                "Declare the minimum required capability in an Agent Pack or tool contract.",
                "Run inside a safe sandbox boundary after owner approval.",
                "Reduce scope or use an existing mediated workflow/action path.",
            ],
            "decided_at": utc_now(),
        }
        decision = dict(decision_base)
        decision["id"] = f"policy_{_json_hash(decision_base)[:16]}"
        event = self.append_audit(
            "policy.sandbox_access.denied",
            scope,
            {"type": "policy_decision", "id": decision["id"]},
            {
                "policy": decision["policy"],
                "capability": capability,
                "target": target,
                "host_operations_executed": 0,
                "reason": decision["reason"],
            },
        )
        return {"policy_decision": decision, "audit_event": event}

    def _workspace_mode_record(self, mode: str, scope: dict[str, str]) -> dict[str, Any]:
        behaviors = {
            "manual": {
                "label": "Manual",
                "action_proposals_allowed": True,
                "autonomous_execution_allowed": False,
                "approval_required_for_execution": True,
                "description": "CornerStone may propose work, but execution requires explicit owner action.",
            },
            "assist": {
                "label": "Assist",
                "action_proposals_allowed": True,
                "autonomous_execution_allowed": False,
                "approval_required_for_execution": True,
                "description": "CornerStone can draft briefs, claims, mission contracts, and action cards without autonomous execution.",
            },
            "autopilot": {
                "label": "Autopilot",
                "action_proposals_allowed": True,
                "autonomous_execution_allowed": True,
                "approval_required_for_execution": False,
                "description": "CornerStone may execute allowed low-risk mission actions inside the active scope and policy boundary.",
            },
            "locked": {
                "label": "Locked",
                "action_proposals_allowed": False,
                "autonomous_execution_allowed": False,
                "approval_required_for_execution": True,
                "description": "CornerStone records evidence and audit only; action proposal and execution are blocked.",
            },
        }
        return {
            "schema_version": "cs.workspace_mode.v0",
            "mode": mode,
            "scope": scope,
            "behaviors": behaviors[mode],
            "available_modes": [
                {
                    "mode": key,
                    "label": value["label"],
                    "autonomous_execution_allowed": value["autonomous_execution_allowed"],
                    "approval_required_for_execution": value["approval_required_for_execution"],
                }
                for key, value in behaviors.items()
            ],
            "updated_at": utc_now(),
        }

    def set_workspace_mode(self, mode: str, scope: dict[str, str]) -> dict[str, Any]:
        record = self._workspace_mode_record(mode, scope)
        _write_json(self.workspace_path(scope), record)
        event = self.append_audit(
            "workspace.mode.set",
            scope,
            {"type": "workspace", "id": scope_key(scope)},
            {
                "mode": mode,
                "autonomous_execution_allowed": record["behaviors"]["autonomous_execution_allowed"],
            },
        )
        return {"workspace_mode": record, "audit_event": event}

    def create_mission_contract(
        self,
        goal: str,
        scope: dict[str, str],
        *,
        claim_id: str | None = None,
        evidence_bundle_id: str | None = None,
    ) -> dict[str, Any]:
        claim = self.get_claim(claim_id) if claim_id else None
        if claim_id and claim is None:
            return {"status": "not_found", "resource": "claim"}
        if claim and claim.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": claim.get("scope")}

        bundle_id = evidence_bundle_id
        if claim and not bundle_id:
            bundle_id = claim.get("evidence_bundle", {}).get("evidence_bundle_id")
        bundle = self.get_evidence_bundle(bundle_id) if bundle_id else None
        if bundle_id and bundle is None:
            return {"status": "not_found", "resource": "evidence_bundle"}
        if bundle and bundle.get("filters") != scope:
            return {"status": "scope_denied", "resource_scope": bundle.get("filters")}

        evidence_artifact_refs = []
        if claim:
            evidence_artifact_refs.extend(claim.get("evidence_bundle", {}).get("artifact_refs", []))
        if bundle:
            evidence_artifact_refs.extend(f"artifact:{item['artifact_id']}" for item in bundle.get("evidence_items", []))
        evidence_artifact_refs = sorted(set(evidence_artifact_refs))
        if not evidence_artifact_refs:
            return {"status": "evidence_required"}

        mode = self.get_workspace_mode(scope)
        contract_base = {
            "schema_version": "cs.mission_goal_contract.v0",
            "status": "draft",
            "goal": goal,
            "scope": scope,
            "workspace_mode": mode["mode"],
            "source_claim": {
                "claim_id": claim.get("claim_id") if claim else None,
                "statement": claim.get("statement") if claim else None,
                "trust_state": claim.get("trust_state") if claim else None,
            },
            "evidence": {
                "evidence_bundle_id": bundle_id,
                "artifact_refs": evidence_artifact_refs,
                "search_snapshot_id": bundle.get("search_snapshot_id") if bundle else None,
            },
            "risk_state": "controlled_policy_required",
            "allowed_actions": [
                "internal_status_update",
                "draft_task",
                "refresh_brief",
            ],
            "forbidden_actions": [
                "external_writeback_without_workflow_action",
                "cross_namespace_access",
                "destructive_change",
                "secret_exfiltration",
            ],
            "success_criteria": [
                "Use attached evidence refs for every durable claim or action.",
                "Record every action proposal, dry-run, policy decision, approval, result, and audit event.",
            ],
            "stop_conditions": [
                "Policy denies scope, egress, connector capability, data sensitivity, or workspace mode.",
                "A required approval is missing.",
                "A requested action is outside the mission contract.",
            ],
            "review_cadence": "owner review required before high-risk execution; after-action audit after every run",
            "escalation_rules": [
                "Escalate high-risk, destructive, sensitive, cross-namespace, or external writeback actions.",
                "Escalate when evidence coverage is insufficient or policy decision is deny.",
            ],
            "evidence_expectations": [
                "Evidence Bundle with at least one artifact reference.",
                "Dry-run diff and expected impact before side-effecting action.",
                "Audit refs for every policy and action transition.",
            ],
            "authority": {
                "may_do": ["propose action cards", "run allowed low-risk internal actions in Autopilot mode"],
                "may_not_do": ["direct external writeback", "cross-namespace action", "destructive action without approval"],
                "requires_escalation": ["high-risk", "external_writeback", "sensitive", "out_of_contract"],
                "controls": ["pause", "stop", "revoke", "reduce_scope"],
            },
            "created_at": utc_now(),
        }
        mission_id = f"mission_{_json_hash(contract_base)[:16]}"
        contract = dict(contract_base)
        contract["mission_id"] = mission_id
        _write_json(self.mission_path(mission_id), contract)
        event = self.append_audit(
            "mission.contract.created",
            scope,
            {"type": "mission", "id": mission_id},
            {
                "claim_id": claim_id,
                "evidence_bundle_id": bundle_id,
                "workspace_mode": mode["mode"],
                "allowed_actions": contract["allowed_actions"],
            },
        )
        return {"mission": contract, "audit_event": event}

    def activate_mission(self, mission_id: str, scope: dict[str, str], mode: str = "autopilot") -> dict[str, Any]:
        mission = self.get_mission(mission_id)
        if mission is None:
            return {"status": "not_found"}
        if mission.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": mission.get("scope")}
        mode_result = self.set_workspace_mode(mode, scope)
        activated = dict(mission)
        activated["status"] = "active"
        activated["workspace_mode"] = mode
        activated["activated_at"] = utc_now()
        activated["authority_view"] = {
            "may_act_in_scope": scope,
            "allowed_actions": activated.get("allowed_actions", []),
            "forbidden_actions": activated.get("forbidden_actions", []),
            "requires_escalation": activated.get("authority", {}).get("requires_escalation", []),
            "pause_stop_revoke": activated.get("authority", {}).get("controls", []),
        }
        _write_json(self.mission_path(mission_id), activated)
        event = self.append_audit(
            "mission.activated",
            scope,
            {"type": "mission", "id": mission_id},
            {
                "mode": mode,
                "allowed_actions": activated.get("allowed_actions", []),
                "forbidden_actions": activated.get("forbidden_actions", []),
            },
        )
        return {"mission": activated, "workspace_mode": mode_result["workspace_mode"], "audit_events": [mode_result["audit_event"], event]}

    def _action_policy(
        self,
        mission: dict[str, Any],
        action_kind: str,
        risk: str,
        scope: dict[str, str],
        connector: str,
        direct: bool = False,
    ) -> dict[str, Any]:
        workspace_mode = self.get_workspace_mode(scope)
        mode = workspace_mode["mode"]
        allowed_actions = set(mission.get("allowed_actions", []))
        external = action_kind == "external_writeback"
        high_risk = risk in {"high", "destructive", "sensitive"} or external

        if direct:
            decision = "deny"
            policy = "workflow_action_path_required"
            reason = "Direct provider writeback is denied. Use a governed Workflow/Action path with dry-run, policy, approval, result, and audit."
            approval_required = True
            can_execute_now = False
            execution_status = "blocked_direct_write"
        elif mode == "locked":
            decision = "deny"
            policy = "workspace_mode_locked"
            reason = "Workspace mode is Locked, so action proposal and execution are disabled."
            approval_required = True
            can_execute_now = False
            execution_status = "blocked_by_workspace_mode"
        elif action_kind not in allowed_actions and not external:
            decision = "escalate"
            policy = "mission_contract_action_scope"
            reason = "Requested action is outside the Mission Goal Contract allowed actions."
            approval_required = True
            can_execute_now = False
            execution_status = "escalated_out_of_contract"
        elif high_risk:
            decision = "requires_approval"
            policy = "high_risk_action_requires_approval"
            reason = "High-risk or external writeback action requires owner approval before execution."
            approval_required = True
            can_execute_now = False
            execution_status = "pending_approval"
        elif mode != "autopilot":
            decision = "deny"
            policy = "workspace_mode_no_autonomous_execution"
            reason = "Workspace mode does not allow autonomous execution."
            approval_required = True
            can_execute_now = False
            execution_status = "blocked_by_workspace_mode"
        else:
            decision = "allow"
            policy = "low_risk_autopilot_allowed"
            reason = "Low-risk action is inside the active Mission Goal Contract and workspace Autopilot boundary."
            approval_required = False
            can_execute_now = True
            execution_status = "ready_to_execute"

        return {
            "schema_version": "cs.policy_decision.v0",
            "id": f"policy_{_json_hash({'mission_id': mission.get('mission_id'), 'action_kind': action_kind, 'risk': risk, 'mode': mode, 'direct': direct})[:16]}",
            "decision": decision,
            "policy": policy,
            "reason": reason,
            "scope": scope,
            "workspace_mode": mode,
            "mission_id": mission.get("mission_id"),
            "action_kind": action_kind,
            "risk": risk,
            "connector": connector,
            "approval_required": approval_required,
            "can_execute_now": can_execute_now,
            "execution_status": execution_status,
            "resolution_path": [
                "Use an active Mission Goal Contract.",
                "Keep action inside allowed actions and owner namespace.",
                "Run dry-run and request owner approval when policy requires it.",
            ],
            "decided_at": utc_now(),
        }

    def propose_action(
        self,
        mission_id: str,
        claim_id: str,
        action_kind: str,
        risk: str,
        scope: dict[str, str],
        *,
        goal: str,
        connector: str = "mock_connector",
        target: str = "mock://local-target",
    ) -> dict[str, Any]:
        mission = self.get_mission(mission_id)
        if mission is None:
            return {"status": "not_found", "resource": "mission"}
        if mission.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": mission.get("scope")}
        claim = self.get_claim(claim_id)
        if claim is None:
            return {"status": "not_found", "resource": "claim"}
        if claim.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": claim.get("scope")}
        if not claim.get("evidence_bundle", {}).get("artifact_refs"):
            return {"status": "evidence_required"}

        policy = self._action_policy(mission, action_kind, risk, scope, connector)
        expected_external_calls = 1 if action_kind == "external_writeback" else 0
        dry_run_base = {
            "schema_version": "cs.action_dry_run.v0",
            "mission_id": mission_id,
            "claim_id": claim_id,
            "action_kind": action_kind,
            "goal": goal,
            "scope": scope,
            "diff": {
                "before": "no side effect applied",
                "after": f"would perform {action_kind} against {target}",
            },
            "expected_impact": {
                "risk": risk,
                "target": target,
                "connector": connector,
                "external_calls": expected_external_calls,
            },
            "policy_decision": policy,
            "created_at": utc_now(),
        }
        dry_run = dict(dry_run_base)
        dry_run["dry_run_id"] = f"dryrun_{_json_hash(dry_run_base)[:16]}"
        card_base = {
            "schema_version": "cs.action_card.v0",
            "mission_id": mission_id,
            "source_claim_id": claim_id,
            "goal": goal,
            "scope": scope,
            "evidence": claim.get("evidence_bundle"),
            "risk": risk,
            "action_kind": action_kind,
            "connector_boundary": {
                "mediated_by": "ConnectorHub",
                "connector": connector,
                "direct_provider_access": False,
                "credentials_exposed_to_agent": False,
                "mocked": True,
            },
            "dry_run": dry_run,
            "policy_decision": policy,
            "approval": {
                "required": policy["approval_required"],
                "status": "not_required" if not policy["approval_required"] else "pending",
                "approver": None,
            },
            "execution": {
                "status": policy["execution_status"],
                "can_execute_now": policy["can_execute_now"],
                "result": None,
            },
            "created_at": utc_now(),
        }
        action_id = f"action_{_json_hash(card_base)[:16]}"
        card = dict(card_base)
        card["action_id"] = action_id
        _write_json(self.action_path(action_id), card)
        event = self.append_audit(
            "action.card.proposed",
            scope,
            {"type": "action", "id": action_id},
            {
                "mission_id": mission_id,
                "claim_id": claim_id,
                "action_kind": action_kind,
                "risk": risk,
                "policy": policy["policy"],
                "decision": policy["decision"],
                "dry_run_id": dry_run["dry_run_id"],
            },
        )
        card["audit_ref"] = f"audit:{event['event_id']}"
        _write_json(self.action_path(action_id), card)
        return {"action_card": card, "audit_event": event}

    def approve_action(self, action_id: str, scope: dict[str, str], approver: str = "owner") -> dict[str, Any]:
        action = self.get_action(action_id)
        if action is None:
            return {"status": "not_found"}
        if action.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": action.get("scope")}
        if not action.get("approval", {}).get("required"):
            return {"status": "approval_not_required", "action_card": action}

        approved = dict(action)
        approved["approval"] = {
            "required": True,
            "status": "approved",
            "approver": approver,
            "approved_at": utc_now(),
        }
        approved["execution"] = dict(approved["execution"])
        approved["execution"]["status"] = "ready_to_execute"
        approved["execution"]["can_execute_now"] = True
        _write_json(self.action_path(action_id), approved)
        event = self.append_audit(
            "action.approved",
            scope,
            {"type": "action", "id": action_id},
            {
                "mission_id": approved.get("mission_id"),
                "approver": approver,
                "policy": approved.get("policy_decision", {}).get("policy"),
            },
        )
        return {"action_card": approved, "audit_event": event}

    def execute_action(self, action_id: str, scope: dict[str, str]) -> dict[str, Any]:
        action = self.get_action(action_id)
        if action is None:
            return {"status": "not_found"}
        if action.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": action.get("scope")}

        mission = self.get_mission(action["mission_id"])
        if mission is None:
            return {"status": "not_found", "resource": "mission"}
        policy = self._action_policy(
            mission,
            action.get("action_kind", ""),
            action.get("risk", ""),
            scope,
            action.get("connector_boundary", {}).get("connector", "mock_connector"),
        )
        approved = action.get("approval", {}).get("status") == "approved"
        if not policy["can_execute_now"] and not approved:
            event = self.append_audit(
                "action.execution.denied",
                scope,
                {"type": "action", "id": action_id},
                {
                    "mission_id": action.get("mission_id"),
                    "policy": policy["policy"],
                    "reason": policy["reason"],
                },
            )
            return {"status": "policy_denied", "policy_decision": policy, "action_card": action, "audit_event": event}

        external = action.get("action_kind") == "external_writeback"
        result_record = {
            "schema_version": "cs.action_result.v0",
            "status": "success",
            "action_id": action_id,
            "mission_id": action.get("mission_id"),
            "side_effect_boundary": "mocked_connector" if external else "local_internal_state",
            "external_http_calls": 0,
            "mock_connector_calls": 1 if external else 0,
            "message": "Action execution was recorded through the governed Workflow/Action path.",
            "executed_at": utc_now(),
        }
        executed = dict(action)
        executed["execution"] = {
            "status": "executed",
            "can_execute_now": False,
            "result": result_record,
        }
        _write_json(self.action_path(action_id), executed)
        event = self.append_audit(
            "action.executed",
            scope,
            {"type": "action", "id": action_id},
            {
                "mission_id": executed.get("mission_id"),
                "action_kind": executed.get("action_kind"),
                "external_http_calls": 0,
                "mock_connector_calls": result_record["mock_connector_calls"],
            },
        )
        return {"status": "executed", "action_card": executed, "action_result": result_record, "audit_event": event}

    def deny_direct_connector_write(self, provider: str, target: str, scope: dict[str, str]) -> dict[str, Any]:
        synthetic_mission = {
            "mission_id": "none",
            "allowed_actions": [],
        }
        policy = self._action_policy(
            synthetic_mission,
            "external_writeback",
            "high",
            scope,
            provider,
            direct=True,
        )
        event = self.append_audit(
            "connector.direct_write.denied",
            scope,
            {"type": "connector", "id": provider},
            {
                "provider": provider,
                "target": target,
                "policy": policy["policy"],
                "direct_provider_access": False,
                "external_http_calls": 0,
            },
        )
        return {"policy_decision": policy, "audit_event": event}

    def ingest_text_artifact(
        self,
        text: str,
        scope: dict[str, str],
        *,
        source_type: str,
        source_ref: str,
        trust: str = "untrusted",
    ) -> dict[str, Any]:
        data = text.encode("utf-8")
        checksum = sha256_bytes(data)
        artifact_id = f"art_{checksum[:16]}"
        original_storage_ref = f"sha256:{checksum}"
        original_path = self.original_dir / checksum
        self.original_dir.mkdir(parents=True, exist_ok=True)
        if not original_path.exists():
            original_path.write_bytes(data)

        existing = self.get_artifact(artifact_id, scope)
        if existing and existing.get("scope") == scope:
            event = self.append_audit(
                "artifact.deduplicated",
                scope,
                {"type": "artifact", "id": artifact_id},
                {"checksum_sha256": checksum, "original_storage_ref": original_storage_ref},
            )
            return {"artifact": existing, "deduplicated": True, "audit_event": event}

        now = utc_now()
        raw_text = data.decode("utf-8", errors="replace")
        redacted_text = redact_text(raw_text)
        derived_text_ref = f"derived/{artifact_id}.txt"
        derived_path = self.artifact_dir / derived_text_ref
        derived_path.parent.mkdir(parents=True, exist_ok=True)
        derived_path.write_text(redacted_text)
        blocked_attempts = detect_unsafe_instructions(raw_text) if trust == "untrusted" else []
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
            "media_type": "text/plain",
            "trust_state": trust,
            "safety": safety,
            "scope": scope,
            "source": {
                "type": source_type,
                "ref": source_ref,
                "ingested_at": now,
            },
            "derived": {
                "status": "ready",
                "media_type": "text/plain",
                "text_ref": derived_text_ref,
                "redacted": redacted_text != raw_text,
            },
            "provenance": {
                "created_at": now,
                "lineage_from": None,
                "transformations": ["hash_calculated", "original_preserved", "derived_text_created"],
            },
        }
        _write_json(self.artifact_path(artifact_id, scope), record)
        event = self.append_audit(
            "artifact.ingested",
            scope,
            {"type": "artifact", "id": artifact_id},
            {
                "checksum_sha256": checksum,
                "original_storage_ref": original_storage_ref,
                "source_type": source_type,
                "derived_status": "ready",
                "unsafe_instruction_detected": safety["unsafe_instruction_detected"],
            },
        )
        return {"artifact": record, "deduplicated": False, "audit_event": event}

    def start_conversation(self, message: str, scope: dict[str, str]) -> dict[str, Any]:
        conversation_base = {
            "schema_version": "cs.conversation.v0",
            "scope": scope,
            "started_from": "natural_message",
            "pre_modeling_required": False,
            "required_setup": {
                "connector_setup": False,
                "model_provider_setup": False,
                "ontology_setup": False,
                "organization_policy_setup": False,
            },
            "created_at": utc_now(),
        }
        conversation_id = f"conv_{_json_hash({**conversation_base, 'message': message})[:16]}"
        artifact_result = self.ingest_text_artifact(
            message,
            scope,
            source_type="conversation_turn",
            source_ref=conversation_id,
            trust="untrusted",
        )
        artifact = artifact_result["artifact"]
        turn = {
            "turn_id": f"turn_{_json_hash({'conversation_id': conversation_id, 'role': 'user', 'message': message})[:16]}",
            "role": "user",
            "content": redact_text(message),
            "artifact_ref": f"artifact:{artifact['artifact_id']}",
            "created_at": utc_now(),
        }
        conversation = dict(conversation_base)
        conversation.update(
            {
                "conversation_id": conversation_id,
                "turns": [turn],
                "source_artifact_id": artifact["artifact_id"],
                "suggested_outputs": [
                    {"type": "Mission Card", "mode": "optional_promotion", "forced": False},
                    {"type": "Knowledge Capsule", "mode": "optional_promotion", "forced": False},
                    {"type": "Claim", "mode": "optional_promotion", "forced": False},
                    {"type": "Action Card", "mode": "optional_promotion", "forced": False},
                    {"type": "Memory", "mode": "optional_promotion", "forced": False},
                    {"type": "Playbook Candidate", "mode": "optional_promotion", "forced": False},
                ],
                "user_can_continue_without_conversion": True,
            }
        )
        _write_json(self.conversation_path(conversation_id), conversation)
        event = self.append_audit(
            "conversation.started",
            scope,
            {"type": "conversation", "id": conversation_id},
            {
                "source_artifact_id": artifact["artifact_id"],
                "suggested_output_types": [item["type"] for item in conversation["suggested_outputs"]],
            },
        )
        return {
            "conversation": conversation,
            "artifact": artifact,
            "audit_events": [artifact_result["audit_event"], event],
        }

    def promote_conversation_to_claim(
        self,
        conversation_id: str,
        statement: str,
        evidence_bundle_id: str,
        scope: dict[str, str],
    ) -> dict[str, Any]:
        conversation = self.get_conversation(conversation_id)
        if conversation is None:
            return {"status": "not_found", "resource": "conversation"}
        if conversation.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": conversation.get("scope")}

        result = self.create_claim_from_evidence_bundle(evidence_bundle_id, statement, scope)
        if result.get("status"):
            return result

        claim = dict(result["claim"])
        claim["source_conversation"] = {
            "conversation_id": conversation_id,
            "turn_refs": [f"conversation_turn:{turn['turn_id']}" for turn in conversation.get("turns", [])],
            "source_artifact_ref": f"artifact:{conversation.get('source_artifact_id')}",
        }
        claim["provenance"] = {
            "created_from": "conversation.promote",
            "source_conversation_id": conversation_id,
            "source_artifact_id": conversation.get("source_artifact_id"),
            "promoted_at": utc_now(),
        }
        _write_json(self.claim_path(claim["claim_id"]), claim)
        event = self.append_audit(
            "conversation.promoted",
            scope,
            {"type": "claim", "id": claim["claim_id"]},
            {
                "conversation_id": conversation_id,
                "evidence_bundle_id": evidence_bundle_id,
                "promoted_kind": "claim",
                "trust_state": claim.get("trust_state"),
            },
        )
        return {"claim": claim, "audit_events": [result["audit_event"], event]}

    def answer_conversation(self, conversation_id: str, question: str, scope: dict[str, str]) -> dict[str, Any]:
        conversation = self.get_conversation(conversation_id)
        if conversation is None:
            return {"status": "not_found", "resource": "conversation"}
        if conversation.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": conversation.get("scope")}

        search_result = self.search(question, **scope)
        snapshot = search_result["snapshot"]
        evidence_refs: list[str] = []
        meaningful_question_terms = {
            term
            for term in search_terms(question)
            if len(term) > 2 and term not in ANSWER_STOP_TERMS
        }
        matched_terms = {
            str(reason.get("matched_term") or reason.get("query_term") or reason.get("query")).lower()
            for result in snapshot.get("results", [])
            for reason in result.get("match_reasons", [])
            if isinstance(reason, dict)
        }
        supported_by_meaningful_match = bool(meaningful_question_terms & matched_terms)
        label = "evidence_backed" if snapshot.get("result_count", 0) > 0 and supported_by_meaningful_match else "insufficient_evidence"
        presented_as_fact = label == "evidence_backed"
        if presented_as_fact:
            evidence_refs.extend(
                ref
                for result_row in snapshot.get("results", [])
                for ref in result_row.get("evidence_refs", [])
            )
        supporting_result_count = snapshot.get("result_count", 0) if presented_as_fact else 0
        answer_base = {
            "schema_version": "cs.conversation_answer.v0",
            "conversation_id": conversation_id,
            "question": question,
            "scope": scope,
            "label": label,
            "trust_state": label,
            "presented_as_fact": presented_as_fact,
            "answer": (
                "The available evidence supports an answer; inspect the attached evidence refs."
                if presented_as_fact
                else "Insufficient evidence. Add or attach source evidence before treating this as fact."
            ),
            "search_snapshot_id": snapshot.get("search_snapshot_id"),
            "search_result_count": snapshot.get("result_count", 0),
            "supporting_result_count": supporting_result_count,
            "meaningful_question_terms": sorted(meaningful_question_terms),
            "matched_terms": sorted(matched_terms),
            "evidence_refs": evidence_refs,
            "unsupported_assertions_labeled": not presented_as_fact,
            "created_at": utc_now(),
        }
        answer = dict(answer_base)
        answer["answer_id"] = f"answer_{_json_hash(answer_base)[:16]}"
        _write_json(self.answer_path(answer["answer_id"]), answer)
        event = self.append_audit(
            "conversation.answer.created",
            scope,
            {"type": "conversation_answer", "id": answer["answer_id"]},
            {
                "conversation_id": conversation_id,
                "label": label,
                "search_snapshot_id": snapshot.get("search_snapshot_id"),
                "search_result_count": snapshot.get("result_count", 0),
            },
        )
        return {"answer": answer, "search_snapshot": snapshot, "audit_events": [search_result["audit_event"], event]}

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
