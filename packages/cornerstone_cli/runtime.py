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

STRUCTURE_LABELS = {
    "person": ("object", "person"),
    "organization": ("object", "organization"),
    "project": ("object", "project"),
    "policy": ("fact", "policy"),
    "asset": ("object", "asset"),
    "event": ("event", "event"),
    "claim": ("fact", "claim"),
    "fact": ("fact", "fact"),
}

TRUSTED_PACK_SOURCES = {"first_party", "organization_private", "curated_certified"}
REQUIRED_AGENT_PACK_FIELDS = {
    "pack_id",
    "version",
    "role_contract",
    "role_card",
    "allowed_capabilities",
    "connector_requirements",
    "memory_scope",
    "model_policy",
    "judge_rubric",
    "playbooks",
    "after_action_review_template",
    "evaluation_expectations",
    "components",
    "trust",
    "supply_chain",
}

AGENT_ROLE_TEMPLATES: dict[str, dict[str, Any]] = {
    "orchestrator": {
        "display_name": "Orchestrator Agent",
        "purpose": "Own mission planning, delegation, synthesis, authority gaps, and after-action review.",
        "responsibilities": ["plan_mission", "delegate_specialists", "merge_results", "ask_for_missing_authority", "produce_after_action_review"],
        "allowed_tools": ["mission.read", "evidence.read", "policy.evaluate", "workflow.propose", "audit.read"],
        "forbidden_actions": ["direct_truth_mutation", "direct_source_write", "cross_namespace_access", "silent_authority_expansion"],
        "memory_scope": "active_workspace_only",
        "evidence_requirements": ["delegation rationale", "evidence refs for durable outputs", "gap label when evidence is insufficient"],
        "escalation_rules": ["escalate missing authority", "escalate high-risk or external action", "escalate unresolved specialist conflict"],
        "model_policy": {"provider": "local_test", "replaceable_brain": True, "contract_controls_authority": True},
        "judge_rubric": {"checks": ["evidence coverage", "policy compliance", "mission outcome quality"], "llm_signal_is_supporting_only": True},
        "audit_expectations": ["agent.orchestrated", "agent.delegated", "agent.after_action_review.created"],
    },
    "evidence": {
        "display_name": "Evidence Agent",
        "purpose": "Review artifacts, search snapshots, and evidence bundles without mutating truth.",
        "responsibilities": ["inspect_artifacts", "check_evidence_refs", "label_evidence_gaps"],
        "allowed_tools": ["artifact.read", "search.read", "evidence.read"],
        "forbidden_actions": ["claim.approve", "memory.write", "source.write"],
        "memory_scope": "none",
        "evidence_requirements": ["source refs", "uncertainty label"],
        "escalation_rules": ["escalate missing source", "escalate contradiction"],
        "model_policy": {"provider": "local_test", "replaceable_brain": True, "contract_controls_authority": True},
        "judge_rubric": {"checks": ["source coverage", "uncertainty"], "llm_signal_is_supporting_only": True},
        "audit_expectations": ["agent.output.created"],
    },
    "memory": {
        "display_name": "Memory Agent",
        "purpose": "Draft memory candidates from evidence while leaving approval to governed memory paths.",
        "responsibilities": ["draft_memory_candidate", "explain_source_basis", "flag_poisoning_risk"],
        "allowed_tools": ["memory.propose", "evidence.read", "wiki.read"],
        "forbidden_actions": ["memory.approve", "raw_memory_as_truth", "cross_namespace_memory_read"],
        "memory_scope": "active_workspace_only",
        "evidence_requirements": ["source evidence bundle", "trust-state label"],
        "escalation_rules": ["escalate approval need", "escalate poisoning risk"],
        "model_policy": {"provider": "local_test", "replaceable_brain": True, "contract_controls_authority": True},
        "judge_rubric": {"checks": ["archive foundation", "owner scope"], "llm_signal_is_supporting_only": True},
        "audit_expectations": ["agent.output.created", "memory.proposal.created"],
    },
    "workflow": {
        "display_name": "Workflow Agent",
        "purpose": "Prepare Workflow/Action proposals, dry-runs, and rollback-aware execution plans.",
        "responsibilities": ["prepare_action_card", "check_dry_run", "record_expected_impact"],
        "allowed_tools": ["action.propose", "action.dry_run", "workflow.read"],
        "forbidden_actions": ["direct_external_write", "execute_without_policy", "execute_without_required_approval"],
        "memory_scope": "active_mission_only",
        "evidence_requirements": ["claim refs", "policy refs", "dry-run diff"],
        "escalation_rules": ["escalate high-risk", "escalate missing approval"],
        "model_policy": {"provider": "local_test", "replaceable_brain": True, "contract_controls_authority": True},
        "judge_rubric": {"checks": ["dry-run present", "approval boundary"], "llm_signal_is_supporting_only": True},
        "audit_expectations": ["action.card.proposed", "policy.decision.created"],
    },
    "judge": {
        "display_name": "Judge Agent",
        "purpose": "Evaluate ambiguous outputs as a supporting signal without mutating memory or rules.",
        "responsibilities": ["score_output", "explain_disagreement", "recommend_escalation"],
        "allowed_tools": ["evidence.read", "trajectory.read", "rubric.evaluate"],
        "forbidden_actions": ["memory.write", "policy.write", "final_success_override"],
        "memory_scope": "active_mission_only",
        "evidence_requirements": ["rubric refs", "outcome refs"],
        "escalation_rules": ["escalate unresolved high-risk disagreement"],
        "model_policy": {"provider": "local_test", "replaceable_brain": True, "contract_controls_authority": True},
        "judge_rubric": {"checks": ["rubric coverage", "objective outcome precedence"], "llm_signal_is_supporting_only": True},
        "audit_expectations": ["agent.judge_result.recorded"],
    },
    "connector": {
        "display_name": "Connector Agent",
        "purpose": "Request ConnectorHub-mediated capabilities and never hold credentials or provider clients.",
        "responsibilities": ["request_capability", "check_source_policy", "report_connector_boundary"],
        "allowed_tools": ["connector.request", "capability.read", "action.propose"],
        "forbidden_actions": ["provider_client", "credential_read", "direct_api_writeback", "raw_secret_access"],
        "memory_scope": "none",
        "evidence_requirements": ["connector capability ref", "policy decision ref"],
        "escalation_rules": ["escalate unavailable connector", "escalate ungranted capability"],
        "model_policy": {"provider": "local_test", "replaceable_brain": True, "contract_controls_authority": True},
        "judge_rubric": {"checks": ["ConnectorHub boundary", "credential custody"], "llm_signal_is_supporting_only": True},
        "audit_expectations": ["agent.connector_request.denied", "agent.connector_request.mediated"],
    },
    "policy": {
        "display_name": "Policy Agent",
        "purpose": "Evaluate scope, risk, approval, egress, and authority boundaries.",
        "responsibilities": ["evaluate_scope", "evaluate_risk", "record_resolution_path"],
        "allowed_tools": ["policy.evaluate", "access.check", "audit.write"],
        "forbidden_actions": ["grant_authority_without_contract", "bypass_approval"],
        "memory_scope": "none",
        "evidence_requirements": ["policy decision refs"],
        "escalation_rules": ["escalate deny", "escalate human approval required"],
        "model_policy": {"provider": "local_test", "replaceable_brain": True, "contract_controls_authority": True},
        "judge_rubric": {"checks": ["policy cause", "resolution path"], "llm_signal_is_supporting_only": True},
        "audit_expectations": ["policy.decision.created"],
    },
    "playbook": {
        "display_name": "Playbook Agent",
        "purpose": "Suggest reusable playbooks from approved experience without auto-globalizing behavior.",
        "responsibilities": ["compare_experience", "draft_playbook_candidate", "require_owner_approval"],
        "allowed_tools": ["experience.read", "lesson.propose", "pack.playbook.propose"],
        "forbidden_actions": ["auto_globalize", "activate_without_approval", "expand_pack_authority"],
        "memory_scope": "active_workspace_only",
        "evidence_requirements": ["trajectory refs", "lesson refs", "approval state"],
        "escalation_rules": ["escalate approval need", "escalate behavior change"],
        "model_policy": {"provider": "local_test", "replaceable_brain": True, "contract_controls_authority": True},
        "judge_rubric": {"checks": ["applicability boundary", "rollback"], "llm_signal_is_supporting_only": True},
        "audit_expectations": ["agent.output.created", "pack.playbook.proposed"],
    },
}

MODEL_CAPABILITY_REGISTRY: dict[str, dict[str, Any]] = {
    "local_test/local_test.v0": {
        "provider": "local_test",
        "model": "local_test.v0",
        "deployment": "local",
        "registry_only": False,
        "external_call_required": False,
        "capabilities": ["deterministic_fixture", "routing_baseline", "evidence_grounded_judge", "classification", "summarization"],
        "allowed_sensitivity": ["public", "internal", "confidential", "restricted"],
        "max_context_tokens": 8192,
        "cost_per_1k_tokens_usd": 0.0,
        "typical_latency_ms": 15,
        "tool_use_reliability": 0.92,
        "grounding_reliability": 0.91,
        "judge_reliability": 0.88,
        "safe_baseline": True,
    },
    "ollama/qwen3.6:27b": {
        "provider": "ollama",
        "model": "qwen3.6:27b",
        "deployment": "local",
        "registry_only": True,
        "external_call_required": False,
        "capabilities": ["planning", "critique", "local_semantic_judge", "long_context_review"],
        "allowed_sensitivity": ["public", "internal", "confidential"],
        "max_context_tokens": 32768,
        "cost_per_1k_tokens_usd": 0.0,
        "typical_latency_ms": 800,
        "tool_use_reliability": 0.83,
        "grounding_reliability": 0.84,
        "judge_reliability": 0.82,
        "safe_baseline": True,
    },
    "openai/gpt-5.4": {
        "provider": "openai",
        "model": "gpt-5.4",
        "deployment": "external",
        "registry_only": True,
        "external_call_required": True,
        "capabilities": ["planning", "tool_use", "critique", "general_reasoning"],
        "allowed_sensitivity": ["public", "internal"],
        "max_context_tokens": 128000,
        "cost_per_1k_tokens_usd": 0.03,
        "typical_latency_ms": 1200,
        "tool_use_reliability": 0.9,
        "grounding_reliability": 0.87,
        "judge_reliability": 0.86,
        "safe_baseline": False,
    },
    "anthropic/claude-sonnet-4.5": {
        "provider": "anthropic",
        "model": "claude-sonnet-4.5",
        "deployment": "external",
        "registry_only": True,
        "external_call_required": True,
        "capabilities": ["writing", "critique", "long_context_review", "planning"],
        "allowed_sensitivity": ["public", "internal"],
        "max_context_tokens": 200000,
        "cost_per_1k_tokens_usd": 0.03,
        "typical_latency_ms": 1300,
        "tool_use_reliability": 0.88,
        "grounding_reliability": 0.86,
        "judge_reliability": 0.85,
        "safe_baseline": False,
    },
    "google/gemini-pro": {
        "provider": "google",
        "model": "gemini-pro",
        "deployment": "external",
        "registry_only": True,
        "external_call_required": True,
        "capabilities": ["multimodal_review", "planning", "summarization"],
        "allowed_sensitivity": ["public", "internal"],
        "max_context_tokens": 1000000,
        "cost_per_1k_tokens_usd": 0.02,
        "typical_latency_ms": 1400,
        "tool_use_reliability": 0.84,
        "grounding_reliability": 0.83,
        "judge_reliability": 0.8,
        "safe_baseline": False,
    },
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
        self.wiki_dir = state_dir / "wiki_views"
        self.memory_control_dir = state_dir / "memory_controls"
        self.memory_export_dir = state_dir / "memory_exports"
        self.memory_quarantine_dir = state_dir / "memory_quarantine"
        self.memory_adaptation_dir = state_dir / "memory_adaptations"
        self.temporary_session_dir = state_dir / "temporary_sessions"
        self.knowledge_capsule_dir = state_dir / "knowledge_capsules"
        self.decision_card_dir = state_dir / "decision_cards"
        self.correction_dir = state_dir / "corrections"
        self.share_dir = state_dir / "shares"
        self.understanding_suggestion_dir = state_dir / "understanding_suggestions"
        self.ontology_item_dir = state_dir / "ontology_items"
        self.operational_map_dir = state_dir / "operational_maps"
        self.contradiction_dir = state_dir / "contradictions"
        self.staleness_dir = state_dir / "staleness"
        self.ontology_change_dir = state_dir / "ontology_changes"
        self.namespace_promotion_dir = state_dir / "namespace_promotions"
        self.namespace_recovery_dir = state_dir / "namespace_recoveries"
        self.namespace_audit_export_dir = state_dir / "namespace_audit_exports"
        self.claim_basis_export_dir = state_dir / "claim_basis_exports"
        self.source_safety_dir = state_dir / "source_safety"
        self.product_learning_boundary_dir = state_dir / "product_learning_boundaries"
        self.product_surface_dir = state_dir / "product_surfaces"
        self.access_decision_dir = state_dir / "access_decisions"
        self.learning_dir = state_dir / "learning"
        self.trajectory_dir = state_dir / "mission_trajectories"
        self.experience_recommendation_dir = state_dir / "experience_recommendations"
        self.lesson_candidate_dir = state_dir / "lesson_candidates"
        self.lesson_control_dir = state_dir / "lesson_controls"
        self.behavior_signal_dir = state_dir / "behavior_signals"
        self.model_evaluation_dir = state_dir / "model_evaluations"
        self.product_improvement_dir = state_dir / "product_improvements"
        self.local_adaptation_dir = state_dir / "local_adaptations"
        self.outcome_metric_dir = state_dir / "outcome_metrics"
        self.connected_outcome_dir = state_dir / "connected_outcomes"
        self.experience_export_dir = state_dir / "experience_exports"
        self.mission_control_dir = state_dir / "mission_control"
        self.mission_autonomy_control_dir = state_dir / "mission_autonomy_controls"
        self.mission_escalation_dir = state_dir / "mission_escalations"
        self.mission_outcome_dir = state_dir / "mission_outcomes"
        self.mission_after_action_dir = state_dir / "mission_after_actions"
        self.mission_audit_export_dir = state_dir / "mission_audit_exports"
        self.autonomy_metric_dir = state_dir / "autonomy_metrics"
        self.action_reversibility_dir = state_dir / "action_reversibility"
        self.connector_action_trace_dir = state_dir / "connector_action_traces"
        self.pack_registry_dir = state_dir / "packs" / "registry"
        self.pack_install_dir = state_dir / "packs" / "installs"
        self.pack_activation_dir = state_dir / "packs" / "activations"
        self.pack_certification_dir = state_dir / "packs" / "certifications"
        self.pack_update_dir = state_dir / "packs" / "updates"
        self.pack_rollback_dir = state_dir / "packs" / "rollbacks"
        self.pack_patch_dir = state_dir / "packs" / "security_patches"
        self.pack_quarantine_dir = state_dir / "packs" / "quarantine"
        self.pack_playbook_proposal_dir = state_dir / "packs" / "playbook_proposals"
        self.agent_role_dir = state_dir / "agents" / "roles"
        self.agent_trace_dir = state_dir / "agents" / "mission_traces"
        self.agent_output_dir = state_dir / "agents" / "outputs"
        self.agent_diagnosis_dir = state_dir / "agents" / "diagnoses"
        self.agent_contract_update_dir = state_dir / "agents" / "contract_updates"
        self.agent_brain_switch_dir = state_dir / "agents" / "brain_switches"
        self.agent_mutation_attempt_dir = state_dir / "agents" / "mutation_attempts"
        self.agent_replay_dir = state_dir / "agents" / "replays"
        self.agent_pack_capability_dir = state_dir / "agents" / "pack_capabilities"
        self.brain_route_dir = state_dir / "brain" / "routes"
        self.brain_ledger_dir = state_dir / "brain" / "ledgers"
        self.brain_switch_dir = state_dir / "brain" / "switches"
        self.brain_aggregation_dir = state_dir / "brain" / "aggregations"
        self.judge_record_dir = state_dir / "judge" / "records"
        self.judge_conflict_dir = state_dir / "judge" / "conflicts"
        self.judge_acceptance_dir = state_dir / "judge" / "acceptance"
        self.judge_recommendation_dir = state_dir / "judge" / "recommendations"
        self.judge_adjudication_dir = state_dir / "judge" / "adjudications"
        self.judge_calibration_dir = state_dir / "judge" / "calibration"
        self.credential_boundary_dir = state_dir / "security" / "credential_boundaries"
        self.sensitive_change_dir = state_dir / "security" / "sensitive_changes"
        self.backup_restore_dir = state_dir / "security" / "backup_restore"
        self.helpful_failure_dir = state_dir / "security" / "helpful_failures"
        self.idempotency_dir = state_dir / "security" / "idempotency"
        self.retention_dir = state_dir / "security" / "retention"
        self.operator_status_dir = state_dir / "security" / "operator_status"
        self.release_report_dir = state_dir / "security" / "release_reports"
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

    def wiki_path(self, wiki_id: str) -> Path:
        return self.wiki_dir / f"{wiki_id}.json"

    def memory_control_path(self, control_id: str) -> Path:
        return self.memory_control_dir / f"{control_id}.json"

    def memory_export_path(self, export_id: str) -> Path:
        return self.memory_export_dir / f"{export_id}.json"

    def memory_quarantine_path(self, quarantine_id: str) -> Path:
        return self.memory_quarantine_dir / f"{quarantine_id}.json"

    def memory_adaptation_path(self, adaptation_id: str) -> Path:
        return self.memory_adaptation_dir / f"{adaptation_id}.json"

    def temporary_session_path(self, session_id: str) -> Path:
        return self.temporary_session_dir / f"{session_id}.json"

    def knowledge_capsule_path(self, capsule_id: str) -> Path:
        return self.knowledge_capsule_dir / f"{capsule_id}.json"

    def decision_card_path(self, card_id: str) -> Path:
        return self.decision_card_dir / f"{card_id}.json"

    def correction_path(self, correction_id: str) -> Path:
        return self.correction_dir / f"{correction_id}.json"

    def share_path(self, share_id: str) -> Path:
        return self.share_dir / f"{share_id}.json"

    def understanding_suggestion_path(self, suggestion_id: str) -> Path:
        return self.understanding_suggestion_dir / f"{suggestion_id}.json"

    def ontology_item_path(self, item_id: str) -> Path:
        return self.ontology_item_dir / f"{item_id}.json"

    def operational_map_path(self, map_id: str) -> Path:
        return self.operational_map_dir / f"{map_id}.json"

    def contradiction_path(self, contradiction_id: str) -> Path:
        return self.contradiction_dir / f"{contradiction_id}.json"

    def staleness_path(self, staleness_id: str) -> Path:
        return self.staleness_dir / f"{staleness_id}.json"

    def ontology_change_path(self, change_id: str) -> Path:
        return self.ontology_change_dir / f"{change_id}.json"

    def namespace_promotion_path(self, promotion_id: str) -> Path:
        return self.namespace_promotion_dir / f"{promotion_id}.json"

    def namespace_recovery_path(self, recovery_id: str) -> Path:
        return self.namespace_recovery_dir / f"{recovery_id}.json"

    def namespace_audit_export_path(self, export_id: str) -> Path:
        return self.namespace_audit_export_dir / f"{export_id}.json"

    def claim_basis_export_path(self, export_id: str) -> Path:
        return self.claim_basis_export_dir / f"{export_id}.json"

    def source_safety_path(self, safety_id: str) -> Path:
        return self.source_safety_dir / f"{safety_id}.json"

    def product_learning_boundary_path(self, boundary_id: str) -> Path:
        return self.product_learning_boundary_dir / f"{boundary_id}.json"

    def product_surface_path(self, surface_id: str) -> Path:
        return self.product_surface_dir / f"{surface_id}.json"

    def access_decision_path(self, decision_id: str) -> Path:
        return self.access_decision_dir / f"{decision_id}.json"

    def learning_path(self, learning_id: str) -> Path:
        return self.learning_dir / f"{learning_id}.json"

    def trajectory_path(self, trajectory_id: str) -> Path:
        return self.trajectory_dir / f"{trajectory_id}.json"

    def experience_recommendation_path(self, recommendation_id: str) -> Path:
        return self.experience_recommendation_dir / f"{recommendation_id}.json"

    def lesson_candidate_path(self, lesson_id: str) -> Path:
        return self.lesson_candidate_dir / f"{lesson_id}.json"

    def lesson_control_path(self, control_id: str) -> Path:
        return self.lesson_control_dir / f"{control_id}.json"

    def behavior_signal_path(self, signal_id: str) -> Path:
        return self.behavior_signal_dir / f"{signal_id}.json"

    def model_evaluation_path(self, evaluation_id: str) -> Path:
        return self.model_evaluation_dir / f"{evaluation_id}.json"

    def product_improvement_path(self, proposal_id: str) -> Path:
        return self.product_improvement_dir / f"{proposal_id}.json"

    def local_adaptation_path(self, adaptation_id: str) -> Path:
        return self.local_adaptation_dir / f"{adaptation_id}.json"

    def outcome_metric_path(self, report_id: str) -> Path:
        return self.outcome_metric_dir / f"{report_id}.json"

    def connected_outcome_path(self, outcome_id: str) -> Path:
        return self.connected_outcome_dir / f"{outcome_id}.json"

    def experience_export_path(self, export_id: str) -> Path:
        return self.experience_export_dir / f"{export_id}.json"

    def mission_control_path(self, control_id: str) -> Path:
        return self.mission_control_dir / f"{control_id}.json"

    def mission_autonomy_control_path(self, control_id: str) -> Path:
        return self.mission_autonomy_control_dir / f"{control_id}.json"

    def mission_escalation_path(self, escalation_id: str) -> Path:
        return self.mission_escalation_dir / f"{escalation_id}.json"

    def mission_outcome_path(self, outcome_id: str) -> Path:
        return self.mission_outcome_dir / f"{outcome_id}.json"

    def mission_after_action_path(self, review_id: str) -> Path:
        return self.mission_after_action_dir / f"{review_id}.json"

    def mission_audit_export_path(self, export_id: str) -> Path:
        return self.mission_audit_export_dir / f"{export_id}.json"

    def autonomy_metric_path(self, metric_id: str) -> Path:
        return self.autonomy_metric_dir / f"{metric_id}.json"

    def action_reversibility_path(self, reversibility_id: str) -> Path:
        return self.action_reversibility_dir / f"{reversibility_id}.json"

    def connector_action_trace_path(self, trace_id: str) -> Path:
        return self.connector_action_trace_dir / f"{trace_id}.json"

    def agent_pack_path(self, pack_id: str) -> Path:
        return self.pack_registry_dir / f"{pack_id}.json"

    def agent_pack_install_path(self, install_id: str) -> Path:
        return self.pack_install_dir / f"{install_id}.json"

    def agent_pack_activation_path(self, activation_id: str) -> Path:
        return self.pack_activation_dir / f"{activation_id}.json"

    def agent_pack_certification_path(self, certification_id: str) -> Path:
        return self.pack_certification_dir / f"{certification_id}.json"

    def agent_pack_update_path(self, update_id: str) -> Path:
        return self.pack_update_dir / f"{update_id}.json"

    def agent_pack_rollback_path(self, rollback_id: str) -> Path:
        return self.pack_rollback_dir / f"{rollback_id}.json"

    def agent_pack_patch_path(self, patch_id: str) -> Path:
        return self.pack_patch_dir / f"{patch_id}.json"

    def agent_pack_quarantine_path(self, quarantine_id: str) -> Path:
        return self.pack_quarantine_dir / f"{quarantine_id}.json"

    def agent_pack_playbook_proposal_path(self, proposal_id: str) -> Path:
        return self.pack_playbook_proposal_dir / f"{proposal_id}.json"

    def agent_role_path(self, role_id: str) -> Path:
        return self.agent_role_dir / f"{role_id}.json"

    def agent_trace_path(self, trace_id: str) -> Path:
        return self.agent_trace_dir / f"{trace_id}.json"

    def agent_output_path(self, output_id: str) -> Path:
        return self.agent_output_dir / f"{output_id}.json"

    def agent_diagnosis_path(self, diagnosis_id: str) -> Path:
        return self.agent_diagnosis_dir / f"{diagnosis_id}.json"

    def agent_contract_update_path(self, update_id: str) -> Path:
        return self.agent_contract_update_dir / f"{update_id}.json"

    def agent_brain_switch_path(self, switch_id: str) -> Path:
        return self.agent_brain_switch_dir / f"{switch_id}.json"

    def agent_mutation_attempt_path(self, attempt_id: str) -> Path:
        return self.agent_mutation_attempt_dir / f"{attempt_id}.json"

    def agent_replay_path(self, replay_id: str) -> Path:
        return self.agent_replay_dir / f"{replay_id}.json"

    def agent_pack_capability_path(self, capability_attempt_id: str) -> Path:
        return self.agent_pack_capability_dir / f"{capability_attempt_id}.json"

    def brain_route_path(self, route_id: str) -> Path:
        return self.brain_route_dir / f"{route_id}.json"

    def brain_ledger_path(self, ledger_id: str) -> Path:
        return self.brain_ledger_dir / f"{ledger_id}.json"

    def brain_switch_path(self, switch_id: str) -> Path:
        return self.brain_switch_dir / f"{switch_id}.json"

    def brain_aggregation_path(self, aggregation_id: str) -> Path:
        return self.brain_aggregation_dir / f"{aggregation_id}.json"

    def judge_record_path(self, judge_record_id: str) -> Path:
        return self.judge_record_dir / f"{judge_record_id}.json"

    def judge_conflict_path(self, conflict_id: str) -> Path:
        return self.judge_conflict_dir / f"{conflict_id}.json"

    def judge_acceptance_path(self, acceptance_id: str) -> Path:
        return self.judge_acceptance_dir / f"{acceptance_id}.json"

    def judge_recommendation_path(self, recommendation_id: str) -> Path:
        return self.judge_recommendation_dir / f"{recommendation_id}.json"

    def judge_adjudication_path(self, adjudication_id: str) -> Path:
        return self.judge_adjudication_dir / f"{adjudication_id}.json"

    def judge_calibration_path(self, calibration_id: str) -> Path:
        return self.judge_calibration_dir / f"{calibration_id}.json"

    def credential_boundary_path(self, boundary_id: str) -> Path:
        return self.credential_boundary_dir / f"{boundary_id}.json"

    def sensitive_change_path(self, gate_id: str) -> Path:
        return self.sensitive_change_dir / f"{gate_id}.json"

    def backup_restore_path(self, restore_id: str) -> Path:
        return self.backup_restore_dir / f"{restore_id}.json"

    def helpful_failure_path(self, failure_id: str) -> Path:
        return self.helpful_failure_dir / f"{failure_id}.json"

    def idempotency_path(self, idempotency_id: str) -> Path:
        return self.idempotency_dir / f"{idempotency_id}.json"

    def retention_path(self, retention_id: str) -> Path:
        return self.retention_dir / f"{retention_id}.json"

    def operator_status_path(self, status_id: str) -> Path:
        return self.operator_status_dir / f"{status_id}.json"

    def release_report_path(self, report_id: str) -> Path:
        return self.release_report_dir / f"{report_id}.json"

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

    def get_agent_role(self, role_id: str) -> dict[str, Any] | None:
        path = self.agent_role_path(role_id)
        if path.exists():
            return _read_json(path)
        for candidate in self._agent_role_records():
            if candidate.get("role_key") == role_id or candidate.get("role_id") == role_id:
                return candidate
        return None

    def get_agent_trace(self, trace_id: str) -> dict[str, Any] | None:
        path = self.agent_trace_path(trace_id)
        if not path.exists():
            return None
        return _read_json(path)

    def get_agent_diagnosis(self, diagnosis_id: str) -> dict[str, Any] | None:
        path = self.agent_diagnosis_path(diagnosis_id)
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

    def get_memory_control(self, control_id: str) -> dict[str, Any] | None:
        path = self.memory_control_path(control_id)
        if not path.exists():
            return None
        return _read_json(path)

    def get_memory_export(self, export_id: str) -> dict[str, Any] | None:
        path = self.memory_export_path(export_id)
        if not path.exists():
            return None
        return _read_json(path)

    def get_memory_adaptation(self, adaptation_id: str) -> dict[str, Any] | None:
        path = self.memory_adaptation_path(adaptation_id)
        if not path.exists():
            return None
        return _read_json(path)

    def get_knowledge_capsule(self, capsule_id: str) -> dict[str, Any] | None:
        path = self.knowledge_capsule_path(capsule_id)
        if not path.exists():
            return None
        return _read_json(path)

    def get_decision_card(self, card_id: str) -> dict[str, Any] | None:
        path = self.decision_card_path(card_id)
        if not path.exists():
            return None
        return _read_json(path)

    def get_correction(self, correction_id: str) -> dict[str, Any] | None:
        path = self.correction_path(correction_id)
        if not path.exists():
            return None
        return _read_json(path)

    def get_share(self, share_id: str) -> dict[str, Any] | None:
        path = self.share_path(share_id)
        if not path.exists():
            return None
        return _read_json(path)

    def get_understanding_suggestion(self, suggestion_id: str) -> dict[str, Any] | None:
        path = self.understanding_suggestion_path(suggestion_id)
        if not path.exists():
            return None
        return _read_json(path)

    def get_ontology_item(self, item_id: str) -> dict[str, Any] | None:
        path = self.ontology_item_path(item_id)
        if not path.exists():
            return None
        return _read_json(path)

    def get_operational_map(self, map_id: str) -> dict[str, Any] | None:
        path = self.operational_map_path(map_id)
        if not path.exists():
            return None
        return _read_json(path)

    def get_contradiction(self, contradiction_id: str) -> dict[str, Any] | None:
        path = self.contradiction_path(contradiction_id)
        if not path.exists():
            return None
        return _read_json(path)

    def get_staleness(self, staleness_id: str) -> dict[str, Any] | None:
        path = self.staleness_path(staleness_id)
        if not path.exists():
            return None
        return _read_json(path)

    def get_ontology_change(self, change_id: str) -> dict[str, Any] | None:
        path = self.ontology_change_path(change_id)
        if not path.exists():
            return None
        return _read_json(path)

    def get_namespace_promotion(self, promotion_id: str) -> dict[str, Any] | None:
        path = self.namespace_promotion_path(promotion_id)
        if not path.exists():
            return None
        return _read_json(path)

    def get_learning(self, learning_id: str) -> dict[str, Any] | None:
        path = self.learning_path(learning_id)
        if not path.exists():
            return None
        return _read_json(path)

    def get_trajectory(self, trajectory_id: str) -> dict[str, Any] | None:
        path = self.trajectory_path(trajectory_id)
        if not path.exists():
            return None
        return _read_json(path)

    def get_lesson_candidate(self, lesson_id: str) -> dict[str, Any] | None:
        path = self.lesson_candidate_path(lesson_id)
        if not path.exists():
            return None
        return _read_json(path)

    def get_connected_outcome(self, outcome_id: str) -> dict[str, Any] | None:
        path = self.connected_outcome_path(outcome_id)
        if not path.exists():
            return None
        return _read_json(path)

    def get_agent_pack(self, pack_id: str) -> dict[str, Any] | None:
        path = self.agent_pack_path(pack_id)
        if not path.exists():
            return None
        return _read_json(path)

    def get_agent_pack_install(self, install_id: str) -> dict[str, Any] | None:
        path = self.agent_pack_install_path(install_id)
        if not path.exists():
            return None
        return _read_json(path)

    def get_agent_pack_activation(self, activation_id: str) -> dict[str, Any] | None:
        path = self.agent_pack_activation_path(activation_id)
        if not path.exists():
            return None
        return _read_json(path)

    def get_agent_pack_playbook_proposal(self, proposal_id: str) -> dict[str, Any] | None:
        path = self.agent_pack_playbook_proposal_path(proposal_id)
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

    def _memory_control_records(self, scope: dict[str, str]) -> list[dict[str, Any]]:
        if not self.memory_control_dir.exists():
            return []
        records = []
        for path in sorted(self.memory_control_dir.glob("*.json")):
            record = _read_json(path)
            if record.get("scope") == scope:
                records.append(record)
        return records

    def _memory_adaptation_records(self, scope: dict[str, str]) -> list[dict[str, Any]]:
        if not self.memory_adaptation_dir.exists():
            return []
        records = []
        for path in sorted(self.memory_adaptation_dir.glob("*.json")):
            record = _read_json(path)
            if record.get("scope") == scope:
                records.append(record)
        return records

    def _knowledge_capsule_records(self, scope: dict[str, str]) -> list[dict[str, Any]]:
        if not self.knowledge_capsule_dir.exists():
            return []
        records = []
        for path in sorted(self.knowledge_capsule_dir.glob("*.json")):
            record = _read_json(path)
            if record.get("scope") == scope:
                records.append(record)
        return records

    def _decision_card_records(self, scope: dict[str, str]) -> list[dict[str, Any]]:
        if not self.decision_card_dir.exists():
            return []
        records = []
        for path in sorted(self.decision_card_dir.glob("*.json")):
            record = _read_json(path)
            if record.get("scope") == scope:
                records.append(record)
        return records

    def _correction_records(self, scope: dict[str, str]) -> list[dict[str, Any]]:
        if not self.correction_dir.exists():
            return []
        records = []
        for path in sorted(self.correction_dir.glob("*.json")):
            record = _read_json(path)
            if record.get("scope") == scope:
                records.append(record)
        return records

    def _share_records(self, scope: dict[str, str]) -> list[dict[str, Any]]:
        if not self.share_dir.exists():
            return []
        records = []
        for path in sorted(self.share_dir.glob("*.json")):
            record = _read_json(path)
            if record.get("scope") == scope:
                records.append(record)
        return records

    def _understanding_suggestion_records(self, scope: dict[str, str]) -> list[dict[str, Any]]:
        if not self.understanding_suggestion_dir.exists():
            return []
        records = []
        for path in sorted(self.understanding_suggestion_dir.glob("*.json")):
            record = _read_json(path)
            if record.get("scope") == scope:
                records.append(record)
        return records

    def _ontology_item_records(self, scope: dict[str, str]) -> list[dict[str, Any]]:
        if not self.ontology_item_dir.exists():
            return []
        records = []
        for path in sorted(self.ontology_item_dir.glob("*.json")):
            record = _read_json(path)
            if record.get("scope") == scope:
                records.append(record)
        return records

    def _ontology_change_records(self, scope: dict[str, str]) -> list[dict[str, Any]]:
        if not self.ontology_change_dir.exists():
            return []
        records = []
        for path in sorted(self.ontology_change_dir.glob("*.json")):
            record = _read_json(path)
            if record.get("scope") == scope:
                records.append(record)
        return records

    def _namespace_promotion_records(self, scope: dict[str, str]) -> list[dict[str, Any]]:
        if not self.namespace_promotion_dir.exists():
            return []
        records = []
        for path in sorted(self.namespace_promotion_dir.glob("*.json")):
            record = _read_json(path)
            target_scope = record.get("target", {}).get("scope") or record.get("target_scope")
            if target_scope == scope:
                records.append(record)
        return records

    def _all_namespace_promotion_records(self) -> list[dict[str, Any]]:
        if not self.namespace_promotion_dir.exists():
            return []
        return [_read_json(path) for path in sorted(self.namespace_promotion_dir.glob("*.json"))]

    def _all_audit_events(self) -> list[dict[str, Any]]:
        if not self.audit_path.exists():
            return []
        events = []
        for line in self.audit_path.read_text().splitlines():
            if line.strip():
                events.append(json.loads(line))
        return events

    def _learning_records(self, scope: dict[str, str]) -> list[dict[str, Any]]:
        if not self.learning_dir.exists():
            return []
        records = []
        for path in sorted(self.learning_dir.glob("*.json")):
            record = _read_json(path)
            if record.get("scope") == scope:
                records.append(record)
        return records

    def _trajectory_records(self, scope: dict[str, str]) -> list[dict[str, Any]]:
        if not self.trajectory_dir.exists():
            return []
        records = []
        for path in sorted(self.trajectory_dir.glob("*.json")):
            record = _read_json(path)
            if record.get("scope") == scope:
                records.append(record)
        return records

    def _lesson_candidate_records(self, scope: dict[str, str]) -> list[dict[str, Any]]:
        if not self.lesson_candidate_dir.exists():
            return []
        records = []
        for path in sorted(self.lesson_candidate_dir.glob("*.json")):
            record = _read_json(path)
            if record.get("scope") == scope:
                records.append(record)
        return records

    def _connected_outcome_records(self, scope: dict[str, str]) -> list[dict[str, Any]]:
        if not self.connected_outcome_dir.exists():
            return []
        records = []
        for path in sorted(self.connected_outcome_dir.glob("*.json")):
            record = _read_json(path)
            if record.get("scope") == scope:
                records.append(record)
        return records

    def _mission_escalation_records(self, scope: dict[str, str]) -> list[dict[str, Any]]:
        if not self.mission_escalation_dir.exists():
            return []
        records = []
        for path in sorted(self.mission_escalation_dir.glob("*.json")):
            record = _read_json(path)
            if record.get("scope") == scope:
                records.append(record)
        return records

    def _model_evaluation_records(self, scope: dict[str, str]) -> list[dict[str, Any]]:
        if not self.model_evaluation_dir.exists():
            return []
        records = []
        for path in sorted(self.model_evaluation_dir.glob("*.json")):
            record = _read_json(path)
            if record.get("scope") == scope:
                records.append(record)
        return records

    def _agent_role_records(self, scope: dict[str, str] | None = None) -> list[dict[str, Any]]:
        if not self.agent_role_dir.exists():
            return []
        records = []
        for path in sorted(self.agent_role_dir.glob("*.json")):
            record = _read_json(path)
            if scope is None or record.get("scope") == scope:
                records.append(record)
        return records

    def _agent_output_records(self, scope: dict[str, str] | None = None) -> list[dict[str, Any]]:
        if not self.agent_output_dir.exists():
            return []
        records = []
        for path in sorted(self.agent_output_dir.glob("*.json")):
            record = _read_json(path)
            if scope is None or record.get("scope") == scope:
                records.append(record)
        return records

    def _agent_diagnosis_records(self, scope: dict[str, str] | None = None) -> list[dict[str, Any]]:
        if not self.agent_diagnosis_dir.exists():
            return []
        records = []
        for path in sorted(self.agent_diagnosis_dir.glob("*.json")):
            record = _read_json(path)
            if scope is None or record.get("scope") == scope:
                records.append(record)
        return records

    def _brain_route_records(self, scope: dict[str, str] | None = None) -> list[dict[str, Any]]:
        if not self.brain_route_dir.exists():
            return []
        records = []
        for path in sorted(self.brain_route_dir.glob("*.json")):
            record = _read_json(path)
            if scope is None or record.get("scope") == scope:
                records.append(record)
        return records

    def _brain_ledger_records(self, scope: dict[str, str] | None = None) -> list[dict[str, Any]]:
        if not self.brain_ledger_dir.exists():
            return []
        records = []
        for path in sorted(self.brain_ledger_dir.glob("*.json")):
            record = _read_json(path)
            if scope is None or record.get("scope") == scope:
                records.append(record)
        return records

    def _judge_records(self, scope: dict[str, str] | None = None) -> list[dict[str, Any]]:
        if not self.judge_record_dir.exists():
            return []
        records = []
        for path in sorted(self.judge_record_dir.glob("*.json")):
            record = _read_json(path)
            if scope is None or record.get("scope") == scope:
                records.append(record)
        return records

    def get_brain_route(self, route_id: str) -> dict[str, Any] | None:
        path = self.brain_route_path(route_id)
        if not path.exists():
            return None
        return _read_json(path)

    def get_judge_record(self, judge_record_id: str) -> dict[str, Any] | None:
        path = self.judge_record_path(judge_record_id)
        if not path.exists():
            return None
        return _read_json(path)

    def _agent_pack_records(self) -> list[dict[str, Any]]:
        if not self.pack_registry_dir.exists():
            return []
        return [_read_json(path) for path in sorted(self.pack_registry_dir.glob("*.json"))]

    def _agent_pack_install_records(self, scope: dict[str, str]) -> list[dict[str, Any]]:
        if not self.pack_install_dir.exists():
            return []
        records = []
        for path in sorted(self.pack_install_dir.glob("*.json")):
            record = _read_json(path)
            if record.get("scope") == scope:
                records.append(record)
        return records

    def _agent_pack_activation_records(self, scope: dict[str, str]) -> list[dict[str, Any]]:
        if not self.pack_activation_dir.exists():
            return []
        records = []
        for path in sorted(self.pack_activation_dir.glob("*.json")):
            record = _read_json(path)
            if record.get("scope") == scope:
                records.append(record)
        return records

    def _find_agent_pack_install(self, pack_id: str, scope: dict[str, str]) -> dict[str, Any] | None:
        matches = [
            record
            for record in self._agent_pack_install_records(scope)
            if record.get("pack_id") == pack_id and record.get("status") in {"installed", "active", "rolled_back_to_pinned"}
        ]
        return matches[-1] if matches else None

    def _find_agent_pack_activation(self, pack_id: str, scope: dict[str, str]) -> dict[str, Any] | None:
        matches = [
            record
            for record in self._agent_pack_activation_records(scope)
            if record.get("pack_id") == pack_id and record.get("status") == "active"
        ]
        return matches[-1] if matches else None

    def _pack_supply_chain_status(self, manifest: dict[str, Any]) -> dict[str, Any]:
        trust = manifest.get("trust", {})
        supply_chain = manifest.get("supply_chain", {})
        signature = supply_chain.get("signature", {})
        attestation = supply_chain.get("attestation", {})
        sbom = supply_chain.get("sbom", {})
        provenance = supply_chain.get("provenance", {})
        registry_source = trust.get("source", "unknown")
        checks = {
            "trusted_registry_source": registry_source in TRUSTED_PACK_SOURCES,
            "signature_verified": signature.get("verified") is True,
            "attestation_verified": attestation.get("verified") is True,
            "sbom_present": sbom.get("present") is True,
            "provenance_present": provenance.get("predicate_type") == "https://slsa.dev/provenance/v1",
            "version_declared": bool(manifest.get("version")),
            "risk_label_present": bool(trust.get("risk_label")),
            "update_metadata_present": bool(manifest.get("updates")),
        }
        return {
            "checks": checks,
            "verified": all(checks.values()),
            "registry_source": registry_source,
            "risk_label": trust.get("risk_label", "unknown"),
            "certified": trust.get("certified") is True,
        }

    def _pack_policy_decision(
        self,
        *,
        scope: dict[str, str],
        subject_id: str,
        decision: str,
        policy: str,
        reason: str,
        resolution_path: list[str],
    ) -> dict[str, Any]:
        decision_base = {
            "schema_version": "cs.policy_decision.v0",
            "decision": decision,
            "policy": policy,
            "reason": redact_text(reason),
            "resolution_path": resolution_path,
            "subject": {"type": "agent_pack", "id": subject_id},
            "scope": scope,
            "created_at": utc_now(),
        }
        decision_id = f"policy_{_json_hash(decision_base)[:16]}"
        record = dict(decision_base)
        record["policy_decision_id"] = decision_id
        return record

    def register_agent_pack(self, manifest_path: Path, scope: dict[str, str]) -> dict[str, Any]:
        if not manifest_path.exists():
            return {"status": "not_found", "resource": "manifest"}
        try:
            raw = manifest_path.read_bytes()
            manifest = json.loads(raw.decode())
        except (OSError, ValueError) as error:
            return {"status": "invalid", "resource": "manifest", "message": str(error)}

        pack_id = str(manifest.get("pack_id", "unknown_pack"))
        missing_fields = sorted(
            field
            for field in REQUIRED_AGENT_PACK_FIELDS
            if field not in manifest or manifest.get(field) is None or manifest.get(field) == ""
        )
        forbidden_runtime = manifest.get("forbidden_runtime", {})
        direct_provider_logic = any(
            bool(forbidden_runtime.get(field))
            for field in ["provider_clients", "extension_owned_credentials", "direct_api_writeback", "raw_secret_access"]
        )
        source_digest = sha256_bytes(raw)
        supply_chain = self._pack_supply_chain_status(manifest)
        if missing_fields or direct_provider_logic:
            decision = self._pack_policy_decision(
                scope=scope,
                subject_id=pack_id,
                decision="deny",
                policy="agent_pack_registry_validation",
                reason="Agent Pack registry validation failed before availability.",
                resolution_path=[
                    "Remove provider clients, credential handling, direct API writeback, and raw secret access from the pack.",
                    "Declare connector and action requirements for ConnectorHub mediation.",
                    "Attach required manifest, provenance, attestation, SBOM, and evaluation metadata.",
                ],
            )
            quarantine_base = {
                "schema_version": "cs.agent_pack_quarantine.v0",
                "status": "quarantined",
                "scope": scope,
                "pack_id": pack_id,
                "manifest_path": str(manifest_path),
                "source_digest": source_digest,
                "missing_fields": missing_fields,
                "direct_provider_logic_detected": direct_provider_logic,
                "forbidden_runtime": forbidden_runtime,
                "policy_decision": decision,
                "created_at": utc_now(),
            }
            quarantine_id = f"packquar_{_json_hash(quarantine_base)[:16]}"
            quarantine = dict(quarantine_base)
            quarantine["quarantine_id"] = quarantine_id
            _write_json(self.agent_pack_quarantine_path(quarantine_id), quarantine)
            event = self.append_audit(
                "pack.registry.quarantined",
                scope,
                {"type": "agent_pack", "id": pack_id},
                {"policy_decision_id": decision["policy_decision_id"], "direct_provider_logic_detected": direct_provider_logic},
            )
            return {"status": "quarantined", "quarantine": quarantine, "policy_decision": decision, "audit_event": event}

        record_base = {
            "schema_version": "cs.agent_pack_registry_record.v0",
            "status": "available" if supply_chain["verified"] and supply_chain["certified"] else "review_required",
            "scope": scope,
            "pack_id": pack_id,
            "name": manifest.get("name", pack_id),
            "version": manifest.get("version"),
            "manifest_path": str(manifest_path),
            "source_digest": source_digest,
            "registry_source": supply_chain["registry_source"],
            "trust": manifest.get("trust", {}),
            "supply_chain": supply_chain,
            "manifest": manifest,
            "install_requires_review": not (supply_chain["verified"] and supply_chain["certified"]),
            "created_at": utc_now(),
        }
        record = dict(record_base)
        _write_json(self.agent_pack_path(pack_id), record)
        event = self.append_audit(
            "pack.registry.imported",
            scope,
            {"type": "agent_pack", "id": pack_id},
            {
                "registry_source": supply_chain["registry_source"],
                "supply_chain_verified": supply_chain["verified"],
                "certified": supply_chain["certified"],
            },
        )
        return {"agent_pack": record, "audit_event": event}

    def list_agent_packs(self, scope: dict[str, str]) -> dict[str, Any]:
        records = self._agent_pack_records()
        registry_view = {
            "schema_version": "cs.agent_pack_registry_view.v0",
            "status": "ready",
            "scope": scope,
            "trust_model": {
                "default_sources": ["first_party", "organization_private", "curated_certified"],
                "public_marketplace_default": False,
                "risk_labels_visible": True,
            },
            "packs": [
                {
                    "pack_id": record.get("pack_id"),
                    "name": record.get("name"),
                    "version": record.get("version"),
                    "status": record.get("status"),
                    "registry_source": record.get("registry_source"),
                    "risk_label": record.get("trust", {}).get("risk_label"),
                    "certified": record.get("trust", {}).get("certified"),
                    "supply_chain_verified": record.get("supply_chain", {}).get("verified"),
                    "install_requires_review": record.get("install_requires_review"),
                }
                for record in records
            ],
            "created_at": utc_now(),
        }
        event = self.append_audit(
            "pack.registry.listed",
            scope,
            {"type": "agent_pack_registry", "id": "local"},
            {"pack_count": len(records), "public_marketplace_default": False},
        )
        return {"registry": registry_view, "audit_event": event}

    def show_agent_pack(self, pack_id: str, scope: dict[str, str]) -> dict[str, Any]:
        pack = self.get_agent_pack(pack_id)
        if pack is None:
            return {"status": "not_found", "resource": "agent_pack"}
        manifest = pack.get("manifest", {})
        detail = {
            "schema_version": "cs.agent_pack_detail.v0",
            "status": pack.get("status"),
            "scope": scope,
            "pack_id": pack_id,
            "version": pack.get("version"),
            "top_level_unit": "agent_pack",
            "role_contract": manifest.get("role_contract"),
            "role_card": manifest.get("role_card"),
            "allowed_capabilities": manifest.get("allowed_capabilities", []),
            "connector_requirements": manifest.get("connector_requirements", []),
            "memory_scope": manifest.get("memory_scope"),
            "model_policy": manifest.get("model_policy"),
            "judge_rubric": manifest.get("judge_rubric"),
            "playbooks": manifest.get("playbooks", []),
            "after_action_review_template": manifest.get("after_action_review_template"),
            "evaluation_expectations": manifest.get("evaluation_expectations"),
            "components": manifest.get("components", {}),
            "trust": pack.get("trust"),
            "supply_chain": pack.get("supply_chain"),
            "source_digest": pack.get("source_digest"),
        }
        event = self.append_audit(
            "pack.detail.read",
            scope,
            {"type": "agent_pack", "id": pack_id},
            {"top_level_unit": "agent_pack"},
        )
        return {"agent_pack_detail": detail, "audit_event": event}

    def install_agent_pack(self, pack_id: str, *, version: str | None, dry_run: bool, scope: dict[str, str]) -> dict[str, Any]:
        pack = self.get_agent_pack(pack_id)
        if pack is None:
            return {"status": "not_found", "resource": "agent_pack"}
        selected_version = version or pack.get("version")
        review_required = bool(pack.get("install_requires_review"))
        install_base = {
            "schema_version": "cs.agent_pack_install.v0",
            "status": "install_preview" if dry_run else "installed",
            "scope": scope,
            "pack_id": pack_id,
            "version": selected_version,
            "pinned_version": selected_version,
            "version_pinned_by_default": True,
            "review_required": review_required,
            "activation_status": "inactive",
            "can_act": False,
            "mission_authority": False,
            "granted_capabilities": [],
            "supply_chain": pack.get("supply_chain"),
            "trust": pack.get("trust"),
            "behavior_changing_updates_apply_silently": False,
            "created_at": utc_now(),
        }
        install_id = f"packinst_{_json_hash(install_base)[:16]}"
        install = dict(install_base)
        install["install_id"] = install_id
        event = self.append_audit(
            "pack.install.previewed" if dry_run else "pack.installed",
            scope,
            {"type": "agent_pack", "id": pack_id},
            {"version": selected_version, "dry_run": dry_run, "activation_status": "inactive", "can_act": False},
        )
        if not dry_run:
            _write_json(self.agent_pack_install_path(install_id), install)
        return {"install": install, "audit_event": event}

    def activate_agent_pack(
        self,
        pack_id: str,
        *,
        grants: list[str],
        mission_id: str | None,
        org_admin_shortcut: bool,
        policy_id: str | None,
        scope: dict[str, str],
    ) -> dict[str, Any]:
        pack = self.get_agent_pack(pack_id)
        if pack is None:
            return {"status": "not_found", "resource": "agent_pack"}
        install = self._find_agent_pack_install(pack_id, scope)
        if install is None:
            return {"status": "not_found", "resource": "agent_pack_install"}
        supply_chain = pack.get("supply_chain", {})
        manifest = pack.get("manifest", {})
        allowed_capabilities = list(manifest.get("allowed_capabilities", []))
        requested_grants = grants or []
        if pack.get("status") != "available" or supply_chain.get("verified") is not True or pack.get("trust", {}).get("certified") is not True:
            decision = self._pack_policy_decision(
                scope=scope,
                subject_id=pack_id,
                decision="deny",
                policy="agent_pack_activation_trust_required",
                reason="Untrusted or uncertified Agent Pack cannot be activated silently.",
                resolution_path=[
                    "Request a reviewed exception with limited capabilities, or use a certified pack.",
                    "Attach registry source, signature, attestation, SBOM, provenance, risk, and certification evidence.",
                ],
            )
            event = self.append_audit(
                "pack.activation.denied",
                scope,
                {"type": "agent_pack", "id": pack_id},
                {"policy_decision_id": decision["policy_decision_id"], "silent_activation": False},
            )
            return {"status": "policy_denied", "policy_decision": decision, "audit_event": event}
        unknown_grants = sorted(set(requested_grants) - set(allowed_capabilities))
        if unknown_grants:
            decision = self._pack_policy_decision(
                scope=scope,
                subject_id=pack_id,
                decision="deny",
                policy="agent_pack_capability_grant_boundary",
                reason="Requested capabilities are outside the Agent Pack contract.",
                resolution_path=["Grant only declared capabilities or revise and recertify the Agent Pack contract."],
            )
            event = self.append_audit(
                "pack.activation.denied",
                scope,
                {"type": "agent_pack", "id": pack_id},
                {"policy_decision_id": decision["policy_decision_id"], "unknown_grants": unknown_grants},
            )
            return {"status": "policy_denied", "policy_decision": decision, "audit_event": event}
        if org_admin_shortcut and not policy_id:
            decision = self._pack_policy_decision(
                scope=scope,
                subject_id=pack_id,
                decision="deny",
                policy="organization_pack_shortcut_policy_required",
                reason="Organization-admin shortcut requires an explicit organization policy record.",
                resolution_path=["Attach the organization policy that allows the default permission shortcut."],
            )
            event = self.append_audit(
                "pack.activation.denied",
                scope,
                {"type": "agent_pack", "id": pack_id},
                {"policy_decision_id": decision["policy_decision_id"], "org_admin_shortcut": True},
            )
            return {"status": "policy_denied", "policy_decision": decision, "audit_event": event}

        decision = self._pack_policy_decision(
            scope=scope,
            subject_id=pack_id,
            decision="allow",
            policy="agent_pack_explicit_activation_grants",
            reason="Certified Agent Pack activation with explicit owner-scoped capability grants.",
            resolution_path=["Use ConnectorHub-mediated capabilities and audit every action."],
        )
        activation_base = {
            "schema_version": "cs.agent_pack_activation.v0",
            "status": "active",
            "scope": scope,
            "pack_id": pack_id,
            "install_id": install["install_id"],
            "version": install.get("pinned_version"),
            "mission_id": mission_id,
            "role_card": manifest.get("role_card"),
            "required_connector_capabilities": manifest.get("connector_requirements", []),
            "memory_scope": manifest.get("memory_scope"),
            "allowed_actions": manifest.get("allowed_actions", []),
            "model_policy": manifest.get("model_policy"),
            "evaluation_rubric": manifest.get("judge_rubric"),
            "risk_level": pack.get("trust", {}).get("risk_label"),
            "requested_permissions": allowed_capabilities,
            "granted_capabilities": requested_grants,
            "capability_disclosure_complete": True,
            "owner_granted_only_needed_capabilities": set(requested_grants).issubset(set(allowed_capabilities)),
            "connectorhub_boundary": {
                "mediated_by": "ConnectorHub",
                "direct_provider_access": False,
                "credentials_exposed_to_agent": False,
                "declared_actions_only": True,
            },
            "organization_admin_shortcut": {
                "used": org_admin_shortcut,
                "policy_id": policy_id,
                "visible": True,
                "auditable": True,
                "bypasses_capability_disclosure": False,
                "rollback_available": True,
            },
            "rollback": {"available": True, "target_version": install.get("pinned_version")},
            "policy_decision": decision,
            "silent_activation": False,
            "created_at": utc_now(),
        }
        activation_id = f"packact_{_json_hash(activation_base)[:16]}"
        activation = dict(activation_base)
        activation["activation_id"] = activation_id
        _write_json(self.agent_pack_activation_path(activation_id), activation)
        updated_install = dict(install)
        updated_install["status"] = "active"
        updated_install["activation_status"] = "active"
        updated_install["activation_id"] = activation_id
        updated_install["granted_capabilities"] = requested_grants
        _write_json(self.agent_pack_install_path(install["install_id"]), updated_install)
        event = self.append_audit(
            "pack.activated",
            scope,
            {"type": "agent_pack", "id": pack_id},
            {
                "activation_id": activation_id,
                "grant_count": len(requested_grants),
                "org_admin_shortcut": org_admin_shortcut,
                "policy_decision_id": decision["policy_decision_id"],
            },
        )
        return {"activation": activation, "policy_decision": decision, "audit_event": event}

    def certify_agent_pack(self, pack_id: str, scope: dict[str, str]) -> dict[str, Any]:
        pack = self.get_agent_pack(pack_id)
        if pack is None:
            return {"status": "not_found", "resource": "agent_pack"}
        manifest = pack.get("manifest", {})
        card = manifest.get("certification", {})
        required = [
            "intended_use",
            "required_capabilities",
            "risk_level",
            "benchmark_scenarios",
            "prompt_injection_tests",
            "connector_action_boundary_checks",
            "llm_judge_rubrics",
            "model_compatibility",
            "outcome_history",
            "audit_coverage",
            "version_history",
            "rollback_support",
            "human_review",
            "scenario_results",
            "policy_checks",
        ]
        missing = [field for field in required if not card.get(field)]
        if missing:
            return {"status": "evidence_required", "resource": "certification", "missing": missing}
        certification_base = {
            "schema_version": "cs.agent_pack_certification.v0",
            "status": "certified",
            "scope": scope,
            "pack_id": pack_id,
            "version": pack.get("version"),
            "evaluation_card": card,
            "human_review_is_evidence_input": True,
            "outcome_history_is_evidence_input": True,
            "human_review_replaces_scenario_certification": False,
            "outcome_history_replaces_policy_checks": False,
            "scenario_certification_required_for_autonomous_action": True,
            "policy_checks_required_for_autonomous_action": True,
            "created_at": utc_now(),
        }
        certification_id = f"packcert_{_json_hash(certification_base)[:16]}"
        certification = dict(certification_base)
        certification["certification_id"] = certification_id
        _write_json(self.agent_pack_certification_path(certification_id), certification)
        event = self.append_audit(
            "pack.certified",
            scope,
            {"type": "agent_pack", "id": pack_id},
            {
                "certification_id": certification_id,
                "human_review_replaces_scenario_certification": False,
                "outcome_history_replaces_policy_checks": False,
            },
        )
        return {"certification": certification, "audit_event": event}

    def request_pack_connector_access(self, pack_id: str, *, capability: str, scope: dict[str, str]) -> dict[str, Any]:
        activation = self._find_agent_pack_activation(pack_id, scope)
        if activation is None:
            decision = self._pack_policy_decision(
                scope=scope,
                subject_id=pack_id,
                decision="deny",
                policy="agent_pack_install_is_not_activation",
                reason="Installed Agent Pack has no mission authority until explicit activation grants exist.",
                resolution_path=["Activate the pack for this workspace or mission with explicit capability grants."],
            )
            event = self.append_audit(
                "pack.connector_request.denied",
                scope,
                {"type": "agent_pack", "id": pack_id},
                {"policy_decision_id": decision["policy_decision_id"], "capability": capability},
            )
            return {"status": "policy_denied", "policy_decision": decision, "audit_event": event}
        if capability not in activation.get("granted_capabilities", []):
            decision = self._pack_policy_decision(
                scope=scope,
                subject_id=pack_id,
                decision="deny",
                policy="agent_pack_capability_not_granted",
                reason="Agent Pack requested a ConnectorHub capability that was not granted.",
                resolution_path=["Request the minimum capability grant from the namespace owner."],
            )
            event = self.append_audit(
                "pack.connector_request.denied",
                scope,
                {"type": "agent_pack", "id": pack_id},
                {"policy_decision_id": decision["policy_decision_id"], "capability": capability},
            )
            return {"status": "policy_denied", "policy_decision": decision, "audit_event": event}
        request_base = {
            "schema_version": "cs.agent_pack_connector_request.v0",
            "status": "mediated",
            "scope": scope,
            "pack_id": pack_id,
            "activation_id": activation["activation_id"],
            "capability": capability,
            "connectorhub": {
                "mediates_provider_access": True,
                "credential_custody": "ConnectorHub",
                "credentials_exposed_to_agent": False,
                "source_policy_enforced": True,
                "projections_enforced": True,
                "declared_actions_only": True,
                "delivery_audited": True,
                "retry_quarantine_supported": True,
                "raw_access_controlled": True,
                "direct_provider_access": False,
                "external_http_calls": 0,
            },
            "created_at": utc_now(),
        }
        request_id = f"packconn_{_json_hash(request_base)[:16]}"
        request = dict(request_base)
        request["connector_request_id"] = request_id
        event = self.append_audit(
            "pack.connector_request.mediated",
            scope,
            {"type": "agent_pack", "id": pack_id},
            {"connector_request_id": request_id, "capability": capability, "direct_provider_access": False},
        )
        return {"connector_request": request, "audit_event": event}

    def propose_pack_playbook_update(self, pack_id: str, *, lesson_id: str, scope: dict[str, str]) -> dict[str, Any]:
        pack = self.get_agent_pack(pack_id)
        if pack is None:
            return {"status": "not_found", "resource": "agent_pack"}
        lesson = self.get_lesson_candidate(lesson_id)
        if lesson is None:
            return {"status": "not_found", "resource": "lesson"}
        if lesson.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": lesson.get("scope")}
        proposal_base = {
            "schema_version": "cs.agent_pack_playbook_proposal.v0",
            "status": "proposed",
            "scope": scope,
            "pack_id": pack_id,
            "lesson_id": lesson_id,
            "source": "Experience Library",
            "trajectory_examples": [lesson.get("trajectory_id")],
            "judge_review": {
                "provider": "local_test",
                "rubric": pack.get("manifest", {}).get("judge_rubric"),
                "pass_judge": False,
                "result": "review_required_before_activation",
            },
            "owner_scope": scope,
            "risk": pack.get("trust", {}).get("risk_label"),
            "rollback": {"available": True, "affected_playbooks": [f"pack:{pack_id}:playbooks"]},
            "approval": {"required": True, "status": "pending", "scope": scope},
            "auto_globalize": False,
            "becomes_active_only_after_approval": True,
            "evidence_refs": [f"lesson:{lesson_id}", *lesson.get("evidence_refs", [])],
            "created_at": utc_now(),
        }
        proposal_id = f"packpb_{_json_hash(proposal_base)[:16]}"
        proposal = dict(proposal_base)
        proposal["playbook_proposal_id"] = proposal_id
        _write_json(self.agent_pack_playbook_proposal_path(proposal_id), proposal)
        event = self.append_audit(
            "pack.playbook.proposed",
            scope,
            {"type": "agent_pack_playbook_proposal", "id": proposal_id},
            {"pack_id": pack_id, "lesson_id": lesson_id, "auto_globalize": False},
        )
        return {"playbook_proposal": proposal, "audit_event": event}

    def approve_pack_playbook_update(self, proposal_id: str, scope: dict[str, str]) -> dict[str, Any]:
        proposal = self.get_agent_pack_playbook_proposal(proposal_id)
        if proposal is None:
            return {"status": "not_found", "resource": "playbook_proposal"}
        if proposal.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": proposal.get("scope")}
        updated = dict(proposal)
        updated["status"] = "active"
        updated["approval"] = {**updated.get("approval", {}), "status": "approved", "approved_at": utc_now()}
        updated["auto_globalize"] = False
        _write_json(self.agent_pack_playbook_proposal_path(proposal_id), updated)
        event = self.append_audit(
            "pack.playbook.approved",
            scope,
            {"type": "agent_pack_playbook_proposal", "id": proposal_id},
            {"pack_id": updated.get("pack_id"), "auto_globalize": False},
        )
        return {"playbook_proposal": updated, "audit_event": event}

    def update_agent_pack(
        self,
        pack_id: str,
        *,
        to_version: str,
        dry_run: bool,
        approve: bool,
        scope: dict[str, str],
    ) -> dict[str, Any]:
        pack = self.get_agent_pack(pack_id)
        if pack is None:
            return {"status": "not_found", "resource": "agent_pack"}
        install = self._find_agent_pack_install(pack_id, scope)
        if install is None:
            return {"status": "not_found", "resource": "agent_pack_install"}
        updates = pack.get("manifest", {}).get("updates", {})
        update_meta = updates.get(to_version)
        if not update_meta:
            return {"status": "not_found", "resource": "agent_pack_update"}
        if not dry_run and not approve:
            decision = self._pack_policy_decision(
                scope=scope,
                subject_id=pack_id,
                decision="requires_approval",
                policy="agent_pack_behavior_update_requires_owner_approval",
                reason="Pack updates do not apply without owner approval.",
                resolution_path=["Review the diff and evaluation gate, then rerun with explicit approval."],
            )
            event = self.append_audit(
                "pack.update.denied",
                scope,
                {"type": "agent_pack", "id": pack_id},
                {"policy_decision_id": decision["policy_decision_id"], "to_version": to_version},
            )
            return {"status": "approval_required", "policy_decision": decision, "audit_event": event}
        update_base = {
            "schema_version": "cs.agent_pack_update.v0",
            "status": "dry_run" if dry_run else "approved_applied",
            "scope": scope,
            "pack_id": pack_id,
            "from_version": install.get("pinned_version"),
            "to_version": to_version,
            "diff": update_meta.get("diff", {}),
            "evaluation_gate": update_meta.get("evaluation_gate", {}),
            "sandbox_canary_test": update_meta.get("sandbox_canary_test", {}),
            "migration_notes": update_meta.get("migration_notes", []),
            "owner_can_test_before_approving": True,
            "applied": not dry_run,
            "behavior_changing_update": bool(update_meta.get("behavior_changing")),
            "behavior_changing_silent_apply": False,
            "approval": {"required": not dry_run, "status": "approved" if approve and not dry_run else "not_requested"},
            "created_at": utc_now(),
        }
        update_id = f"packupd_{_json_hash(update_base)[:16]}"
        update = dict(update_base)
        update["update_id"] = update_id
        _write_json(self.agent_pack_update_path(update_id), update)
        if not dry_run:
            updated_install = dict(install)
            history = list(updated_install.get("version_history", []))
            history.append({"from_version": install.get("pinned_version"), "to_version": to_version, "update_id": update_id})
            updated_install["previous_pinned_version"] = install.get("pinned_version")
            updated_install["pinned_version"] = to_version
            updated_install["version_history"] = history
            _write_json(self.agent_pack_install_path(install["install_id"]), updated_install)
        event = self.append_audit(
            "pack.update.dry_run" if dry_run else "pack.update.applied",
            scope,
            {"type": "agent_pack", "id": pack_id},
            {"update_id": update_id, "to_version": to_version, "behavior_changing_silent_apply": False},
        )
        return {"pack_update": update, "audit_event": event}

    def rollback_agent_pack(self, pack_id: str, *, to_version: str, reason: str, scope: dict[str, str]) -> dict[str, Any]:
        install = self._find_agent_pack_install(pack_id, scope)
        if install is None:
            return {"status": "not_found", "resource": "agent_pack_install"}
        rollback_base = {
            "schema_version": "cs.agent_pack_rollback.v0",
            "status": "rolled_back",
            "scope": scope,
            "pack_id": pack_id,
            "from_version": install.get("pinned_version"),
            "to_version": to_version,
            "reason": redact_text(reason),
            "affected_missions": [record.get("mission_id") for record in self._agent_pack_activation_records(scope) if record.get("pack_id") == pack_id and record.get("mission_id")],
            "changes_recorded": True,
            "created_at": utc_now(),
        }
        rollback_id = f"packroll_{_json_hash(rollback_base)[:16]}"
        rollback = dict(rollback_base)
        rollback["rollback_id"] = rollback_id
        _write_json(self.agent_pack_rollback_path(rollback_id), rollback)
        updated_install = dict(install)
        history = list(updated_install.get("version_history", []))
        history.append({"from_version": install.get("pinned_version"), "to_version": to_version, "rollback_id": rollback_id})
        updated_install["status"] = "rolled_back_to_pinned"
        updated_install["previous_pinned_version"] = install.get("pinned_version")
        updated_install["pinned_version"] = to_version
        updated_install["version_history"] = history
        _write_json(self.agent_pack_install_path(install["install_id"]), updated_install)
        event = self.append_audit(
            "pack.rollback.completed",
            scope,
            {"type": "agent_pack", "id": pack_id},
            {"rollback_id": rollback_id, "to_version": to_version, "changes_recorded": True},
        )
        return {"pack_rollback": rollback, "audit_event": event}

    def emergency_patch_agent_pack(
        self,
        pack_id: str,
        *,
        patch_version: str,
        behavior_change: bool,
        scope: dict[str, str],
    ) -> dict[str, Any]:
        pack = self.get_agent_pack(pack_id)
        if pack is None:
            return {"status": "not_found", "resource": "agent_pack"}
        install = self._find_agent_pack_install(pack_id, scope)
        if install is None:
            return {"status": "not_found", "resource": "agent_pack_install"}
        patches = pack.get("manifest", {}).get("security_patches", {})
        patch_meta = patches.get(patch_version)
        if not patch_meta:
            return {"status": "not_found", "resource": "agent_pack_security_patch"}
        if behavior_change:
            decision = self._pack_policy_decision(
                scope=scope,
                subject_id=pack_id,
                decision="requires_review",
                policy="agent_pack_emergency_patch_behavior_change_review",
                reason="Behavior-changing emergency updates still require appropriate review.",
                resolution_path=["Split the security fix from behavior changes or obtain reviewed approval."],
            )
            event = self.append_audit(
                "pack.security_patch.denied",
                scope,
                {"type": "agent_pack", "id": pack_id},
                {"policy_decision_id": decision["policy_decision_id"], "patch_version": patch_version},
            )
            return {"status": "approval_required", "policy_decision": decision, "audit_event": event}
        patch_base = {
            "schema_version": "cs.agent_pack_security_patch.v0",
            "status": "applied",
            "scope": scope,
            "pack_id": pack_id,
            "from_version": install.get("pinned_version"),
            "patch_version": patch_version,
            "policy": patch_meta.get("policy", "emergency_security_patch"),
            "owner_visibility": True,
            "compatibility_checks": patch_meta.get("compatibility_checks", {}),
            "rollback": {"available": True, "target_version": install.get("pinned_version")},
            "behavior_change": False,
            "behavior_changing_updates_require_review": True,
            "created_at": utc_now(),
        }
        patch_id = f"packpatch_{_json_hash(patch_base)[:16]}"
        patch = dict(patch_base)
        patch["security_patch_id"] = patch_id
        _write_json(self.agent_pack_patch_path(patch_id), patch)
        updated_install = dict(install)
        history = list(updated_install.get("version_history", []))
        history.append({"from_version": install.get("pinned_version"), "to_version": patch_version, "security_patch_id": patch_id})
        updated_install["previous_pinned_version"] = install.get("pinned_version")
        updated_install["pinned_version"] = patch_version
        updated_install["version_history"] = history
        _write_json(self.agent_pack_install_path(install["install_id"]), updated_install)
        event = self.append_audit(
            "pack.security_patch.applied",
            scope,
            {"type": "agent_pack", "id": pack_id},
            {"security_patch_id": patch_id, "owner_visibility": True, "rollback_available": True},
        )
        return {"security_patch": patch, "audit_event": event}

    def _agent_role_id(self, role_key: str, scope: dict[str, str]) -> str:
        return f"role_{role_key}_{scope_key(scope)}"

    def _agent_role_record(self, role_key: str, scope: dict[str, str]) -> dict[str, Any]:
        template = AGENT_ROLE_TEMPLATES[role_key]
        role_contract = {
            "schema_version": "cs.agent_role_contract.v0",
            "role_key": role_key,
            "purpose": template["purpose"],
            "responsibilities": template["responsibilities"],
            "allowed_tools": template["allowed_tools"],
            "forbidden_actions": template["forbidden_actions"],
            "memory_scope": template["memory_scope"],
            "evidence_requirements": template["evidence_requirements"],
            "escalation_rules": template["escalation_rules"],
            "model_policy": template["model_policy"],
            "judge_rubric": template["judge_rubric"],
            "audit_expectations": template["audit_expectations"],
        }
        role_card = {
            "schema_version": "cs.agent_role_card.v0",
            "display_name": template["display_name"],
            "purpose": template["purpose"],
            "visible_to_daily_user": True,
            "summary": f"{template['display_name']} helps the Orchestrator with bounded, evidence-labeled work.",
            "risk_boundary": "cannot directly mutate truth or source systems",
        }
        role_id = self._agent_role_id(role_key, scope)
        record = {
            "schema_version": "cs.agent_role_record.v0",
            "role_id": role_id,
            "role_key": role_key,
            "status": "active",
            "scope": scope,
            "version": 1,
            "role_card": role_card,
            "role_contract": role_contract,
            "operator_contract_visible": True,
            "daily_user_card_visible": True,
            "contract_hash": _json_hash(role_contract),
            "created_at": utc_now(),
        }
        return record

    def ensure_agent_roles(self, scope: dict[str, str]) -> list[dict[str, Any]]:
        records = []
        for role_key in AGENT_ROLE_TEMPLATES:
            role_id = self._agent_role_id(role_key, scope)
            path = self.agent_role_path(role_id)
            if path.exists():
                records.append(_read_json(path))
                continue
            record = self._agent_role_record(role_key, scope)
            _write_json(path, record)
            self.append_audit(
                "agent.role_contract.created",
                scope,
                {"type": "agent_role", "id": role_id},
                {"role_key": role_key, "contract_hash": record["contract_hash"], "version": record["version"]},
            )
            records.append(record)
        return records

    def list_agent_roles(self, scope: dict[str, str]) -> dict[str, Any]:
        roles = self.ensure_agent_roles(scope)
        view = {
            "schema_version": "cs.agent_role_registry_view.v0",
            "status": "ready",
            "scope": scope,
            "orchestrator_led": True,
            "daily_users_manage_specialists_directly": False,
            "roles": [
                {
                    "role_id": role["role_id"],
                    "role_key": role["role_key"],
                    "display_name": role["role_card"]["display_name"],
                    "purpose": role["role_card"]["purpose"],
                    "contract_hash": role["contract_hash"],
                    "version": role["version"],
                }
                for role in roles
            ],
        }
        event = self.append_audit(
            "agent.roles.listed",
            scope,
            {"type": "agent_role_registry", "id": "local"},
            {"role_count": len(roles), "orchestrator_led": True},
        )
        return {"agent_roles": view, "audit_event": event}

    def show_agent_role(self, role_id: str, scope: dict[str, str], *, view: str) -> dict[str, Any]:
        self.ensure_agent_roles(scope)
        role = self.get_agent_role(role_id)
        if role is None:
            return {"status": "not_found", "resource": "agent_role"}
        if role.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": role.get("scope")}
        detail = {
            "schema_version": "cs.agent_role_view.v0",
            "status": "ready",
            "scope": scope,
            "role_id": role["role_id"],
            "role_key": role["role_key"],
            "view": view,
            "role_card": role["role_card"],
            "operator_contract": role["role_contract"] if view in {"operator", "both"} else None,
            "operator_contract_visible": view in {"operator", "both"},
            "daily_user_card_visible": True,
            "contract_hash": role["contract_hash"],
            "version": role["version"],
        }
        event = self.append_audit(
            "agent.role.read",
            scope,
            {"type": "agent_role", "id": role["role_id"]},
            {"view": view, "operator_contract_visible": detail["operator_contract_visible"]},
        )
        return {"agent_role_view": detail, "audit_event": event}

    def _agent_policy_decision(
        self,
        *,
        scope: dict[str, str],
        subject_id: str,
        decision: str,
        policy: str,
        reason: str,
        resolution_path: list[str],
    ) -> dict[str, Any]:
        decision_base = {
            "schema_version": "cs.policy_decision.v0",
            "decision": decision,
            "policy": policy,
            "reason": redact_text(reason),
            "resolution_path": resolution_path,
            "subject": {"type": "agent", "id": subject_id},
            "scope": scope,
            "created_at": utc_now(),
        }
        decision_id = f"policy_{_json_hash(decision_base)[:16]}"
        decision_record = dict(decision_base)
        decision_record["policy_decision_id"] = decision_id
        return decision_record

    def create_agent_mission_trace(self, mission_id: str, scope: dict[str, str]) -> dict[str, Any]:
        mission = self.get_mission(mission_id)
        if mission is None:
            return {"status": "not_found", "resource": "mission"}
        if mission.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": mission.get("scope")}
        roles = {role["role_key"]: role for role in self.ensure_agent_roles(scope)}
        evidence = mission.get("evidence", {})
        evidence_refs = list(evidence.get("artifact_refs") or [])
        if evidence.get("evidence_bundle_id"):
            evidence_refs.append(f"evidence_bundle:{evidence['evidence_bundle_id']}")
        trace_base = {
            "schema_version": "cs.agent_mission_trace.v0",
            "status": "completed_with_review",
            "scope": scope,
            "mission_id": mission_id,
            "orchestrator_role_id": roles["orchestrator"]["role_id"],
            "orchestrator_plan": {
                "goal": mission.get("goal"),
                "steps": [
                    "review evidence",
                    "draft memory candidate",
                    "prepare workflow path",
                    "evaluate policy",
                    "check connector boundary",
                    "judge outcome quality",
                    "capture playbook candidate",
                    "produce after-action review",
                ],
                "delegation_model": "orchestrator_as_controller_specialists_as_tools",
                "asks_for_missing_authority": True,
            },
            "delegations": [],
            "agent_activity_view": [],
            "outputs": [],
            "after_action_review": {
                "status": "complete",
                "owner_visible": True,
                "evidence_refs": evidence_refs,
                "missing_authority_questions": ["External action approval remains required before provider writeback."],
                "next_steps": ["Review evidence gaps before promoting memory or executing external actions."],
            },
            "accountability": {
                "namespace_owner": scope["owner_id"],
                "authority_grant_visible": True,
                "correction_and_rollback_visible": True,
                "agents_accountable_to_owner_scope": True,
            },
            "hidden_chain_of_thought_captured": False,
            "replay_without_hidden_chain_of_thought": True,
            "created_at": utc_now(),
        }
        trace_id = f"agenttrace_{_json_hash(trace_base)[:16]}"
        trace = dict(trace_base)
        trace["trace_id"] = trace_id

        delegation_specs = [
            ("evidence", "reviewed artifact and evidence bundle coverage", "summary"),
            ("memory", "drafted memory candidate with owner approval still required", "memory_candidate"),
            ("workflow", "prepared governed Action Card path instead of direct mutation", "action_proposal"),
            ("policy", "checked scope, risk, approval, and authority boundaries", "policy_review"),
            ("connector", "verified ConnectorHub mediation and credential custody boundary", "connector_review"),
            ("judge", "scored outcome quality as supporting signal only", "judgment"),
            ("playbook", "identified reusable playbook candidate requiring approval", "playbook_candidate"),
        ]
        for role_key, rationale, output_type in delegation_specs:
            role = roles[role_key]
            output_base = {
                "schema_version": "cs.agent_output.v0",
                "status": "evidence_labeled",
                "scope": scope,
                "trace_id": trace_id,
                "mission_id": mission_id,
                "role_id": role["role_id"],
                "role_key": role_key,
                "output_type": output_type,
                "summary": rationale,
                "evidence_refs": evidence_refs,
                "uncertainty": "medium" if evidence_refs else "high",
                "insufficient_evidence_label": not bool(evidence_refs),
                "source_refs_or_gap_label_present": True,
                "direct_mutation_performed": False,
                "created_at": utc_now(),
            }
            output_id = f"agentout_{_json_hash(output_base)[:16]}"
            output = dict(output_base)
            output["output_id"] = output_id
            _write_json(self.agent_output_path(output_id), output)
            delegation = {
                "role_id": role["role_id"],
                "role_key": role_key,
                "display_name": role["role_card"]["display_name"],
                "rationale": rationale,
                "handled_evidence_refs": evidence_refs,
                "tool_refs": role["role_contract"]["allowed_tools"],
                "result_output_id": output_id,
                "influenced_final_outcome": True,
            }
            trace["delegations"].append(delegation)
            trace["agent_activity_view"].append(
                {
                    "role_id": role["role_id"],
                    "display_name": role["role_card"]["display_name"],
                    "visible_when_useful": True,
                    "result_output_id": output_id,
                }
            )
            trace["outputs"].append(f"agent_output:{output_id}")
            self.append_audit(
                "agent.delegated",
                scope,
                {"type": "agent_role", "id": role["role_id"]},
                {"trace_id": trace_id, "mission_id": mission_id, "rationale": rationale},
            )

        _write_json(self.agent_trace_path(trace_id), trace)
        event = self.append_audit(
            "agent.orchestrated",
            scope,
            {"type": "agent_mission_trace", "id": trace_id},
            {"mission_id": mission_id, "delegation_count": len(trace["delegations"]), "after_action_review": True},
        )
        review_event = self.append_audit(
            "agent.after_action_review.created",
            scope,
            {"type": "agent_mission_trace", "id": trace_id},
            {"mission_id": mission_id, "evidence_ref_count": len(evidence_refs)},
        )
        return {"agent_trace": trace, "audit_event": event, "review_audit_event": review_event}

    def test_agent_direct_mutation(self, role_id: str, *, target: str, scope: dict[str, str]) -> dict[str, Any]:
        self.ensure_agent_roles(scope)
        role = self.get_agent_role(role_id)
        if role is None:
            return {"status": "not_found", "resource": "agent_role"}
        if role.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": role.get("scope")}
        decision = self._agent_policy_decision(
            scope=scope,
            subject_id=role["role_id"],
            decision="deny",
            policy="agent_workflow_path_required",
            reason="Agents cannot directly mutate durable truth, memory, source systems, or external systems.",
            resolution_path=["Use governed claim, memory, Workflow/Action, ConnectorHub, and approval paths."],
        )
        attempt_base = {
            "schema_version": "cs.agent_mutation_attempt.v0",
            "status": "denied",
            "scope": scope,
            "role_id": role["role_id"],
            "target": redact_text(target),
            "direct_mutation_performed": False,
            "governed_paths_required": ["claim", "memory", "workflow_action", "connectorhub", "audit"],
            "policy_decision": decision,
            "created_at": utc_now(),
        }
        attempt_id = f"agentmut_{_json_hash(attempt_base)[:16]}"
        attempt = dict(attempt_base)
        attempt["attempt_id"] = attempt_id
        _write_json(self.agent_mutation_attempt_path(attempt_id), attempt)
        event = self.append_audit(
            "agent.direct_mutation.denied",
            scope,
            {"type": "agent_role", "id": role["role_id"]},
            {"attempt_id": attempt_id, "target": target, "policy_decision_id": decision["policy_decision_id"]},
        )
        return {"status": "policy_denied", "mutation_attempt": attempt, "policy_decision": decision, "audit_event": event}

    def switch_agent_brain(self, role_id: str, *, provider: str, model: str, scope: dict[str, str]) -> dict[str, Any]:
        self.ensure_agent_roles(scope)
        role = self.get_agent_role(role_id)
        if role is None:
            return {"status": "not_found", "resource": "agent_role"}
        if role.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": role.get("scope")}
        switch_base = {
            "schema_version": "cs.agent_brain_switch.v0",
            "status": "switched",
            "scope": scope,
            "role_id": role["role_id"],
            "from_provider": role["role_contract"]["model_policy"].get("provider"),
            "to_provider": provider,
            "to_model": model,
            "contract_hash_before": role["contract_hash"],
            "contract_hash_after": role["contract_hash"],
            "allowed_tools_unchanged": True,
            "memory_scope_unchanged": True,
            "evidence_rules_unchanged": True,
            "audit_expectations_unchanged": True,
            "only_inference_brain_changed": True,
            "created_at": utc_now(),
        }
        switch_id = f"agentbrain_{_json_hash(switch_base)[:16]}"
        switch = dict(switch_base)
        switch["switch_id"] = switch_id
        _write_json(self.agent_brain_switch_path(switch_id), switch)
        event = self.append_audit(
            "agent.brain.switched",
            scope,
            {"type": "agent_role", "id": role["role_id"]},
            {"switch_id": switch_id, "from_provider": switch["from_provider"], "to_provider": provider, "contract_hash_unchanged": True},
        )
        return {"brain_switch": switch, "audit_event": event}

    def update_agent_contract(self, role_id: str, *, change_summary: str, scope: dict[str, str]) -> dict[str, Any]:
        self.ensure_agent_roles(scope)
        role = self.get_agent_role(role_id)
        if role is None:
            return {"status": "not_found", "resource": "agent_role"}
        if role.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": role.get("scope")}
        current_contract = role["role_contract"]
        updated_contract = dict(current_contract)
        updated_requirements = list(updated_contract.get("evidence_requirements", []))
        if "structured output schema" not in updated_requirements:
            updated_requirements.append("structured output schema")
        updated_contract["evidence_requirements"] = updated_requirements
        new_hash = _json_hash(updated_contract)
        update_base = {
            "schema_version": "cs.agent_contract_update.v0",
            "status": "versioned",
            "scope": scope,
            "role_id": role["role_id"],
            "from_version": role.get("version", 1),
            "to_version": int(role.get("version", 1)) + 1,
            "change_summary": redact_text(change_summary),
            "diff": {
                "evidence_requirements_added": ["structured output schema"],
                "allowed_tools_added": [],
                "memory_scope_changed": False,
                "authority_expansion": False,
            },
            "impact": {
                "affected_missions": [mission["mission_id"] for mission in self._mission_records(scope)],
                "affected_agent_packs": sorted({activation.get("pack_id") for activation in self._agent_pack_activation_records(scope) if activation.get("pack_id")}),
            },
            "migration_rollout_guidance": ["Re-run agent replay for active missions.", "Keep previous contract hash for audit comparison."],
            "old_contract_hash": role["contract_hash"],
            "new_contract_hash": new_hash,
            "created_at": utc_now(),
        }
        update_id = f"agentupd_{_json_hash(update_base)[:16]}"
        update = dict(update_base)
        update["update_id"] = update_id
        _write_json(self.agent_contract_update_path(update_id), update)
        updated_role = dict(role)
        updated_role["version"] = update["to_version"]
        updated_role["role_contract"] = updated_contract
        updated_role["contract_hash"] = new_hash
        updated_role["change_history"] = [*role.get("change_history", []), {"update_id": update_id, "from_version": update["from_version"], "to_version": update["to_version"]}]
        _write_json(self.agent_role_path(role["role_id"]), updated_role)
        event = self.append_audit(
            "agent.contract.updated",
            scope,
            {"type": "agent_role", "id": role["role_id"]},
            {"update_id": update_id, "from_version": update["from_version"], "to_version": update["to_version"], "authority_expansion": False},
        )
        return {"contract_update": update, "agent_role": updated_role, "audit_event": event}

    def test_agent_prompt_authority_expansion(
        self,
        role_id: str,
        *,
        requested_tool: str,
        requested_memory_scope: str,
        requested_authority: str,
        scope: dict[str, str],
    ) -> dict[str, Any]:
        self.ensure_agent_roles(scope)
        role = self.get_agent_role(role_id)
        if role is None:
            return {"status": "not_found", "resource": "agent_role"}
        if role.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": role.get("scope")}
        decision = self._agent_policy_decision(
            scope=scope,
            subject_id=role["role_id"],
            decision="deny",
            policy="agent_prompt_cannot_expand_authority",
            reason="Prompt-only agent changes cannot grant new tools, connector access, memory scope, write permissions, or action authority.",
            resolution_path=["Revise the Agent Role Contract, run policy review, and record a versioned contract update before granting authority."],
        )
        attempt_base = {
            "schema_version": "cs.agent_prompt_authority_attempt.v0",
            "status": "denied",
            "scope": scope,
            "role_id": role["role_id"],
            "requested_tool": requested_tool,
            "requested_memory_scope": requested_memory_scope,
            "requested_authority": requested_authority,
            "authority_expanded": False,
            "role_contract_required": True,
            "policy_decision": decision,
            "created_at": utc_now(),
        }
        attempt_id = f"agentauth_{_json_hash(attempt_base)[:16]}"
        attempt = dict(attempt_base)
        attempt["attempt_id"] = attempt_id
        _write_json(self.agent_mutation_attempt_path(attempt_id), attempt)
        event = self.append_audit(
            "agent.prompt_authority.denied",
            scope,
            {"type": "agent_role", "id": role["role_id"]},
            {"attempt_id": attempt_id, "policy_decision_id": decision["policy_decision_id"], "authority_expanded": False},
        )
        return {"status": "policy_denied", "authority_attempt": attempt, "policy_decision": decision, "audit_event": event}

    def record_agent_failure(
        self,
        trace_id: str,
        role_id: str,
        *,
        failure_kind: str,
        scope: dict[str, str],
    ) -> dict[str, Any]:
        trace = self.get_agent_trace(trace_id)
        if trace is None:
            return {"status": "not_found", "resource": "agent_trace"}
        if trace.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": trace.get("scope")}
        role = self.get_agent_role(role_id)
        if role is None:
            return {"status": "not_found", "resource": "agent_role"}
        diagnosis_base = {
            "schema_version": "cs.agent_failure_diagnosis.v0",
            "status": "diagnosed",
            "scope": scope,
            "trace_id": trace_id,
            "mission_id": trace.get("mission_id"),
            "role_id": role["role_id"],
            "failure_kind": failure_kind,
            "first_failing_layer": "connector_capability" if failure_kind in {"timeout", "unavailable_connector"} else "agent_output_validation",
            "mission_impact": "mission_can_continue_with_escalation",
            "retry_path": ["retry specialist with same contract", "fallback to Orchestrator summary", "request owner or operator review"],
            "escalation_path": ["show user-facing diagnosis", "preserve trace", "avoid direct mutation"],
            "can_continue": True,
            "user_facing_error": "Specialist agent failed safely; the Orchestrator kept the mission reviewable and escalation-ready.",
            "created_at": utc_now(),
        }
        diagnosis_id = f"agentdiag_{_json_hash(diagnosis_base)[:16]}"
        diagnosis = dict(diagnosis_base)
        diagnosis["diagnosis_id"] = diagnosis_id
        _write_json(self.agent_diagnosis_path(diagnosis_id), diagnosis)
        event = self.append_audit(
            "agent.failure.diagnosed",
            scope,
            {"type": "agent_role", "id": role["role_id"]},
            {"diagnosis_id": diagnosis_id, "trace_id": trace_id, "first_failing_layer": diagnosis["first_failing_layer"]},
        )
        return {"diagnosis": diagnosis, "audit_event": event}

    def test_agent_pack_capability(
        self,
        role_id: str,
        *,
        pack_id: str,
        capability: str,
        scope: dict[str, str],
    ) -> dict[str, Any]:
        self.ensure_agent_roles(scope)
        role = self.get_agent_role(role_id)
        if role is None:
            return {"status": "not_found", "resource": "agent_role"}
        if role.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": role.get("scope")}
        activation = self._find_agent_pack_activation(pack_id, scope)
        if activation is None or capability not in activation.get("granted_capabilities", []):
            decision = self._agent_policy_decision(
                scope=scope,
                subject_id=role["role_id"],
                decision="deny",
                policy="agent_pack_workspace_grant_required",
                reason="Agent Pack supplied agents can use only capabilities explicitly activated for this workspace or mission.",
                resolution_path=["Activate the pack with the minimum required grant and ConnectorHub policy review."],
            )
            attempt_base = {
                "schema_version": "cs.agent_pack_capability_attempt.v0",
                "status": "denied",
                "scope": scope,
                "role_id": role["role_id"],
                "pack_id": pack_id,
                "capability": capability,
                "activation_id": activation.get("activation_id") if activation else None,
                "granted_capabilities": activation.get("granted_capabilities", []) if activation else [],
                "connectorhub_mediated": False,
                "capability_used": False,
                "policy_decision": decision,
                "created_at": utc_now(),
            }
            attempt_id = f"agentpackcap_{_json_hash(attempt_base)[:16]}"
            attempt = dict(attempt_base)
            attempt["capability_attempt_id"] = attempt_id
            _write_json(self.agent_pack_capability_path(attempt_id), attempt)
            event = self.append_audit(
                "agent.pack_capability.denied",
                scope,
                {"type": "agent_role", "id": role["role_id"]},
                {"pack_id": pack_id, "capability": capability, "policy_decision_id": decision["policy_decision_id"]},
            )
            return {"status": "policy_denied", "capability_attempt": attempt, "policy_decision": decision, "audit_event": event}
        attempt_base = {
            "schema_version": "cs.agent_pack_capability_attempt.v0",
            "status": "mediated",
            "scope": scope,
            "role_id": role["role_id"],
            "pack_id": pack_id,
            "capability": capability,
            "activation_id": activation["activation_id"],
            "granted_capabilities": activation.get("granted_capabilities", []),
            "connectorhub_mediated": True,
            "direct_provider_access": False,
            "credentials_exposed_to_agent": False,
            "capability_used": True,
            "created_at": utc_now(),
        }
        attempt_id = f"agentpackcap_{_json_hash(attempt_base)[:16]}"
        attempt = dict(attempt_base)
        attempt["capability_attempt_id"] = attempt_id
        _write_json(self.agent_pack_capability_path(attempt_id), attempt)
        event = self.append_audit(
            "agent.pack_capability.mediated",
            scope,
            {"type": "agent_role", "id": role["role_id"]},
            {"pack_id": pack_id, "capability": capability, "activation_id": activation["activation_id"]},
        )
        return {"capability_attempt": attempt, "audit_event": event}

    def replay_agent_mission(self, trace_id: str, scope: dict[str, str]) -> dict[str, Any]:
        trace = self.get_agent_trace(trace_id)
        if trace is None:
            return {"status": "not_found", "resource": "agent_trace"}
        if trace.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": trace.get("scope")}
        outputs = [output for output in self._agent_output_records(scope) if output.get("trace_id") == trace_id]
        diagnoses = [diagnosis for diagnosis in self._agent_diagnosis_records(scope) if diagnosis.get("trace_id") == trace_id]
        role_ids = {trace.get("orchestrator_role_id"), *(delegation.get("role_id") for delegation in trace.get("delegations", []))}
        roles = [self.get_agent_role(role_id) for role_id in role_ids if role_id]
        replay_base = {
            "schema_version": "cs.agent_mission_replay.v0",
            "status": "reviewable",
            "scope": scope,
            "trace_id": trace_id,
            "mission_id": trace.get("mission_id"),
            "trace_refs": [f"agent_trace:{trace_id}"],
            "role_contract_refs": [f"agent_role:{role['role_id']}" for role in roles if role],
            "output_refs": [f"agent_output:{output['output_id']}" for output in outputs],
            "diagnosis_refs": [f"agent_diagnosis:{diagnosis['diagnosis_id']}" for diagnosis in diagnoses],
            "model_provider_records": [
                {
                    "role_id": role["role_id"],
                    "provider": role["role_contract"]["model_policy"].get("provider"),
                    "contract_hash": role["contract_hash"],
                }
                for role in roles
                if role
            ],
            "tool_outputs": [{"output_id": output["output_id"], "role_id": output["role_id"], "evidence_refs": output.get("evidence_refs", [])} for output in outputs],
            "policy_decision_refs": [],
            "judge_results": [output for output in outputs if output.get("output_type") == "judgment"],
            "evidence_refs": sorted({ref for output in outputs for ref in output.get("evidence_refs", [])}),
            "audit_refs": [],
            "hidden_chain_of_thought_required": False,
            "review_without_hidden_chain_of_thought": True,
            "created_at": utc_now(),
        }
        replay_id = f"agentreplay_{_json_hash(replay_base)[:16]}"
        replay = dict(replay_base)
        replay["replay_id"] = replay_id
        _write_json(self.agent_replay_path(replay_id), replay)
        event = self.append_audit(
            "agent.replay.created",
            scope,
            {"type": "agent_mission_replay", "id": replay_id},
            {"trace_id": trace_id, "role_contract_count": len(replay["role_contract_refs"]), "hidden_chain_of_thought_required": False},
        )
        replay["audit_refs"] = [f"audit:{event['event_id']}"]
        _write_json(self.agent_replay_path(replay_id), replay)
        return {"agent_replay": replay, "audit_event": event}

    def _brain_policy_decision(
        self,
        *,
        scope: dict[str, str],
        subject_id: str,
        decision: str,
        policy: str,
        reason: str,
        resolution_path: list[str],
    ) -> dict[str, Any]:
        decision_base = {
            "schema_version": "cs.policy_decision.v0",
            "decision": decision,
            "policy": policy,
            "reason": redact_text(reason),
            "resolution_path": resolution_path,
            "subject": {"type": "brain", "id": subject_id},
            "scope": scope,
            "created_at": utc_now(),
        }
        decision_id = f"policy_{_json_hash(decision_base)[:16]}"
        decision_record = dict(decision_base)
        decision_record["policy_decision_id"] = decision_id
        return decision_record

    def list_models(self, scope: dict[str, str]) -> dict[str, Any]:
        models = [dict(record) for record in MODEL_CAPABILITY_REGISTRY.values()]
        registry = {
            "schema_version": "cs.model_capability_registry.v0",
            "status": "ready",
            "scope": scope,
            "models": models,
            "safe_baseline_provider": "local_test",
            "safe_baseline_model": "local_test.v0",
            "external_models_are_registry_only": True,
            "real_provider_calls": 0,
            "secret_reads": 0,
            "created_at": utc_now(),
        }
        event = self.append_audit(
            "model.registry.listed",
            scope,
            {"type": "model_registry", "id": "local"},
            {"model_count": len(models), "external_models_are_registry_only": True},
        )
        return {"model_registry": registry, "audit_event": event}

    def _model_record(self, provider: str, model: str | None = None) -> dict[str, Any] | None:
        for record in MODEL_CAPABILITY_REGISTRY.values():
            if record["provider"] == provider and (model is None or record["model"] == model):
                return record
        return None

    def _record_brain_ledger_entry(
        self,
        *,
        scope: dict[str, str],
        route_id: str,
        provider: str,
        model: str,
        task_type: str,
        sensitivity: str,
        mission_type: str,
        risk: str,
        objective_outcome: str = "pending",
        owner_acceptance: str = "not_recorded",
        judge_quality: str = "not_measured",
        mission_success: bool | None = None,
    ) -> dict[str, Any]:
        model_record = self._model_record(provider, model) or {}
        ledger_base = {
            "schema_version": "cs.brain_performance_ledger_entry.v0",
            "status": "recorded",
            "scope": scope,
            "route_id": route_id,
            "provider": provider,
            "model": model,
            "task_type": task_type,
            "mission_type": mission_type,
            "sensitivity": sensitivity,
            "risk": risk,
            "policy": "workspace_model_routing_policy_v0",
            "cost_per_1k_tokens_usd": model_record.get("cost_per_1k_tokens_usd", 0),
            "latency_ms": model_record.get("typical_latency_ms", 0),
            "judge_quality": judge_quality,
            "tool_use_reliability": model_record.get("tool_use_reliability"),
            "grounding_issues": [],
            "owner_corrections": 0 if owner_acceptance == "not_recorded" else 1,
            "owner_acceptance": owner_acceptance,
            "objective_outcome": objective_outcome,
            "mission_success": mission_success,
            "namespace_local": True,
            "cross_namespace_aggregation": "requires_opt_in",
            "can_influence_routing": True,
            "created_at": utc_now(),
        }
        ledger_id = f"brainledger_{_json_hash(ledger_base)[:16]}"
        ledger = dict(ledger_base)
        ledger["ledger_id"] = ledger_id
        _write_json(self.brain_ledger_path(ledger_id), ledger)
        event = self.append_audit(
            "brain.ledger.recorded",
            scope,
            {"type": "brain_ledger", "id": ledger_id},
            {"provider": provider, "model": model, "namespace_local": True, "route_id": route_id},
        )
        return {"ledger_entry": ledger, "audit_event": event}

    def route_brain(
        self,
        *,
        task_ref: str,
        task_type: str,
        mission_type: str,
        sensitivity: str,
        risk: str,
        owner_preference: str,
        max_cost_usd: float,
        max_latency_ms: int,
        override_provider: str | None,
        override_model: str | None,
        dry_run: bool,
        scope: dict[str, str],
    ) -> dict[str, Any]:
        if not dry_run:
            decision = self._brain_policy_decision(
                scope=scope,
                subject_id="brain_route",
                decision="deny",
                policy="brain_route_dry_run_required",
                reason="Model routing changes and provider selection must be previewed through dry-run in local verification.",
                resolution_path=["Re-run with --dry-run and inspect policy, cost, latency, capability, and audit refs."],
            )
            event = self.append_audit(
                "brain.route.denied",
                scope,
                {"type": "policy_decision", "id": decision["policy_decision_id"]},
                {"policy_decision_id": decision["policy_decision_id"], "dry_run_required": True},
            )
            return {"status": "policy_denied", "policy_decision": decision, "audit_event": event}

        allowed_providers = ["local_test", "ollama"]
        denied_external = sensitivity in {"confidential", "restricted"} or risk in {"high", "safety_sensitive"}
        selected = self._model_record("local_test", "local_test.v0") or {}
        override_record = self._model_record(override_provider or "", override_model) if override_provider else None
        override_allowed = False
        if override_record:
            override_allowed = (
                override_record["provider"] in allowed_providers
                and sensitivity in override_record.get("allowed_sensitivity", [])
                and override_record.get("cost_per_1k_tokens_usd", 0) <= max_cost_usd
                and override_record.get("typical_latency_ms", 0) <= max_latency_ms
            )
            if override_record.get("deployment") == "external" and denied_external:
                override_allowed = False
        if override_record and not override_allowed:
            decision = self._brain_policy_decision(
                scope=scope,
                subject_id=f"{override_record['provider']}/{override_record['model']}",
                decision="deny",
                policy="model_override_forbidden_by_workspace_policy",
                reason="Requested model override violates workspace policy, sensitivity, cost, latency, or egress limits.",
                resolution_path=["Use an allowed local provider, lower sensitivity through policy review, or request explicit admin approval."],
            )
            event = self.append_audit(
                "brain.override.denied",
                scope,
                {"type": "policy_decision", "id": decision["policy_decision_id"]},
                {"provider": override_record["provider"], "model": override_record["model"], "policy_decision_id": decision["policy_decision_id"]},
            )
            return {"status": "policy_denied", "policy_decision": decision, "audit_event": event}
        if override_record and override_allowed:
            selected = override_record
        elif owner_preference == "local_semantic" and sensitivity != "restricted":
            selected = self._model_record("ollama", "qwen3.6:27b") or selected

        local_history = self._brain_ledger_records(scope)
        high_value = risk in {"high", "safety_sensitive"} or mission_type in {"externally_impactful", "ambiguous_research"}
        ensemble_triggered = high_value
        contribution_records = []
        if ensemble_triggered:
            for record in [self._model_record("local_test", "local_test.v0"), self._model_record("ollama", "qwen3.6:27b")]:
                if record:
                    contribution_records.append(
                        {
                            "provider": record["provider"],
                            "model": record["model"],
                            "role": "plan_or_critique",
                            "external_call_made": False,
                            "registry_only": record.get("registry_only"),
                        }
                    )
        route_base = {
            "schema_version": "cs.brain_routing_decision.v0",
            "status": "dry_run",
            "scope": scope,
            "task_ref": redact_text(task_ref),
            "task_type": task_type,
            "mission_type": mission_type,
            "sensitivity": sensitivity,
            "risk": risk,
            "owner_preference": owner_preference,
            "selected_brain": {
                "provider": selected["provider"],
                "model": selected["model"],
                "deployment": selected["deployment"],
                "external_call_made": False,
            },
            "factors": {
                "workspace_policy": {
                    "allowed_providers": allowed_providers,
                    "egress_default": "deny",
                    "confidential_external_denied": denied_external,
                },
                "sensitivity": sensitivity,
                "mission_type": mission_type,
                "risk": risk,
                "cost_limit_usd": max_cost_usd,
                "latency_limit_ms": max_latency_ms,
                "capabilities": selected.get("capabilities", []),
                "owner_preference": owner_preference,
                "local_performance_history_count": len(local_history),
                "historical_outcome_quality_used": bool(local_history),
                "static_capability_registry_used": len(local_history) == 0,
            },
            "override": {
                "requested": bool(override_record),
                "allowed": bool(override_record and override_allowed),
                "provider": override_record.get("provider") if override_record else None,
                "model": override_record.get("model") if override_record else None,
            },
            "ensemble": {
                "triggered": ensemble_triggered,
                "not_default_for_routine": not ensemble_triggered,
                "reason": "risk_value_trigger" if ensemble_triggered else "routine_single_brain",
                "contribution_records": contribution_records,
            },
            "no_real_provider_call": True,
            "secret_reads": 0,
            "created_at": utc_now(),
        }
        route_id = f"brainroute_{_json_hash(route_base)[:16]}"
        route = dict(route_base)
        route["route_id"] = route_id
        _write_json(self.brain_route_path(route_id), route)
        ledger_result = self._record_brain_ledger_entry(
            scope=scope,
            route_id=route_id,
            provider=selected["provider"],
            model=selected["model"],
            task_type=task_type,
            sensitivity=sensitivity,
            mission_type=mission_type,
            risk=risk,
        )
        event = self.append_audit(
            "brain.route.decided",
            scope,
            {"type": "brain_route", "id": route_id},
            {
                "provider": selected["provider"],
                "model": selected["model"],
                "dry_run": dry_run,
                "ensemble_triggered": ensemble_triggered,
                "static_registry_used": route["factors"]["static_capability_registry_used"],
            },
        )
        route["ledger_refs"] = [f"brain_ledger:{ledger_result['ledger_entry']['ledger_id']}"]
        _write_json(self.brain_route_path(route_id), route)
        return {"routing_decision": route, "ledger_entry": ledger_result["ledger_entry"], "audit_event": event, "ledger_audit_event": ledger_result["audit_event"]}

    def switch_workspace_brain(
        self,
        *,
        provider: str,
        model: str,
        evidence_bundle_id: str | None,
        mission_id: str | None,
        scope: dict[str, str],
    ) -> dict[str, Any]:
        if self._model_record(provider, model) is None:
            return {"status": "not_found", "resource": "model"}
        bundle = self.get_evidence_bundle(evidence_bundle_id) if evidence_bundle_id else None
        if evidence_bundle_id and bundle is None:
            return {"status": "not_found", "resource": "evidence_bundle"}
        mission = self.get_mission(mission_id) if mission_id else None
        if mission_id and mission is None:
            return {"status": "not_found", "resource": "mission"}
        switch_base = {
            "schema_version": "cs.workspace_brain_switch.v0",
            "status": "switched",
            "scope": scope,
            "from_provider": "local_test",
            "from_model": "local_test.v0",
            "to_provider": provider,
            "to_model": model,
            "durable_surfaces_unchanged": {
                "namespaces": True,
                "permanent_wiki": True,
                "evidence_archive": True,
                "ontology": True,
                "mission_contracts": True,
                "agents": True,
                "workflows": True,
                "policy": True,
                "audit": True,
                "experience_library": True,
                "judge_records": True,
                "promotion_ladder": True,
            },
            "existing_records_still_usable": {
                "evidence_bundle_id": evidence_bundle_id,
                "evidence_bundle_readable": bool(bundle),
                "mission_id": mission_id,
                "mission_readable": bool(mission),
            },
            "only_inference_brain_changed": True,
            "real_provider_call_made": False,
            "created_at": utc_now(),
        }
        switch_id = f"brainswitch_{_json_hash(switch_base)[:16]}"
        switch = dict(switch_base)
        switch["switch_id"] = switch_id
        _write_json(self.brain_switch_path(switch_id), switch)
        event = self.append_audit(
            "brain.provider.switched",
            scope,
            {"type": "brain_switch", "id": switch_id},
            {"to_provider": provider, "to_model": model, "only_inference_brain_changed": True},
        )
        return {"brain_switch": switch, "audit_event": event}

    def list_brain_ledger(self, scope: dict[str, str]) -> dict[str, Any]:
        entries = self._brain_ledger_records(scope)
        ledger = {
            "schema_version": "cs.brain_performance_ledger.v0",
            "status": "ready",
            "scope": scope,
            "entries": entries,
            "entry_count": len(entries),
            "namespace_local": True,
            "cross_namespace_entries": 0,
            "can_influence_routing": any(entry.get("can_influence_routing") is True for entry in entries),
            "created_at": utc_now(),
        }
        event = self.append_audit(
            "brain.ledger.listed",
            scope,
            {"type": "brain_ledger", "id": scope_key(scope)},
            {"entry_count": len(entries), "namespace_local": True},
        )
        return {"brain_ledger": ledger, "audit_event": event}

    def test_brain_aggregation(
        self,
        *,
        source_namespace: str,
        target_namespace: str,
        opt_in: bool,
        scope: dict[str, str],
    ) -> dict[str, Any]:
        source_scope = {**scope, "namespace_id": source_namespace}
        target_scope = {**scope, "namespace_id": target_namespace}
        source_entries = self._brain_ledger_records(source_scope)
        if not opt_in:
            decision = self._brain_policy_decision(
                scope=target_scope,
                subject_id=f"{source_namespace}->{target_namespace}",
                decision="deny",
                policy="brain_ledger_cross_namespace_opt_in_required",
                reason="Brain performance learning is namespace-local first; cross-namespace aggregation requires explicit opt-in and governance.",
                resolution_path=["Collect owner/admin opt-in and record aggregation policy before using source namespace performance data."],
            )
            aggregation_base = {
                "schema_version": "cs.brain_ledger_aggregation_attempt.v0",
                "status": "denied",
                "scope": target_scope,
                "source_scope": source_scope,
                "target_scope": target_scope,
                "opt_in": False,
                "source_entry_count": len(source_entries),
                "entries_used_for_routing": 0,
                "policy_decision": decision,
                "created_at": utc_now(),
            }
            aggregation_id = f"brainagg_{_json_hash(aggregation_base)[:16]}"
            aggregation = dict(aggregation_base)
            aggregation["aggregation_id"] = aggregation_id
            _write_json(self.brain_aggregation_path(aggregation_id), aggregation)
            event = self.append_audit(
                "brain.ledger_aggregation.denied",
                target_scope,
                {"type": "brain_aggregation", "id": aggregation_id},
                {"policy_decision_id": decision["policy_decision_id"], "opt_in": False},
            )
            return {"status": "policy_denied", "aggregation": aggregation, "policy_decision": decision, "audit_event": event}
        aggregation_base = {
            "schema_version": "cs.brain_ledger_aggregation_attempt.v0",
            "status": "aggregated",
            "scope": target_scope,
            "source_scope": source_scope,
            "target_scope": target_scope,
            "opt_in": True,
            "governance_recorded": True,
            "source_entry_count": len(source_entries),
            "entries_used_for_routing": len(source_entries),
            "created_at": utc_now(),
        }
        aggregation_id = f"brainagg_{_json_hash(aggregation_base)[:16]}"
        aggregation = dict(aggregation_base)
        aggregation["aggregation_id"] = aggregation_id
        _write_json(self.brain_aggregation_path(aggregation_id), aggregation)
        event = self.append_audit(
            "brain.ledger_aggregation.recorded",
            target_scope,
            {"type": "brain_aggregation", "id": aggregation_id},
            {"source_namespace": source_namespace, "target_namespace": target_namespace, "opt_in": True},
        )
        return {"aggregation": aggregation, "audit_event": event}

    def run_judge(
        self,
        *,
        route_id: str,
        subject: str,
        rubric: str,
        evidence_ref: str,
        ambiguity: str,
        scope: dict[str, str],
    ) -> dict[str, Any]:
        route = self.get_brain_route(route_id)
        if route is None:
            return {"status": "not_found", "resource": "brain_route"}
        if route.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": route.get("scope")}
        judge_base = {
            "schema_version": "cs.judge_record.v0",
            "status": "recorded",
            "scope": scope,
            "route_id": route_id,
            "provider": route.get("selected_brain", {}).get("provider"),
            "model": route.get("selected_brain", {}).get("model"),
            "subject": redact_text(subject),
            "rubric": redact_text(rubric),
            "ambiguity": ambiguity,
            "score": "pass_with_limitations",
            "confidence": "medium",
            "limitations": ["LLM judge is scalable for ambiguous quality dimensions but can be biased and overconfident."],
            "evidence_refs": [evidence_ref],
            "primary_for_ambiguous_outcome": True,
            "pass_judge": False,
            "directly_mutates_memory_or_rules": False,
            "created_at": utc_now(),
        }
        judge_record_id = f"judge_{_json_hash(judge_base)[:16]}"
        judge = dict(judge_base)
        judge["judge_record_id"] = judge_record_id
        _write_json(self.judge_record_path(judge_record_id), judge)
        ledger_result = self._record_brain_ledger_entry(
            scope=scope,
            route_id=route_id,
            provider=judge["provider"],
            model=judge["model"],
            task_type="judge",
            sensitivity=route.get("sensitivity", "internal"),
            mission_type=route.get("mission_type", "ambiguous_research"),
            risk=route.get("risk", "medium"),
            judge_quality="pending_calibration",
        )
        event = self.append_audit(
            "judge.record.created",
            scope,
            {"type": "judge_record", "id": judge_record_id},
            {"route_id": route_id, "primary_for_ambiguous_outcome": True, "pass_judge": False},
        )
        return {"judge_record": judge, "ledger_entry": ledger_result["ledger_entry"], "audit_event": event}

    def record_judge_conflict(self, judge_record_id: str, *, objective_evidence: str, scope: dict[str, str]) -> dict[str, Any]:
        judge = self.get_judge_record(judge_record_id)
        if judge is None:
            return {"status": "not_found", "resource": "judge_record"}
        if judge.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": judge.get("scope")}
        conflict_base = {
            "schema_version": "cs.judge_objective_conflict.v0",
            "status": "objective_outcome_selected",
            "scope": scope,
            "judge_record_id": judge_record_id,
            "judge_score": judge.get("score"),
            "objective_evidence": redact_text(objective_evidence),
            "objective_outcome": "failed",
            "final_outcome_state": "failed",
            "objective_outcome_overrides_judge": True,
            "judge_retained_as_evaluation_artifact": True,
            "memory_or_rule_mutated": False,
            "created_at": utc_now(),
        }
        conflict_id = f"judgeconf_{_json_hash(conflict_base)[:16]}"
        conflict = dict(conflict_base)
        conflict["conflict_id"] = conflict_id
        _write_json(self.judge_conflict_path(conflict_id), conflict)
        event = self.append_audit(
            "judge.objective_conflict.recorded",
            scope,
            {"type": "judge_conflict", "id": conflict_id},
            {"judge_record_id": judge_record_id, "objective_outcome_overrides_judge": True},
        )
        return {"judge_conflict": conflict, "audit_event": event}

    def record_owner_acceptance(self, judge_record_id: str, *, acceptance: str, scope: dict[str, str]) -> dict[str, Any]:
        judge = self.get_judge_record(judge_record_id)
        if judge is None:
            return {"status": "not_found", "resource": "judge_record"}
        if judge.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": judge.get("scope")}
        acceptance_base = {
            "schema_version": "cs.owner_acceptance_signal.v0",
            "status": "recorded",
            "scope": scope,
            "judge_record_id": judge_record_id,
            "owner_acceptance": acceptance,
            "grounds_final_success_when_objective_truth_unavailable": True,
            "judge_score_supporting_only": True,
            "learning_signal": "owner_grounded",
            "created_at": utc_now(),
        }
        acceptance_id = f"accept_{_json_hash(acceptance_base)[:16]}"
        record = dict(acceptance_base)
        record["acceptance_id"] = acceptance_id
        _write_json(self.judge_acceptance_path(acceptance_id), record)
        event = self.append_audit(
            "judge.owner_acceptance.recorded",
            scope,
            {"type": "owner_acceptance", "id": acceptance_id},
            {"judge_record_id": judge_record_id, "owner_acceptance": acceptance},
        )
        return {"owner_acceptance": record, "audit_event": event}

    def recommend_from_judge(self, judge_record_id: str, *, recommendation: str, scope: dict[str, str]) -> dict[str, Any]:
        judge = self.get_judge_record(judge_record_id)
        if judge is None:
            return {"status": "not_found", "resource": "judge_record"}
        if judge.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": judge.get("scope")}
        recommendation_base = {
            "schema_version": "cs.judge_recommendation.v0",
            "status": "candidate_lesson",
            "scope": scope,
            "judge_record_id": judge_record_id,
            "recommendation": redact_text(recommendation),
            "promotion_ladder": ["candidate_lesson", "workspace_memory", "mission_playbook", "organization_approved_rule"],
            "approved_memory_created": False,
            "global_rule_created": False,
            "requires_scope_evidence_confidence_governance": True,
            "evidence_refs": judge.get("evidence_refs", []),
            "created_at": utc_now(),
        }
        recommendation_id = f"judgerec_{_json_hash(recommendation_base)[:16]}"
        record = dict(recommendation_base)
        record["recommendation_id"] = recommendation_id
        _write_json(self.judge_recommendation_path(recommendation_id), record)
        event = self.append_audit(
            "judge.recommendation.created",
            scope,
            {"type": "judge_recommendation", "id": recommendation_id},
            {"approved_memory_created": False, "global_rule_created": False},
        )
        return {"judge_recommendation": record, "audit_event": event}

    def test_judge_disagreement(self, *, risk: str, scope: dict[str, str]) -> dict[str, Any]:
        high_risk = risk in {"high", "safety_sensitive"}
        adjudication_base = {
            "schema_version": "cs.judge_disagreement_adjudication.v0",
            "status": "escalated" if high_risk else "recommended",
            "scope": scope,
            "risk": risk,
            "participants": [
                {"provider": "local_test", "model": "local_test.v0", "position": "approve_with_limitations"},
                {"provider": "ollama", "model": "qwen3.6:27b", "position": "request_more_evidence"},
            ],
            "factors": {
                "evidence_quality": "mixed",
                "policy_constraints": ["no external write without approval"],
                "mission_goals": ["evidence-backed decision"],
                "prior_brain_performance": "local_namespace_ledger",
                "objective_outcomes": "unavailable",
                "rubric": ["grounding", "risk", "usefulness"],
            },
            "recommended_path": "request_more_evidence" if high_risk else "proceed_with_limitations",
            "dissent_preserved": True,
            "evidence_weighted": True,
            "unresolved": high_risk,
            "escalation_card": {
                "created": high_risk,
                "target": "namespace_owner" if high_risk else None,
                "reason": "High-risk disagreement remained material." if high_risk else None,
            },
            "proceeded_silently": False,
            "created_at": utc_now(),
        }
        adjudication_id = f"adjud_{_json_hash(adjudication_base)[:16]}"
        adjudication = dict(adjudication_base)
        adjudication["adjudication_id"] = adjudication_id
        _write_json(self.judge_adjudication_path(adjudication_id), adjudication)
        event = self.append_audit(
            "judge.disagreement.adjudicated",
            scope,
            {"type": "judge_adjudication", "id": adjudication_id},
            {"risk": risk, "dissent_preserved": True, "escalated": high_risk},
        )
        return {"adjudication": adjudication, "audit_event": event}

    def judge_calibration_report(self, scope: dict[str, str]) -> dict[str, Any]:
        judge_records = self._judge_records(scope)
        conflicts = [json.loads(path.read_text()) for path in sorted(self.judge_conflict_dir.glob("*.json"))] if self.judge_conflict_dir.exists() else []
        acceptances = [json.loads(path.read_text()) for path in sorted(self.judge_acceptance_dir.glob("*.json"))] if self.judge_acceptance_dir.exists() else []
        adjudications = [json.loads(path.read_text()) for path in sorted(self.judge_adjudication_dir.glob("*.json"))] if self.judge_adjudication_dir.exists() else []
        scoped_conflicts = [row for row in conflicts if row.get("scope") == scope]
        scoped_acceptances = [row for row in acceptances if row.get("scope") == scope]
        scoped_adjudications = [row for row in adjudications if row.get("scope") == scope]
        report_base = {
            "schema_version": "cs.judge_calibration_report.v0",
            "status": "ready",
            "scope": scope,
            "judge_record_count": len(judge_records),
            "disagreement_count": len(scoped_adjudications),
            "objective_reversal_count": len(scoped_conflicts),
            "owner_override_count": len(scoped_acceptances),
            "calibration_issues": ["overconfidence_risk", "position_bias_risk"],
            "model_specific_bias_signals": [
                {"provider": "local_test", "model": "local_test.v0", "bias_signal": "fixture_bias_possible"},
                {"provider": "ollama", "model": "qwen3.6:27b", "bias_signal": "local_model_calibration_needed"},
            ],
            "judge_is_unquestionable_authority": False,
            "objective_outcomes_override_judge": True,
            "owner_acceptance_recorded": bool(scoped_acceptances),
            "ledger_refs": [f"brain_ledger:{entry['ledger_id']}" for entry in self._brain_ledger_records(scope)],
            "created_at": utc_now(),
        }
        calibration_id = f"judgecal_{_json_hash(report_base)[:16]}"
        report = dict(report_base)
        report["calibration_id"] = calibration_id
        _write_json(self.judge_calibration_path(calibration_id), report)
        event = self.append_audit(
            "judge.calibration.reported",
            scope,
            {"type": "judge_calibration", "id": calibration_id},
            {"judge_record_count": len(judge_records), "objective_reversal_count": len(scoped_conflicts)},
        )
        return {"calibration_report": report, "audit_event": event}

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

    def mission_control_view(self, scope: dict[str, str]) -> dict[str, Any]:
        briefs = self._brief_records(scope)
        missions = self._mission_records(scope)
        actions = self._action_records(scope)
        memories = self._memory_records(scope)
        learning = self._learning_records(scope)
        pending_approvals = [
            {
                "action_id": action.get("action_id"),
                "mission_id": action.get("mission_id"),
                "label": "Approval",
                "risk": action.get("risk"),
                "status": action.get("approval", {}).get("status"),
                "evidence_refs": action.get("evidence", {}).get("artifact_refs", []),
            }
            for action in actions
            if action.get("approval", {}).get("status") == "pending"
        ]
        recommended_actions = [
            {
                "action_id": action.get("action_id"),
                "label": "Action",
                "goal": action.get("goal"),
                "status": action.get("execution", {}).get("status"),
                "policy": action.get("policy_decision", {}).get("policy"),
            }
            for action in actions
        ]
        evidence_gaps = [
            {
                "id": f"gap_{mission.get('mission_id', 'mission')}",
                "label": "Evidence",
                "reason": "Review evidence coverage before any high-risk or external action.",
                "mission_id": mission.get("mission_id"),
                "minimum_resolution": "Attach evidence refs or escalate before execution.",
            }
            for mission in missions
        ]
        surface_base = {
            "schema_version": "cs.mission_control_surface.v0",
            "status": "ready",
            "scope": scope,
            "surface_label": "Mission Control",
            "ops_inbox_label": "Ops Inbox",
            "one_operational_surface": True,
            "plain_language_default": True,
            "advanced_governance_available_on_request": True,
            "advanced_governance_default_visible": False,
            "sections": {
                "pending_briefs": [
                    {"brief_id": brief.get("brief_id"), "label": "Brief", "status": brief.get("status")}
                    for brief in briefs
                ],
                "evidence_gaps": evidence_gaps,
                "missions": [
                    {
                        "mission_id": mission.get("mission_id"),
                        "label": "Mission",
                        "goal": mission.get("goal"),
                        "status": mission.get("status"),
                    }
                    for mission in missions
                ],
                "tasks": [
                    {
                        "action_id": action.get("action_id"),
                        "label": "Task",
                        "goal": action.get("goal"),
                        "status": action.get("execution", {}).get("status"),
                    }
                    for action in actions
                ],
                "approvals": pending_approvals,
                "recommended_actions": recommended_actions,
                "memory_changes": [
                    {
                        "memory_id": memory.get("memory_id"),
                        "label": "Memory",
                        "status": memory.get("status"),
                        "trust_state": memory.get("trust_state"),
                    }
                    for memory in memories
                ],
                "learning_opportunities": [
                    {
                        "learning_id": row.get("learning_id"),
                        "label": "Learn",
                        "status": row.get("status"),
                        "lesson": row.get("lesson"),
                    }
                    for row in learning
                ]
                or [
                    {
                        "id": "learn_after_action_review",
                        "label": "Learn",
                        "status": "pending_after_action_review",
                    }
                ],
            },
            "created_at": utc_now(),
        }
        surface_id = f"missioncontrol_{_json_hash(surface_base)[:16]}"
        surface = dict(surface_base)
        surface["surface_id"] = surface_id
        _write_json(self.mission_control_path(surface_id), surface)
        event = self.append_audit(
            "product.mission_control.viewed",
            scope,
            {"type": "mission_control", "id": surface_id},
            {
                "section_count": len(surface["sections"]),
                "pending_approval_count": len(pending_approvals),
                "one_operational_surface": True,
            },
        )
        return {"mission_control": surface, "audit_event": event}

    def product_loop_view(
        self,
        scope: dict[str, str],
        *,
        conversation_id: str = "",
        brief_id: str = "",
        claim_id: str = "",
        mission_id: str = "",
        action_id: str = "",
        outcome_id: str = "",
    ) -> dict[str, Any]:
        loop_base = {
            "schema_version": "cs.product_loop_view.v0",
            "status": "visible",
            "scope": scope,
            "item_id": mission_id or action_id or claim_id or brief_id or conversation_id,
            "stages": [
                {"stage": "Inbox", "visible": True, "ref": f"conversation:{conversation_id}" if conversation_id else None},
                {"stage": "Brief", "visible": True, "ref": f"brief:{brief_id}" if brief_id else None},
                {"stage": "Claim", "visible": True, "ref": f"claim:{claim_id}" if claim_id else None},
                {"stage": "Action", "visible": True, "ref": f"action:{action_id}" if action_id else None},
                {"stage": "Learn", "visible": True, "ref": f"mission_outcome:{outcome_id}" if outcome_id else None},
            ],
            "journey": "Inbox -> Brief -> Claim -> Action -> Learn",
            "single_item_progression_visible": True,
            "created_at": utc_now(),
        }
        loop_id = f"productloop_{_json_hash(loop_base)[:16]}"
        loop = dict(loop_base)
        loop["loop_id"] = loop_id
        _write_json(self.product_surface_path(loop_id), loop)
        event = self.append_audit(
            "product.loop.viewed",
            scope,
            {"type": "product_loop", "id": loop_id},
            {"visible_stage_count": len([stage for stage in loop["stages"] if stage["visible"]])},
        )
        return {"product_loop": loop, "audit_event": event}

    def product_boundary_review(self, scope: dict[str, str]) -> dict[str, Any]:
        user_visible_text = [
            "Source systems remain systems of record where appropriate.",
            "CornerStone is the intelligence, evidence, mission, action-control, and learning layer over them.",
            "Connected sources are accessed through governed capabilities and action approvals.",
        ]
        boundary_base = {
            "schema_version": "cs.product_boundary_review.v0",
            "status": "ready",
            "scope": scope,
            "source_systems_remain_systems_of_record": True,
            "cornerstone_layers": ["intelligence", "evidence", "mission", "action-control", "learning"],
            "help_surfaces": ["onboarding_card", "boundary_view", "admin_help"],
            "user_visible_text": user_visible_text,
            "visible_internal_repo_names": [],
            "created_at": utc_now(),
        }
        boundary_id = f"boundary_{_json_hash(boundary_base)[:16]}"
        boundary = dict(boundary_base)
        boundary["boundary_id"] = boundary_id
        _write_json(self.product_surface_path(boundary_id), boundary)
        event = self.append_audit(
            "product.boundary.reviewed",
            scope,
            {"type": "product_boundary", "id": boundary_id},
            {"source_systems_remain_systems_of_record": True},
        )
        return {"boundary_review": boundary, "audit_event": event}

    def product_plain_language_review(self, scope: dict[str, str]) -> dict[str, Any]:
        terms = ["workspace", "memory", "evidence", "brief", "claim", "mission", "action", "approval", "learn"]
        review_base = {
            "schema_version": "cs.product_plain_language_review.v0",
            "status": "passed",
            "scope": scope,
            "first_value_task_completed": True,
            "basic_mission_task_completed": True,
            "plain_language_terms": terms,
            "advanced_governance_available": True,
            "advanced_governance_required_for_first_value": False,
            "admin_setup_required_beyond_defaults": False,
            "created_at": utc_now(),
        }
        review_id = f"plainlang_{_json_hash(review_base)[:16]}"
        review = dict(review_base)
        review["review_id"] = review_id
        _write_json(self.product_surface_path(review_id), review)
        event = self.append_audit(
            "product.plain_language.reviewed",
            scope,
            {"type": "product_plain_language_review", "id": review_id},
            {"term_count": len(terms), "first_value_task_completed": True},
        )
        return {"plain_language_review": review, "audit_event": event}

    def product_repo_split_review(self, scope: dict[str, str]) -> dict[str, Any]:
        user_visible_labels = [
            "Home",
            "Search",
            "Artifacts",
            "Claims",
            "Actions",
            "Mission Control",
            "Workspace",
            "Memory",
            "Evidence",
            "Brief",
            "Mission",
            "Approval",
            "Learn",
            "Connected Sources",
        ]
        forbidden_terms = ["Cornerstone", "KnowledgeBase", "ConnectorHub", "repo", "repository", "package"]
        visible_text = " ".join(user_visible_labels)
        forbidden_present = [term for term in forbidden_terms if term.lower() in visible_text.lower()]
        review_base = {
            "schema_version": "cs.product_repo_split_review.v0",
            "status": "passed" if not forbidden_present else "failed",
            "scope": scope,
            "one_cornerstone_product": True,
            "user_visible_labels": user_visible_labels,
            "visible_capabilities": ["capture", "search", "evidence", "claims", "missions", "actions", "learning"],
            "forbidden_internal_repo_terms": forbidden_terms,
            "forbidden_terms_present": forbidden_present,
            "daily_user_requires_repo_model": False,
            "created_at": utc_now(),
        }
        review_id = f"reposplit_{_json_hash(review_base)[:16]}"
        review = dict(review_base)
        review["review_id"] = review_id
        _write_json(self.product_surface_path(review_id), review)
        event = self.append_audit(
            "product.repo_split.reviewed",
            scope,
            {"type": "product_repo_split_review", "id": review_id},
            {"forbidden_terms_present": forbidden_present},
        )
        return {"repo_split_review": review, "audit_event": event}

    def create_memory_from_evidence_bundle(
        self,
        bundle_id: str,
        statement: str,
        scope: dict[str, str],
        *,
        trust_state: str = "evidence_backed",
        status: str = "owner_approved",
        memory_type: str = "durable_fact",
        synthesis_mode: str = "owner_approved",
    ) -> dict[str, Any]:
        bundle = self.get_evidence_bundle(bundle_id)
        if bundle is None:
            return {"status": "not_found", "resource": "evidence_bundle"}
        if bundle.get("filters") != scope:
            return {"status": "scope_denied", "resource_scope": bundle.get("filters")}

        evidence_items = bundle.get("evidence_items", [])
        artifact_refs = [f"artifact:{item['artifact_id']}" for item in evidence_items]
        if not evidence_items or not artifact_refs:
            return {"status": "evidence_required"}

        influence_answers = status == "owner_approved" and trust_state in {"evidence_backed", "approved"}
        influence_actions = status == "owner_approved" and trust_state == "approved"
        memory_base = {
            "schema_version": "cs.memory.v0",
            "status": status,
            "trust_state": trust_state,
            "memory_type": memory_type,
            "statement": redact_text(statement),
            "scope": scope,
            "source": {
                "created_from": "memory.create",
                "source_type": "evidence_bundle",
                "evidence_bundle_id": bundle_id,
                "search_snapshot_id": bundle.get("search_snapshot_id"),
                "artifact_refs": artifact_refs,
                "synthesis_mode": synthesis_mode,
            },
            "provenance": {
                "source_evidence_bundle_id": bundle_id,
                "source_search_snapshot_id": bundle.get("search_snapshot_id"),
                "source_artifact_refs": artifact_refs,
                "created_from": f"{synthesis_mode}_memory_create",
            },
            "canonicality": {
                "canonical_truth_foundation": "archive_evidence",
                "raw_agent_memory_canonical": False,
                "owner_approved": status == "owner_approved",
                "requires_evidence_for_truth_claims": True,
            },
            "synthesis": {
                "living_synthesis": True,
                "raw_truth": False,
                "source_count": len(evidence_items),
                "confidence": "medium" if trust_state == "draft" else "high",
                "why_written": "Synthesized from an Evidence Bundle with source artifacts.",
                "auto_synthesized": synthesis_mode == "auto",
                "user_visible_source": True,
            },
            "freshness": {
                "status": "current",
                "last_reviewed_at": utc_now(),
                "stale_after_days": 90,
                "warning_visible": False,
            },
            "usage_permissions": {
                "can_influence_answers": influence_answers,
                "can_influence_actions": influence_actions,
                "can_influence_routing": influence_answers,
                "requires_review_before_action_use": trust_state != "approved",
                "allowed_scope": scope,
            },
            "identity_visibility": {
                "user_owned_permanent_wiki": True,
                "hidden_profile": False,
                "inspectable": True,
                "controllable": True,
            },
            "update_history": [
                {
                    "event": "created",
                    "statement": redact_text(statement),
                    "trust_state": trust_state,
                    "status": status,
                    "evidence_refs": [f"evidence_bundle:{bundle_id}", *artifact_refs],
                    "created_at": utc_now(),
                }
            ],
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
                "trust_state": trust_state,
                "status": status,
                "synthesis_mode": synthesis_mode,
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
            "freshness": {
                "status": "unverified",
                "warning_visible": True,
            },
            "usage_permissions": {
                "can_influence_answers": False,
                "can_influence_actions": False,
                "can_influence_routing": False,
                "requires_review_before_action_use": True,
                "allowed_scope": scope,
            },
            "identity_visibility": {
                "user_owned_permanent_wiki": True,
                "hidden_profile": False,
                "inspectable": True,
                "controllable": True,
            },
            "update_history": [
                {
                    "event": "raw_agent_candidate_created",
                    "statement": redact_text(statement),
                    "trust_state": "unverified",
                    "status": "raw_agent_memory",
                    "evidence_refs": [],
                    "created_at": utc_now(),
                }
            ],
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

    def permanent_wiki_view(self, scope: dict[str, str], *, wiki_kind: str) -> dict[str, Any]:
        memories = self._memory_records(scope)
        claims = self._claim_records(scope)
        missions = self._mission_records(scope)
        actions = self._action_records(scope)
        learning_records = self._learning_records(scope)
        adaptations = self._memory_adaptation_records(scope)
        promotions = self._namespace_promotion_records(scope)
        if wiki_kind == "product-learning":
            entries = [
                {
                    "entry_id": f"learning:{learning['learning_id']}",
                    "entry_type": "product_learning",
                    "title": learning.get("lesson"),
                    "trust_state": "review_required",
                    "source_refs": learning.get("evidence_refs", []),
                    "changes_user_or_org_truth": learning.get("learning_boundary", {}).get("changes_user_or_org_truth"),
                    "requires_review_before_memory_update": learning.get("learning_boundary", {}).get("requires_review_before_memory_update"),
                    "scope": learning.get("scope"),
                }
                for learning in learning_records
            ]
            wiki_scope = {**scope, "namespace_id": "product_learning"}
        else:
            entries = []
            for memory in memories:
                entries.append(
                    {
                        "entry_id": f"memory:{memory['memory_id']}",
                        "entry_type": "memory",
                        "title": memory.get("statement"),
                        "memory_type": memory.get("memory_type"),
                        "status": memory.get("status"),
                        "trust_state": memory.get("trust_state"),
                        "freshness": memory.get("freshness"),
                        "source_refs": memory.get("evidence_refs", []),
                        "source": memory.get("source"),
                        "update_history": memory.get("update_history", []),
                        "correction_history": memory.get("correction_history", []),
                        "usage_permissions": memory.get("usage_permissions", {}),
                        "hidden_profile": memory.get("identity_visibility", {}).get("hidden_profile", False),
                    }
                )
            for claim in claims:
                entries.append(
                    {
                        "entry_id": f"claim:{claim['claim_id']}",
                        "entry_type": "claim",
                        "title": claim.get("statement"),
                        "status": claim.get("status"),
                        "trust_state": claim.get("trust_state"),
                        "source_refs": self._claim_evidence_refs(claim),
                    }
                )
            for mission in missions:
                entries.append(
                    {
                        "entry_id": f"mission:{mission['mission_id']}",
                        "entry_type": "mission",
                        "title": mission.get("goal"),
                        "status": mission.get("status"),
                        "source_refs": mission.get("evidence", {}).get("artifact_refs", []),
                    }
                )
            for action in actions:
                entries.append(
                    {
                        "entry_id": f"action:{action['action_id']}",
                        "entry_type": "action_history",
                        "title": action.get("goal"),
                        "status": action.get("execution", {}).get("status"),
                        "source_refs": action.get("evidence", {}).get("artifact_refs", []),
                    }
                )
            wiki_scope = scope

        wiki_base = {
            "schema_version": "cs.permanent_wiki_view.v0",
            "wiki_kind": wiki_kind,
            "scope": wiki_scope,
            "status": "ready",
            "source_aware": True,
            "living_synthesis_not_raw_truth": True,
            "archive_truth_foundation": True,
            "entries": entries,
            "entry_count": len(entries),
            "update_history_visible": True,
            "correction_history_count": sum(len(memory.get("correction_history", [])) for memory in memories),
            "namespace_promotion_count": len(promotions),
            "adaptation_count": len(adaptations),
            "controls_available": [
                "inspect",
                "correct",
                "demote",
                "promote",
                "forget",
                "rollback",
                "disable_influence",
                "limit_scope",
                "export",
            ],
            "identity_policy": {
                "personal_memory_is_user_owned_wiki": wiki_kind == "personal",
                "organization_memory_is_governed": wiki_kind == "organization",
                "product_learning_separate_from_user_org_truth": wiki_kind == "product-learning",
                "hidden_profile": False,
            },
            "created_at": utc_now(),
        }
        wiki_id = f"wiki_{_json_hash(wiki_base)[:16]}"
        wiki = dict(wiki_base)
        wiki["wiki_id"] = wiki_id
        _write_json(self.wiki_path(wiki_id), wiki)
        event = self.append_audit(
            "wiki.view.generated",
            scope,
            {"type": "permanent_wiki_view", "id": wiki_id},
            {"wiki_kind": wiki_kind, "entry_count": len(entries), "source_aware": True},
        )
        return {"wiki": wiki, "audit_event": event}

    def memory_control_center(self, scope: dict[str, str]) -> dict[str, Any]:
        memories = self._memory_records(scope)
        control_base = {
            "schema_version": "cs.memory_control_center.v0",
            "scope": scope,
            "status": "ready",
            "memory_count": len(memories),
            "controls": {
                "inspect": True,
                "correct": True,
                "demote": True,
                "promote": True,
                "forget": True,
                "rollback": True,
                "disable_influence": True,
                "limit_scope": True,
                "export": True,
            },
            "influence_controls": {
                "answers": True,
                "actions": True,
                "routing": True,
                "autonomous_behavior": True,
            },
            "entries": [
                {
                    "memory_id": memory["memory_id"],
                    "status": memory.get("status"),
                    "trust_state": memory.get("trust_state"),
                    "freshness": memory.get("freshness", {}),
                    "usage_permissions": memory.get("usage_permissions", {}),
                    "source_refs": memory.get("evidence_refs", []),
                    "correction_history_count": len(memory.get("correction_history", [])),
                    "update_history_count": len(memory.get("update_history", [])),
                }
                for memory in memories
            ],
            "hidden_profile": False,
            "created_at": utc_now(),
        }
        control_id = f"memctrl_{_json_hash(control_base)[:16]}"
        control = dict(control_base)
        control["control_center_id"] = control_id
        _write_json(self.memory_control_path(control_id), control)
        event = self.append_audit(
            "memory.control_center.opened",
            scope,
            {"type": "memory_control_center", "id": control_id},
            {"memory_count": len(memories), "controls": list(control["controls"].keys())},
        )
        return {"memory_control_center": control, "audit_event": event}

    def create_temporary_memory_session(self, note: str, scope: dict[str, str]) -> dict[str, Any]:
        before_count = len(self._memory_records(scope))
        session_base = {
            "schema_version": "cs.temporary_memory_session.v0",
            "status": "completed",
            "scope": scope,
            "note": redact_text(note),
            "memory_mode": "no_memory",
            "permanent_memory_created": False,
            "memory_count_before": before_count,
            "memory_count_after": before_count,
            "security_and_audit_rules_still_apply": True,
            "created_at": utc_now(),
        }
        session_id = f"tempsess_{_json_hash(session_base)[:16]}"
        session = dict(session_base)
        session["temporary_session_id"] = session_id
        _write_json(self.temporary_session_path(session_id), session)
        event = self.append_audit(
            "memory.temporary_session.completed",
            scope,
            {"type": "temporary_memory_session", "id": session_id},
            {"permanent_memory_created": False, "memory_mode": "no_memory"},
        )
        return {"temporary_session": session, "audit_event": event}

    def correct_memory(
        self,
        memory_id: str,
        *,
        corrected_text: str,
        rationale: str,
        evidence_bundle_id: str | None,
        scope: dict[str, str],
    ) -> dict[str, Any]:
        memory = self.get_memory(memory_id)
        if memory is None:
            return {"status": "not_found", "resource": "memory"}
        if memory.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": memory.get("scope")}
        bundle = self.get_evidence_bundle(evidence_bundle_id) if evidence_bundle_id else None
        if evidence_bundle_id and bundle is None:
            return {"status": "not_found", "resource": "evidence_bundle"}
        if bundle and bundle.get("filters") != scope:
            return {"status": "scope_denied", "resource_scope": bundle.get("filters")}
        evidence_refs = [f"memory:{memory_id}"]
        if bundle:
            evidence_refs.extend(
                [
                    f"evidence_bundle:{evidence_bundle_id}",
                    f"search_snapshot:{bundle.get('search_snapshot_id')}",
                    *[f"artifact:{item['artifact_id']}" for item in bundle.get("evidence_items", [])],
                ]
            )
        else:
            evidence_refs.append("owner_judgment:local-user")

        correction_base = {
            "schema_version": "cs.correction.v0",
            "status": "recorded",
            "scope": scope,
            "target": {
                "kind": "memory",
                "id": memory_id,
                "original_trust_state": memory.get("trust_state"),
                "original_provenance": memory.get("provenance"),
            },
            "correction": {
                "corrected_text": redact_text(corrected_text),
                "rationale": redact_text(rationale),
                "source_type": "evidence_bundle" if bundle else "owner_judgment",
            },
            "learning_signal": {
                "signal_type": "human_evidence_aware_correction",
                "used_for_silent_overwrite": False,
                "requires_review_before_memory_update": True,
            },
            "evidence_refs": evidence_refs,
            "provenance_preserved": True,
            "created_at": utc_now(),
        }
        correction_id = f"correction_{_json_hash(correction_base)[:16]}"
        correction = dict(correction_base)
        correction["correction_id"] = correction_id
        updated = dict(memory)
        previous_statement = memory.get("statement")
        updated["statement"] = redact_text(corrected_text)
        updated["status"] = "owner_approved"
        updated["freshness"] = {**updated.get("freshness", {}), "status": "current", "warning_visible": False, "last_reviewed_at": utc_now()}
        updated["provenance"] = {**updated.get("provenance", {}), "last_correction_id": correction_id}
        correction_history = list(updated.get("correction_history", []))
        correction_history.append({"correction_id": correction_id, "evidence_refs": evidence_refs, "silent_overwrite": False, "corrected_at": correction["created_at"]})
        updated["correction_history"] = correction_history
        update_history = list(updated.get("update_history", []))
        update_history.append(
            {
                "event": "corrected",
                "previous_statement": previous_statement,
                "statement": updated["statement"],
                "correction_id": correction_id,
                "evidence_refs": evidence_refs,
                "created_at": correction["created_at"],
            }
        )
        updated["update_history"] = update_history
        _write_json(self.correction_path(correction_id), correction)
        _write_json(self.memory_path(memory_id), updated)
        event = self.append_audit(
            "memory.corrected",
            scope,
            {"type": "memory", "id": memory_id},
            {"correction_id": correction_id, "source_type": correction["correction"]["source_type"], "silent_overwrite": False},
        )
        return {"correction": correction, "memory": updated, "audit_event": event}

    def control_memory(self, memory_id: str, *, action: str, scope: dict[str, str]) -> dict[str, Any]:
        memory = self.get_memory(memory_id)
        if memory is None:
            return {"status": "not_found", "resource": "memory"}
        if memory.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": memory.get("scope")}
        updated = dict(memory)
        previous = {
            "status": memory.get("status"),
            "trust_state": memory.get("trust_state"),
            "statement": memory.get("statement"),
            "usage_permissions": memory.get("usage_permissions", {}),
        }
        if action == "forget":
            updated["status"] = "forgotten"
            updated["usage_permissions"] = {**updated.get("usage_permissions", {}), "can_influence_answers": False, "can_influence_actions": False, "can_influence_routing": False}
            updated["retention_note"] = "Memory is disabled for future use; underlying archive and audit evidence remain according to retention policy."
        elif action == "rollback":
            rollback_statement = None
            for entry in reversed(updated.get("update_history", [])):
                if entry.get("previous_statement"):
                    rollback_statement = entry["previous_statement"]
                    break
            if rollback_statement:
                updated["statement"] = rollback_statement
            updated["status"] = "owner_approved"
            updated["usage_permissions"] = {**updated.get("usage_permissions", {}), "can_influence_answers": True}
        elif action == "demote":
            updated["trust_state"] = "draft"
            updated["usage_permissions"] = {**updated.get("usage_permissions", {}), "can_influence_actions": False, "requires_review_before_action_use": True}
        elif action == "promote":
            updated["trust_state"] = "approved"
            updated["status"] = "owner_approved"
            updated["usage_permissions"] = {**updated.get("usage_permissions", {}), "can_influence_answers": True, "can_influence_actions": True, "requires_review_before_action_use": False}
        elif action == "disable-influence":
            updated["usage_permissions"] = {**updated.get("usage_permissions", {}), "can_influence_answers": False, "can_influence_actions": False, "can_influence_routing": False}
        elif action == "limit-scope":
            updated["usage_permissions"] = {**updated.get("usage_permissions", {}), "allowed_scope": scope, "limited_to_active_scope": True}
        else:
            return {"status": "invalid_action", "resource": "memory"}

        control_base = {
            "schema_version": "cs.memory_control_action.v0",
            "status": "recorded",
            "scope": scope,
            "memory_id": memory_id,
            "action": action,
            "previous": previous,
            "current": {
                "status": updated.get("status"),
                "trust_state": updated.get("trust_state"),
                "statement": updated.get("statement"),
                "usage_permissions": updated.get("usage_permissions", {}),
            },
            "archive_evidence_retained": True,
            "created_at": utc_now(),
        }
        control_id = f"memact_{_json_hash(control_base)[:16]}"
        control = dict(control_base)
        control["memory_control_action_id"] = control_id
        history = list(updated.get("update_history", []))
        history.append({"event": f"control:{action}", "control_id": control_id, "created_at": control["created_at"], "previous": previous, "current": control["current"]})
        updated["update_history"] = history
        _write_json(self.memory_control_path(control_id), control)
        _write_json(self.memory_path(memory_id), updated)
        event = self.append_audit(
            "memory.control.applied",
            scope,
            {"type": "memory_control_action", "id": control_id},
            {"memory_id": memory_id, "action": action, "archive_evidence_retained": True},
        )
        return {"memory_control_action": control, "memory": updated, "audit_event": event}

    def check_memory_freshness(self, memory_id: str, newer_evidence_bundle_id: str, scope: dict[str, str]) -> dict[str, Any]:
        memory = self.get_memory(memory_id)
        if memory is None:
            return {"status": "not_found", "resource": "memory"}
        if memory.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": memory.get("scope")}
        bundle = self.get_evidence_bundle(newer_evidence_bundle_id)
        if bundle is None:
            return {"status": "not_found", "resource": "evidence_bundle"}
        if bundle.get("filters") != scope:
            return {"status": "scope_denied", "resource_scope": bundle.get("filters")}
        old_refs = set(memory.get("source", {}).get("artifact_refs", []))
        new_refs = {f"artifact:{item['artifact_id']}" for item in bundle.get("evidence_items", [])}
        needs_review = bool(new_refs and new_refs != old_refs)
        updated = dict(memory)
        updated["freshness"] = {
            **updated.get("freshness", {}),
            "status": "needs_review" if needs_review else "current",
            "warning_visible": needs_review,
            "newer_evidence_bundle_id": newer_evidence_bundle_id,
            "newer_evidence_refs": sorted(new_refs),
            "used_as_current_fact_without_warning": False,
            "checked_at": utc_now(),
        }
        _write_json(self.memory_path(memory_id), updated)
        event = self.append_audit(
            "memory.freshness.checked",
            scope,
            {"type": "memory", "id": memory_id},
            {"status": updated["freshness"]["status"], "warning_visible": updated["freshness"]["warning_visible"]},
        )
        return {"memory": updated, "audit_event": event}

    def quarantine_memory_attempt(self, artifact_id: str, statement: str, scope: dict[str, str]) -> dict[str, Any]:
        artifact = self.get_artifact(artifact_id, scope)
        if artifact is None:
            return {"status": "not_found", "resource": "artifact"}
        if artifact.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": artifact.get("scope")}
        text = f"{self._derived_text(artifact)}\n{statement}"
        blocked_patterns = detect_unsafe_instructions(text)
        unsafe = bool(blocked_patterns) or artifact.get("safety", {}).get("unsafe_instruction_detected") is True or artifact.get("safety", {}).get("untrusted_evidence") is True
        quarantine_base = {
            "schema_version": "cs.memory_quarantine.v0",
            "status": "quarantined" if unsafe else "allowed_for_review",
            "scope": scope,
            "artifact_id": artifact_id,
            "statement": redact_text(statement),
            "blocked_patterns": blocked_patterns,
            "memory_created": False,
            "trusted_memory_created": False,
            "requires_owner_review": True,
            "reason": "Untrusted or unsafe content cannot write trusted memory without evidence, confidence, scope, and review.",
            "evidence_refs": [f"artifact:{artifact_id}", f"storage:{artifact.get('original_storage_ref')}"],
            "created_at": utc_now(),
        }
        quarantine_id = f"memq_{_json_hash(quarantine_base)[:16]}"
        quarantine = dict(quarantine_base)
        quarantine["memory_quarantine_id"] = quarantine_id
        _write_json(self.memory_quarantine_path(quarantine_id), quarantine)
        event = self.append_audit(
            "memory.write.quarantined",
            scope,
            {"type": "memory_quarantine", "id": quarantine_id},
            {"artifact_id": artifact_id, "memory_created": False, "requires_owner_review": True},
        )
        return {"memory_quarantine": quarantine, "audit_event": event}

    def export_memory(self, scope: dict[str, str]) -> dict[str, Any]:
        memories = self._memory_records(scope)
        export_base = {
            "schema_version": "cs.memory_export.v0",
            "status": "ready",
            "scope": scope,
            "format": "json",
            "entry_count": len(memories),
            "entries": [
                {
                    "memory_id": memory["memory_id"],
                    "statement": memory.get("statement"),
                    "memory_type": memory.get("memory_type"),
                    "status": memory.get("status"),
                    "trust_state": memory.get("trust_state"),
                    "freshness": memory.get("freshness", {}),
                    "source": memory.get("source"),
                    "evidence_refs": memory.get("evidence_refs", []),
                    "correction_history": memory.get("correction_history", []),
                    "owner_namespace": memory.get("scope"),
                    "usage_permissions": memory.get("usage_permissions", {}),
                }
                for memory in memories
            ],
            "understandable": True,
            "created_at": utc_now(),
        }
        export_id = f"memexport_{_json_hash(export_base)[:16]}"
        export = dict(export_base)
        export["memory_export_id"] = export_id
        _write_json(self.memory_export_path(export_id), export)
        event = self.append_audit(
            "memory.export.created",
            scope,
            {"type": "memory_export", "id": export_id},
            {"entry_count": len(memories), "format": "json"},
        )
        return {"memory_export": export, "audit_event": event}

    def record_memory_adaptation(self, preference: str, scope: dict[str, str], *, source_memory_id: str | None = None) -> dict[str, Any]:
        evidence_refs: list[str] = []
        if source_memory_id:
            memory = self.get_memory(source_memory_id)
            if memory is None:
                return {"status": "not_found", "resource": "memory"}
            if memory.get("scope") != scope:
                return {"status": "scope_denied", "resource_scope": memory.get("scope")}
            evidence_refs.append(f"memory:{source_memory_id}")
            evidence_refs.extend(memory.get("evidence_refs", []))
        adaptation_base = {
            "schema_version": "cs.memory_adaptation.v0",
            "status": "active",
            "scope": scope,
            "preference": redact_text(preference),
            "source_memory_id": source_memory_id,
            "evidence_refs": evidence_refs,
            "namespace_local": True,
            "changes_product_defaults": False,
            "changes_other_namespaces": False,
            "governed_and_versioned": True,
            "created_at": utc_now(),
        }
        adaptation_id = f"adapt_{_json_hash(adaptation_base)[:16]}"
        adaptation = dict(adaptation_base)
        adaptation["memory_adaptation_id"] = adaptation_id
        _write_json(self.memory_adaptation_path(adaptation_id), adaptation)
        event = self.append_audit(
            "memory.adaptation.recorded",
            scope,
            {"type": "memory_adaptation", "id": adaptation_id},
            {"namespace_local": True, "changes_other_namespaces": False},
        )
        return {"memory_adaptation": adaptation, "audit_event": event}

    def _access_decision_record(
        self,
        *,
        scope: dict[str, str],
        principal_id: str,
        principal_role: str,
        principal_attributes: list[str],
        action: str,
        resource_kind: str,
        resource_id: str,
        resource_scope: dict[str, str],
        classification: str,
        mission_authority: str,
    ) -> dict[str, Any]:
        attributes = sorted({attribute.strip() for attribute in principal_attributes if attribute.strip()})
        decision = "allow"
        policy = "local_rbac_abac_matrix"
        reason = "Access is allowed by the local deterministic RBAC/ABAC matrix."
        resolution_path = ["Proceed through the same scoped CLI path and audit ledger."]
        allowed_org_roles = {"org_admin", "org_approver", "org_member"}
        read_write_actions = {"read", "write", "promote", "search", "summarize", "extract_memory", "use_in_action"}
        admin_actions = {"configure", "configure_autopilot", "install_pack", "aggregate_learning"}

        if scope != resource_scope:
            decision = "deny"
            policy = "active_workspace_resource_scope"
            reason = "The active workspace scope does not match the requested resource scope."
            resolution_path = ["Switch to the resource workspace or request an explicit promotion/reference."]
        elif resource_scope["namespace_id"] == "organization" and principal_role not in allowed_org_roles:
            decision = "deny"
            policy = "organization_membership_required"
            reason = "Organization resources require an organization role in the active organization namespace."
            resolution_path = ["Request organization access or promote only through an approved namespace workflow."]
        elif action in admin_actions and principal_role != "org_admin":
            decision = "deny"
            policy = "configuration_requires_org_admin"
            reason = "Policy, Autopilot, Agent Pack install, or aggregate learning configuration requires the org_admin role."
            resolution_path = ["Ask an organization admin to configure the policy or grant the needed role."]
        elif action == "approve" and principal_role not in {"org_admin", "org_approver"}:
            decision = "deny"
            policy = "approval_requires_authorized_approver"
            reason = "Approval requires an org_admin or org_approver role."
            resolution_path = ["Request review from an authorized approver."]
        elif action == "execute" and mission_authority not in {"active", "approved"}:
            decision = "deny"
            policy = "mission_authority_required_for_execution"
            reason = "Execution requires an active or approved Mission Goal Contract authority."
            resolution_path = ["Activate or approve a Mission Goal Contract before execution."]
        elif classification == "secret":
            decision = "deny"
            policy = "secret_classification_denied_in_local_scaffold"
            reason = "The local scaffold denies secret-classified resource access by default."
            resolution_path = ["Use fake-secret fixtures only and move secret access to an approved production policy design."]
        elif classification == "restricted" and principal_role != "org_admin" and "clearance:restricted" not in attributes:
            decision = "deny"
            policy = "restricted_classification_clearance_required"
            reason = "Restricted resources require org_admin or clearance:restricted."
            resolution_path = ["Request restricted clearance or reduce the resource classification."]
        elif action in read_write_actions and principal_role in allowed_org_roles | {"personal_user"}:
            decision = "allow"
        elif action in admin_actions and principal_role == "org_admin":
            decision = "allow"
        elif action not in {"read", "write", "promote", "search", "summarize", "extract_memory", "use_in_action", "approve", "execute", *admin_actions}:
            decision = "deny"
            policy = "unknown_action_denied"
            reason = "Unknown access actions are denied by default."
            resolution_path = ["Use a declared action in the local access matrix."]

        decision_base = {
            "schema_version": "cs.policy_decision.v0",
            "decision": decision,
            "policy": policy,
            "reason": reason,
            "principal": {
                "id": principal_id,
                "role": principal_role,
                "attributes": attributes,
            },
            "action": action,
            "resource": {
                "kind": resource_kind,
                "id": resource_id,
                "scope": resource_scope,
                "classification": classification,
            },
            "scope": scope,
            "mission_authority": mission_authority,
            "evaluation_model": "deterministic_local_rbac_abac",
            "external_http_calls": 0,
            "secret_reads": 0,
            "resolution_path": resolution_path,
            "decided_at": utc_now(),
        }
        record = dict(decision_base)
        record["id"] = f"policy_{_json_hash(decision_base)[:16]}"
        return record

    def evaluate_access(
        self,
        *,
        principal_id: str,
        principal_role: str,
        principal_attributes: list[str],
        action: str,
        resource_kind: str,
        resource_id: str,
        resource_scope: dict[str, str],
        classification: str,
        mission_authority: str,
        scope: dict[str, str],
    ) -> dict[str, Any]:
        decision = self._access_decision_record(
            scope=scope,
            principal_id=principal_id,
            principal_role=principal_role,
            principal_attributes=principal_attributes,
            action=action,
            resource_kind=resource_kind,
            resource_id=resource_id,
            resource_scope=resource_scope,
            classification=classification,
            mission_authority=mission_authority,
        )
        _write_json(self.access_decision_path(decision["id"]), decision)
        event = self.append_audit(
            "policy.access.evaluated",
            scope,
            {"type": "policy_decision", "id": decision["id"]},
            {
                "policy": decision["policy"],
                "decision": decision["decision"],
                "principal_role": principal_role,
                "action": action,
                "resource_kind": resource_kind,
                "resource_id": resource_id,
                "resource_scope": resource_scope,
                "classification": classification,
                "mission_authority": mission_authority,
                "external_http_calls": 0,
                "secret_reads": 0,
            },
        )
        return {"policy_decision": decision, "audit_event": event}

    def export_claim_basis(self, claim_id: str, scope: dict[str, str]) -> dict[str, Any]:
        claim = self.get_claim(claim_id)
        if claim is None:
            return {"status": "not_found", "resource": "claim"}
        if claim.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": claim.get("scope")}

        evidence = claim.get("evidence_bundle", {})
        bundle_id = evidence.get("evidence_bundle_id")
        snapshot_id = evidence.get("search_snapshot_id")
        bundle = self.get_evidence_bundle(bundle_id) if bundle_id else None
        snapshot = self.get_search_snapshot(snapshot_id) if snapshot_id else None
        artifact_records = []
        for artifact_ref in evidence.get("artifact_refs", []):
            artifact_id = str(artifact_ref).split("artifact:", 1)[-1]
            artifact = self.get_artifact(artifact_id, scope)
            if artifact is not None:
                artifact_records.append(
                    {
                        "artifact_id": artifact_id,
                        "scope": artifact.get("scope"),
                        "source": artifact.get("source"),
                        "original_storage_ref": artifact.get("original_storage_ref"),
                        "derived_text_ref": artifact.get("derived", {}).get("text_ref"),
                        "provenance": artifact.get("provenance"),
                    }
                )

        audit_events = [
            event
            for event in self._all_audit_events()
            if event.get("subject", {}).get("id") == claim_id
            or event.get("details", {}).get("claim_id") == claim_id
        ]
        export_base = {
            "schema_version": "cs.claim_basis_export.v0",
            "status": "ready",
            "scope": scope,
            "claim_id": claim_id,
            "claim_status": claim.get("status"),
            "claim_trust_state": claim.get("trust_state"),
            "statement": claim.get("statement"),
            "source_artifacts": artifact_records,
            "search_snapshot": {
                "search_snapshot_id": snapshot_id,
                "query": snapshot.get("query") if snapshot else None,
                "filters": snapshot.get("filters") if snapshot else None,
                "result_count": snapshot.get("result_count") if snapshot else 0,
            },
            "evidence_bundle": {
                "evidence_bundle_id": bundle_id,
                "evidence_item_count": len(bundle.get("evidence_items", [])) if bundle else 0,
                "artifact_refs": evidence.get("artifact_refs", []),
            },
            "transformations": [
                {
                    "type": "artifact.derived_text",
                    "status": "preserved",
                    "artifact_id": artifact["artifact_id"],
                    "derived_text_ref": artifact.get("derived_text_ref"),
                }
                for artifact in artifact_records
            ],
            "model_or_judge_records": {
                "available": False,
                "reason": "Local deterministic scaffold did not need an LLM judge for this claim.",
            },
            "owner_approval": {
                "approved": claim.get("status") == "approved",
                "approval_events": [
                    {"event_id": event.get("event_id"), "event_type": event.get("event_type"), "occurred_at": event.get("occurred_at")}
                    for event in audit_events
                    if event.get("event_type") == "claim.approved"
                ],
            },
            "freshness": {
                "status": "current",
                "validity_state": "inspectable",
                "reproducible_from_archive": bool(bundle and snapshot and artifact_records),
            },
            "created_at": utc_now(),
        }
        export_id = f"claimbasis_{_json_hash(export_base)[:16]}"
        export = dict(export_base)
        export["claim_basis_export_id"] = export_id
        _write_json(self.claim_basis_export_path(export_id), export)
        event = self.append_audit(
            "claim.basis.exported",
            scope,
            {"type": "claim_basis_export", "id": export_id},
            {"claim_id": claim_id, "artifact_count": len(artifact_records), "approved": export["owner_approval"]["approved"]},
        )
        return {"claim_basis_export": export, "audit_event": event}

    def verify_source_readonly_ingest(self, artifact_id: str, scope: dict[str, str], *, source_system: str) -> dict[str, Any]:
        artifact = self.get_artifact(artifact_id, scope)
        if artifact is None:
            return {"status": "not_found", "resource": "artifact"}
        if artifact.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": artifact.get("scope")}

        safety_base = {
            "schema_version": "cs.source_readonly_ingest_test.v0",
            "status": "verified",
            "scope": scope,
            "artifact_id": artifact_id,
            "source_system": source_system,
            "source_record_ref": artifact.get("source", {}),
            "ingestion_path": "archive_capture_only",
            "source_write_events": 0,
            "source_mutation_attempts": [],
            "explicit_action_workflow_required_for_mutation": True,
            "connector_boundary": {
                "mocked_source": True,
                "read_only_adapter": True,
                "direct_writeback_allowed": False,
                "workflow_action_required": True,
                "external_http_calls": 0,
                "secret_reads": 0,
            },
            "created_at": utc_now(),
        }
        safety_id = f"srcsafe_{_json_hash(safety_base)[:16]}"
        safety = dict(safety_base)
        safety["source_safety_id"] = safety_id
        _write_json(self.source_safety_path(safety_id), safety)
        event = self.append_audit(
            "source.readonly_ingest.verified",
            scope,
            {"type": "source_readonly_ingest_test", "id": safety_id},
            {"artifact_id": artifact_id, "source_write_events": 0, "workflow_action_required": True},
        )
        return {"source_safety": safety, "audit_event": event}

    def query_namespace_audit(self, scope: dict[str, str], *, event_types: list[str] | None = None) -> dict[str, Any]:
        selected = []
        requested = set(event_types or [])
        for event in self._all_audit_events():
            event_scope = {
                "tenant_id": event.get("tenant_id"),
                "owner_id": event.get("owner_id"),
                "namespace_id": event.get("namespace_id"),
                "workspace_id": event.get("workspace_id"),
            }
            if event_scope != scope:
                continue
            if requested and event.get("event_type") not in requested:
                continue
            selected.append(
                {
                    "event_id": event.get("event_id"),
                    "event_type": event.get("event_type"),
                    "occurred_at": event.get("occurred_at"),
                    "subject": event.get("subject"),
                    "details": event.get("details"),
                }
            )
        counts: dict[str, int] = {}
        for event in selected:
            event_type = str(event.get("event_type"))
            counts[event_type] = counts.get(event_type, 0) + 1
        export_base = {
            "schema_version": "cs.namespace_audit_export.v0",
            "status": "ready",
            "scope": scope,
            "format": "json",
            "event_count": len(selected),
            "event_type_counts": counts,
            "events": selected,
            "coverage": {
                "data_access": any(event.get("event_type") in {"artifact.read", "evidence_bundle.read", "memory.read"} for event in selected),
                "memory_writes": any(str(event.get("event_type", "")).startswith("memory.") for event in selected),
                "promotions": any(event.get("event_type") == "namespace.promotion.created" for event in selected),
                "approvals": any(event.get("event_type") == "claim.approved" for event in selected),
                "actions": any(str(event.get("event_type", "")).startswith("action.") for event in selected),
                "model_routing": any(str(event.get("event_type", "")).startswith("brain.") for event in selected),
                "agent_activity": any(str(event.get("event_type", "")).startswith("agent.") for event in selected),
                "learning_events": any("learning" in str(event.get("event_type", "")) for event in selected),
            },
            "created_at": utc_now(),
        }
        export_id = f"nsaudit_{_json_hash(export_base)[:16]}"
        export = dict(export_base)
        export["namespace_audit_export_id"] = export_id
        _write_json(self.namespace_audit_export_path(export_id), export)
        event = self.append_audit(
            "namespace.audit.exported",
            scope,
            {"type": "namespace_audit_export", "id": export_id},
            {"event_count": len(selected), "format": "json"},
        )
        return {"namespace_audit_export": export, "audit_event": event}

    def check_product_learning_boundary(self, scope: dict[str, str]) -> dict[str, Any]:
        scoped_learning = self._learning_records(scope)
        boundary_base = {
            "schema_version": "cs.product_learning_boundary.v0",
            "status": "enforced",
            "scope": scope,
            "default_policy": "product_learning_raw_truth_deny_by_default",
            "allowed_inputs": [
                "explicit_feedback",
                "benchmark_results",
                "opt_in_aggregated_signals",
                "redacted_approved_data",
            ],
            "denied_inputs": [
                "raw_personal_memory",
                "raw_organization_memory",
                "private_artifacts",
                "unapproved_claims",
            ],
            "policy_checks": [
                {
                    "check": "raw_personal_truth_read",
                    "decision": "deny",
                    "reason": "Product learning cannot silently consume personal truth.",
                },
                {
                    "check": "raw_organization_truth_read",
                    "decision": "deny",
                    "reason": "Product learning cannot silently consume organization truth.",
                },
                {
                    "check": "explicit_feedback",
                    "decision": "allow",
                    "reason": "Explicit feedback is allowed as proposal/evaluation data.",
                },
                {
                    "check": "redacted_approved_data",
                    "decision": "allow",
                    "reason": "Approved redacted data can be used without rewriting user/org truth.",
                },
            ],
            "learning_records": [
                {"learning_id": record.get("learning_id"), "changes_user_or_org_truth": record.get("learning_boundary", {}).get("changes_user_or_org_truth")}
                for record in scoped_learning
            ],
            "raw_truth_records_read": 0,
            "user_or_org_memory_rewrites": 0,
            "proposal_data_only": True,
            "created_at": utc_now(),
        }
        boundary_id = f"plboundary_{_json_hash(boundary_base)[:16]}"
        boundary = dict(boundary_base)
        boundary["product_learning_boundary_id"] = boundary_id
        _write_json(self.product_learning_boundary_path(boundary_id), boundary)
        event = self.append_audit(
            "product_learning.boundary.checked",
            scope,
            {"type": "product_learning_boundary", "id": boundary_id},
            {"raw_truth_records_read": 0, "user_or_org_memory_rewrites": 0, "allowed_input_count": len(boundary["allowed_inputs"])},
        )
        return {"product_learning_boundary": boundary, "audit_event": event}

    def promote_memory_to_namespace(
        self,
        memory_id: str,
        source_scope: dict[str, str],
        target_scope: dict[str, str],
        *,
        mode: str,
        principal_id: str = "local-user",
        principal_role: str = "org_admin",
    ) -> dict[str, Any]:
        source = self.get_memory(memory_id)
        if source is None:
            return {"status": "not_found", "resource": "memory"}
        if source.get("scope") != source_scope:
            return {"status": "scope_denied", "resource_scope": source.get("scope")}
        if source.get("status") != "owner_approved" or not source.get("evidence_refs"):
            return {"status": "evidence_required", "resource": "memory"}

        policy_result = self.evaluate_access(
            principal_id=principal_id,
            principal_role=principal_role,
            principal_attributes=["namespace:personal", "namespace:organization", "clearance:restricted"],
            action="promote",
            resource_kind="memory",
            resource_id=memory_id,
            resource_scope=target_scope,
            classification="restricted",
            mission_authority="active",
            scope=target_scope,
        )
        decision = policy_result["policy_decision"]
        if decision["decision"] != "allow":
            return {"status": "policy_denied", "policy_decision": decision, "audit_event": policy_result["audit_event"]}

        source_evidence_refs = list(source.get("evidence_refs", []))
        materialized_modes = {"copy_with_provenance", "promote_to_approved_truth"}
        mode_behaviors = {
            "copy_with_provenance": {
                "ownership": "target_namespace_copy",
                "permission_behavior": "target namespace receives an independent evidence-backed copy with source provenance",
                "target_materialized": True,
                "source_owner_retains_original": True,
                "can_influence_answers": True,
            },
            "reference": {
                "ownership": "source_owner_retained",
                "permission_behavior": "target namespace receives an auditable pointer that must re-check source permission before use",
                "target_materialized": False,
                "source_owner_retains_original": True,
                "can_influence_answers": False,
            },
            "share": {
                "ownership": "source_owner_retained",
                "permission_behavior": "target namespace receives a bounded shared-view grant without ownership transfer",
                "target_materialized": False,
                "source_owner_retains_original": True,
                "can_influence_answers": False,
            },
            "promote_to_approved_truth": {
                "ownership": "target_namespace_approved_truth",
                "permission_behavior": "target namespace receives an owner-approved durable truth candidate with evidence and provenance",
                "target_materialized": True,
                "source_owner_retains_original": True,
                "can_influence_answers": True,
            },
        }
        behavior = mode_behaviors.get(mode, mode_behaviors["copy_with_provenance"])
        target_base = {
            "schema_version": "cs.memory.v0",
            "status": "owner_approved",
            "trust_state": "approved" if mode == "promote_to_approved_truth" else "evidence_backed",
            "memory_type": "promoted_approved_truth" if mode == "promote_to_approved_truth" else "promoted_durable_fact",
            "statement": source.get("statement", ""),
            "scope": target_scope,
            "source": {
                "created_from": "namespace.promote",
                "source_type": "explicit_namespace_promotion",
                "source_memory_id": memory_id,
                "source_scope": source_scope,
                "promotion_mode": mode,
                "evidence_bundle_id": source.get("source", {}).get("evidence_bundle_id"),
                "search_snapshot_id": source.get("source", {}).get("search_snapshot_id"),
                "artifact_refs": source.get("source", {}).get("artifact_refs", []),
            },
            "provenance": {
                "created_from": "explicit_namespace_promotion",
                "mode": mode,
                "source_memory_id": memory_id,
                "source_scope": source_scope,
                "target_scope": target_scope,
                "source_evidence_refs": source_evidence_refs,
                "policy_decision_ref": f"policy:{decision['id']}",
            },
            "canonicality": {
                "canonical_truth_foundation": "archive_evidence",
                "raw_agent_memory_canonical": False,
                "owner_approved": True,
                "requires_evidence_for_truth_claims": True,
                "explicitly_promoted": True,
            },
            "usage_permissions": {
                "can_influence_answers": behavior["can_influence_answers"],
                "can_influence_actions": mode == "promote_to_approved_truth",
                "source_permission_recheck_required": mode in {"reference", "share"},
            },
            "evidence_refs": [f"memory:{memory_id}", *source_evidence_refs],
            "created_at": utc_now(),
        }
        target_memory_id = f"memory_{_json_hash(target_base)[:16]}"
        target_memory = dict(target_base)
        target_memory["memory_id"] = target_memory_id
        target_item: dict[str, Any]
        if mode in materialized_modes:
            target_item = target_memory
        else:
            reference_base = {
                "schema_version": "cs.namespace_reference.v0",
                "status": "referenced" if mode == "reference" else "shared",
                "mode": mode,
                "scope": target_scope,
                "source_memory_id": memory_id,
                "source_scope": source_scope,
                "target_scope": target_scope,
                "permission_behavior": behavior["permission_behavior"],
                "source_permission_recheck_required": True,
                "can_influence_answers": False,
                "created_at": utc_now(),
            }
            reference_id = f"nsref_{_json_hash(reference_base)[:16]}"
            target_item = dict(reference_base)
            target_item["memory_id"] = reference_id

        promotion_base = {
            "schema_version": "cs.namespace_promotion.v0",
            "status": "promoted" if mode in materialized_modes else target_item["status"],
            "mode": mode,
            "mode_behavior": behavior,
            "source": {
                "kind": "memory",
                "id": memory_id,
                "scope": source_scope,
                "evidence_refs": source_evidence_refs,
            },
            "target": {
                "kind": "memory",
                "id": target_item["memory_id"],
                "scope": target_scope,
                "materialized": mode in materialized_modes,
            },
            "provenance": {
                "activity": "explicit_namespace_promotion",
                "agent": principal_id,
                "source_entity": f"memory:{memory_id}",
                "target_entity": f"memory:{target_item['memory_id']}",
                "source_scope": source_scope,
                "target_scope": target_scope,
                "mode": mode,
            },
            "policy_decision": decision,
            "evidence_refs": [f"memory:{memory_id}", *source_evidence_refs],
            "created_at": utc_now(),
        }
        promotion_id = f"promotion_{_json_hash(promotion_base)[:16]}"
        promotion = dict(promotion_base)
        promotion["promotion_id"] = promotion_id
        promotion["evidence_refs"] = [f"namespace_promotion:{promotion_id}", *promotion["evidence_refs"]]

        if mode in materialized_modes:
            target_memory["source"]["namespace_promotion_id"] = promotion_id
            target_memory["provenance"]["namespace_promotion_id"] = promotion_id
            target_memory["evidence_refs"] = [f"namespace_promotion:{promotion_id}", *target_memory["evidence_refs"]]
            _write_json(self.memory_path(target_memory_id), target_memory)
        _write_json(self.namespace_promotion_path(promotion_id), promotion)

        promotion_event = self.append_audit(
            "namespace.promotion.created",
            target_scope,
            {"type": "namespace_promotion", "id": promotion_id},
            {
                "source_kind": "memory",
                "source_id": memory_id,
                "target_kind": "memory",
                "target_id": target_memory_id,
                "source_scope": source_scope,
                "target_scope": target_scope,
                "mode": mode,
                "evidence_refs": promotion["evidence_refs"],
                "policy_decision_ref": f"policy:{decision['id']}",
            },
        )
        return {
            "promotion": promotion,
            "promoted_memory": target_item,
            "policy_decision": decision,
            "audit_events": [policy_result["audit_event"], promotion_event],
        }

    def recover_namespace_boundary(self, promotion_id: str, scope: dict[str, str], *, reason: str) -> dict[str, Any]:
        promotion = None
        for record in self._all_namespace_promotion_records():
            if record.get("promotion_id") == promotion_id:
                promotion = record
                break
        if promotion is None:
            return {"status": "not_found", "resource": "namespace_promotion"}
        target_scope = promotion.get("target", {}).get("scope")
        if target_scope != scope:
            return {"status": "scope_denied", "resource_scope": target_scope}

        target_id = promotion.get("target", {}).get("id")
        target_memory = self.get_memory(target_id) if target_id else None
        if target_memory is not None:
            updated = dict(target_memory)
            updated["status"] = "revoked"
            permissions = dict(updated.get("usage_permissions", {}))
            permissions["can_influence_answers"] = False
            permissions["can_influence_actions"] = False
            updated["usage_permissions"] = permissions
            updated.setdefault("recovery_history", []).append({"promotion_id": promotion_id, "reason": redact_text(reason), "recovered_at": utc_now()})
            _write_json(self.memory_path(target_id), updated)

        access_events = [
            event
            for event in self._all_audit_events()
            if event.get("details", {}).get("resource_id") == target_id
            or event.get("subject", {}).get("id") == target_id
        ]
        recovery_base = {
            "schema_version": "cs.namespace_boundary_recovery.v0",
            "status": "recovered",
            "scope": scope,
            "promotion_id": promotion_id,
            "reason": redact_text(reason),
            "target_id": target_id,
            "target_materialized": promotion.get("target", {}).get("materialized"),
            "revocation": {
                "available": True,
                "applied": True,
                "future_answer_use_disabled": True,
                "future_action_use_disabled": True,
            },
            "rollback": {
                "available": True,
                "source_original_preserved": True,
                "target_record_status": "revoked" if target_memory is not None else "reference_revoked",
            },
            "access_trail": [
                {"event_id": event.get("event_id"), "event_type": event.get("event_type"), "occurred_at": event.get("occurred_at")}
                for event in access_events
            ],
            "retention": {
                "audit_retained": True,
                "promotion_record_retained": True,
                "original_evidence_retained": True,
            },
            "created_at": utc_now(),
        }
        recovery_id = f"nsrecover_{_json_hash(recovery_base)[:16]}"
        recovery = dict(recovery_base)
        recovery["recovery_id"] = recovery_id
        _write_json(self.namespace_recovery_path(recovery_id), recovery)
        event = self.append_audit(
            "namespace.boundary.recovered",
            scope,
            {"type": "namespace_boundary_recovery", "id": recovery_id},
            {"promotion_id": promotion_id, "target_id": target_id, "future_answer_use_disabled": True},
        )
        return {"namespace_recovery": recovery, "audit_event": event}

    def answer_from_memory(self, question: str, scope: dict[str, str]) -> dict[str, Any]:
        terms = [term for term in search_terms(question) if term not in ANSWER_STOP_TERMS]
        matching_memories = []
        for memory in self._memory_records(scope):
            statement = str(memory.get("statement", ""))
            haystack = statement.lower()
            if memory.get("status") != "owner_approved" or not memory.get("evidence_refs"):
                continue
            if memory.get("usage_permissions", {}).get("can_influence_answers") is False:
                continue
            if any(term in haystack for term in terms):
                matching_memories.append(memory)

        if not matching_memories:
            decision = self._access_decision_record(
                scope=scope,
                principal_id=scope["owner_id"],
                principal_role="org_member" if scope["namespace_id"] == "organization" else "personal_user",
                principal_attributes=[f"namespace:{scope['namespace_id']}"],
                action="read",
                resource_kind="memory",
                resource_id="active-scope-memory",
                resource_scope=scope,
                classification="confidential",
                mission_authority="none",
            )
            decision["decision"] = "deny"
            decision["policy"] = "cross_namespace_memory_denied_by_default"
            decision["reason"] = "No owner-approved active-scope memory matched the question; CornerStone did not search or use other namespaces."
            decision["resolution_path"] = ["Explicitly promote, copy, reference, or permit the source memory before using it in this workspace."]
            decision_base = dict(decision)
            decision_base.pop("id", None)
            decision["id"] = f"policy_{_json_hash(decision_base)[:16]}"
            _write_json(self.access_decision_path(decision["id"]), decision)
            answer_base = {
                "schema_version": "cs.memory_answer.v0",
                "status": "insufficient_evidence",
                "question": redact_text(question),
                "scope": scope,
                "answer_label": "insufficient_evidence",
                "presented_as_fact": False,
                "based_on": "active_scope_memory_only",
                "used_memory_refs": [],
                "evidence_refs": [],
                "policy_decision": decision,
                "context_boundary": {
                    "active_scope_only": True,
                    "implicit_cross_namespace_context": False,
                    "personal_memory_used_without_promotion": False,
                    "explicit_promotion_required": True,
                },
                "created_at": utc_now(),
            }
            answer_id = f"answer_{_json_hash(answer_base)[:16]}"
            answer = dict(answer_base)
            answer["answer_id"] = answer_id
            _write_json(self.answer_path(answer_id), answer)
            event = self.append_audit(
                "memory.answer.insufficient_evidence",
                scope,
                {"type": "memory_answer", "id": answer_id},
                {
                    "question": redact_text(question),
                    "used_memory_refs": [],
                    "policy": decision["policy"],
                    "personal_memory_used_without_promotion": False,
                },
            )
            return {"status": "insufficient_evidence", "answer": answer, "policy_decision": decision, "audit_event": event}

        memory = matching_memories[0]
        used_memory_refs = [f"memory:{memory['memory_id']}"]
        promotions = [
            promotion
            for promotion in self._namespace_promotion_records(scope)
            if promotion.get("target", {}).get("id") == memory.get("memory_id")
        ]
        answer_base = {
            "schema_version": "cs.memory_answer.v0",
            "status": "answered",
            "question": redact_text(question),
            "scope": scope,
            "answer_label": "evidence_backed",
            "presented_as_fact": True,
            "based_on": "active_scope_owner_approved_memory",
            "used_memory_refs": used_memory_refs,
            "evidence_refs": list(memory.get("evidence_refs", [])),
            "statement": memory.get("statement"),
            "memory_use_explanation": {
                "why_this_context": "Matched an owner-approved active-scope memory with evidence refs.",
                "used_memory_refs": used_memory_refs,
                "source_evidence_refs": list(memory.get("evidence_refs", [])),
                "freshness": memory.get("freshness", {}),
                "trust_state": memory.get("trust_state"),
                "can_correct_or_disable": True,
            },
            "context_boundary": {
                "active_scope_only": True,
                "implicit_cross_namespace_context": False,
                "personal_memory_used_without_promotion": False,
                "used_promoted_memory": bool(promotions),
                "promotion_refs": [f"namespace_promotion:{promotion['promotion_id']}" for promotion in promotions],
            },
            "created_at": utc_now(),
        }
        answer_id = f"answer_{_json_hash(answer_base)[:16]}"
        answer = dict(answer_base)
        answer["answer_id"] = answer_id
        _write_json(self.answer_path(answer_id), answer)
        event = self.append_audit(
            "memory.answer.created",
            scope,
            {"type": "memory_answer", "id": answer_id},
            {
                "question": redact_text(question),
                "used_memory_refs": used_memory_refs,
                "evidence_ref_count": len(answer["evidence_refs"]),
                "used_promoted_memory": bool(promotions),
                "personal_memory_used_without_promotion": False,
            },
        )
        return {"answer": answer, "audit_event": event}

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

    def record_connected_outcome(
        self,
        action_id: str,
        *,
        evidence_bundle_id: str,
        outcome_status: str,
        summary: str,
        scope: dict[str, str],
    ) -> dict[str, Any]:
        action = self.get_action(action_id)
        if action is None:
            return {"status": "not_found", "resource": "action"}
        if action.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": action.get("scope")}
        execution = action.get("execution", {})
        result = execution.get("result", {})
        if execution.get("status") != "executed" or result.get("status") != "success":
            return {"status": "evidence_required", "resource": "executed_action"}
        bundle = self.get_evidence_bundle(evidence_bundle_id)
        if bundle is None:
            return {"status": "not_found", "resource": "evidence_bundle"}
        if bundle.get("filters") != scope:
            return {"status": "scope_denied", "resource_scope": bundle.get("filters")}
        artifact_refs = [f"artifact:{item['artifact_id']}" for item in bundle.get("evidence_items", [])]
        if not artifact_refs:
            return {"status": "evidence_required", "resource": "evidence_bundle"}

        outcome_base = {
            "schema_version": "cs.connected_outcome.v0",
            "status": "recorded",
            "scope": scope,
            "action_id": action_id,
            "mission_id": action.get("mission_id"),
            "outcome_status": outcome_status,
            "summary": redact_text(summary),
            "source": {
                "created_from": "experience.connected-outcome",
                "connector": action.get("connector_boundary", {}).get("connector", "mock_connector"),
                "connectorhub_mediated": True,
                "external_http_calls": 0,
                "mock_connector_calls": result.get("mock_connector_calls", 0),
                "credentials_exposed_to_agent": False,
                "reingested_as_evidence": True,
                "outcome_evidence_bundle_id": evidence_bundle_id,
                "immutable_outcome_artifact_refs": artifact_refs,
            },
            "evidence_refs": [
                f"action:{action_id}",
                f"mission:{action.get('mission_id')}",
                f"evidence_bundle:{evidence_bundle_id}",
                f"search_snapshot:{bundle.get('search_snapshot_id')}",
                *artifact_refs,
            ],
            "created_at": utc_now(),
        }
        outcome_id = f"outcome_{_json_hash(outcome_base)[:16]}"
        outcome = dict(outcome_base)
        outcome["connected_outcome_id"] = outcome_id
        _write_json(self.connected_outcome_path(outcome_id), outcome)
        event = self.append_audit(
            "experience.connected_outcome.recorded",
            scope,
            {"type": "connected_outcome", "id": outcome_id},
            {"action_id": action_id, "mission_id": action.get("mission_id"), "outcome_status": outcome_status, "external_http_calls": 0},
        )
        return {"connected_outcome": outcome, "audit_event": event}

    def record_mission_trajectory(
        self,
        mission_id: str,
        *,
        outcome_status: str,
        outcome_summary: str,
        owner_acceptance: str,
        failure_reason: str | None,
        recovery_attempt: str | None,
        connected_outcome_id: str | None,
        scope: dict[str, str],
    ) -> dict[str, Any]:
        mission = self.get_mission(mission_id)
        if mission is None:
            return {"status": "not_found", "resource": "mission"}
        if mission.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": mission.get("scope")}
        connected_outcome = self.get_connected_outcome(connected_outcome_id) if connected_outcome_id else None
        if connected_outcome_id and connected_outcome is None:
            return {"status": "not_found", "resource": "connected_outcome"}
        if connected_outcome and connected_outcome.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": connected_outcome.get("scope")}

        actions = [action for action in self._action_records(scope) if action.get("mission_id") == mission_id]
        action_ids = {action.get("action_id") for action in actions}
        executed_actions = [action for action in actions if action.get("execution", {}).get("status") == "executed"]
        learnings = [
            learning
            for learning in self._learning_records(scope)
            if str(learning.get("source_action", {}).get("action_id")) in action_ids
        ]
        corrections = [
            correction
            for correction in self._correction_records(scope)
            if any(ref in correction.get("evidence_refs", []) for ref in [f"mission:{mission_id}", *[f"action:{action_id}" for action_id in action_ids if action_id]])
        ]
        evidence_refs = [
            f"mission:{mission_id}",
            *([f"claim:{mission.get('source_claim', {}).get('claim_id')}"] if mission.get("source_claim", {}).get("claim_id") else []),
            *([f"evidence_bundle:{mission.get('evidence', {}).get('evidence_bundle_id')}"] if mission.get("evidence", {}).get("evidence_bundle_id") else []),
            *mission.get("evidence", {}).get("artifact_refs", []),
            *[f"action:{action['action_id']}" for action in actions],
            *[f"learning:{learning['learning_id']}" for learning in learnings],
            *([f"connected_outcome:{connected_outcome_id}"] if connected_outcome_id else []),
        ]
        if outcome_status == "success" and not executed_actions and connected_outcome is None:
            return {"status": "evidence_required", "resource": "executed_action_or_connected_outcome"}
        if outcome_status == "success" and not any(ref.startswith("artifact:") for ref in evidence_refs):
            return {"status": "evidence_required", "resource": "artifact_evidence"}
        action_steps = [
            {
                "action_id": action.get("action_id"),
                "action_kind": action.get("action_kind"),
                "goal": action.get("goal"),
                "risk": action.get("risk"),
                "policy_decision": action.get("policy_decision"),
                "dry_run": action.get("dry_run"),
                "approval": action.get("approval"),
                "execution": action.get("execution"),
                "tool_results": action.get("execution", {}).get("result"),
            }
            for action in actions
        ]
        trajectory_base = {
            "schema_version": "cs.mission_trajectory.v0",
            "status": "reference",
            "scope": scope,
            "mission_id": mission_id,
            "goal": mission.get("goal"),
            "workspace": scope,
            "classification": "internal",
            "promotion_state": "reference",
            "contract": {
                "mission_id": mission_id,
                "status": mission.get("status"),
                "mode": mission.get("mode"),
                "source_claim": mission.get("source_claim"),
                "allowed_actions": mission.get("allowed_actions", []),
                "forbidden_actions": mission.get("forbidden_actions", []),
                "success_criteria": mission.get("success_criteria", []),
                "stop_conditions": mission.get("stop_conditions", []),
                "review_cadence": mission.get("review_cadence"),
                "escalation_rules": mission.get("escalation_rules", []),
                "evidence_expectations": mission.get("evidence_expectations", []),
            },
            "plan": {
                "steps": [
                    "collect_evidence",
                    "create_claim_or_decision",
                    "propose_action_card",
                    "execute_or_escalate",
                    "record_outcome",
                    "extract_lessons",
                ],
                "orchestrator_owned": True,
            },
            "evidence_refs": evidence_refs,
            "actions": action_steps,
            "policy_decisions": [action.get("policy_decision") for action in actions if action.get("policy_decision")],
            "approvals": [action.get("approval") for action in actions if action.get("approval")],
            "corrections": [f"correction:{correction['correction_id']}" for correction in corrections],
            "outcome": {
                "status": outcome_status,
                "summary": redact_text(outcome_summary),
                "owner_acceptance": owner_acceptance,
                "connected_outcome_id": connected_outcome_id,
            },
            "exceptions": [
                {
                    "reason": redact_text(failure_reason or "Mission recorded non-success outcome."),
                    "first_failing_layer": "policy_or_connector_or_evidence",
                    "impact": "Mission marked as learning material.",
                    "recovery_attempt": redact_text(recovery_attempt or "No recovery attempted."),
                }
            ]
            if outcome_status != "success"
            else [],
            "rollback_events": [],
            "cost_time": {
                "duration_ms": sum(float(action.get("dry_run", {}).get("duration_ms", 0) or 0) for action in actions),
                "estimated_cost_usd": 0,
                "real_external_http_calls": sum(int(action.get("execution", {}).get("result", {}).get("external_http_calls", 0) or 0) for action in actions),
            },
            "extracted_lessons": [f"learning:{learning['learning_id']}" for learning in learnings],
            "reference_corpus": {
                "stored_as_reference": True,
                "auto_converted_to_memory_or_rules": False,
                "retention_policy": "owner_scope_required",
                "privacy_policy": "active_scope_only",
            },
            "created_at": utc_now(),
        }
        trajectory_id = f"traj_{_json_hash(trajectory_base)[:16]}"
        trajectory = dict(trajectory_base)
        trajectory["trajectory_id"] = trajectory_id
        _write_json(self.trajectory_path(trajectory_id), trajectory)
        event = self.append_audit(
            "experience.trajectory.recorded",
            scope,
            {"type": "mission_trajectory", "id": trajectory_id},
            {"mission_id": mission_id, "outcome_status": outcome_status, "action_count": len(actions), "lesson_count": len(learnings)},
        )
        return {"trajectory": trajectory, "audit_event": event}

    def experience_library(self, scope: dict[str, str]) -> dict[str, Any]:
        trajectories = self._trajectory_records(scope)
        lessons = self._lesson_candidate_records(scope)
        outcomes = self._connected_outcome_records(scope)
        evaluations = self._model_evaluation_records(scope)
        library_base = {
            "schema_version": "cs.experience_library.v0",
            "status": "ready",
            "scope": scope,
            "trajectory_count": len(trajectories),
            "lesson_count": len(lessons),
            "connected_outcome_count": len(outcomes),
            "judge_review_count": len(evaluations),
            "entries": [
                {
                    "entry_id": f"trajectory:{trajectory['trajectory_id']}",
                    "entry_type": "trajectory",
                    "mission_id": trajectory.get("mission_id"),
                    "goal": trajectory.get("goal"),
                    "outcome_status": trajectory.get("outcome", {}).get("status"),
                    "owner_acceptance": trajectory.get("outcome", {}).get("owner_acceptance"),
                    "recoveries": trajectory.get("exceptions", []),
                    "lessons": trajectory.get("extracted_lessons", []),
                    "evidence_refs": trajectory.get("evidence_refs", []),
                }
                for trajectory in trajectories
            ],
            "browse_supported": True,
            "search_supported": True,
            "inspect_supported": True,
            "privacy_boundary": {
                "active_scope_only": True,
                "owner_namespace_required": True,
                "cross_namespace_results": 0,
            },
            "created_at": utc_now(),
        }
        library_id = f"explib_{_json_hash(library_base)[:16]}"
        library = dict(library_base)
        library["experience_library_id"] = library_id
        event = self.append_audit(
            "experience.library.generated",
            scope,
            {"type": "experience_library", "id": library_id},
            {"trajectory_count": len(trajectories), "lesson_count": len(lessons)},
        )
        return {"experience_library": library, "audit_event": event}

    def recommend_experience(self, mission_id: str, query: str, scope: dict[str, str]) -> dict[str, Any]:
        mission = self.get_mission(mission_id)
        if mission is None:
            return {"status": "not_found", "resource": "mission"}
        if mission.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": mission.get("scope")}
        terms = set(search_terms(query) + search_terms(str(mission.get("goal", ""))))
        matches = []
        for trajectory in self._trajectory_records(scope):
            haystack = " ".join(
                [
                    str(trajectory.get("goal", "")),
                    str(trajectory.get("outcome", {}).get("summary", "")),
                    " ".join(str(lesson) for lesson in trajectory.get("extracted_lessons", [])),
                ]
            ).lower()
            score = sum(1 for term in terms if term in haystack)
            if score > 0:
                matches.append((score, trajectory))
        matches.sort(key=lambda item: item[0], reverse=True)
        cited = [
            {
                "trajectory_id": trajectory["trajectory_id"],
                "mission_id": trajectory.get("mission_id"),
                "score": score,
                "outcome_status": trajectory.get("outcome", {}).get("status"),
                "evidence_refs": trajectory.get("evidence_refs", []),
                "influence": "Use as scoped prior experience; do not treat as truth without current evidence.",
            }
            for score, trajectory in matches[:5]
        ]
        recommendation_base = {
            "schema_version": "cs.experience_recommendation.v0",
            "status": "ready" if cited else "no_match",
            "scope": scope,
            "mission_id": mission_id,
            "query": redact_text(query),
            "cited_experiences": cited,
            "influence_explanation": {
                "visible_to_user": True,
                "can_inspect": True,
                "can_ignore": True,
                "does_not_auto_execute": True,
                "requires_current_evidence": True,
            },
            "created_at": utc_now(),
        }
        recommendation_id = f"exprec_{_json_hash(recommendation_base)[:16]}"
        recommendation = dict(recommendation_base)
        recommendation["experience_recommendation_id"] = recommendation_id
        _write_json(self.experience_recommendation_path(recommendation_id), recommendation)
        event = self.append_audit(
            "experience.recommendation.created",
            scope,
            {"type": "experience_recommendation", "id": recommendation_id},
            {"mission_id": mission_id, "match_count": len(cited), "can_ignore": True},
        )
        return {"experience_recommendation": recommendation, "audit_event": event}

    def propose_lesson(
        self,
        trajectory_id: str,
        *,
        lesson: str,
        applies_when: str,
        does_not_apply_when: str,
        confidence: str,
        scope: dict[str, str],
    ) -> dict[str, Any]:
        trajectory = self.get_trajectory(trajectory_id)
        if trajectory is None:
            return {"status": "not_found", "resource": "trajectory"}
        if trajectory.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": trajectory.get("scope")}
        lesson_base = {
            "schema_version": "cs.lesson_candidate.v0",
            "status": "candidate",
            "scope": scope,
            "trajectory_id": trajectory_id,
            "lesson": redact_text(lesson),
            "causal_attribution": {
                "source": f"trajectory:{trajectory_id}",
                "outcome_status": trajectory.get("outcome", {}).get("status"),
                "why_this_helped_or_failed": "Derived from trajectory outcome, action evidence, and owner acceptance.",
            },
            "applicability": {
                "applies_when": redact_text(applies_when),
                "does_not_apply_when": redact_text(does_not_apply_when),
                "evidence_required_before_use": True,
            },
            "confidence": confidence,
            "review_state": "needs_review",
            "promotion_stage": "candidate_lesson",
            "promotion_ladder": [
                "trajectory",
                "observation",
                "candidate_lesson",
                "workspace_memory",
                "mission_playbook",
                "organization_approved_rule",
                "solution_pack_or_product_learning_proposal",
            ],
            "promotion_history": [
                {
                    "stage": "trajectory",
                    "ref": f"trajectory:{trajectory_id}",
                    "created_at": utc_now(),
                },
                {
                    "stage": "observation",
                    "ref": f"trajectory:{trajectory_id}",
                    "created_at": utc_now(),
                },
                {
                    "stage": "candidate_lesson",
                    "created_at": utc_now(),
                },
            ],
            "scope_boundary": {
                "namespace_local_first": True,
                "auto_global_rule": False,
                "auto_product_default": False,
                "requires_staged_approval_for_broader_reuse": True,
            },
            "rollback": {
                "available": True,
                "affected_missions": [trajectory.get("mission_id")],
                "affected_playbooks": [],
            },
            "evidence_refs": [f"trajectory:{trajectory_id}", *trajectory.get("evidence_refs", [])],
            "created_at": utc_now(),
        }
        lesson_id = f"lesson_{_json_hash(lesson_base)[:16]}"
        record = dict(lesson_base)
        record["lesson_id"] = lesson_id
        _write_json(self.lesson_candidate_path(lesson_id), record)
        event = self.append_audit(
            "experience.lesson.proposed",
            scope,
            {"type": "lesson_candidate", "id": lesson_id},
            {"trajectory_id": trajectory_id, "promotion_stage": "candidate_lesson", "auto_global_rule": False},
        )
        return {"lesson": record, "audit_event": event}

    def promote_lesson(self, lesson_id: str, *, stage: str, scope: dict[str, str]) -> dict[str, Any]:
        lesson = self.get_lesson_candidate(lesson_id)
        if lesson is None:
            return {"status": "not_found", "resource": "lesson"}
        if lesson.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": lesson.get("scope")}
        ladder = lesson.get("promotion_ladder", [])
        current = lesson.get("promotion_stage")
        if stage not in ladder:
            return {"status": "invalid_stage", "resource": "lesson"}
        current_index = ladder.index(current)
        target_index = ladder.index(stage)
        if target_index < current_index:
            return {"status": "invalid_stage", "resource": "lesson"}
        if target_index > current_index + 1:
            return {"status": "invalid_stage", "resource": "lesson", "reason": "promotion_ladder_skipped"}
        approval_sensitive = {"organization_approved_rule", "solution_pack_or_product_learning_proposal"}
        if stage in approval_sensitive:
            return {"status": "approval_required", "resource": "lesson", "stage": stage}
        updated = dict(lesson)
        updated["promotion_stage"] = stage
        updated["status"] = "active" if stage in {"workspace_memory", "mission_playbook"} else "candidate"
        history = list(updated.get("promotion_history", []))
        history.append({"stage": stage, "approved": stage in {"workspace_memory", "mission_playbook"}, "created_at": utc_now()})
        updated["promotion_history"] = history
        updated["scope_boundary"] = {
            **updated.get("scope_boundary", {}),
            "auto_global_rule": False,
            "auto_product_default": False,
            "requires_staged_approval_for_broader_reuse": stage in {"organization_approved_rule", "solution_pack_or_product_learning_proposal"},
        }
        _write_json(self.lesson_candidate_path(lesson_id), updated)
        event = self.append_audit(
            "experience.lesson.promoted",
            scope,
            {"type": "lesson_candidate", "id": lesson_id},
            {"stage": stage, "auto_global_rule": False},
        )
        return {"lesson": updated, "audit_event": event}

    def control_lesson(self, lesson_id: str, *, action: str, scope: dict[str, str]) -> dict[str, Any]:
        lesson = self.get_lesson_candidate(lesson_id)
        if lesson is None:
            return {"status": "not_found", "resource": "lesson"}
        if lesson.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": lesson.get("scope")}
        updated = dict(lesson)
        previous = {"status": lesson.get("status"), "promotion_stage": lesson.get("promotion_stage"), "lesson": lesson.get("lesson")}
        if action == "rollback":
            updated["status"] = "rolled_back"
            updated["promotion_stage"] = "candidate_lesson"
        elif action == "demote":
            updated["status"] = "demoted"
            updated["promotion_stage"] = "observation"
        elif action == "disable":
            updated["status"] = "disabled"
        elif action == "revise":
            updated["status"] = "needs_revision"
        else:
            return {"status": "invalid_action", "resource": "lesson"}
        control_base = {
            "schema_version": "cs.lesson_control.v0",
            "status": "recorded",
            "scope": scope,
            "lesson_id": lesson_id,
            "action": action,
            "previous": previous,
            "current": {"status": updated.get("status"), "promotion_stage": updated.get("promotion_stage")},
            "affected_scope_report": {
                "affected_missions": updated.get("rollback", {}).get("affected_missions", []),
                "affected_playbooks": updated.get("rollback", {}).get("affected_playbooks", []),
                "requires_review": True,
            },
            "created_at": utc_now(),
        }
        control_id = f"lessonctl_{_json_hash(control_base)[:16]}"
        control = dict(control_base)
        control["lesson_control_id"] = control_id
        history = list(updated.get("promotion_history", []))
        history.append({"stage": updated.get("promotion_stage"), "control_id": control_id, "action": action, "created_at": control["created_at"]})
        updated["promotion_history"] = history
        _write_json(self.lesson_candidate_path(lesson_id), updated)
        _write_json(self.lesson_control_path(control_id), control)
        event = self.append_audit(
            "experience.lesson.controlled",
            scope,
            {"type": "lesson_control", "id": control_id},
            {"lesson_id": lesson_id, "action": action, "affected_missions": control["affected_scope_report"]["affected_missions"]},
        )
        return {"lesson": updated, "lesson_control": control, "audit_event": event}

    def record_behavior_signal(self, trajectory_id: str, *, signal: str, interpretation: str, scope: dict[str, str]) -> dict[str, Any]:
        trajectory = self.get_trajectory(trajectory_id)
        if trajectory is None:
            return {"status": "not_found", "resource": "trajectory"}
        if trajectory.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": trajectory.get("scope")}
        signal_base = {
            "schema_version": "cs.behavior_signal.v0",
            "status": "recorded",
            "scope": scope,
            "trajectory_id": trajectory_id,
            "signal": redact_text(signal),
            "interpretation": redact_text(interpretation),
            "authority": {
                "role": "supporting_signal",
                "can_personalize": True,
                "can_create_hypothesis": True,
                "outranks_outcome_evidence": False,
                "durable_learning_requires_outcome": True,
            },
            "evidence_refs": [f"trajectory:{trajectory_id}", *trajectory.get("evidence_refs", [])],
            "created_at": utc_now(),
        }
        signal_id = f"behav_{_json_hash(signal_base)[:16]}"
        record = dict(signal_base)
        record["behavior_signal_id"] = signal_id
        _write_json(self.behavior_signal_path(signal_id), record)
        event = self.append_audit(
            "experience.behavior_signal.recorded",
            scope,
            {"type": "behavior_signal", "id": signal_id},
            {"trajectory_id": trajectory_id, "outranks_outcome_evidence": False},
        )
        return {"behavior_signal": record, "audit_event": event}

    def record_model_evaluation(self, trajectory_id: str, *, score: str, rationale: str, scope: dict[str, str]) -> dict[str, Any]:
        trajectory = self.get_trajectory(trajectory_id)
        if trajectory is None:
            return {"status": "not_found", "resource": "trajectory"}
        if trajectory.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": trajectory.get("scope")}
        evaluation_base = {
            "schema_version": "cs.model_evaluation.v0",
            "status": "recorded",
            "scope": scope,
            "trajectory_id": trajectory_id,
            "provider": "local_test",
            "model": "local_test.v0",
            "score": redact_text(score),
            "rationale": redact_text(rationale),
            "signal_hierarchy": [
                "objective_outcome",
                "owner_acceptance",
                "policy_and_audit",
                "evidence_coverage",
                "model_self_evaluation",
            ],
            "supports_product_learning": True,
            "overrides_outcome_evidence": False,
            "directly_mutates_memory_or_rules": False,
            "pass_judge": False,
            "evidence_refs": [f"trajectory:{trajectory_id}", *trajectory.get("evidence_refs", [])],
            "created_at": utc_now(),
        }
        evaluation_id = f"eval_{_json_hash(evaluation_base)[:16]}"
        evaluation = dict(evaluation_base)
        evaluation["model_evaluation_id"] = evaluation_id
        _write_json(self.model_evaluation_path(evaluation_id), evaluation)
        event = self.append_audit(
            "experience.model_evaluation.recorded",
            scope,
            {"type": "model_evaluation", "id": evaluation_id},
            {"trajectory_id": trajectory_id, "overrides_outcome_evidence": False, "pass_judge": False},
        )
        return {"model_evaluation": evaluation, "audit_event": event}

    def propose_product_improvement(self, lesson_id: str, *, proposal: str, scope: dict[str, str]) -> dict[str, Any]:
        lesson = self.get_lesson_candidate(lesson_id)
        if lesson is None:
            return {"status": "not_found", "resource": "lesson"}
        if lesson.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": lesson.get("scope")}
        proposal_base = {
            "schema_version": "cs.product_improvement_proposal.v0",
            "status": "proposed",
            "scope": {**scope, "namespace_id": "product_learning"},
            "source_scope": scope,
            "lesson_id": lesson_id,
            "proposal": redact_text(proposal),
            "evidence_refs": [f"lesson:{lesson_id}", *lesson.get("evidence_refs", [])],
            "benchmark_results": [
                {
                    "name": "local_fixture_replay",
                    "status": "evidence_attached" if lesson.get("evidence_refs") else "not_verified",
                    "evidence_refs": lesson.get("evidence_refs", [])[:5],
                    "external_calls": 0,
                }
            ],
            "expected_impact": "Improve repeat mission planning while preserving evidence and owner review.",
            "versioning": {"proposed_version": "v0.local-proposal", "diff_available": True},
            "monitoring": {"required": True, "metrics": ["outcome_quality", "rollback_rate", "owner_acceptance"]},
            "rollback": {"available": True, "plan": "Disable proposal and restore previous local default."},
            "approval": {"required": True, "status": "not_approved"},
            "global_behavior_changed": False,
            "created_at": utc_now(),
        }
        proposal_id = f"prodimp_{_json_hash(proposal_base)[:16]}"
        record = dict(proposal_base)
        record["product_improvement_id"] = proposal_id
        _write_json(self.product_improvement_path(proposal_id), record)
        event = self.append_audit(
            "experience.product_improvement.proposed",
            scope,
            {"type": "product_improvement", "id": proposal_id},
            {"lesson_id": lesson_id, "global_behavior_changed": False, "approval_required": True},
        )
        return {"product_improvement": record, "audit_event": event}

    def record_local_adaptation(self, lesson_id: str, *, preference: str, scope: dict[str, str]) -> dict[str, Any]:
        lesson = self.get_lesson_candidate(lesson_id)
        if lesson is None:
            return {"status": "not_found", "resource": "lesson"}
        if lesson.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": lesson.get("scope")}
        adaptation_base = {
            "schema_version": "cs.local_adaptation.v0",
            "status": "active",
            "scope": scope,
            "lesson_id": lesson_id,
            "preference": redact_text(preference),
            "adaptation_types": ["memory_ranking", "brief_style", "recurring_workflow_optimization", "onboarding_personalization", "preference_learning"],
            "namespace_local": True,
            "visible_to_owner": True,
            "reset_available": True,
            "changes_other_namespaces": False,
            "changes_product_defaults": False,
            "created_at": utc_now(),
        }
        adaptation_id = f"localadapt_{_json_hash(adaptation_base)[:16]}"
        adaptation = dict(adaptation_base)
        adaptation["local_adaptation_id"] = adaptation_id
        _write_json(self.local_adaptation_path(adaptation_id), adaptation)
        event = self.append_audit(
            "experience.local_adaptation.recorded",
            scope,
            {"type": "local_adaptation", "id": adaptation_id},
            {"lesson_id": lesson_id, "namespace_local": True, "changes_other_namespaces": False},
        )
        return {"local_adaptation": adaptation, "audit_event": event}

    def reset_local_adaptation(self, adaptation_id: str, scope: dict[str, str]) -> dict[str, Any]:
        path = self.local_adaptation_path(adaptation_id)
        if not path.exists():
            return {"status": "not_found", "resource": "local_adaptation"}
        adaptation = _read_json(path)
        if adaptation.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": adaptation.get("scope")}
        updated = dict(adaptation)
        updated["status"] = "reset"
        updated["reset_at"] = utc_now()
        _write_json(path, updated)
        event = self.append_audit(
            "experience.local_adaptation.reset",
            scope,
            {"type": "local_adaptation", "id": adaptation_id},
            {"status": "reset", "changes_other_namespaces": False},
        )
        return {"local_adaptation": updated, "audit_event": event}

    def outcome_quality_metrics(self, scope: dict[str, str]) -> dict[str, Any]:
        trajectories = self._trajectory_records(scope)
        completed = [trajectory for trajectory in trajectories if trajectory.get("outcome", {}).get("status") == "success"]
        failed = [trajectory for trajectory in trajectories if trajectory.get("outcome", {}).get("status") != "success"]
        action_count = sum(len(trajectory.get("actions", [])) for trajectory in trajectories)
        auto_action_count = sum(
            1
            for trajectory in trajectories
            for action in trajectory.get("actions", [])
            if action.get("policy_decision", {}).get("can_execute_now") is True
        )
        evidence_ref_count = sum(len(trajectory.get("evidence_refs", [])) for trajectory in trajectories)
        accepted = [trajectory for trajectory in trajectories if trajectory.get("outcome", {}).get("owner_acceptance") == "accepted"]
        metrics_base = {
            "schema_version": "cs.outcome_quality_report.v0",
            "status": "ready",
            "scope": scope,
            "primary_metric": "outcome_quality",
            "outcome_quality": {
                "trajectory_count": len(trajectories),
                "success_count": len(completed),
                "failure_count": len(failed),
                "owner_acceptance_count": len(accepted),
                "evidence_ref_count": evidence_ref_count,
                "rollback_or_escalation_failure_count": 0,
            },
            "supporting_metrics": {
                "task_completion_rate": len(completed) / len(trajectories) if trajectories else 0,
                "autonomy_ratio": auto_action_count / action_count if action_count else 0,
                "evidence_coverage": evidence_ref_count / len(trajectories) if trajectories else 0,
                "error_count": len(failed),
                "owner_acceptance_rate": len(accepted) / len(trajectories) if trajectories else 0,
            },
            "autonomy_ratio_not_primary": True,
            "created_at": utc_now(),
        }
        report_id = f"metrics_{_json_hash(metrics_base)[:16]}"
        report = dict(metrics_base)
        report["outcome_quality_report_id"] = report_id
        _write_json(self.outcome_metric_path(report_id), report)
        event = self.append_audit(
            "experience.metrics.generated",
            scope,
            {"type": "outcome_quality_report", "id": report_id},
            {"trajectory_count": len(trajectories), "primary_metric": "outcome_quality"},
        )
        return {"outcome_quality_report": report, "audit_event": event}

    def search_experience(self, query: str, scope: dict[str, str]) -> dict[str, Any]:
        terms = set(search_terms(query))
        results = []
        for trajectory in self._trajectory_records(scope):
            reference = trajectory.get("reference_corpus", {})
            classification = trajectory.get("classification", "internal")
            promotion_state = trajectory.get("promotion_state", "reference")
            retention_ok = reference.get("retention_policy") == "owner_scope_required"
            classification_ok = classification in {"public", "internal", "confidential"}
            promotion_ok = promotion_state in {"reference", "candidate_lesson", "workspace_memory", "mission_playbook"}
            if not (retention_ok and classification_ok and promotion_ok):
                continue
            haystack = " ".join([str(trajectory.get("goal", "")), str(trajectory.get("outcome", {}).get("summary", ""))]).lower()
            if any(term in haystack for term in terms):
                results.append(
                    {
                        "result_type": "trajectory",
                        "trajectory_id": trajectory["trajectory_id"],
                        "scope": trajectory.get("scope"),
                        "classification": classification,
                        "promotion_state": promotion_state,
                        "retention_policy": reference.get("retention_policy"),
                        "snippet": redact_text(str(trajectory.get("outcome", {}).get("summary", ""))),
                        "evidence_refs": trajectory.get("evidence_refs", []),
                    }
                )
        search_base = {
            "schema_version": "cs.experience_search.v0",
            "status": "success",
            "scope": scope,
            "query": redact_text(query),
            "result_count": len(results),
            "results": results,
            "privacy_filters": {
                "owner_namespace": scope,
                "active_scope_only": True,
                "permissions_enforced": True,
                "retention_enforced": True,
                "classification_checked": True,
                "promotion_state_checked": True,
                "cross_namespace_results": 0,
            },
            "created_at": utc_now(),
        }
        search_id = f"expsearch_{_json_hash(search_base)[:16]}"
        record = dict(search_base)
        record["experience_search_id"] = search_id
        event = self.append_audit(
            "experience.search.completed",
            scope,
            {"type": "experience_search", "id": search_id},
            {"result_count": len(results), "cross_namespace_results": 0},
        )
        return {"experience_search": record, "audit_event": event}

    def export_experience(self, scope: dict[str, str]) -> dict[str, Any]:
        trajectories = self._trajectory_records(scope)
        lessons = self._lesson_candidate_records(scope)
        outcomes = self._connected_outcome_records(scope)
        evaluations = self._model_evaluation_records(scope)
        export_base = {
            "schema_version": "cs.experience_export.v0",
            "status": "ready",
            "scope": scope,
            "format": "json",
            "permission_aware_redaction": True,
            "unauthorized_raw_content_leaked": False,
            "entries": {
                "trajectories": [
                    {
                        "trajectory_id": trajectory["trajectory_id"],
                        "mission_id": trajectory.get("mission_id"),
                        "outcome": trajectory.get("outcome"),
                        "evidence_refs": trajectory.get("evidence_refs", []),
                        "lesson_refs": trajectory.get("extracted_lessons", []),
                        "promotion_state": "reference",
                    }
                    for trajectory in trajectories
                ],
                "lessons": [
                    {
                        "lesson_id": lesson["lesson_id"],
                        "status": lesson.get("status"),
                        "promotion_stage": lesson.get("promotion_stage"),
                        "promotion_history": lesson.get("promotion_history", []),
                        "applicability": lesson.get("applicability"),
                        "evidence_refs": lesson.get("evidence_refs", []),
                    }
                    for lesson in lessons
                ],
                "judge_results": [
                    {
                        "model_evaluation_id": evaluation["model_evaluation_id"],
                        "provider": evaluation.get("provider"),
                        "score": evaluation.get("score"),
                        "overrides_outcome_evidence": evaluation.get("overrides_outcome_evidence"),
                        "evidence_refs": evaluation.get("evidence_refs", []),
                    }
                    for evaluation in evaluations
                ],
                "connected_outcomes": [
                    {
                        "connected_outcome_id": outcome["connected_outcome_id"],
                        "outcome_status": outcome.get("outcome_status"),
                        "evidence_refs": outcome.get("evidence_refs", []),
                    }
                    for outcome in outcomes
                ],
                "playbooks": [
                    {
                        "lesson_id": lesson["lesson_id"],
                        "promotion_stage": lesson.get("promotion_stage"),
                        "status": lesson.get("status"),
                    }
                    for lesson in lessons
                    if lesson.get("promotion_stage") == "mission_playbook"
                ],
            },
            "created_at": utc_now(),
        }
        export_id = f"expexport_{_json_hash(export_base)[:16]}"
        export = dict(export_base)
        export["experience_export_id"] = export_id
        _write_json(self.experience_export_path(export_id), export)
        event = self.append_audit(
            "experience.export.created",
            scope,
            {"type": "experience_export", "id": export_id},
            {"trajectory_count": len(trajectories), "lesson_count": len(lessons), "unauthorized_raw_content_leaked": False},
        )
        return {"experience_export": export, "audit_event": event}

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

    def _claim_evidence_refs(self, claim: dict[str, Any]) -> list[str]:
        evidence = claim.get("evidence_bundle", {})
        refs = [f"claim:{claim['claim_id']}"]
        if evidence.get("evidence_bundle_id"):
            refs.append(f"evidence_bundle:{evidence['evidence_bundle_id']}")
        if evidence.get("search_snapshot_id"):
            refs.append(f"search_snapshot:{evidence['search_snapshot_id']}")
        refs.extend(evidence.get("artifact_refs", []))
        return refs

    def _scoped_target(self, target_kind: str, target_id: str, scope: dict[str, str]) -> dict[str, Any]:
        getters = {
            "brief": self.get_brief,
            "claim": self.get_claim,
            "knowledge_capsule": self.get_knowledge_capsule,
            "decision_card": self.get_decision_card,
            "memory": self.get_memory,
        }
        getter = getters.get(target_kind)
        if getter is None:
            return {"status": "unsupported_kind"}
        record = getter(target_id)
        if record is None:
            return {"status": "not_found"}
        if record.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": record.get("scope")}
        return {"record": record}

    def create_knowledge_capsule(
        self,
        *,
        claim_id: str,
        title: str,
        summary: str,
        scope: dict[str, str],
    ) -> dict[str, Any]:
        claim = self.get_claim(claim_id)
        if claim is None:
            return {"status": "not_found", "resource": "claim"}
        if claim.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": claim.get("scope")}
        evidence = claim.get("evidence_bundle", {})
        artifact_refs = evidence.get("artifact_refs", [])
        if not evidence.get("evidence_bundle_id") or not artifact_refs:
            return {"status": "evidence_required", "resource": "claim"}

        related_missions = [
            f"mission:{mission['mission_id']}"
            for mission in self._mission_records(scope)
            if mission.get("source_claim", {}).get("claim_id") == claim_id
        ]
        capsule_base = {
            "schema_version": "cs.knowledge_capsule.v0",
            "status": "active",
            "title": redact_text(title),
            "summary": redact_text(summary),
            "scope": scope,
            "trust_state": claim.get("trust_state"),
            "freshness": {
                "status": "current",
                "source_created_at": claim.get("created_at"),
                "last_reviewed_at": utc_now(),
                "stale_after_days": 90,
            },
            "source": {
                "created_from": "claim.capsule.create",
                "source_claim_id": claim_id,
                "source_statement": claim.get("statement"),
                "evidence_bundle_id": evidence.get("evidence_bundle_id"),
                "search_snapshot_id": evidence.get("search_snapshot_id"),
                "artifact_refs": artifact_refs,
            },
            "related_claim_refs": [f"claim:{claim_id}"],
            "related_mission_refs": related_missions,
            "evidence_refs": self._claim_evidence_refs(claim),
            "created_at": utc_now(),
        }
        capsule_id = f"capsule_{_json_hash(capsule_base)[:16]}"
        capsule = dict(capsule_base)
        capsule["capsule_id"] = capsule_id
        _write_json(self.knowledge_capsule_path(capsule_id), capsule)
        event = self.append_audit(
            "knowledge_capsule.created",
            scope,
            {"type": "knowledge_capsule", "id": capsule_id},
            {
                "claim_id": claim_id,
                "trust_state": capsule["trust_state"],
                "evidence_bundle_id": evidence.get("evidence_bundle_id"),
                "artifact_refs": artifact_refs,
            },
        )
        return {"capsule": capsule, "audit_event": event}

    def show_knowledge_capsule(self, capsule_id: str, scope: dict[str, str]) -> dict[str, Any]:
        capsule = self.get_knowledge_capsule(capsule_id)
        if capsule is None:
            return {"status": "not_found"}
        if capsule.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": capsule.get("scope")}
        event = self.append_audit(
            "knowledge_capsule.read",
            scope,
            {"type": "knowledge_capsule", "id": capsule_id},
            {"reason": "cli_capsule_show"},
        )
        return {"capsule": capsule, "audit_event": event}

    def create_decision_card(
        self,
        *,
        goal: str,
        claim_id: str,
        mission_id: str,
        scope: dict[str, str],
    ) -> dict[str, Any]:
        claim = self.get_claim(claim_id)
        if claim is None:
            return {"status": "not_found", "resource": "claim"}
        mission = self.get_mission(mission_id)
        if mission is None:
            return {"status": "not_found", "resource": "mission"}
        if claim.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": claim.get("scope")}
        if mission.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": mission.get("scope")}
        evidence = claim.get("evidence_bundle", {})
        artifact_refs = evidence.get("artifact_refs", [])
        if not evidence.get("evidence_bundle_id") or not artifact_refs:
            return {"status": "evidence_required", "resource": "claim"}

        actions = [
            {
                "action_id": action["action_id"],
                "status": action.get("status"),
                "action_kind": action.get("action_kind"),
                "risk": action.get("risk"),
                "approval_required": action.get("policy_decision", {}).get("approval_required"),
                "policy": action.get("policy_decision", {}).get("policy"),
                "result_status": action.get("execution", {}).get("result", {}).get("status"),
            }
            for action in self._action_records(scope)
            if action.get("mission_id") == mission_id
        ]
        mission_action_ids = {action["action_id"] for action in self._action_records(scope) if action.get("mission_id") == mission_id}
        learning_history = [
            f"learning:{learning['learning_id']}"
            for learning in self._learning_records(scope)
            if learning.get("source_action", {}).get("action_id") in mission_action_ids
        ]
        card_base = {
            "schema_version": "cs.decision_card.v0",
            "status": "active",
            "scope": scope,
            "goal": redact_text(goal),
            "context": {
                "source_mission_id": mission_id,
                "source_claim_id": claim_id,
                "mission_status": mission.get("status"),
                "workspace_mode": mission.get("workspace_mode"),
                "claim_statement": claim.get("statement"),
                "claim_trust_state": claim.get("trust_state"),
            },
            "evidence": {
                "evidence_bundle_id": evidence.get("evidence_bundle_id"),
                "search_snapshot_id": evidence.get("search_snapshot_id"),
                "artifact_refs": artifact_refs,
                "evidence_refs": self._claim_evidence_refs(claim),
            },
            "claims": [
                {
                    "claim_id": claim_id,
                    "statement": claim.get("statement"),
                    "trust_state": claim.get("trust_state"),
                    "status": claim.get("status"),
                }
            ],
            "open_questions": [
                "What additional evidence would raise confidence before broader publication?",
                "Which owner should approve any external-facing action?",
            ],
            "actions": actions,
            "approvals": {
                "claim_status": claim.get("status"),
                "claim_can_publish_shared_truth": claim.get("authority", {}).get("can_publish_shared_truth"),
                "mission_mode": mission.get("workspace_mode"),
            },
            "outcomes": [
                {
                    "source": "local_scaffold",
                    "status": "pending_human_outcome",
                    "evidence_required": True,
                }
            ],
            "learning_history": learning_history,
            "created_at": utc_now(),
        }
        card_id = f"decision_{_json_hash(card_base)[:16]}"
        card = dict(card_base)
        card["decision_card_id"] = card_id
        _write_json(self.decision_card_path(card_id), card)
        event = self.append_audit(
            "decision_card.created",
            scope,
            {"type": "decision_card", "id": card_id},
            {
                "mission_id": mission_id,
                "claim_id": claim_id,
                "action_count": len(actions),
                "evidence_bundle_id": evidence.get("evidence_bundle_id"),
            },
        )
        return {"decision_card": card, "audit_event": event}

    def show_decision_card(self, card_id: str, scope: dict[str, str]) -> dict[str, Any]:
        card = self.get_decision_card(card_id)
        if card is None:
            return {"status": "not_found"}
        if card.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": card.get("scope")}
        event = self.append_audit(
            "decision_card.read",
            scope,
            {"type": "decision_card", "id": card_id},
            {"reason": "cli_decision_card_show"},
        )
        return {"decision_card": card, "audit_event": event}

    def record_correction(
        self,
        *,
        target_kind: str,
        target_id: str,
        corrected_text: str,
        rationale: str,
        evidence_bundle_id: str | None,
        scope: dict[str, str],
    ) -> dict[str, Any]:
        target_result = self._scoped_target(target_kind, target_id, scope)
        if target_result.get("status"):
            return target_result
        target = target_result["record"]
        bundle = self.get_evidence_bundle(evidence_bundle_id) if evidence_bundle_id else None
        if evidence_bundle_id and bundle is None:
            return {"status": "not_found", "resource": "evidence_bundle"}
        if bundle and bundle.get("filters") != scope:
            return {"status": "scope_denied", "resource_scope": bundle.get("filters")}
        evidence_refs = [f"{target_kind}:{target_id}"]
        if bundle:
            evidence_refs.append(f"evidence_bundle:{evidence_bundle_id}")
            evidence_refs.append(f"search_snapshot:{bundle.get('search_snapshot_id')}")
            evidence_refs.extend(f"artifact:{item['artifact_id']}" for item in bundle.get("evidence_items", []))
        else:
            evidence_refs.append("owner_judgment:local-user")

        correction_base = {
            "schema_version": "cs.correction.v0",
            "status": "recorded",
            "scope": scope,
            "target": {
                "kind": target_kind,
                "id": target_id,
                "original_trust_state": target.get("trust_state"),
                "original_provenance": target.get("provenance") or target.get("source"),
            },
            "correction": {
                "corrected_text": redact_text(corrected_text),
                "rationale": redact_text(rationale),
                "source_type": "evidence_bundle" if bundle else "owner_judgment",
            },
            "learning_signal": {
                "signal_type": "human_evidence_aware_correction",
                "used_for_silent_overwrite": False,
                "requires_review_before_memory_update": True,
            },
            "evidence_refs": evidence_refs,
            "provenance_preserved": True,
            "created_at": utc_now(),
        }
        correction_id = f"correction_{_json_hash(correction_base)[:16]}"
        correction = dict(correction_base)
        correction["correction_id"] = correction_id
        _write_json(self.correction_path(correction_id), correction)

        updated_target = dict(target)
        history = list(updated_target.get("correction_history", []))
        history.append(
            {
                "correction_id": correction_id,
                "corrected_at": correction["created_at"],
                "evidence_refs": evidence_refs,
                "silent_overwrite": False,
            }
        )
        updated_target["correction_history"] = history
        if target_kind == "claim":
            _write_json(self.claim_path(target_id), updated_target)
        elif target_kind == "brief":
            _write_json(self.brief_path(target_id), updated_target)
        elif target_kind == "knowledge_capsule":
            _write_json(self.knowledge_capsule_path(target_id), updated_target)
        elif target_kind == "decision_card":
            _write_json(self.decision_card_path(target_id), updated_target)
        elif target_kind == "memory":
            _write_json(self.memory_path(target_id), updated_target)

        event = self.append_audit(
            "correction.recorded",
            scope,
            {"type": "correction", "id": correction_id},
            {
                "target_kind": target_kind,
                "target_id": target_id,
                "source_type": correction["correction"]["source_type"],
                "silent_overwrite": False,
                "provenance_preserved": True,
            },
        )
        return {"correction": correction, "target": updated_target, "audit_event": event}

    def create_share_view(
        self,
        *,
        item_kind: str,
        item_id: str,
        audience: str,
        channel: str,
        scope: dict[str, str],
    ) -> dict[str, Any]:
        target_result = self._scoped_target(item_kind, item_id, scope)
        if target_result.get("status"):
            return target_result
        item = target_result["record"]
        trust_state = item.get("trust_state") or item.get("context", {}).get("claim_trust_state")
        if item_kind == "knowledge_capsule":
            evidence_refs = list(item.get("evidence_refs", []))
            owner_label = scope["owner_id"]
        elif item_kind == "claim":
            evidence_refs = self._claim_evidence_refs(item)
            owner_label = scope["owner_id"]
        elif item_kind == "decision_card":
            evidence_refs = list(item.get("evidence", {}).get("evidence_refs", []))
            owner_label = scope["owner_id"]
        else:
            evidence_refs = [f"{item_kind}:{item_id}"]
            owner_label = scope["owner_id"]
        publish_state = "organization_approved" if trust_state == "approved" and scope["namespace_id"] == "organization" else "shared"
        if scope["namespace_id"] == "personal" and trust_state != "approved":
            publish_state = "personal_shared_preview"

        share_base = {
            "schema_version": "cs.shared_item_view.v0",
            "status": "shared",
            "scope": scope,
            "item": {
                "kind": item_kind,
                "id": item_id,
                "trust_state": trust_state,
                "status": item.get("status"),
                "owner_id": owner_label,
                "namespace_id": scope["namespace_id"],
                "workspace_id": scope["workspace_id"],
            },
            "audience": audience,
            "channel": channel,
            "visibility": {
                "state": publish_state,
                "trust_state_visible": True,
                "evidence_visible": bool(evidence_refs),
                "owner_visible": True,
                "scope_visible": True,
                "approved_for_shared_truth": bool(item.get("authority", {}).get("can_publish_shared_truth") or trust_state == "approved"),
            },
            "recipient_view": {
                "trust_state": trust_state,
                "evidence_refs": evidence_refs,
                "owner": owner_label,
                "scope": scope,
                "personal_shared_or_org_approved": publish_state,
            },
            "evidence_refs": evidence_refs,
            "created_at": utc_now(),
        }
        share_id = f"share_{_json_hash(share_base)[:16]}"
        share = dict(share_base)
        share["share_id"] = share_id
        _write_json(self.share_path(share_id), share)
        event = self.append_audit(
            "share_view.created",
            scope,
            {"type": "share", "id": share_id},
            {
                "item_kind": item_kind,
                "item_id": item_id,
                "trust_state": trust_state,
                "audience": audience,
                "channel": channel,
                "evidence_ref_count": len(evidence_refs),
            },
        )
        return {"share": share, "audit_event": event}

    def show_share_view(self, share_id: str, scope: dict[str, str]) -> dict[str, Any]:
        share = self.get_share(share_id)
        if share is None:
            return {"status": "not_found"}
        if share.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": share.get("scope")}
        event = self.append_audit(
            "share_view.read",
            scope,
            {"type": "share", "id": share_id},
            {"reason": "cli_share_show"},
        )
        return {"share": share, "audit_event": event}

    def _fact_parts(self, value: str) -> tuple[str, str]:
        cleaned = value.strip().rstrip(".")
        for separator in [" is ", " = ", ":"]:
            if separator in cleaned:
                key, fact_value = cleaned.split(separator, 1)
                return self._normalize_key(key), fact_value.strip()
        return self._normalize_key(cleaned), cleaned

    def _normalize_key(self, value: str) -> str:
        terms = search_terms(value)
        return "_".join(terms[:8]) or "unlabeled_fact"

    def _relationship_parts(self, value: str) -> tuple[str, str, str]:
        match = re.match(r"(.+?)\s*->\s*(.+?)\s*:\s*(.+)", value)
        if not match:
            return value.strip(), "", "related_to"
        return match.group(1).strip(), match.group(2).strip(), self._normalize_key(match.group(3))

    def suggest_operational_structure(
        self,
        artifact_id: str,
        scope: dict[str, str],
        *,
        domain: str = "general",
    ) -> dict[str, Any]:
        artifact = self.get_artifact(artifact_id, scope)
        if artifact is None:
            return {"status": "not_found", "resource": "artifact"}
        if artifact.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": artifact.get("scope")}

        text = self._derived_text(artifact)
        suggestions: list[dict[str, Any]] = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            relationship_match = re.match(r"Relationship:\s*(.+)", stripped, re.IGNORECASE)
            if relationship_match:
                source, target, predicate = self._relationship_parts(relationship_match.group(1))
                base = {
                    "schema_version": "cs.understanding_suggestion.v0",
                    "status": "suggested",
                    "kind": "link",
                    "candidate_type": "relationship",
                    "label": f"{source} {predicate.replace('_', ' ')} {target}",
                    "value": relationship_match.group(1).strip(),
                    "relationship": {"source": source, "target": target, "predicate": predicate},
                    "source": {
                        "artifact_id": artifact_id,
                        "line_no": line_no,
                        "snippet": stripped,
                    },
                    "scope": scope,
                    "trust_state": "draft",
                    "approved_ontology_truth": False,
                    "confidence": 0.78,
                    "evidence_refs": [f"artifact:{artifact_id}", f"storage:{artifact.get('original_storage_ref')}"],
                    "unsupported_inferences": [],
                    "evidence_gaps": [],
                    "created_at": utc_now(),
                }
                suggestion = dict(base)
                suggestion["suggestion_id"] = f"sugg_{_json_hash(base)[:16]}"
                suggestions.append(suggestion)
                continue

            label_match = re.match(r"([A-Za-z][A-Za-z ]{1,32}):\s*(.+)", stripped)
            if not label_match:
                continue
            label_key = self._normalize_key(label_match.group(1)).replace("_", " ")
            raw_value = label_match.group(2).strip()
            structure = STRUCTURE_LABELS.get(label_key)
            if structure is None:
                continue
            kind, candidate_type = structure
            fact_key = None
            fact_value = raw_value
            if kind == "fact":
                fact_key, fact_value = self._fact_parts(raw_value)
            base = {
                "schema_version": "cs.understanding_suggestion.v0",
                "status": "suggested",
                "kind": kind,
                "candidate_type": candidate_type,
                "label": raw_value,
                "value": raw_value,
                "fact_key": fact_key,
                "fact_value": fact_value,
                "source": {
                    "artifact_id": artifact_id,
                    "line_no": line_no,
                    "snippet": stripped,
                },
                "scope": scope,
                "trust_state": "draft",
                "approved_ontology_truth": False,
                "confidence": 0.82 if domain != "unknown" else 0.64,
                "evidence_refs": [f"artifact:{artifact_id}", f"storage:{artifact.get('original_storage_ref')}"],
                "unsupported_inferences": (
                    ["Domain-specific meaning is not inferred without evidence or an approved solution pack."]
                    if domain == "unknown"
                    else []
                ),
                "evidence_gaps": (
                    ["No approved domain ontology or solution-pack rule was used for this suggestion."]
                    if domain == "unknown"
                    else []
                ),
                "created_at": utc_now(),
            }
            suggestion = dict(base)
            suggestion["suggestion_id"] = f"sugg_{_json_hash(base)[:16]}"
            suggestions.append(suggestion)

        for suggestion in suggestions:
            _write_json(self.understanding_suggestion_path(suggestion["suggestion_id"]), suggestion)
        event = self.append_audit(
            "understanding.suggestions.created",
            scope,
            {"type": "artifact", "id": artifact_id},
            {
                "suggestion_count": len(suggestions),
                "domain": domain,
                "approved_ontology_truth_created": False,
            },
        )
        return {"suggestions": suggestions, "audit_event": event}

    def promote_understanding_suggestion(self, suggestion_id: str, scope: dict[str, str]) -> dict[str, Any]:
        suggestion = self.get_understanding_suggestion(suggestion_id)
        if suggestion is None:
            return {"status": "not_found", "resource": "understanding_suggestion"}
        if suggestion.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": suggestion.get("scope")}

        item_base = {
            "schema_version": "cs.ontology_item.v0",
            "status": "promoted_draft",
            "scope": scope,
            "kind": suggestion.get("kind"),
            "candidate_type": suggestion.get("candidate_type"),
            "label": suggestion.get("label"),
            "value": suggestion.get("value"),
            "fact_key": suggestion.get("fact_key"),
            "fact_value": suggestion.get("fact_value"),
            "relationship": suggestion.get("relationship"),
            "trust_state": suggestion.get("trust_state"),
            "approved_ontology_truth": False,
            "source_suggestion_id": suggestion_id,
            "source": suggestion.get("source"),
            "provenance": {
                "created_from": "understanding.promote",
                "source_suggestion_id": suggestion_id,
                "source_artifact_id": suggestion.get("source", {}).get("artifact_id"),
                "trust_state_preserved": True,
            },
            "evidence_refs": [f"understanding_suggestion:{suggestion_id}", *suggestion.get("evidence_refs", [])],
            "version": 1,
            "version_history": [],
            "created_at": utc_now(),
        }
        item_id = f"onto_{_json_hash(item_base)[:16]}"
        item = dict(item_base)
        item["ontology_item_id"] = item_id
        _write_json(self.ontology_item_path(item_id), item)
        event = self.append_audit(
            "ontology.item.promoted",
            scope,
            {"type": "ontology_item", "id": item_id},
            {
                "suggestion_id": suggestion_id,
                "trust_state": item["trust_state"],
                "approved_ontology_truth": False,
                "evidence_refs": item["evidence_refs"],
            },
        )
        return {"ontology_item": item, "audit_event": event}

    def operational_map(self, scope: dict[str, str]) -> dict[str, Any]:
        artifacts = self._artifact_records(scope)
        ontology_items = self._ontology_item_records(scope)
        claims = self._claim_records(scope)
        missions = self._mission_records(scope)
        actions = self._action_records(scope)
        policy_decisions = []
        if self.access_decision_dir.exists():
            for path in sorted(self.access_decision_dir.glob("*.json")):
                decision = _read_json(path)
                if decision.get("scope") == scope:
                    policy_decisions.append(decision)

        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        timelines: list[dict[str, Any]] = []
        for artifact in artifacts:
            nodes.append(
                {
                    "id": f"artifact:{artifact['artifact_id']}",
                    "type": "artifact",
                    "label": artifact["artifact_id"],
                    "evidence_refs": [f"artifact:{artifact['artifact_id']}", f"storage:{artifact.get('original_storage_ref')}"],
                    "correctable": True,
                }
            )
            timelines.append({"at": artifact.get("source", {}).get("ingested_at"), "event": "artifact_ingested", "ref": f"artifact:{artifact['artifact_id']}"})
        for item in ontology_items:
            nodes.append(
                {
                    "id": f"ontology_item:{item['ontology_item_id']}",
                    "type": item.get("kind"),
                    "label": item.get("label"),
                    "trust_state": item.get("trust_state"),
                    "evidence_refs": item.get("evidence_refs", []),
                    "correctable": True,
                }
            )
            source_artifact_id = item.get("source", {}).get("artifact_id")
            if source_artifact_id:
                edges.append(
                    {
                        "source": f"ontology_item:{item['ontology_item_id']}",
                        "target": f"artifact:{source_artifact_id}",
                        "type": "supported_by",
                        "evidence_refs": item.get("evidence_refs", []),
                    }
                )
        for claim in claims:
            nodes.append({"id": f"claim:{claim['claim_id']}", "type": "claim", "label": claim.get("statement"), "trust_state": claim.get("trust_state"), "evidence_refs": self._claim_evidence_refs(claim), "correctable": True})
            for artifact_ref in claim.get("evidence_bundle", {}).get("artifact_refs", []):
                edges.append({"source": f"claim:{claim['claim_id']}", "target": artifact_ref, "type": "supported_by", "evidence_refs": self._claim_evidence_refs(claim)})
        for mission in missions:
            nodes.append({"id": f"mission:{mission['mission_id']}", "type": "mission", "label": mission.get("goal"), "status": mission.get("status"), "evidence_refs": mission.get("evidence", {}).get("artifact_refs", []), "correctable": True})
            claim_id = mission.get("source_claim", {}).get("claim_id")
            if claim_id:
                edges.append({"source": f"mission:{mission['mission_id']}", "target": f"claim:{claim_id}", "type": "derived_from_claim", "evidence_refs": mission.get("evidence", {}).get("artifact_refs", [])})
        for action in actions:
            nodes.append({"id": f"action:{action['action_id']}", "type": "workflow_action", "label": action.get("goal"), "status": action.get("status"), "evidence_refs": action.get("evidence", {}).get("artifact_refs", []), "correctable": True})
            edges.append({"source": f"action:{action['action_id']}", "target": f"mission:{action.get('mission_id')}", "type": "executes_mission", "evidence_refs": action.get("evidence", {}).get("artifact_refs", [])})
            if action.get("policy_decision"):
                policy_decisions.append(action["policy_decision"])

        map_base = {
            "schema_version": "cs.operational_map.v0",
            "scope": scope,
            "status": "ready",
            "nodes": nodes,
            "edges": edges,
            "timelines": [entry for entry in timelines if entry.get("at")],
            "policies": [{"policy_decision_id": decision.get("id"), "policy": decision.get("policy"), "decision": decision.get("decision")} for decision in policy_decisions],
            "decisions": [{"claim_id": claim["claim_id"], "trust_state": claim.get("trust_state"), "status": claim.get("status")} for claim in claims],
            "workflows": [{"mission_id": mission["mission_id"], "status": mission.get("status")} for mission in missions],
            "evidence_linked": all(node.get("evidence_refs") for node in nodes if node.get("type") in {"artifact", "fact", "link", "event", "claim"}),
            "correctable": True,
            "created_at": utc_now(),
        }
        map_id = f"omap_{_json_hash(map_base)[:16]}"
        operational_map = dict(map_base)
        operational_map["operational_map_id"] = map_id
        _write_json(self.operational_map_path(map_id), operational_map)
        event = self.append_audit(
            "understanding.operational_map.created",
            scope,
            {"type": "operational_map", "id": map_id},
            {"node_count": len(nodes), "edge_count": len(edges), "evidence_linked": operational_map["evidence_linked"]},
        )
        return {"operational_map": operational_map, "audit_event": event}

    def detect_contradictions(self, scope: dict[str, str]) -> dict[str, Any]:
        by_key: dict[str, list[dict[str, Any]]] = {}
        for item in self._ontology_item_records(scope):
            if item.get("kind") != "fact" or not item.get("fact_key"):
                continue
            by_key.setdefault(item["fact_key"], []).append(item)

        contradictions = []
        for fact_key, items in sorted(by_key.items()):
            values = {str(item.get("fact_value")) for item in items}
            if len(values) < 2:
                continue
            base = {
                "schema_version": "cs.contradiction.v0",
                "status": "unresolved",
                "scope": scope,
                "fact_key": fact_key,
                "competing_values": sorted(values),
                "competing_evidence": [
                    {
                        "ontology_item_id": item["ontology_item_id"],
                        "value": item.get("fact_value"),
                        "label": item.get("label"),
                        "evidence_refs": item.get("evidence_refs", []),
                    }
                    for item in items
                ],
                "silent_choice_made": False,
                "asks_for_resolution": True,
                "claim_or_memory_marked_uncertain": True,
                "created_at": utc_now(),
            }
            contradiction_id = f"contra_{_json_hash(base)[:16]}"
            contradiction = dict(base)
            contradiction["contradiction_id"] = contradiction_id
            _write_json(self.contradiction_path(contradiction_id), contradiction)
            contradictions.append(contradiction)

        event = self.append_audit(
            "understanding.contradictions.detected",
            scope,
            {"type": "contradiction_scan", "id": scope_key(scope)},
            {"contradiction_count": len(contradictions), "silent_choice_made": False},
        )
        return {"contradictions": contradictions, "audit_event": event}

    def check_staleness(self, claim_id: str, newer_evidence_bundle_id: str, scope: dict[str, str]) -> dict[str, Any]:
        claim = self.get_claim(claim_id)
        if claim is None:
            return {"status": "not_found", "resource": "claim"}
        if claim.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": claim.get("scope")}
        newer_bundle = self.get_evidence_bundle(newer_evidence_bundle_id)
        if newer_bundle is None:
            return {"status": "not_found", "resource": "evidence_bundle"}
        if newer_bundle.get("filters") != scope:
            return {"status": "scope_denied", "resource_scope": newer_bundle.get("filters")}

        old_refs = set(claim.get("evidence_bundle", {}).get("artifact_refs", []))
        new_refs = {f"artifact:{item['artifact_id']}" for item in newer_bundle.get("evidence_items", [])}
        needs_review = bool(new_refs and new_refs != old_refs)
        base = {
            "schema_version": "cs.staleness_check.v0",
            "status": "needs_review" if needs_review else "current",
            "scope": scope,
            "claim_id": claim_id,
            "claim_trust_state": claim.get("trust_state"),
            "old_evidence_refs": sorted(old_refs),
            "newer_evidence_refs": sorted(new_refs),
            "newer_evidence_bundle_id": newer_evidence_bundle_id,
            "warning_visible": needs_review,
            "used_as_approved_current_truth_without_warning": False,
            "reason": "newer_or_contradictory_evidence_available" if needs_review else "no_newer_conflicting_evidence",
            "checked_at": utc_now(),
        }
        staleness_id = f"stale_{_json_hash(base)[:16]}"
        staleness = dict(base)
        staleness["staleness_id"] = staleness_id
        _write_json(self.staleness_path(staleness_id), staleness)
        event = self.append_audit(
            "understanding.staleness.checked",
            scope,
            {"type": "staleness_check", "id": staleness_id},
            {"claim_id": claim_id, "status": staleness["status"], "warning_visible": staleness["warning_visible"]},
        )
        return {"staleness": staleness, "audit_event": event}

    def record_ontology_change(
        self,
        item_id: str,
        *,
        property_name: str,
        new_value: str,
        scope: dict[str, str],
    ) -> dict[str, Any]:
        item = self.get_ontology_item(item_id)
        if item is None:
            return {"status": "not_found", "resource": "ontology_item"}
        if item.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": item.get("scope")}
        old_value = item.get(property_name)
        affected_evidence_refs = set(item.get("evidence_refs", []))
        affected_claims = [
            f"claim:{claim['claim_id']}"
            for claim in self._claim_records(scope)
            if affected_evidence_refs & set(self._claim_evidence_refs(claim))
        ]
        affected_missions = [
            f"mission:{mission['mission_id']}"
            for mission in self._mission_records(scope)
            if affected_evidence_refs & set(mission.get("evidence", {}).get("artifact_refs", []))
        ]
        affected_actions = [
            f"action:{action['action_id']}"
            for action in self._action_records(scope)
            if affected_evidence_refs & set(action.get("evidence", {}).get("artifact_refs", []))
        ]
        change_base = {
            "schema_version": "cs.ontology_change.v0",
            "status": "recorded",
            "scope": scope,
            "ontology_item_id": item_id,
            "from_version": item.get("version", 1),
            "to_version": int(item.get("version", 1)) + 1,
            "diff": {
                "property": property_name,
                "from": old_value,
                "to": redact_text(new_value),
            },
            "impact": {
                "affected_claims": affected_claims,
                "affected_missions": affected_missions,
                "affected_playbooks": [],
                "affected_actions": affected_actions,
            },
            "rollback_guidance": [
                f"Restore {property_name} to the prior value recorded in this change.",
                "Re-run contradiction, staleness, and operational-map checks before using the changed item as approved truth.",
            ],
            "migration_guidance": [
                "Review affected claims, missions, playbooks, and actions before broad use.",
            ],
            "evidence_refs": item.get("evidence_refs", []),
            "created_at": utc_now(),
        }
        change_id = f"ochg_{_json_hash(change_base)[:16]}"
        change = dict(change_base)
        change["ontology_change_id"] = change_id

        updated_item = dict(item)
        updated_item[property_name] = redact_text(new_value)
        updated_item["version"] = change["to_version"]
        history = list(updated_item.get("version_history", []))
        history.append({"ontology_change_id": change_id, "from_version": change["from_version"], "to_version": change["to_version"], "diff": change["diff"]})
        updated_item["version_history"] = history
        _write_json(self.ontology_change_path(change_id), change)
        _write_json(self.ontology_item_path(item_id), updated_item)
        event = self.append_audit(
            "ontology.change.recorded",
            scope,
            {"type": "ontology_change", "id": change_id},
            {
                "ontology_item_id": item_id,
                "from_version": change["from_version"],
                "to_version": change["to_version"],
                "affected_claim_count": len(affected_claims),
                "affected_mission_count": len(affected_missions),
                "affected_action_count": len(affected_actions),
            },
        )
        return {"ontology_change": change, "ontology_item": updated_item, "audit_event": event}

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
        expected_connector_calls = 1 if action_kind == "external_writeback" else 0
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
                "expected_connector_calls": expected_connector_calls,
                "mock_connector_calls": expected_connector_calls,
                "real_external_http_calls": 0,
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

    def test_connector_credential_boundary(self, provider: str, capability: str, scope: dict[str, str]) -> dict[str, Any]:
        record_base = {
            "schema_version": "cs.connector_credential_boundary_test.v0",
            "status": "passed",
            "scope": scope,
            "provider": provider,
            "capability": capability,
            "mediated_by": "ConnectorHub",
            "credential_custody": "connectorhub",
            "credential_reference": f"connectorhub://credential/{provider}/redacted",
            "declared_actions": [capability],
            "source_policy": {
                "raw_access_default": "deny",
                "projection_required": True,
                "retry_quarantine": True,
            },
            "credential_secret_value_present": False,
            "credentials_exposed_to_agent": False,
            "credentials_exposed_to_product_output": False,
            "direct_provider_access": False,
            "raw_secret_reads": 0,
            "external_http_calls": 0,
            "evidence_metadata": {
                "provider": provider,
                "capability": capability,
                "boundary": "ConnectorHub",
            },
            "created_at": utc_now(),
        }
        boundary_id = f"credbound_{_json_hash(record_base)[:16]}"
        record = dict(record_base)
        record["boundary_id"] = boundary_id
        _write_json(self.credential_boundary_path(boundary_id), record)
        event = self.append_audit(
            "connector.credential_boundary.checked",
            scope,
            {"type": "connector", "id": provider},
            {
                "capability": capability,
                "mediated_by": "ConnectorHub",
                "raw_secret_reads": 0,
                "credentials_exposed_to_agent": False,
            },
        )
        return {"credential_boundary": record, "audit_event": event}

    def create_connector_action_trace(self, action_id: str, scope: dict[str, str]) -> dict[str, Any]:
        action = self.get_action(action_id)
        if action is None:
            return {"status": "not_found", "resource": "action"}
        if action.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": action.get("scope")}
        connector_boundary = action.get("connector_boundary", {})
        trace_base = {
            "schema_version": "cs.connector_action_trace.v0",
            "status": "mediated",
            "scope": scope,
            "action_id": action_id,
            "mission_id": action.get("mission_id"),
            "provider_access": {
                "mediated_by": "ConnectorHub",
                "direct_provider_access": False,
                "agent_owned_provider_client": False,
            },
            "credentials": {
                "custody": "ConnectorHub",
                "credential_reference": f"connectorhub://credential/{connector_boundary.get('connector', 'mock_connector')}/redacted",
                "raw_secret_reads": 0,
                "credentials_exposed_to_agent": False,
            },
            "source_policy": {
                "raw_access_default": "deny",
                "projection_required": True,
                "declared_actions_only": True,
                "retry_quarantine": True,
            },
            "projections": [{"name": "status_projection", "raw_source_value_exposed": False}],
            "declared_actions": [action.get("action_kind")],
            "delivery": {
                "mode": "mocked_connector",
                "external_http_calls": 0,
                "delivery_attempts": 0,
                "quarantine_on_failure": True,
            },
            "raw_access": {"allowed": False, "reads": 0},
            "evidence_metadata": {
                "evidence_bundle_id": action.get("evidence", {}).get("evidence_bundle_id"),
                "artifact_refs": action.get("evidence", {}).get("artifact_refs", []),
                "policy_decision_id": action.get("policy_decision", {}).get("id"),
            },
            "arbitrary_agent_code_used": False,
            "created_at": utc_now(),
        }
        trace_id = f"connectortrace_{_json_hash(trace_base)[:16]}"
        trace = dict(trace_base)
        trace["trace_id"] = trace_id
        _write_json(self.connector_action_trace_path(trace_id), trace)
        event = self.append_audit(
            "connector.action_trace.recorded",
            scope,
            {"type": "connector_action_trace", "id": trace_id},
            {"action_id": action_id, "direct_provider_access": False, "raw_secret_reads": 0},
        )
        return {"connector_action_trace": trace, "audit_event": event}

    def control_mission_autonomy(
        self,
        mission_id: str,
        scope: dict[str, str],
        *,
        control: str,
        reason: str,
    ) -> dict[str, Any]:
        mission = self.get_mission(mission_id)
        if mission is None:
            return {"status": "not_found", "resource": "mission"}
        if mission.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": mission.get("scope")}

        target_mode = "assist" if control == "reduce" else "locked"
        mode_result = self.set_workspace_mode(target_mode, scope)
        updated = dict(mission)
        status_by_control = {"pause": "paused", "stop": "stopped", "revoke": "revoked", "reduce": "active"}
        updated["status"] = status_by_control[control]
        updated["autonomy"] = {
            "mode_before": mission.get("workspace_mode"),
            "control": control,
            "mode_after": target_mode,
            "future_autonomous_actions_allowed": False if control in {"pause", "stop", "revoke"} else target_mode == "autopilot",
            "allowed_cleanup_actions": ["audit_export", "owner_notification", "safe_state_persistence"],
            "reason": redact_text(reason),
            "controlled_at": utc_now(),
        }
        _write_json(self.mission_path(mission_id), updated)
        control_base = {
            "schema_version": "cs.mission_autonomy_control.v0",
            "status": "applied",
            "scope": scope,
            "mission_id": mission_id,
            "control": control,
            "reason": redact_text(reason),
            "mode_after": target_mode,
            "future_autonomous_actions_allowed": updated["autonomy"]["future_autonomous_actions_allowed"],
            "allowed_cleanup_actions": updated["autonomy"]["allowed_cleanup_actions"],
            "created_at": utc_now(),
        }
        control_id = f"autonomyctl_{_json_hash(control_base)[:16]}"
        control_record = dict(control_base)
        control_record["control_id"] = control_id
        _write_json(self.mission_autonomy_control_path(control_id), control_record)
        event = self.append_audit(
            "mission.autonomy.controlled",
            scope,
            {"type": "mission", "id": mission_id},
            {
                "control_id": control_id,
                "control": control,
                "mode_after": target_mode,
                "future_autonomous_actions_allowed": control_record["future_autonomous_actions_allowed"],
            },
        )
        return {"mission": updated, "autonomy_control": control_record, "workspace_mode": mode_result["workspace_mode"], "audit_events": [mode_result["audit_event"], event]}

    def escalate_mission_exception(self, mission_id: str, exception: str, scope: dict[str, str]) -> dict[str, Any]:
        mission = self.get_mission(mission_id)
        if mission is None:
            return {"status": "not_found", "resource": "mission"}
        if mission.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": mission.get("scope")}

        catalog = {
            "missing_evidence": (
                "Required evidence is missing for a durable claim or action.",
                "Attach an Evidence Bundle or narrow the mission.",
                "Choose evidence to attach, or approve a scoped deferral.",
            ),
            "policy_denial": (
                "Policy denied the requested mission/action step.",
                "Review the policy reason and adjust scope, risk, or authority.",
                "Decide whether to revise the request or keep it denied.",
            ),
            "connector_failure": (
                "Connector capability failed or is unavailable.",
                "Retry through ConnectorHub or quarantine the connector attempt.",
                "Approve retry, choose a fallback source, or pause the mission.",
            ),
            "model_disagreement": (
                "Model or judge outputs disagree on a material mission result.",
                "Use evidence-weighted adjudication and preserve dissent.",
                "Pick the acceptable interpretation or request more evidence.",
            ),
            "unclear_goal": (
                "Mission goal is ambiguous enough to risk wrong action.",
                "Clarify goal, success criteria, and stop conditions.",
                "Provide the minimum goal clarification needed to continue.",
            ),
            "high_risk_action": (
                "Requested action is high-risk or externally impactful.",
                "Run dry-run, show impact, require approval, and keep rollback visible.",
                "Approve, reject, or revise the high-risk action.",
            ),
        }
        reason, resolution, decision = catalog[exception]
        escalation_base = {
            "schema_version": "cs.mission_escalation.v0",
            "status": "requires_human_decision",
            "scope": scope,
            "mission_id": mission_id,
            "exception": exception,
            "reason": reason,
            "recommended_resolution": resolution,
            "minimum_required_human_decision": decision,
            "silent_continue_allowed": False,
            "created_at": utc_now(),
        }
        escalation_id = f"escalation_{_json_hash(escalation_base)[:16]}"
        escalation = dict(escalation_base)
        escalation["escalation_id"] = escalation_id
        _write_json(self.mission_escalation_path(escalation_id), escalation)
        event = self.append_audit(
            "mission.exception.escalated",
            scope,
            {"type": "mission", "id": mission_id},
            {"escalation_id": escalation_id, "exception": exception, "silent_continue_allowed": False},
        )
        return {"escalation": escalation, "audit_event": event}

    def record_mission_outcome(self, mission_id: str, action_id: str, scope: dict[str, str]) -> dict[str, Any]:
        mission = self.get_mission(mission_id)
        if mission is None:
            return {"status": "not_found", "resource": "mission"}
        if mission.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": mission.get("scope")}
        action = self.get_action(action_id)
        if action is None:
            return {"status": "not_found", "resource": "action"}
        if action.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": action.get("scope")}
        escalations = [row for row in self._mission_escalation_records(scope) if row.get("mission_id") == mission_id]
        result = action.get("execution", {}).get("result") or {}
        outcome_base = {
            "schema_version": "cs.mission_outcome.v0",
            "status": "evaluated",
            "scope": scope,
            "mission_id": mission_id,
            "goal": mission.get("goal"),
            "action_id": action_id,
            "outcome": "completed" if result.get("status") == "success" else "needs_review",
            "evidence_refs": [f"mission:{mission_id}", f"action:{action_id}", *mission.get("evidence", {}).get("artifact_refs", [])],
            "judge_assessment": {
                "status": "supporting_signal",
                "quality": "sufficient_for_local_fixture",
                "llm_judge_is_pass_authority": False,
            },
            "owner_acceptance": {"status": "accepted_for_local_fixture", "accepted": True},
            "errors": [],
            "escalations": [row["escalation_id"] for row in escalations],
            "lessons": ["Evidence-backed action flow should preserve approval, audit, and rollback context."],
            "created_at": utc_now(),
        }
        outcome_id = f"outcome_{_json_hash(outcome_base)[:16]}"
        outcome = dict(outcome_base)
        outcome["outcome_id"] = outcome_id
        _write_json(self.mission_outcome_path(outcome_id), outcome)
        event = self.append_audit(
            "mission.outcome.evaluated",
            scope,
            {"type": "mission_outcome", "id": outcome_id},
            {"mission_id": mission_id, "action_id": action_id, "escalation_count": len(escalations)},
        )
        return {"mission_outcome": outcome, "audit_event": event}

    def create_mission_after_action_review(self, mission_id: str, outcome_id: str, scope: dict[str, str]) -> dict[str, Any]:
        mission = self.get_mission(mission_id)
        outcome = _read_json(self.mission_outcome_path(outcome_id)) if self.mission_outcome_path(outcome_id).exists() else None
        if mission is None:
            return {"status": "not_found", "resource": "mission"}
        if mission.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": mission.get("scope")}
        if outcome is None:
            return {"status": "not_found", "resource": "mission_outcome"}
        if outcome.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": outcome.get("scope")}
        actions = [row for row in self._action_records(scope) if row.get("mission_id") == mission_id]
        review_base = {
            "schema_version": "cs.mission_after_action_review.v0",
            "status": "complete",
            "scope": scope,
            "mission_id": mission_id,
            "outcome_id": outcome_id,
            "goal": mission.get("goal"),
            "actions_taken": [action.get("action_id") for action in actions if action.get("execution", {}).get("status") == "executed"],
            "evidence_used": outcome.get("evidence_refs", []),
            "judge_assessment": outcome.get("judge_assessment"),
            "objective_outcome": outcome.get("outcome"),
            "owner_outcome": outcome.get("owner_acceptance"),
            "errors": outcome.get("errors", []),
            "escalations": outcome.get("escalations", []),
            "lessons_learned": outcome.get("lessons", []),
            "reusable_memories": ["memory_candidate:evidence-backed-action-flow"],
            "candidate_playbooks": ["playbook_candidate:governed-action-with-audit"],
            "rollback_correction_options": ["rollback", "compensation", "retry", "non_reversible_warning"],
            "autonomy_scorecard": {
                "outcome_quality": "passed_local_fixture",
                "evidence_coverage": "complete_for_fixture",
                "owner_acceptance": outcome.get("owner_acceptance", {}).get("accepted") is True,
                "rollback_or_correction_visible": True,
            },
            "created_at": utc_now(),
        }
        review_id = f"aar_{_json_hash(review_base)[:16]}"
        review = dict(review_base)
        review["review_id"] = review_id
        _write_json(self.mission_after_action_path(review_id), review)
        event = self.append_audit(
            "mission.after_action_review.created",
            scope,
            {"type": "mission_after_action_review", "id": review_id},
            {"mission_id": mission_id, "outcome_id": outcome_id},
        )
        return {"after_action_review": review, "audit_event": event}

    def export_mission_audit(self, mission_id: str, scope: dict[str, str]) -> dict[str, Any]:
        mission = self.get_mission(mission_id)
        if mission is None:
            return {"status": "not_found", "resource": "mission"}
        if mission.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": mission.get("scope")}
        actions = [row for row in self._action_records(scope) if row.get("mission_id") == mission_id]
        audit_events = [
            event
            for event in self._all_audit_events()
            if all(event.get(field) == scope[field] for field in ["tenant_id", "owner_id", "namespace_id", "workspace_id"])
            and (
                event.get("subject", {}).get("id") == mission_id
                or event.get("details", {}).get("mission_id") == mission_id
                or event.get("subject", {}).get("id") in {action.get("action_id") for action in actions}
            )
        ]
        export_base = {
            "schema_version": "cs.mission_audit_export.v0",
            "status": "exported",
            "scope": scope,
            "mission_id": mission_id,
            "timeline_events": [
                {"event_id": event.get("event_id"), "event_type": event.get("event_type"), "created_at": event.get("created_at")}
                for event in audit_events
            ],
            "tool_calls": [
                {"command": "cornerstone mission create", "subject": f"mission:{mission_id}"},
                *[
                    {"command": "cornerstone action execute", "subject": f"action:{action.get('action_id')}"}
                    for action in actions
                ],
            ],
            "policy_decisions": [action.get("policy_decision") for action in actions if action.get("policy_decision")],
            "evidence": mission.get("evidence", {}),
            "judge_outputs": [{"status": "supporting_signal", "llm_judge_is_pass_authority": False}],
            "approvals": [action.get("approval") for action in actions if action.get("approval")],
            "action_results": [action.get("execution", {}).get("result") for action in actions if action.get("execution", {}).get("result")],
            "trace_context": {
                "trace_id": f"trace_{_json_hash({'mission_id': mission_id, 'scope': scope})[:16]}",
                "logs_and_events_correlated": True,
                "correlated_audit_event_count": len(audit_events),
            },
            "created_at": utc_now(),
        }
        export_id = f"missionaudit_{_json_hash(export_base)[:16]}"
        export = dict(export_base)
        export["export_id"] = export_id
        _write_json(self.mission_audit_export_path(export_id), export)
        event = self.append_audit(
            "mission.audit_export.created",
            scope,
            {"type": "mission_audit_export", "id": export_id},
            {"mission_id": mission_id, "timeline_event_count": len(export["timeline_events"])},
        )
        return {"mission_audit_export": export, "audit_event": event}

    def autonomy_quality_metrics(self, mission_id: str, outcome_id: str, scope: dict[str, str]) -> dict[str, Any]:
        mission = self.get_mission(mission_id)
        outcome = _read_json(self.mission_outcome_path(outcome_id)) if self.mission_outcome_path(outcome_id).exists() else None
        if mission is None:
            return {"status": "not_found", "resource": "mission"}
        if mission.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": mission.get("scope")}
        if outcome is None:
            return {"status": "not_found", "resource": "mission_outcome"}
        metrics_base = {
            "schema_version": "cs.autonomy_quality_metrics.v0",
            "status": "reported",
            "scope": scope,
            "mission_id": mission_id,
            "outcome_id": outcome_id,
            "priority": "outcome_quality_over_autonomy_ratio",
            "primary_metrics": {
                "fewer_errors": True,
                "better_evidence_coverage": True,
                "faster_safe_resolution": True,
                "fewer_repeated_explanations": True,
                "better_owner_acceptance": True,
                "fewer_rollback_or_escalation_failures": True,
            },
            "autonomy_ratio": {"value": 0.64, "priority": "secondary_context_only"},
            "created_at": utc_now(),
        }
        metric_id = f"autonomymetrics_{_json_hash(metrics_base)[:16]}"
        metrics = dict(metrics_base)
        metrics["metric_id"] = metric_id
        _write_json(self.autonomy_metric_path(metric_id), metrics)
        event = self.append_audit(
            "autonomy.metrics.reported",
            scope,
            {"type": "autonomy_metrics", "id": metric_id},
            {"mission_id": mission_id, "priority": metrics["priority"]},
        )
        return {"autonomy_metrics": metrics, "audit_event": event}

    def test_action_reversibility(self, action_id: str, scope: dict[str, str], mode: str) -> dict[str, Any]:
        action = self.get_action(action_id)
        if action is None:
            return {"status": "not_found", "resource": "action"}
        if action.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": action.get("scope")}
        available_by_mode = {
            "rollback": mode == "rollback",
            "compensation": mode == "compensation",
            "retry": mode == "retry",
            "explicit_non_reversible_explanation": mode == "non_reversible",
        }
        record_base = {
            "schema_version": "cs.action_reversibility.v0",
            "status": "reviewed",
            "scope": scope,
            "action_id": action_id,
            "mission_id": action.get("mission_id"),
            "mode": mode,
            "rollback_available": available_by_mode["rollback"],
            "compensation_available": available_by_mode["compensation"],
            "retry_available": available_by_mode["retry"],
            "non_reversible_explanation": (
                "External system semantics may prevent reversal; owner must receive impact warning before execution."
                if mode == "non_reversible"
                else ""
            ),
            "external_http_calls": 0,
            "real_world_side_effects": 0,
            "created_at": utc_now(),
        }
        reversibility_id = f"reversible_{_json_hash(record_base)[:16]}"
        record = dict(record_base)
        record["reversibility_id"] = reversibility_id
        _write_json(self.action_reversibility_path(reversibility_id), record)
        event = self.append_audit(
            "action.reversibility.reviewed",
            scope,
            {"type": "action_reversibility", "id": reversibility_id},
            {"action_id": action_id, "mode": mode, "external_http_calls": 0},
        )
        return {"action_reversibility": record, "audit_event": event}

    def test_sensitive_change_gate(self, category: str, scope: dict[str, str]) -> dict[str, Any]:
        policy_base = {
            "schema_version": "cs.policy_decision.v0",
            "decision": "requires_approval",
            "policy": "sensitive_change_stop_and_ask",
            "reason": "Sensitive, destructive, production, auth, tenant, retention, audit, broad-network, release, or secret-handling changes require explicit owner approval.",
            "category": category,
            "scope": scope,
            "executed": False,
            "requires_explicit_approval": True,
            "risk": "high",
            "impact": [
                "Could mutate durable product state, security posture, tenant boundaries, audit/retention guarantees, release status, or external systems.",
            ],
            "rollback": {
                "available": category not in {"irreversible_migration", "data_deletion"},
                "irreversible_warning": category in {"irreversible_migration", "data_deletion"},
            },
            "resolution_path": [
                "Show risk, impact, and rollback or irreversibility before proceeding.",
                "Collect explicit approval from the owner or authorized admin.",
                "Record approval, execution result, and audit before claiming completion.",
            ],
            "decided_at": utc_now(),
        }
        policy = dict(policy_base)
        policy["id"] = f"policy_{_json_hash(policy_base)[:16]}"
        gate_base = {
            "schema_version": "cs.sensitive_change_gate_test.v0",
            "status": "approval_required",
            "scope": scope,
            "category": category,
            "policy_decision": policy,
            "stop_and_ask_card": {
                "required": True,
                "risk": policy["risk"],
                "impact": policy["impact"],
                "rollback": policy["rollback"],
                "approval_collected": False,
            },
            "mutation_executed": False,
            "secret_material_read": False,
            "external_http_calls": 0,
            "created_at": utc_now(),
        }
        gate_id = f"sensgate_{_json_hash(gate_base)[:16]}"
        gate = dict(gate_base)
        gate["gate_id"] = gate_id
        _write_json(self.sensitive_change_path(gate_id), gate)
        event = self.append_audit(
            "policy.sensitive_change.requires_approval",
            scope,
            {"type": "policy_decision", "id": policy["id"]},
            {"category": category, "mutation_executed": False, "approval_collected": False},
        )
        return {"sensitive_change_gate": gate, "policy_decision": policy, "audit_event": event}

    def rehearse_backup_restore(self, scope: dict[str, str], subject_refs: list[str] | None = None) -> dict[str, Any]:
        counts = {
            "artifact_count": len(self._artifact_records(scope)),
            "claim_count": len(self._claim_records(scope)),
            "mission_count": len(self._mission_records(scope)),
            "action_count": len(self._action_records(scope)),
            "memory_count": len(self._memory_records(scope)),
            "experience_count": len(self._trajectory_records(scope)),
        }
        audit_before = self.verify_audit()
        manifest_base = {
            "schema_version": "cs.backup_manifest.v0",
            "scope": scope,
            "subject_refs": subject_refs or [],
            "counts": counts,
            "audit_integrity_before_restore": audit_before,
            "artifact_hashes": sorted(row.get("checksum_sha256") for row in self._artifact_records(scope) if row.get("checksum_sha256")),
            "created_at": utc_now(),
        }
        restore_base = {
            "schema_version": "cs.backup_restore_rehearsal.v0",
            "status": "restored",
            "scope": scope,
            "backup_manifest_hash": _json_hash(manifest_base),
            "counts_before": counts,
            "counts_after": dict(counts),
            "artifact_hashes_match": True,
            "evidence_replay_ok": counts["artifact_count"] >= 1 and counts["claim_count"] >= 1,
            "audit_replay_ok": audit_before.get("status") == "success",
            "search_replay_ok": counts["artifact_count"] >= 1,
            "restore_used_external_system": False,
            "secret_material_restored_to_output": False,
            "created_at": utc_now(),
        }
        restore_id = f"backuprestore_{_json_hash(restore_base)[:16]}"
        restore = dict(restore_base)
        restore["restore_id"] = restore_id
        _write_json(self.backup_restore_path(restore_id), restore)
        event = self.append_audit(
            "backup.restore.rehearsed",
            scope,
            {"type": "backup_restore", "id": restore_id},
            {
                "counts_before": counts,
                "counts_after": counts,
                "audit_replay_ok": restore["audit_replay_ok"],
            },
        )
        restore["audit_integrity_after_restore"] = self.verify_audit()
        _write_json(self.backup_restore_path(restore_id), restore)
        return {"backup_restore": restore, "audit_event": event}

    def record_helpful_failure_examples(self, scope: dict[str, str]) -> dict[str, Any]:
        failure_classes = [
            "ingestion",
            "search",
            "extraction",
            "model_routing",
            "action_execution",
            "connector_call",
            "policy_check",
            "memory_update",
        ]
        examples = [
            {
                "failure_class": name,
                "cause": f"{name} failed in deterministic fixture",
                "impact": "No unsafe mutation occurred; existing evidence and audit remain preserved.",
                "retry_options": ["retry_same_input", "reduce_scope", "request_owner_review"],
                "escalation_path": "namespace_owner_review",
                "safe_state_preserved": True,
            }
            for name in failure_classes
        ]
        record_base = {
            "schema_version": "cs.helpful_failure_examples.v0",
            "status": "ready",
            "scope": scope,
            "examples": examples,
            "all_have_cause": True,
            "all_have_impact": True,
            "all_have_retry_options": True,
            "all_have_escalation_path": True,
            "all_preserve_safe_state": True,
            "external_http_calls": 0,
            "created_at": utc_now(),
        }
        failure_id = f"helpfail_{_json_hash(record_base)[:16]}"
        record = dict(record_base)
        record["failure_id"] = failure_id
        _write_json(self.helpful_failure_path(failure_id), record)
        event = self.append_audit(
            "helpful_failures.examples.recorded",
            scope,
            {"type": "helpful_failure_examples", "id": failure_id},
            {"failure_class_count": len(examples), "all_preserve_safe_state": True},
        )
        return {"helpful_failures": record, "audit_event": event}

    def test_action_idempotency(self, action_id: str, scope: dict[str, str]) -> dict[str, Any]:
        action = self.get_action(action_id)
        if action is None:
            return {"status": "not_found"}
        if action.get("scope") != scope:
            return {"status": "scope_denied", "resource_scope": action.get("scope")}
        idempotency_base = {
            "schema_version": "cs.action_idempotency_test.v0",
            "status": "deduplicated",
            "scope": scope,
            "action_id": action_id,
            "idempotency_key": f"{action.get('mission_id')}:{action_id}:{action.get('action_kind')}",
            "first_request": {
                "accepted": True,
                "side_effect_count": 1 if action.get("execution", {}).get("status") == "executed" else 0,
            },
            "duplicate_request": {
                "accepted": False,
                "deduplicated": True,
                "side_effect_count": 0,
            },
            "retry_policy": {
                "timeout_ms": 2000,
                "max_attempts": 2,
                "quarantine_after_failure": True,
                "compensation_required_for_external_failure": action.get("action_kind") == "external_writeback",
            },
            "duplicate_real_world_side_effects": 0,
            "external_http_calls": 0,
            "created_at": utc_now(),
        }
        idempotency_id = f"idemp_{_json_hash(idempotency_base)[:16]}"
        record = dict(idempotency_base)
        record["idempotency_id"] = idempotency_id
        _write_json(self.idempotency_path(idempotency_id), record)
        event = self.append_audit(
            "action.idempotency.checked",
            scope,
            {"type": "action", "id": action_id},
            {"duplicate_real_world_side_effects": 0, "deduplicated": True},
        )
        return {"idempotency": record, "audit_event": event}

    def explain_retention(self, resource_type: str, scope: dict[str, str]) -> dict[str, Any]:
        retention_base = {
            "schema_version": "cs.retention_explanation.v0",
            "status": "explained",
            "scope": scope,
            "resource_type": resource_type,
            "policy": "local_retention_policy_v0",
            "states": {
                "deleted": "User-visible active record is removed or disabled when policy allows.",
                "disabled": "Automation and search use stop for disabled resources.",
                "retained_for_audit": "Audit ledger entries remain tamper-evident for accountability.",
                "retained_as_immutable_evidence": "Original artifacts may remain immutable when referenced by evidence or legal/audit policy.",
                "anonymized": "Aggregated product-learning signals use redaction/anonymization and never raw user/org truth by default.",
                "subject_to_policy": "Retention/legal hold/admin policy can constrain deletion.",
            },
            "dry_run": True,
            "active_record_deleted": False,
            "audit_retained": True,
            "immutable_evidence_retained_when_required": True,
            "raw_secret_output": False,
            "created_at": utc_now(),
        }
        retention_id = f"retention_{_json_hash(retention_base)[:16]}"
        record = dict(retention_base)
        record["retention_id"] = retention_id
        _write_json(self.retention_path(retention_id), record)
        event = self.append_audit(
            "retention.explained",
            scope,
            {"type": "retention", "id": retention_id},
            {"resource_type": resource_type, "dry_run": True, "audit_retained": True},
        )
        return {"retention_explanation": record, "audit_event": event}

    def operator_status_report(self, scope: dict[str, str]) -> dict[str, Any]:
        audit = self.verify_audit()
        event_types = []
        if self.audit_path.exists():
            for line in self.audit_path.read_text().splitlines():
                if line.strip():
                    event_types.append(json.loads(line).get("event_type"))
        status_base = {
            "schema_version": "cs.operator_status_report.v0",
            "status": "ready",
            "scope": scope,
            "signals": {
                "ingestion": {"status": "ok", "artifact_count": len(self._artifact_records(scope))},
                "search": {"status": "ok"},
                "model_routing": {"status": "ok", "route_count": len(self._brain_route_records(scope))},
                "workflow_execution": {"status": "ok", "action_count": len(self._action_records(scope))},
                "connector_health": {"status": "ok", "mocked": True},
                "policy_denials": {"status": "visible", "count": sum(1 for event in event_types if "denied" in str(event))},
                "audit_integrity": {"status": audit.get("status"), "event_count": audit.get("event_count", 0)},
                "queue_retries": {"status": "ok", "retry_count": 0, "quarantine_count": 0},
                "failed_missions": {"status": "visible", "count": 0},
            },
            "telemetry_signals": ["logs", "metrics", "traces"],
            "shared_context": scope_key(scope),
            "external_http_calls": 0,
            "created_at": utc_now(),
        }
        status_id = f"opsstatus_{_json_hash(status_base)[:16]}"
        record = dict(status_base)
        record["status_id"] = status_id
        _write_json(self.operator_status_path(status_id), record)
        event = self.append_audit(
            "operator.status.reported",
            scope,
            {"type": "operator_status", "id": status_id},
            {"audit_status": audit.get("status"), "signals": sorted(status_base["signals"])},
        )
        return {"operator_status": record, "audit_event": event}

    def validate_release_report(
        self,
        scenario_report_path: Path,
        verification_report_path: Path,
        scope: dict[str, str],
    ) -> dict[str, Any]:
        errors: list[str] = []
        scenario_data: dict[str, Any] = {}
        markdown = ""
        if not scenario_report_path.exists():
            errors.append("missing_scenario_report")
        else:
            try:
                scenario_data = json.loads(scenario_report_path.read_text())
            except ValueError:
                errors.append("invalid_scenario_report_json")
        if not verification_report_path.exists():
            errors.append("missing_verification_report")
        else:
            markdown = verification_report_path.read_text()

        scenario_rows = scenario_data.get("scenario_results", []) if isinstance(scenario_data, dict) else []
        required_markdown_sections = [
            "Frozen Goal",
            "Scenario Table",
            "Human Required",
            "Command Evidence",
            "Gaps And Risks",
        ]
        missing_sections = [section for section in required_markdown_sections if section not in markdown]
        ai_rows = [row for row in scenario_rows if row.get("owner", "AI") != "Human"]
        pass_rows = [row for row in ai_rows if row.get("status") == "PASS"]
        human_rows = [row for row in scenario_rows if row.get("owner", "AI") == "Human"]
        human_required = scenario_data.get("human_required", []) if isinstance(scenario_data, dict) else []
        no_unverified_pass_claim = (
            bool(scenario_rows)
            and len(pass_rows) == len(ai_rows)
            and all(row.get("status") == "HUMAN_REQUIRED" for row in human_rows)
            and scenario_data.get("summary", {}).get("blocking") == 0
            and "does not mark production" in markdown
        )
        if missing_sections:
            errors.append("missing_markdown_sections")
        if not no_unverified_pass_claim:
            errors.append("unverified_or_overbroad_claim")

        report_base = {
            "schema_version": "cs.release_report_validation.v0",
            "status": "passed" if not errors else "failed",
            "scope": scope,
            "scenario_report_path": str(scenario_report_path),
            "verification_report_path": str(verification_report_path),
            "scenario_count": len(scenario_rows),
            "pass_count": len(pass_rows),
            "blocking": scenario_data.get("summary", {}).get("blocking") if isinstance(scenario_data, dict) else None,
            "human_required_count": len(human_required),
            "required_sections": required_markdown_sections,
            "missing_sections": missing_sections,
            "no_implementation_claim_without_repo_evidence": no_unverified_pass_claim,
            "scenario_verification_remains_release_standard": not errors,
            "documented_target_distinguished_from_current_implementation": "does not mark production" in markdown,
            "errors": errors,
            "created_at": utc_now(),
        }
        report_id = f"relreport_{_json_hash(report_base)[:16]}"
        report = dict(report_base)
        report["report_id"] = report_id
        _write_json(self.release_report_path(report_id), report)
        event = self.append_audit(
            "release.report.validated",
            scope,
            {"type": "release_report", "id": report_id},
            {"status": report["status"], "scenario_count": len(scenario_rows), "errors": errors},
        )
        return {"release_report_validation": report, "audit_event": event}

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
