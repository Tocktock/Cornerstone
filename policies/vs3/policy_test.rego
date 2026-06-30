package cornerstone.vs3

base_input := {
	"schema_version": "cs.policy_input.vs3.v0",
	"trace_id": "trace_vs3_rego_test",
	"subject": {
		"principal_id": "principal_alice",
		"roles": ["member"],
		"membership_revision": "memrev-alpha-001",
		"revoked": false,
	},
	"scope": {
		"tenant_id": "tenant_alpha",
		"namespace_id": "personal",
		"workspace_id": "alpha-home",
	},
	"resource": {
		"resource_id": "artifact_alpha_001",
		"tenant_id": "tenant_alpha",
		"namespace_id": "personal",
		"classification": "internal",
	},
	"action": "artifact.read",
	"risk": "low",
	"policy_path": "artifact.read",
	"mission_authority": {
		"mission_id": "mission_alpha",
		"authorized": true,
		"authority_ref": "mission-alpha-grant",
	},
	"data_scope": {
		"scope": "same_workspace",
		"purpose": "evidence_read",
	},
	"approval": {
		"required": false,
		"status": "not_required",
	},
	"capability": {
		"declared": true,
		"connectorhub_mediated": true,
		"tool_manifest_ref": "tool:none",
	},
	"model_policy": {
		"provider": "local_test",
		"route": "local_test.default",
		"sensitivity": "internal",
	},
	"environment": {
		"deployment": "local_test",
		"workspace_mode": "personal",
		"network_zone": "loopback",
	},
}

test_allow_artifact_read if {
	result := data.cornerstone.vs3.decision with input as base_input
	result.decision == "allow"
	result.bundle_revision == "vs3-rego-local-v1"
}

test_model_policy_denied if {
	result := data.cornerstone.vs3.decision with input as object.union(base_input, {"model_policy": {"provider": "external", "route": "external_unapproved", "sensitivity": "restricted"}})
	result.decision == "deny"
	result.reason_codes[_] == "model_policy_denied"
}

test_unknown_policy_defaults_deny if {
	result := data.cornerstone.vs3.decision with input as object.union(base_input, {"policy_path": "unknown"})
	result.decision == "deny"
	result.reason_codes[_] == "unknown_policy_default_deny"
}

test_unexpected_authoritative_attribute_denied if {
	result := data.cornerstone.vs3.decision with input as object.union(base_input, {"subject": object.union(base_input.subject, {"tenant_id": "tenant_forged"})})
	result.decision == "deny"
	result.reason_codes[_] == "unexpected_authoritative_attribute"
}
