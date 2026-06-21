package cornerstone.vs2

base_input := {
	"schema_version": "cs.policy_input.vs2.v1",
	"trace_id": "trace_policy_test",
	"scope": {
		"tenant_id": "tenant_a",
		"namespace_id": "personal",
		"workspace_id": "default",
	},
	"action": "artifact.read",
	"risk": "low",
	"policy_path": "artifact.read",
	"subject": {
		"principal_id": "user_a",
		"roles": ["owner"],
		"membership_revision": "memrev-001",
		"revoked": false,
	},
	"resource": {
		"resource_id": "artifact_a",
		"tenant_id": "tenant_a",
		"namespace_id": "personal",
		"classification": "internal",
	},
	"mission_authority": {
		"mission_id": "mission_alpha",
		"authorized": true,
		"authority_ref": "authority_alpha_owner",
	},
	"data_scope": {
		"scope": "tenant",
		"purpose": "artifact_read",
	},
	"capability": {
		"declared": true,
		"connectorhub_mediated": true,
	},
	"approval": {
		"required": false,
		"status": "not_required",
	},
	"environment": {
		"deployment": "local",
		"workspace_mode": "assist",
	},
}

test_owner_read_allowed if {
	d := decision with input as base_input
	d.decision == "allow"
	d.bundle_revision == "vs2-rego-local-v1"
}

test_member_write_denied if {
	i := object.union(base_input, {"action": "artifact.write", "subject": object.union(base_input.subject, {"roles": ["member"]})})
	d := decision with input as i
	d.decision == "deny"
	"role_not_allowed" in d.reason_codes
}

test_cross_tenant_scope_denied if {
	i := object.union(base_input, {"resource": object.union(base_input.resource, {"tenant_id": "tenant_b"})})
	d := decision with input as i
	d.decision == "deny"
	"cross_tenant_scope" in d.reason_codes
}

test_revoked_principal_denied if {
	i := object.union(base_input, {"subject": object.union(base_input.subject, {"revoked": true})})
	d := decision with input as i
	d.decision == "deny"
	"revoked_principal" in d.reason_codes
}

test_high_risk_requires_approval if {
	i := object.union(base_input, {"risk": "high"})
	d := decision with input as i
	d.decision == "deny"
	"high_risk_requires_approval" in d.reason_codes
}

test_secret_classification_denied if {
	i := object.union(base_input, {"resource": object.union(base_input.resource, {"classification": "secret"})})
	d := decision with input as i
	d.decision == "deny"
	"secret_classification_denied" in d.reason_codes
}

test_mission_authority_required if {
	i := object.union(base_input, {"mission_authority": object.union(base_input.mission_authority, {"authorized": false})})
	d := decision with input as i
	d.decision == "deny"
	"mission_authority_required" in d.reason_codes
}

test_cross_tenant_data_scope_denied if {
	i := object.union(base_input, {"data_scope": object.union(base_input.data_scope, {"scope": "cross_tenant"})})
	d := decision with input as i
	d.decision == "deny"
	"data_scope_denied" in d.reason_codes
}

test_external_workspace_mode_denied if {
	i := object.union(base_input, {"environment": object.union(base_input.environment, {"workspace_mode": "external"})})
	d := decision with input as i
	d.decision == "deny"
	"workspace_mode_denied" in d.reason_codes
}

test_unexpected_authoritative_attribute_denied if {
	i := object.union(base_input, {"subject": object.union(base_input.subject, {"tenant_id": "tenant_b"})})
	d := decision with input as i
	d.decision == "deny"
	"unexpected_authoritative_attribute" in d.reason_codes
}

test_high_risk_approved_allowed if {
	i := object.union(base_input, {"risk": "high", "approval": {"required": true, "status": "approved"}})
	d := decision with input as i
	d.decision == "allow"
}

test_connectorhub_capability_required if {
	i := object.union(base_input, {"capability": {"declared": false, "connectorhub_mediated": false}})
	d := decision with input as i
	d.decision == "deny"
	"connectorhub_capability_required" in d.reason_codes
}

test_unknown_policy_default_deny if {
	i := object.union(base_input, {"policy_path": "unknown"})
	d := decision with input as i
	d.decision == "deny"
	"unknown_policy_default_deny" in d.reason_codes
}

test_invalid_schema_fails_closed if {
	i := object.remove(base_input, {"schema_version"})
	d := decision with input as i
	d.decision == "deny"
	"invalid_schema" in d.reason_codes
}
