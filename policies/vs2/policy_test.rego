package cornerstone.vs2

base_input := {
	"schema_version": "cs.policy_input.vs2.v0",
	"tenant_id": "tenant_a",
	"namespace_id": "personal",
	"workspace_id": "default",
	"action": "read",
	"classification": "internal",
	"risk": "low",
	"policy_path": "artifact.read",
	"subject": {
		"principal_id": "user_a",
		"role": "owner",
		"revoked": false,
	},
	"resource": {
		"resource_id": "artifact_a",
		"tenant_id": "tenant_a",
		"namespace_id": "personal",
	},
	"capability": {
		"declared": true,
		"connectorhub_mediated": true,
	},
}

test_owner_read_allowed if {
	d := decision with input as base_input
	d.decision == "allow"
	d.bundle_revision == "vs2-rego-local-v1"
}

test_member_write_denied if {
	i := object.union(base_input, {"action": "write", "subject": object.union(base_input.subject, {"role": "member"})})
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
	i := object.union(base_input, {"classification": "secret"})
	d := decision with input as i
	d.decision == "deny"
	"secret_classification_denied" in d.reason_codes
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
