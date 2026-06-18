package cornerstone.vs2

default allow := false

valid_schema if {
	input.schema_version == "cs.policy_input.vs2.v0"
	is_string(input.subject.principal_id)
	is_string(input.subject.role)
	is_string(input.tenant_id)
	is_string(input.namespace_id)
	is_string(input.workspace_id)
	is_string(input.resource.resource_id)
	is_string(input.resource.tenant_id)
	is_string(input.resource.namespace_id)
	is_string(input.action)
	is_string(input.classification)
	is_string(input.risk)
}

same_scope if {
	input.tenant_id == input.resource.tenant_id
	input.namespace_id == input.resource.namespace_id
}

role_allowed if {
	input.subject.role == "owner"
}

role_allowed if {
	input.subject.role == "admin"
}

role_allowed if {
	input.subject.role == "member"
	input.action == "read"
}

revoked if {
	input.subject.revoked == true
}

risk_allowed if {
	input.risk != "high"
}

classification_allowed if {
	input.classification != "secret"
}

capability_allowed if {
	input.capability.declared == true
	input.capability.connectorhub_mediated == true
}

allow if {
	valid_schema
	same_scope
	role_allowed
	risk_allowed
	classification_allowed
	capability_allowed
	not revoked
	input.policy_path != "unknown"
}

deny contains "invalid_schema" if {
	not valid_schema
}

deny contains "cross_tenant_scope" if {
	valid_schema
	not same_scope
}

deny contains "role_not_allowed" if {
	valid_schema
	not role_allowed
}

deny contains "revoked_principal" if {
	valid_schema
	revoked
}

deny contains "high_risk_requires_approval" if {
	valid_schema
	input.risk == "high"
}

deny contains "secret_classification_denied" if {
	valid_schema
	input.classification == "secret"
}

deny contains "connectorhub_capability_required" if {
	valid_schema
	not capability_allowed
}

deny contains "unknown_policy_default_deny" if {
	valid_schema
	input.policy_path == "unknown"
}

decision := {
	"decision": "allow",
	"reason_codes": [],
	"policy_path": "cornerstone.vs2/allow",
	"bundle_revision": "vs2-rego-local-v1",
} if {
	allow
}

decision := {
	"decision": "deny",
	"reason_codes": reasons,
	"policy_path": "cornerstone.vs2/deny",
	"bundle_revision": "vs2-rego-local-v1",
} if {
	not allow
	reasons := [reason | reason := deny[_]]
}
