package cornerstone.vs3

default allow := false

valid_schema if {
	input.schema_version == "cs.policy_input.vs3.v0"
	is_string(input.trace_id)
	is_string(input.subject.principal_id)
	is_array(input.subject.roles)
	is_string(input.subject.membership_revision)
	is_boolean(input.subject.revoked)
	is_string(input.scope.tenant_id)
	is_string(input.scope.namespace_id)
	is_string(input.scope.workspace_id)
	is_string(input.resource.resource_id)
	is_string(input.resource.tenant_id)
	is_string(input.resource.namespace_id)
	is_string(input.resource.classification)
	is_string(input.action)
	is_string(input.risk)
	is_string(input.policy_path)
	is_string(input.mission_authority.mission_id)
	is_boolean(input.mission_authority.authorized)
	is_string(input.mission_authority.authority_ref)
	is_string(input.data_scope.scope)
	is_string(input.data_scope.purpose)
	is_boolean(input.capability.declared)
	is_boolean(input.capability.connectorhub_mediated)
	is_string(input.capability.tool_manifest_ref)
	is_string(input.model_policy.provider)
	is_string(input.model_policy.route)
	is_string(input.model_policy.sensitivity)
	is_boolean(input.approval.required)
	is_string(input.approval.status)
	is_string(input.environment.deployment)
	is_string(input.environment.workspace_mode)
	is_string(input.environment.network_zone)
}

same_scope if {
	input.scope.tenant_id == input.resource.tenant_id
	input.scope.namespace_id == input.resource.namespace_id
}

role_allowed if {
	input.subject.roles[_] == "owner"
}

role_allowed if {
	input.subject.roles[_] == "admin"
}

role_allowed if {
	input.subject.roles[_] == "member"
	input.action == "artifact.read"
}

revoked if {
	input.subject.revoked == true
}

risk_allowed if {
	input.risk != "high"
}

risk_allowed if {
	input.risk == "high"
	input.approval.status == "approved"
}

classification_allowed if {
	input.resource.classification != "secret"
}

mission_allowed if {
	input.mission_authority.authorized == true
}

data_scope_allowed if {
	input.data_scope.scope != "cross_tenant"
}

workspace_mode_allowed if {
	input.environment.workspace_mode != "external"
}

capability_allowed if {
	input.capability.declared == true
	input.capability.connectorhub_mediated == true
}

model_policy_allowed if {
	input.model_policy.provider == "local_test"
	input.model_policy.sensitivity != "restricted"
	input.model_policy.route != "external_unapproved"
}

model_policy_allowed if {
	input.model_policy.provider == "approved_internal"
	input.model_policy.sensitivity == "internal"
}

unexpected_authoritative_attrs if {
	object.get(input.subject, "tenant_id", null) != null
}

unexpected_authoritative_attrs if {
	object.get(input.subject, "role_override", null) != null
}

unexpected_authoritative_attrs if {
	object.get(input.model_policy, "caller_authorized", null) != null
}

allow if {
	valid_schema
	same_scope
	role_allowed
	risk_allowed
	classification_allowed
	mission_allowed
	data_scope_allowed
	workspace_mode_allowed
	capability_allowed
	model_policy_allowed
	not revoked
	not unexpected_authoritative_attrs
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
	input.approval.status != "approved"
}

deny contains "secret_classification_denied" if {
	valid_schema
	input.resource.classification == "secret"
}

deny contains "mission_authority_required" if {
	valid_schema
	not mission_allowed
}

deny contains "data_scope_denied" if {
	valid_schema
	not data_scope_allowed
}

deny contains "workspace_mode_denied" if {
	valid_schema
	not workspace_mode_allowed
}

deny contains "connectorhub_capability_required" if {
	valid_schema
	not capability_allowed
}

deny contains "model_policy_denied" if {
	valid_schema
	not model_policy_allowed
}

deny contains "unexpected_authoritative_attribute" if {
	valid_schema
	unexpected_authoritative_attrs
}

deny contains "unknown_policy_default_deny" if {
	valid_schema
	input.policy_path == "unknown"
}

decision := {
	"decision": "allow",
	"reason_codes": [],
	"policy_path": "cornerstone.vs3/allow",
	"bundle_revision": "vs3-rego-local-v1",
} if {
	allow
}

decision := {
	"decision": "deny",
	"reason_codes": reasons,
	"policy_path": "cornerstone.vs3/deny",
	"bundle_revision": "vs3-rego-local-v1",
} if {
	not allow
	reasons := [reason | reason := deny[_]]
}
