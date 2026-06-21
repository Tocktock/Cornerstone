CREATE SCHEMA IF NOT EXISTS cs;

CREATE ROLE cornerstone_schema_owner NOLOGIN NOSUPERUSER NOBYPASSRLS;
CREATE ROLE cornerstone_app LOGIN NOSUPERUSER NOBYPASSRLS;
CREATE ROLE cornerstone_identity NOLOGIN NOSUPERUSER NOBYPASSRLS;
CREATE ROLE cornerstone_migrator LOGIN NOSUPERUSER NOBYPASSRLS;
CREATE ROLE cornerstone_maintenance LOGIN NOSUPERUSER NOBYPASSRLS;
CREATE ROLE cornerstone_auditor NOLOGIN NOSUPERUSER NOBYPASSRLS;

GRANT USAGE ON SCHEMA cs TO cornerstone_app;

CREATE TABLE IF NOT EXISTS cs.principals (
  principal_id text PRIMARY KEY,
  display_name text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS cs.tenants (
  tenant_id text PRIMARY KEY,
  display_name text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS cs.memberships (
  membership_id text PRIMARY KEY,
  principal_id text NOT NULL REFERENCES cs.principals(principal_id),
  tenant_id text NOT NULL REFERENCES cs.tenants(tenant_id),
  namespace_id text NOT NULL,
  workspace_id text NOT NULL,
  owner_id text NOT NULL,
  roles text[] NOT NULL,
  membership_revision text NOT NULL,
  session_version integer NOT NULL DEFAULT 1,
  revoked_at timestamptz
);

ALTER TABLE cs.memberships ENABLE ROW LEVEL SECURITY;
ALTER TABLE cs.memberships FORCE ROW LEVEL SECURITY;

CREATE OR REPLACE FUNCTION cs.resolve_membership(
  p_principal_id text,
  p_membership_id text,
  p_session_version integer
) RETURNS TABLE (
  principal_id text,
  tenant_id text,
  namespace_id text,
  workspace_id text,
  owner_id text,
  roles text[],
  membership_revision text,
  session_version integer,
  revoked boolean
) LANGUAGE sql
SECURITY DEFINER
SET search_path = cs, pg_temp
AS $$
  SELECT
    m.principal_id,
    m.tenant_id,
    m.namespace_id,
    m.workspace_id,
    m.owner_id,
    m.roles,
    m.membership_revision,
    m.session_version,
    false AS revoked
  FROM memberships m
  WHERE m.principal_id = p_principal_id
    AND m.membership_id = p_membership_id
    AND m.session_version = p_session_version
    AND m.revoked_at IS NULL
$$;

REVOKE ALL ON FUNCTION cs.resolve_membership(text, text, integer) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION cs.resolve_membership(text, text, integer) TO cornerstone_app;
