CREATE SCHEMA IF NOT EXISTS cs;

CREATE ROLE cornerstone_schema_owner NOLOGIN NOSUPERUSER NOBYPASSRLS;
CREATE ROLE cornerstone_app LOGIN NOSUPERUSER NOBYPASSRLS;
CREATE ROLE cornerstone_migrator LOGIN NOSUPERUSER NOBYPASSRLS;
CREATE ROLE cornerstone_maintenance LOGIN NOSUPERUSER NOBYPASSRLS;

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
