CREATE TABLE IF NOT EXISTS cs.artifacts (
  tenant_id text NOT NULL,
  namespace_id text NOT NULL,
  owner_id text NOT NULL,
  workspace_id text NOT NULL,
  artifact_id text NOT NULL,
  classification text NOT NULL DEFAULT 'internal',
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now(),
  audit_ref text,
  PRIMARY KEY (tenant_id, artifact_id)
);

CREATE TABLE IF NOT EXISTS cs.derived_representations (LIKE cs.artifacts INCLUDING ALL);
CREATE TABLE IF NOT EXISTS cs.search_snapshots (LIKE cs.artifacts INCLUDING ALL);
CREATE TABLE IF NOT EXISTS cs.evidence_bundles (LIKE cs.artifacts INCLUDING ALL);
CREATE TABLE IF NOT EXISTS cs.claims (LIKE cs.artifacts INCLUDING ALL);
CREATE TABLE IF NOT EXISTS cs.ontology_objects (LIKE cs.artifacts INCLUDING ALL);
CREATE TABLE IF NOT EXISTS cs.ontology_links (LIKE cs.artifacts INCLUDING ALL);
CREATE TABLE IF NOT EXISTS cs.action_cards (LIKE cs.artifacts INCLUDING ALL);
CREATE TABLE IF NOT EXISTS cs.workflow_runs (LIKE cs.artifacts INCLUDING ALL);
CREATE TABLE IF NOT EXISTS cs.policy_decisions (LIKE cs.artifacts INCLUDING ALL);
CREATE TABLE IF NOT EXISTS cs.jobs (LIKE cs.artifacts INCLUDING ALL);
CREATE TABLE IF NOT EXISTS cs.idempotency_keys (LIKE cs.artifacts INCLUDING ALL);
CREATE TABLE IF NOT EXISTS cs.egress_grants (LIKE cs.artifacts INCLUDING ALL);
CREATE TABLE IF NOT EXISTS cs.migration_quarantine (LIKE cs.artifacts INCLUDING ALL);
