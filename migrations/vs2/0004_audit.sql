CREATE TABLE IF NOT EXISTS cs.audit_events (
  tenant_id text NOT NULL,
  namespace_id text NOT NULL,
  owner_id text NOT NULL,
  workspace_id text NOT NULL,
  event_id text NOT NULL,
  event_type text NOT NULL,
  actor text NOT NULL,
  action text NOT NULL,
  subject jsonb NOT NULL,
  decision_id text,
  policy_revision text,
  evidence_refs jsonb NOT NULL DEFAULT '[]'::jsonb,
  previous_hash text NOT NULL,
  event_hash text NOT NULL,
  trace_id text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, event_id)
);

ALTER TABLE cs.audit_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE cs.audit_events FORCE ROW LEVEL SECURITY;
CREATE POLICY audit_events_tenant_scope ON cs.audit_events
  FOR ALL TO cornerstone_app
  USING (
    tenant_id = current_setting('app.tenant_id', true)
    AND namespace_id = current_setting('app.namespace_id', true)
    AND owner_id = current_setting('app.owner_id', true)
    AND workspace_id = current_setting('app.workspace_id', true)
  )
  WITH CHECK (
    tenant_id = current_setting('app.tenant_id', true)
    AND namespace_id = current_setting('app.namespace_id', true)
    AND owner_id = current_setting('app.owner_id', true)
    AND workspace_id = current_setting('app.workspace_id', true)
  );

GRANT SELECT, INSERT ON cs.audit_events TO cornerstone_app;

CREATE POLICY audit_events_auditor_read ON cs.audit_events
  FOR SELECT TO cornerstone_auditor
  USING (true);

GRANT USAGE ON SCHEMA cs TO cornerstone_auditor;
GRANT SELECT ON cs.audit_events TO cornerstone_auditor;
