CREATE TABLE IF NOT EXISTS cs.operator_metrics (
  tenant_id text NOT NULL,
  namespace_id text NOT NULL,
  owner_id text NOT NULL,
  workspace_id text NOT NULL,
  metric_id text NOT NULL,
  metric_name text NOT NULL,
  metric_value numeric NOT NULL,
  labels jsonb NOT NULL DEFAULT '{}'::jsonb,
  trace_id text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, metric_id)
);

CREATE TABLE IF NOT EXISTS cs.status_records (
  tenant_id text NOT NULL,
  namespace_id text NOT NULL,
  owner_id text NOT NULL,
  workspace_id text NOT NULL,
  status_id text NOT NULL,
  component text NOT NULL,
  status text NOT NULL,
  detail jsonb NOT NULL DEFAULT '{}'::jsonb,
  trace_id text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, status_id)
);

CREATE TABLE IF NOT EXISTS cs.tenant_exports (
  tenant_id text NOT NULL,
  namespace_id text NOT NULL,
  owner_id text NOT NULL,
  workspace_id text NOT NULL,
  export_id text NOT NULL,
  export_type text NOT NULL,
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  row_count integer NOT NULL,
  payload_hash text NOT NULL,
  trace_id text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, export_id)
);

DO $$
DECLARE
  table_name text;
BEGIN
  FOREACH table_name IN ARRAY ARRAY[
    'operator_metrics',
    'status_records',
    'tenant_exports'
  ]
  LOOP
    EXECUTE format('ALTER TABLE cs.%I ENABLE ROW LEVEL SECURITY', table_name);
    EXECUTE format('ALTER TABLE cs.%I FORCE ROW LEVEL SECURITY', table_name);
    EXECUTE format('DROP POLICY IF EXISTS %I ON cs.%I', table_name || '_tenant_scope', table_name);
    EXECUTE format(
      'CREATE POLICY %I ON cs.%I FOR ALL TO cornerstone_app USING (
        tenant_id = current_setting(''app.tenant_id'', true)
        AND namespace_id = current_setting(''app.namespace_id'', true)
        AND owner_id = current_setting(''app.owner_id'', true)
        AND workspace_id = current_setting(''app.workspace_id'', true)
      ) WITH CHECK (
        tenant_id = current_setting(''app.tenant_id'', true)
        AND namespace_id = current_setting(''app.namespace_id'', true)
        AND owner_id = current_setting(''app.owner_id'', true)
        AND workspace_id = current_setting(''app.workspace_id'', true)
      )',
      table_name || '_tenant_scope',
      table_name
    );
    EXECUTE format('GRANT SELECT, INSERT ON cs.%I TO cornerstone_app', table_name);
  END LOOP;
END $$;
