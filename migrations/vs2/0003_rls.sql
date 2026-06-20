DO $$
DECLARE
  table_name text;
BEGIN
  FOREACH table_name IN ARRAY ARRAY[
    'artifacts',
    'derived_representations',
    'search_snapshots',
    'evidence_bundles',
    'claims',
    'ontology_objects',
    'ontology_links',
    'action_cards',
    'workflow_runs',
    'policy_decisions',
    'jobs',
    'idempotency_keys',
    'egress_grants',
    'migration_quarantine'
  ]
  LOOP
    EXECUTE format('ALTER TABLE cs.%I ENABLE ROW LEVEL SECURITY', table_name);
    EXECUTE format('ALTER TABLE cs.%I FORCE ROW LEVEL SECURITY', table_name);
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
    EXECUTE format('GRANT SELECT, INSERT, UPDATE, DELETE ON cs.%I TO cornerstone_app', table_name);
  END LOOP;
END $$;
