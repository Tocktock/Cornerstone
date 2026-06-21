CREATE TABLE IF NOT EXISTS cs.artifact_references (
  tenant_id text NOT NULL,
  namespace_id text NOT NULL,
  owner_id text NOT NULL,
  workspace_id text NOT NULL,
  reference_id text NOT NULL,
  classification text NOT NULL DEFAULT 'internal',
  source_artifact_id text NOT NULL,
  target_artifact_id text NOT NULL,
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  audit_ref text,
  created_at timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, reference_id),
  CONSTRAINT artifact_references_source_artifact_fkey
    FOREIGN KEY (tenant_id, source_artifact_id)
    REFERENCES cs.artifacts(tenant_id, artifact_id),
  CONSTRAINT artifact_references_target_artifact_fkey
    FOREIGN KEY (tenant_id, target_artifact_id)
    REFERENCES cs.artifacts(tenant_id, artifact_id)
);

ALTER TABLE cs.artifact_references ENABLE ROW LEVEL SECURITY;
ALTER TABLE cs.artifact_references FORCE ROW LEVEL SECURITY;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_policies
    WHERE schemaname = 'cs'
      AND tablename = 'artifact_references'
      AND policyname = 'artifact_references_tenant_scope'
  ) THEN
    CREATE POLICY artifact_references_tenant_scope
      ON cs.artifact_references
      FOR ALL
      TO cornerstone_app
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
  END IF;
END $$;

GRANT SELECT, INSERT, UPDATE, DELETE ON cs.artifact_references TO cornerstone_app;
