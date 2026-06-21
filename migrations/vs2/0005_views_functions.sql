CREATE OR REPLACE VIEW cs.safe_artifact_counts
WITH (security_invoker = true)
AS
SELECT
  tenant_id,
  namespace_id,
  owner_id,
  workspace_id,
  count(*)::integer AS row_count
FROM cs.artifacts
GROUP BY tenant_id, namespace_id, owner_id, workspace_id;

GRANT SELECT ON cs.safe_artifact_counts TO cornerstone_app;

CREATE OR REPLACE FUNCTION cs.visible_artifact_ids()
RETURNS TABLE (
  artifact_id text,
  tenant_id text,
  namespace_id text,
  owner_id text,
  workspace_id text,
  payload jsonb
)
LANGUAGE sql
SECURITY INVOKER
SET search_path = cs, pg_temp
AS $$
  SELECT artifact_id, tenant_id, namespace_id, owner_id, workspace_id, payload
  FROM artifacts
  ORDER BY artifact_id
$$;

REVOKE ALL ON FUNCTION cs.visible_artifact_ids() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION cs.visible_artifact_ids() TO cornerstone_app;

CREATE OR REPLACE FUNCTION cs.unsafe_all_artifacts()
RETURNS TABLE (
  artifact_id text,
  tenant_id text,
  namespace_id text,
  owner_id text,
  workspace_id text,
  payload jsonb
)
LANGUAGE sql
SECURITY DEFINER
SET search_path = cs, pg_temp
AS $$
  SELECT artifact_id, tenant_id, namespace_id, owner_id, workspace_id, payload
  FROM artifacts
  ORDER BY artifact_id
$$;

REVOKE ALL ON FUNCTION cs.unsafe_all_artifacts() FROM PUBLIC;
