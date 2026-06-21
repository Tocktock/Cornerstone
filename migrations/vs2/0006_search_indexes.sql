CREATE INDEX IF NOT EXISTS search_snapshots_payload_fts_idx
  ON cs.search_snapshots
  USING gin (to_tsvector('simple', payload::text));

CREATE INDEX IF NOT EXISTS ontology_objects_payload_fts_idx
  ON cs.ontology_objects
  USING gin (to_tsvector('simple', payload::text));
