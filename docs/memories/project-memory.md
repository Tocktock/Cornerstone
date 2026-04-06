# Project Memory

## Current posture

Cornerstone is in a documentation-first phase. The current priority is to make the functional contract explicit enough that implementation can follow without inventing product behavior.

## Recent clarifications that changed the docs set

The current docs set explicitly clarified the following areas:

1. **Permission model**
   - Membership uses base roles: `owner`, `admin`, and `member`.
   - Functional work uses scoped capabilities: `manage_connectors`, `operate`, `review`, and `own_domain`.
   - Human-facing labels map onto those capabilities rather than creating a second hidden permission system.

2. **Deterministic review scope**
   - Every reviewable shared object exposes exactly one canonical `review_domain`.
   - Review authorization is evaluated from that `review_domain` and the actor’s explicit review grants.

3. **Source-backed trust contract**
   - Member-facing outputs may be approved using hidden support when policy allows.
   - They may only be labeled `source_backed` when the current consumer can inspect at least one visible support item.
   - Otherwise they must disclose `restricted_support` or remain hidden if policy requires that.

4. **Personal sources**
   - Personal sources remain secondary.
   - They do not directly influence shared official context.
   - They must be explicitly promoted into a workspace as `PromotedSupport` before they can support shared outputs.

5. **Serving contract**
   - Human UI, API, and MCP surfaces share one canonical contract for response kinds and payload shapes.
   - Required fields, enums, and cardinality are now part of the functional contract.

6. **Canonical state vocabulary**
   - Shared surfaces use one stable exposed vocabulary for lifecycle, verification, support visibility, freshness, and source-connection state.

## Deferred questions

The following areas may still evolve, but they no longer block P0 implementation work:
- how strict default workspace policy values should be
- whether a workspace wants to allow member-facing `restricted_support` publication by default
- whether future milestones need richer domain-routing behavior than the current deterministic `review_domain` rules

7. **Ontology layer**
   - The docs now define a canonical ontology layer that owns abstract classes, concrete semantic object kinds, graph predicates, support-item semantics, and projection ontology.
   - Feature specs should now refer upward to the ontology when they need to define what a concept, relation, support item, or decision *is*.

8. **Symptom-first verification**
   - The implementation now treats the test environment itself as a product surface.
   - Fast domain checks, Postgres-backed integration and contract checks, and browser symptom tests are all wired into one local command path.
   - The one-shot operator entrypoint is `./run-all.sh check`, with the full corpus smoke remaining opt-in via `--with-corpus`.

9. **Local dev database isolation**
   - The normal dev Compose stack now uses a P0-specific Postgres volume name.
   - This prevents the rewritten schema from accidentally booting against the legacy pre-P0 local database volume.
