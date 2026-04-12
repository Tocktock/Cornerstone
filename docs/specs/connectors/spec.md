# Connectors

## Summary

Connectors are how Cornerstone continuously and recoverably stays aligned with external source systems.

They should feel simple to set up, preserve source provenance, and support eventual correctness rather than promising strict real-time sync for every provider.

## Why this exists

Cornerstone only works if it can continuously ingest real organizational context from the places where work already happens.

Those places include:
- document systems
- conversation systems
- engineering systems
- uploaded snapshots or exports

## Scope and owned behavior

This spec owns:
- connector setup expectations
- connector template philosophy
- connector template registry behavior
- runtime-mode policy for connector bootstrap and provider binding
- provider-binding expectations
- source visibility classes
- connector sync modes
- shared-source management permissions
- personal-source connector behavior
- source-native fallback principles

## Primary users

- workspace owners and admins
- members granted workspace-wide `manage_connectors`
- personal-source owners when personal context is enabled

## Shared-source management rules

- Shared workspace connectors are workspace assets.
- Creating, updating, rebinding, pausing, resuming, or removing a shared connector requires workspace-wide `manage_connectors`.
- `operate` alone does not allow shared-source setup or credential changes.
- Owners and admins always satisfy the connector-management requirement.

## Current behavior

- Shared workspace sources are the primary connector path.
- Personal sources are optional and secondary.
- Connectors should prefer template-led setup paths over raw provider configuration fields when possible.
- Connector setup may use provider-native browse, pickers, or source search where that improves correctness without breaking workspace or provenance rules.
- The default design bias is zero-config for end users and low-config for connector managers.
- Connectors do not publish official meaning directly. They create or update source memory.

## Supported provider classes

Cornerstone should support provider classes such as:
- document systems
- conversation systems
- engineering systems
- uploads and snapshots

The exact provider sequence is replaceable. The category model is not.

## Connector template model

Templates are the default setup mechanism.

Template intent matters more than provider internals:
- some templates feed general member-visible context
- some templates primarily feed review and evidence collection
- some templates are snapshots instead of live sync sources
- some templates exist only for personal context preparation and later promotion

The template registry is code-defined in P0 and is not itself a canonical long-lived entity.

Every template must define at least:
- `template_key`
- provider class
- scope kind expected during setup
- default `visibility_class`
- recommended sync cadence
- supported content unit summary
- provenance capability summary
- preview requirements

Connection setup may use ephemeral helper objects such as setup sessions, previews, or sync checkpoints, but those helpers do not replace the canonical `SourceConnection`.

## Provider binding model

- Shared-source provider binding is manager-only.
- Cornerstone may store provider credentials or auth payloads internally, but a `SourceConnection` may expose only a workspace-bound credential reference and safe account summary.
- A persisted `SourceConnection` must preserve its workspace-bound credential reference, selected provider scope payload, and sync checkpoint payload so shared connector recovery survives backend restarts and local database reuse.
- Binding and rebind flows must preserve workspace boundary and must not leak provider secrets to non-manager actors.
- In P0, user-facing binding should prefer OAuth-style flows when the provider supports them.

## Runtime-mode policy

Cornerstone has two backend-owned runtime modes:
- `mock`
- `production`

The runtime mode is deployment-controlled and is not user-switchable in the UI.

Local operator tooling may expose dedicated launchers such as development and production-profile scripts, but those launchers are convenience wrappers only. They must preselect runtime-mode inputs without redefining the canonical meaning of `mock` or `production`.
Local production-like tooling should also isolate persistence from mock/demo tooling so a production-profile workspace does not inherit demo-seeded source state through a shared local database volume.

Connector behavior must follow these rules:
- In `mock`, demo fixture bootstrap and demo provider binding may remain available.
- In `production`, demo fixture bootstrap is disabled.
- In `production`, provider-binding flows must never silently create demo credentials.
- In `production`, missing OAuth configuration must surface as a clear configuration error instead of falling back to demo behavior.
- In `production`, live provider sync paths are the only valid sync paths for shared workspace connectors.

## Visibility model

- `member_visible` sources contribute to normal retrieval, concepts, graph views, and answers.
- `evidence_only` sources contribute to review, validation, concept support, and decision support, but stay hidden from normal member-facing retrieval by default.
- Visibility is resolved inside the current context boundary.
- Snapshot or evidence-heavy templates may default to `evidence_only`.

## Personal connectors

When personal context is enabled:
- a personal connector belongs to a personal context, not directly to a workspace
- only the personal-source owner may inspect its full contents by default
- workspace admins, connector managers, operators, and reviewers do not automatically gain access to personal connector contents or credentials
- personal artifacts and evidence do not directly support shared official outputs
- selected personal material may be promoted into a workspace only by creating `PromotedSupport`

## Sync modes

Connectors may use one or more of the following modes:
- `polling`
- `scheduled_sync`
- `webhook`
- `hybrid`
- `snapshot_upload`

Cornerstone requires eventual correctness and provenance preservation, not strict real-time sync for every provider.

Connector implementations may maintain richer internal run and checkpoint detail, but consumer-facing state must continue to map to canonical `source_connection_state` and `freshness_state`.

Minimal per-run operational history may be persisted for observability and recovery as long as it does not redefine the canonical connector contract.

## Key workflows

- Shared source setup:
  - choose a provider or template
  - authorize or bind the source
  - browse, paste, or otherwise select the target scope
  - confirm sync and visibility defaults
  - preview a safe sample before save
- Source management:
  - inspect sync health
  - view connection details and recent runs
  - change visibility or schedule when permitted
  - re-run sync or rebind a broken source
- Evidence setup:
  - select an evidence-oriented template
  - import the source or upload the snapshot
  - make it available to review and validation without exposing it to normal member retrieval
- Personal-source preparation:
  - connect a private source inside a personal context
  - prepare or inspect candidate support privately
  - explicitly create `PromotedSupport` for a workspace when sharing is intended
- Source-native recovery:
  - when normalized retrieval is not enough, authorized actors may open the original source or use provider-native search while staying within the same context and provenance rules

## Constraints and non-goals

- Connectors are not the place where official meaning is approved.
- Connectors must not require ordinary members to manage shared workspace connector credentials or raw resource identifiers.
- Production runtime must not fall back to demo provider credentials once `runtime_mode=production`.
- The exact provider list is replaceable.
- Query-time chat tools are not the primary connector model.

## Related docs

- [../ai-operator-surfaces/spec.md](../ai-operator-surfaces/spec.md)
- [../workspace-and-access/spec.md](../workspace-and-access/spec.md)
- [../sync-and-provenance/spec.md](../sync-and-provenance/spec.md)
- [./notion/spec.md](./notion/spec.md)
- [../../decisions/0006-template-first-zero-config-connectors.md](../../decisions/0006-template-first-zero-config-connectors.md)
- [../../decisions/0009-access-uses-base-roles-plus-scoped-capabilities.md](../../decisions/0009-access-uses-base-roles-plus-scoped-capabilities.md)
- [../../decisions/0016-workspace-bound-provider-credentials-back-shared-connectors.md](../../decisions/0016-workspace-bound-provider-credentials-back-shared-connectors.md)
- [../../decisions/0011-personal-sources-remain-separate-until-explicitly-promoted-into-shared-context.md](../../decisions/0011-personal-sources-remain-separate-until-explicitly-promoted-into-shared-context.md)
