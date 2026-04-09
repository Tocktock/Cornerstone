# AI Operator Surfaces

## Summary

This spec defines how Cornerstone is exposed to model operators such as Codex, Claude, Gemini, and other MCP-capable clients.

Cornerstone remains the canonical semantic and governance layer. Model-facing surfaces are replaceable transports and clients over that contract.

## Why this exists

Cornerstone serves both humans and AI. Without a dedicated spec for model-facing behavior:
- source-ingestion connectors and model-facing tools would blur together
- host-specific agent behavior would leak into product semantics
- auth and scope behavior could drift from the canonical serving contract
- MCP transport details could be mistaken for the product contract itself

This spec keeps those boundaries explicit.

## Scope and owned behavior

This spec owns:
- the `ai operator surfaces` workstream name
- the distinction between source-ingestion connectors and model-facing tools
- the first public model-facing transport
- read-only MCP tool exposure in v1
- model-facing auth and consumer-scope behavior
- boundaries between Cornerstone semantics and host-specific agent customization

## Naming rule

- `connectors` means upstream source-ingestion integrations such as document, conversation, engineering, and snapshot providers.
- `ai operator surfaces` means model-facing transports and tool surfaces consumed by external AI hosts.
- Query-time model tools are not connector templates and must not be documented as connectors.

## Canonical ownership rule

- Cornerstone’s serving contract remains the canonical semantic contract for all consumers.
- MCP is the first model-facing transport, not the canonical product contract.
- Model hosts such as Codex, Claude, and Gemini are replaceable clients over Cornerstone.
- Host-specific instructions, skills, and orchestration behavior stay outside Cornerstone’s core semantics unless a separate product spec explicitly adopts them.

## V1 transport

### Public endpoint

- V1 exposes one public MCP HTTP surface at `/mcp`.
- `/api/v1/mcp/read` is not a supported compatibility surface in v1.

### V1 exposure

- V1 exposes MCP tools only.
- V1 does not expose MCP prompts.
- V1 does not expose MCP resources or resource templates.
- Tool results must preserve the same canonical envelope JSON used by other serving surfaces.

## V1 tool set

V1 exposes exactly these read-only tools:
- `search_context`
- `get_concept`
- `get_relation`
- `get_decision`
- `get_answer`
- `get_graph_slice`
- `follow_provenance`

### Tool argument rules

- `search_context(query, consumer_scope?)`
- `get_answer(query, consumer_scope?)`
- `get_concept(resource_id, consumer_scope?)`
- `get_relation(resource_id, consumer_scope?)`
- `get_decision(resource_id, consumer_scope?)`
- `get_graph_slice(root?, consumer_scope?)`
- `follow_provenance(resource_kind, resource_id, consumer_scope?)`

`consumer_scope` may be omitted. When omitted, the actor’s preferred consumer scope must be used.

### Output rule

- MCP tools must return canonical contract envelopes without MCP-specific semantic reshaping.
- Transport packaging may differ only when the mapping to canonical fields is lossless.
- Trust, provenance, support-visibility, and state semantics must remain identical to the serving contract.

## Auth and access behavior

- MCP uses the same bearer-token actor model as the REST API.
- Model-facing requests run inside the same workspace boundary as the authenticating actor.
- Service and AI actors use the same role-and-capability model as other actors.
- A missing or invalid bearer token is an authentication failure.
- A requested `consumer_scope` that exceeds the actor’s allowed scope is an authorization failure.
- Member visibility, review visibility, and admin visibility must match the existing access rules for the same actor and scope.

## Non-goals

- This spec does not define source-ingestion connector setup or provider bindings.
- This spec does not define host-specific Codex, Claude, or Gemini prompt packs.
- This spec does not authorize write, review, or officialization tools in v1.
- This spec does not make MCP canonical over the serving contract.

## Related docs

- [../serving-contract/spec.md](../serving-contract/spec.md)
- [../workspace-and-access/spec.md](../workspace-and-access/spec.md)
- [../connectors/spec.md](../connectors/spec.md)
- [../retrieval-and-answers/spec.md](../retrieval-and-answers/spec.md)
- [../../decisions/0012-serving-contract-is-canonical-across-ui-api-and-mcp-surfaces.md](../../decisions/0012-serving-contract-is-canonical-across-ui-api-and-mcp-surfaces.md)
- [../../decisions/0017-mcp-is-the-first-model-facing-transport-for-ai-operator-surfaces.md](../../decisions/0017-mcp-is-the-first-model-facing-transport-for-ai-operator-surfaces.md)
