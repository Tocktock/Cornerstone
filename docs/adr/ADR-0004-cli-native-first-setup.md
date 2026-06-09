# ADR-0004 - CLI-Native-First Setup

**Date:** 2026-06-09
**Status:** Accepted as setup-planning authority; no runtime implementation yet.
**Owner:** JiYong / Tars

## Context

CornerStone already has a CLI-native-first release gate: no native `cornerstone ...` CLI path means no feature PASS. The VS-0 scaffold must preserve that gate instead of treating CLI as a later wrapper around API behavior.

## Decision

The scaffold must plan `cornerstone-cli` as a first-class package that uses the same domain, policy, workflow, evidence, and audit paths as API/UI.

```text
cornerstone-cli
  -> cornerstone-core use cases
  -> policy / evidence / workflow / audit services
  -> db / object storage / model router / ConnectorHub adapter
```

Remote mode may call the API. Local mode may call domain services directly. Both modes must produce the same policy decisions, evidence refs, workflow records, and audit refs.

## Required First Command Groups

```text
cornerstone version
cornerstone health
cornerstone ready
cornerstone workspace
cornerstone namespace
cornerstone artifact
cornerstone derived
cornerstone search
cornerstone snapshot
cornerstone evidence
cornerstone brief
cornerstone claim
cornerstone action
cornerstone workflow
cornerstone approval
cornerstone policy
cornerstone audit
cornerstone scenario
```

## JSON Output Contract

Every `--json` response should include:

```json
{
  "schema_version": "cs.cli.v0",
  "command": "cornerstone artifact ingest",
  "status": "success | denied | failed | approval_required | evidence_missing",
  "tenant_id": "...",
  "owner_id": "...",
  "namespace_id": "...",
  "workspace_id": "...",
  "ids": {},
  "evidence_refs": [],
  "audit_refs": [],
  "policy_decision_refs": [],
  "errors": []
}
```

## Exit-Code Baseline

| Exit code | Meaning |
|---:|---|
| 0 | Success |
| 1 | Invalid input or validation error |
| 2 | Policy, permission, namespace, or workspace denial |
| 3 | Not found, conflict, duplicate, or stale state |
| 4 | Evidence missing or trust-state violation |
| 5 | Runtime, dependency, or storage failure |
| 6 | Human approval required |
| 7 | Connector/provider unavailable or out of scope |
| 8 | Unsafe output, secret, prompt-injection, or egress denial |

## Consequences

Positive:

- Operators and agents get deterministic, scriptable evidence paths.
- CLI transcripts can become scenario evidence.
- CLI cannot bypass policy/evidence/audit boundaries.

Costs:

- Every product feature needs CLI parity design before implementation.
- Docs, scenario contracts, CLI help, JSON schemas, and reports must stay synchronized.

## Non-Decision

This ADR does not implement the CLI package. It defines the scaffold expectation that future feature work must satisfy.
