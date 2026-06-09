# ADR-0006 - Agent Guide for VS-0 Setup Phase

**Date:** 2026-06-09
**Status:** Accepted as setup-planning authority.
**Owner:** JiYong / Tars

## Context

Future coding agents need a compact setup-phase instruction that prevents them from turning the setup plan into unapproved product implementation.

## Decision

Use this instruction for VS-0 setup agents:

```markdown
Task: VS-0 project setup planning and scaffold work only. Do not implement product features until approved.

Research:
- Verify latest compatible Python, Node, FastAPI, Typer, Next.js, PostgreSQL, pgvector, OPA, Wasmtime, uv, pnpm versions.
- Prefer latest stable compatible versions.
- Prefer LTS over Current for production runtimes.

Scenario-first:
- Freeze Goal, Constraints, Success Criteria, Out of Scope, MUST_PASS, REGRESSION, and Human Required.
- Do not code before the setup contract is approved.

CLI-native:
- Every feature must declare CLI commands before implementation.
- No CLI, no feature PASS.
- CLI must use the same domain/policy/workflow/evidence/audit paths as API/UI.
- CLI must support `--json`, stable exit codes, evidence refs, audit refs, and dry-run for mutations.

Boundary:
- Product core owns meaning, claims, briefs, actions, approvals, and workspace state.
- Archive/Evidence owns immutable artifacts, hashes, provenance, derived docs/chunks/search.
- ConnectorHub owns provider access, credentials, source policy, and declared external actions.
- Agent memory is never source of truth.

Verification:
- PASS requires command output, test output, source refs, API response, CLI transcript, audit record, or human approval evidence.
- If not run, mark NOT_RUN.
- If evidence is insufficient, mark NOT_VERIFIED.
```

## Consequences

Positive:

- Agents have a reusable setup-phase guardrail.
- Scenario-first and CLI-native-first remain visible before code exists.

Costs:

- Agents must still read the full root instructions and SoT. This guide is not a replacement for authority order.
