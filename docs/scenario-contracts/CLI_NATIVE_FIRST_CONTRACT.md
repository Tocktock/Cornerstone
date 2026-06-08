# CornerStone CLI Native-First Contract

**Date:** 2026-06-08
**Owner:** JiYong / Tars
**Status:** Mandatory execution overlay for every CornerStone feature.

## Hard rule

**No CLI, no feature PASS.**

Every shipped CornerStone feature must have a native, reproducible `cornerstone ...` CLI path before it can be marked `PASS`, `done`, release-ready, or accepted.

This contract is an execution overlay. It does not replace the canonical product goal or the 206-product-scenario standard. It changes the release gate:

```text
Feature PASS = product behavior verified + policy/evidence/audit verified + native CLI path verified
```

A feature with UI/API behavior but no CLI path is `FAIL` or `NOT_VERIFIED` for release. It cannot be waived as “CLI later.”

## What counts as a feature

A feature is any shipped capability that can do one or more of the following:

1. read, create, update, approve, execute, export, import, forget, restore, or verify product state;
2. create user-visible output, generated knowledge, evidence, memory, claim, mission, action, policy decision, workflow result, or audit event;
3. affect tenant, namespace, owner, workspace, trust state, permissions, connector capabilities, model routing, learning, security, retention, or release status;
4. appear in UI navigation, public API docs, scenario contracts, CLI help, onboarding, admin/operator flows, generated reports, solution packs, extension packs, or automation workflows.

Private helper functions, local refactors, hidden indexes, and internal library code do not need their own CLI command. If an internal change creates or changes product behavior, that behavior must be covered through the owning feature command.

Purely visual UI quality can require human review, but the underlying state/read/action/export/verification path still requires CLI support.

## CLI parity requirements

Every feature must have:

| Requirement | Required evidence |
|---|---|
| Native command | `cornerstone ...` command and help text. |
| Shared backend path | CLI uses the same Product / Archive / Connector / Workflow / Policy / Audit path as UI/API. |
| Machine output | `--json` output with stable schema, `schema_version`, IDs, status, evidence refs, audit refs, and errors where relevant. |
| Human output | Safe, understandable default output for operators. |
| Exit codes | Stable documented exit code contract. |
| Context scope | Tenant, owner, namespace, and workspace are explicit or inspectable. |
| Policy enforcement | Permission, classification, egress, approval, and mission/workspace policy are enforced. |
| Evidence refs | Evidence/search/provenance refs emitted when the feature depends on evidence. |
| Audit refs | Audit refs emitted for state-changing or security-relevant behavior. |
| Dry-run | Mutating, risky, external, sensitive, or destructive commands support dry-run and block execution without required dry-run/approval. |
| Secret safety | Secrets are never printed to stdout, stderr, logs, reports, screenshots, or transcripts. |
| Fixture/transcript | Deterministic CLI fixture or transcript verifies success and failure behavior. |
| Docs sync | CLI help, examples, scenario contract, API/UI docs, and verification report agree. |

Raw SQL, direct Python scripts, `curl` calls, ad-hoc admin scripts, generated connector glue, and manual database edits do not satisfy CLI parity. They may help implementation, but release-ready feature access must be through the native `cornerstone ...` CLI.

## Required scenario-contract fields

Every feature scenario contract must include:

```markdown
CLI Parity:
- Command group:
- Read command:
- Create/update command:
- Dry-run command, if mutating:
- Approval/execution command, if applicable:
- JSON schema path:
- Exit codes covered:
- Fixture/transcript path:
- Evidence refs emitted:
- Audit refs emitted:
- Same backend path evidence:
- CLI status: PASS / FAIL / NOT_VERIFIED / NOT_RUN / HUMAN_REQUIRED / OUT_OF_SCOPE
```

A missing `CLI Parity` section is a release-blocking contract defect.

## Baseline command groups

The native CLI namespace must cover all current and planned CornerStone capability families:

```text
cornerstone health|ready|version
cornerstone login|logout|whoami|config|profile
cornerstone tenant|principal|membership
cornerstone namespace|workspace
cornerstone artifact|derived|redaction|classification
cornerstone search|snapshot
cornerstone evidence|provenance
cornerstone brief|claim|capsule|decision
cornerstone memory|wiki
cornerstone mission|mode|autopilot
cornerstone action|workflow|approval
cornerstone policy|access|egress
cornerstone audit|backup|restore|observe
cornerstone connector|capability|provider
cornerstone agent|role|orchestrator
cornerstone model|brain|judge|router
cornerstone trajectory|experience|lesson|learning
cornerstone pack|extension|registry|tool
cornerstone scenario|release
```

## Exit-code baseline

| Exit code | Meaning |
|---:|---|
| 0 | Success |
| 1 | Invalid input, validation error, missing argument, or incompatible flags |
| 2 | Policy, permission, classification, tenant, namespace, or workspace denial |
| 3 | Not found, conflict, stale state, duplicate, version mismatch, or idempotency conflict |
| 4 | Evidence missing, unsupported assertion, verification gap, or trust-state violation |
| 5 | Runtime failure, dependency failure, storage failure, or internal error |
| 6 | Human approval required before continuation |
| 7 | Connector/provider unavailable, external capability unavailable, or declared out of scope |
| 8 | Unsafe output blocked, secret redaction failure, prompt-injection/tool-hijack attempt, or egress denial |

## Required release report fields

Every release or milestone verification report must include:

```markdown
CLI Parity Summary:
| Feature / Scenario | CLI command(s) | JSON schema | Exit-code tests | Evidence/audit refs | Same backend path | Status |
|---|---|---|---|---|---|---|
```

Any non-`PASS` AI-verifiable CLI parity row blocks release.

## CLI scenarios

## CS-CLI-001 — Root CLI exists

**Type:** MUST_PASS
**Scenario:** A user or operator installs CornerStone locally and runs the native CLI.
**Must-pass outcome:** `cornerstone --help` and `cornerstone version --json` work without external credentials and identify product version, CLI schema version, and configured endpoint/local mode.
**Verification evidence:** CLI transcript with command, JSON output, and exit code 0.

## CS-CLI-002 — Every feature contract declares CLI parity

**Type:** MUST_PASS
**Scenario:** A feature scenario contract is frozen.
**Must-pass outcome:** The contract includes the required CLI Parity section. Missing CLI parity blocks `PASS`.
**Verification evidence:** Scenario contract row with CLI command, schema, exit codes, fixture/transcript, evidence refs, audit refs, and same-backend-path evidence.

## CS-CLI-003 — CLI cannot bypass safe boundaries

**Type:** MUST_PASS
**Scenario:** CLI invokes ingestion, search, claim, action, audit, connector, model, memory, learning, or admin behavior.
**Must-pass outcome:** CLI uses the same domain, policy, workflow, evidence, and audit paths as UI/API; it never directly mutates durable state or source systems.
**Verification evidence:** Source review or integration test proving shared handler/service path.

## CS-CLI-004 — CLI output is scriptable

**Type:** MUST_PASS
**Scenario:** A user runs a CLI command with `--json`.
**Must-pass outcome:** JSON output is stable, schema-versioned, and includes IDs, status, evidence refs, audit refs, and structured errors where relevant.
**Verification evidence:** JSON schema fixture and CLI transcript.

## CS-CLI-005 — CLI enforces workspace and namespace context

**Type:** MUST_PASS
**Scenario:** A CLI command reads or writes context-bearing product state.
**Must-pass outcome:** Tenant, owner, namespace, and workspace are explicit through flags/config/default context and visible in JSON output. Cross-namespace behavior is blocked unless explicit promotion/reference policy allows it.
**Verification evidence:** Cross-namespace CLI fixture with allowed and denied cases.

## CS-CLI-006 — CLI mutations require dry-run and approval when needed

**Type:** MUST_PASS
**Scenario:** CLI command proposes internal mutation, external mutation, destructive action, sensitive access, or high-risk operation.
**Must-pass outcome:** Dry-run is available and required before execution. Approval/policy requirements are enforced with exit code 6 or 2 where appropriate.
**Verification evidence:** Dry-run, policy-denial, approval-required, and execution transcripts.

## CS-CLI-007 — CLI emits evidence and audit refs

**Type:** MUST_PASS
**Scenario:** CLI command creates or changes artifacts, search snapshots, evidence bundles, briefs, claims, actions, workflow runs, policy decisions, memory, or learning records.
**Must-pass outcome:** Output includes evidence refs and/or audit refs as applicable.
**Verification evidence:** CLI transcript linked to stored evidence/audit records.

## CS-CLI-008 — CLI failure behavior is stable and safe

**Type:** MUST_PASS
**Scenario:** CLI command fails due to validation, policy, missing evidence, missing approval, stale state, unavailable connector, egress denial, prompt-injection block, or secret redaction block.
**Must-pass outcome:** Command exits with documented code and prints helpful, redacted, structured error output.
**Verification evidence:** Failure fixture transcripts for each error class.

## CS-CLI-009 — CLI scenarios verify release readiness

**Type:** MUST_PASS
**Scenario:** A milestone or release candidate is checked.
**Must-pass outcome:** `cornerstone scenario status`, `cornerstone scenario verify`, or equivalent local command produces scenario evidence, including per-feature CLI parity status.
**Verification evidence:** Scenario report output with CLI transcript refs and exit codes.

## CS-CLI-010 — CLI docs and help stay synchronized

**Type:** REGRESSION_GUARD
**Scenario:** A feature command changes behavior, flags, output, or errors.
**Must-pass outcome:** CLI help, examples, scenario contract, JSON schema, command matrix, and release report template are updated together.
**Verification evidence:** Docs/check script output.
