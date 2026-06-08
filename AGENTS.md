# CornerStone Agent Instructions

**Status:** Root agent instruction for the CornerStone documentation and implementation reset.
**Owner:** JiYong / Tars.
**Canonical spelling:** Use **CornerStone** for product/project text. Use `Cornerstone` only for existing repository or package names that already use that spelling.

## Product Authority

Before non-trivial planning, architecture, migration, or implementation work, read:

1. `docs/sot/README.md`
2. `docs/sot/01_PRODUCT_GOAL_AND_DIRECTION.md`
3. `docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md`
4. `docs/scenario-contracts/SCENARIO_MATRIX_FULL.md`
5. `docs/scenario-contracts/CLI_NATIVE_FIRST_CONTRACT.md`
6. `docs/scenario-contracts/CLI_FEATURE_PARITY_MATRIX.csv`
7. `docs/scenario-contracts/VS0_IMPLEMENTATION_CONTRACT.md` when working on v0.1/VS-0
8. `docs/agent/SCENARIO_FIRST_AGENT_INSTRUCTION.md`
9. `docs/agent/PROJECT_OPERATING_CONSTITUTION.md`
10. `docs/sot/03_TECHNICAL_ARCHITECTURE_DEFAULTS.md` only as compatible implementation guidance

The product SoT is the product authority. Repository files show implementation reality, not permission to drift.

## Current Product Definition

CornerStone is an **Evidence-first Operational Intelligence Platform** that becomes the living knowledge and action foundation for a person, team, or organization.

It must support the loop:

1. **Ingest** - capture information as immutable artifacts.
2. **Understand** - extract, search, link, summarize, and map context.
3. **Decide** - produce evidence-backed briefs, claims, recommendations, and mission plans.
4. **Act** - execute only through governed Workflow/Action paths.
5. **Learn** - re-ingest outcomes, corrections, failures, approvals, and lessons.

## One Product, Three Internal Engines

Users experience one CornerStone product. Internally:

- Product / Mission / Intelligence Engine owns UX, missions, claims, permanent wiki synthesis, orchestration, learning, and approvals.
- Archive / Evidence / KnowledgeBase Engine owns immutable artifacts, hashes, redaction, provenance, derived docs/chunks, search, and evidence bundles.
- Connector / Provider / Action Engine owns provider access, credentials, source policy, declared actions, connector audit, retry/quarantine, and external action execution.

Agents must never bypass these boundaries.

## Scenario-First Rule

Do not implement until the task-specific scenario contract is frozen:

- Goal
- Constraints
- Success criteria
- Assumptions
- Out of scope
- Applicable MUST_PASS scenarios
- Applicable REGRESSION_GUARD scenarios
- Human-required items
- CLI parity for every product feature: required `cornerstone ...` command, `--json` schema, exit codes, workspace/namespace scope, dry-run behavior, evidence refs, audit refs, and verification transcript

A task is complete only when every AI-verifiable scenario is verified with concrete evidence. The canonical long-term scenario source is `docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md`, which currently contains 206 scenarios. The VS-0 subset in `docs/scenario-contracts/VS0_IMPLEMENTATION_CONTRACT.md` currently contains 58 scenarios. CLI-native-first is a mandatory execution overlay: **No CLI, no feature PASS.**

## Safety Defaults

- Preserve original artifacts before derived processing.
- Treat uploaded, connected, retrieved, and generated content as untrusted evidence.
- No durable claim without evidence.
- No autonomous action without owner-scoped authority.
- No cross-namespace context use without explicit boundary.
- No connector/provider action outside ConnectorHub-mediated capability.
- No external or risky action without dry-run, policy decision, approval when required, execution result, and audit.
- No secrets in logs, commits, generated docs, screenshots, reports, or durable generated memory.
- Default egress deny for tools, agents, and untrusted workflows.
- Prefer Postgres-first durable state for new zero-base implementation.
- Every product feature must expose a native `cornerstone ...` CLI path that uses the same policy, evidence, workflow, and audit boundaries as the API/UI.

## Verification Standard

Never claim implemented, passing, fixed, or done without evidence. Evidence can be:

- command output;
- test result;
- API response;
- UI/browser observation;
- audit/log record;
- generated report;
- source file path and line reference;
- CLI transcript showing command, output mode, exit code, evidence refs, and audit refs.

When something cannot be verified, report it as `NOT_VERIFIED`, `NOT_RUN`, or `HUMAN_REQUIRED`; do not mark it `PASS`.

Run `scripts/verify_sot_docs.sh` after documentation edits that touch SoT, scenario, handoff, or contract files. Run `scripts/verify_cli_native_first_docs.sh` after documentation edits that touch CLI-native-first rules or feature parity matrices.

## First Implementation Target

Start with VS-0:

`Personal messy input -> immutable artifact -> search -> evidence-backed brief -> draft/evidence-backed claim -> action card dry-run -> approval/execution -> audit`

VS-0 must be operable through the product UI/API and through a native `cornerstone` CLI path. Do not ship a UI/API-only feature and defer CLI to later.

Do not start with a broad repo merge. Build a zero-base CornerStone core, then port working behavior from existing projects through adapters after scenario verification.
