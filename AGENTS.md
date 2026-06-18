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
7. `docs/scenario-contracts/LOCAL_VERIFICATION_PLANE_V0.md` before designing local/CI scenario verification, fixture corpora, model harnesses, or release gates
8. `docs/design/DESIGN_SYSTEM_CONTRACT_V0_3.md` before UI, design-system, component, or frontend work
9. `docs/design/DESIGN_CONCEPT_SYSTEM_V0_3.md` and `docs/design/tokens/cornerstone_design_tokens_v0_3.json` for visual concept and tokens
10. `docs/scenario-contracts/VS0_SCAFFOLD_CONTRACT.md` before VS-0 scaffold or feature coding
11. `docs/verification-reports/VS0_SCAFFOLD_READINESS_REPORT_V0.md` before VS-0 scaffold implementation
12. `docs/scenario-contracts/VS0_IMPLEMENTATION_CONTRACT.md` when working on v0.1/VS-0
13. `docs/scenario-contracts/VS0_PRODUCT_RUNTIME_READINESS_CONTRACT.md` when working on local VS0 runtime readiness
14. `docs/scenario-contracts/VS0_RUNTIME_ACCEPTANCE_AND_HARDENING_CONTRACT.md` when working on local VS0 runtime acceptance evidence
15. `docs/scenario-contracts/VS0_EVIDENCE_CLEANUP_AND_INTERACTIVE_UI_LOOP_CONTRACT.md` before implementing the VS0 evidence-cleanup and interactive UI loop
16. `docs/scenario-contracts/VS0_EVUX_CLEAN_SIGNOFF_GOVERNANCE_CONTRACT.md` before implementing VS0 EVUX clean sign-off governance
17. `docs/scenario-contracts/VS0_OPERATOR_ACCEPTANCE_UI_GATE_CONTRACT.md` before implementing VS0 operator UI acceptance work or starting full VS-1 implementation
18. `docs/verification-reports/VS0_OPERATOR_ACCEPTANCE_UI_REVIEW_2026-06-14.md` before claiming VS0 human operator UX acceptance or moving full VS-1 onto the main implementation track
19. `docs/scenario-contracts/VS2_POLICY_TENANCY_EGRESS_CONTRACT.md` before implementing VS2 policy, tenant isolation, default egress deny, OPA/Rego, Postgres RLS, authorization, or runtime egress controls
20. `docs/verification-reports/VS2_POLICY_TENANCY_EGRESS_CURRENT_STATE_2026-06-19.md` before claiming VS2 implementation status or starting sensitive VS2 code changes
21. `docs/agent/SCENARIO_FIRST_AGENT_INSTRUCTION.md`
22. `docs/agent/PROJECT_OPERATING_CONSTITUTION.md`
23. `docs/sot/03_TECHNICAL_ARCHITECTURE_DEFAULTS.md` only as compatible implementation guidance

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

## Local Verification Plane

Before changing local scenario verification, fixture corpora, model harnesses, CLI transcript capture, evidence validators, policy/security gates, or release gates, read:

`docs/scenario-contracts/LOCAL_VERIFICATION_PLANE_V0.md`

Agents must treat local verification as a CornerStone product acceptance surface, not an ad-hoc test folder:

- Prefer `cornerstone scenario verify ... --json` as the product-level verification entry point.
- Use a deterministic `local_test` model provider as the required baseline for local/CI scenario gates.
- Use Ollama or another local LLM only as a model backend for semantic smoke tests; never use the LLM as the judge of `PASS`.
- Validate `PASS` through deterministic checks over artifacts, evidence bundles, policy decisions, workflow/action records, audit events, CLI transcripts, and scenario reports.
- Capture negative evidence for safety scenarios, including zero unauthorized tool calls, zero unauthorized action cards, zero egress, and zero unredacted secret leaks where prohibited.
- Mark unavailable, subjective, external-provider, or production-only checks as `HUMAN_REQUIRED` with required evidence instead of `PASS`.

## Design System

Before UI, frontend, component, or design-system work, read:

`docs/design/DESIGN_SYSTEM_CONTRACT_V0_3.md`

CornerStone's visual doctrine is **Calm Surface. Deep Evidence. Safe Action.**

Agents must preserve:

- light-first calm workspace and admin surfaces;
- small standard navigation with Home, Search, Artifacts, Claims, and Actions as the normal-user default;
- admin controls separated into admin context;
- evidence, policy, approval, and audit as progressively disclosed detail;
- trust/risk labels that use text plus color, never color alone;
- no dark command-center, chatbot-only, connector-admin-first, or ontology-first default experience.

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

Run `scripts/verify_sot_docs.sh` after documentation edits that touch SoT, scenario, handoff, design, or contract files. Run `scripts/verify_cli_native_first_docs.sh` after documentation edits that touch CLI-native-first rules or feature parity matrices. Run `scripts/verify_design_system_docs.sh` after documentation edits that touch design-system rules or tokens. Run `scripts/verify_vs0_scaffold_readiness_docs.sh` after documentation edits that touch VS-0 scaffold readiness or implementation gating.

## First Implementation Target

Start with the VS-0 scaffold contract before feature coding:

`docs/scenario-contracts/VS0_SCAFFOLD_CONTRACT.md`

Before starting scaffold implementation, read:

`docs/verification-reports/VS0_SCAFFOLD_READINESS_REPORT_V0.md`

The current approved next implementation scope is VS-0 scaffold foundation only after preflight and approval. Do not start VS-0 product features from this gate.

Then start VS-0:

`Personal messy input -> immutable artifact -> search -> evidence-backed brief -> draft/evidence-backed claim -> action card dry-run -> approval/execution -> audit`

VS-0 must be operable through the product UI/API and through a native `cornerstone` CLI path. Do not ship a UI/API-only feature and defer CLI to later.

Do not start with a broad repo merge. Build a zero-base CornerStone core, then port working behavior from existing projects through adapters after scenario verification.
