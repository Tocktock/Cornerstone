# CornerStone Scenario Pass Freeze - 2026-06-09

Status: frozen pre-coding contract for the active scenario-pass goal.
Owner: JiYong / Tars.
Scope: full frozen scenario set from `docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md`.

## Goal

Make CornerStone pass the frozen scenario set without diluting the product standard or adding unrelated features.

Done means:

- every AI-verifiable `MUST_PASS` and `REGRESSION_GUARD` scenario in the frozen set is `PASS` with concrete evidence;
- no AI-verifiable scenario remains `FAIL`, `NOT_RUN`, or `NOT_VERIFIED`;
- human-only items are explicitly listed with required action, expected evidence, and release impact;
- Artifact, Evidence, Audit, Action, policy, prompt-injection, tenant/namespace, and ConnectorHub safety boundaries are preserved;
- each scenario batch is committed only after its scenario evidence passes.

## Current Evidence

Repository state before coding:

- `git status --porcelain=v1`: clean.
- `scripts/verify_sot_docs.sh`: PASS for 206 full scenarios, 58 VS-0 scenarios, CLI native-first docs, local verification plane docs, design-system docs, and VS-0 scaffold-readiness docs.
- `command -v cornerstone`: no output; no native CLI is installed or available in this checkout.
- `git ls-files | wc -l`: 44 tracked files.
- No runtime source tree, package manifest, tests, fixture corpus, scenario registry, local model provider, validator implementation, API, UI, or `cornerstone scenario verify` command exists yet.

This evidence proves documentation wiring only. It does not prove product scenario behavior.

## Scenario Scope

The frozen full set is:

| Set | Total | MUST_PASS | REGRESSION_GUARD | Explicit HUMAN_REQUIRED rows |
|---|---:|---:|---:|---:|
| Full scenario standard | 206 | 184 | 22 | 0 |
| VS-0 implementation subset | 58 | 52 | 6 | 0 |

The full set remains the final target. The VS-0 subset is the first product slice, not a replacement for the full standard.

The current repo gate still requires VS-0 scaffold foundation before VS-0 product features. The first coherent implementation batch must therefore create verification/scaffold capability, not claim artifact/search/claim/action behavior.

## Assumptions

- The user has authorized implementation work toward the full scenario goal, but has not approved new production dependencies, broad lockfile churn, auth/authz changes, tenant/security model changes, external calls, secret access, or destructive actions.
- Local deterministic verification must work without live external credentials.
- Ollama may be used only as an optional semantic smoke model. It cannot judge `PASS`.
- Source text, fixtures, model output, connector payloads, and tool output are untrusted evidence.
- Until a scenario registry exists, AI-verifiable versus human-only ownership cannot be mechanically enforced for all 206 scenarios.

## Out Of Scope Before First Code Batch

- Marking the full 206 scenarios `PASS`.
- Implementing VS-0 product features before scaffold/CLI/scenario verification exists.
- Live external connector writeback.
- Production deployment.
- Real provider credential use.
- Secret handling beyond fake-secret fixtures.
- Auth/authz, cryptography, data-retention, or tenant-isolation changes without explicit approval.
- Adding new production dependencies or lockfiles without explicit approval.
- Treating docs verification as product behavior verification.

## Checklist

- [x] Read `README.md`.
- [x] Read root `AGENTS.md`.
- [x] Read required SoT, scenario, CLI-native-first, local-verification, scaffold, VS-0, ADR, and agent-process docs.
- [x] Inspect current repo structure, scripts, git status, branch, remote, and implementation absence.
- [x] Run baseline documentation verification.
- [x] Research current practices for RAG evaluation, durable agent workflows, policy-as-code, provenance, observability, supply-chain evidence, and LLM application security.
- [x] Freeze scenario counts and first implementation boundary.
- [ ] Build scenario registry and evidence report skeleton.
- [ ] Build native `cornerstone` CLI scaffold for version/health/ready/scenario commands.
- [ ] Add deterministic fixtures and validators.
- [ ] Add local `local_test` model provider.
- [ ] Add optional Ollama smoke path after deterministic baseline exists.
- [ ] Add unit/integration/e2e/security/policy/audit checks.
- [ ] Verify each scenario batch and commit only after PASS evidence exists.

## MUST_PASS And REGRESSION Freeze

Applicable `MUST_PASS`: all 184 full-set `MUST_PASS` rows from `SCENARIO_MATRIX_FULL.md`.

Applicable `REGRESSION_GUARD`: all 22 full-set `REGRESSION_GUARD` rows from `SCENARIO_MATRIX_FULL.md`, including `CS-AUTO-020`, `CS-SEC-020`, and `CS-REG-001` through `CS-REG-020`.

First product subset after scaffold capability exists: the 58 VS-0 rows in `VS0_IMPLEMENTATION_CONTRACT.md`.

First implementation batch before product behavior: VS-0 scaffold and verification-plane bootstrap, with no product-feature PASS claims.

## Human Required

| ID | Why AI Cannot Verify | Required Human Action | Expected Evidence | Release Impact |
|---|---|---|---|---|
| H-FREEZE-001 | New production dependencies and lockfiles are approval-gated. | Approve the specific scaffold dependency set before dependency files are added. | Written approval naming dependency scope, or explicit approval to use ADR-0002 targets. | Blocks dependency-based scaffold implementation. |
| H-FREEZE-002 | Some full-set scenarios require subjective product/UX judgment, real external accounts, production data, compliance/security approval, or irreversible operations. | Review the generated scenario registry once implemented and approve `HUMAN_REQUIRED` classifications. | Approved registry/report rows with required evidence per scenario. | Blocks full release PASS until classified and evidenced. |
| H-FREEZE-003 | Local Ollama availability and model choice may depend on the user's machine and preferred pinned model. | Confirm or install the local Ollama model to use for semantic smoke tests if not already available. | `ollama list` / model digest / smoke run log. | Does not block deterministic PASS; blocks Ollama-specific smoke evidence. |

## Research Decision

Best-fit direction for this repo:

- Start with a no-new-dependency scenario registry, CLI/report skeleton, and deterministic validators so the repo can stop making false PASS claims and can produce machine-readable coverage.
- Add ADR-0002 production dependencies only after explicit approval, then move toward the documented FastAPI/Typer/Postgres/OPA/Next stack with lockfiles.
- Use Postgres-first durable state once dependency approval is granted; use PostgreSQL full-text search and pgvector rather than a separate vector service for the first durable scaffold.
- Use OPA/Rego-compatible policy decisions for authorization, egress, approval, and capability gates after dependency/tooling approval; keep initial bootstrap policy logic deterministic and auditable.
- Use LangGraph-style durable execution concepts as a reference for future workflow/mission implementation, but do not add agent orchestration dependencies before the core evidence/policy/audit boundary exists.
- Use deterministic RAG and scenario evaluation: retrieval quality, grounding, robustness, safety, and evidence refs are checked by validators; LLM output is never the PASS judge.
- Use W3C PROV concepts for artifact/evidence provenance shape and OpenTelemetry-compatible trace/log/metric concepts for future observability.
- Use OWASP LLM risk classes as security fixture drivers for prompt injection, sensitive output, excessive agency, vector/embedding weakness, supply-chain, and overreliance checks.
- Use SLSA/in-toto-style provenance ideas for future extension and supply-chain evidence; do not activate external Agent Packs or provider clients without signed/versioned evidence and policy gates.

## First Batch Definition

Batch 1 must make the following true without claiming VS-0 product behavior:

- native `cornerstone` command is locally invocable from the repo;
- `cornerstone --help`, `cornerstone version --json`, `cornerstone health --json`, and `cornerstone ready --json` return stable JSON and exit codes;
- `cornerstone scenario list --json` enumerates the frozen full and VS-0 scenario rows from repository sources;
- `cornerstone scenario coverage --json` fails if required scenarios are missing from the registry;
- `cornerstone scenario verify vs0-scaffold --json` can verify scaffold-only documentation/runtime-readiness checks and honestly report `PASS`, `FAIL`, `NOT_VERIFIED`, `NOT_RUN`, or `HUMAN_REQUIRED`;
- docs verification scripts remain passing;
- the final report distinguishes scaffold PASS from product-feature NOT_VERIFIED.

Batch 1 can be committed only after its own scaffold scenarios pass with evidence. It must not mark any product scenario PASS unless the product behavior is actually verified.
