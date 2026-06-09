# CornerStone VS-0 Scaffold Readiness Report v0

Summary:
- Verdict: proceed only to VS-0 scaffold implementation after approval and preflight.
- Scope: documentation, local verification plane, scaffold contract, and current developer environment readiness.
- Date: 2026-06-09
- Owner: JiYong / Tars
- Commit: NOT_COMMITTED

## Goal

Record the current implementation gate before runtime scaffold work begins.

This report applies the scaffold-readiness assessment to the current checkout. It preserves the key decision:

```text
Next implementation target: VS-0 Scaffold Foundation.
Do not proceed directly to VS-0 product features.
Do not claim local verification or scaffold PASS until runtime code, CLI paths, and scenario verification exist and run.
```

## Readiness Decision

| Scope | Proceed? | Reason |
|---|---:|---|
| Re-apply Local Verification Plane docs patch | No | The repo already contains `LOCAL_VERIFICATION_PLANE_V0.md`, read-order wiring, manifest wiring, and verification script wiring. |
| Implement VS-0 product features | No | Feature behavior requires scaffold/runtime files, native CLI command paths, storage, policy, workflow, audit, and local scenario verification first. |
| Implement VS-0 scaffold | Yes, conditionally | Planning docs and contracts are ready; production dependency additions still require approval and preflight evidence. |
| Claim setup works locally | No | Runtime scaffold files and `cornerstone ...` commands do not exist yet. |
| Claim Local Verification Plane is implemented | No | It is a planning and verification contract only; no runtime runner, corpus, validators, providers, or CLI commands exist yet. |

## Current Repository Authority

The repo is documentation-ready for the scaffold step when these files remain wired:

- `AGENTS.md`
- `README.md`
- `docs/sot/README.md`
- `docs/sot/sot_manifest.yaml`
- `docs/scenario-contracts/LOCAL_VERIFICATION_PLANE_V0.md`
- `docs/scenario-contracts/VS0_SCAFFOLD_CONTRACT.md`
- `docs/scenario-contracts/VS0_IMPLEMENTATION_CONTRACT.md`
- `docs/adr/ADR-0002-framework-and-version-policy.md`
- `docs/verification-reports/template.md`
- `scripts/verify_sot_docs.sh`
- `scripts/verify_cli_native_first_docs.sh`
- `scripts/verify_local_verification_plane_docs.sh`

The design-system contract is also active for any UI shell work:

- `docs/design/DESIGN_SYSTEM_CONTRACT_V0_3.md`
- `docs/design/DESIGN_CONCEPT_SYSTEM_V0_3.md`
- `docs/design/tokens/cornerstone_design_tokens_v0_3.json`
- `scripts/verify_design_system_docs.sh`

## Current Environment Evidence

Local tool inspection in this checkout produced:

```text
git      git version 2.50.1 (Apple Git-155)
python   NOT_INSTALLED as `python`
python3  Python 3.14.5
node     v25.9.0
npm      11.12.1
pnpm     NOT_INSTALLED
uv       uv 0.11.7 (Homebrew 2026-04-15 aarch64-apple-darwin)
docker   Docker version 29.4.1, build 055a478ea9
pytest   NOT_INSTALLED
ruff     NOT_INSTALLED
mypy     NOT_INSTALLED
```

Environment gaps against `ADR-0002` and `VS0_SCAFFOLD_CONTRACT.md`:

- `python3` is on target family, but `python` is missing.
- Node.js is newer than the target `24.x LTS`; scaffold work should pin/use Node 24.x for verification.
- `pnpm` is missing.
- `uv` is present but older than the ADR target `0.11.19`.
- Docker is present.
- `pytest`, `ruff`, and `mypy` are missing until the Python scaffold environment exists.
- No runtime scaffold, lockfiles, database services, or `cornerstone` CLI exist yet.

## Required Preflight Before Scaffold Implementation

Run before adding scaffold files:

```bash
git status --porcelain=v1
git diff

scripts/verify_sot_docs.sh
scripts/verify_cli_native_first_docs.sh
scripts/verify_local_verification_plane_docs.sh
scripts/verify_design_system_docs.sh
scripts/verify_vs0_scaffold_readiness_docs.sh

python3 --version      # expect 3.14.x
node --version         # expect 24.x LTS for scaffold verification
pnpm --version         # expect 11.x per ADR
uv --version           # expect approved compatible version
docker --version
```

If these checks fail, fix the environment or report the failure before scaffold implementation.

## Recommended Next Implementation Scope

Feature / Task:

- Implement **VS-0 Scaffold Foundation**, not VS-0 product features.

Goal:

- Add the runtime scaffolding needed for future VS-0 implementation while preserving scenario-first, CLI-native-first, local verification, and design-system rules.

Allowed initial implementation work:

- root setup files;
- Python workspace metadata;
- Node/pnpm workspace metadata;
- Docker Compose skeleton;
- FastAPI health/ready skeleton;
- Typer `cornerstone` CLI skeleton;
- deterministic `cornerstone version --json`;
- placeholder `cornerstone scenario` command group;
- scaffold verification commands and report output;
- no product-feature behavior yet;
- no live connector/provider integration;
- no external writeback;
- no production database schema beyond minimal scaffold bootstrap unless the scaffold scenario contract explicitly includes it.

Do not implement artifact/search/claim/action/audit behavior until scaffold checks pass.

## Scenario Verification

| ID | Type | Expected Result | Verification Method | Evidence | Status |
|---|---|---|---|---|---|
| VS0-RDY-001 | MUST_PASS | Current authority docs identify Local Verification Plane and VS-0 scaffold as the gate before feature work. | Source review plus docs verification scripts. | `README.md`, `AGENTS.md`, `docs/sot/README.md`, `docs/sot/sot_manifest.yaml`. | PASS |
| VS0-RDY-002 | MUST_PASS | Local Verification Plane is classified as documentation-only until implemented. | Source review. | `docs/scenario-contracts/LOCAL_VERIFICATION_PLANE_V0.md` status and non-goal. | PASS |
| VS0-RDY-003 | MUST_PASS | Current environment fully matches scaffold target tools. | Local tool version inspection. | Tool output above; Node is ahead of target and `pnpm`, `pytest`, `ruff`, `mypy` are missing. | FAIL |
| VS0-RDY-004 | MUST_PASS | Next implementation scope is VS-0 scaffold only, not product features. | Report review. | Readiness decision and recommended scope. | PASS |
| VS0-RDY-005 | REGRESSION_GUARD | Do not claim implementation PASS without concrete runtime, CLI, and scenario evidence. | Report review. | This report explicitly refuses local setup, Local Verification Plane implementation, and product-feature PASS claims. | PASS |

## Human Required

| ID | Why AI Cannot Verify | Required Human Action | Expected Evidence | Release Impact |
|---|---|---|---|---|
| H-SCAF-RDY-001 | Adding production dependencies, lockfiles, or scaffold runtime packages is an approval gate. | Approve VS-0 scaffold implementation scope and dependency additions before code starts. | Written approval or explicit "proceed with scaffold implementation." | Blocks scaffold implementation. |
| H-SCAF-RDY-002 | Scaffold setup cannot be marked working until runtime files and commands exist. | Run scaffold verification after implementation. | Logs for local start, CLI, Docker Compose, tests, lint, typecheck, build, and scenario verifier. | Blocks setup PASS claim. |

## Tool / Process Evidence

- Inputs inspected:
  - pasted scaffold-readiness assessment;
  - `AGENTS.md`;
  - `README.md`;
  - `docs/sot/README.md`;
  - `docs/sot/sot_manifest.yaml`;
  - `docs/scenario-contracts/LOCAL_VERIFICATION_PLANE_V0.md`;
  - `docs/scenario-contracts/VS0_SCAFFOLD_CONTRACT.md`;
  - `docs/adr/ADR-0002-framework-and-version-policy.md`;
  - repo verification scripts.
- Current behavior reverse-engineered:
  - repo is documentation-ready for the scaffold gate;
  - runtime scaffold is not implemented;
  - native `cornerstone ...` CLI does not exist yet.
- Files or artifacts changed:
  - this readiness report;
  - SoT/readme/roadmap wiring;
  - readiness verification script.
- Commands/checks run:
  - local tool version inspection;
  - docs verification scripts after edits.
- Checks not run:
  - Docker Compose config for scaffold services;
  - `cornerstone ...` CLI commands;
  - pytest/ruff/mypy/pnpm checks;
  - API/UI runtime checks.

## Verification Gaps

- No scaffold runtime files exist yet.
- No `cornerstone` CLI exists yet.
- No fixture corpus, scenario registry, local model provider, validators, or scenario runner exists yet.
- No local/on-prem start command exists yet.
- Node 24.x LTS and pnpm 11.x are not available in the current shell.

## Risks

- Starting product features before scaffold verification would violate the scaffold-first contract.
- Treating Local Verification Plane docs as implementation would create false PASS claims.
- Adding dependencies without explicit approval would violate the project stop-and-ask gate.
- Verifying scaffold work on Node 25 instead of Node 24 LTS could hide target-runtime issues.

## Verdict

- AI-verifiable scope: needs-follow-up for runtime scaffold implementation.
- Human/release gate: needs-human-verification for dependency/scaffold approval.
- Implementation gate: clear only for VS-0 scaffold implementation after preflight and approval; blocked for product-feature implementation.
