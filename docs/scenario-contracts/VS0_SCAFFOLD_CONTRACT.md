# CornerStone VS-0 Scaffold Contract

**Date:** 2026-06-09
**Owner:** JiYong / Tars
**Status:** Frozen setup-planning contract; no product feature implementation yet.

## Feature / Task

Prepare the project structure and setup policy so future implementation can satisfy the VS-0 product loop without starting feature coding prematurely.

## Goal

Prepare the repo for this future flow:

```text
Personal messy input
-> immutable artifact
-> searchable derived representation
-> evidence-backed brief
-> draft/evidence-backed claim
-> action card dry-run
-> approval/execution
-> audit trail
```

## Success Criteria

- Setup/version policy ADRs exist and name the latest compatible baseline.
- Monorepo setup direction exists without creating unapproved runtime dependencies.
- CLI-native-first is preserved as a first-class setup requirement.
- Product/archive/connector domain boundaries are documented for scaffold work.
- A setup-phase verification report template exists.
- Future scaffold implementation has explicit commands to verify, but those commands are not claimed PASS until run.

## Constraints

### Product / UX

- One visible product: CornerStone.
- Do not expose Cornerstone, KnowledgeBase, or ConnectorHub as three required user mental models.
- Do not start with feature UX before setup contract approval.

### Data / State

- No production database schema or runtime storage is introduced by this planning contract.
- Future durable state must remain PostgreSQL-first.
- Future artifacts, evidence, claims, actions, and audit records must keep owner, namespace, provenance, trust state, evidence refs, and audit refs where relevant.

### Permission / Security

- No production dependency is added without approval.
- No secret, credential, provider access, or external writeback is introduced.
- Default egress deny remains the setup direction.
- CLI must not bypass policy/evidence/workflow/audit boundaries.

### Compatibility / Format

- Python target: 3.14.x.
- Node target: 24.x LTS.
- PostgreSQL target: 18.x.
- Package versions follow ADR-0002 and lockfile verification when scaffold files are introduced.
- TypeScript 6.0.3 is a candidate only if Next build/typecheck pass; TypeScript 5.9.x fallback is allowed for first scaffold.

### Operational / Environment

- The future scaffold must provide one-command local/on-prem start.
- Tests must not require live external credentials.
- CLI transcripts are required for feature PASS once implementation exists.

## Assumptions

- This is setup planning, not product implementation.
- The existing full scenario standard and VS-0 implementation contract remain authoritative.
- Existing repos remain references/adapters until imported through scenario-verified boundaries.
- Current docs verification scripts are the proof surface for this planning change.

## Out of Scope Before Scaffold Implementation Approval

- Runtime code.
- Feature code.
- Dependency installation.
- Lockfile creation.
- Docker service startup.
- Database migrations.
- Live ConnectorHub provider E2E.
- External writeback.
- Production deployment.

## Required Documents Before Feature Coding

```text
docs/adr/ADR-0002-framework-and-version-policy.md
docs/adr/ADR-0003-monorepo-setup.md
docs/adr/ADR-0004-cli-native-first-setup.md
docs/adr/ADR-0005-domain-boundaries.md
docs/adr/ADR-0006-agent-guide.md
docs/scenario-contracts/VS0_SCAFFOLD_CONTRACT.md
docs/verification-reports/template.md
```

## Planned Root Setup Files

These files are planned for a later scaffold implementation step:

```text
.python-version
.node-version
pyproject.toml
uv.lock
package.json
pnpm-workspace.yaml
pnpm-lock.yaml
docker-compose.yml
.env.example
Makefile
```

## Planned Verification Commands for Scaffold Implementation

```bash
python --version
node --version
pnpm --version
uv --version

scripts/verify_sot_docs.sh
scripts/verify_cli_native_first_docs.sh

docker compose config

cornerstone --help
cornerstone version --json
cornerstone health --json
cornerstone ready --json
cornerstone scenario list --json
cornerstone scenario verify vs0-scaffold --json

pytest
ruff check
ruff format --check
mypy
pnpm typecheck
pnpm build
```

## Scenario Contract

| ID | Type | Expected Result | Verification Method | Evidence Required | Status |
|---|---|---|---|---|---|
| VS0-SCAF-001 | MUST_PASS | Setup docs define latest compatible runtime and framework baseline without installing dependencies. | Source review plus docs verification script. | ADR-0002, `scripts/verify_sot_docs.sh` output. | NOT_VERIFIED until checked per change |
| VS0-SCAF-002 | MUST_PASS | Monorepo setup direction preserves one product with internal engine boundaries. | Source review. | ADR-0003 and ADR-0005 line refs. | NOT_VERIFIED until checked per change |
| VS0-SCAF-003 | MUST_PASS | CLI-native-first is part of setup, not deferred. | CLI docs verification script. | `scripts/verify_cli_native_first_docs.sh` output. | NOT_VERIFIED until checked per change |
| VS0-SCAF-004 | MUST_PASS | Future scaffold commands are declared but not falsely marked passing. | Source review. | Planned command list and NOT_RUN/NOT_VERIFIED reporting. | NOT_VERIFIED until checked per change |
| VS0-SCAF-005 | MUST_PASS | Setup work does not add production dependencies or feature code. | Git diff review. | Changed files list. | NOT_VERIFIED until checked per change |
| VS0-SCAF-006 | MUST_PASS | Verification report template can record scenario evidence, CLI parity, human-required items, and gaps. | Source review. | `docs/verification-reports/template.md`. | NOT_VERIFIED until checked per change |
| VS0-SCAF-R01 | REGRESSION_GUARD | Existing 206 full scenarios, 58 VS-0 scenarios, and CLI-native-first gate remain wired. | Local verification scripts. | `scripts/verify_sot_docs.sh` and `scripts/verify_cli_native_first_docs.sh` output. | NOT_VERIFIED until checked per change |

## Human Required

| ID | Why AI Cannot Verify | Required Human Action | Expected Evidence | Release Impact |
|---|---|---|---|---|
| H-SCAF-001 | Adding production dependencies is an approval gate. | Approve stack/version policy before scaffold implementation adds dependencies or lockfiles. | Written approval. | Blocks scaffold implementation. |
| H-SCAF-002 | Setup "works locally" requires future runtime files and command execution after implementation. | Run scaffold commands after implementation exists. | Command logs for local start, CLI, tests, lint, typecheck, build. | Blocks setup PASS claim. |

## Verdict Rule

The VS-0 scaffold cannot be marked complete if any AI-verifiable scaffold scenario is `FAIL`, `NOT_VERIFIED`, or `NOT_RUN`.
