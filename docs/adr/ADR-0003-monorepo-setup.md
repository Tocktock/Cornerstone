# ADR-0003 - VS-0 Monorepo Setup Direction

**Date:** 2026-06-09
**Status:** Accepted as setup-planning authority; no runtime implementation yet.
**Owner:** JiYong / Tars

## Context

CornerStone must be one coherent product, not three visible products. Existing Cornerstone, KnowledgeBase, and Connector-Hub work remains reference/adaptor material until imported through scenario-verified boundaries.

## Decision

Plan the zero-base scaffold as a single product monorepo:

```text
Cornerstone/
  apps/
    web/

  services/
    api/
    worker/

  packages/
    cornerstone-core/
    cornerstone-db/
    cornerstone-cli/
    cornerstone-policy/
    cornerstone-models/
    cornerstone-connectors/
    cornerstone-audit/

  policies/
    opa/

  fixtures/
    vs0/
      personal_messy_input/
      unknown_format/
      secret_redaction/
      prompt_injection/
      cross_namespace/
      action_dry_run/
      audit_tamper/

  tests/
    unit/
    integration/
    scenario/
    cli_transcripts/

  docs/
    adr/
    scenario-contracts/
    verification-reports/
    agent/

  infra/
    postgres/
    docker/
```

## Planned Root Setup Files

These are planned for the scaffold implementation step, not created by this ADR:

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

## Required Setup Properties

- Latest compatible versions from ADR-0002.
- Pinned lockfiles.
- One-command local start.
- Scenario verification from day 0.
- CLI-native-first feature contract.
- No direct mutation bypass.
- Local/on-prem-friendly operation before live connector/provider requirements.

## Consequences

Positive:

- The product shell can stay coherent while internal engines remain separated.
- Tests and fixtures can cover UI/API/CLI parity from the beginning.
- The repo can port existing behavior by capability instead of merging old repos wholesale.

Costs:

- The monorepo setup must enforce boundaries by package ownership and tests, not just by documentation.
- Workspace tooling must stay simple enough for first-use local setup.

## Non-Decision

This ADR does not create scaffold directories or runtime configuration. Implementation starts only after `docs/scenario-contracts/VS0_SCAFFOLD_CONTRACT.md` is accepted for coding.
