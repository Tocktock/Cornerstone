# ADR-0001 — Reset Product SoT Around New CornerStone Goal

**Date:** 2026-06-08  
**Status:** Accepted as documentation reset draft  
**Owner:** JiYong / Tars

## Context

CornerStone previously had an older `project-sot.md` that claimed to be the only SoT. That document contained useful technical defaults, but its product framing is now out of date.

The newer product goal defines CornerStone as a living, evidence-first, autonomous operational intelligence platform for personal and organizational knowledge. The newer scenario standard defines what must pass before capabilities or releases can claim complete.

## Decision

Create a SoT bundle:

1. `docs/sot/01_PRODUCT_GOAL_AND_DIRECTION.md` — canonical product goal.
2. `docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md` — canonical scenario and release gate.
3. `docs/sot/03_TECHNICAL_ARCHITECTURE_DEFAULTS.md` — compatible technical defaults only.
4. `docs/scenario-contracts/VS0_IMPLEMENTATION_CONTRACT.md` — first implementation freeze.

Archive the old `project-sot.md` as superseded historical evidence. Do not allow it to claim current product authority.

## Consequences

### Positive

- Product direction becomes clear.
- Docs stop fighting each other.
- Implementation can start from scenario-first VS-0.
- Compatible technical defaults are preserved without preserving old product identity.
- Existing repositories can be converged safely through adapters instead of big-bang merge.

### Negative / costs

- Existing README/roadmap/docs must be reviewed and updated.
- Some old terminology must be removed or archived.
- Agents must inspect the new SoT bundle before coding.
- Scenario verification/reporting adds process, but prevents false “done” claims.

## Alternatives considered

### Keep old `project-sot.md` as the only SoT

Rejected. It conflicts with the new product goal and would preserve outdated product identity.

### Delete all old docs immediately

Rejected for now. Old docs contain useful technical evidence and references. Archive first; delete only after links and dependencies are updated.

### Merge everything into one giant SoT file

Rejected. A bundle is easier to maintain: product goal, scenario standard, technical defaults, implementation contracts, and ADRs have different purposes.

## Implementation note

The first engineering target remains VS-0+:

`Personal messy input → immutable artifact → search → evidence-backed brief → claim → action card dry-run/approval/execution → audit`

This is the smallest slice that proves CornerStone is not just chatbot, search, archive, connector, or automation.
