# CornerStone Primary Goal Instruction Following

**Status:** Active primary goal instruction-following overlay for long-running scenario-first work.
**Owner:** JiYong / Tars
**Date:** 2026-06-29
**Applies to:** CornerStone implementation, ConnectorHub adoption work, VS milestone continuation, and any long-running `/goal`-driven engineering loop.

---

## Purpose

This document defines how agents should follow broad CornerStone goals without turning every iteration into an unbounded research and implementation effort.

It preserves the scenario-first direction:

- work from frozen scenario contracts;
- solve scenarios as independent delivery slices;
- keep strict proof boundaries;
- document implementation decisions and evidence;
- never promote human-required or external-provider evidence to local `PASS`.

It adds an execution constraint:

> Continue the full direction, but work in small verified slices with explicit checkpoints before widening scope.

---

## Primary Goal Instruction

Use this instruction when continuing CornerStone ConnectorHub, VS milestone, or scenario-first implementation work:

```markdown
Continue CornerStone ConnectorHub scenario-first development, but operate in small verified slices.

For each slice:
1. Freeze a short scenario contract.
2. Inspect the relevant product/docs/code context.
3. Use senior review lenses, but keep research bounded.
4. Implement the smallest complete AI-verifiable solution.
5. Refactor only for the current slice's correctness and maintainability.
6. Run targeted tests/checks yourself.
7. Document evidence, remaining HUMAN_REQUIRED gates, and the decision before moving on.

Do not claim production readiness, external-provider readiness, human acceptance, or security acceptance without the matching proof surface. Keep local proof, integration proof, staging/prod proof, PR state, and human review separate.

Work on one slice at a time and checkpoint before starting the next one.
```

---

## Full Balanced Goal Prompt

Use the longer version when starting a broad goal that may span multiple scenarios:

```markdown
/goal

CornerStone ConnectorHub work must continue scenario-first, but execution must be balanced and incremental.

Maintain the original direction:
- Treat each Must Pass Scenario or tightly related scenario group as an independent delivery slice.
- Preserve product value, domain correctness, architecture, data contracts, reliability, security, observability, performance, testability, maintainability, and migration feasibility as review lenses.
- Keep proof boundaries strict: local tests, browser checks, integration tests, staging/prod evidence, human review, and PR state must not be merged into one claim.
- Never promote HUMAN_REQUIRED or external-provider evidence to PASS based only on local proof.

Execution limits:
- Work on only one delivery slice at a time unless I explicitly approve a wider batch.
- Before coding, write a short slice contract: goal, scope, non-scope, scenarios, proof needed, human-required items, and done criteria.
- Research from multiple senior perspectives, but timebox it to the smallest useful set: product/domain, architecture/data contract, security/reliability, and verification.
- Prefer the smallest complete implementation that can be verified.
- Refactor only where it directly improves the current slice or prevents obvious follow-on risk.
- Run the relevant automated checks yourself.
- Stop at a checkpoint before starting the next slice.

Checkpoint format after each slice:
- What changed
- What passed
- What remains HUMAN_REQUIRED
- What was deliberately not done
- Evidence: commands, files, reports, browser/API observations
- Recommendation: continue / pause / ask human review / open PR

Speed rule:
- If a scenario needs real external systems, production credentials, human trust review, or business acceptance, document it as HUMAN_REQUIRED and replace AI-verifiable parts with local integration tests where possible.
- Do not spend time trying to prove what cannot be proven locally.
- If verification is slow, first add a narrow deterministic verifier instead of running a broad suite repeatedly.

Done means:
- The current slice has a documented decision trail.
- AI-verifiable checks passed with evidence.
- Human-required gates remain explicit.
- The repo is clean or remaining changes are explained.
```

---

## Operating Rules

### 1. Slice first, map second

Start each continuation by identifying the smallest delivery slice that still advances the larger goal. A slice may be:

- one `MUST_PASS` scenario;
- one tightly related scenario group;
- one human-required gate preparation;
- one verifier or evidence-layout improvement;
- one integration rehearsal that replaces an unverifiable human claim with local proof.

Do not start a broad implementation pass until the slice boundary is explicit.

### 2. Keep the review lenses, reduce the depth

Every slice should still be viewed through these lenses:

- product value;
- domain correctness;
- architecture and data contracts;
- reliability and security;
- observability and performance;
- testability and maintainability;
- migration and rollout feasibility.

The agent should not write a full essay for every lens. It should record only decisions, risks, and evidence that materially affect the current slice.

### 3. Bound research before coding

Before implementation, inspect the relevant product authority, scenario contract, code, tests, and reports. Stop research when the slice has enough information to define:

- expected behavior;
- affected files or commands;
- verification method;
- proof boundary;
- risks that could change the implementation.

If more research would not change the current slice, continue to implementation.

### 4. Preserve proof boundaries

The following proof surfaces are separate:

- local deterministic tests;
- local browser or API observations;
- local integration rehearsals;
- staging evidence;
- production evidence;
- external provider evidence;
- human acceptance;
- PR review and merge state.

A pass in one surface cannot be used as a pass in another surface unless the scenario explicitly defines that surface as sufficient.

### 5. Use `HUMAN_REQUIRED` honestly

Use `HUMAN_REQUIRED` only when verification needs a person, unavailable credentials, external provider state, production access, business judgment, or subjective acceptance.

When possible, replace the AI-verifiable portion with a local deterministic or local integration test, while keeping the remaining human decision explicit.

### 6. Checkpoint before widening

After each slice, stop and record:

- what changed;
- what passed;
- what remains human-required;
- what was deliberately not done;
- evidence paths and commands;
- whether the next slice should continue, pause, or wait for human review.

The next slice should not begin until this checkpoint exists.

---

## Relationship to Existing Agent Instructions

This document does not replace:

- `docs/sot/README.md`;
- `docs/sot/01_PRODUCT_GOAL_AND_DIRECTION.md`;
- `docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md`;
- `docs/scenario-contracts/LOCAL_VERIFICATION_PLANE_V0.md`;
- `docs/agent/SCENARIO_FIRST_AGENT_INSTRUCTION.md`;
- `docs/agent/PROJECT_OPERATING_CONSTITUTION.md`.

It is an execution overlay for long-running goals. If another instruction requires scenario-first verification, evidence-backed claims, CLI-native feature proof, or strict human-required separation, that requirement still applies.

---

## Decision Record

This instruction was created after ConnectorHub VS2/VS3 continuation work showed that the original goal direction was correct but too slow when applied as a full-depth process to every iteration.

The adopted decision is:

- keep scenario-first development;
- keep whole-system review lenses;
- keep strict proof boundaries;
- reduce each execution unit to one verified slice;
- checkpoint before continuing.

