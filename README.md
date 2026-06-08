# CornerStone Documentation Reset Pack

**Date:** 2026-06-08  
**Owner:** JiYong / Tars  
**Status:** Documentation reset package for the changed CornerStone product goal  
**Canonical spelling:** Use **CornerStone** for the product/project. Use `Cornerstone` only for existing repository/package names that already use that spelling. Treat `conerstone` as CornerStone unless intentionally referring to another name.

## Purpose

This package rewrites the documentation authority before implementation starts.

The product goal has changed. CornerStone is no longer defined primarily as a Palantir/OpenClaw-style enterprise/logistics product. The current product authority is:

> **CornerStone is a living, evidence-first, autonomous operational intelligence platform for personal and organizational knowledge. It remembers with evidence, understands with context, acts with governed autonomy, and improves through experience.**

The docs here make that goal explicit and give coding agents a clean source of truth.

## Apply order

1. Copy this package into the future `CornerStone/` repository root.
2. Move old conflicting docs into `docs/archive/superseded/` instead of deleting them immediately.
3. Replace root README/product explanations with this package's `README.md` and `docs/sot/README.md`.
4. Use `docs/sot/01_PRODUCT_GOAL_AND_DIRECTION.md` and `docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md` as product authority.
5. Use `docs/sot/03_TECHNICAL_ARCHITECTURE_DEFAULTS.md` only as compatible implementation guidance.
6. Freeze implementation work with `docs/scenario-contracts/VS0_IMPLEMENTATION_CONTRACT.md` before coding.

## Documentation map

| File | Role |
|---|---|
| `docs/sot/README.md` | SoT authority, precedence, and document index |
| `docs/sot/01_PRODUCT_GOAL_AND_DIRECTION.md` | Canonical product goal and direction |
| `docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md` | Canonical scenario/release gate standard |
| `docs/sot/03_TECHNICAL_ARCHITECTURE_DEFAULTS.md` | Technical defaults that remain compatible with the new product goal |
| `docs/sot/04_DOCUMENT_REPLACEMENT_AND_DEPRECATION_PLAN.md` | What to replace, archive, or rewrite |
| `docs/sot/sot_manifest.yaml` | Machine-readable SoT manifest for agents/CI |
| `docs/scenario-contracts/VS0_IMPLEMENTATION_CONTRACT.md` | First zero-base implementation contract |
| `docs/architecture/ONE_PRODUCT_THREE_ENGINES.md` | One-product / three-engine architecture boundary |
| `docs/implementation/ZERO_BASE_IMPLEMENTATION_ROADMAP.md` | Milestone sequence from zero base |
| `docs/adr/ADR-0001-product-sot-reset.md` | Decision record for the documentation reset |
| `AGENTS.md` | Agent operating instructions for the new repo |

## Non-negotiable reset decision

Old docs that claim to be the **only SoT** must be replaced or archived. They can remain as historical evidence, but not as product authority.

The older technical SoT still contributes useful defaults such as Postgres-first storage, RLS, OPA policy, tamper-evident audit, artifact/evidence/action workflow, and one-command local start. Those defaults are preserved in `03_TECHNICAL_ARCHITECTURE_DEFAULTS.md` only where they support the new product goal.
