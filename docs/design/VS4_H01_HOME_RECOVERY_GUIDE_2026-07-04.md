# VS4-H01 Home Recovery Guide

**Date:** 2026-07-04 KST  
**Owner:** JiYong / Tars  
**Status:** Recovery guide; not implementation evidence  
**Scope:** Product Alpha Home / landing workspace recovery

## Decision

`VS4-H01` is rejected. VS5 external-user sessions must not start from the current Home UI.

The current surface is structurally rich but visually wrong for first value: it exposes review, package, scenario, readiness, and workflow complexity before the user understands what CornerStone does.

Recovery target:

> Drop anything, or ask what we know — then get a brief with receipts.

## Read first

Before implementation, read:

1. `docs/adr/ADR-0007-product-value-first-reset.md`
2. `docs/sot/01_PRODUCT_GOAL_AND_DIRECTION.md` Part 0
3. `docs/sot/05_PRODUCT_VALUE_VERIFICATION_STANDARD.md`
4. `docs/scenario-contracts/VS5_CITATION_GROUNDED_BRIEF_CONTRACT.md`
5. `docs/design/DESIGN_SYSTEM_CONTRACT_V0_3.md`
6. `docs/design/DESIGN_CONCEPT_SYSTEM_V0_3.md`
7. `docs/design/reference-images/README.md`
8. `docs/design/reference-images/cornerstone-reference-07-home-upload-ask.png`
9. `docs/verification-reports/VS4_H01_UI_UX_REJECTION_REPORT_2026-07-04.md`
10. `reports/human-gates/vs4/filled-records/VS4-H01.review-record.json`
11. `packages/cornerstone_cli/product_runtime.py`

Reference images are design direction, not PASS evidence. Human acceptance remains human-owned.

## Product spine

```text
Drop / Ask -> Evidence-backed Brief -> Decision -> Audit
```

Home should translate this into normal-user language:

```text
Drop anything -> Ask what we know -> Get a brief with receipts -> Decide with source links
```

## Home must show first

The first viewport should contain:

- small standard left navigation: Home, Search, Artifacts, Claims, Actions;
- prominent global search;
- hero: `Drop anything, or ask what we know.`;
- large Drop zone;
- prominent Ask box;
- suggested prompts;
- recent items;
- knowledge-state summary;
- suggested next steps;
- recent activity rail.

Suggested copy:

```text
CornerStone preserves your sources, builds a brief you can defend, and keeps every important statement tied to evidence.
```

If VS5 model-backed behavior is not implemented yet, use honest preview copy:

```text
Current local preview preserves sources and prepares reviewable work. Model-backed briefs are the active VS5 build.
```

## Home must not show first

The first viewport must not expose:

- scenario verifier language;
- `VS4-H01` gate language;
- human-gate package language;
- package paths;
- raw command lists;
- dense audit IDs;
- implementation readiness counters;
- admin / connector / ontology / policy-first content;
- production, on-prem, final-security, live-provider, external-user, or human-acceptance claims;
- earned trust labels that have not been earned by output-specific citation checks.

Allowed first-viewport safety copy should stay short:

```text
Local preview
No external send
Sources preserved
Evidence required for decisions
```

## Move displaced content

| Current Home content | New location |
|---|---|
| Product Alpha review handoff | Review / Proof route or below-fold owner-only panel |
| Review packet | Human-gate package files and validation reports |
| Package paths and commands | Collapsed proof drawer |
| Raw audit IDs | Evidence/Audit detail drawer or Audit page |
| Readiness counters | Verification details drawer or scenario report |
| Ops Inbox lanes | Below first-value Home section or separate daily work area |
| Claim, Memory/Wiki, Action details | Secondary surfaces after work item creation |
| Learn review | Review/Learn surface |
| Admin/connector/policy internals | Admin context only |

## Required implementation contract

Feature / Task:

```text
VS4-H01 UI Recovery: Reference-aligned Product Alpha Home
```

Goal:

```text
Rebuild Product Alpha Home so a first-time user sees a calm, reference-aligned Drop / Ask first-value path, while evidence/proof/safety details remain available through progressive disclosure.
```

Out of scope:

- VS5 model-backed generation;
- new ingest connectors;
- live external writeback;
- memory promotion machinery;
- ontology expansion;
- production hardening, tenancy, SSO, or on-prem claims.

## Acceptance scenarios

| ID | Type | Expected result | Verification | Owner |
|---|---|---|---|---|
| HREC-S01 | MUST_PASS | Desktop Home first viewport contains nav, global search, hero, Drop zone, Ask box, prompts, recent/context regions, and activity rail. | Screenshot + DOM review | AI |
| HREC-S02 | MUST_PASS | First viewport omits scenario, package, raw command, dense audit, readiness-counter, and admin-first content. | DOM absence scan + screenshot | AI |
| HREC-S03 | MUST_PASS | Drop and Ask remain keyboard reachable and prepare local reviewable work without live writeback. | Browser interaction proof or VS4 scenario report | AI |
| HREC-S04 | MUST_PASS | Evidence, audit, policy, local-mode, and review proof remain reachable through progressive disclosure. | Browser/DOM review | AI |
| HREC-S05 | MUST_PASS | Mobile preserves Drop/Ask first-value hierarchy without body-level horizontal overflow. | Mobile screenshot + overflow check | AI |
| HREC-R01 | REGRESSION | Existing artifact/evidence/claim/action/audit structural behavior remains intact. | Relevant VS4 scenario checks | AI |
| HREC-R02 | REGRESSION | No production/on-prem/final-security/live-provider/external-user/human-acceptance overclaim is introduced. | Claim-language scan/source review | AI |
| HREC-R03 | REGRESSION | Template/fallback output is not presented as earned `evidence_backed` unless output-specific citation checks exist. | DOM/source review + relevant tests | AI |
| HREC-H01 | HUMAN_REQUIRED | Owner records `APPROVE`, `APPROVE_WITH_EXCEPTIONS`, or `REJECT` for redesigned Home. | Filled owner review record + screenshots | Human |

## Verification guidance

Run the smallest safe relevant set and report exact evidence. Preferred checks:

```sh
git status --porcelain=v1
scripts/verify_design_system_docs.sh
scripts/verify_sot_docs.sh
PATH="$PWD:$PATH" cornerstone scenario verify vs4-product-alpha-ui-daily-loop --json
PATH="$PWD:$PATH" cornerstone human-gate package --scope vs4 --json
```

Before retrying owner review, produce desktop and mobile screenshots, DOM evidence, absence evidence for forbidden first-viewport terms, updated verifier output, and a filled `VS4-H01` review record.

## Retry gate

Do not start VS5 external-user sessions until AI-verifiable Home recovery rows are verified and `HREC-H01` has a dated owner decision.
