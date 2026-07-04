# VS4-H01 Home Recovery — AI Agent Handoff Prompt

Use this prompt for the next coding agent that will implement the rejected landing/Home UI recovery.

---

## Prompt

You are working on `Tocktock/Cornerstone` for CornerStone.

Task:

```text
Implement VS4-H01 UI Recovery: Reference-aligned Product Alpha Home.
```

The owner fully rejected the current VS4 Product Alpha Home UI. The rejection is not only about Drop / Ask discoverability. It is a full visual hierarchy, information architecture, and human-friendly workflow mismatch. The current Home feels like a scenario verifier / review package / workflow debugger instead of a calm product workspace.

Your job is to rebuild the Home / landing workspace so a first-time user immediately understands:

```text
Drop anything, or ask what we know — then get a brief with receipts.
```

Do not begin coding until you freeze the scenario contract below and reverse-engineer current behavior from repo evidence.

---

## Mandatory read list

Read these before implementation:

1. `AGENTS.md`
2. `README.md`
3. `docs/adr/ADR-0007-product-value-first-reset.md`
4. `docs/sot/01_PRODUCT_GOAL_AND_DIRECTION.md` Part 0
5. `docs/sot/05_PRODUCT_VALUE_VERIFICATION_STANDARD.md`
6. `docs/scenario-contracts/VS5_CITATION_GROUNDED_BRIEF_CONTRACT.md`
7. `docs/design/DESIGN_SYSTEM_CONTRACT_V0_3.md`
8. `docs/design/DESIGN_CONCEPT_SYSTEM_V0_3.md`
9. `docs/design/reference-images/README.md`
10. `docs/design/reference-images/cornerstone-reference-07-home-upload-ask.png`
11. `docs/verification-reports/VS4_H01_UI_UX_REJECTION_REPORT_2026-07-04.md`
12. `docs/design/VS4_H01_HOME_RECOVERY_GUIDE_2026-07-04.md`
13. `reports/human-gates/vs4/filled-records/VS4-H01.review-record.json`
14. `packages/cornerstone_cli/product_runtime.py`
15. Existing relevant tests/scripts for VS4 Product Alpha UI and browser proof.

Treat docs and reports as evidence/requirements, not proof of implementation. Treat reference images as design direction, not PASS evidence. Human acceptance remains human-owned.

---

## Frozen goal

Rebuild Product Alpha Home so the first viewport is a calm, reference-aligned product workspace centered on:

```text
Drop / Ask -> Evidence-backed Brief -> Decision -> Audit
```

Normal user wording should be:

```text
Drop anything -> Ask what we know -> Get a brief with receipts -> Decide with source links
```

The first viewport should not force the user to understand scenario gates, package reports, CLI parity, audit internals, policy internals, ontology, connectors, memory machinery, action machinery, or Product Alpha review mechanics.

---

## Success criteria

The implementation is acceptable only if all of these are true:

1. Desktop Home first viewport follows the Home reference direction: left nav, global search, hero, large Drop zone, prominent Ask box, suggested prompts, recent items, knowledge-state summary, suggested next steps, and right-side recent activity.
2. A first-time user can identify Drop or Ask as the first action within five seconds.
3. Normal Home first viewport does not expose scenario verifier language, human-gate package language, package paths, raw command lists, dense audit IDs, implementation readiness counters, or admin/connector/ontology/policy-first content.
4. Evidence, audit, policy, local-mode, and review proof remain reachable through progressive disclosure.
5. Drop and Ask remain keyboard reachable and connected to the existing local reviewable-work behavior.
6. Mobile preserves the same first-value hierarchy without body-level horizontal overflow.
7. No new overclaim is introduced: no production readiness, on-prem readiness, final security acceptance, live-provider readiness, external-user validation, or human UX acceptance claim.
8. Template/fallback output is not presented as earned `evidence_backed` unless output-specific VS5 citation checks exist.

---

## Out of scope

Do not implement these in this task:

- VS5 model-backed generation;
- new ingest connectors;
- live provider calls or external writeback;
- memory promotion machinery;
- ontology expansion;
- new scenario/report families beyond focused recovery evidence;
- production hardening, tenancy, SSO, or on-prem claims;
- broad refactor unrelated to Home recovery.

---

## Scenario contract

| ID | Type | Expected result | Verification method | Evidence required | Owner |
|---|---|---|---|---|---|
| HREC-S01 | MUST_PASS | Desktop Home first viewport contains standard nav, global search, hero, large Drop zone, prominent Ask box, suggested prompts, recent/context regions, and activity rail. | Browser screenshot + DOM review | Screenshot path + DOM markers | AI |
| HREC-S02 | MUST_PASS | First viewport omits scenario, package, raw command, dense audit, readiness-counter, and admin-first content. | DOM absence scan + screenshot review | Absence-check output | AI |
| HREC-S03 | MUST_PASS | Drop and Ask remain keyboard reachable and prepare local reviewable work without live writeback. | Browser interaction proof / existing VS4 verifier | Browser proof or scenario report | AI |
| HREC-S04 | MUST_PASS | Evidence, audit, policy, local-mode, and review proof remain reachable through progressive disclosure. | Browser/DOM review | Proof drawer/detail evidence | AI |
| HREC-S05 | MUST_PASS | Mobile preserves Drop/Ask first-value hierarchy and has no body-level horizontal overflow. | Mobile screenshot + overflow check | Mobile screenshot/proof | AI |
| HREC-R01 | REGRESSION | Existing artifact/evidence/claim/action/audit structural behavior remains intact. | Relevant VS4 structural checks | Scenario report | AI |
| HREC-R02 | REGRESSION | No production/on-prem/final-security/live-provider/external-user/human-acceptance overclaim is introduced. | Claim-language scan/source review | Scan output or source refs | AI |
| HREC-R03 | REGRESSION | Template/fallback output is not presented as earned `evidence_backed` unless output-specific citation checks exist. | DOM/source review + relevant tests | Source refs / report | AI |
| HREC-H01 | HUMAN_REQUIRED | Owner accepts, accepts with exceptions, or rejects redesigned Home. | Human walkthrough | Filled owner review record + screenshots | Human |

Do not mark any AI-verifiable scenario PASS without concrete evidence. Do not mark `HREC-H01` PASS yourself.

---

## Implementation guidance

Start by reverse-engineering the current Home in `packages/cornerstone_cli/product_runtime.py`.

Likely change shape:

1. Keep the existing structural substrate and runtime-backed work behavior.
2. Rebuild the first viewport into a product-first Home shell:
   - small standard nav;
   - prominent global search;
   - hero: `Drop anything, or ask what we know.`;
   - large Drop zone;
   - prominent Ask box;
   - suggested prompts;
   - recent items;
   - knowledge states;
   - suggested next steps;
   - recent activity rail.
3. Move Product Alpha review handoff, review packet, package paths, raw commands, readiness counters, and scenario/human-gate details out of first viewport.
4. Keep proof/evidence/safety reachable through collapsed details or a review/proof section.
5. Preserve existing `data-vs4-drop-zone` and `data-vs4-ask-box` markers unless you update tests/verifiers consistently.
6. Add Home region markers if useful for verification:

```html
data-vs4-home-region="hero"
data-vs4-home-region="drop-zone"
data-vs4-home-region="ask-box"
data-vs4-home-region="suggested-prompts"
data-vs4-home-region="recent-items"
data-vs4-home-region="knowledge-states"
data-vs4-home-region="next-steps"
data-vs4-home-region="recent-activity"
```

7. Do not use `evidence_backed` or `presented_as_fact` for template/fallback outputs unless output-specific citation checks actually earn the label.

---

## Verification commands to prefer

Run the smallest safe relevant set. Suggested commands:

```sh
git status --porcelain=v1
git diff -- packages/cornerstone_cli/product_runtime.py docs/design docs/agent
scripts/verify_design_system_docs.sh
scripts/verify_sot_docs.sh
PATH="$PWD:$PATH" cornerstone runtime serve --port 8787
PATH="$PWD:$PATH" cornerstone scenario verify vs4-product-alpha-ui-daily-loop --json
PATH="$PWD:$PATH" cornerstone human-gate package --scope vs4 --json
```

If you add a focused Home recovery browser-proof script or verifier, run it and include the exact output paths.

Expected recovery evidence before human review:

```text
reports/browser/vs4-h01-home-recovery/home.png
reports/browser/vs4-h01-home-recovery/home.dom.html
reports/browser/vs4-h01-home-recovery-mobile/home.png
reports/browser/vs4-h01-home-recovery-mobile/home.dom.html
```

---

## Required final report format

Use this exact structure:

```markdown
Summary:
Goal:
Scenario Verification:
| ID | Type | Expected Result | Verification Method | Evidence | Status |
|---|---|---|---|---|---|
Human Required:
| ID | Why AI Cannot Verify | Required Human Action | Expected Evidence | Release Impact |
|---|---|---|---|---|
Tool / Process Evidence:
Failure Reverse Engineering:
Verification Gaps:
Risks:
Confidence:
Verdict:
```

Verdict rules:

- If any AI-verifiable MUST_PASS or REGRESSION item is `FAIL`, `NOT_VERIFIED`, or `NOT_RUN`, do not claim AI-verifiable scope done.
- If all AI-verifiable rows pass but `HREC-H01` is still unreviewed, verdict is:

```text
AI-verifiable scope: done
Human/release gate: needs-human-verification
```

- VS5 external-user sessions remain blocked until owner review is recorded.
