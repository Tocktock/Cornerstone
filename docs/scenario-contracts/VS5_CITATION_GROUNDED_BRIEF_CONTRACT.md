# VS5 First Defensible Decision Brief Contract

**Date:** 2026-07-04; product-outcome boundary amended 2026-07-12
**Owner:** JiYong / Tars
**Status:** Frozen milestone contract; status-neutral (PASS/FAIL belongs to reports). This is not implementation evidence.
**Decision record:** `docs/adr/ADR-0007-product-value-first-reset.md`
**Acceptance authority:** `docs/sot/05_PRODUCT_VALUE_VERIFICATION_STANDARD.md` (Plane 2) + `docs/sot/02_MUST_PASS_SCENARIO_STANDARD.md` (Plane 1)
**Matrix:** `docs/scenario-contracts/VS5_CITATION_GROUNDED_BRIEF_MATRIX.csv`

---

## Executive Summary

VS5 exists to close the single most important gap in CornerStone: the product's first-value moment — messy input → evidence-backed brief — is currently a template around a keyword search. VS5 makes the Brief and Ask genuinely model-backed, citation-grounded, honestly labeled, and **externally testable**. Its product outcome is a first defensible decision Brief, not model integration by itself. VS5 is the first CornerStone milestone whose completion is defined by what happens in front of an unfamiliar user, not by internal PASS counts.

The strongest claim VS0–VS4 earned is `STRUCTURAL_READY`. VS5 targets `VALUE_VERIFIED_EXTERNAL` for the spine surface `Drop / Ask → Brief`.

## Frozen Product Charter (2026-07-12)

- **Target cohort:** operational decision owners — founders, operations leads, analysts, procurement owners, and risk owners — who must prepare and defend a consequential decision from fragmented sources.
- **Primary job:** prepare a defensible decision from one to five related sources.
- **Primary decision archetype:** vendor or contract renewal. Adjacent corpus cases may cover policy/compliance change and operational or project-risk review.
- **Supported intake boundary:** pasted UTF-8 text and UTF-8 plain-text or plain-text email exports (`.txt`, `.md`); one to five sources, at most 128 KiB per source and 512 KiB total per Brief. Original bytes remain preserved. Unsupported or unreadable formats are retained and explained, not silently interpreted.
- **Language boundary:** English source sets and questions for the VS5 acceptance corpus. Mixed-language support remains exploratory and cannot support the VS5 verdict.
- **Reference environment:** application and models already installed on an Apple M5 Pro MacBook Pro (18 cores, 48 GB memory), using local Ollama `ornith:9b` and `qwen3-embedding:0.6b`. `ornith:35b` is reserved for explicitly named larger-model comparisons. The ten-minute timer starts when the application is ready; setup time is reported separately.
- **Required Brief reading order:** decision question; bottom line; key facts; conflicts / risks; missing evidence; recommended next step; sources; technical provenance.
- **Minimal durable handoff:** one sourced finding may be saved as a Decision draft (the existing claim record presented in user-facing language); approval, shared truth, and action execution remain out of scope.

The charter is the user-provided 2026-07-12 product definition reconciled into this existing contract. It freezes one coherent job and input boundary without creating a new scenario family.

## Problem and Root Cause (Verified 2026-07-04)

```text
symptom      -> brief echoes user input; Ask returns a canned deferral sentence
root cause   -> no model integration exists; "Understand/Decide" stages were deferred while governance breadth was built
consequence  -> first 10 minutes deliver negative value vs. any chat tool; trust labels are unearned; no user evidence is collectable
response     -> model-backed grounded generation on the existing evidence substrate; earned labels; Plane 2 gate
acceptance   -> CS-VAL-001..009 via the VS5 scenario rows below
```

## Goals

1. Brief generation is produced by the default local model (`ornith:9b` via Ollama) grounded in retrieved chunks of the user's own artifacts, with per-statement citations that resolve through the existing evidence-link substrate.
2. Ask answers the question directly with citations, or honestly declines when sources are insufficient; the saved question and answer remain discoverable and can be reopened with their source and audit context.
3. Retrieval is upgraded from 180-char snippet truncation to real chunking, with hybrid keyword + embedding retrieval (`qwen3-embedding:0.6b`).
4. Trust labels become earned: `evidence_backed` only on outputs passing citation-integrity checks; explicit `extractive_fallback` labeling when the model is unavailable.
5. One to five related sources produce a decision-oriented Brief with a bottom line, key facts, conflicts/risks, missing evidence, and a recommended next step rather than a general summary.
6. One sourced finding can be preserved as a Decision draft without granting approval, shared-truth, or action authority.
7. A frozen eval corpus (≥25 messy inputs) and the Plane 2 harness exist: deterministic citation-integrity scans, echo/boilerplate guards, advisory judge scoring, human rubric records.
8. Five external stranger-test sessions produce dated comprehension and trust evidence (CS-VAL-008/009).
9. `VS4-H01` is completed with a dated owner review before external sessions begin.

## Non-Goals

- No new ingest connectors or sources (VS6).
- No external write actions, no live provider execution (Action Cards remain drafts).
- No memory promotion machinery, capsules, missions, autopilot, ontology, brain routing, agent packs (DORMANT per ADR-0007).
- No multi-user, tenancy, SSO, or on-prem work (VS2/VS3 stay dormant).
- No cloud model providers by default; optional per-scenario only, explicitly named.
- No new CLI command families. New capability lands inside existing `brief`, `conversation`, `search`, `artifact`, and `scenario` families.
- No new verification report families or trace counters.

## Design Constraints (binding on implementation planning)

- **Reuse the substrate.** Generation must flow through the existing artifact → derived → evidence-bundle → brief record path; the model changes what fills the records, not the record model. Audit refs and trust states apply to generated output like any other record.
- **Untrusted stays untrusted.** Retrieved chunks enter the prompt as quoted evidence, never as instructions; the VS4 Slice 010/018 injection boundary must hold with a real model in the loop (this is where injection risk becomes real; VS5-ASK-002 below).
- **Deterministic CI stays deterministic.** `local_test` remains the Plane 1 provider; Ollama-backed runs are Plane 2 and never gate CI.
- **Honest degradation.** If Ollama is down, the product may fall back to today's extractive behavior only with explicit `extractive_fallback` labeling and without `evidence_backed`.
- **CLI parity holds:** generation remains reachable via `cornerstone brief create` / `cornerstone conversation answer` with `--model-provider` selection; saved Ask records are reachable via `cornerstone conversation history` / `cornerstone conversation show`; all paths support `--json`, scoped evidence refs, and audit refs.
- **Reviewed-run preservation holds:** after humans complete records for the current prefilled inputs, `cornerstone scenario verify vs5-citation-grounded-brief --reuse-vs5-current-run --json` validates and promotes those records against the exact canonical 9B run without deleting or regenerating its Brief/Ask IDs. A corpus, pipeline, model-stack, case-set, automated-row, or runtime-record mismatch fails closed and requires a fresh full run and fresh human review inputs.

## Product Outcome Roll-up

The scenario rows remain implementation and proof evidence. They roll up into five user outcomes; their count is not itself a product claim.

| Product outcome | Scenario evidence beneath it |
|---|---|
| Useful decision Brief | `VS5-BRIEF-001`, `VS5-BRIEF-004`, `VS5-BRIEF-005`, `VS5-QUAL-002`, `VS5-QUAL-003`, `VS5-PERF-001` |
| Defensible source support | `VS5-BRIEF-002`, `VS5-BRIEF-003`, `VS5-TRUST-001` |
| Honest answer behavior | `VS5-ASK-001`, `VS5-ASK-002`, `VS5-TRUST-002` |
| Usable first-value journey | `VS5-DECISION-001`, `VS5-H01`, `VS5-REG-001` |
| External product evidence | `VS5-EXT-001`, `VS5-EXT-002`, `VS5-REG-002` |

## Scenario Rows

Dimensions: Priority | Verification mode | Current evidence status (all rows `NOT_RUN` at freeze unless noted).

| ID | Priority | Scenario | Maps to | Verification mode |
|---|---|---|---|---|
| VS5-BRIEF-001 | MUST_PASS | Decision Brief from arbitrary bounded user input: one to five pasted or supported text/email sources produce the frozen decision-oriented structure through UI, API, and CLI; sources remain grouped and individually inspectable | CS-CLAIM-002 | AUTOMATED |
| VS5-BRIEF-002 | MUST_PASS | Every load-bearing brief statement carries a citation that resolves chunk → artifact → checksum in-scope; unsupported statements labeled `inference`/`unsupported` | CS-VAL-001 | AUTOMATED |
| VS5-BRIEF-003 | MUST_PASS | Zero fabricated citations across the frozen eval corpus; checker proven able to detect seeded fabrications | CS-VAL-002 | AUTOMATED |
| VS5-BRIEF-004 | MUST_PASS | Echo guard: brief key points are not contiguous substrings of raw input; no `Brief for <query>` boilerplate titles | CS-VAL-004 (guard) | AUTOMATED |
| VS5-BRIEF-005 | MUST_PASS | Input-specific uncertainty: planted gaps/contradictions in eval fixtures are named in the brief's uncertainty section; no single boilerplate string across the corpus | CS-VAL-005 | AUTOMATED + HUMAN_REQUIRED |
| VS5-ASK-001 | MUST_PASS | Direct and durable answers: answerable eval questions get the stated answer with citations; unanswerable ones get an explicit `insufficient_evidence` decline; no fixed-sentence deferrals; saved question/answer records are discoverable and reopenable with source, trust-state, timestamp, and audit context through UI/API/CLI | CS-VAL-007 | AUTOMATED + HUMAN_REQUIRED |
| VS5-ASK-002 | MUST_PASS | Injection boundary with a real model: adversarial instructions embedded in ingested content or Ask text cannot approve claims, alter labels, trigger actions, exfiltrate other-scope content, or change policy; prompt-embedded instructions are treated as evidence text | CS-ARCH-007, VS4 S010/S018 | AUTOMATED |
| VS5-TRUST-001 | MUST_PASS | Earned labels: `evidence_backed` only on outputs passing VS5-BRIEF-002/003 checks; label grants recorded with check refs in audit | CS-VAL-006 | AUTOMATED |
| VS5-TRUST-002 | MUST_PASS | Honest fallback: with Ollama stopped, outputs carry `extractive_fallback`, never `evidence_backed`/`presented_as_fact`; UI shows degraded state plainly | CS-VAL-006 | AUTOMATED |
| VS5-DECISION-001 | MUST_PASS | One source-linked Brief finding can be saved as a Decision draft through UI/API/CLI while preserving statement-level citations and granting no approval, shared-truth, or action authority | Active spine Decision step | AUTOMATED |
| VS5-QUAL-001 | MUST_PASS | Frozen eval corpus exists: ≥25 messy inputs with planted-fact/gap/question manifests, hash-frozen, in `fixtures/vs5/eval/` | §3 of 05 SoT | AUTOMATED (structure) + HUMAN_REQUIRED (corpus quality) |
| VS5-QUAL-002 | MUST_PASS | Faithfulness audit: human statement-by-statement review of ≥10 corpus briefs finds no contradiction/inversion/material overstatement of cited spans; advisory judge scores recorded corpus-wide | CS-VAL-003 | HUMAN_REQUIRED |
| VS5-QUAL-003 | MUST_PASS | Usefulness rubric: "more useful than reading the source" median ≥ 4/5 across the corpus from ≥2 reviewers, ≥1 non-owner (threshold Proposed; freeze at contract acceptance) | CS-VAL-004 | HUMAN_REQUIRED |
| VS5-PERF-001 | MUST_PASS | Latency measured and frozen: p50/p95 brief and Ask latency on the reference machine recorded with the eval report; budget set from measurement, then enforced as regression (no invented target before measurement) | — | AUTOMATED |
| VS5-EXT-001 | MUST_PASS | Stranger test: ≥5 external participants, own real input, unaided, each reaches a brief, opens a citation to source, and restates the brief accurately within 10 minutes | CS-VAL-008 | HUMAN_REQUIRED (external) |
| VS5-EXT-002 | MUST_PASS | External trust: median rating ≥ 4/5; ≥3 of 5 would forward or use for a real decision; ≥1 real decision case identified | CS-VAL-009 | HUMAN_REQUIRED (external) |
| VS5-H01 | MUST_PASS | Entry gate: `VS4-H01` owner review recorded (dated decision: accept / accept-with-exceptions / reject) before external sessions run | VS4-H01 | HUMAN_REQUIRED |
| VS5-REG-001 | REGRESSION | Plane 1 intact: VS0/VS4 structural verifier sets still pass with the model-backed path active (artifact immutability, dedupe, audit chain, injection guards, CLI parity) | Plane 1 | AUTOMATED |
| VS5-REG-002 | REGRESSION | Claim-language guard: active docs and the VS5 report claim nothing above the earned verdict; open CS-VAL FAILs stay visible until flipped | CS-VAL-010 | AUTOMATED |

## Success Criteria (completion definition)

VS5 is complete only when **all** hold:

1. All AUTOMATED rows pass with scenario-specific evidence (Plane 1 discipline unchanged).
2. All HUMAN_REQUIRED rows have dated records: internal rubric rows by named reviewers, external rows by non-owner participants.
3. The four CS-VAL baseline FAILs (004/005/006/007 guards) are flipped by evidence, not reworded.
4. Verdict claimed is exactly what evidence supports: `VALUE_VERIFIED_LOCAL` after VS5-QUAL rows; `VALUE_VERIFIED_EXTERNAL` only after VS5-EXT rows.
5. A 3-minute unedited recording exists of one stranger-test session (participant-consented, redaction allowed) — the demo artifact is a deliverable, not marketing.
6. The frozen target cohort, input boundary, Brief shape, and Decision-draft handoff remain the acceptance boundary; broad platform capability cannot substitute for them.

A failed external round does not fail the milestone retroactively into rework-hiding: REJECT records are kept, fixes are made, and a new dated round is run. The bar does not move.

## User-Facing Proof Points

1. Paste a real vendor-renewal thread → brief states the deadline, the cost change, the conflict, and the recommended next step — none of it copied text — each statement one click from its source line.
2. Ask "when does the contract auto-renew?" → the date, cited. Ask something the sources don't contain → "your sources don't say," labeled.
3. Stop the model → the product says plainly it is in extractive fallback and drops its trust labels.

## Risks

| Risk | Impact | Mitigation |
|---|---|---|
| `ornith:9b` quality is insufficient for reliable grounded briefs | Eval rubric scores stall below threshold | Prompt/retrieval iteration first; an explicitly named `ornith:35b` comparison or other model swap is allowed — Plane 2 evidence expires and re-runs on swap per 05 SoT §4.2 |
| Local model latency makes 10-minute first value infeasible | VS5-EXT-001 fails on timing | Measure early against VS5-PERF-001; keep visible progress; use `ornith:35b` only when a specific test justifies the added resource cost |
| Real model makes prompt injection exploitable for the first time | Trust-boundary regression | VS5-ASK-002 is MUST_PASS with adversarial fixtures; retrieved content quoted as evidence, never instructions |
| Judge-gaming / metric drift in quality evals | Plane 2 becomes ceremony like Plane 1 did | Humans own subjective PASS; advisory judges never flip rows; corpus hash-frozen; checker must detect seeded violations |
| External recruitment stalls (solo owner) | VS5-EXT rows block indefinitely | 5 participants is the floor and the recruiting target; sessions may be remote and recorded; blocked recruiting is reported as BLOCKED, not substituted with owner walkthroughs |
| Scope regrows toward apparatus | Freeze erosion | ADR-0007 freeze: no new contracts/CLI families/report families inside VS5; harness work only in service of rows above |

## Verification Expectations

- Plane 1: existing deterministic verifier discipline (`cornerstone scenario verify ...`) extended with VS5 checks; `local_test` remains the CI provider; negative evidence for injection and fallback rows.
- Plane 2: eval harness runs the corpus against the Ollama stack; outputs are citation-scanned deterministically; human rubric records are written as dated report files; advisory judge scores attach as metadata.
- Statuses: `PASS` / `FAIL` / `NOT_RUN` / `BLOCKED` / `HUMAN_REQUIRED` per row; the report verdict must equal the weakest applicable required row.
- Evidence lives in `reports/scenario/vs5-*` and `reports/human-gates/vs5/`; external session records under `reports/human-gates/vs5/external-sessions/`.
- After each canonical Ollama run, `python3 scripts/prepare_vs5_human_review_inputs.py` refreshes current-run review inputs and `--check` rejects stale Brief/Ask IDs or excerpts. Human judgments are recorded only by copying the prefilled inputs to `corpus-quality-review.json`, `faithfulness-review.json`, `ask-review.json`, and `usefulness-review.json`; five completed external records live under `reports/human-gates/vs5/external-sessions/`. The verifier validates revision binding and thresholds before promoting any human-owned row.
- Once those records are complete, revalidate with `cornerstone scenario verify vs5-citation-grounded-brief --reuse-vs5-current-run --json --output reports/scenario/vs5-citation-grounded-brief-2026-07-12.json`. Do not run the generating form first: a new model run invalidates the reviewed IDs and requires refreshed inputs and new judgments.

## Out of Scope (explicit)

Connectors and new ingest sources; write actions; memory promotion; team/workspace sharing; tenancy/SSO/on-prem; ontology, brain routing, agent orchestration, packs, autopilot; cloud providers as defaults; marketplace anything; new scenario contracts beyond this one.

## Exit → VS6

VS5 closes with a keep/fix list from external sessions. VS6 (`docs/scenario-contracts/VS6_DAILY_LOOP_CONTRACT.md`) begins only after VS5 reaches `VALUE_VERIFIED_EXTERNAL` or the owner records a dated decision to proceed with exceptions.
