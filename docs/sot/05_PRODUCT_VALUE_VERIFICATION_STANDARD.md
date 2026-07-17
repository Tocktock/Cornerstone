# CornerStone Product Value Verification Standard — Plane 2 Acceptance SoT

**Date:** 2026-07-04
**Owner:** JiYong / Tars
**Status:** Active acceptance authority for all product-value claims (binding on VS5 and later; retroactively binding on any new value claim about existing surfaces)
**Decision record:** `docs/adr/ADR-0007-product-value-first-reset.md`
**Relationship to `02_MUST_PASS_SCENARIO_STANDARD.md`:** The 216-scenario matrix remains the long-term behavior index and structural release gate (Plane 1). The ten CS-VAL definitions in this document supply the second, mandatory verification plane (Plane 2) that structural evidence cannot substitute for.

---

## 0. Why This Document Exists

The 2026-07-04 product review proved a failure mode the existing standard could not catch: **every structural gate can pass while the product produces zero user value.** The live evidence: a brief whose "key points" were the user's own input truncated mid-word, and an Ask feature that answered every question with the canned sentence "The available evidence supports an answer; inspect the attached evidence refs" — labeled `evidence_backed` and `presented_as_fact: true`.

Plane 1 verifies that the machine is honest about its records.
Plane 2 verifies that the machine is worth using.

Both are required. Neither substitutes for the other.

## 1. The Two Verification Planes

| Plane | Question it answers | Judge | Evidence types | Can support |
|---|---|---|---|---|
| **Plane 1 — Structural** | Are records, boundaries, labels, and audit trails intact and honest? | Deterministic validators only | CLI transcripts, schema checks, audit-chain verification, negative evidence counters, browser proofs | `STRUCTURAL_READY` |
| **Plane 2 — Product value** | Would a real user trust, understand, and use this output? | Deterministic citation-integrity checks + human rubric scoring + external users. Local LLM judges are **advisory only** and never own PASS | Eval-corpus reports, citation-integrity scans, dated human rubric records, external-user session records | `VALUE_VERIFIED_LOCAL`, `VALUE_VERIFIED_EXTERNAL` |

### 1.1 Verdict ladder

1. `STRUCTURAL_READY` — all applicable Plane 1 rows pass. Says nothing about usefulness. This is the strongest verdict any pre-VS5 milestone (VS0–VS4) has earned or can earn.
2. `VALUE_VERIFIED_LOCAL` — Plane 1 holds and CS-VAL-001..007 pass on the frozen eval corpus with dated human rubric records.
3. `VALUE_VERIFIED_EXTERNAL` — Plane 2 external rows (CS-VAL-008, CS-VAL-009) pass with external participants who are not JiYong/Tars.

No milestone, release note, README, report, or roadmap may use the words "useful," "understands," "product value," "product-ready," or equivalent claims above the verdict actually earned. `STRUCTURAL_READY` milestones must say so.

### 1.2 Model assumptions for Plane 2

- Generation: local Ollama **`ornith:9b`** (verified installed 2026-07-12). `ornith:35b` is opt-in for explicitly named larger-model comparisons.
- Embeddings: local Ollama **`qwen3-embedding:0.6b`** (verified installed 2026-07-04).
- Plane 1 CI baseline remains the deterministic `local_test` provider; Plane 2 quality runs use the local Ollama stack.
- External model providers (Claude, GPT, Gemini) are optional and future-facing; a scenario that assumes one must name it and mark itself `EXTERNAL_ENVIRONMENT`.
- LLM-as-judge (including `ornith:9b` judging its own outputs) may produce advisory scores recorded alongside evidence; it may never flip a row to PASS. Humans own subjective PASS; deterministic checks own mechanical PASS.

## 2. CS-VAL Scenario Family — Product Value MUST-PASS

Scenario dimensions follow the adaptive standard: **Priority** (MUST_PASS / REGRESSION) and **Verification mode** (AUTOMATED / HUMAN_REQUIRED / AUTOMATED+HUMAN). Run-specific status belongs only in the canonical generated VS5 report, whose corpus, pipeline, runtime-state, and verification-contract bindings must all match the current revision. This contract deliberately does not freeze dated PASS counts into its own hash.

These rows are canonical acceptance authority. They were folded into the generated 216-row matrix during VS5 on 2026-07-12 (see ADR-0007); this document remains their authoritative definition.

### Scenario index

| ID | Priority | Scenario | Verification mode | Acceptance state |
|---|---|---|---|---|
| CS-VAL-001 | MUST_PASS | Every load-bearing brief statement carries a resolvable citation | AUTOMATED | REPORT_OWNED — must PASS on the current bound run |
| CS-VAL-002 | MUST_PASS | Zero fabricated citations | AUTOMATED | REPORT_OWNED — must PASS on the current bound run |
| CS-VAL-003 | MUST_PASS | Brief statements are faithful to their cited spans | AUTOMATED (advisory) + HUMAN_REQUIRED | HUMAN_REQUIRED — current bound outputs plus dated human review |
| CS-VAL-004 | MUST_PASS | Brief synthesizes beyond extraction | AUTOMATED (guard) + HUMAN_REQUIRED | HUMAN_REQUIRED — current bound outputs plus dated human review |
| CS-VAL-005 | MUST_PASS | Uncertainty and gaps are input-specific, not boilerplate | AUTOMATED (guard) + HUMAN_REQUIRED | HUMAN_REQUIRED — current bound outputs plus dated human review |
| CS-VAL-006 | MUST_PASS | Trust labels are earned, never decorative | AUTOMATED | REPORT_OWNED — must PASS on the current bound run |
| CS-VAL-007 | MUST_PASS | Ask answers the question or honestly declines | AUTOMATED (guard) + HUMAN_REQUIRED | HUMAN_REQUIRED — current bound outputs plus dated human review |
| CS-VAL-008 | MUST_PASS | An unfamiliar user reaches a traceable brief in 10 minutes | HUMAN_REQUIRED (external) | HUMAN_REQUIRED — five eligible external sessions |
| CS-VAL-009 | MUST_PASS | External users trust the brief and would use it | HUMAN_REQUIRED (external) | HUMAN_REQUIRED — threshold evidence from the same external round |
| CS-VAL-010 | REGRESSION | No claim above earned verdict anywhere in active docs/reports | AUTOMATED | REPORT_OWNED — must PASS on the current bound run |

Automated guards establish only the machine-owned portion of these rows. Their human-owned portions remain `HUMAN_REQUIRED`; automated readiness does not establish usefulness or product value. A prior report becomes stale immediately when any bound corpus, generation pipeline, runtime-state, performance budget, verifier, or acceptance-contract input changes.

### CS-VAL-001 — Every load-bearing brief statement carries a resolvable citation

- **Intent / risk addressed:** A brief that cannot show its sources is indistinguishable from confident fabrication; source traceability is the product's core promise.
- **Given** an artifact ingested from arbitrary user input, **when** a brief is generated, **then** every key point, finding, number, date, and recommendation classified as load-bearing carries at least one citation ref (artifact + chunk/span), and every ref resolves to a stored artifact and derived chunk in the same owner scope.
- **Expected observable result:** Machine-checkable citation graph: brief → evidence link → chunk → artifact → original checksum, with zero dangling refs. Statements without support must be explicitly labeled `inference` or `unsupported`, not silently uncited.
- **Verification method:** Deterministic citation-resolution scan over every brief in the frozen eval corpus.
- **PASS evidence:** Eval report with per-brief citation-resolution results; zero unresolved refs; unsupported-statement labels present where applicable.

### CS-VAL-002 — Zero fabricated citations

- **Intent / risk addressed:** A model-backed brief can cite sources that do not say what the brief claims, or do not exist. One fabricated citation destroys the "receipts" promise.
- **Given** any generated brief with quoted or paraphrase-anchored citations, **when** each citation is checked, **then** the cited span exists in the cited source (exact for quotes; anchor-overlap for paraphrases), for 100% of citations in the eval corpus. Tolerance: zero.
- **Verification method:** Deterministic span-in-source verification (this is mechanically checkable and does not need a judge).
- **PASS evidence:** Citation-integrity scan report: `fabricated_citation_count=0` across the full eval corpus, plus the scan tool's own fixture test showing it detects seeded fabrications (the checker must be proven able to fail).
- **Negative evidence:** A seeded-fabrication fixture run where the checker reports the planted violations.

### CS-VAL-003 — Brief statements are faithful to their cited spans

- **Intent / risk addressed:** Citations can resolve and still be misrepresented (wrong number, inverted meaning, dropped qualifier).
- **Given** the eval corpus briefs, **when** a human reviews a sample of ≥10 briefs statement-by-statement against cited spans, **then** no statement contradicts, inverts, or materially overstates its source. Advisory judge scores (local model) are recorded for the full corpus but decide nothing.
- **Verification mode:** AUTOMATED (advisory judge, full corpus) + HUMAN_REQUIRED (sample audit owns PASS).
- **PASS evidence:** Dated human faithfulness-audit record with per-statement findings; advisory score distribution attached.

### CS-VAL-004 — Brief synthesizes beyond extraction

- **Intent / risk addressed:** The 2026-07-04 baseline brief was the input echoed back. A brief must be worth more than reading the input.
- **Given** a messy multi-fact input (e.g., a vendor-renewal thread containing a deadline, a cost change, a dependency, and a contradiction), **when** the brief is generated, **then** it reorganizes content by decision relevance: extracts deadlines/amounts/owners, surfaces conflicts, and states what the input means for the user's next step — none of which is a contiguous substring of the input.
- **Deterministic guard (AUTOMATED):** brief key points must not be ≥N-character contiguous substrings of the raw input (echo detection); title must not be `Brief for <query>` boilerplate.
- **Human rubric (owns PASS):** "Is this brief more useful than reading the source?" scored per rubric on the eval corpus; median ≥ 4/5 (threshold **Proposed**, freeze at VS5 contract acceptance).
- **PASS evidence:** Echo-guard scan + dated rubric records from ≥2 reviewers, at least one non-owner.

### CS-VAL-005 — Uncertainty and gaps are input-specific, not boilerplate

- **Intent / risk addressed:** Hardcoded uncertainty text ("Add more sources before using it as broad organizational truth") trains users to ignore uncertainty — the opposite of evidence-first.
- **Given** inputs with distinct, known gaps (missing date, conflicting figures, single-source claim), **when** briefs are generated, **then** each brief's uncertainty section names the specific gap in that input, and inputs with sufficient evidence do not carry false-doubt boilerplate.
- **Deterministic guard:** every uncertainty row is nonempty, non-generic, explicitly typed `presented_as_fact: false`, and bound either to a mechanically detected source-declared absence / whole-bundle coverage check or to the question-specific `HUMAN_REQUIRED` evidence-need path; no single normalized string may appear on more than 20% of corpus briefs. Exact lexical matches to planted gap labels are diagnostic only because they cannot prove semantic completeness.
- **Human rubric (owns semantic PASS):** the dated review sample confirms that generated uncertainty addresses the planted decision-evidence gaps and does not falsely claim evidence is absent when the packet contains it. A model-suggested evidence need is never upgraded to a source fact by the automated guard.
- **PASS evidence:** Corpus structure/variation scan plus the bound human gap-and-conflict review record.

### CS-VAL-006 — Trust labels are earned, never decorative

- **Intent / risk addressed:** As of 2026-07-04, templated non-answers carry `evidence_backed` and `presented_as_fact: true`. Unearned labels are worse than no labels: they launder unverified output.
- **Given** any output (brief, answer, claim suggestion), **then**: `evidence_backed` may appear only when CS-VAL-001/002 checks pass for that specific output; templated, extractive-fallback, or model-unavailable outputs must carry an explicit `extractive_fallback` / `template` label and may never carry `evidence_backed` or `presented_as_fact`; label assignment is recorded in the audit trail with the check refs that earned it.
- **Verification method:** Deterministic label-audit over eval corpus + forced-fallback runs (model stopped) verifying honest degraded labeling.
- **PASS evidence:** Label-audit report, including a model-down run showing fallback outputs correctly labeled.
- **Current status authority:** The canonical bound report owns run status; this row acts as a regression gate on every fresh run.

### CS-VAL-007 — Ask answers the question or honestly declines

- **Intent / risk addressed:** The baseline Ask returns "inspect the attached evidence refs" — deferring the user's work back to the user while claiming support.
- **Given** a question whose answer exists in ingested sources, **when** asked, **then** the response states the answer directly (e.g., "The Acme contract auto-renews on Aug 1; cancellation requires notice by Jul 1") with citations per CS-VAL-001/002. **Given** a question whose answer is not in the sources, **then** the response says so plainly ("Your sources don't contain the renewal date") and is labeled accordingly — no canned deferral, no fabricated answer.
- **Deterministic guard:** answers must not match a fixed-sentence set across distinct questions; insufficiency responses must be labeled `insufficient_evidence`, not `evidence_backed`.
- **PASS evidence:** Eval Q&A report (answerable + unanswerable question sets) + human sample audit.

### CS-VAL-008 — An unfamiliar user reaches a traceable brief in 10 minutes

- **Intent / risk addressed:** The product has never been touched by anyone but its owner. First-value speed and comprehension cannot be self-certified.
- **Given** ≥5 external participants (not JiYong/Tars, no prior exposure), each with their own real messy input, **when** they use CornerStone unaided, **then** each reaches a generated brief, opens at least one citation to its source, and can restate in their own words what the brief says and where it came from, within 10 minutes.
- **Verification mode:** HUMAN_REQUIRED (external). Owner walkthroughs, screenshots, or browser proofs cannot satisfy this row.
- **PASS evidence:** Dated session records per participant: recording or observer notes, task timing, comprehension restatement, participant identifier (redaction allowed), and the brief/citation refs exercised.

### CS-VAL-009 — External users trust the brief and would use it

- **Intent / risk addressed:** Comprehension without trust produces no adoption. This is the category proof.
- **Given** the CS-VAL-008 participants, **then**: median trust/usefulness rating ≥ 4/5 (Proposed; freeze at VS5 acceptance), ≥3 of 5 say they would forward the brief or use it for a real decision, and at least one participant identifies a real decision of their own the brief materially helped.
- **PASS evidence:** Dated rating records + quoted participant statements + the identified decision case. REJECT outcomes are recorded as evidence too; a failed round does not lower the bar, it schedules the next round.

### CS-VAL-010 — No claim above earned verdict (overclaim regression)

- **Intent / risk addressed:** The repository's own history shows claim inflation by vocabulary ("Product Alpha ready") on structural evidence.
- **Given** active docs (README, SoT bundle, active contracts, closure/checkpoint reports), **then** no active document states or implies usefulness, understanding, or product value beyond the current verdict ladder position; automated VS5 readiness is labeled separately from every open human row.
- **Verification method:** Deterministic claim-language scan (forbidden-claim phrase list vs. current verdict) + human review at each milestone close.
- **PASS evidence:** Claim-scan report per milestone close.

## 3. Eval Corpus Requirements

- Formal VS5 acceptance location: `fixtures/vs5/edgar-eval/`.
- ≥25 messy real-domain decision cases, each using one to five inspectable sources and a manifest with planted facts, planted gaps, and answerable and unanswerable questions. The corpus as a whole must include at least three provenance-supported contradiction, supersession, or scope-difference cases; contradictions are never invented merely to fill a fixture field.
- Every source must preserve and hash-bind authoritative retrieval metadata, raw bytes, normalized text, and the exact bounded upload span used by CornerStone. A manifest containing only generated inline prose is not sufficient for faithfulness review.
- Frozen by hash before scoring; extended or replaced only by an explicit dated contract amendment. The earlier inline synthetic corpus at `fixtures/vs5/eval/manifest.json` is superseded by the 2026-07-17 SEC EDGAR corpus and cannot support a current verdict or human record.
- Synthetic safety probes, including prompt injection, remain isolated negative controls and are excluded from quality-case counts, latency samples, and human quality samples.
- Grows with consented real external-user inputs from CS-VAL-008 sessions onward; private participant inputs are not redistributed without explicit consent.

## 4. Boundary Rules

1. Plane 2 does not weaken Plane 1. Deterministic structural gates, negative evidence, CLI parity, and audit verification remain mandatory.
2. Plane 2 evidence expires with major generation-path changes: swapping the model, prompt scheme, or retrieval pipeline invalidates prior `VALUE_VERIFIED_*` verdicts until the eval corpus is re-run (structural Plane 1 evidence does not expire this way).
3. Local Plane 2 evidence is not external evidence. `VALUE_VERIFIED_LOCAL` must never be presented as user validation.
4. External-user evidence must come from participants who are not the owner and had no part in building CornerStone.
5. This standard governs the spine (Drop/Ask → Brief → Decision → Audit). Dormant systems (ConnectorHub, brain routing, agent orchestration, ontology, autopilot, capsules, packs — see `02_MUST_PASS_SCENARIO_STANDARD.md` §2.4) acquire Plane 2 obligations only if reactivated.
