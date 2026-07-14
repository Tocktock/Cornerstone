# ADR-0007: Product-Value-First Reset (Drop → Brief → Decision → Audit)

**Date:** 2026-07-04
**Owner:** JiYong / Tars
**Status:** Accepted
**Supersedes:** Breadth-first milestone sequencing (VS2/VS3/ConnectorHub expansion before product value); "Evidence-first Operational Intelligence Platform" as the primary external framing.

## Context

An executive product review on 2026-07-04 audited the repository by running the product, reading the runtime code, and auditing the verification system. The findings that force this decision:

1. **The governance substrate is real.** Content-addressed immutable artifacts, forced-untrusted user input, hash-chained tamper-evident audit, evidence links, trust states, CLI/API/UI parity, and Chrome-CDP browser proofs all exist and work (Verified).
2. **The intelligence layer does not exist.** There is no model integration anywhere. `brief create` returns the first search snippets (180-char truncations of the user's own input) plus hardcoded uncertainty strings. `conversation answer` returns the canned sentence "The available evidence supports an answer; inspect the attached evidence refs" for every answerable question (Verified by live test, 2026-07-04).
3. **Trust labels are currently unearned.** Templated outputs carry `evidence_backed` and `presented_as_fact: true`. This violates the spirit of "Do not claim more than evidence supports" from inside the product itself (Verified).
4. **Verification optimizes for itself.** 27/28 VS4 rows PASS while the product cannot produce one insight. Structural PASS counts (206-scenario standard, 25 VS4 slices, 243 verification reports) measure contract-shape consistency, not product value (Verified).
5. **No external human has ever used the product.** No user research, market, pricing, or external demo artifact exists anywhere in the documentation (Verified absence).

## Decision

1. **The active product spine is `Drop / Ask → Evidence-backed Brief → Decision → Audit`.** Everything on the spine is active. Everything off the spine is dormant until user evidence pulls it back.
   - "Decision" is the user-facing promotion of a Brief finding (implemented today as the claim record). Memory/Wiki candidates and Action Cards remain visible as review drafts but are not part of the active value claim.
2. **Two verification planes, both required for completion claims:**
   - **Plane 1 — Structural (existing):** deterministic checks over schemas, boundaries, trust states, audit chains, CLI transcripts, and regressions. Plane 1 alone can support only the verdict `STRUCTURAL_READY`.
   - **Plane 2 — Product value (new):** grounding, citation integrity, faithfulness, usefulness, uncertainty honesty, and external-user comprehension/trust, defined in `docs/sot/05_PRODUCT_VALUE_VERIFICATION_STANDARD.md` (CS-VAL family). No milestone, release, or report may claim product value, understanding, or usefulness from Plane 1 evidence alone.
3. **Local model stack is the default assumption** for all test and scenario planning: Ollama `ornith:9b` for generation and `qwen3-embedding:0.6b` for embeddings (`ornith:9b` verified installed locally, 2026-07-12). `ornith:35b` is opt-in when a specific comparison or test requires the larger model. External providers are optional, future-facing, and must be named per-scenario when assumed. The deterministic `local_test` provider remains the Plane 1 CI baseline.
4. **Milestone sequence is VS5 → VS6 → VS7:**
   - **VS5** — Citation-Grounded Brief: model-backed Brief and Ask, earned trust labels, eval corpus, external stranger test (`docs/scenario-contracts/VS5_CITATION_GROUNDED_BRIEF_CONTRACT.md`).
   - **VS6** — Daily Loop: one read-only ingest source, self-filling Ops Inbox, digest, retrieval quality, ~20 external users (`docs/scenario-contracts/VS6_DAILY_LOOP_CONTRACT.md`).
   - **VS7** — Wedge Validation: design partners, willingness-to-pay evidence, keep/kill decisions on off-spine surfaces (`docs/scenario-contracts/VS7_WEDGE_VALIDATION_CONTRACT.md`).
5. **Dormancy register.** ConnectorHub expansion, VS2 production tenancy/egress, VS3 on-prem/security closure, brain routing/ensembles/judge ledger, agent orchestration/packs, ontology promotion machinery, autopilot modes, capsules/decision-cards as separate objects, and memory promotion machinery are **DORMANT**: preserved as strategic direction and frozen contracts, excluded from active roadmaps, reports, and product claims. Reactivation requires a dated user-evidence rationale recorded against VS6/VS7 findings.
6. **Scope freeze.** No new scenario contracts, verification report families, trace counters, or CLI command families may be added outside the spine until VS5 closes. Verification-apparatus work is capped as a supporting activity, not a deliverable.
7. **Trust-label correction is a MUST_PASS obligation** (CS-VAL-006): outputs that are templated, extractive-fallback, or unverified must not carry `evidence_backed` or `presented_as_fact` labels. The current `conversation answer` behavior is an acknowledged open FAIL against this scenario until VS5 fixes it.

## Consequences

- The canonical 206-scenario standard remains authoritative for long-term behavior, but 6 of its 13 families are now explicitly dormant for planning (see `02_MUST_PASS_SCENARIO_STANDARD.md` §2.4). The count-pinned matrix machinery (206 rows) is intentionally left untouched in this reset; CS-VAL rows are canonicalized into the matrix at the next registry regeneration, which requires a code change and is scheduled inside VS5.
- README and SoT bundle are rewritten around the spine, honest current state, and the next external proof point instead of scenario counts and verification apparatus.
- The next meaningful proof point is external: an unfamiliar user, on their own messy input, receives a brief they understand, trust, and could act on, with every load-bearing statement traceable to its source — inside 10 minutes.
- VS4-H01 (owner UX acceptance) remains open and is the entry gate for VS5 external testing.

## Alternatives considered

- **Continue breadth-first (VS2/VS3/ConnectorHub completion):** rejected — grows the exoskeleton around a missing brain; no path to user evidence.
- **Full pivot away from evidence/governance:** rejected — the substrate is the differentiator once intelligence exists; evidence-first principles are preserved unchanged.
- **Add CS-VAL rows directly to the 206 matrix now:** rejected for this reset — requires `scenarios.py` registry changes (out of scope for a documentation reset) and would break count guards; scheduled inside VS5 instead.
