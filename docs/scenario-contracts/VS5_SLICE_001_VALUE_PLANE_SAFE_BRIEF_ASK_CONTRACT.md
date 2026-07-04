# VS5 Slice 001 Contract - Ollama-Backed Brief/Ask Path

**Date:** 2026-07-04
**Owner:** JiYong / Tars
**Status:** Frozen slice contract; status-neutral. PASS/FAIL belongs to implementation evidence and verifier output.
**Parent milestone:** `docs/scenario-contracts/VS5_CITATION_GROUNDED_BRIEF_CONTRACT.md`
**Acceptance authority:** `docs/sot/05_PRODUCT_VALUE_VERIFICATION_STANDARD.md` (CS-VAL-001/002/006/007) and ADR-0007.

## Goal

Start the real VS5 intelligence path by making Brief and Ask optionally use the local Ollama stack:

- generation model: `ornith:35b`
- embedding model: `qwen3-embedding:0.6b`
- deterministic baseline: `local_test` remains the Plane 1 CI provider only

This slice creates the smallest model-backed path that can be verified: scoped chunk retrieval, quoted evidence prompts, model-produced Brief/Ask text, citation refs to stored artifact/chunk/span records, deterministic citation checks, earned trust labels, and honest model-down fallback.

This slice does **not** complete VS5, does not claim product value, does not run the full eval corpus, and does not satisfy external-user validation.

## Success Criteria

1. `cornerstone brief create ... --model-provider ollama --generation-model ornith:35b --embedding-model qwen3-embedding:0.6b --json` runs against local Ollama and returns a model-backed brief when the local models are available.
2. `cornerstone conversation answer ... --model-provider ollama --generation-model ornith:35b --embedding-model qwen3-embedding:0.6b --json` uses the same retrieval/generation path and either answers directly with citations or returns `insufficient_evidence`.
3. Retrieved evidence enters prompts only as quoted evidence blocks with stable `evidence_chunk:<id>` refs; artifact text is never treated as executable instructions.
4. Generated load-bearing Brief and Ask statements carry citation refs to stored evidence chunks, with chunk span metadata resolving back to the scoped artifact derived text.
5. `evidence_backed` and `presented_as_fact` are assigned only when deterministic citation-resolution/span checks pass for that exact output. Otherwise outputs use `draft`, `insufficient_evidence`, or `extractive_fallback`.
6. If Ollama generation or embedding is unavailable, Brief/Ask degrade to explicit `extractive_fallback`, never `evidence_backed` or `presented_as_fact`.
7. Prompt-injection content inside artifacts or Ask text cannot approve claims, alter labels, trigger actions, call providers, exfiltrate other-scope content, or change policy.
8. Existing Plane 1 docs/matrix checks and targeted structural flows remain intact.

## Constraints

- Extend only existing `brief`, `conversation`, `search`, `artifact`, and `scenario` surfaces.
- Do not add a new CLI command family.
- Preserve `local_test` as the deterministic CI baseline.
- Use only local Ollama for model-backed work in this slice; no cloud providers.
- Retrieved and ingested content is evidence, never instructions.
- No live providers, external writeback, memory promotion, ontology expansion, agent expansion, brain routing, trace-counter families, or report-family expansion.

## Assumptions

- Local Ollama is available at the configured base URL for model-backed verification, with `ornith:35b` and `qwen3-embedding:0.6b` installed.
- Citation integrity in this slice means deterministic citation resolution and span-in-source checks. Human faithfulness/usefulness review remains later VS5 work.
- `local_test` behavior may continue returning fallback-safe structural outputs for CI and legacy Plane 1 checks.
- Existing claim records may still use `evidence_backed` where their existing claim-specific evidence rules pass; this slice targets Brief and Ask output labels.

## Out of Scope

- Full VS5 completion.
- The >=25 item eval corpus, human rubric records, external stranger tests, advisory judge scoring, and latency budgets.
- UI redesign.
- New ingest connectors, write actions, memory promotion, ontology, agent/brain routing, autopilot, or packs.
- Cloud model providers.

## MUST_PASS Scenarios

| ID | Expected Result | Verification |
|---|---|---|
| S01 | `brief create` accepts `--model-provider ollama --generation-model ornith:35b --embedding-model qwen3-embedding:0.6b` and returns model-backed JSON when Ollama is available. | CLI transcript + JSON output |
| S02 | Brief generation retrieves chunked evidence from the user's artifact and sends only quoted evidence blocks to the model. | source review + adversarial fixture |
| S03 | Generated Brief key points are model-produced statements with citation refs to stored artifact/chunk/span records. | CLI/API JSON inspection |
| S04 | `conversation answer` uses the same Ollama-backed retrieval/generation path and answers directly or returns `insufficient_evidence`. | CLI transcript + JSON output |
| S05 | `evidence_backed` appears only when citation-resolution/span checks pass for the exact output; otherwise labels are fallback/draft/insufficient. | deterministic label audit |
| S06 | Forced model-down runs degrade to `extractive_fallback` with no `evidence_backed` or `presented_as_fact` labels. | forced model-down test |
| S07 | Prompt-injection text inside artifacts or Ask text cannot approve claims, alter labels, trigger actions, call providers, exfiltrate other-scope content, or change policy. | adversarial fixture + negative evidence counters |

## REGRESSION Scenarios

| ID | Expected Result | Verification |
|---|---|---|
| R01 | Existing Plane 1 structural docs and matrix checks still pass. | `scripts/verify_sot_docs.sh`; `python3 scripts/verify_scenario_matrix.py`; `git diff --check` |
| R02 | Artifact, search, evidence bundle, brief, conversation, claim, and audit JSON contracts remain backward-compatible unless explicitly versioned. | targeted CLI/API checks |

## Human-Required Gates

| ID | Expected Result | Verification |
|---|---|---|
| H01 | Human usefulness/faithfulness and external stranger-test rows remain unclaimed. | final report only |

## Verification Commands and Required Evidence

- `python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs5_slice_001_ollama_brief_ask`
- `cornerstone scenario verify vs5-slice-001 --model-provider ollama --json`
- Targeted CLI transcript using a temporary state dir:
  - `cornerstone artifact ingest --text ... --source user_paste --state-dir <tmp> --json`
  - `cornerstone search query ... --state-dir <tmp> --json`
  - `cornerstone evidence bundle create --search-snapshot-id <id> --state-dir <tmp> --json`
  - `cornerstone brief create --evidence-bundle-id <id> --model-provider ollama --generation-model ornith:35b --embedding-model qwen3-embedding:0.6b --state-dir <tmp> --json`
  - `cornerstone conversation start --message ... --state-dir <tmp> --json`
  - `cornerstone conversation answer <id> --question ... --model-provider ollama --generation-model ornith:35b --embedding-model qwen3-embedding:0.6b --state-dir <tmp> --json`
  - forced model-down equivalents using an unavailable Ollama URL
- `scripts/verify_sot_docs.sh`
- `python3 scripts/verify_scenario_matrix.py`
- `git diff --check`

Required evidence surfaces: source file line refs for provider selection, chunk storage, prompt boundary, citation checks, label assignment, fallback behavior, and negative evidence counters; CLI JSON proving model-backed and model-down paths; unit test result; scenario verifier result; explicit note that H01 remains HUMAN_REQUIRED.
