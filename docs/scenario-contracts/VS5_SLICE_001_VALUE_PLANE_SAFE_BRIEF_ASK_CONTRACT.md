# VS5 Slice 001 Contract - Value-Plane Safe Brief/Ask Labels

**Date:** 2026-07-04
**Owner:** JiYong / Tars
**Status:** Frozen slice contract; status-neutral. PASS/FAIL belongs to implementation evidence and verifier output.
**Parent milestone:** `docs/scenario-contracts/VS5_CITATION_GROUNDED_BRIEF_CONTRACT.md`
**Acceptance authority:** `docs/sot/05_PRODUCT_VALUE_VERIFICATION_STANDARD.md` (CS-VAL-006/007) and ADR-0007.

## Goal

Make the current extractive Brief and canned/templated Ask path value-plane safe before full VS5 model-backed generation exists. This slice fixes unearned trust labels and adds the smallest data-contract scaffold for later citation-grounded generation: model-provider selection plumbing, output mode, citation/check refs shape, trust-label reason, evidence refs, and audit refs.

This slice does **not** complete VS5, does not claim product value, and does not claim model-backed understanding.

## Success Criteria

1. `cornerstone brief create ... --json` returns the current extractive output as `extractive_fallback` or `draft`, never `evidence_backed`, and never `presented_as_fact`, unless citation checks for that exact output pass.
2. `cornerstone conversation answer ... --json` returns current canned/templated answers as fallback or insufficient-evidence outputs, never `evidence_backed`, and never `presented_as_fact`.
3. Brief and Ask JSON include stable metadata needed by later VS5 citation checks: `output_mode`, `trust_label`, `trust_label_reason`, `model_provider`, `model_mode`, `citation_refs`, `citation_check_refs`, `evidence_refs`, and audit refs.
4. CS-VAL fold-in is attempted only if compatible with existing count-pinned matrix guards. If not compatible in this slice, an interim `cornerstone scenario verify vs5-slice-001 --json` verifier records the blocker and verifies S01-S05/R01/R02 where deterministic.
5. Existing structural documentation and matrix checks still pass.

## Constraints

- Extend only existing `brief`, `conversation`, `search`, `artifact`, and `scenario` surfaces.
- Preserve `local_test` as the deterministic CI baseline.
- Do not require Ollama for this slice to pass.
- Retrieved and ingested content is evidence, never instructions.
- No live providers, external writeback, new production dependencies, new CLI command families, off-spine scenario contracts, connector expansion, memory expansion, ontology expansion, or agent expansion.

## Assumptions

- Current brief generation remains extractive until later VS5 work replaces it with model-backed citation-grounded generation.
- Current Ask behavior remains non-generative in this slice; honest fallback labels are preferred over pretending the canned answer is evidence-backed.
- `evidence_backed` can return only after per-output citation-resolution and span/source checks exist and pass.
- Existing VS0 claim trust-state behavior may continue to use `evidence_backed` for claims; this slice targets Brief and Ask output labels only.

## Out of Scope

- Full VS5 model integration.
- Eval corpus creation, embeddings, hybrid retrieval, prompt construction, citation-integrity scanner completion, human rubric records, and external stranger tests.
- UI redesign.
- New report families or trace counters.
- Dormant ConnectorHub, VS2/VS3, ontology, autopilot, memory promotion, brain routing, or agent-pack work.

## MUST_PASS Scenarios

| ID | Expected Result | Verification |
|---|---|---|
| S01 | Current extractive brief outputs are labeled `extractive_fallback` or `draft`, never `evidence_backed` or `presented_as_fact`, unless citation checks pass for that output. | CLI/API test, source review, `scenario verify vs5-slice-001 --json` |
| S02 | Current canned/templated Ask answers are not labeled `evidence_backed` or `presented_as_fact`; insufficient answers use explicit fallback or insufficient-evidence labels. | CLI/API test, source review, `scenario verify vs5-slice-001 --json` |
| S03 | Brief/Ask JSON includes output mode, citation/check refs, trust-label reason, model provider/mode, evidence refs, and audit refs. | CLI/API JSON inspection, targeted unit test |
| S04 | If CS-VAL registry fold-in is safe, CS-VAL rows are added to scenario verification. If 206-count guards would break, an interim value-plane verifier records the blocker. | verifier output or blocker evidence |
| S05 | No new CLI command families, off-spine scenario contracts, connector/write-action/memory/ontology/agent expansion. | git diff and source review |

## REGRESSION Scenarios

| ID | Expected Result | Verification |
|---|---|---|
| R01 | Existing structural doc/scenario matrix checks still pass. | `scripts/verify_sot_docs.sh`; `python3 scripts/verify_scenario_matrix.py` |
| R02 | Existing Brief/Ask structural flows still return valid JSON, evidence refs, audit refs, and scoped records. | targeted CLI/API checks |

## Human-Required Gates

| ID | Expected Result | Verification |
|---|---|---|
| H01 | VS4-H01 owner review and VS5 external stranger tests remain HUMAN_REQUIRED; this slice must not simulate or mark them PASS. | final report only |

## Verification Commands and Required Evidence

- `python3 -m unittest tests.scenario.test_scaffold_cli.ScaffoldCliTests.test_vs5_slice_001_value_plane_safe_brief_ask`
- `cornerstone scenario verify vs5-slice-001 --json`
- Targeted CLI transcript using a temporary state dir:
  - `cornerstone artifact ingest --text ... --source user_paste --state-dir <tmp> --json`
  - `cornerstone search query ... --state-dir <tmp> --json`
  - `cornerstone evidence bundle create --search-snapshot-id <id> --state-dir <tmp> --json`
  - `cornerstone brief create --evidence-bundle-id <id> --model-provider local_test --state-dir <tmp> --json`
  - `cornerstone conversation start --message ... --state-dir <tmp> --json`
  - `cornerstone conversation answer <id> --question ... --model-provider local_test --state-dir <tmp> --json`
- `scripts/verify_sot_docs.sh`
- `python3 scripts/verify_scenario_matrix.py`
- `git diff --check`

Required evidence surfaces: source file line refs for label assignment, CLI JSON snippets proving labels/metadata/audit refs, unit test result, scenario verifier result, and explicit note that H01 remains HUMAN_REQUIRED.
