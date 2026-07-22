# VS5 Human Evidence Package

These records complete only the human-owned rows of the frozen VS5 contract. Automation may prepare evidence, but it must not fill reviewer judgments or participant outcomes.

Required order:

1. `VS4-H01` is currently `APPROVE_WITH_EXCEPTIONS` (2026-07-12), so external sessions are authorized. The Ask History exception has been implemented and AI-verified; a separate human confirmation of that follow-up has not been recorded.
2. After every canonical 9B corpus run, execute `python3 scripts/prepare_vs5_human_review_inputs.py` and then `python3 scripts/prepare_vs5_human_review_inputs.py --check`. This binds the review inputs to the current Brief IDs and excerpts instead of a stale model run.
3. Review the frozen corpus itself in `corpus-quality-review.prefilled.json`; confirm cohort fit, domain specificity, realistic messiness, and representative multi-source/conflict/gap coverage.
4. Review at least 10 current-model Briefs statement by statement for faithfulness. `faithfulness-review.prefilled.json` contains the selected current-run statements, citations, exact source excerpts, generated gaps/next steps, and the frozen corpus's planted gap/contradiction terms. Complete both the statement judgments and `gap_and_conflict_review`; only human judgments remain blank.
5. Review at least 10 current-run answerable/unanswerable Ask pairs in `ask-review.prefilled.json`; confirm directness, faithfulness, plain insufficiency, and absence of unsupported facts.
6. Collect usefulness ratings from at least two reviewers, including one non-owner. `usefulness-review.prefilled.json` pins all current-run corpus Briefs and leaves ratings and rationale blank.
7. Run five unaided external stranger-test sessions and store one dated record per participant in `external-sessions/`. Retain one consented three-minute unedited recording reference.
8. After all records are complete, run `PATH="$PWD:$PATH" cornerstone scenario verify vs5-citation-grounded-brief --reuse-vs5-current-run --json --output reports/scenario/vs5-citation-grounded-brief-2026-07-12.json`. This validates the human evidence without regenerating the Brief and Ask IDs that were reviewed. If it reports a stale reusable run, perform a fresh full 9B run, refresh the prefilled inputs, and review the new revision instead of bypassing the mismatch.

Follow each current template's binding fields. Corpus-derived records name the frozen corpus hash; model-output reviews additionally name the model stack and prompt/retrieval revision; every completed record names its date and responsible reviewer or participant role. Changing the model, prompt scheme, retrieval pipeline, or corpus invalidates dependent Plane 2 evidence.

Prefilled files are review inputs, not evidence of acceptance. Do not rename them to the canonical filled-record paths until a reviewer has completed every required judgment field.

Canonical completed paths are `corpus-quality-review.json`, `faithfulness-review.json`, `ask-review.json`, and `usefulness-review.json`. Keep the `.prefilled.json` files unchanged as current-run inputs; copy one to its canonical path, complete it, and set a dated explicit `decision`. The VS5 verifier ignores incomplete, stale-revision, or threshold-failing records.
