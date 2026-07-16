# VS5 External Session Records

Use this protocol only after the canonical VS5 report and all four external templates have been regenerated for the exact model and pipeline revision being tested. Run one non-counting pilot first, record its privacy-preserving participant hash in `round.json`, and keep that participant outside the formal cohort.

## Participant and environment

- Recruit five different non-owner operational decision owners who have never used or helped build CornerStone. JiYong/Tars cannot count.
- Give each participant a clean workspace with no preloaded or unrelated artifacts.
- The participant supplies their own real, messy 1–5-source decision input. Do not use the frozen eval corpus, a facilitator-created sample, or another participant's input.
- Ask for recording consent before starting. At least one formal session must have a consented, unedited recording of three minutes or longer. Keep recordings and sensitive source text outside the repository; record only redacted references.

## Freeze one formal round before testing

Copy `../external-round.template.json` to `round.json` before any formal timer starts. Give the round a unique dated ID, record the coordinator, and preregister exactly five privacy-preserving stable participant hashes in attempt order. Use random study identifiers salted outside this repository; never hash an email address directly. Set `status` to `REGISTERED` and `decision` to `APPROVE` only after the cohort is frozen.

All five preregistered attempts count. Save them only as `session-01.json` through `session-05.json`; a failed or low-rated attempt cannot be replaced or omitted. A failed round remains evidence. If fixes require a new round, archive the entire old round under a dated subdirectory and preregister five new unfamiliar participants. Archived and pilot participant hashes cannot overlap a later formal cohort. Pilot session records stay outside the formal directory.

## Exact unaided task

Read this once, then do not coach or point at controls:

> Using your own source material, create a decision Brief in CornerStone. Open at least one citation and inspect its original source. When you are done, explain the Brief's conclusion and which source evidence supports it.

Start the timer immediately after reading the task. Stop it only when the participant has created the Brief, opened a citation, inspected the source, and finished both explanations. Questions about product controls count as failure of `unaided`; record what happened without helping during the timed attempt.

## Evidence to record

Copy `../external-session.template.json` to `session-01.json` through `session-05.json` and complete every field. Use `status: COMPLETED` and `decision: ACCEPT` only for a completed qualifying attempt; preserve failures with a rejection decision. Record the exact timestamps, `brief_id`, opened `citation_ref`, inspected artifact reference, participant's own words for both explanations, rating rationales, forwarding/use rationale, and an observer's accuracy assessment. A non-empty restatement alone is not a pass: both the conclusion and source-basis explanations must be accurate.

For each attempt, copy `../external-runtime-evidence.template.json` to `evidence/session-0N.runtime-evidence.json`. Populate it from the retained session export, including hashes for the Brief record, opened citation chunk, and inspected source artifact. Bind `runtime_evidence_manifest_path` in the session record to that repository-relative manifest. Do not store source text in Git.

Record failures honestly. At least one valid case must be a genuine decision the participant needed to make and must include the participant's own material-help statement. For the qualifying recording, retain a custody reference, SHA-256, verified duration, consent reference and timestamp, and assessor identity; JSON assertions do not replace the final human evidence audit.

## Final human evidence audit

After all five attempts, copy `../external-evidence-audit.template.json` to `evidence-audit.json`. Codex may calculate the round, session, and runtime-manifest hashes, but a named human must inspect the retained recruitment attestations, source/runtime custody, participant explanations, observer assessments, and recording/consent evidence. Complete every verification field only from that inspection, add at least three distinct custody/evidence references, and set `status: COMPLETED` plus `decision: ACCEPT` only when the evidence is authentic and the preregistered cohort was not replaced or cherry-picked. This audit is the formal human authenticity boundary; structurally plausible JSON alone cannot pass either external row.

After all records are present, run:

```bash
PATH="$PWD:$PATH" cornerstone scenario verify vs5-citation-grounded-brief \
  --reuse-vs5-current-run --json \
  --output reports/scenario/vs5-citation-grounded-brief-2026-07-12.json
```

Do not run a generating verification command after the reviewed Brief and Ask IDs are frozen. If reuse reports a stale run, regenerate the full 9B run, refresh every review input, and review that new revision.

The formal exit thresholds are five distinct valid sessions within ten minutes, one qualifying recording, trust and usefulness medians of at least 4/5, at least three participants willing to forward or use the Brief, and at least one real decision materially helped.
