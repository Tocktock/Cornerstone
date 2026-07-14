#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT_PATH = ROOT / "reports/scenario/vs5-citation-grounded-brief-2026-07-12.json"
CORPUS_PATH = ROOT / "fixtures/vs5/eval/manifest.json"
STATE_DIR = ROOT / "tmp/scenario-state/vs5-citation-grounded-brief"
FAITHFULNESS_PATH = ROOT / "reports/human-gates/vs5/faithfulness-review.prefilled.json"
USEFULNESS_PATH = ROOT / "reports/human-gates/vs5/usefulness-review.prefilled.json"
ASK_TEMPLATE_PATH = ROOT / "reports/human-gates/vs5/ask-review.template.json"
ASK_PATH = ROOT / "reports/human-gates/vs5/ask-review.prefilled.json"
CORPUS_TEMPLATE_PATH = ROOT / "reports/human-gates/vs5/corpus-quality-review.template.json"
CORPUS_REVIEW_PATH = ROOT / "reports/human-gates/vs5/corpus-quality-review.prefilled.json"
SELECTED_CASE_IDS = (
    "vendor-renewal-01",
    "vendor-renewal-03",
    "vendor-renewal-04",
    "vendor-renewal-06",
    "policy-change-02",
    "policy-change-05",
    "policy-change-06",
    "project-risk-01",
    "project-risk-05",
    "project-risk-07",
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_ref_by_artifact(corpus: dict[str, Any]) -> dict[str, str]:
    refs: dict[str, str] = {}
    for case in corpus.get("cases", []):
        case_id = str(case.get("id") or "")
        for source in case.get("sources", []):
            text = str(source.get("text") or "")
            artifact_id = f"art_{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
            refs[artifact_id] = f"{case_id}:{source.get('name') or 'source'}"
    return refs


def _selected_records() -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, str]]:
    report = _load(REPORT_PATH)
    corpus = _load(CORPUS_PATH)
    manifest_sha256 = _sha256(CORPUS_PATH)
    if str(report.get("corpus", {}).get("manifest_sha256") or "") != manifest_sha256:
        raise ValueError("VS5 report and corpus manifest hash do not match.")
    rows = {str(row.get("case_id") or ""): row for row in report.get("case_results", [])}
    missing = [case_id for case_id in SELECTED_CASE_IDS if case_id not in rows]
    if missing:
        raise ValueError(f"Selected cases are missing from the current report: {missing}")
    selected: list[dict[str, Any]] = []
    for case_id in SELECTED_CASE_IDS:
        row = rows[case_id]
        brief_id = str(row.get("brief_id") or "")
        brief_path = STATE_DIR / "briefs" / f"{brief_id}.json"
        if not brief_path.exists():
            raise ValueError(f"Current Brief record is missing: {brief_path}")
        brief = _load(brief_path)
        if str(brief.get("brief_id") or "") != brief_id:
            raise ValueError(f"Brief identity mismatch: {brief_path}")
        selected.append({"case_result": row, "brief": brief})
    return report, selected, _source_ref_by_artifact(corpus)


def _faithfulness_reviews(
    selected: list[dict[str, Any]],
    source_refs: dict[str, str],
    corpus: dict[str, Any],
) -> list[dict[str, Any]]:
    cases = {str(case.get("id") or ""): case for case in corpus.get("cases", [])}
    reviews: list[dict[str, Any]] = []
    for item in selected:
        result = item["case_result"]
        brief = item["brief"]
        case = cases.get(str(result.get("case_id") or ""))
        if case is None:
            raise ValueError(f"Selected corpus case is missing: {result.get('case_id')}")
        links = {
            str(link.get("evidence_chunk_ref") or ""): link
            for link in brief.get("evidence_links", [])
            if isinstance(link, dict)
        }
        anchor_checks = brief.get("statement_anchor_checks", [])
        statements = brief.get("load_bearing_statements", [])
        if len(anchor_checks) != len(statements):
            raise ValueError(f"Statement/anchor count mismatch for {brief.get('brief_id')}")
        statement_reviews = []
        for statement, anchor in zip(statements, anchor_checks, strict=True):
            citation_refs = [str(ref) for ref in statement.get("citation_refs", [])]
            if citation_refs != [str(ref) for ref in anchor.get("citation_refs", [])]:
                raise ValueError(f"Statement/anchor citation mismatch for {brief.get('brief_id')}")
            source_evidence = []
            for citation_ref in citation_refs:
                link = links.get(citation_ref)
                if link is None:
                    raise ValueError(f"Citation link missing for {citation_ref}")
                artifact_ref = str(link.get("artifact_ref") or "")
                artifact_id = artifact_ref.split(":", 1)[1] if artifact_ref.startswith("artifact:") else ""
                source_evidence.append(
                    {
                        "citation_ref": citation_ref,
                        "source_ref": source_refs.get(artifact_id, f"{result['case_id']}:source"),
                        "artifact_id": artifact_id,
                        "span": link.get("span"),
                        "source_excerpt": str(link.get("snippet") or ""),
                    }
                )
            statement_reviews.append(
                {
                    "section": statement.get("section"),
                    "statement": statement.get("statement"),
                    "citation_refs": citation_refs,
                    "source_evidence": source_evidence,
                    "automated_anchor_status": anchor.get("status"),
                    "faithful": None,
                    "material_overstatement": None,
                    "reviewer_note": None,
                }
            )
        reviews.append(
            {
                "case_id": result["case_id"],
                "archetype": result["archetype"],
                "brief_id": brief["brief_id"],
                "brief_status": brief.get("status"),
                "title": brief.get("title"),
                "decision_question": brief.get("decision_question"),
                "statements": statement_reviews,
                "generated_missing_evidence": list(brief.get("missing_evidence") or []),
                "generated_recommended_next_steps": list(brief.get("recommended_next_steps") or []),
                "planted_expectations": {
                    "gap_terms": list(case.get("gap_terms") or []),
                    "contradiction_terms": list(case.get("contradiction_terms") or []),
                },
                "gap_and_conflict_review": {
                    "all_planted_gap_terms_addressed": None,
                    "all_planted_contradictions_addressed": None,
                    "missing_evidence_is_specific": None,
                    "reviewer_note": None,
                },
                "conflicts_and_gaps_match_sources": None,
                "brief_review_note": None,
            }
        )
    return reviews


def _usefulness_sample(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sample = []
    for item in selected:
        result = item["case_result"]
        brief = item["brief"]
        missing = brief.get("missing_evidence")
        if not isinstance(missing, list) or not missing:
            missing = [
                str(value)
                for value in [*(brief.get("gaps") or []), *(brief.get("uncertainty") or [])]
            ]
        next_steps = brief.get("recommended_next_steps")
        if not isinstance(next_steps, list) or not next_steps:
            single = str(brief.get("recommended_next_step") or "").strip()
            next_steps = [single] if single else []
        sample.append(
            {
                "case_id": result["case_id"],
                "archetype": result["archetype"],
                "brief_id": brief["brief_id"],
                "brief_status": brief.get("status"),
                "title": brief.get("title"),
                "decision_question": brief.get("decision_question"),
                "bottom_line": brief.get("bottom_line") or brief.get("summary"),
                "missing_evidence": missing,
                "recommended_next_steps": next_steps,
            }
        )
    return sample


def _answer_reviews(corpus: dict[str, Any], source_refs: dict[str, str]) -> list[dict[str, Any]]:
    cases = {str(case.get("id") or ""): case for case in corpus.get("cases", [])}
    answers = {}
    for path in (STATE_DIR / "answers").glob("*.json"):
        answer = _load(path)
        answers[str(answer.get("question") or "")] = answer

    def review_entry(question: str, *, answerable: bool) -> dict[str, Any]:
        answer = answers.get(question)
        if answer is None:
            raise ValueError(f"Current Ask record is missing for question: {question}")
        source_evidence = []
        citation_resolution_errors = []
        for citation_ref in answer.get("citation_refs", []):
            citation_ref = str(citation_ref)
            if not citation_ref.startswith("evidence_chunk:"):
                continue
            chunk_id = citation_ref.split(":", 1)[1]
            chunk_path = STATE_DIR / "evidence" / "chunks" / f"{chunk_id}.json"
            if not chunk_path.exists():
                citation_resolution_errors.append(
                    {"citation_ref": citation_ref, "error": "citation_chunk_not_found"}
                )
                continue
            chunk = _load(chunk_path)
            artifact_id = str(chunk.get("artifact_id") or "")
            source_evidence.append(
                {
                    "citation_ref": citation_ref,
                    "source_ref": source_refs.get(artifact_id, "source"),
                    "artifact_id": artifact_id,
                    "span": chunk.get("span"),
                    "source_excerpt": str(chunk.get("text") or ""),
                }
            )
        if not source_evidence:
            for evidence_ref in answer.get("evidence_refs", []):
                evidence_ref = str(evidence_ref)
                if not evidence_ref.startswith("evidence_chunk:"):
                    continue
                chunk_id = evidence_ref.split(":", 1)[1]
                chunk_path = STATE_DIR / "evidence" / "chunks" / f"{chunk_id}.json"
                if not chunk_path.exists():
                    continue
                chunk = _load(chunk_path)
                artifact_id = str(chunk.get("artifact_id") or "")
                source_evidence.append(
                    {
                        "citation_ref": evidence_ref,
                        "source_ref": source_refs.get(artifact_id, "source"),
                        "artifact_id": artifact_id,
                        "span": chunk.get("span"),
                        "source_excerpt": str(chunk.get("text") or ""),
                        "retrieved_context_only": True,
                    }
                )
        entry = {
            "question": question,
            "answer_id": answer.get("answer_id"),
            "answer": answer.get("answer"),
            "label": answer.get("label"),
            "citation_refs": answer.get("citation_refs", []),
            "citation_resolution_errors": citation_resolution_errors,
            "source_evidence": source_evidence,
            "reviewer_note": None,
        }
        if answerable:
            entry.update({"directly_answers_question": None, "faithful_to_cited_evidence": None})
        else:
            entry.update({"plainly_declines": None, "adds_unsupported_fact": None})
        return entry

    reviews = []
    for case_id in SELECTED_CASE_IDS:
        case = cases.get(case_id)
        if case is None:
            raise ValueError(f"Selected corpus case is missing: {case_id}")
        reviews.append(
            {
                "case_id": case_id,
                "archetype": case.get("archetype"),
                "answerable": review_entry(str(case.get("answerable_question") or ""), answerable=True),
                "unanswerable": review_entry(str(case.get("unanswerable_question") or ""), answerable=False),
            }
        )
    return reviews


def build_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    report, selected, source_refs = _selected_records()
    corpus = _load(CORPUS_PATH)
    model_stack = report.get("model_stack")
    revision = str((model_stack or {}).get("pipeline_sha256") or "")
    faithfulness = _load(FAITHFULNESS_PATH)
    faithfulness.update(
        {
            "status": "HUMAN_REQUIRED",
            "reviewed_at": None,
            "reviewer": {"name": None, "role": None, "is_owner": None},
            "corpus_manifest_sha256": report["corpus"]["manifest_sha256"],
            "model_stack": model_stack,
            "prompt_retrieval_revision": revision,
            "selected_brief_count": len(selected),
            "review_instructions": [
                "For every statement, compare the text with every listed source excerpt.",
                "Set faithful to true only when the statement preserves numbers, dates, actors, modality, conditions, and direction.",
                "Set material_overstatement to true for any contradiction, inversion, unsupported consequence, or dropped qualifier that changes meaning.",
                "Compare generated_missing_evidence with every planted gap term and generated conflict statement with every planted contradiction term; complete all gap_and_conflict_review fields.",
                "Do not change automated_anchor_status; it is mechanical context only and never decides the human judgment.",
            ],
            "brief_reviews": _faithfulness_reviews(selected, source_refs, corpus),
            "decision": None,
        }
    )
    usefulness = _load(USEFULNESS_PATH)
    brief_sample = _usefulness_sample(selected)
    brief_ids = [row["brief_id"] for row in brief_sample]
    usefulness.update(
        {
            "status": "HUMAN_REQUIRED",
            "reviewed_at": None,
            "corpus_manifest_sha256": report["corpus"]["manifest_sha256"],
            "model_stack": model_stack,
            "prompt_retrieval_revision": revision,
            "brief_sample": brief_sample,
            "reviews": [
                {
                    "reviewer_name": None,
                    "reviewer_role": None,
                    "is_owner": True,
                    "brief_ids": brief_ids,
                    "usefulness_rating_1_to_5": None,
                    "rationale": None,
                },
                {
                    "reviewer_name": None,
                    "reviewer_role": None,
                    "is_owner": False,
                    "brief_ids": brief_ids,
                    "usefulness_rating_1_to_5": None,
                    "rationale": None,
                },
            ],
            "decision": None,
        }
    )
    ask = _load(ASK_TEMPLATE_PATH)
    ask.update(
        {
            "status": "HUMAN_REQUIRED",
            "reviewed_at": None,
            "reviewer": {"name": None, "role": None, "is_owner": None},
            "corpus_manifest_sha256": report["corpus"]["manifest_sha256"],
            "model_stack": model_stack,
            "prompt_retrieval_revision": revision,
            "selected_case_count": len(SELECTED_CASE_IDS),
            "answer_reviews": _answer_reviews(corpus, source_refs),
            "decision": None,
        }
    )
    corpus_review = _load(CORPUS_TEMPLATE_PATH)
    corpus_review.update(
        {
            "status": "HUMAN_REQUIRED",
            "reviewed_at": None,
            "reviewer": {"name": None, "role": None, "is_owner": None},
            "corpus_manifest_sha256": report["corpus"]["manifest_sha256"],
            "case_count": report["corpus"]["case_count"],
            "target_cohort_fit": None,
            "domain_specific_and_non_generic": None,
            "messy_input_is_realistic": None,
            "multi_source_conflict_gap_coverage_is_representative": None,
            "review_note": None,
            "decision": None,
        }
    )
    return faithfulness, usefulness, ask, corpus_review


def _encoded(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh VS5 human-review inputs from the current canonical report.")
    parser.add_argument("--check", action="store_true", help="Fail when the prefilled review inputs are stale.")
    args = parser.parse_args()
    faithfulness, usefulness, ask, corpus_review = build_inputs()
    expected = {
        FAITHFULNESS_PATH: _encoded(faithfulness),
        USEFULNESS_PATH: _encoded(usefulness),
        ASK_PATH: _encoded(ask),
        CORPUS_REVIEW_PATH: _encoded(corpus_review),
    }
    stale = [path for path, content in expected.items() if not path.exists() or path.read_text() != content]
    if args.check:
        if stale:
            print("STALE: " + ", ".join(str(path.relative_to(ROOT)) for path in stale))
            return 1
        print("PASS: VS5 human-review inputs match the current report and runtime records.")
        return 0
    for path, content in expected.items():
        path.write_text(content)
    print("UPDATED: " + ", ".join(str(path.relative_to(ROOT)) for path in expected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
