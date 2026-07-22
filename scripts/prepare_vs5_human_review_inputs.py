#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PACKAGES_DIR = ROOT / "packages"
if str(PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGES_DIR))

from cornerstone_cli.vs5_corpus import (  # noqa: E402
    load_vs5_corpus,
    validate_vs5_corpus_freeze,
)


REPORT_PATH = ROOT / "reports/scenario/vs5-citation-grounded-brief-2026-07-12.json"
CORPUS_RELATIVE_PATH = "fixtures/vs5/edgar-eval/manifest.json"
CORPUS_FREEZE_RELATIVE_PATH = "fixtures/vs5/edgar-eval/freeze.json"
STATE_DIR = ROOT / "tmp/scenario-state/vs5-citation-grounded-brief"
FAITHFULNESS_PATH = ROOT / "reports/human-gates/vs5/faithfulness-review.prefilled.json"
USEFULNESS_PATH = ROOT / "reports/human-gates/vs5/usefulness-review.prefilled.json"
ASK_TEMPLATE_PATH = ROOT / "reports/human-gates/vs5/ask-review.template.json"
ASK_PATH = ROOT / "reports/human-gates/vs5/ask-review.prefilled.json"
CORPUS_TEMPLATE_PATH = ROOT / "reports/human-gates/vs5/corpus-quality-review.template.json"
CORPUS_REVIEW_PATH = ROOT / "reports/human-gates/vs5/corpus-quality-review.prefilled.json"
EXTERNAL_SESSION_TEMPLATE_PATH = ROOT / "reports/human-gates/vs5/external-session.template.json"
EXTERNAL_ROUND_TEMPLATE_PATH = ROOT / "reports/human-gates/vs5/external-round.template.json"
EXTERNAL_RUNTIME_EVIDENCE_TEMPLATE_PATH = ROOT / "reports/human-gates/vs5/external-runtime-evidence.template.json"
EXTERNAL_EVIDENCE_AUDIT_TEMPLATE_PATH = ROOT / "reports/human-gates/vs5/external-evidence-audit.template.json"
VS4_H01_PATH = ROOT / "reports/human-gates/vs4/filled-records/VS4-H01.review-record.json"


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _load_bound_corpus() -> tuple[dict[str, Any], dict[str, Any]]:
    corpus, binding = load_vs5_corpus(ROOT, CORPUS_RELATIVE_PATH)
    validate_vs5_corpus_freeze(ROOT, CORPUS_FREEZE_RELATIVE_PATH, binding)
    return corpus, binding


def _human_review_case_ids(corpus: dict[str, Any]) -> tuple[str, ...]:
    raw_case_ids = corpus.get("human_review_case_ids")
    if not isinstance(raw_case_ids, list):
        raise ValueError("The VS5 corpus does not declare human_review_case_ids.")
    case_ids = tuple(str(case_id).strip() for case_id in raw_case_ids)
    if len(case_ids) != 10 or len(set(case_ids)) != 10 or any(not case_id for case_id in case_ids):
        raise ValueError("The VS5 human-review sample must contain exactly ten unique case IDs.")
    return case_ids


def _source_bindings_by_case(
    corpus: dict[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    bindings: dict[str, dict[str, dict[str, Any]]] = {}
    for case in corpus.get("cases", []):
        case_id = str(case.get("id") or "")
        case_bindings = bindings.setdefault(case_id, {})
        for source in case.get("sources", []):
            text = str(source.get("text") or "")
            artifact_id = f"art_{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
            source_ref = str(source.get("source_ref") or source.get("upload_path") or "")
            if not source_ref:
                raise ValueError(f"Corpus source has no upload source_ref: {source.get('source_id')}")
            binding = {
                "source_ref": source_ref,
                "source_id": str(source.get("source_id") or ""),
                "source_url": str(source.get("source_url") or ""),
                "filing_index_url": str(source.get("filing_index_url") or ""),
                "accession_number": str(source.get("accession_number") or ""),
                "form_type": str(source.get("form_type") or ""),
                "filing_date": str(source.get("filing_date") or ""),
                "raw_path": str(source.get("raw_path") or ""),
                "raw_sha256": str(source.get("raw_sha256") or ""),
                "normalized_path": str(source.get("normalized_path") or ""),
                "normalized_sha256": str(source.get("normalized_sha256") or ""),
                "upload_path": str(source.get("upload_path") or ""),
                "upload_sha256": str(source.get("upload_sha256") or ""),
            }
            existing = case_bindings.get(artifact_id)
            if existing is not None and existing != binding:
                raise ValueError(
                    f"One case maps identical upload text to multiple sources: {case_id}:{artifact_id}"
                )
            case_bindings[artifact_id] = binding
    return bindings


def _required_source_binding(
    source_bindings: dict[str, dict[str, dict[str, Any]]],
    case_id: str,
    artifact_id: str,
) -> dict[str, Any]:
    binding = source_bindings.get(case_id, {}).get(artifact_id)
    if not case_id or not artifact_id or binding is None:
        raise ValueError(
            "Runtime artifact is not bound to this case's corpus upload: "
            f"{case_id or 'missing-case'}:{artifact_id or 'missing-artifact'}"
        )
    return dict(binding)


def _selected_records(
    corpus: dict[str, Any],
    corpus_binding: dict[str, Any],
    review_case_ids: tuple[str, ...],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    report = _load(REPORT_PATH)
    report_corpus = report.get("corpus")
    if not isinstance(report_corpus, dict):
        raise ValueError("The VS5 report has no corpus binding.")
    if report_corpus.get("binding") != corpus_binding:
        raise ValueError("VS5 report and exact corpus binding do not match.")
    if report_corpus.get("manifest_sha256") != corpus_binding["manifest_sha256"]:
        raise ValueError("VS5 report and corpus manifest hash do not match.")
    rows = {str(row.get("case_id") or ""): row for row in report.get("case_results", [])}
    missing = [case_id for case_id in review_case_ids if case_id not in rows]
    if missing:
        raise ValueError(f"Selected cases are missing from the current report: {missing}")
    selected: list[dict[str, Any]] = []
    for case_id in review_case_ids:
        row = rows[case_id]
        brief_id = str(row.get("brief_id") or "")
        brief_path = STATE_DIR / "briefs" / f"{brief_id}.json"
        if not brief_path.exists():
            raise ValueError(f"Current Brief record is missing: {brief_path}")
        brief = _load(brief_path)
        if str(brief.get("brief_id") or "") != brief_id:
            raise ValueError(f"Brief identity mismatch: {brief_path}")
        selected.append({"case_result": row, "brief": brief})
    return report, selected, _source_bindings_by_case(corpus)


def _faithfulness_reviews(
    selected: list[dict[str, Any]],
    source_bindings: dict[str, dict[str, dict[str, Any]]],
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
                chunk_id = (
                    citation_ref.split(":", 1)[1]
                    if citation_ref.startswith("evidence_chunk:")
                    else ""
                )
                chunk_path = STATE_DIR / "evidence" / "chunks" / f"{chunk_id}.json"
                if not chunk_path.exists():
                    raise ValueError(f"Citation chunk missing for {citation_ref}")
                chunk = _load(chunk_path)
                if (
                    str(chunk.get("artifact_id") or "") != artifact_id
                    or chunk.get("span") != link.get("span")
                ):
                    raise ValueError(f"Citation link/chunk mismatch for {citation_ref}")
                source_evidence.append(
                    {
                        "citation_ref": citation_ref,
                        **_required_source_binding(
                            source_bindings,
                            str(result["case_id"]),
                            artifact_id,
                        ),
                        "artifact_id": artifact_id,
                        "span": chunk.get("span"),
                        "source_excerpt": str(chunk.get("text") or ""),
                    }
                )
            statement_reviews.append(
                {
                    "section": statement.get("section"),
                    "statement_type": statement.get("statement_type"),
                    "presented_as_fact": statement.get("presented_as_fact"),
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
                "generated_conflicts_risks": list(brief.get("conflicts_risks") or []),
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


def _usefulness_sample(
    selected: list[dict[str, Any]],
    corpus: dict[str, Any],
) -> list[dict[str, Any]]:
    cases = {str(case.get("id") or ""): case for case in corpus.get("cases", [])}
    sample = []
    for item in selected:
        result = item["case_result"]
        brief = item["brief"]
        case = cases.get(str(result.get("case_id") or ""))
        if case is None:
            raise ValueError(f"Selected corpus case is missing: {result.get('case_id')}")
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
        statements = [
            {
                "section": str(row.get("section") or ""),
                "statement": str(row.get("statement") or ""),
                "citation_refs": [str(ref) for ref in row.get("citation_refs", [])],
            }
            for row in brief.get("load_bearing_statements", [])
            if isinstance(row, dict) and str(row.get("statement") or "").strip()
        ]
        source_set = []
        for source in case.get("sources", []):
            if not isinstance(source, dict):
                continue
            text = str(source.get("text") or "")
            text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if text_sha256 != str(source.get("upload_sha256") or ""):
                raise ValueError(
                    f"Loaded source text no longer matches upload_sha256: {source.get('source_id')}"
                )
            source_set.append(
                {
                    "name": str(source.get("name") or "source"),
                    "source_id": str(source.get("source_id") or ""),
                    "source_ref": str(source.get("source_ref") or source.get("upload_path") or ""),
                    "source_url": str(source.get("source_url") or ""),
                    "final_url": str(source.get("final_url") or ""),
                    "accession_number": str(source.get("accession_number") or ""),
                    "form_type": str(source.get("form_type") or ""),
                    "filing_date": str(source.get("filing_date") or ""),
                    "exhibit_number": str(source.get("exhibit_number") or ""),
                    "document_name": str(source.get("document_name") or ""),
                    "retrieved_at": str(source.get("retrieved_at") or ""),
                    "verified_at": str(source.get("verified_at") or ""),
                    "raw_path": str(source.get("raw_path") or ""),
                    "raw_sha256": str(source.get("raw_sha256") or ""),
                    "normalized_path": str(source.get("normalized_path") or ""),
                    "normalized_sha256": str(source.get("normalized_sha256") or ""),
                    "upload_path": str(source.get("upload_path") or ""),
                    "upload_sha256": str(source.get("upload_sha256") or ""),
                    "upload_bytes": source.get("upload_bytes"),
                    "text": text,
                    "sha256": text_sha256,
                }
            )
        sample.append(
            {
                "case_id": result["case_id"],
                "archetype": result["archetype"],
                "brief_id": brief["brief_id"],
                "brief_status": brief.get("status"),
                "title": brief.get("title"),
                "decision_question": brief.get("decision_question"),
                "bottom_line": brief.get("bottom_line") or brief.get("summary"),
                "key_facts": [
                    row["statement"] for row in statements if row["section"] == "key_facts"
                ],
                "conflicts_risks": [
                    row["statement"] for row in statements if row["section"] == "conflicts_risks"
                ],
                "missing_evidence": missing,
                "recommended_next_steps": next_steps,
                "load_bearing_statements": statements,
                "source_set": source_set,
            }
        )
    return sample


def _answer_reviews(
    corpus: dict[str, Any],
    source_bindings: dict[str, dict[str, dict[str, Any]]],
    review_case_ids: tuple[str, ...],
) -> list[dict[str, Any]]:
    cases = {str(case.get("id") or ""): case for case in corpus.get("cases", [])}
    answers = {}
    for path in (STATE_DIR / "answers").glob("*.json"):
        answer = _load(path)
        answers[str(answer.get("question") or "")] = answer

    def review_entry(case_id: str, question: str, *, answerable: bool) -> dict[str, Any]:
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
                    **_required_source_binding(source_bindings, case_id, artifact_id),
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
                        **_required_source_binding(source_bindings, case_id, artifact_id),
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
            "supporting_result_count": answer.get("supporting_result_count"),
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
    for case_id in review_case_ids:
        case = cases.get(case_id)
        if case is None:
            raise ValueError(f"Selected corpus case is missing: {case_id}")
        reviews.append(
            {
                "case_id": case_id,
                "archetype": case.get("archetype"),
                "answerable": review_entry(
                    case_id,
                    str(case.get("answerable_question") or ""),
                    answerable=True,
                ),
                "unanswerable": review_entry(
                    case_id,
                    str(case.get("unanswerable_question") or ""),
                    answerable=False,
                ),
            }
        )
    return reviews


def build_inputs() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    corpus, corpus_binding = _load_bound_corpus()
    review_case_ids = _human_review_case_ids(corpus)
    report, selected, source_bindings = _selected_records(
        corpus,
        corpus_binding,
        review_case_ids,
    )
    all_case_ids = tuple(str(case.get("id") or "") for case in corpus.get("cases", []))
    if not all_case_ids or any(not case_id for case_id in all_case_ids):
        raise ValueError("The VS5 corpus contains an empty case ID.")
    _, all_selected, _ = _selected_records(
        corpus,
        corpus_binding,
        all_case_ids,
    )
    model_stack = report.get("model_stack")
    revision = str((model_stack or {}).get("pipeline_sha256") or "")
    faithfulness = _load(FAITHFULNESS_PATH)
    faithfulness.update(
        {
            "status": "HUMAN_REQUIRED",
            "reviewed_at": None,
            "reviewer": {"name": None, "role": None, "is_owner": None},
            "corpus_manifest_sha256": corpus_binding["manifest_sha256"],
            "corpus_bundle_sha256": corpus_binding["bundle_sha256"],
            "corpus_source_manifest_sha256": corpus_binding["source_files"]["manifest_sha256"],
            "model_stack": model_stack,
            "prompt_retrieval_revision": revision,
            "selected_brief_count": len(selected),
            "review_instructions": [
                "For every statement, compare the text with every listed source excerpt.",
                "Set faithful to true only when the statement preserves numbers, dates, actors, modality, conditions, and direction.",
                "Set material_overstatement to true for any contradiction, inversion, unsupported consequence, or dropped qualifier that changes meaning.",
                "Review bottom-line decision_synthesis as a cited assessment, not as a sourced fact; reject it if its rationale does not support the recommendation.",
                "Every recommended_next_steps row must be present in statements, remain presented_as_fact=false, cite its factual basis, and avoid inventing an approver, signer, owner, obligation, or deadline.",
                "Review generated_missing_evidence and generated_conflicts_risks together as the uncertainty surface: missing_evidence is for absent information, while known failures and unresolved conditions belong in conflicts_risks. Confirm every planted gap and contradiction term appears in the appropriate section, then complete all gap_and_conflict_review fields.",
                "Do not change automated_anchor_status; it is mechanical context only and never decides the human judgment.",
            ],
            "brief_reviews": _faithfulness_reviews(selected, source_bindings, corpus),
            "decision": None,
        }
    )
    usefulness = _load(USEFULNESS_PATH)
    brief_sample = _usefulness_sample(all_selected, corpus)
    brief_ids = [row["brief_id"] for row in brief_sample]
    usefulness.update(
        {
            "status": "HUMAN_REQUIRED",
            "reviewed_at": None,
            "corpus_manifest_sha256": corpus_binding["manifest_sha256"],
            "corpus_bundle_sha256": corpus_binding["bundle_sha256"],
            "corpus_source_manifest_sha256": corpus_binding["source_files"]["manifest_sha256"],
            "model_stack": model_stack,
            "prompt_retrieval_revision": revision,
            "brief_sample": brief_sample,
            "review_instructions": [
                "Read every original source in source_set before rating its Brief.",
                "Review the complete Brief: bottom line, key facts, conflicts/risks, missing evidence, and recommended next steps.",
                "Rate 4 or 5 only when the Brief materially reduces decision-preparation work compared with reading the sources while preserving important nuance.",
                "Review all corpus Briefs and explain concrete strengths or defects in the rationale.",
            ],
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
            "corpus_manifest_sha256": corpus_binding["manifest_sha256"],
            "corpus_bundle_sha256": corpus_binding["bundle_sha256"],
            "corpus_source_manifest_sha256": corpus_binding["source_files"]["manifest_sha256"],
            "model_stack": model_stack,
            "prompt_retrieval_revision": revision,
            "selected_case_count": len(review_case_ids),
            "answer_reviews": _answer_reviews(corpus, source_bindings, review_case_ids),
            "decision": None,
        }
    )
    corpus_review = _load(CORPUS_TEMPLATE_PATH)
    corpus_review.update(
        {
            "status": "HUMAN_REQUIRED",
            "reviewed_at": None,
            "reviewer": {"name": None, "role": None, "is_owner": None},
            "corpus_manifest_sha256": corpus_binding["manifest_sha256"],
            "corpus_bundle_sha256": corpus_binding["bundle_sha256"],
            "corpus_source_manifest_sha256": corpus_binding["source_files"]["manifest_sha256"],
            "case_count": corpus_binding["case_count"],
            "target_cohort_fit": None,
            "domain_specific_and_non_generic": None,
            "messy_input_is_realistic": None,
            "multi_source_conflict_gap_coverage_is_representative": None,
            "review_note": None,
            "decision": None,
        }
    )
    external_session_template = _load(EXTERNAL_SESSION_TEMPLATE_PATH)
    external_session_template.update(
        {
            "status": "HUMAN_REQUIRED_EXTERNAL",
            "corpus_manifest_sha256": corpus_binding["manifest_sha256"],
            "corpus_bundle_sha256": corpus_binding["bundle_sha256"],
            "corpus_source_manifest_sha256": corpus_binding["source_files"]["manifest_sha256"],
            "model_stack": model_stack,
            "prompt_retrieval_revision": revision,
        }
    )
    h01_record = _load(VS4_H01_PATH)
    external_round_template = _load(EXTERNAL_ROUND_TEMPLATE_PATH)
    external_round_template.update(
        {
            "status": "HUMAN_REQUIRED_EXTERNAL",
            "corpus_manifest_sha256": corpus_binding["manifest_sha256"],
            "corpus_bundle_sha256": corpus_binding["bundle_sha256"],
            "corpus_source_manifest_sha256": corpus_binding["source_files"]["manifest_sha256"],
            "model_stack": model_stack,
            "prompt_retrieval_revision": revision,
            "prerequisite": {
                "vs4_h01_decision": h01_record.get("decision"),
                "vs4_h01_reviewed_at": h01_record.get("reviewed_at"),
                "vs4_h01_record": "reports/human-gates/vs4/filled-records/VS4-H01.review-record.json",
            },
        }
    )
    external_runtime_evidence_template = _load(EXTERNAL_RUNTIME_EVIDENCE_TEMPLATE_PATH)
    external_runtime_evidence_template["prompt_retrieval_revision"] = revision
    external_evidence_audit_template = _load(EXTERNAL_EVIDENCE_AUDIT_TEMPLATE_PATH)
    external_evidence_audit_template.update(
        {
            "status": "HUMAN_REQUIRED_EXTERNAL",
            "corpus_manifest_sha256": corpus_binding["manifest_sha256"],
            "corpus_bundle_sha256": corpus_binding["bundle_sha256"],
            "corpus_source_manifest_sha256": corpus_binding["source_files"]["manifest_sha256"],
            "model_stack": model_stack,
            "prompt_retrieval_revision": revision,
        }
    )
    return (
        faithfulness,
        usefulness,
        ask,
        corpus_review,
        external_session_template,
        external_round_template,
        external_runtime_evidence_template,
        external_evidence_audit_template,
    )


def _encoded(value: dict[str, Any]) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh VS5 human-review inputs from the current canonical report.")
    parser.add_argument("--check", action="store_true", help="Fail when the prefilled review inputs are stale.")
    args = parser.parse_args()
    (
        faithfulness,
        usefulness,
        ask,
        corpus_review,
        external_session_template,
        external_round_template,
        external_runtime_evidence_template,
        external_evidence_audit_template,
    ) = build_inputs()
    expected = {
        FAITHFULNESS_PATH: _encoded(faithfulness),
        USEFULNESS_PATH: _encoded(usefulness),
        ASK_PATH: _encoded(ask),
        CORPUS_REVIEW_PATH: _encoded(corpus_review),
        EXTERNAL_SESSION_TEMPLATE_PATH: _encoded(external_session_template),
        EXTERNAL_ROUND_TEMPLATE_PATH: _encoded(external_round_template),
        EXTERNAL_RUNTIME_EVIDENCE_TEMPLATE_PATH: _encoded(external_runtime_evidence_template),
        EXTERNAL_EVIDENCE_AUDIT_TEMPLATE_PATH: _encoded(external_evidence_audit_template),
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
