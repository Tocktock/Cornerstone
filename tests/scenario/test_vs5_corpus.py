from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Callable

from cornerstone_cli.vs5_corpus import (
    CORPUS_ID,
    CORPUS_SCHEMA,
    EXPECTED_PROVENANCE_POLICY,
    FREEZE_SCHEMA,
    Vs5CorpusIntegrityError,
    load_vs5_corpus,
    normalize_edgar_filing_html,
)


MANIFEST_PATH = "fixtures/vs5/edgar-eval/manifest.json"
CORPUS_DIRECTORY = Path("fixtures/vs5/edgar-eval")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write(root: Path, relative: str, value: bytes) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def _make_source(
    root: Path,
    *,
    case_id: str,
    case_index: int,
    source_index: int,
    text: str,
) -> tuple[dict[str, Any], str]:
    sequence = case_index * 10 + source_index + 1
    cik = 9_000_000 + sequence
    accession = f"{cik:010d}-26-{sequence:06d}"
    compact_accession = accession.replace("-", "")
    raw = f"<html><body><p>{text}</p></body></html>".encode("utf-8")
    normalized = normalize_edgar_filing_html(raw)
    upload = normalized
    raw_sha256 = _sha256(raw)
    normalized_bytes = normalized.encode("utf-8")
    normalized_sha256 = _sha256(normalized_bytes)
    upload_bytes = upload.encode("utf-8")
    upload_sha256 = _sha256(upload_bytes)
    raw_path = f"fixtures/vs5/edgar-eval/_shared/raw/sha256/{raw_sha256}.html"
    normalized_path = (
        "fixtures/vs5/edgar-eval/_shared/normalized/sha256/"
        f"{normalized_sha256}.txt"
    )
    upload_path = (
        f"fixtures/vs5/edgar-eval/{case_id}/upload/{source_index + 1:02d}.txt"
    )
    _write(root, raw_path, raw)
    _write(root, normalized_path, normalized_bytes)
    _write(root, upload_path, upload_bytes)
    archive_root = (
        f"https://www.sec.gov/Archives/edgar/data/{cik}/{compact_accession}"
    )
    source = {
        "source_id": f"{case_id}-source-{source_index + 1:02d}",
        "case_id": case_id,
        "name": f"{source_index + 1:02d}-case-document",
        "source_order": source_index + 1,
        "cik": str(cik),
        "source_url": f"{archive_root}/document.htm",
        "final_url": f"{archive_root}/document.htm",
        "filing_index_url": f"{archive_root}/{accession}-index.html",
        "accession_number": accession,
        "http_status": 200,
        "filing_index_http_status": 200,
        "content_type": "text/html",
        "encoding": "utf-8",
        "normalizer": "cornerstone_edgar_visible_text_v1",
        "raw_path": raw_path,
        "raw_sha256": raw_sha256,
        "raw_bytes": len(raw),
        "normalized_path": normalized_path,
        "normalized_sha256": normalized_sha256,
        "normalized_bytes": len(normalized_bytes),
        "upload_path": upload_path,
        "upload_sha256": upload_sha256,
        "upload_bytes": len(upload_bytes),
        "upload_span_in_normalized": {
            "coordinate_system": "unicode_code_points",
            "char_start": 0,
            "char_end": len(normalized),
            "sha256": upload_sha256,
        },
        "supersedes": [],
        "incorporated_by_reference": [],
    }
    return source, normalized


def _support(source: dict[str, Any], normalized: str, term: str) -> dict[str, Any]:
    start = normalized.index(term)
    return {
        "source_id": source["source_id"],
        "normalized_char_start": start,
        "normalized_char_end": start + len(term),
        "match_strategy": "case_insensitive_literal",
    }


def _build_fixture(root: Path) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for case_index in range(25):
        case_id = f"case-{case_index:02d}"
        fact = f"FactCase{case_index:02d}"
        answer = f"AnswerCase{case_index:02d}"
        gap = f"GapCase{case_index:02d}"
        prior_claim = f"PriorCase{case_index:02d} obligation"
        current_claim = f"CurrentCase{case_index:02d} obligation"
        first_text = (
            f"{fact}. {answer}. {gap}. CommonPacketTerm. {prior_claim}."
        )
        first_source, first_normalized = _make_source(
            root,
            case_id=case_id,
            case_index=case_index,
            source_index=0,
            text=first_text,
        )
        sources = [first_source]
        normalized_by_source = {first_source["source_id"]: first_normalized}
        contradictions: list[dict[str, Any]] = []
        contradiction_terms: list[str] = []
        if case_index < 3:
            second_source, second_normalized = _make_source(
                root,
                case_id=case_id,
                case_index=case_index,
                source_index=1,
                text=f"CommonPacketTerm. {current_claim}.",
            )
            sources.append(second_source)
            second_source["supersedes"] = [first_source["name"]]
            normalized_by_source[second_source["source_id"]] = second_normalized
            proposition = f"Current clause supersedes prior case {case_index:02d}"
            contradiction_terms = [proposition]
            contradictions = [
                {
                    "term": proposition,
                    "classification": "supersession",
                    "sides": [
                        {
                            "side": "prior",
                            "source_name": first_source["name"],
                            "claim": prior_claim,
                            "support_span": _support(
                                first_source, first_normalized, prior_claim
                            ),
                        },
                        {
                            "side": "current",
                            "source_name": second_source["name"],
                            "claim": current_claim,
                            "support_span": _support(
                                second_source, second_normalized, current_claim
                            ),
                        },
                    ],
                }
            ]
        answerable_question = f"What is the answer token for case {case_index:02d}?"
        unanswerable_question = (
            f"What unrecorded final decision was made for case {case_index:02d}?"
        )
        case = {
            "id": case_id,
            "issuer": f"Issuer {case_index:02d}",
            "cik": first_source["cik"],
            "archetype": "contract_review",
            "as_of_date": "2026-01-01",
            "decision_question": f"Should the owner proceed with case {case_index:02d}?",
            "decision_owner": "operational contract owner",
            "operational_decision": f"Should the owner proceed with case {case_index:02d}?",
            "source_count": len(sources),
            "upload_bundle_bytes": sum(source["upload_bytes"] for source in sources),
            "planted_fact_terms": [fact],
            "gap_terms": [gap],
            "contradiction_terms": contradiction_terms,
            "answerable_question": answerable_question,
            "answer_terms": [answer],
            "unanswerable_question": unanswerable_question,
            "document_relationships": ["agreement -> amendment"],
            "annotations": {
                "facts": [
                    {
                        "term": fact,
                        "support_spans": [_support(first_source, first_normalized, fact)],
                    }
                ],
                "gaps": [
                    {
                        "term": gap,
                        "full_packet_search_terms": [gap],
                        "full_normalized_occurrence_count": sum(
                            normalized.casefold().count(gap.casefold())
                            for normalized in normalized_by_source.values()
                        ),
                        "interpretation": "topic_present_but_decision_evidence_incomplete",
                    }
                ],
                "contradictions": contradictions,
                "answerable_question": {
                    "question": answerable_question,
                    "answer_terms": [
                        {
                            "term": answer,
                            "support_spans": [
                                _support(first_source, first_normalized, answer)
                            ],
                        }
                    ],
                    "unique_support_source_id": first_source["source_id"],
                    "support_scope": "exactly_one_case_upload_source",
                },
                "unanswerable_question": {
                    "question": unanswerable_question,
                    "packet_search_scope": "all normalized full-text sources",
                    "exact_question_occurrence_count": 0,
                    "deterministic_status": "literal_question_absent",
                    "subjective_unanswerability_status": "HUMAN_REQUIRED",
                    "reason": "A human must assess substantive unanswerability.",
                },
            },
            "sources": sources,
        }
        cases.append(case)

    manifest = {
        "schema_version": CORPUS_SCHEMA,
        "corpus_id": CORPUS_ID,
        "language": "en",
        "target_cohort": "operational decision owners",
        "domain": "SEC EDGAR commercial contracts and issuer disclosures",
        "provenance_policy": EXPECTED_PROVENANCE_POLICY,
        "retrieval_policy": {
            "official_hosts": ["sec.gov", "www.sec.gov"],
            "maximum_requests_per_second": 5.0,
            "identifying_user_agent_required": True,
            "user_agent_not_persisted": True,
        },
        "intake_limits": {
            "source_count_min": 1,
            "source_count_max": 5,
            "max_source_bytes": 128 * 1024,
            "max_case_bytes": 512 * 1024,
        },
        "case_count": len(cases),
        "source_count": sum(case["source_count"] for case in cases),
        "human_review_case_ids": [case["id"] for case in cases[:10]],
        "cases": cases,
    }
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")
    _write(root, MANIFEST_PATH, manifest_bytes)
    freeze = {
        "schema_version": FREEZE_SCHEMA,
        "corpus_id": CORPUS_ID,
        "manifest_path": MANIFEST_PATH,
        "manifest_sha256": _sha256(manifest_bytes),
        "case_count": manifest["case_count"],
        "source_count": manifest["source_count"],
    }
    _write(
        root,
        str(CORPUS_DIRECTORY / "freeze.json"),
        (json.dumps(freeze, indent=2) + "\n").encode("utf-8"),
    )
    return manifest


def _rewrite_manifest(root: Path, manifest: dict[str, Any]) -> None:
    _write(
        root,
        MANIFEST_PATH,
        (json.dumps(manifest, indent=2) + "\n").encode("utf-8"),
    )


class Vs5FormalCorpusTrustTest(unittest.TestCase):
    def _assert_mutation_rejected(
        self,
        mutation: Callable[[Path, dict[str, Any]], None],
        message: str,
    ) -> None:
        with tempfile.TemporaryDirectory(prefix="cornerstone-vs5-corpus-") as directory:
            root = Path(directory)
            manifest = _build_fixture(root)
            mutation(root, manifest)
            _rewrite_manifest(root, manifest)
            with self.assertRaisesRegex(Vs5CorpusIntegrityError, message):
                load_vs5_corpus(root, MANIFEST_PATH)

    def test_minimal_formal_corpus_fixture_passes_all_loader_invariants(self) -> None:
        with tempfile.TemporaryDirectory(prefix="cornerstone-vs5-corpus-") as directory:
            root = Path(directory)
            _build_fixture(root)
            corpus, binding = load_vs5_corpus(root, MANIFEST_PATH)
        self.assertEqual(corpus["case_count"], 25)
        self.assertEqual(binding["source_count"], 28)

    def test_rejects_stale_provenance_and_intake_metadata(self) -> None:
        mutations = (
            (lambda _root, manifest: manifest.__setitem__("language", "ko"), "language"),
            (
                lambda _root, manifest: manifest["intake_limits"].__setitem__(
                    "max_source_bytes", 999_999
                ),
                "intake_limits",
            ),
            (
                lambda _root, manifest: manifest.__setitem__("source_count", 27),
                "source_count",
            ),
        )
        for mutation, message in mutations:
            with self.subTest(message=message):
                self._assert_mutation_rejected(mutation, message)

    def test_rejects_fact_and_answer_annotation_tampering(self) -> None:
        def stale_fact(_root: Path, manifest: dict[str, Any]) -> None:
            manifest["cases"][0]["annotations"]["facts"][0]["term"] = "OtherFact"

        self._assert_mutation_rejected(stale_fact, "facts.*declared terms")

        def unknown_fact_source(_root: Path, manifest: dict[str, Any]) -> None:
            support = manifest["cases"][0]["annotations"]["facts"][0]["support_spans"][0]
            support["source_id"] = "unknown-source"

        self._assert_mutation_rejected(unknown_fact_source, "unknown source_id")

        def out_of_bounds_answer_span(_root: Path, manifest: dict[str, Any]) -> None:
            support = manifest["cases"][0]["annotations"]["answerable_question"][
                "answer_terms"
            ][0]["support_spans"][0]
            support["normalized_char_end"] = 1_000_000

        self._assert_mutation_rejected(
            out_of_bounds_answer_span, "outside the normalized source"
        )

        def answer_in_two_uploads(root: Path, manifest: dict[str, Any]) -> None:
            case = manifest["cases"][0]
            source = case["sources"][0]
            normalized = (root / source["normalized_path"]).read_text(encoding="utf-8")
            term = "CommonPacketTerm"
            case["answer_terms"] = [term]
            answer_annotation = case["annotations"]["answerable_question"]
            answer_annotation["answer_terms"] = [
                {"term": term, "support_spans": [_support(source, normalized, term)]}
            ]

        self._assert_mutation_rejected(answer_in_two_uploads, "every and only upload source")

    def test_rejects_gap_and_unanswerable_annotation_tampering(self) -> None:
        def stale_gap_count(_root: Path, manifest: dict[str, Any]) -> None:
            manifest["cases"][0]["annotations"]["gaps"][0][
                "full_normalized_occurrence_count"
            ] = 0

        self._assert_mutation_rejected(stale_gap_count, "occurrence count is stale")

        def stale_gap_interpretation(_root: Path, manifest: dict[str, Any]) -> None:
            manifest["cases"][0]["annotations"]["gaps"][0][
                "interpretation"
            ] = "literal_absence"

        self._assert_mutation_rejected(
            stale_gap_interpretation, "interpretation is inconsistent"
        )

        def answerable_unanswerable_question(_root: Path, manifest: dict[str, Any]) -> None:
            case = manifest["cases"][0]
            case["unanswerable_question"] = "CommonPacketTerm"
            case["annotations"]["unanswerable_question"]["question"] = "CommonPacketTerm"

        self._assert_mutation_rejected(
            answerable_unanswerable_question, "occurs literally in the source packet"
        )

    def test_rejects_one_sided_or_inexact_contradiction_evidence(self) -> None:
        def same_source(_root: Path, manifest: dict[str, Any]) -> None:
            sides = manifest["cases"][0]["annotations"]["contradictions"][0]["sides"]
            sides[1] = dict(sides[0])
            sides[1]["side"] = "current"

        self._assert_mutation_rejected(same_source, "distinct sources and claims")

        def case_changed_claim(_root: Path, manifest: dict[str, Any]) -> None:
            side = manifest["cases"][0]["annotations"]["contradictions"][0]["sides"][0]
            side["claim"] = str(side["claim"]).upper()

        self._assert_mutation_rejected(case_changed_claim, "does not exactly support")

        def missing_supersession_metadata(
            _root: Path, manifest: dict[str, Any]
        ) -> None:
            manifest["cases"][0]["sources"][1]["supersedes"] = []

        self._assert_mutation_rejected(
            missing_supersession_metadata,
            "supersession annotation is not declared",
        )

    def test_rejects_invalid_document_relationship_metadata(self) -> None:
        def unknown_relationship(_root: Path, manifest: dict[str, Any]) -> None:
            manifest["cases"][3]["sources"][0]["supersedes"] = ["unknown-source"]

        self._assert_mutation_rejected(unknown_relationship, "unknown source")

        def forward_relationship(_root: Path, manifest: dict[str, Any]) -> None:
            manifest["cases"][0]["sources"][0]["incorporated_by_reference"] = [
                manifest["cases"][0]["sources"][1]["name"]
            ]

        self._assert_mutation_rejected(forward_relationship, "an earlier source")

    def test_rejects_extra_files_directories_and_symlinks(self) -> None:
        def extra_file(root: Path, _manifest: dict[str, Any]) -> None:
            _write(root, str(CORPUS_DIRECTORY / "stale.txt"), b"stale")

        self._assert_mutation_rejected(extra_file, "file inventory is not exact")

        def extra_directory(root: Path, _manifest: dict[str, Any]) -> None:
            (root / CORPUS_DIRECTORY / "stale-directory").mkdir()

        self._assert_mutation_rejected(extra_directory, "directory inventory is not exact")

        with tempfile.TemporaryDirectory(prefix="cornerstone-vs5-corpus-") as directory:
            root = Path(directory)
            _build_fixture(root)
            (root / CORPUS_DIRECTORY / "linked-source").symlink_to(
                root / CORPUS_DIRECTORY / "manifest.json"
            )
            with self.assertRaisesRegex(Vs5CorpusIntegrityError, "must not contain symlinks"):
                load_vs5_corpus(root, MANIFEST_PATH)


if __name__ == "__main__":
    unittest.main()
