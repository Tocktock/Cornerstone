from __future__ import annotations

import hashlib
import html
import json
import os
import re
from copy import deepcopy
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
CORPUS_SCHEMA = "cs.vs5_edgar_eval_manifest.v1"
FREEZE_SCHEMA = "cs.vs5_edgar_eval_freeze.v1"
CORPUS_ID = "vs5-sec-edgar-commercial-contracts-2026-07-17"
MAX_SOURCE_BYTES = 128 * 1024
MAX_CASE_BYTES = 512 * 1024
EXPECTED_PROVENANCE_POLICY = (
    "Source payloads are verbatim substrings of deterministic full-text "
    "normalizations of official SEC filing HTML; no source text is paraphrased."
)


class Vs5CorpusIntegrityError(ValueError):
    """Raised when the formal evaluation corpus cannot prove local provenance."""


class _EdgarVisibleTextParser(HTMLParser):
    """Deterministically retain visible filing text without source rewriting."""

    _SKIP_TAGS = {"script", "style", "noscript", "template"}
    _BLOCK_TAGS = {
        "address", "article", "aside", "blockquote", "br", "caption", "dd",
        "div", "dl", "dt", "figcaption", "figure", "footer", "form", "h1",
        "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "main", "nav",
        "ol", "p", "pre", "section", "table", "tbody", "td", "tfoot", "th",
        "thead", "tr", "ul",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        if lowered in self._SKIP_TAGS:
            self._skip_depth += 1
        elif not self._skip_depth and lowered in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if not self._skip_depth and tag.lower() in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in self._SKIP_TAGS:
            if self._skip_depth:
                self._skip_depth -= 1
        elif not self._skip_depth and lowered in self._BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def normalize_edgar_filing_html(raw: bytes) -> str:
    """Return the canonical visible-text rendering derived solely from filing bytes."""

    decoded = raw.decode("utf-8", errors="replace")
    parser = _EdgarVisibleTextParser()
    parser.feed(decoded)
    parser.close()
    value = html.unescape("".join(parser.parts)).replace("\xa0", " ")
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    lines = []
    for line in value.split("\n"):
        compact = re.sub(r"[\t\f\v ]+", " ", line).strip()
        lines.append(compact)
    value = "\n".join(lines)
    value = re.sub(r"\n{3,}", "\n\n", value).strip()
    return value + "\n"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _required_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Vs5CorpusIntegrityError(f"{field} must be a nonempty string")
    return value


def _required_term_list(value: Any, *, field: str, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "a list" if allow_empty else "a nonempty list"
        raise Vs5CorpusIntegrityError(f"{field} must be {qualifier} of strings")
    if any(not isinstance(term, str) or not term.strip() for term in value):
        raise Vs5CorpusIntegrityError(f"{field} must contain only nonempty strings")
    if len({term.casefold() for term in value}) != len(value):
        raise Vs5CorpusIntegrityError(f"{field} must not contain duplicate terms")
    return list(value)


def _required_int(value: Any, *, field: str, minimum: int = 0) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < minimum
    ):
        raise Vs5CorpusIntegrityError(f"{field} must be an integer >= {minimum}")
    return value


def _validate_manifest_metadata(corpus: dict[str, Any]) -> None:
    if corpus.get("corpus_id") != CORPUS_ID:
        raise Vs5CorpusIntegrityError(f"corpus_id must be {CORPUS_ID}")
    expected_strings = {
        "language": "en",
        "target_cohort": "operational decision owners",
        "domain": "SEC EDGAR commercial contracts and issuer disclosures",
        "provenance_policy": EXPECTED_PROVENANCE_POLICY,
    }
    for field, expected in expected_strings.items():
        if corpus.get(field) != expected:
            raise Vs5CorpusIntegrityError(
                f"formal corpus {field} does not match the frozen provenance contract"
            )

    retrieval_policy = corpus.get("retrieval_policy")
    expected_retrieval_policy = {
        "official_hosts": ["sec.gov", "www.sec.gov"],
        "maximum_requests_per_second": 5.0,
        "identifying_user_agent_required": True,
        "user_agent_not_persisted": True,
    }
    if retrieval_policy != expected_retrieval_policy:
        raise Vs5CorpusIntegrityError(
            "formal corpus retrieval_policy does not match the frozen SEC policy"
        )

    intake_limits = corpus.get("intake_limits")
    expected_intake_limits = {
        "source_count_min": 1,
        "source_count_max": 5,
        "max_source_bytes": MAX_SOURCE_BYTES,
        "max_case_bytes": MAX_CASE_BYTES,
    }
    if intake_limits != expected_intake_limits:
        raise Vs5CorpusIntegrityError(
            "formal corpus intake_limits do not match the frozen limits"
        )


def _checked_repo_file(root: Path, relative_path: Any, *, field: str) -> Path:
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise Vs5CorpusIntegrityError(f"{field} is missing or invalid")
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise Vs5CorpusIntegrityError(f"{field} must be a repository-relative path")

    root = root.resolve()
    candidate = root / relative
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise Vs5CorpusIntegrityError(f"{field} must not traverse a symlink")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as error:
        raise Vs5CorpusIntegrityError(f"{field} does not exist: {relative_path}") from error
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise Vs5CorpusIntegrityError(f"{field} is not a regular file under the repository")
    return resolved


def _checked_hash(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or SHA256_PATTERN.fullmatch(value) is None:
        raise Vs5CorpusIntegrityError(f"{field} must be a lowercase SHA-256")
    return value


def _validate_sec_source_metadata(source: dict[str, Any]) -> None:
    accession = str(source.get("accession_number") or "")
    if re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession) is None:
        raise Vs5CorpusIntegrityError("accession_number is not a canonical SEC accession")
    cik = str(source.get("cik") or "")
    if not cik.isdigit() or not cik.strip("0"):
        raise Vs5CorpusIntegrityError("source cik must be a nonzero decimal identifier")
    canonical_cik = str(int(cik))
    accession_compact = accession.replace("-", "")
    expected_path_prefix = f"/Archives/edgar/data/{canonical_cik}/{accession_compact}/"

    for field in ("source_url", "final_url", "filing_index_url"):
        value = str(source.get(field) or "")
        parsed = urlparse(value)
        if (
            parsed.scheme != "https"
            or parsed.hostname not in {"sec.gov", "www.sec.gov"}
            or not parsed.path.startswith(expected_path_prefix)
            or parsed.query
            or parsed.fragment
        ):
            raise Vs5CorpusIntegrityError(
                f"{field} is not the expected canonical SEC filing URL"
            )
    if source.get("final_url") != source.get("source_url"):
        raise Vs5CorpusIntegrityError("final_url must equal the verified source_url")
    if not str(source["filing_index_url"]).endswith(f"/{accession}-index.html"):
        raise Vs5CorpusIntegrityError("filing_index_url does not match accession_number")
    if source.get("http_status") != 200 or source.get("filing_index_http_status") != 200:
        raise Vs5CorpusIntegrityError("SEC source and filing-index status must both be 200")
    if source.get("content_type") != "text/html":
        raise Vs5CorpusIntegrityError("SEC source content_type must be text/html")
    if source.get("encoding") != "utf-8":
        raise Vs5CorpusIntegrityError("normalized SEC source encoding must be utf-8")


def _load_bound_file(
    root: Path,
    source: dict[str, Any],
    *,
    path_field: str,
    hash_field: str,
    size_field: str,
) -> tuple[Path, bytes]:
    path = _checked_repo_file(root, source.get(path_field), field=path_field)
    content = path.read_bytes()
    expected_hash = _checked_hash(source.get(hash_field), field=hash_field)
    if _sha256_bytes(content) != expected_hash:
        raise Vs5CorpusIntegrityError(
            f"{hash_field} does not match {source.get(path_field)}"
        )
    expected_size = source.get(size_field)
    if (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size < 0
        or expected_size != len(content)
    ):
        raise Vs5CorpusIntegrityError(
            f"{size_field} does not match {source.get(path_field)}"
        )
    return path, content


def load_vs5_corpus_source(root: Path, source: dict[str, Any]) -> dict[str, Any]:
    """Load one verbatim upload and prove its raw/normalized/span bindings."""

    if not isinstance(source, dict):
        raise Vs5CorpusIntegrityError("corpus source must be an object")
    if "text" in source:
        raise Vs5CorpusIntegrityError(
            "formal corpus sources must use hash-bound upload_path, not inline text"
        )
    _validate_sec_source_metadata(source)

    _, raw_bytes = _load_bound_file(
        root,
        source,
        path_field="raw_path",
        hash_field="raw_sha256",
        size_field="raw_bytes",
    )
    _, normalized_bytes = _load_bound_file(
        root,
        source,
        path_field="normalized_path",
        hash_field="normalized_sha256",
        size_field="normalized_bytes",
    )
    _, upload_bytes = _load_bound_file(
        root,
        source,
        path_field="upload_path",
        hash_field="upload_sha256",
        size_field="upload_bytes",
    )
    try:
        normalized_text = normalized_bytes.decode("utf-8")
        upload_text = upload_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise Vs5CorpusIntegrityError(
            "normalized and upload corpus files must be UTF-8"
        ) from error
    if source.get("normalizer") != "cornerstone_edgar_visible_text_v1":
        raise Vs5CorpusIntegrityError("source normalizer is not the frozen EDGAR normalizer")
    if normalize_edgar_filing_html(raw_bytes) != normalized_text:
        raise Vs5CorpusIntegrityError(
            "normalized text is not the canonical visible-text rendering of raw bytes"
        )

    span = source.get("upload_span_in_normalized")
    if not isinstance(span, dict):
        raise Vs5CorpusIntegrityError("upload_span_in_normalized is missing")
    if span.get("coordinate_system") != "unicode_code_points":
        raise Vs5CorpusIntegrityError(
            "upload span coordinate_system must be unicode_code_points"
        )
    start = span.get("char_start")
    end = span.get("char_end")
    if (
        not isinstance(start, int)
        or isinstance(start, bool)
        or not isinstance(end, int)
        or isinstance(end, bool)
        or start < 0
        or end < start
        or end > len(normalized_text)
    ):
        raise Vs5CorpusIntegrityError("upload span is outside normalized text")
    if normalized_text[start:end] != upload_text:
        raise Vs5CorpusIntegrityError(
            "upload text is not the declared normalized-text span"
        )
    span_sha256 = _checked_hash(span.get("sha256"), field="upload span sha256")
    if span_sha256 != _sha256_bytes(upload_bytes):
        raise Vs5CorpusIntegrityError("upload span sha256 does not match upload bytes")

    loaded = deepcopy(source)
    loaded["text"] = upload_text
    loaded["source_ref"] = str(source["upload_path"])
    loaded["integrity"] = {
        "status": "passed",
        "raw_sha256": _sha256_bytes(raw_bytes),
        "normalized_sha256": _sha256_bytes(normalized_bytes),
        "upload_sha256": _sha256_bytes(upload_bytes),
        "normalized_span_verified": True,
    }
    return loaded


def _literal_support_source_ids(
    term: str,
    source_by_id: dict[str, dict[str, Any]],
) -> set[str]:
    folded = term.casefold()
    return {
        source_id
        for source_id, record in source_by_id.items()
        if folded in str(record["upload"]).casefold()
    }


def _validate_support_span(
    *,
    case_id: str,
    term: str,
    support: Any,
    source_by_id: dict[str, dict[str, Any]],
    field: str,
    exact_case: bool = False,
) -> str:
    if not isinstance(support, dict):
        raise Vs5CorpusIntegrityError(f"case {case_id} {field} must be an object")
    if support.get("match_strategy") != "case_insensitive_literal":
        raise Vs5CorpusIntegrityError(
            f"case {case_id} {field} must use case_insensitive_literal"
        )
    source_id = _required_text(support.get("source_id"), field=f"{field}.source_id")
    record = source_by_id.get(source_id)
    if record is None:
        raise Vs5CorpusIntegrityError(
            f"case {case_id} {field} references an unknown source_id"
        )
    start = _required_int(
        support.get("normalized_char_start"),
        field=f"{field}.normalized_char_start",
    )
    end = _required_int(
        support.get("normalized_char_end"),
        field=f"{field}.normalized_char_end",
    )
    normalized = str(record["normalized"])
    if end <= start or end > len(normalized):
        raise Vs5CorpusIntegrityError(
            f"case {case_id} {field} is outside the normalized source"
        )
    supported_text = normalized[start:end]
    text_matches = (
        supported_text == term
        if exact_case
        else supported_text.casefold() == term.casefold()
    )
    if not text_matches:
        raise Vs5CorpusIntegrityError(
            f"case {case_id} {field} does not exactly support {term!r}"
        )

    upload_span = record["source"].get("upload_span_in_normalized")
    upload_start = int(upload_span["char_start"])
    upload_end = int(upload_span["char_end"])
    if start < upload_start or end > upload_end:
        raise Vs5CorpusIntegrityError(
            f"case {case_id} {field} is not inside the bound upload span"
        )
    relative_start = start - upload_start
    relative_end = end - upload_start
    upload_text = str(record["upload"])[relative_start:relative_end]
    upload_matches = (
        upload_text == term
        if exact_case
        else upload_text.casefold() == term.casefold()
    )
    if not upload_matches:
        raise Vs5CorpusIntegrityError(
            f"case {case_id} {field} does not exactly match the upload text"
        )
    return source_id


def _validate_supported_term_annotations(
    *,
    case_id: str,
    field: str,
    annotations: Any,
    expected_terms: list[str],
    source_by_id: dict[str, dict[str, Any]],
) -> list[set[str]]:
    if not isinstance(annotations, list):
        raise Vs5CorpusIntegrityError(f"case {case_id} annotations.{field} must be a list")
    annotated_terms = [
        row.get("term") if isinstance(row, dict) else None
        for row in annotations
    ]
    if annotated_terms != expected_terms:
        raise Vs5CorpusIntegrityError(
            f"case {case_id} annotations.{field} terms do not equal the declared terms"
        )

    support_sets: list[set[str]] = []
    for index, (term, annotation) in enumerate(zip(expected_terms, annotations)):
        supports = annotation.get("support_spans")
        if not isinstance(supports, list) or not supports:
            raise Vs5CorpusIntegrityError(
                f"case {case_id} annotations.{field}[{index}] needs support_spans"
            )
        declared_ids = {
            _validate_support_span(
                case_id=case_id,
                term=term,
                support=support,
                source_by_id=source_by_id,
                field=f"annotations.{field}[{index}].support_spans",
            )
            for support in supports
        }
        if len(declared_ids) != len(supports):
            raise Vs5CorpusIntegrityError(
                f"case {case_id} annotations.{field}[{index}] repeats a support source"
            )
        actual_ids = _literal_support_source_ids(term, source_by_id)
        if declared_ids != actual_ids:
            raise Vs5CorpusIntegrityError(
                f"case {case_id} annotations.{field}[{index}] does not bind "
                f"every and only upload source supporting {term!r}"
            )
        support_sets.append(declared_ids)
    return support_sets


def _validate_gap_annotations(
    *,
    case_id: str,
    annotations: Any,
    expected_terms: list[str],
    source_by_id: dict[str, dict[str, Any]],
) -> None:
    if not isinstance(annotations, list):
        raise Vs5CorpusIntegrityError(f"case {case_id} annotations.gaps must be a list")
    annotated_terms = [
        row.get("term") if isinstance(row, dict) else None
        for row in annotations
    ]
    if annotated_terms != expected_terms:
        raise Vs5CorpusIntegrityError(
            f"case {case_id} annotations.gaps terms do not equal gap_terms"
        )
    for index, (term, annotation) in enumerate(zip(expected_terms, annotations)):
        if annotation.get("full_packet_search_terms") != [term]:
            raise Vs5CorpusIntegrityError(
                f"case {case_id} annotations.gaps[{index}] search terms are stale"
            )
        occurrence_count = sum(
            str(record["normalized"]).casefold().count(term.casefold())
            for record in source_by_id.values()
        )
        declared_count = _required_int(
            annotation.get("full_normalized_occurrence_count"),
            field=f"case {case_id} annotations.gaps[{index}].full_normalized_occurrence_count",
        )
        if declared_count != occurrence_count:
            raise Vs5CorpusIntegrityError(
                f"case {case_id} annotations.gaps[{index}] occurrence count is stale"
            )
        expected_interpretation = (
            "literal_absence"
            if occurrence_count == 0
            else "topic_present_but_decision_evidence_incomplete"
        )
        if annotation.get("interpretation") != expected_interpretation:
            raise Vs5CorpusIntegrityError(
                f"case {case_id} annotations.gaps[{index}] interpretation is inconsistent"
            )


def _validate_contradiction_annotations(
    *,
    case_id: str,
    annotations: Any,
    expected_terms: list[str],
    source_by_id: dict[str, dict[str, Any]],
) -> None:
    if not isinstance(annotations, list):
        raise Vs5CorpusIntegrityError(
            f"case {case_id} annotations.contradictions must be a list"
        )
    annotated_terms = [
        row.get("term") if isinstance(row, dict) else None
        for row in annotations
    ]
    if annotated_terms != expected_terms:
        raise Vs5CorpusIntegrityError(
            f"case {case_id} contradiction annotations do not equal contradiction_terms"
        )
    source_id_by_name: dict[str, str] = {}
    for source_id, record in source_by_id.items():
        source_name = _required_text(
            record["source"].get("name"),
            field=f"case {case_id} source.name",
        )
        if source_name in source_id_by_name:
            raise Vs5CorpusIntegrityError(f"case {case_id} source names must be unique")
        source_id_by_name[source_name] = source_id

    for index, contradiction in enumerate(annotations):
        if contradiction.get("classification") not in {
            "contradiction",
            "scope_difference",
            "supersession",
        }:
            raise Vs5CorpusIntegrityError(
                f"case {case_id} contradiction {index} classification is unsupported"
            )
        sides = contradiction.get("sides")
        if not isinstance(sides, list) or len(sides) != 2:
            raise Vs5CorpusIntegrityError(
                f"case {case_id} contradiction {index} must have exactly two sides"
            )
        if [side.get("side") if isinstance(side, dict) else None for side in sides] != [
            "prior",
            "current",
        ]:
            raise Vs5CorpusIntegrityError(
                f"case {case_id} contradiction {index} sides must be prior then current"
            )
        side_source_ids: set[str] = set()
        side_claims: set[str] = set()
        for side_index, side in enumerate(sides):
            source_name = _required_text(
                side.get("source_name"),
                field=f"case {case_id} contradiction side source_name",
            )
            expected_source_id = source_id_by_name.get(source_name)
            if expected_source_id is None:
                raise Vs5CorpusIntegrityError(
                    f"case {case_id} contradiction side names an unknown source"
                )
            claim = _required_text(
                side.get("claim"),
                field=f"case {case_id} contradiction side claim",
            )
            support_source_id = _validate_support_span(
                case_id=case_id,
                term=claim,
                support=side.get("support_span"),
                source_by_id=source_by_id,
                field=f"annotations.contradictions[{index}].sides[{side_index}].support_span",
                exact_case=True,
            )
            if support_source_id != expected_source_id:
                raise Vs5CorpusIntegrityError(
                    f"case {case_id} contradiction side source_name and source_id disagree"
                )
            side_source_ids.add(support_source_id)
            side_claims.add(claim.casefold())
        if len(side_source_ids) != 2 or len(side_claims) != 2:
            raise Vs5CorpusIntegrityError(
                f"case {case_id} contradiction sides need distinct sources and claims"
            )
        if contradiction.get("classification") == "supersession":
            prior_source_name = str(sides[0]["source_name"])
            current_source_name = str(sides[1]["source_name"])
            current_source_id = source_id_by_name[current_source_name]
            current_source = source_by_id[current_source_id]["source"]
            if prior_source_name not in current_source["supersedes"]:
                raise Vs5CorpusIntegrityError(
                    f"case {case_id} supersession annotation is not declared by "
                    "the current source metadata"
                )


def _validate_document_relationships(
    *,
    case_id: str,
    source_by_id: dict[str, dict[str, Any]],
) -> None:
    source_id_by_name: dict[str, str] = {}
    source_order_by_name: dict[str, int] = {}
    for source_id, record in source_by_id.items():
        source = record["source"]
        source_name = _required_text(
            source.get("name"),
            field=f"case {case_id} source.name",
        )
        if source_name in source_id_by_name:
            raise Vs5CorpusIntegrityError(f"case {case_id} source names must be unique")
        source_order = _required_int(
            source.get("source_order"),
            field=f"case {case_id} source.source_order",
            minimum=1,
        )
        if source_order in source_order_by_name.values():
            raise Vs5CorpusIntegrityError(
                f"case {case_id} source_order values must be unique"
            )
        source_id_by_name[source_name] = source_id
        source_order_by_name[source_name] = source_order

    for source_name, source_id in source_id_by_name.items():
        source = source_by_id[source_id]["source"]
        for field in ("supersedes", "incorporated_by_reference"):
            relationships = source.get(field)
            if not isinstance(relationships, list) or any(
                not isinstance(target, str) or not target.strip()
                for target in relationships
            ):
                raise Vs5CorpusIntegrityError(
                    f"case {case_id} source {source_name} {field} must be a list "
                    "of nonempty source names"
                )
            if len(relationships) != len(set(relationships)):
                raise Vs5CorpusIntegrityError(
                    f"case {case_id} source {source_name} {field} repeats a source"
                )
            for target in relationships:
                if target == source_name:
                    raise Vs5CorpusIntegrityError(
                        f"case {case_id} source {source_name} {field} references itself"
                    )
                if target not in source_id_by_name:
                    raise Vs5CorpusIntegrityError(
                        f"case {case_id} source {source_name} {field} references "
                        "an unknown source"
                    )
                if source_order_by_name[target] >= source_order_by_name[source_name]:
                    raise Vs5CorpusIntegrityError(
                        f"case {case_id} source {source_name} {field} must reference "
                        "an earlier source"
                    )


def _validate_case_annotations(
    *,
    case: dict[str, Any],
    source_by_id: dict[str, dict[str, Any]],
) -> None:
    case_id = str(case["id"])
    annotations = case.get("annotations")
    if not isinstance(annotations, dict):
        raise Vs5CorpusIntegrityError(f"case {case_id} annotations must be an object")
    expected_annotation_fields = {
        "facts",
        "gaps",
        "contradictions",
        "answerable_question",
        "unanswerable_question",
    }
    if set(annotations) != expected_annotation_fields:
        raise Vs5CorpusIntegrityError(
            f"case {case_id} annotation fields do not match the v1 schema"
        )

    _validate_document_relationships(case_id=case_id, source_by_id=source_by_id)

    fact_terms = _required_term_list(
        case.get("planted_fact_terms"),
        field=f"case {case_id} planted_fact_terms",
    )
    gap_terms = _required_term_list(
        case.get("gap_terms"),
        field=f"case {case_id} gap_terms",
    )
    contradiction_terms = _required_term_list(
        case.get("contradiction_terms"),
        field=f"case {case_id} contradiction_terms",
        allow_empty=True,
    )
    answer_terms = _required_term_list(
        case.get("answer_terms"),
        field=f"case {case_id} answer_terms",
    )
    _validate_supported_term_annotations(
        case_id=case_id,
        field="facts",
        annotations=annotations["facts"],
        expected_terms=fact_terms,
        source_by_id=source_by_id,
    )
    _validate_gap_annotations(
        case_id=case_id,
        annotations=annotations["gaps"],
        expected_terms=gap_terms,
        source_by_id=source_by_id,
    )
    _validate_contradiction_annotations(
        case_id=case_id,
        annotations=annotations["contradictions"],
        expected_terms=contradiction_terms,
        source_by_id=source_by_id,
    )

    answer_annotation = annotations["answerable_question"]
    if not isinstance(answer_annotation, dict):
        raise Vs5CorpusIntegrityError(
            f"case {case_id} answerable_question annotation must be an object"
        )
    if answer_annotation.get("question") != case.get("answerable_question"):
        raise Vs5CorpusIntegrityError(
            f"case {case_id} answerable question annotation is stale"
        )
    answer_support_sets = _validate_supported_term_annotations(
        case_id=case_id,
        field="answerable_question.answer_terms",
        annotations=answer_annotation.get("answer_terms"),
        expected_terms=answer_terms,
        source_by_id=source_by_id,
    )
    answer_source_ids = set().union(*answer_support_sets)
    if len(answer_source_ids) != 1:
        raise Vs5CorpusIntegrityError(
            f"case {case_id} answer terms must resolve to exactly one upload source"
        )
    unique_answer_source_id = next(iter(answer_source_ids))
    if (
        answer_annotation.get("unique_support_source_id") != unique_answer_source_id
        or answer_annotation.get("support_scope") != "exactly_one_case_upload_source"
    ):
        raise Vs5CorpusIntegrityError(
            f"case {case_id} answer support-source annotation is stale"
        )

    unanswerable_annotation = annotations["unanswerable_question"]
    if not isinstance(unanswerable_annotation, dict):
        raise Vs5CorpusIntegrityError(
            f"case {case_id} unanswerable_question annotation must be an object"
        )
    unanswerable_question = str(case["unanswerable_question"])
    exact_count = sum(
        str(record["normalized"]).casefold().count(unanswerable_question.casefold())
        for record in source_by_id.values()
    )
    if exact_count:
        raise Vs5CorpusIntegrityError(
            f"case {case_id} unanswerable question occurs literally in the source packet"
        )
    declared_exact_count = _required_int(
        unanswerable_annotation.get("exact_question_occurrence_count"),
        field=f"case {case_id} unanswerable exact_question_occurrence_count",
    )
    if (
        unanswerable_annotation.get("question") != unanswerable_question
        or unanswerable_annotation.get("packet_search_scope")
        != "all normalized full-text sources"
        or declared_exact_count != exact_count
        or unanswerable_annotation.get("deterministic_status") != "literal_question_absent"
        or unanswerable_annotation.get("subjective_unanswerability_status")
        != "HUMAN_REQUIRED"
        or not str(unanswerable_annotation.get("reason") or "").strip()
    ):
        raise Vs5CorpusIntegrityError(
            f"case {case_id} unanswerable-question annotation is stale or incomplete"
        )


def _validate_exact_corpus_inventory(
    *,
    root: Path,
    manifest_path: Path,
    manifest_relative_path: str,
    sources: list[dict[str, Any]],
) -> None:
    manifest_relative = Path(manifest_relative_path)
    freeze_relative = manifest_relative.parent / "freeze.json"
    freeze_path = _checked_repo_file(root, str(freeze_relative), field="corpus freeze path")
    corpus_root = manifest_path.parent.resolve()
    expected_files = {manifest_path.resolve(), freeze_path.resolve()}
    for source in sources:
        for kind in ("raw", "normalized", "upload"):
            bound_path = _checked_repo_file(
                root,
                source.get(f"{kind}_path"),
                field=f"{kind}_path",
            )
            if not bound_path.is_relative_to(corpus_root):
                raise Vs5CorpusIntegrityError(
                    f"{kind}_path must remain inside the formal corpus directory"
                )
            expected_files.add(bound_path)

    actual_files: set[Path] = set()
    actual_directories: set[Path] = set()
    for directory, directory_names, file_names in os.walk(corpus_root, followlinks=False):
        directory_path = Path(directory)
        actual_directories.add(directory_path.resolve())
        for name in [*directory_names, *file_names]:
            child = directory_path / name
            if child.is_symlink():
                raise Vs5CorpusIntegrityError(
                    f"formal corpus inventory must not contain symlinks: {child}"
                )
        for name in file_names:
            child = directory_path / name
            if not child.is_file():
                raise Vs5CorpusIntegrityError(
                    f"formal corpus inventory contains a non-regular file: {child}"
                )
            actual_files.add(child.resolve())

    if actual_files != expected_files:
        extra = sorted(str(path.relative_to(corpus_root)) for path in actual_files - expected_files)
        missing = sorted(
            str(path.relative_to(corpus_root))
            for path in expected_files - actual_files
        )
        raise Vs5CorpusIntegrityError(
            f"formal corpus file inventory is not exact; extra={extra}, missing={missing}"
        )
    expected_directories = {corpus_root}
    for path in expected_files:
        cursor = path.parent
        while cursor != corpus_root:
            expected_directories.add(cursor)
            cursor = cursor.parent
        expected_directories.add(corpus_root)
    if actual_directories != expected_directories:
        extra = sorted(
            str(path.relative_to(corpus_root))
            for path in actual_directories - expected_directories
        )
        raise Vs5CorpusIntegrityError(
            f"formal corpus directory inventory is not exact; extra={extra}"
        )


def _source_file_binding(sources: list[dict[str, Any]]) -> dict[str, Any]:
    entries_by_path: dict[str, dict[str, Any]] = {}
    for source in sources:
        for kind in ("raw", "normalized", "upload"):
            path = str(source[f"{kind}_path"])
            entry = {
                "path": path,
                "sha256": str(source[f"{kind}_sha256"]),
                "size_bytes": int(source[f"{kind}_bytes"]),
                "kind": kind,
            }
            existing = entries_by_path.get(path)
            if existing is not None and existing != entry:
                raise Vs5CorpusIntegrityError(
                    f"one corpus path has conflicting bindings: {path}"
                )
            entries_by_path[path] = entry
    entries = [entries_by_path[path] for path in sorted(entries_by_path)]
    digest_payload = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return {
        "schema_version": "cs.vs5_corpus_source_binding.v1",
        "entry_count": len(entries),
        "file_count": len(entries),
        "total_file_bytes": sum(entry["size_bytes"] for entry in entries),
        "manifest_sha256": _sha256_bytes(digest_payload),
        "entries": entries,
    }


def load_vs5_corpus(
    root: Path,
    manifest_relative_path: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and validate the complete real-source VS5 formal corpus."""

    manifest_path = _checked_repo_file(
        root,
        manifest_relative_path,
        field="corpus manifest path",
    )
    try:
        corpus = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Vs5CorpusIntegrityError("corpus manifest is not valid UTF-8 JSON") from error
    if not isinstance(corpus, dict) or corpus.get("schema_version") != CORPUS_SCHEMA:
        raise Vs5CorpusIntegrityError(f"corpus schema must be {CORPUS_SCHEMA}")
    _validate_manifest_metadata(corpus)
    cases = corpus.get("cases")
    if not isinstance(cases, list) or len(cases) != 25:
        raise Vs5CorpusIntegrityError("formal corpus must contain exactly 25 cases")
    if _required_int(corpus.get("case_count"), field="case_count") != len(cases):
        raise Vs5CorpusIntegrityError("case_count does not match cases")

    loaded = deepcopy(corpus)
    loaded_cases: list[dict[str, Any]] = []
    all_loaded_sources: list[dict[str, Any]] = []
    case_ids: set[str] = set()
    source_ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise Vs5CorpusIntegrityError("each corpus case must be an object")
        case_id = _required_text(case.get("id"), field="case.id")
        if not case_id or case_id in case_ids:
            raise Vs5CorpusIntegrityError("corpus case IDs must be unique and nonempty")
        case_ids.add(case_id)
        sources = case.get("sources")
        if not isinstance(sources, list) or not 1 <= len(sources) <= 5:
            raise Vs5CorpusIntegrityError("each corpus case must contain one to five sources")
        required_text_fields = (
            "decision_question",
            "answerable_question",
            "unanswerable_question",
        )
        for field in required_text_fields:
            _required_text(case.get(field), field=f"case {case_id} {field}")
        if case.get("operational_decision") != case.get("decision_question"):
            raise Vs5CorpusIntegrityError(
                f"case {case_id} operational_decision must equal decision_question"
            )
        if (
            _required_int(
                case.get("source_count"),
                field=f"case {case_id} source_count",
                minimum=1,
            )
            != len(sources)
        ):
            raise Vs5CorpusIntegrityError(f"case {case_id} source_count is incorrect")

        loaded_sources: list[dict[str, Any]] = []
        source_by_id: dict[str, dict[str, Any]] = {}
        for source in sources:
            loaded_source = load_vs5_corpus_source(root, source)
            if loaded_source.get("case_id") != case_id:
                raise Vs5CorpusIntegrityError(
                    f"source {loaded_source.get('source_id')} has the wrong case_id"
                )
            source_id = _required_text(
                loaded_source.get("source_id"),
                field=f"case {case_id} source_id",
            )
            if not source_id or source_id in source_ids:
                raise Vs5CorpusIntegrityError(
                    "formal corpus source IDs must be unique and nonempty"
                )
            source_ids.add(source_id)
            loaded_sources.append(loaded_source)
            all_loaded_sources.append(loaded_source)
            normalized_path = _checked_repo_file(
                root,
                loaded_source.get("normalized_path"),
                field="normalized_path",
            )
            try:
                normalized_text = normalized_path.read_text(encoding="utf-8")
            except UnicodeDecodeError as error:
                raise Vs5CorpusIntegrityError(
                    f"case {case_id} normalized source is not UTF-8"
                ) from error
            source_by_id[source_id] = {
                "source": loaded_source,
                "normalized": normalized_text,
                "upload": loaded_source["text"],
            }
        declared_bundle_bytes = _required_int(
            case.get("upload_bundle_bytes"),
            field=f"case {case_id} upload_bundle_bytes",
        )
        if declared_bundle_bytes != sum(
            int(source["upload_bytes"]) for source in loaded_sources
        ):
            raise Vs5CorpusIntegrityError(
                f"case {case_id} upload_bundle_bytes is incorrect"
            )
        if any(int(source["upload_bytes"]) > MAX_SOURCE_BYTES for source in loaded_sources):
            raise Vs5CorpusIntegrityError(f"case {case_id} exceeds the per-source limit")
        if sum(int(source["upload_bytes"]) for source in loaded_sources) > MAX_CASE_BYTES:
            raise Vs5CorpusIntegrityError(f"case {case_id} exceeds the bundle limit")
        _validate_case_annotations(case=case, source_by_id=source_by_id)
        loaded_cases.append({**deepcopy(case), "sources": loaded_sources})

    if (
        _required_int(corpus.get("source_count"), field="source_count", minimum=1)
        != len(all_loaded_sources)
    ):
        raise Vs5CorpusIntegrityError("source_count does not match corpus sources")
    contradiction_case_count = sum(
        bool(list(case.get("contradiction_terms") or [])) for case in loaded_cases
    )
    if contradiction_case_count < 3:
        raise Vs5CorpusIntegrityError(
            "formal corpus must contain at least three declared contradiction cases"
        )
    review_ids = corpus.get("human_review_case_ids")
    if (
        not isinstance(review_ids, list)
        or len(review_ids) != 10
        or len(set(review_ids)) != 10
        or not set(review_ids) <= case_ids
        or review_ids != [str(case["id"]) for case in loaded_cases[:10]]
    ):
        raise Vs5CorpusIntegrityError(
            "human_review_case_ids must be the first ten unique corpus IDs"
        )

    _validate_exact_corpus_inventory(
        root=root,
        manifest_path=manifest_path,
        manifest_relative_path=manifest_relative_path,
        sources=all_loaded_sources,
    )

    loaded["cases"] = loaded_cases
    manifest_sha256 = _sha256_bytes(manifest_path.read_bytes())
    source_binding = _source_file_binding(all_loaded_sources)
    binding_payload = json.dumps(
        {
            "manifest_sha256": manifest_sha256,
            "source_manifest_sha256": source_binding["manifest_sha256"],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    binding = {
        "schema_version": "cs.vs5_corpus_binding.v1",
        "manifest_path": manifest_relative_path,
        "manifest_sha256": manifest_sha256,
        "case_count": len(loaded_cases),
        "source_count": len(all_loaded_sources),
        "bundle_sha256": _sha256_bytes(binding_payload),
        "source_files": source_binding,
    }
    return loaded, binding


def validate_vs5_corpus_freeze(
    root: Path,
    freeze_relative_path: str,
    binding: dict[str, Any],
) -> dict[str, Any]:
    freeze_path = _checked_repo_file(
        root,
        freeze_relative_path,
        field="corpus freeze path",
    )
    try:
        freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise Vs5CorpusIntegrityError("corpus freeze is not valid UTF-8 JSON") from error
    if not isinstance(freeze, dict) or freeze.get("schema_version") != FREEZE_SCHEMA:
        raise Vs5CorpusIntegrityError(f"corpus freeze schema must be {FREEZE_SCHEMA}")
    expected = {
        "manifest_path": binding["manifest_path"],
        "manifest_sha256": binding["manifest_sha256"],
        "case_count": binding["case_count"],
        "source_count": binding["source_count"],
    }
    for field, value in expected.items():
        if freeze.get(field) != value:
            raise Vs5CorpusIntegrityError(f"corpus freeze {field} is stale")
    freeze_bundle = freeze.get("bundle_sha256")
    if freeze_bundle is not None and freeze_bundle != binding["bundle_sha256"]:
        raise Vs5CorpusIntegrityError("corpus freeze bundle_sha256 is stale")
    source_manifest = freeze.get("source_manifest_sha256")
    if (
        source_manifest is not None
        and source_manifest != binding["source_files"]["manifest_sha256"]
    ):
        raise Vs5CorpusIntegrityError("corpus freeze source_manifest_sha256 is stale")
    return freeze
