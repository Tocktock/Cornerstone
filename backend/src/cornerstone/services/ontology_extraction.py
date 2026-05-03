from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from cornerstone.config import Settings
from cornerstone.schemas import (
    Concept,
    ConceptCandidate,
    CreateOntologyExtractionRunRequest,
    EvidenceFragment,
    OntologyCandidateStatus,
    OntologyExtractionProvider,
    OntologyExtractionRun,
    OntologyExtractionRunResponse,
    OntologyExtractionRunStatus,
    RelationCandidate,
    RelationType,
    normalize_concept_term,
    utc_now,
)
from cornerstone.store import NotFoundError


@dataclass(slots=True)
class _ConceptDraft:
    name: str
    proposed_definition: str
    concept_type: str
    evidence_fragment_ids: set[str] = field(default_factory=set)
    confidence: float = 0.55
    rationale: str | None = None


@dataclass(slots=True)
class _RelationDraft:
    source_name: str
    target_name: str
    relation_type: RelationType
    evidence_fragment_ids: set[str] = field(default_factory=set)
    confidence: float = 0.7
    rationale: str = "Extracted from explicit relation language in evidence."


@dataclass(slots=True)
class OntologyExtractionDraft:
    concepts: list[_ConceptDraft]
    relations: list[_RelationDraft]
    warning_count: int = 0


class _StrictConceptPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    definition: str = Field(min_length=1)
    concept_type: str = Field(default="domain_concept", min_length=1)
    evidence_fragment_ids: list[str] = Field(min_length=1)
    confidence: float = Field(default=0.7, ge=0, le=1)
    rationale: str | None = None


class _StrictRelationPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_name: str = Field(min_length=1)
    target_name: str = Field(min_length=1)
    relation_type: RelationType
    evidence_fragment_ids: list[str] = Field(min_length=1)
    confidence: float = Field(default=0.7, ge=0, le=1)
    rationale: str = Field(default="Proposed by live ontology provider.", min_length=1)


class _StrictOntologyPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concepts: list[_StrictConceptPayload] = Field(default_factory=list)
    relations: list[_StrictRelationPayload] = Field(default_factory=list)


class OntologyExtractor(Protocol):
    """Provider boundary for ontology extraction.

    v1.4.0 ships a local deterministic provider so the candidate persistence and
    API contracts can be tested without external credentials. A live LLM provider
    can implement this protocol later while preserving the same validation rules.
    """

    provider: OntologyExtractionProvider
    model_name: str
    prompt_version: str

    def extract(
        self,
        *,
        evidence_fragments: list[EvidenceFragment],
        focus_concept: str | None,
    ) -> OntologyExtractionDraft:
        ...


class LocalRuleBasedOntologyExtractor:
    provider = OntologyExtractionProvider.LOCAL_RULE_BASED
    model_name = "local-rule-based-ontology-extractor-v1.4.0"
    prompt_version = "ontology-extraction-v1.4.0"

    _definition_pattern = re.compile(
        r"^(?P<name>[A-Za-z][A-Za-z0-9 /_-]{1,80}?)\s+"
        r"(?P<verb>is|are|means|refers to|describes)\s+"
        r"(?P<definition>.+)$",
        re.IGNORECASE,
    )
    _relation_patterns: tuple[tuple[RelationType, re.Pattern[str]], ...] = (
        (
            RelationType.PRECEDES,
            re.compile(r"^(?P<source>.+?)\s+precedes\s+(?P<target>.+)$", re.IGNORECASE),
        ),
        (
            RelationType.FOLLOWS,
            re.compile(r"^(?P<source>.+?)\s+follows\s+(?P<target>.+)$", re.IGNORECASE),
        ),
        (
            RelationType.DEPENDS_ON,
            re.compile(r"^(?P<source>.+?)\s+depends\s+on\s+(?P<target>.+)$", re.IGNORECASE),
        ),
        (
            RelationType.GOVERNED_BY,
            re.compile(r"^(?P<source>.+?)\s+is\s+governed\s+by\s+(?P<target>.+)$", re.IGNORECASE),
        ),
        (
            RelationType.UPDATES,
            re.compile(r"^(?P<source>.+?)\s+updates\s+(?P<target>.+)$", re.IGNORECASE),
        ),
        (
            RelationType.VALIDATES,
            re.compile(r"^(?P<source>.+?)\s+validates\s+(?P<target>.+)$", re.IGNORECASE),
        ),
        (
            RelationType.TRIGGERS,
            re.compile(r"^(?P<source>.+?)\s+triggers\s+(?P<target>.+)$", re.IGNORECASE),
        ),
        (
            RelationType.BLOCKS,
            re.compile(r"^(?P<source>.+?)\s+blocks\s+(?P<target>.+)$", re.IGNORECASE),
        ),
        (
            RelationType.PRODUCES,
            re.compile(r"^(?P<source>.+?)\s+produces\s+(?P<target>.+)$", re.IGNORECASE),
        ),
        (
            RelationType.CONSUMES,
            re.compile(r"^(?P<source>.+?)\s+consumes\s+(?P<target>.+)$", re.IGNORECASE),
        ),
    )

    def extract(
        self,
        *,
        evidence_fragments: list[EvidenceFragment],
        focus_concept: str | None,
    ) -> OntologyExtractionDraft:
        concept_by_normalized: dict[str, _ConceptDraft] = {}
        relation_by_key: dict[tuple[str, str, RelationType], _RelationDraft] = {}
        warning_count = 0

        for evidence in evidence_fragments:
            for sentence in _split_sentences(evidence.text):
                definition_match = self._definition_pattern.match(sentence)
                if definition_match:
                    raw_name = definition_match.group("name")
                    name = _clean_term(raw_name)
                    definition = _clean_definition(definition_match.group("definition"))
                    if name and definition:
                        _merge_concept(
                            concept_by_normalized,
                            name=name,
                            proposed_definition=definition,
                            concept_type=_infer_concept_type(definition),
                            evidence_fragment_id=evidence.id,
                            confidence=0.72,
                            rationale="Extracted from explicit definition-style evidence.",
                        )
                    else:
                        warning_count += 1

                matched_relation = False
                for relation_type, pattern in self._relation_patterns:
                    relation_match = pattern.match(sentence)
                    if relation_match is None:
                        continue
                    source_name = _clean_term(relation_match.group("source"))
                    target_name = _clean_term(relation_match.group("target"))
                    if not source_name or not target_name:
                        warning_count += 1
                        continue
                    normalized_source = normalize_concept_term(source_name)
                    normalized_target = normalize_concept_term(target_name)
                    if normalized_source == normalized_target:
                        warning_count += 1
                        continue
                    _merge_placeholder_concept(
                        concept_by_normalized,
                        name=source_name,
                        evidence_fragment_id=evidence.id,
                    )
                    _merge_placeholder_concept(
                        concept_by_normalized,
                        name=target_name,
                        evidence_fragment_id=evidence.id,
                    )
                    key = (normalized_source, normalized_target, relation_type)
                    draft = relation_by_key.get(key)
                    if draft is None:
                        draft = _RelationDraft(
                            source_name=source_name,
                            target_name=target_name,
                            relation_type=relation_type,
                            evidence_fragment_ids={evidence.id},
                            confidence=0.74,
                            rationale=f"Evidence states that {source_name} {relation_type.value} {target_name}.",
                        )
                        relation_by_key[key] = draft
                    else:
                        draft.evidence_fragment_ids.add(evidence.id)
                    matched_relation = True
                    break
                if not matched_relation and focus_concept:
                    _maybe_add_focus_mention(
                        concept_by_normalized,
                        sentence=sentence,
                        focus_concept=focus_concept,
                        evidence_fragment_id=evidence.id,
                    )

        return OntologyExtractionDraft(
            concepts=list(concept_by_normalized.values()),
            relations=list(relation_by_key.values()),
            warning_count=warning_count,
        )


class LiveLlmOntologyExtractor:
    provider = OntologyExtractionProvider.LIVE_LLM

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model_name = settings.ontology_live_llm_model_name
        self.prompt_version = settings.ontology_live_llm_prompt_version

    def extract(
        self,
        *,
        evidence_fragments: list[EvidenceFragment],
        focus_concept: str | None,
    ) -> OntologyExtractionDraft:
        if not self.settings.ontology_live_llm_enabled:
            raise ValueError(
                "Live ontology provider is disabled. Set ONTOLOGY_LIVE_LLM_ENABLED=true and configure the provider explicitly."
            )
        payload_text = self._load_payload_text(
            evidence_fragments=evidence_fragments,
            focus_concept=focus_concept,
        )
        return _draft_from_strict_payload(
            payload_text,
            evidence_fragments=evidence_fragments,
        )

    def _load_payload_text(
        self,
        *,
        evidence_fragments: list[EvidenceFragment],
        focus_concept: str | None,
    ) -> str:
        if self.settings.ontology_live_llm_fixture_response_json.strip():
            return self.settings.ontology_live_llm_fixture_response_json
        if not self.settings.ontology_live_llm_api_url.strip():
            raise ValueError("ONTOLOGY_LIVE_LLM_API_URL is required for live ontology extraction.")
        request_payload = {
            "model": self.model_name,
            "promptVersion": self.prompt_version,
            "focusConcept": focus_concept,
            "evidence": [
                {
                    "id": fragment.id,
                    "text": fragment.text,
                    "artifactId": fragment.artifact_id,
                    "sourceExternalId": fragment.provenance.source_external_id,
                }
                for fragment in evidence_fragments
            ],
            "responseContract": {
                "concepts": [
                    {
                        "name": "string",
                        "definition": "string",
                        "concept_type": "string",
                        "evidence_fragment_ids": ["known-evidence-id"],
                        "confidence": 0.0,
                        "rationale": "string",
                    }
                ],
                "relations": [
                    {
                        "source_name": "string",
                        "target_name": "string",
                        "relation_type": "precedes",
                        "evidence_fragment_ids": ["known-evidence-id"],
                        "confidence": 0.0,
                        "rationale": "string",
                    }
                ],
            },
        }
        data = json.dumps(request_payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.settings.ontology_live_llm_api_key:
            headers["Authorization"] = f"Bearer {self.settings.ontology_live_llm_api_key}"
        request = urllib.request.Request(
            self.settings.ontology_live_llm_api_url,
            data=data,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.settings.ontology_live_llm_timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise ValueError(f"Live ontology provider request failed: {exc.reason}") from exc
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and "choices" in parsed:
            try:
                content = parsed["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError("Live ontology provider returned an unsupported choices payload.") from exc
            return str(content)
        return raw


class OntologyExtractionService:
    def __init__(
        self,
        store: object,
        *,
        extractor: OntologyExtractor | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.store = store
        self.extractor = extractor
        self.settings = settings or Settings()

    def create_run(self, request: CreateOntologyExtractionRunRequest) -> OntologyExtractionRunResponse:
        extractor = self.extractor or _extractor_for_request(request.provider, self.settings)
        if request.provider != extractor.provider:
            raise ValueError(f"Unsupported ontology extraction provider: {request.provider}")

        evidence_fragments = self._load_evidence_fragments(request)
        if not evidence_fragments:
            raise ValueError("Ontology extraction scope did not resolve to any EvidenceFragments.")

        evidence_fragments = evidence_fragments[: request.max_evidence_fragments]
        run = OntologyExtractionRun(
            provider=extractor.provider,
            model_name=request.model_name or extractor.model_name,
            prompt_version=request.prompt_version or extractor.prompt_version,
            status=OntologyExtractionRunStatus.RUNNING,
            requested_by=request.requested_by,
            focus_concept=request.focus_concept,
            evidence_fragment_ids=[item.id for item in evidence_fragments],
            artifact_ids=request.artifact_ids,
            started_at=utc_now(),
        )
        self.store.add_ontology_extraction_run(run)

        try:
            draft = extractor.extract(
                evidence_fragments=evidence_fragments,
                focus_concept=request.focus_concept,
            )
            _validate_draft_evidence(draft, evidence_fragments=evidence_fragments)
            existing_concept_ids = _existing_concept_ids_by_normalized_term(self.store.list_concepts())
            concept_candidates = self._persist_concept_candidates(
                run=run,
                drafts=draft.concepts,
                existing_concept_ids=existing_concept_ids,
            )
            concept_ids_by_normalized_name = {
                candidate.normalized_name: candidate.id for candidate in concept_candidates
            }
            relation_candidates = self._persist_relation_candidates(
                run=run,
                drafts=draft.relations,
                concept_candidate_ids_by_normalized_name=concept_ids_by_normalized_name,
                existing_concept_ids=existing_concept_ids,
            )
            run.status = OntologyExtractionRunStatus.COMPLETED
            run.completed_at = utc_now()
            run.concept_candidate_count = len(concept_candidates)
            run.relation_candidate_count = len(relation_candidates)
            run.warning_count = draft.warning_count
            run = self.store.update_ontology_extraction_run(run)
            return OntologyExtractionRunResponse(
                run=run,
                concept_candidates=concept_candidates,
                relation_candidates=relation_candidates,
            )
        except Exception as exc:
            run.status = OntologyExtractionRunStatus.FAILED
            run.error = str(exc)
            run.completed_at = utc_now()
            self.store.update_ontology_extraction_run(run)
            raise

    def get_run_response(self, run_id: str) -> OntologyExtractionRunResponse:
        run = self.store.get_ontology_extraction_run(run_id)
        return OntologyExtractionRunResponse(
            run=run,
            concept_candidates=self.store.list_concept_candidates(extraction_run_id=run_id),
            relation_candidates=self.store.list_relation_candidates(extraction_run_id=run_id),
        )

    def _load_evidence_fragments(self, request: CreateOntologyExtractionRunRequest) -> list[EvidenceFragment]:
        seen: set[str] = set()
        results: list[EvidenceFragment] = []

        for evidence_id in request.evidence_fragment_ids:
            evidence = self.store.get_evidence_fragment(evidence_id)
            if evidence.id not in seen:
                results.append(evidence)
                seen.add(evidence.id)

        for artifact_id in request.artifact_ids:
            for evidence in self.store.list_evidence_fragments(artifact_id=artifact_id):
                if evidence.id not in seen:
                    results.append(evidence)
                    seen.add(evidence.id)

        return results

    def _persist_concept_candidates(
        self,
        *,
        run: OntologyExtractionRun,
        drafts: list[_ConceptDraft],
        existing_concept_ids: dict[str, str],
    ) -> list[ConceptCandidate]:
        candidates: list[ConceptCandidate] = []
        for draft in drafts:
            normalized_name = normalize_concept_term(draft.name)
            if not normalized_name:
                continue
            candidate = ConceptCandidate(
                extraction_run_id=run.id,
                name=_display_name(draft.name),
                normalized_name=normalized_name,
                aliases=[],
                proposed_definition=draft.proposed_definition,
                concept_type=draft.concept_type,
                evidence_fragment_ids=sorted(draft.evidence_fragment_ids),
                confidence=draft.confidence,
                status=OntologyCandidateStatus.PENDING,
                matched_existing_concept_id=existing_concept_ids.get(normalized_name),
                rationale=draft.rationale,
            )
            candidates.append(self.store.add_concept_candidate(candidate))
        return candidates

    def _persist_relation_candidates(
        self,
        *,
        run: OntologyExtractionRun,
        drafts: list[_RelationDraft],
        concept_candidate_ids_by_normalized_name: dict[str, str],
        existing_concept_ids: dict[str, str],
    ) -> list[RelationCandidate]:
        candidates: list[RelationCandidate] = []
        for draft in drafts:
            normalized_source = normalize_concept_term(draft.source_name)
            normalized_target = normalize_concept_term(draft.target_name)
            if not normalized_source or not normalized_target or normalized_source == normalized_target:
                continue
            candidate = RelationCandidate(
                extraction_run_id=run.id,
                source_name=_display_name(draft.source_name),
                target_name=_display_name(draft.target_name),
                normalized_source_name=normalized_source,
                normalized_target_name=normalized_target,
                source_candidate_id=concept_candidate_ids_by_normalized_name.get(normalized_source),
                target_candidate_id=concept_candidate_ids_by_normalized_name.get(normalized_target),
                source_concept_id=existing_concept_ids.get(normalized_source),
                target_concept_id=existing_concept_ids.get(normalized_target),
                relation_type=draft.relation_type,
                evidence_fragment_ids=sorted(draft.evidence_fragment_ids),
                confidence=draft.confidence,
                rationale=draft.rationale,
                status=OntologyCandidateStatus.PENDING,
            )
            candidates.append(self.store.add_relation_candidate(candidate))
        return candidates


def _split_sentences(text: str) -> list[str]:
    normalized = " ".join(text.strip().split())
    if not normalized:
        return []
    return [part.strip(" \t\r\n.;:") for part in re.split(r"(?<=[.!?])\s+", normalized) if part.strip(" \t\r\n.;:")]


def _clean_term(value: str) -> str:
    cleaned = value.strip(" \t\r\n\"'`.,;:!?()[]{}")
    cleaned = re.sub(r"^(the|a|an)\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned)
    # Keep noun-like leading chunks and drop common trailing qualifiers that make
    # relation endpoints too sentence-like.
    cleaned = re.sub(r"\s+(after|before|when|where|because)\s+.*$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def _clean_definition(value: str) -> str:
    cleaned = value.strip(" \t\r\n\"'`.,;:!?()[]{}")
    return re.sub(r"\s+", " ", cleaned)


def _display_name(value: str) -> str:
    cleaned = _clean_term(value)
    if cleaned.islower() or cleaned.isupper():
        return cleaned.title()
    return cleaned[0].upper() + cleaned[1:] if cleaned else cleaned


def _infer_concept_type(definition: str) -> str:
    lower = definition.casefold()
    if "process" in lower or "workflow" in lower:
        return "process"
    if "policy" in lower or "rule" in lower:
        return "policy"
    if "team" in lower or "owner" in lower:
        return "team"
    if "system" in lower or "service" in lower or "application" in lower:
        return "system"
    if "metric" in lower or "measure" in lower or "indicator" in lower:
        return "metric"
    if "document" in lower or "guide" in lower:
        return "document"
    return "domain_concept"


def _merge_concept(
    concept_by_normalized: dict[str, _ConceptDraft],
    *,
    name: str,
    proposed_definition: str,
    concept_type: str,
    evidence_fragment_id: str,
    confidence: float,
    rationale: str,
) -> None:
    normalized = normalize_concept_term(name)
    if not normalized:
        return
    existing = concept_by_normalized.get(normalized)
    if existing is None:
        concept_by_normalized[normalized] = _ConceptDraft(
            name=_display_name(name),
            proposed_definition=proposed_definition,
            concept_type=concept_type,
            evidence_fragment_ids={evidence_fragment_id},
            confidence=confidence,
            rationale=rationale,
        )
        return
    existing.evidence_fragment_ids.add(evidence_fragment_id)
    if existing.confidence < confidence:
        existing.proposed_definition = proposed_definition
        existing.concept_type = concept_type
        existing.confidence = confidence
        existing.rationale = rationale


def _merge_placeholder_concept(
    concept_by_normalized: dict[str, _ConceptDraft],
    *,
    name: str,
    evidence_fragment_id: str,
) -> None:
    normalized = normalize_concept_term(name)
    if not normalized or normalized in concept_by_normalized:
        if normalized:
            concept_by_normalized[normalized].evidence_fragment_ids.add(evidence_fragment_id)
        return
    display_name = _display_name(name)
    concept_by_normalized[normalized] = _ConceptDraft(
        name=display_name,
        proposed_definition=(
            f"{display_name} is mentioned in source evidence. "
            "A reviewer should confirm the official definition before promotion."
        ),
        concept_type="domain_concept",
        evidence_fragment_ids={evidence_fragment_id},
        confidence=0.52,
        rationale="Created as a relation endpoint mentioned in evidence.",
    )


def _maybe_add_focus_mention(
    concept_by_normalized: dict[str, _ConceptDraft],
    *,
    sentence: str,
    focus_concept: str,
    evidence_fragment_id: str,
) -> None:
    if normalize_concept_term(focus_concept) not in normalize_concept_term(sentence):
        return
    _merge_placeholder_concept(
        concept_by_normalized,
        name=focus_concept,
        evidence_fragment_id=evidence_fragment_id,
    )


def _existing_concept_ids_by_normalized_term(concepts: list[Concept]) -> dict[str, str]:
    result: dict[str, str] = {}
    for concept in concepts:
        result[normalize_concept_term(concept.name)] = concept.id
        for alias in concept.aliases:
            result[normalize_concept_term(alias)] = concept.id
    return result


def _extractor_for_request(
    provider: OntologyExtractionProvider,
    settings: Settings,
) -> OntologyExtractor:
    if provider == OntologyExtractionProvider.LOCAL_RULE_BASED:
        return LocalRuleBasedOntologyExtractor()
    if provider == OntologyExtractionProvider.LIVE_LLM:
        return LiveLlmOntologyExtractor(settings)
    raise ValueError(f"Unsupported ontology extraction provider: {provider}")


def _draft_from_strict_payload(
    payload_text: str,
    *,
    evidence_fragments: list[EvidenceFragment],
) -> OntologyExtractionDraft:
    try:
        raw = json.loads(payload_text)
        payload = _StrictOntologyPayload.model_validate(raw)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"Live ontology provider returned invalid structured output: {exc}") from exc
    draft = OntologyExtractionDraft(
        concepts=[
            _ConceptDraft(
                name=item.name,
                proposed_definition=item.definition,
                concept_type=item.concept_type,
                evidence_fragment_ids=set(item.evidence_fragment_ids),
                confidence=item.confidence,
                rationale=item.rationale or "Proposed by live ontology provider.",
            )
            for item in payload.concepts
        ],
        relations=[
            _RelationDraft(
                source_name=item.source_name,
                target_name=item.target_name,
                relation_type=item.relation_type,
                evidence_fragment_ids=set(item.evidence_fragment_ids),
                confidence=item.confidence,
                rationale=item.rationale,
            )
            for item in payload.relations
        ],
    )
    _validate_draft_evidence(draft, evidence_fragments=evidence_fragments)
    return draft


def _validate_draft_evidence(
    draft: OntologyExtractionDraft,
    *,
    evidence_fragments: list[EvidenceFragment],
) -> None:
    known_ids = {fragment.id for fragment in evidence_fragments}
    for concept in draft.concepts:
        if not concept.evidence_fragment_ids:
            raise ValueError(f"Concept '{concept.name}' is missing evidence ids.")
        unknown = sorted(concept.evidence_fragment_ids - known_ids)
        if unknown:
            raise ValueError(f"Concept '{concept.name}' references unknown evidence ids: {', '.join(unknown)}")
    for relation in draft.relations:
        if not relation.evidence_fragment_ids:
            raise ValueError(
                f"Relation '{relation.source_name} {relation.relation_type} {relation.target_name}' is missing evidence ids."
            )
        unknown = sorted(relation.evidence_fragment_ids - known_ids)
        if unknown:
            raise ValueError(
                f"Relation '{relation.source_name} {relation.relation_type} {relation.target_name}' references unknown evidence ids: {', '.join(unknown)}"
            )
        if normalize_concept_term(relation.source_name) == normalize_concept_term(relation.target_name):
            raise ValueError("Relation source and target must differ.")
