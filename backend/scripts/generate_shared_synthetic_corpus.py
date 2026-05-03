#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DATASET_ID = "cornerstone-shared-synthetic-corpus-v1"
DATASET_VERSION = "1.0.0"
DOMAIN = "temperature-controlled specialty pharmacy logistics"
ORG_NAME = "HelioPharm Cold Chain Operations"
FIXED_GENERATED_AT = "2026-05-03"


@dataclass(frozen=True)
class ConceptProfile:
    slug: str
    name: str
    aliases: tuple[str, ...]
    category: str
    owner: str
    definition: str
    metric: str
    requirement: str
    policy: str
    decision: str
    example: str
    risk: str
    open_question: str


CONCEPTS: tuple[ConceptProfile, ...] = (
    ConceptProfile(
        slug="lane-qualification",
        name="Lane Qualification",
        aliases=("qualified lane", "route qualification", "lane validation"),
        category="logistics_control",
        owner="Quality Operations",
        definition="Lane Qualification means a documented approval that a transport lane can protect product quality under expected seasonal, carrier, customs, and handoff conditions.",
        metric="The lane qualification score is the weighted sum of seasonal thermal margin, carrier handoff reliability, customs dwell exposure, and recovery access.",
        requirement="A lane must not be released for commercial biologic movement until a qualification packet includes route risk scoring, validated shipper fit, and escalation contacts.",
        policy="The policy requires every qualified lane to be rechecked before summer and winter operating windows.",
        decision="The review board decided that any lane with two unresolved excursion near misses in ninety days moves back to pilot status.",
        example="For instance, the Seoul to Singapore oncology lane uses active monitoring during airport dwell because the customs hold risk is material.",
        risk="The main risk is silent route drift after a carrier subcontractor changes the airport handoff pattern.",
        open_question="Should lane qualification include a separate score for weekend customs staffing?",
    ),
    ConceptProfile(
        slug="stability-budget",
        name="Stability Budget",
        aliases=("thermal budget", "stability allowance", "excursion budget"),
        category="quality_science",
        owner="Product Quality",
        definition="Stability Budget means the remaining approved exposure allowance for a product lot after considering temperature, duration, and product-specific stability evidence.",
        metric="The budget is tracked in minutes by temperature band and is reduced only by reviewed excursion evidence.",
        requirement="A release decision must preserve a nonnegative stability budget for every shipped unit and every documented handoff.",
        policy="The policy requires budget calculation to use the product stability memo, not the shipment average temperature.",
        decision="The quality council decided that estimated budget recovery is prohibited unless the stability memo explicitly names the recovery method.",
        example="For instance, a ten minute 8.7 C event may consume more budget than a thirty minute 7.9 C event for a narrow 2-8 C biologic.",
        risk="The main risk is aggregating sensor readings before applying the product-specific temperature band.",
        open_question="Should the stability budget service expose remaining budget as a graph edge attribute?",
    ),
    ConceptProfile(
        slug="temperature-excursion",
        name="Temperature Excursion",
        aliases=("thermal excursion", "out-of-range event", "temperature breach"),
        category="quality_event",
        owner="Quality Operations",
        definition="Temperature Excursion means an observed or inferred product exposure outside the approved temperature range for a material duration.",
        metric="Excursion severity is scored by peak deviation, duration, sensor confidence, lane state, and remaining stability budget.",
        requirement="A temperature excursion must create a deviation triage record before the shipment can be released or discarded.",
        policy="The policy requires raw sensor files to be retained with the release evidence packet for every excursion.",
        decision="The incident review group decided that inferred excursions from missing sensor intervals require the same triage path as observed excursions.",
        example="For instance, a gap between gateway pings can indicate a possible excursion if the shipper was on an airport tarmac during high heat.",
        risk="The main risk is treating the carrier delivered timestamp as proof that no excursion occurred.",
        open_question="Should inferred excursions be visible to customers before quality review is complete?",
    ),
    ConceptProfile(
        slug="chain-of-custody",
        name="Chain of Custody",
        aliases=("custody chain", "handoff chain", "custody evidence"),
        category="compliance_record",
        owner="Compliance",
        definition="Chain of Custody means the ordered record of accountable handoffs from pack-out through patient site receipt.",
        metric="Custody completeness is measured by signed handoffs, timestamp consistency, geofence match, and identity verification.",
        requirement="Every custody handoff must include actor identity, location, timestamp, package seal state, and exception notes when applicable.",
        policy="The policy requires custody evidence to be immutable after release except through a correction record.",
        decision="The compliance forum decided that courier app signatures are acceptable only when paired with device identity and route geofence evidence.",
        example="For instance, a depot supervisor handoff without seal state is incomplete even when the driver signature is present.",
        risk="The main risk is relying on carrier milestone labels that do not identify the accountable actor.",
        open_question="Should custody gaps automatically reduce the graph trust label for affected concepts?",
    ),
    ConceptProfile(
        slug="validated-shipper",
        name="Validated Shipper",
        aliases=("qualified shipper", "thermal shipper", "packout system"),
        category="packaging_system",
        owner="Packaging Engineering",
        definition="Validated Shipper means a packaging configuration proven to maintain the target temperature range for a defined duration and payload profile.",
        metric="Shipper suitability is measured by qualified duration, payload mass, refrigerant conditioning state, and lane risk margin.",
        requirement="A validated shipper must match product temperature class, planned lane duration, and contingency hold time.",
        policy="The policy requires a new validation or engineering assessment after any insulation material, coolant, or payload bracket changes.",
        decision="The packaging board decided that passive shippers may not be used on high humidity customs lanes without extra condensation checks.",
        example="For instance, a 96 hour shipper can fail a 42 hour route when the payload mass is below the validated bracket.",
        risk="The main risk is treating nameplate duration as universal instead of validation-scope-specific.",
        open_question="Should shipper validation results be modeled as first-class evidence nodes?",
    ),
    ConceptProfile(
        slug="quality-hold",
        name="Quality Hold",
        aliases=("QA hold", "release hold", "product hold"),
        category="release_control",
        owner="Quality Assurance",
        definition="Quality Hold means a controlled state that prevents product release until quality evidence is reviewed and disposition is recorded.",
        metric="Hold aging is measured by elapsed hours, missing evidence count, patient impact tier, and next required reviewer.",
        requirement="A shipment under quality hold must not be released to inventory, patient scheduling, or billing systems.",
        policy="The policy requires quality hold reason codes to reference the triggering evidence and the required disposition path.",
        decision="The release board decided that stability budget uncertainty always creates a quality hold until Product Quality signs the calculation.",
        example="For instance, a custody gap and a sensor gap can share one quality hold when they affect the same physical shipment.",
        risk="The main risk is a downstream system clearing the hold because carrier delivery is complete.",
        open_question="Should quality hold state expire if no reviewer acts within the service-level target?",
    ),
    ConceptProfile(
        slug="deviation-triage",
        name="Deviation Triage",
        aliases=("deviation assessment", "event triage", "quality triage"),
        category="quality_workflow",
        owner="Quality Operations",
        definition="Deviation Triage means the first structured assessment that classifies a quality event, assigns ownership, and selects the disposition path.",
        metric="Triage quality is measured by classification accuracy, evidence completeness, owner assignment, and time to first disposition.",
        requirement="Deviation triage must identify whether the event affects product quality, patient scheduling, regulatory reporting, or only operational performance.",
        policy="The policy requires triage notes to separate observed facts from inferred hypotheses.",
        decision="The quality operations team decided that triage can close low-risk paperwork deviations only when no product exposure uncertainty exists.",
        example="For instance, a missing receiver signature is paperwork-only when custody is otherwise proven by geofence and device evidence.",
        risk="The main risk is closing a deviation before stability budget and custody evidence are reviewed together.",
        open_question="Should triage confidence be computed by rule or assigned by the reviewer?",
    ),
    ConceptProfile(
        slug="capa",
        name="CAPA",
        aliases=("corrective preventive action", "corrective action", "preventive action"),
        category="quality_system",
        owner="Quality Systems",
        definition="CAPA means a corrective and preventive action record that addresses root cause, containment, verification, and recurrence prevention.",
        metric="CAPA effectiveness is measured by repeat-event rate, verification evidence quality, owner timeliness, and control durability.",
        requirement="A CAPA must link to the triggering deviation, the root cause statement, effectiveness criteria, and closure evidence.",
        policy="The policy requires preventive actions to name the control that will detect recurrence before patient impact.",
        decision="The CAPA council decided that training-only actions are insufficient for recurring lane qualification failures.",
        example="For instance, a revised carrier checklist is weak unless paired with audit sampling and route risk score recalibration.",
        risk="The main risk is writing corrective actions that address symptoms but not route or packaging control design.",
        open_question="Should CAPA effectiveness be visible in the official ontology graph?",
    ),
    ConceptProfile(
        slug="calibration-drift",
        name="Calibration Drift",
        aliases=("sensor drift", "logger drift", "calibration variance"),
        category="measurement_control",
        owner="Metrology",
        definition="Calibration Drift means movement of a measurement device away from accepted tolerance between calibration events.",
        metric="Drift risk is measured by days since calibration, observed variance, device family history, and excursion decision dependency.",
        requirement="A sensor with unresolved calibration drift must not be used as the sole evidence for product release.",
        policy="The policy requires paired sensor comparison when drift affects a shipment with limited stability budget.",
        decision="The metrology board decided that post-use calibration failure reopens any release decision that depended on that device.",
        example="For instance, a logger reading 7.8 C may not prove compliance if its paired sensor reads 8.4 C and drift is unresolved.",
        risk="The main risk is using a single sensor trace without checking calibration certificate status.",
        open_question="Should calibration drift automatically downgrade evidence freshness or trust state?",
    ),
    ConceptProfile(
        slug="refrigerant-conditioning",
        name="Refrigerant Conditioning",
        aliases=("coolant conditioning", "gel pack conditioning", "phase-change conditioning"),
        category="packaging_process",
        owner="Packaging Operations",
        definition="Refrigerant Conditioning means preparing coolant to the validated temperature and physical state before pack-out.",
        metric="Conditioning readiness is measured by chamber dwell time, surface probe confirmation, batch count, and pack-out delay.",
        requirement="Refrigerant conditioning must match the shipper validation recipe and must be recorded before pack-out starts.",
        policy="The policy requires a restart of conditioning when coolant sits outside the staging window.",
        decision="The packaging operations review decided that visual frost inspection is not evidence of correct phase state.",
        example="For instance, a gel pack conditioned for frozen use can damage 2-8 C product when placed in a narrow payload cavity.",
        risk="The main risk is substituting a local packing habit for the validated conditioning recipe.",
        open_question="Should conditioning chamber telemetry be attached to each release evidence packet?",
    ),
    ConceptProfile(
        slug="route-risk-scoring",
        name="Route Risk Scoring",
        aliases=("lane risk score", "transport risk score", "route risk model"),
        category="risk_model",
        owner="Network Planning",
        definition="Route Risk Scoring means the calculation that estimates product quality risk across route, weather, customs, carrier, and contingency factors.",
        metric="The score uses thermal margin, dwell exposure, carrier reliability, contingency depot access, and seasonal volatility.",
        requirement="A high route risk score must trigger either enhanced monitoring, a stronger shipper, or a quality-approved exception.",
        policy="The policy requires score inputs to be auditable and reproducible from source evidence.",
        decision="The network planning group decided that weekend border crossings add risk unless a named customs broker is on call.",
        example="For instance, a short route can have high risk when airport dwell occurs during a heat advisory.",
        risk="The main risk is allowing planners to override risk factors without a decision record.",
        open_question="Should route risk scoring be recalculated after every deviation or only at lane review?",
    ),
    ConceptProfile(
        slug="release-evidence-packet",
        name="Release Evidence Packet",
        aliases=("release packet", "disposition packet", "QA evidence packet"),
        category="release_record",
        owner="Quality Assurance",
        definition="Release Evidence Packet means the collected evidence used to justify shipment disposition and downstream release state.",
        metric="Packet readiness is measured by required evidence completeness, reviewer signoff, exception closure, and unresolved question count.",
        requirement="A release evidence packet must include stability budget calculation, custody evidence, sensor files, deviation records, and final disposition.",
        policy="The policy requires release packets to preserve source evidence links instead of copied summary text only.",
        decision="The QA board decided that release packets are the canonical audit object for shipped specialty pharmacy product.",
        example="For instance, a packet can cite a CAPA for prevention but still needs shipment-specific stability evidence.",
        risk="The main risk is approving release from a narrative summary without the raw evidence trail.",
        open_question="Should the release evidence packet become the default graph focus for audit queries?",
    ),
    ConceptProfile(
        slug="fefo-allocation",
        name="FEFO Allocation",
        aliases=("first-expiry-first-out", "expiry allocation", "lot allocation"),
        category="inventory_control",
        owner="Inventory Operations",
        definition="FEFO Allocation means selecting inventory by earliest usable expiry while respecting quality state, route feasibility, and patient scheduling constraints.",
        metric="FEFO effectiveness is measured by expiry waste, release eligibility, patient promise adherence, and avoidable exception rate.",
        requirement="FEFO allocation must exclude units on quality hold and units without enough stability budget for the planned lane.",
        policy="The policy requires allocation logic to treat release eligibility as stronger than expiry priority.",
        decision="The inventory council decided that patient-critical shipments may bypass normal FEFO order only with a decision record.",
        example="For instance, the earliest expiring lot cannot be allocated if its release evidence packet is incomplete.",
        risk="The main risk is optimizing expiry while ignoring route risk and quality hold state.",
        open_question="Should FEFO allocation consume graph answers directly or only read materialized release state?",
    ),
    ConceptProfile(
        slug="sensor-pairing",
        name="Sensor Pairing",
        aliases=("dual logger pairing", "paired sensors", "sensor redundancy"),
        category="measurement_control",
        owner="Metrology",
        definition="Sensor Pairing means assigning two compatible measurement devices to a shipment so readings can validate each other and expose drift or placement error.",
        metric="Pairing reliability is measured by device compatibility, placement separation, time synchronization, and drift agreement.",
        requirement="Sensor pairing must be used when shipment value, stability budget, or regulatory category requires redundant evidence.",
        policy="The policy requires paired sensors to be time-synchronized before pack-out and reconciled after receipt.",
        decision="The metrology team decided that sensor pairing is mandatory for pediatric oncology biologic lanes.",
        example="For instance, a top-cavity logger and payload-core logger can reveal a local placement artifact.",
        risk="The main risk is pairing two devices from the same suspect calibration batch.",
        open_question="Should paired sensor disagreement create automatic deviation triage?",
    ),
    ConceptProfile(
        slug="quarantine-workflow",
        name="Quarantine Workflow",
        aliases=("inventory quarantine", "segregation workflow", "blocked stock workflow"),
        category="inventory_control",
        owner="Warehouse Quality",
        definition="Quarantine Workflow means the physical and system segregation process for units that cannot be released until quality disposition is complete.",
        metric="Quarantine performance is measured by segregation accuracy, hold aging, scan compliance, and release reconciliation.",
        requirement="Quarantine workflow must prevent pick, pack, transfer, billing, and patient dispatch for affected units.",
        policy="The policy requires physical labels and system status to agree before a unit is considered quarantined.",
        decision="The warehouse quality team decided that scanner override is prohibited for quarantined biologic inventory.",
        example="For instance, a unit can be physically in the quarantine cage but still unsafe if warehouse management status is available.",
        risk="The main risk is a manual inventory move that bypasses the quality hold state.",
        open_question="Should quarantine workflow publish events to patient scheduling when a critical dose is affected?",
    ),
    ConceptProfile(
        slug="gdp-audit-readiness",
        name="GDP Audit Readiness",
        aliases=("good distribution practice audit", "GDP readiness", "distribution audit readiness"),
        category="compliance_program",
        owner="Compliance",
        definition="GDP Audit Readiness means the ability to prove distribution controls, evidence integrity, and quality decisions during a good distribution practice audit.",
        metric="Readiness is measured by trace completeness, decision record coverage, training currency, and evidence retrieval time.",
        requirement="GDP audit readiness must show how each release decision links to source evidence, reviewer identity, and applicable procedure.",
        policy="The policy requires audit evidence to be retrievable without exposing unrelated patient or commercial data.",
        decision="The compliance steering group decided that graph responses may support audits only when citations are reviewable and source-backed.",
        example="For instance, an auditor can ask why a shipment was released after a thermal event and receive a cited packet summary.",
        risk="The main risk is a confident answer without a verifiable evidence chain.",
        open_question="Should audit queries default to evidence-only mode until the official graph is reviewed?",
    ),
)


RELATIONS: tuple[dict[str, str], ...] = (
    {"from": "Lane Qualification", "to": "Route Risk Scoring", "type": "depends_on"},
    {"from": "Route Risk Scoring", "to": "Lane Qualification", "type": "informs"},
    {"from": "Temperature Excursion", "to": "Deviation Triage", "type": "triggers"},
    {"from": "Temperature Excursion", "to": "Quality Hold", "type": "triggers"},
    {"from": "Quality Hold", "to": "Release Evidence Packet", "type": "requires"},
    {"from": "Deviation Triage", "to": "CAPA", "type": "may_create"},
    {"from": "CAPA", "to": "Lane Qualification", "type": "can_update"},
    {"from": "Validated Shipper", "to": "Stability Budget", "type": "protects"},
    {"from": "Refrigerant Conditioning", "to": "Validated Shipper", "type": "precondition_for"},
    {"from": "Calibration Drift", "to": "Temperature Excursion", "type": "can_invalidate"},
    {"from": "Sensor Pairing", "to": "Calibration Drift", "type": "detects"},
    {"from": "Chain of Custody", "to": "Release Evidence Packet", "type": "supports"},
    {"from": "FEFO Allocation", "to": "Stability Budget", "type": "must_preserve"},
    {"from": "Quarantine Workflow", "to": "Quality Hold", "type": "implements"},
    {"from": "GDP Audit Readiness", "to": "Release Evidence Packet", "type": "requires"},
    {"from": "GDP Audit Readiness", "to": "Chain of Custody", "type": "requires"},
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the shared Cornerstone synthetic corpus.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "test-data" / "shared-synthetic-corpus",
    )
    args = parser.parse_args()
    generate(args.output_root)
    return 0


def generate(output_root: Path) -> None:
    documents_dir = output_root / "documents"
    output_root.mkdir(parents=True, exist_ok=True)
    documents_dir.mkdir(parents=True, exist_ok=True)

    documents: list[dict[str, Any]] = []
    source_objects: list[dict[str, Any]] = []
    doc_index = 1
    for concept in CONCEPTS:
        for artifact_type, builder in (
            ("sop", _build_sop_document),
            ("decision_record", _build_decision_document),
            ("field_report", _build_field_report_document),
        ):
            doc_id = f"hpcc-v1-{concept.slug}-{artifact_type}"
            filename = f"{doc_index:03d}-{doc_id}.md"
            relative_path = f"documents/{filename}"
            text = builder(concept, doc_id)
            (documents_dir / filename).write_text(text, encoding="utf-8")
            word_count = len(re.findall(r"\b[\w-]+\b", text))
            sentence_count = len(re.findall(r"[^.!?\n]+[.!?]", text))
            visibility = "evidence_only" if artifact_type == "field_report" else "member_visible"
            metadata = {
                "documentId": doc_id,
                "title": _title_for(concept, artifact_type),
                "path": relative_path,
                "artifactType": artifact_type,
                "visibility": visibility,
                "primaryConcept": concept.name,
                "expectedConcepts": _expected_concepts_for(concept),
                "expectedEvidenceTypes": ("definition", "requirement", "policy", "decision", "example", "open_question"),
                "wordCount": word_count,
                "sentenceCount": sentence_count,
                "sourceUrl": f"synthetic://{DATASET_ID}/{doc_id}",
                "sourceUpdatedAt": None,
                "reviewIntent": _review_intent_for(artifact_type),
                "tags": (DOMAIN, concept.category, artifact_type, "synthetic", "not-public-data"),
            }
            documents.append(metadata)
            source_objects.append(
                {
                    "sourceExternalId": doc_id,
                    "title": metadata["title"],
                    "contentPath": relative_path,
                    "sourceUrl": metadata["sourceUrl"],
                    "sourceUpdatedAt": metadata["sourceUpdatedAt"],
                    "sourceObjectType": artifact_type,
                    "providerMetadata": {
                        "provider": "shared_synthetic_corpus",
                        "datasetId": DATASET_ID,
                        "datasetVersion": DATASET_VERSION,
                        "visibility": visibility,
                        "primaryConcept": concept.name,
                        "domain": DOMAIN,
                    },
                }
            )
            doc_index += 1

    glossary = _build_glossary_document()
    glossary_path = "documents/000-hpcc-v1-domain-glossary.md"
    (documents_dir / Path(glossary_path).name).write_text(glossary, encoding="utf-8")
    documents.insert(
        0,
        {
            "documentId": "hpcc-v1-domain-glossary",
            "title": "HelioPharm Cold Chain Domain Glossary",
            "path": glossary_path,
            "artifactType": "glossary",
            "visibility": "member_visible",
            "primaryConcept": "Domain Glossary",
            "expectedConcepts": [concept.name for concept in CONCEPTS],
            "expectedEvidenceTypes": ("definition", "policy", "requirement"),
            "wordCount": len(re.findall(r"\b[\w-]+\b", glossary)),
            "sentenceCount": len(re.findall(r"[^.!?\n]+[.!?]", glossary)),
            "sourceUrl": f"synthetic://{DATASET_ID}/hpcc-v1-domain-glossary",
            "sourceUpdatedAt": None,
            "reviewIntent": "Use as broad concept vocabulary and alias coverage.",
            "tags": (DOMAIN, "glossary", "synthetic", "not-public-data"),
        },
    )
    source_objects.insert(
        0,
        {
            "sourceExternalId": "hpcc-v1-domain-glossary",
            "title": "HelioPharm Cold Chain Domain Glossary",
            "contentPath": glossary_path,
            "sourceUrl": f"synthetic://{DATASET_ID}/hpcc-v1-domain-glossary",
            "sourceUpdatedAt": None,
            "sourceObjectType": "glossary",
            "providerMetadata": {
                "provider": "shared_synthetic_corpus",
                "datasetId": DATASET_ID,
                "datasetVersion": DATASET_VERSION,
                "visibility": "member_visible",
                "domain": DOMAIN,
            },
        },
    )

    manifest = {
        "datasetId": DATASET_ID,
        "version": DATASET_VERSION,
        "generatedAt": FIXED_GENERATED_AT,
        "organization": ORG_NAME,
        "domain": DOMAIN,
        "description": (
            "A synthetic, domain-specific organizational corpus for Cornerstone ingestion, "
            "evidence review, ontology extraction, graph serving, and evaluation tests."
        ),
        "dataPolicy": {
            "syntheticOnly": True,
            "publicDatasetDependency": False,
            "containsRealPeople": False,
            "containsSecrets": False,
            "intendedUse": "Shared project test data and demos; not normative product documentation.",
        },
        "coverage": {
            "documentCount": len(documents),
            "sourceObjectCount": len(source_objects),
            "conceptCount": len(CONCEPTS),
            "expectedRelationCount": len(RELATIONS),
            "artifactTypes": sorted({document["artifactType"] for document in documents}),
            "visibilityModes": sorted({document["visibility"] for document in documents}),
            "totalWords": sum(int(document["wordCount"]) for document in documents),
            "totalSentences": sum(int(document["sentenceCount"]) for document in documents),
        },
        "files": {
            "documentsDirectory": "documents",
            "sourceObjectsJsonl": "source_objects.jsonl",
            "expectedOntology": "expected_ontology.json",
            "evaluationTasks": "evaluation_tasks.json",
        },
        "documents": documents,
    }
    (output_root / "manifest.json").write_text(_json(manifest), encoding="utf-8")

    with (output_root / "source_objects.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for item in source_objects:
            handle.write(json.dumps(item, ensure_ascii=True, sort_keys=True) + "\n")

    (output_root / "expected_ontology.json").write_text(
        _json(
            {
                "datasetId": DATASET_ID,
                "version": DATASET_VERSION,
                "concepts": [
                    {
                        "canonicalName": concept.name,
                        "aliases": list(concept.aliases),
                        "category": concept.category,
                        "owner": concept.owner,
                        "definitionCue": concept.definition,
                    }
                    for concept in CONCEPTS
                ],
                "relations": [
                    {
                        **relation,
                        "expectedEvidenceCue": _relation_evidence_cue(relation),
                        "officializationPolicy": "candidate_only_until_human_review",
                    }
                    for relation in RELATIONS
                ],
                "negativeControls": [
                    {
                        "query": "Does HelioPharm approve room-temperature shipment for frozen gene therapy without quality review?",
                        "expectedTrustLabel": "unsupported",
                        "reason": "The corpus never permits room-temperature release for frozen therapy.",
                    },
                    {
                        "query": "Can billing clear a quality hold after carrier delivery?",
                        "expectedTrustLabel": "unsupported",
                        "reason": "Quality hold policy blocks billing until quality disposition.",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    (output_root / "evaluation_tasks.json").write_text(_json(_evaluation_tasks()), encoding="utf-8")
    (output_root / "README.md").write_text(_readme(manifest), encoding="utf-8")


def _title_for(concept: ConceptProfile, artifact_type: str) -> str:
    if artifact_type == "sop":
        return f"{concept.name} Control SOP"
    if artifact_type == "decision_record":
        return f"{concept.name} Governance Decision Record"
    return f"{concept.name} Field Evidence Report"


def _expected_concepts_for(concept: ConceptProfile) -> list[str]:
    related: list[str] = [concept.name]
    for relation in RELATIONS:
        if relation["from"] == concept.name:
            related.append(relation["to"])
        if relation["to"] == concept.name:
            related.append(relation["from"])
    return sorted(set(related))


def _review_intent_for(artifact_type: str) -> str:
    if artifact_type == "sop":
        return "Review as policy and requirement evidence before officialization."
    if artifact_type == "decision_record":
        return "Review as decision evidence and relation justification."
    return "Review as evidence-only operational signal; do not officialize without corroboration."


def _build_sop_document(concept: ConceptProfile, doc_id: str) -> str:
    return f"""# {concept.name} Control SOP

Document id: {doc_id}.
Dataset: {DATASET_ID}.
Organization: {ORG_NAME}.
Artifact type: SOP.
Visibility: member_visible.

## Purpose
{concept.definition}
This SOP is synthetic and was created for Cornerstone shared project testing.
The owner of {concept.name} is {concept.owner}.
The process is part of {DOMAIN}.
The intended evidence state is reviewable and source-backed.

## Control Policy
{concept.policy}
{concept.requirement}
The procedure must preserve source evidence instead of relying on copied summaries.
The reviewer must be able to identify the source document, source object id, and quote range.
The control owner shall update the decision record when the procedure changes.
The control owner shall mark obsolete guidance as superseded instead of deleting it.

## Operational Criteria
{concept.metric}
The daily operating review must compare the metric with the latest shipment evidence.
The weekly quality review must inspect exceptions that affect patient delivery commitments.
The monthly governance review must identify repeated patterns and assign accountable actions.
If the evidence is incomplete, the answer must remain evidence-only or unsupported.
If the evidence is conflicted, the official graph must not hide the conflict.

## Expert Notes
{concept.example}
{concept.risk}
The system should treat a strong narrative without citations as insufficient evidence.
The system should prefer reviewed evidence over stale or unreviewed operational notes.
The system should preserve distinction between a policy, a requirement, a decision, an example, and an open question.
The process is intentionally domain-specific and does not depend on any public benchmark dataset.

## Review Questions
{concept.open_question}
Can the reviewer trace every official claim about {concept.name} to a concrete source artifact?
Should this SOP create a candidate relation in the ontology graph?
"""


def _build_decision_document(concept: ConceptProfile, doc_id: str) -> str:
    related = _expected_concepts_for(concept)
    primary_related = next((item for item in related if item != concept.name), "Release Evidence Packet")
    return f"""# {concept.name} Governance Decision Record

Document id: {doc_id}.
Dataset: {DATASET_ID}.
Organization: {ORG_NAME}.
Artifact type: decision_record.
Visibility: member_visible.

## Context
{concept.definition}
The governance group reviewed the effect of {concept.name} on {primary_related}.
The decision was recorded to test Cornerstone decision provenance and official graph review.
The group treated this record as synthetic but operationally realistic.

## Decision
{concept.decision}
The decision must be linked to evidence fragments from the SOP and field report before officialization.
The decision must not overwrite contrary evidence from unresolved incidents.
The decision shall create a candidate relation only when source evidence names both concepts.
The decision owner is {concept.owner}.

## Rationale
{concept.metric}
{concept.policy}
{concept.requirement}
The rationale is that specialty pharmacy logistics needs auditable quality controls before product release.
The rationale is also that patient scheduling can be harmed by confident but unsupported operational answers.
The expected graph behavior is candidate-only until a human reviewer approves the concept or relation.

## Consequences
{concept.example}
{concept.risk}
If future evidence contradicts this record, the graph should show conflicted or partially supported state.
If a downstream integration asks for candidates, the response must reject candidate bypass unless explicitly allowed by a reviewed contract.
If the record is superseded, the new record must preserve the old source link and decision context.

## Open Questions
{concept.open_question}
Should {concept.name} be visible in customer-facing answers or restricted to internal quality review?
"""


def _build_field_report_document(concept: ConceptProfile, doc_id: str) -> str:
    return f"""# {concept.name} Field Evidence Report

Document id: {doc_id}.
Dataset: {DATASET_ID}.
Organization: {ORG_NAME}.
Artifact type: field_report.
Visibility: evidence_only.

## Situation
A synthetic operations team observed a recurring issue related to {concept.name}.
{concept.definition}
The field report is evidence-only because it may include local observations that need reviewer confirmation.
The report should support extraction tests without becoming official truth automatically.

## Observed Evidence
{concept.example}
{concept.metric}
The observed event affected a shipment planning review, a quality review, and a release readiness discussion.
The field team recorded timestamps, actor roles, source system references, and unresolved assumptions.
The field team did not record real customer names, patient identifiers, or production secrets.

## Required Follow Up
{concept.requirement}
{concept.policy}
The reviewer must compare this field evidence with the governance decision record.
The reviewer must decide whether the evidence supports a new concept, a relation, or only an operational note.
The reviewer must reject any claim that is not supported by the source text.

## Risk and Example
{concept.risk}
{concept.example}
The example is intentionally repeated in another artifact so tests can verify duplicate evidence handling.
The report can create useful candidate evidence but should not directly mutate the official graph.
The report is useful for testing stale, conflicted, and evidence-only answer behavior.

## Open Question
{concept.open_question}
Should this field report remain evidence-only after a decision record cites it?
"""


def _build_glossary_document() -> str:
    lines = [
        "# HelioPharm Cold Chain Domain Glossary",
        "",
        f"Document id: hpcc-v1-domain-glossary.",
        f"Dataset: {DATASET_ID}.",
        f"Organization: {ORG_NAME}.",
        "Artifact type: glossary.",
        "Visibility: member_visible.",
        "",
        "## Scope",
        f"This glossary defines the synthetic {DOMAIN} vocabulary used across the shared test corpus.",
        "The glossary is designed for Cornerstone evidence extraction, review queues, ontology extraction, graph serving, and evaluation tests.",
        "The glossary is not copied from a public dataset and does not describe a real company.",
        "Every term must remain source-backed before it becomes official graph knowledge.",
        "",
        "## Terms",
    ]
    for concept in CONCEPTS:
        alias_text = ", ".join(concept.aliases)
        lines.extend(
            [
                f"### {concept.name}",
                concept.definition,
                f"Aliases for {concept.name} are {alias_text}.",
                concept.requirement,
                concept.policy,
                concept.open_question,
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def _relation_evidence_cue(relation: dict[str, str]) -> str:
    return f"{relation['from']} {relation['type'].replace('_', ' ')} {relation['to']}."


def _evaluation_tasks() -> dict[str, Any]:
    tasks = [
        {
            "taskId": "hpcc-eval-quality-hold-release",
            "query": "Why does a shipment with uncertain stability budget enter quality hold?",
            "expectedTrustLabel": "official_after_review",
            "expectedAnswerFragments": ["stability budget uncertainty", "quality hold", "Product Quality signs"],
            "requiredConcepts": ["Stability Budget", "Quality Hold", "Release Evidence Packet"],
            "requiredRelations": ["Temperature Excursion->Quality Hold", "Quality Hold->Release Evidence Packet"],
        },
        {
            "taskId": "hpcc-eval-lane-risk",
            "query": "How should lane qualification use route risk scoring?",
            "expectedTrustLabel": "official_after_review",
            "expectedAnswerFragments": ["thermal margin", "carrier reliability", "qualification packet"],
            "requiredConcepts": ["Lane Qualification", "Route Risk Scoring"],
            "requiredRelations": ["Lane Qualification->Route Risk Scoring", "Route Risk Scoring->Lane Qualification"],
        },
        {
            "taskId": "hpcc-eval-custody-audit",
            "query": "What evidence is needed for GDP audit readiness after a thermal event?",
            "expectedTrustLabel": "official_after_review",
            "expectedAnswerFragments": ["custody evidence", "reviewer identity", "source evidence"],
            "requiredConcepts": ["GDP Audit Readiness", "Chain of Custody", "Release Evidence Packet"],
            "requiredRelations": ["GDP Audit Readiness->Release Evidence Packet", "GDP Audit Readiness->Chain of Custody"],
        },
        {
            "taskId": "hpcc-eval-negative-billing-hold",
            "query": "Can billing clear a quality hold after carrier delivery?",
            "expectedTrustLabel": "unsupported",
            "expectedAnswerFragments": ["must not be released", "billing systems"],
            "requiredConcepts": [],
            "requiredRelations": [],
        },
    ]
    return {
        "datasetId": DATASET_ID,
        "version": DATASET_VERSION,
        "tasks": tasks,
    }


def _readme(manifest: dict[str, Any]) -> str:
    coverage = manifest["coverage"]
    return f"""# Shared Synthetic Corpus

This directory contains `{DATASET_ID}` version `{DATASET_VERSION}`.

It is a fully synthetic organizational dataset for {ORG_NAME}, a fictional team operating in {DOMAIN}.
It was created for project-wide Cornerstone testing and demos.
It does not rely on public benchmark datasets, real customers, real employees, real shipment ids, real Notion pages, or production secrets.

## Coverage

- Documents: {coverage["documentCount"]}
- Source objects: {coverage["sourceObjectCount"]}
- Concepts: {coverage["conceptCount"]}
- Expected relations: {coverage["expectedRelationCount"]}
- Total words: {coverage["totalWords"]}
- Total sentences: {coverage["totalSentences"]}
- Artifact types: {", ".join(coverage["artifactTypes"])}
- Visibility modes: {", ".join(coverage["visibilityModes"])}

## Files

- `manifest.json`: dataset inventory, metadata, document paths, and expected review intent.
- `source_objects.jsonl`: source-object descriptors that can be loaded by ingestion helpers.
- `expected_ontology.json`: expected concept and relation coverage for ontology extraction and graph tests.
- `evaluation_tasks.json`: reusable query/evaluation seeds for grounded context and graph tests.
- `documents/*.md`: synthetic source documents.

## Intended Uses

Use this corpus for manual upload tests, connector-normalized ingestion tests, evidence review queues, ontology candidate extraction, graph visualization, release-readiness demos, and external integration contract examples.

Normal tests should load a subset when speed matters.
Full project or smoke tests can load all source objects when they need representative volume.
"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
