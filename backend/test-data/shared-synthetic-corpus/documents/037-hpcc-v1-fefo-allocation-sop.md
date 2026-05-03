# FEFO Allocation Control SOP

Document id: hpcc-v1-fefo-allocation-sop.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: SOP.
Visibility: member_visible.

## Purpose
FEFO Allocation means selecting inventory by earliest usable expiry while respecting quality state, route feasibility, and patient scheduling constraints.
This SOP is synthetic and was created for Cornerstone shared project testing.
The owner of FEFO Allocation is Inventory Operations.
The process is part of temperature-controlled specialty pharmacy logistics.
The intended evidence state is reviewable and source-backed.

## Control Policy
The policy requires allocation logic to treat release eligibility as stronger than expiry priority.
FEFO allocation must exclude units on quality hold and units without enough stability budget for the planned lane.
The procedure must preserve source evidence instead of relying on copied summaries.
The reviewer must be able to identify the source document, source object id, and quote range.
The control owner shall update the decision record when the procedure changes.
The control owner shall mark obsolete guidance as superseded instead of deleting it.

## Operational Criteria
FEFO effectiveness is measured by expiry waste, release eligibility, patient promise adherence, and avoidable exception rate.
The daily operating review must compare the metric with the latest shipment evidence.
The weekly quality review must inspect exceptions that affect patient delivery commitments.
The monthly governance review must identify repeated patterns and assign accountable actions.
If the evidence is incomplete, the answer must remain evidence-only or unsupported.
If the evidence is conflicted, the official graph must not hide the conflict.

## Expert Notes
For instance, the earliest expiring lot cannot be allocated if its release evidence packet is incomplete.
The main risk is optimizing expiry while ignoring route risk and quality hold state.
The system should treat a strong narrative without citations as insufficient evidence.
The system should prefer reviewed evidence over stale or unreviewed operational notes.
The system should preserve distinction between a policy, a requirement, a decision, an example, and an open question.
The process is intentionally domain-specific and does not depend on any public benchmark dataset.

## Review Questions
Should FEFO allocation consume graph answers directly or only read materialized release state?
Can the reviewer trace every official claim about FEFO Allocation to a concrete source artifact?
Should this SOP create a candidate relation in the ontology graph?
