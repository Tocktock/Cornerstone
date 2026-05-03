# Stability Budget Control SOP

Document id: hpcc-v1-stability-budget-sop.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: SOP.
Visibility: member_visible.

## Purpose
Stability Budget means the remaining approved exposure allowance for a product lot after considering temperature, duration, and product-specific stability evidence.
This SOP is synthetic and was created for Cornerstone shared project testing.
The owner of Stability Budget is Product Quality.
The process is part of temperature-controlled specialty pharmacy logistics.
The intended evidence state is reviewable and source-backed.

## Control Policy
The policy requires budget calculation to use the product stability memo, not the shipment average temperature.
A release decision must preserve a nonnegative stability budget for every shipped unit and every documented handoff.
The procedure must preserve source evidence instead of relying on copied summaries.
The reviewer must be able to identify the source document, source object id, and quote range.
The control owner shall update the decision record when the procedure changes.
The control owner shall mark obsolete guidance as superseded instead of deleting it.

## Operational Criteria
The budget is tracked in minutes by temperature band and is reduced only by reviewed excursion evidence.
The daily operating review must compare the metric with the latest shipment evidence.
The weekly quality review must inspect exceptions that affect patient delivery commitments.
The monthly governance review must identify repeated patterns and assign accountable actions.
If the evidence is incomplete, the answer must remain evidence-only or unsupported.
If the evidence is conflicted, the official graph must not hide the conflict.

## Expert Notes
For instance, a ten minute 8.7 C event may consume more budget than a thirty minute 7.9 C event for a narrow 2-8 C biologic.
The main risk is aggregating sensor readings before applying the product-specific temperature band.
The system should treat a strong narrative without citations as insufficient evidence.
The system should prefer reviewed evidence over stale or unreviewed operational notes.
The system should preserve distinction between a policy, a requirement, a decision, an example, and an open question.
The process is intentionally domain-specific and does not depend on any public benchmark dataset.

## Review Questions
Should the stability budget service expose remaining budget as a graph edge attribute?
Can the reviewer trace every official claim about Stability Budget to a concrete source artifact?
Should this SOP create a candidate relation in the ontology graph?
