# Validated Shipper Control SOP

Document id: hpcc-v1-validated-shipper-sop.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: SOP.
Visibility: member_visible.

## Purpose
Validated Shipper means a packaging configuration proven to maintain the target temperature range for a defined duration and payload profile.
This SOP is synthetic and was created for Cornerstone shared project testing.
The owner of Validated Shipper is Packaging Engineering.
The process is part of temperature-controlled specialty pharmacy logistics.
The intended evidence state is reviewable and source-backed.

## Control Policy
The policy requires a new validation or engineering assessment after any insulation material, coolant, or payload bracket changes.
A validated shipper must match product temperature class, planned lane duration, and contingency hold time.
The procedure must preserve source evidence instead of relying on copied summaries.
The reviewer must be able to identify the source document, source object id, and quote range.
The control owner shall update the decision record when the procedure changes.
The control owner shall mark obsolete guidance as superseded instead of deleting it.

## Operational Criteria
Shipper suitability is measured by qualified duration, payload mass, refrigerant conditioning state, and lane risk margin.
The daily operating review must compare the metric with the latest shipment evidence.
The weekly quality review must inspect exceptions that affect patient delivery commitments.
The monthly governance review must identify repeated patterns and assign accountable actions.
If the evidence is incomplete, the answer must remain evidence-only or unsupported.
If the evidence is conflicted, the official graph must not hide the conflict.

## Expert Notes
For instance, a 96 hour shipper can fail a 42 hour route when the payload mass is below the validated bracket.
The main risk is treating nameplate duration as universal instead of validation-scope-specific.
The system should treat a strong narrative without citations as insufficient evidence.
The system should prefer reviewed evidence over stale or unreviewed operational notes.
The system should preserve distinction between a policy, a requirement, a decision, an example, and an open question.
The process is intentionally domain-specific and does not depend on any public benchmark dataset.

## Review Questions
Should shipper validation results be modeled as first-class evidence nodes?
Can the reviewer trace every official claim about Validated Shipper to a concrete source artifact?
Should this SOP create a candidate relation in the ontology graph?
