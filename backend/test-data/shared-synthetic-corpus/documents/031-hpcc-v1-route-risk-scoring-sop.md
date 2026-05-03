# Route Risk Scoring Control SOP

Document id: hpcc-v1-route-risk-scoring-sop.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: SOP.
Visibility: member_visible.

## Purpose
Route Risk Scoring means the calculation that estimates product quality risk across route, weather, customs, carrier, and contingency factors.
This SOP is synthetic and was created for Cornerstone shared project testing.
The owner of Route Risk Scoring is Network Planning.
The process is part of temperature-controlled specialty pharmacy logistics.
The intended evidence state is reviewable and source-backed.

## Control Policy
The policy requires score inputs to be auditable and reproducible from source evidence.
A high route risk score must trigger either enhanced monitoring, a stronger shipper, or a quality-approved exception.
The procedure must preserve source evidence instead of relying on copied summaries.
The reviewer must be able to identify the source document, source object id, and quote range.
The control owner shall update the decision record when the procedure changes.
The control owner shall mark obsolete guidance as superseded instead of deleting it.

## Operational Criteria
The score uses thermal margin, dwell exposure, carrier reliability, contingency depot access, and seasonal volatility.
The daily operating review must compare the metric with the latest shipment evidence.
The weekly quality review must inspect exceptions that affect patient delivery commitments.
The monthly governance review must identify repeated patterns and assign accountable actions.
If the evidence is incomplete, the answer must remain evidence-only or unsupported.
If the evidence is conflicted, the official graph must not hide the conflict.

## Expert Notes
For instance, a short route can have high risk when airport dwell occurs during a heat advisory.
The main risk is allowing planners to override risk factors without a decision record.
The system should treat a strong narrative without citations as insufficient evidence.
The system should prefer reviewed evidence over stale or unreviewed operational notes.
The system should preserve distinction between a policy, a requirement, a decision, an example, and an open question.
The process is intentionally domain-specific and does not depend on any public benchmark dataset.

## Review Questions
Should route risk scoring be recalculated after every deviation or only at lane review?
Can the reviewer trace every official claim about Route Risk Scoring to a concrete source artifact?
Should this SOP create a candidate relation in the ontology graph?
