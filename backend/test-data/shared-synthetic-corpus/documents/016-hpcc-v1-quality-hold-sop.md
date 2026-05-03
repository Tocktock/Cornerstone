# Quality Hold Control SOP

Document id: hpcc-v1-quality-hold-sop.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: SOP.
Visibility: member_visible.

## Purpose
Quality Hold means a controlled state that prevents product release until quality evidence is reviewed and disposition is recorded.
This SOP is synthetic and was created for Cornerstone shared project testing.
The owner of Quality Hold is Quality Assurance.
The process is part of temperature-controlled specialty pharmacy logistics.
The intended evidence state is reviewable and source-backed.

## Control Policy
The policy requires quality hold reason codes to reference the triggering evidence and the required disposition path.
A shipment under quality hold must not be released to inventory, patient scheduling, or billing systems.
The procedure must preserve source evidence instead of relying on copied summaries.
The reviewer must be able to identify the source document, source object id, and quote range.
The control owner shall update the decision record when the procedure changes.
The control owner shall mark obsolete guidance as superseded instead of deleting it.

## Operational Criteria
Hold aging is measured by elapsed hours, missing evidence count, patient impact tier, and next required reviewer.
The daily operating review must compare the metric with the latest shipment evidence.
The weekly quality review must inspect exceptions that affect patient delivery commitments.
The monthly governance review must identify repeated patterns and assign accountable actions.
If the evidence is incomplete, the answer must remain evidence-only or unsupported.
If the evidence is conflicted, the official graph must not hide the conflict.

## Expert Notes
For instance, a custody gap and a sensor gap can share one quality hold when they affect the same physical shipment.
The main risk is a downstream system clearing the hold because carrier delivery is complete.
The system should treat a strong narrative without citations as insufficient evidence.
The system should prefer reviewed evidence over stale or unreviewed operational notes.
The system should preserve distinction between a policy, a requirement, a decision, an example, and an open question.
The process is intentionally domain-specific and does not depend on any public benchmark dataset.

## Review Questions
Should quality hold state expire if no reviewer acts within the service-level target?
Can the reviewer trace every official claim about Quality Hold to a concrete source artifact?
Should this SOP create a candidate relation in the ontology graph?
