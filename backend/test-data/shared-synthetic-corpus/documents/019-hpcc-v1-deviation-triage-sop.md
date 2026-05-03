# Deviation Triage Control SOP

Document id: hpcc-v1-deviation-triage-sop.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: SOP.
Visibility: member_visible.

## Purpose
Deviation Triage means the first structured assessment that classifies a quality event, assigns ownership, and selects the disposition path.
This SOP is synthetic and was created for Cornerstone shared project testing.
The owner of Deviation Triage is Quality Operations.
The process is part of temperature-controlled specialty pharmacy logistics.
The intended evidence state is reviewable and source-backed.

## Control Policy
The policy requires triage notes to separate observed facts from inferred hypotheses.
Deviation triage must identify whether the event affects product quality, patient scheduling, regulatory reporting, or only operational performance.
The procedure must preserve source evidence instead of relying on copied summaries.
The reviewer must be able to identify the source document, source object id, and quote range.
The control owner shall update the decision record when the procedure changes.
The control owner shall mark obsolete guidance as superseded instead of deleting it.

## Operational Criteria
Triage quality is measured by classification accuracy, evidence completeness, owner assignment, and time to first disposition.
The daily operating review must compare the metric with the latest shipment evidence.
The weekly quality review must inspect exceptions that affect patient delivery commitments.
The monthly governance review must identify repeated patterns and assign accountable actions.
If the evidence is incomplete, the answer must remain evidence-only or unsupported.
If the evidence is conflicted, the official graph must not hide the conflict.

## Expert Notes
For instance, a missing receiver signature is paperwork-only when custody is otherwise proven by geofence and device evidence.
The main risk is closing a deviation before stability budget and custody evidence are reviewed together.
The system should treat a strong narrative without citations as insufficient evidence.
The system should prefer reviewed evidence over stale or unreviewed operational notes.
The system should preserve distinction between a policy, a requirement, a decision, an example, and an open question.
The process is intentionally domain-specific and does not depend on any public benchmark dataset.

## Review Questions
Should triage confidence be computed by rule or assigned by the reviewer?
Can the reviewer trace every official claim about Deviation Triage to a concrete source artifact?
Should this SOP create a candidate relation in the ontology graph?
