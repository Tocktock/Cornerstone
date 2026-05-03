# Temperature Excursion Control SOP

Document id: hpcc-v1-temperature-excursion-sop.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: SOP.
Visibility: member_visible.

## Purpose
Temperature Excursion means an observed or inferred product exposure outside the approved temperature range for a material duration.
This SOP is synthetic and was created for Cornerstone shared project testing.
The owner of Temperature Excursion is Quality Operations.
The process is part of temperature-controlled specialty pharmacy logistics.
The intended evidence state is reviewable and source-backed.

## Control Policy
The policy requires raw sensor files to be retained with the release evidence packet for every excursion.
A temperature excursion must create a deviation triage record before the shipment can be released or discarded.
The procedure must preserve source evidence instead of relying on copied summaries.
The reviewer must be able to identify the source document, source object id, and quote range.
The control owner shall update the decision record when the procedure changes.
The control owner shall mark obsolete guidance as superseded instead of deleting it.

## Operational Criteria
Excursion severity is scored by peak deviation, duration, sensor confidence, lane state, and remaining stability budget.
The daily operating review must compare the metric with the latest shipment evidence.
The weekly quality review must inspect exceptions that affect patient delivery commitments.
The monthly governance review must identify repeated patterns and assign accountable actions.
If the evidence is incomplete, the answer must remain evidence-only or unsupported.
If the evidence is conflicted, the official graph must not hide the conflict.

## Expert Notes
For instance, a gap between gateway pings can indicate a possible excursion if the shipper was on an airport tarmac during high heat.
The main risk is treating the carrier delivered timestamp as proof that no excursion occurred.
The system should treat a strong narrative without citations as insufficient evidence.
The system should prefer reviewed evidence over stale or unreviewed operational notes.
The system should preserve distinction between a policy, a requirement, a decision, an example, and an open question.
The process is intentionally domain-specific and does not depend on any public benchmark dataset.

## Review Questions
Should inferred excursions be visible to customers before quality review is complete?
Can the reviewer trace every official claim about Temperature Excursion to a concrete source artifact?
Should this SOP create a candidate relation in the ontology graph?
