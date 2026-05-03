# Temperature Excursion Field Evidence Report

Document id: hpcc-v1-temperature-excursion-field_report.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: field_report.
Visibility: evidence_only.

## Situation
A synthetic operations team observed a recurring issue related to Temperature Excursion.
Temperature Excursion means an observed or inferred product exposure outside the approved temperature range for a material duration.
The field report is evidence-only because it may include local observations that need reviewer confirmation.
The report should support extraction tests without becoming official truth automatically.

## Observed Evidence
For instance, a gap between gateway pings can indicate a possible excursion if the shipper was on an airport tarmac during high heat.
Excursion severity is scored by peak deviation, duration, sensor confidence, lane state, and remaining stability budget.
The observed event affected a shipment planning review, a quality review, and a release readiness discussion.
The field team recorded timestamps, actor roles, source system references, and unresolved assumptions.
The field team did not record real customer names, patient identifiers, or production secrets.

## Required Follow Up
A temperature excursion must create a deviation triage record before the shipment can be released or discarded.
The policy requires raw sensor files to be retained with the release evidence packet for every excursion.
The reviewer must compare this field evidence with the governance decision record.
The reviewer must decide whether the evidence supports a new concept, a relation, or only an operational note.
The reviewer must reject any claim that is not supported by the source text.

## Risk and Example
The main risk is treating the carrier delivered timestamp as proof that no excursion occurred.
For instance, a gap between gateway pings can indicate a possible excursion if the shipper was on an airport tarmac during high heat.
The example is intentionally repeated in another artifact so tests can verify duplicate evidence handling.
The report can create useful candidate evidence but should not directly mutate the official graph.
The report is useful for testing stale, conflicted, and evidence-only answer behavior.

## Open Question
Should inferred excursions be visible to customers before quality review is complete?
Should this field report remain evidence-only after a decision record cites it?
