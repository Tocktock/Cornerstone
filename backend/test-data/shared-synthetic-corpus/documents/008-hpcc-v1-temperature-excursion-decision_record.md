# Temperature Excursion Governance Decision Record

Document id: hpcc-v1-temperature-excursion-decision_record.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: decision_record.
Visibility: member_visible.

## Context
Temperature Excursion means an observed or inferred product exposure outside the approved temperature range for a material duration.
The governance group reviewed the effect of Temperature Excursion on Calibration Drift.
The decision was recorded to test Cornerstone decision provenance and official graph review.
The group treated this record as synthetic but operationally realistic.

## Decision
The incident review group decided that inferred excursions from missing sensor intervals require the same triage path as observed excursions.
The decision must be linked to evidence fragments from the SOP and field report before officialization.
The decision must not overwrite contrary evidence from unresolved incidents.
The decision shall create a candidate relation only when source evidence names both concepts.
The decision owner is Quality Operations.

## Rationale
Excursion severity is scored by peak deviation, duration, sensor confidence, lane state, and remaining stability budget.
The policy requires raw sensor files to be retained with the release evidence packet for every excursion.
A temperature excursion must create a deviation triage record before the shipment can be released or discarded.
The rationale is that specialty pharmacy logistics needs auditable quality controls before product release.
The rationale is also that patient scheduling can be harmed by confident but unsupported operational answers.
The expected graph behavior is candidate-only until a human reviewer approves the concept or relation.

## Consequences
For instance, a gap between gateway pings can indicate a possible excursion if the shipper was on an airport tarmac during high heat.
The main risk is treating the carrier delivered timestamp as proof that no excursion occurred.
If future evidence contradicts this record, the graph should show conflicted or partially supported state.
If a downstream integration asks for candidates, the response must reject candidate bypass unless explicitly allowed by a reviewed contract.
If the record is superseded, the new record must preserve the old source link and decision context.

## Open Questions
Should inferred excursions be visible to customers before quality review is complete?
Should Temperature Excursion be visible in customer-facing answers or restricted to internal quality review?
