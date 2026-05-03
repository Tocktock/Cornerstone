# Calibration Drift Governance Decision Record

Document id: hpcc-v1-calibration-drift-decision_record.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: decision_record.
Visibility: member_visible.

## Context
Calibration Drift means movement of a measurement device away from accepted tolerance between calibration events.
The governance group reviewed the effect of Calibration Drift on Sensor Pairing.
The decision was recorded to test Cornerstone decision provenance and official graph review.
The group treated this record as synthetic but operationally realistic.

## Decision
The metrology board decided that post-use calibration failure reopens any release decision that depended on that device.
The decision must be linked to evidence fragments from the SOP and field report before officialization.
The decision must not overwrite contrary evidence from unresolved incidents.
The decision shall create a candidate relation only when source evidence names both concepts.
The decision owner is Metrology.

## Rationale
Drift risk is measured by days since calibration, observed variance, device family history, and excursion decision dependency.
The policy requires paired sensor comparison when drift affects a shipment with limited stability budget.
A sensor with unresolved calibration drift must not be used as the sole evidence for product release.
The rationale is that specialty pharmacy logistics needs auditable quality controls before product release.
The rationale is also that patient scheduling can be harmed by confident but unsupported operational answers.
The expected graph behavior is candidate-only until a human reviewer approves the concept or relation.

## Consequences
For instance, a logger reading 7.8 C may not prove compliance if its paired sensor reads 8.4 C and drift is unresolved.
The main risk is using a single sensor trace without checking calibration certificate status.
If future evidence contradicts this record, the graph should show conflicted or partially supported state.
If a downstream integration asks for candidates, the response must reject candidate bypass unless explicitly allowed by a reviewed contract.
If the record is superseded, the new record must preserve the old source link and decision context.

## Open Questions
Should calibration drift automatically downgrade evidence freshness or trust state?
Should Calibration Drift be visible in customer-facing answers or restricted to internal quality review?
