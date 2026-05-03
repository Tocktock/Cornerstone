# Sensor Pairing Governance Decision Record

Document id: hpcc-v1-sensor-pairing-decision_record.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: decision_record.
Visibility: member_visible.

## Context
Sensor Pairing means assigning two compatible measurement devices to a shipment so readings can validate each other and expose drift or placement error.
The governance group reviewed the effect of Sensor Pairing on Calibration Drift.
The decision was recorded to test Cornerstone decision provenance and official graph review.
The group treated this record as synthetic but operationally realistic.

## Decision
The metrology team decided that sensor pairing is mandatory for pediatric oncology biologic lanes.
The decision must be linked to evidence fragments from the SOP and field report before officialization.
The decision must not overwrite contrary evidence from unresolved incidents.
The decision shall create a candidate relation only when source evidence names both concepts.
The decision owner is Metrology.

## Rationale
Pairing reliability is measured by device compatibility, placement separation, time synchronization, and drift agreement.
The policy requires paired sensors to be time-synchronized before pack-out and reconciled after receipt.
Sensor pairing must be used when shipment value, stability budget, or regulatory category requires redundant evidence.
The rationale is that specialty pharmacy logistics needs auditable quality controls before product release.
The rationale is also that patient scheduling can be harmed by confident but unsupported operational answers.
The expected graph behavior is candidate-only until a human reviewer approves the concept or relation.

## Consequences
For instance, a top-cavity logger and payload-core logger can reveal a local placement artifact.
The main risk is pairing two devices from the same suspect calibration batch.
If future evidence contradicts this record, the graph should show conflicted or partially supported state.
If a downstream integration asks for candidates, the response must reject candidate bypass unless explicitly allowed by a reviewed contract.
If the record is superseded, the new record must preserve the old source link and decision context.

## Open Questions
Should paired sensor disagreement create automatic deviation triage?
Should Sensor Pairing be visible in customer-facing answers or restricted to internal quality review?
