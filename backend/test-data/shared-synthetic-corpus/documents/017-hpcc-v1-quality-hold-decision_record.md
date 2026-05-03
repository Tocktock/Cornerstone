# Quality Hold Governance Decision Record

Document id: hpcc-v1-quality-hold-decision_record.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: decision_record.
Visibility: member_visible.

## Context
Quality Hold means a controlled state that prevents product release until quality evidence is reviewed and disposition is recorded.
The governance group reviewed the effect of Quality Hold on Quarantine Workflow.
The decision was recorded to test Cornerstone decision provenance and official graph review.
The group treated this record as synthetic but operationally realistic.

## Decision
The release board decided that stability budget uncertainty always creates a quality hold until Product Quality signs the calculation.
The decision must be linked to evidence fragments from the SOP and field report before officialization.
The decision must not overwrite contrary evidence from unresolved incidents.
The decision shall create a candidate relation only when source evidence names both concepts.
The decision owner is Quality Assurance.

## Rationale
Hold aging is measured by elapsed hours, missing evidence count, patient impact tier, and next required reviewer.
The policy requires quality hold reason codes to reference the triggering evidence and the required disposition path.
A shipment under quality hold must not be released to inventory, patient scheduling, or billing systems.
The rationale is that specialty pharmacy logistics needs auditable quality controls before product release.
The rationale is also that patient scheduling can be harmed by confident but unsupported operational answers.
The expected graph behavior is candidate-only until a human reviewer approves the concept or relation.

## Consequences
For instance, a custody gap and a sensor gap can share one quality hold when they affect the same physical shipment.
The main risk is a downstream system clearing the hold because carrier delivery is complete.
If future evidence contradicts this record, the graph should show conflicted or partially supported state.
If a downstream integration asks for candidates, the response must reject candidate bypass unless explicitly allowed by a reviewed contract.
If the record is superseded, the new record must preserve the old source link and decision context.

## Open Questions
Should quality hold state expire if no reviewer acts within the service-level target?
Should Quality Hold be visible in customer-facing answers or restricted to internal quality review?
