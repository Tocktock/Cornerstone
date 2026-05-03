# Validated Shipper Governance Decision Record

Document id: hpcc-v1-validated-shipper-decision_record.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: decision_record.
Visibility: member_visible.

## Context
Validated Shipper means a packaging configuration proven to maintain the target temperature range for a defined duration and payload profile.
The governance group reviewed the effect of Validated Shipper on Refrigerant Conditioning.
The decision was recorded to test Cornerstone decision provenance and official graph review.
The group treated this record as synthetic but operationally realistic.

## Decision
The packaging board decided that passive shippers may not be used on high humidity customs lanes without extra condensation checks.
The decision must be linked to evidence fragments from the SOP and field report before officialization.
The decision must not overwrite contrary evidence from unresolved incidents.
The decision shall create a candidate relation only when source evidence names both concepts.
The decision owner is Packaging Engineering.

## Rationale
Shipper suitability is measured by qualified duration, payload mass, refrigerant conditioning state, and lane risk margin.
The policy requires a new validation or engineering assessment after any insulation material, coolant, or payload bracket changes.
A validated shipper must match product temperature class, planned lane duration, and contingency hold time.
The rationale is that specialty pharmacy logistics needs auditable quality controls before product release.
The rationale is also that patient scheduling can be harmed by confident but unsupported operational answers.
The expected graph behavior is candidate-only until a human reviewer approves the concept or relation.

## Consequences
For instance, a 96 hour shipper can fail a 42 hour route when the payload mass is below the validated bracket.
The main risk is treating nameplate duration as universal instead of validation-scope-specific.
If future evidence contradicts this record, the graph should show conflicted or partially supported state.
If a downstream integration asks for candidates, the response must reject candidate bypass unless explicitly allowed by a reviewed contract.
If the record is superseded, the new record must preserve the old source link and decision context.

## Open Questions
Should shipper validation results be modeled as first-class evidence nodes?
Should Validated Shipper be visible in customer-facing answers or restricted to internal quality review?
