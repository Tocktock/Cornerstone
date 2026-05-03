# Route Risk Scoring Governance Decision Record

Document id: hpcc-v1-route-risk-scoring-decision_record.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: decision_record.
Visibility: member_visible.

## Context
Route Risk Scoring means the calculation that estimates product quality risk across route, weather, customs, carrier, and contingency factors.
The governance group reviewed the effect of Route Risk Scoring on Lane Qualification.
The decision was recorded to test Cornerstone decision provenance and official graph review.
The group treated this record as synthetic but operationally realistic.

## Decision
The network planning group decided that weekend border crossings add risk unless a named customs broker is on call.
The decision must be linked to evidence fragments from the SOP and field report before officialization.
The decision must not overwrite contrary evidence from unresolved incidents.
The decision shall create a candidate relation only when source evidence names both concepts.
The decision owner is Network Planning.

## Rationale
The score uses thermal margin, dwell exposure, carrier reliability, contingency depot access, and seasonal volatility.
The policy requires score inputs to be auditable and reproducible from source evidence.
A high route risk score must trigger either enhanced monitoring, a stronger shipper, or a quality-approved exception.
The rationale is that specialty pharmacy logistics needs auditable quality controls before product release.
The rationale is also that patient scheduling can be harmed by confident but unsupported operational answers.
The expected graph behavior is candidate-only until a human reviewer approves the concept or relation.

## Consequences
For instance, a short route can have high risk when airport dwell occurs during a heat advisory.
The main risk is allowing planners to override risk factors without a decision record.
If future evidence contradicts this record, the graph should show conflicted or partially supported state.
If a downstream integration asks for candidates, the response must reject candidate bypass unless explicitly allowed by a reviewed contract.
If the record is superseded, the new record must preserve the old source link and decision context.

## Open Questions
Should route risk scoring be recalculated after every deviation or only at lane review?
Should Route Risk Scoring be visible in customer-facing answers or restricted to internal quality review?
