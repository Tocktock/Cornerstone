# HelioPharm Cold Chain Domain Glossary

Document id: hpcc-v1-domain-glossary.
Dataset: cornerstone-shared-synthetic-corpus-v1.
Organization: HelioPharm Cold Chain Operations.
Artifact type: glossary.
Visibility: member_visible.

## Scope
This glossary defines the synthetic temperature-controlled specialty pharmacy logistics vocabulary used across the shared test corpus.
The glossary is designed for Cornerstone evidence extraction, review queues, ontology extraction, graph serving, and evaluation tests.
The glossary is not copied from a public dataset and does not describe a real company.
Every term must remain source-backed before it becomes official graph knowledge.

## Terms
### Lane Qualification
Lane Qualification means a documented approval that a transport lane can protect product quality under expected seasonal, carrier, customs, and handoff conditions.
Aliases for Lane Qualification are qualified lane, route qualification, lane validation.
A lane must not be released for commercial biologic movement until a qualification packet includes route risk scoring, validated shipper fit, and escalation contacts.
The policy requires every qualified lane to be rechecked before summer and winter operating windows.
Should lane qualification include a separate score for weekend customs staffing?

### Stability Budget
Stability Budget means the remaining approved exposure allowance for a product lot after considering temperature, duration, and product-specific stability evidence.
Aliases for Stability Budget are thermal budget, stability allowance, excursion budget.
A release decision must preserve a nonnegative stability budget for every shipped unit and every documented handoff.
The policy requires budget calculation to use the product stability memo, not the shipment average temperature.
Should the stability budget service expose remaining budget as a graph edge attribute?

### Temperature Excursion
Temperature Excursion means an observed or inferred product exposure outside the approved temperature range for a material duration.
Aliases for Temperature Excursion are thermal excursion, out-of-range event, temperature breach.
A temperature excursion must create a deviation triage record before the shipment can be released or discarded.
The policy requires raw sensor files to be retained with the release evidence packet for every excursion.
Should inferred excursions be visible to customers before quality review is complete?

### Chain of Custody
Chain of Custody means the ordered record of accountable handoffs from pack-out through patient site receipt.
Aliases for Chain of Custody are custody chain, handoff chain, custody evidence.
Every custody handoff must include actor identity, location, timestamp, package seal state, and exception notes when applicable.
The policy requires custody evidence to be immutable after release except through a correction record.
Should custody gaps automatically reduce the graph trust label for affected concepts?

### Validated Shipper
Validated Shipper means a packaging configuration proven to maintain the target temperature range for a defined duration and payload profile.
Aliases for Validated Shipper are qualified shipper, thermal shipper, packout system.
A validated shipper must match product temperature class, planned lane duration, and contingency hold time.
The policy requires a new validation or engineering assessment after any insulation material, coolant, or payload bracket changes.
Should shipper validation results be modeled as first-class evidence nodes?

### Quality Hold
Quality Hold means a controlled state that prevents product release until quality evidence is reviewed and disposition is recorded.
Aliases for Quality Hold are QA hold, release hold, product hold.
A shipment under quality hold must not be released to inventory, patient scheduling, or billing systems.
The policy requires quality hold reason codes to reference the triggering evidence and the required disposition path.
Should quality hold state expire if no reviewer acts within the service-level target?

### Deviation Triage
Deviation Triage means the first structured assessment that classifies a quality event, assigns ownership, and selects the disposition path.
Aliases for Deviation Triage are deviation assessment, event triage, quality triage.
Deviation triage must identify whether the event affects product quality, patient scheduling, regulatory reporting, or only operational performance.
The policy requires triage notes to separate observed facts from inferred hypotheses.
Should triage confidence be computed by rule or assigned by the reviewer?

### CAPA
CAPA means a corrective and preventive action record that addresses root cause, containment, verification, and recurrence prevention.
Aliases for CAPA are corrective preventive action, corrective action, preventive action.
A CAPA must link to the triggering deviation, the root cause statement, effectiveness criteria, and closure evidence.
The policy requires preventive actions to name the control that will detect recurrence before patient impact.
Should CAPA effectiveness be visible in the official ontology graph?

### Calibration Drift
Calibration Drift means movement of a measurement device away from accepted tolerance between calibration events.
Aliases for Calibration Drift are sensor drift, logger drift, calibration variance.
A sensor with unresolved calibration drift must not be used as the sole evidence for product release.
The policy requires paired sensor comparison when drift affects a shipment with limited stability budget.
Should calibration drift automatically downgrade evidence freshness or trust state?

### Refrigerant Conditioning
Refrigerant Conditioning means preparing coolant to the validated temperature and physical state before pack-out.
Aliases for Refrigerant Conditioning are coolant conditioning, gel pack conditioning, phase-change conditioning.
Refrigerant conditioning must match the shipper validation recipe and must be recorded before pack-out starts.
The policy requires a restart of conditioning when coolant sits outside the staging window.
Should conditioning chamber telemetry be attached to each release evidence packet?

### Route Risk Scoring
Route Risk Scoring means the calculation that estimates product quality risk across route, weather, customs, carrier, and contingency factors.
Aliases for Route Risk Scoring are lane risk score, transport risk score, route risk model.
A high route risk score must trigger either enhanced monitoring, a stronger shipper, or a quality-approved exception.
The policy requires score inputs to be auditable and reproducible from source evidence.
Should route risk scoring be recalculated after every deviation or only at lane review?

### Release Evidence Packet
Release Evidence Packet means the collected evidence used to justify shipment disposition and downstream release state.
Aliases for Release Evidence Packet are release packet, disposition packet, QA evidence packet.
A release evidence packet must include stability budget calculation, custody evidence, sensor files, deviation records, and final disposition.
The policy requires release packets to preserve source evidence links instead of copied summary text only.
Should the release evidence packet become the default graph focus for audit queries?

### FEFO Allocation
FEFO Allocation means selecting inventory by earliest usable expiry while respecting quality state, route feasibility, and patient scheduling constraints.
Aliases for FEFO Allocation are first-expiry-first-out, expiry allocation, lot allocation.
FEFO allocation must exclude units on quality hold and units without enough stability budget for the planned lane.
The policy requires allocation logic to treat release eligibility as stronger than expiry priority.
Should FEFO allocation consume graph answers directly or only read materialized release state?

### Sensor Pairing
Sensor Pairing means assigning two compatible measurement devices to a shipment so readings can validate each other and expose drift or placement error.
Aliases for Sensor Pairing are dual logger pairing, paired sensors, sensor redundancy.
Sensor pairing must be used when shipment value, stability budget, or regulatory category requires redundant evidence.
The policy requires paired sensors to be time-synchronized before pack-out and reconciled after receipt.
Should paired sensor disagreement create automatic deviation triage?

### Quarantine Workflow
Quarantine Workflow means the physical and system segregation process for units that cannot be released until quality disposition is complete.
Aliases for Quarantine Workflow are inventory quarantine, segregation workflow, blocked stock workflow.
Quarantine workflow must prevent pick, pack, transfer, billing, and patient dispatch for affected units.
The policy requires physical labels and system status to agree before a unit is considered quarantined.
Should quarantine workflow publish events to patient scheduling when a critical dose is affected?

### GDP Audit Readiness
GDP Audit Readiness means the ability to prove distribution controls, evidence integrity, and quality decisions during a good distribution practice audit.
Aliases for GDP Audit Readiness are good distribution practice audit, GDP readiness, distribution audit readiness.
GDP audit readiness must show how each release decision links to source evidence, reviewer identity, and applicable procedure.
The policy requires audit evidence to be retrievable without exposing unrelated patient or commercial data.
Should audit queries default to evidence-only mode until the official graph is reviewed?
