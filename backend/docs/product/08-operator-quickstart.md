# Operator Quickstart

This Cornerstone quickstart shows the product loop, not a full production deployment.

The goal is to prove:

```text
manual source data
→ evidence
→ candidates
→ review
→ official graph
→ evaluation
→ SSOT readiness
```

## 1. Start the backend

```bash
cornerstone api --reload
```

Check health:

```bash
curl http://localhost:8000/healthz
```

## 2. Create a manual source

```bash
curl -X POST http://localhost:8000/v1/sources \
  -H 'Content-Type: application/json' \
  -d '{
    "type": "manual",
    "name": "Settlement Demo Source",
    "productionEnabled": true
  }'
```

Save the returned source id as `SOURCE_ID`.

## 3. Upload or paste settlement text

```bash
curl -X POST "http://localhost:8000/v1/manual-sources/$SOURCE_ID/uploads/text" \
  -H 'Content-Type: application/json' \
  -d '{
    "objects": [
      {
        "title": "Settlement Notes",
        "content": "Settlement is the process of finalizing financial obligations. Clearing happens before settlement. Reconciliation validates settlement results. Settlement updates the ledger after obligations are finalized."
      }
    ],
    "queueOntologyReExtraction": true,
    "ontologyFocusConcept": "Settlement"
  }'
```

The upload creates Artifacts and EvidenceFragments. It does not create official graph truth.

## 4. Run re-extraction if queued

```bash
curl -X POST http://localhost:8000/v1/ontology/re-extraction-runs/<run-id>/run \
  -H 'Content-Type: application/json' \
  -d '{"requestedBy":"operator@example.com"}'
```

This creates ConceptCandidates and RelationCandidates only.

## 5. Review evidence

```bash
cornerstone evidence queue
cornerstone evidence review <evidence-id> --reviewer reviewer@example.com
```

## 6. Review ontology candidates

Use the candidate review endpoints:

```http
POST /v1/ontology/concept-candidates/{candidateId}/approve
POST /v1/ontology/relation-candidates/{candidateId}/approve
```

Approval still requires evidence and officialization gates.

## 7. Query the official graph

```bash
curl "http://localhost:8000/v1/ontology/graph?concept=Settlement&depth=1&mode=official"
```

Expected result shape:

```text
Settlement official graph
- direct nodes
- direct edges
- citations
- review provenance
- trust label
- candidate summary
```

## 8. Run ontology evaluation

```bash
curl -X POST http://localhost:8000/v1/evaluations/ontology/run \
  -H 'Content-Type: application/json' \
  -d '{"focusConcept":"Settlement","depth":1,"mode":"official"}'
```

## 9. Run SSOT readiness

```bash
curl "http://localhost:8000/v1/ontology/ssot/readiness?focusConcept=Settlement&depth=1&mode=official"
```

Or through CLI:

```bash
cornerstone proof run --ssot-readiness --ontology-focus-concept Settlement --json
```

## Fast proof path

For a checklist-driven proof:

```bash
cornerstone proof run \
  --ontology-loop \
  --confirm-ontology-mutation \
  --ontology-focus-concept Settlement \
  --reviewer reviewer@example.com \
  --json
```

Then:

```bash
cornerstone proof run \
  --ssot-readiness \
  --ontology-focus-concept Settlement \
  --json
```

## Operator quickstart acceptance checklist

| Check ID | Requirement | Measurement | Status |
|---|---|---|---|
| PROD-OPS-01 | Quickstart proves the full product loop. | Lists source → readiness sequence. | complete |
| PROD-OPS-02 | Manual source path is included. | Shows source creation and upload/text sync. | complete |
| PROD-OPS-03 | Candidate-only boundary is preserved. | Upload and re-extraction sections state no official truth. | complete |
| PROD-OPS-04 | Review steps are included. | Evidence and candidate review are explicit. | complete |
| PROD-OPS-05 | Evaluation and readiness are included. | Includes ontology evaluation and SSOT readiness commands. | complete |
