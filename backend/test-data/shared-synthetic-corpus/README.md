# Shared Synthetic Corpus

This directory contains `cornerstone-shared-synthetic-corpus-v1` version `1.0.0`.

It is a fully synthetic organizational dataset for HelioPharm Cold Chain Operations, a fictional team operating in temperature-controlled specialty pharmacy logistics.
It was created for project-wide Cornerstone testing and demos.
It does not rely on public benchmark datasets, real customers, real employees, real shipment ids, real Notion pages, or production secrets.

## Coverage

- Documents: 49
- Source objects: 49
- Concepts: 16
- Expected relations: 16
- Total words: 18382
- Total sentences: 1449
- Artifact types: decision_record, field_report, glossary, sop
- Visibility modes: evidence_only, member_visible

## Files

- `manifest.json`: dataset inventory, metadata, document paths, and expected review intent.
- `source_objects.jsonl`: source-object descriptors that can be loaded by ingestion helpers.
- `expected_ontology.json`: expected concept and relation coverage for ontology extraction and graph tests.
- `evaluation_tasks.json`: reusable query/evaluation seeds for grounded context and graph tests.
- `documents/*.md`: synthetic source documents.

## Intended Uses

Use this corpus for manual upload tests, connector-normalized ingestion tests, evidence review queues, ontology candidate extraction, graph visualization, release-readiness demos, and external integration contract examples.

Normal tests should load a subset when speed matters.
Full project or smoke tests can load all source objects when they need representative volume.
