# 110 - Quality and Containerization Upgrade

Status: Approved

## Decision

The reference implementation moves to a split workspace:

- `backend/` for the FastAPI API and curation engine
- `frontend/` for a dedicated React + Vite UI
- `compose.yml` for a multi-container local environment

## Rationale

The previous shape mixed review UI into the backend and left the repository structurally incomplete. This upgrade fixes three things:

1. the backend becomes a coherent domain-first API project with integration tests
2. the frontend becomes an independently buildable application with stronger CSS and layout discipline
3. the stack becomes easy to run through Docker Compose with Postgres, health checks, and explicit service boundaries

## Non-replaceable alignment

This change preserves the SoT:

- Cornerstone remains the official context layer, not a source replacement
- Concept, Relation, Decision Record, and Evidence remain first-class
- reviewability and provenance remain the gate for official knowledge

## Replaceable choices selected here

- React + Vite for the frontend
- Postgres in Docker Compose for local containerized development
- SQLite for fast integration tests
