# Cornerstone

Cornerstone is the shared organizational context layer for humans and AI.

This repository upgrades the earlier reference implementation into a split, container-ready workspace with a stronger backend domain model, dedicated integration tests, and a standalone frontend.

## What changed

- **Backend quality**
  - coherent domain-first package structure
  - stronger SQLAlchemy data model with unique constraints and first-class association tables
  - clearer review invariants around evidence and accepted decision lineage
  - integration tests that exercise sync, officialization, and answer assembly end-to-end
- **Frontend quality**
  - dedicated React + Vite application
  - improved visual design and layout through a single CSS system
  - API-driven pages for glossary, graph, decisions, artifacts, sources, and review
- **Container workflow**
  - `compose.yml` for Postgres + backend + frontend
  - separate Dockerfiles for backend and frontend
  - local environment that mirrors the service split

## Workspace layout

```text
backend/         FastAPI API, sync engine, domain model, integration tests
frontend/        React + Vite application and CSS system
demo_sources/    demo markdown artifacts for the filesystem connector
docs/sot/        product SoT documents
docs/spec/       product and domain specs
specs/           implementation-level project decisions
compose.yml      multi-container local development stack
```

## Local development

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest
uvicorn cornerstone.main:app --reload
```

Backend API:
- `http://localhost:8000/`
- `http://localhost:8000/docs`
- `http://localhost:8000/api/v1/health`

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend UI:
- `http://localhost:5173`

### Sample-data + Ollama

To run against the exported sample corpus instead of the curated demo sources, point the backend at `sample-data/` and enable Ollama-backed answering:

```bash
cd backend
source .venv/bin/activate
CORNERSTONE_SOURCE_ROOT=../sample-data \
CORNERSTONE_OLLAMA_ENABLED=true \
CORNERSTONE_OLLAMA_CHAT_MODEL=qwen3:0.6b \
CORNERSTONE_OLLAMA_EMBEDDING_MODEL=qwen3-embedding:0.6b \
uvicorn cornerstone.main:app --reload
```

This keeps glossary/graph officialization rules unchanged, while `/api/v1/answers` can retrieve raw source evidence from unstructured artifacts and summarize it through the lightweight local Ollama model.

## Docker Compose

```bash
cp .env.example .env
docker compose up --build
```

Services:
- frontend: `http://localhost:5173`
- backend: `http://localhost:8000`
- postgres: `localhost:5432`

## Verified locally

- backend integration suite passes
- frontend production build passes
- backend application boots and serves the health endpoint

## Design commitments preserved from the SoT

- Cornerstone does **not** replace source systems.
- `Concept`, `ConceptRelation`, `DecisionRecord`, and `EvidenceFragment` remain first-class.
- Glossary and graph remain **projections** over the canonical model.
- Human review remains the gate for official knowledge.
- Provenance and accepted decision lineage remain essential for officialization.

## Recommended next steps

1. add a Notion connector on top of the existing connector contract
2. add authn/authz for real reviewer roles
3. add richer graph visualization and filtering
4. add async worker execution for larger sync workloads
