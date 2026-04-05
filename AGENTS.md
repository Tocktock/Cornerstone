# Cornerstone Repository Rules

These rules are top-priority instructions for this repository and must always be followed.

## Documentation governance

1. This project must be developed in a spec-driven manner.
2. `/docs` is the canonical documentation home for this repository.
3. Feature behavior and requirements must live under `/docs/specs/`.
4. Durable architectural decisions and technical invariants must live under `/docs/decisions/`.
5. Traceability records must be maintained under `/docs/memories/`.
6. Conversations, intentions, deprecated content, and rationale behind design or implementation decisions must be recorded in `/docs/memories/`.
7. Every feature change must update the relevant spec in the same change.
8. Files outside `/docs` may contain short summaries and links, but they must not become shadow Sources of Truth for feature behavior.
9. Existing high-level reference documents under `/docs/sot/` and legacy `/docs/spec/` remain valid inputs until migrated, but new feature-level documentation must not be added there when `/docs/specs/`, `/docs/decisions/`, or `/docs/memories/` is the correct home.

## Documentation precedence

- Product identity, goals, and legacy high-level references: `/docs/sot/`, `/docs/spec/`
- Feature behavior and requirements: `/docs/specs/`
- Architectural decisions and durable technical rules: `/docs/decisions/`
- Traceability, rationale history, deprecations, and implementation intent: `/docs/memories/`

## Required workflow

1. Before implementation, create or update the relevant spec under `/docs/specs/`.
2. When a durable architectural or technical rule changes, add or update a decision record under `/docs/decisions/`.
3. During design or behavior discussion, record intent and rationale under `/docs/memories/`.
4. After implementation changes, update the relevant spec and memory note in the same change.

## Documentation organization

- Use feature-first folders under `/docs/specs/`.
- Use feature-first folders under `/docs/memories/`.
- Use numbered filenames for decision records: `000x-<slug>.md`.
- Use date-prefixed filenames for memory notes: `YYYY-MM-DD-<type>-<slug>.md`.
- Keep new docs in English by default unless a document explicitly needs bilingual content.
