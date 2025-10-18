# Cornerstone Sample Data

This directory hosts a tiny, fully anonymized dataset you can use to preview Cornerstone without uploading private documents. The samples mirror the structure used by the ingestion pipeline and analytics services.

## Contents

- `projects.json` – Three demo projects (`default-project`, `logistics`, `product-support`) seeded with placeholder descriptions.
- `documents/default-project.json` – A handful of Markdown knowledge base entries (incident runbooks, onboarding steps) with synthetic text.
- `personas.json` – Default persona configuration with safe overrides.

All text is fictional; no customer data or credentials are included.

## Loading the Samples

1. Copy the files into your writable data directory (defaults to `data/`).
   ```bash
   cp -R samples/* data/
   ```
2. Start the application and browse the Search, Support, and Analytics pages to explore the seeded content.

You can delete or replace the sample files at any time—the ingestion scripts and UI upload flow will rebuild the data directory from scratch.
