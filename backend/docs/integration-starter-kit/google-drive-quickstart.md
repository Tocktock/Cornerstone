# Google Drive Quickstart

## Goal

Use Google Drive as a Cornerstone source for selected Google Docs or plain text files.

## Local mock mode

Mock mode is enabled by default:

```bash
GOOGLE_DRIVE_MOCK_EXTERNAL_API=true
```

In mock mode the connector returns one ingestible Google Doc fixture and one non-ingestible Google Sheet fixture. This is useful for API/CLI contract testing.

## Live mode

Set:

```bash
export GOOGLE_DRIVE_MOCK_EXTERNAL_API=false
export GOOGLE_DRIVE_CLIENT_ID='<google-oauth-client-id>'
export GOOGLE_DRIVE_CLIENT_SECRET='<google-oauth-client-secret>'
```

Use the standard connector flow:

```bash
# create connection intent through API or Source Studio
# open authorization URL
# complete OAuth callback
# test connection
# discover files
# select supported files
# run sync worker
```

## Supported file types

In v1.2.0:

```text
Google Docs → exported as text/plain
plain text files → downloaded as text
```

Deferred:

```text
Google Sheets
Google Slides
PDFs
folders
binary files
```

## Safety

Do not store Google access tokens in CLI config, docs, reports, or committed files.
