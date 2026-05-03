# v1.2.0 — Google Drive Connector

## Purpose

Add Google Drive as the next runtime connector after Notion while keeping the core Cornerstone model provider-neutral.

The connector supports this first slice:

```text
Google OAuth
→ connection test
→ Drive file discovery
→ source selection
→ selected Google Doc/text-file ingestion
→ Artifact
→ EvidenceFragment
```

## Scope

Included:

```text
- Google Drive catalog entry
- OAuth authorization URL generation
- OAuth token exchange boundary
- credential reuse through existing ConnectorCredential model
- connection test through Drive files.list
- file discovery through Drive files.list
- selected Google Docs export as text/plain
- selected plain text file download
- normalized SourceObject creation
- Artifact/EvidenceFragment creation through existing sync worker
- provider error mapping for auth, permission, not found, rate limit, and provider outage
```

Deferred:

```text
- Google Sheets ingestion
- Google Slides ingestion
- PDF extraction/OCR
- folder-as-source semantics
- Drive push notifications/webhooks
- per-file Google Picker UX
- live Google Drive E2E script
```

## Supported Object Matrix

| Object type | Discoverable | Ingestible | Notes |
|---|---:|---:|---|
| Google Doc | Yes | Yes | Exported through Drive `files.export` as `text/plain`. |
| Plain text file | Yes | Yes | Downloaded through Drive media download. |
| Google Sheet | Yes | No | Requires spreadsheet-specific extraction policy. |
| Google Slide | Yes | No | Requires presentation extraction policy. |
| PDF | Yes | No | Requires PDF text extraction/OCR policy. |
| Folder | Yes | No | Folders organize files but are not evidence themselves. |
| Binary/other file | Yes | No | Deferred until file-type policy exists. |

## Environment Variables

```text
GOOGLE_DRIVE_CLIENT_ID=...
GOOGLE_DRIVE_CLIENT_SECRET=...
GOOGLE_DRIVE_OAUTH_AUTHORIZE_URL=https://accounts.google.com/o/oauth2/v2/auth
GOOGLE_DRIVE_OAUTH_TOKEN_URL=https://oauth2.googleapis.com/token
GOOGLE_DRIVE_MOCK_EXTERNAL_API=true
GOOGLE_DRIVE_DISCOVERY_QUERY=trashed = false
GOOGLE_DRIVE_EXPORT_MIME_TYPE=text/plain
GOOGLE_DRIVE_PAGE_SIZE=25
```

## Source Admin Flow

```text
1. Create Google Drive connection intent.
2. Open returned authorization URL.
3. Complete OAuth callback.
4. Test connection.
5. Discover files.
6. Select supported Google Docs/text files.
7. Queue sync job.
8. Run worker.
9. Review resulting EvidenceFragments.
```

## Trust Rules

Google Drive data follows the same trust model as Notion/manual sources:

```text
- OAuth success is not source trust.
- Connection test is separate from authorization.
- Discovery metadata is not evidence.
- Only selected ingestible files can produce Artifacts.
- Evidence starts unreviewed.
- Official context still requires reviewer approval.
```

## Known Limitations

```text
- Sheets, Slides, PDFs, folders, and binary files are discovery-only in this slice.
- Live Google Drive E2E runner is bundled, but requires operator-provided Google OAuth access token and shared file ID.
- Google API OAuth verification and consent-screen setup are deployment-owner responsibilities.
- Production mode requires live Google Drive credentials and rejects mock Google Drive mode.
```
