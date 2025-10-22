# Issue Reporting Guide

This guide shows what a well-formed issue looks like when using the default template at `.github/ISSUE_TEMPLATE/issue_template.md`.

## Example Issue

```
Title: [Issue]: Scheduler sync job fails after nightly deployment

## Summary

Nightly scheduler deployments fail after upgrading `scheduler-core` to 2.1.1; the sync job aborts with a 500 when building the daily task queue.

## Steps to Reproduce

1. Deploy commit 739b8d9 to the staging environment.
2. Trigger the nightly sync job via the `/sync` endpoint.

## Expected Behavior

The sync job completes and schedules the next day's tasks.

## Actual Behavior

The job returns HTTP 500 with `Unable to load workspace configuration`, and no tasks are scheduled.

## Impact

Operations must schedule tasks manually, delaying partner onboarding by several hours.

## Environment

- Commit/Version: 739b8d9 (main)
- Deployment/Runtime: Staging scheduler service
- Operating System: Ubuntu 22.04 LTS
- Other Context: Upgrade to `scheduler-core` 2.1.1 landed earlier the same day.

## Additional Context

Relevant log excerpt:
```text
2024-08-19T03:15:02Z ERROR sync-runner failed to load workspace config
```
```

Use this as a starting point and adjust the content (title, environment details, logs, etc.) to match the issue you are reporting.
