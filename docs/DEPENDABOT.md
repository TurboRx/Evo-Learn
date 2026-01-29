# Dependabot Configuration

This document describes the Dependabot configuration for the Evo-Learn repository.

## Overview

Dependabot is configured to automatically check for dependency updates across three package ecosystems:
- Python (pip)
- Node.js (npm)
- GitHub Actions

## Configuration Files

### `.github/dependabot.yml`

Main Dependabot configuration that defines:
- Update schedule (weekly)
- Package ecosystems to monitor
- Grouping strategy to minimize PRs
- Label assignments for categorization

### `.github/workflows/dependabot-auto-merge.yml`

Workflow that automatically enables auto-merge for Dependabot PRs.

#### Recent Updates (January 2026)

Updated to comply with GitHub's changes to Dependabot PR handling:

1. **Explicit Permissions Block**: Added required permissions for modern GitHub Actions security model
   ```yaml
   permissions:
     pull-requests: write  # Required for auto-merge
     contents: write       # Required for workflow operations
   ```

2. **Explicit PR Number**: Added `pull-request-number` parameter for clarity and compatibility
   ```yaml
   pull-request-number: ${{ github.event.pull_request.number }}
   ```

These changes ensure compatibility with GitHub's move away from comment-based commands (`@dependabot merge`, etc.) towards explicit workflow-based automation.

## How It Works

1. **Dependabot scans** for outdated dependencies weekly
2. **Creates grouped PRs** for related updates to minimize noise
3. **Auto-merge workflow** automatically enables auto-merge on Dependabot PRs
4. **CI tests run** automatically on all PRs
5. **Auto-merge completes** once CI passes (squash merge)

## Update Policy

The configuration ignores patch and minor version updates to reduce noise. However:
- **Major version updates** are NOT ignored and will be created when detected
- **Security updates** are NOT ignored and will be created immediately when detected, regardless of the weekly schedule

This policy balances keeping dependencies up-to-date with minimizing disruption from frequent minor updates.

## Customization

To modify the configuration:
- Edit `.github/dependabot.yml` for update frequency, grouping, or ignore rules
- Edit `.github/workflows/dependabot-auto-merge.yml` to change merge behavior

## References

- [Dependabot Documentation](https://docs.github.com/en/code-security/dependabot)
- [GitHub Actions Permissions](https://docs.github.com/en/actions/security-guides/automatic-token-authentication#permissions-for-the-github_token)
- [GitHub Changelog: Changes to Dependabot PR Commands (Jan 2026)](https://github.blog/changelog/2026-01-27-changes-to-github-dependabot-pull-request-comment-commands/)
