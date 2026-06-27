# Git Setup

Iteration 005 initializes `quant-platform` as the Git repository root.

## Local Repository

The local repository should be initialized in:

```text
quant-platform/
```

Default branch:

```powershell
git branch -M main
```

## Remote

Configured remote:

```text
https://github.com/franiatesting-tech/quantfinance
```

Check it with:

```powershell
git remote -v
```

## Ignored Content

The repository excludes:

- `.env` and `.env.*` files with local secrets.
- Python caches and build artifacts.
- Local raw/processed/registry data.
- Generated reports.
- PDFs under `docs/literature/pdfs/*.pdf` unless permissions are explicitly confirmed.

Metadata and bibliography maps may be committed. Secrets, real data, generated outputs, and unlicensed PDFs must not be committed.

## Commit And Push

Use:

```powershell
git status
git add .
git commit -m "Add read-only real data research pipeline foundation"
git push -u origin main
```

If GitHub asks for credentials, use a GitHub token or Git Credential Manager. If authentication fails, do not retry blindly; verify account access and token scopes first.
