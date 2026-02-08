# RUNBOOK.md

Operational commands for local development and Docker runtime.

## Local Start

Backend:

```powershell
.\start-backend.ps1
```

Frontend:

```powershell
.\start-frontend.ps1
```

## Docker Start / Rebuild

Start or refresh everything:

```bash
docker compose up --build -d
```

Status:

```bash
docker compose ps
```

Stop:

```bash
docker compose down
```

## Quick Health Checks

Backend:

```powershell
Invoke-WebRequest -UseBasicParsing http://localhost:8000/health
```

Frontend:

```powershell
Invoke-WebRequest -UseBasicParsing http://localhost:3000
```

History API:

```powershell
Invoke-RestMethod -Method Get http://localhost:8000/api/magi/history?limit=5&offset=0 | ConvertTo-Json -Depth 6
```

## Next.js Cache Recovery

Use when dev server shows module/chunk inconsistencies (e.g. missing `_document`, missing chunk files).

```powershell
if (Test-Path frontend\.next) { cmd /c rmdir /s /q frontend\.next }
cd frontend
npm run dev
```

For production build verification:

```powershell
if (Test-Path frontend\.next) { cmd /c rmdir /s /q frontend\.next }
cd frontend
npm run build
```

## Backend Warnings

- `trio._core._multierror RuntimeWarning` can appear depending on environment hooks.
- The app suppresses this warning in code; if seen in ad-hoc scripts, it is usually non-fatal.

## Provider Quota Errors

- If a single agent shows `ERROR` with Gemini, check backend logs for `429` / `RESOURCE_EXHAUSTED`.
- This indicates provider quota/rate-limit, not an app crash.
- Wait and retry, or raise provider quota/billing limits.

## History DB

- Default DB path: `backend/data/magi.db`
- Override path with `MAGI_DB_PATH` in `backend/.env`
- If history is not needed, you can remove the DB file while backend is stopped and it will be recreated on next start.
