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

Thread delete API:

```powershell
Invoke-RestMethod -Method Delete http://localhost:8000/api/magi/history/thread/<thread_id>
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

## Fresh Mode (Latest Info)

- Set `TAVILY_API_KEY` in `backend/.env` to enable Fresh mode web retrieval.
- Optional tuning:
  - `FRESH_MAX_RESULTS` (default `3`, max `10`)
  - `FRESH_CACHE_TTL_SECONDS` (default `1800`)
  - `FRESH_SEARCH_DEPTH` (`basic` or `advanced`)
  - `FRESH_PRIMARY_TOPIC` (`general` or `news`, default `general`)
- If key is missing or Tavily request fails, backend falls back to normal prompt (run does not fail).

## History DB

- Default DB path: `backend/data/magi.db`
- Override path with `MAGI_DB_PATH` in `backend/.env`
- Recommended override (Docker/Local共通): `MAGI_DB_PATH=data/magi.db`
- Docker uses bind mount `./backend/data:/app/data`; if this mount is absent, history can be lost on container recreation.
- If history is not needed, you can remove the DB file while backend is stopped and it will be recreated on next start.

## Consensus Modes

- `cost` / `balance`: normal peer-vote consensus.
- `performance` / `ultra`: strict debate consensus (`min_criticisms=2`).
- If strict mode yields consensus errors, inspect backend logs for `strict debate requires at least ... criticisms`.
