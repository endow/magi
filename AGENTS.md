# AGENTS.md

This file defines working rules for coding agents in this repository.

## Scope

- Applies to the whole repository (`backend/`, `frontend/`, root files).
- If multiple instruction files exist, the closest file in the directory tree has priority.

## Read Order (Before Any Change)

1. `AGENTS.md` (this file)
2. `SPEC.md`
3. `README.md`
4. Files directly related to the requested change

## Core Rules

- Keep implementation aligned with `SPEC.md`.
- Avoid over-engineering; prefer minimum working changes.
- Do not commit secrets (`.env`, API keys, tokens).
- Keep model/provider settings externalized in `backend/config.json`.
- Preserve partial-failure behavior (one model failure must not crash whole run).

## Required Validation

After backend changes:

- `python -m pytest backend/tests -q`

After frontend changes:

- `npm run build` (in `frontend/`)
- `npm run test:e2e` (in `frontend/`, when UI behavior changed)

## Docker Policy

- If code/config used by runtime changed, rebuild and restart containers:
  - `docker compose up --build -d`
- Verify runtime:
  - `docker compose ps`
  - Backend health: `http://localhost:8000/health`

## Ops / Troubleshooting Reference

- See `RUNBOOK.md` for operational commands:
  - Next.js cache cleanup (`frontend/.next`)
  - Docker rebuild/restart
  - common local recovery steps

