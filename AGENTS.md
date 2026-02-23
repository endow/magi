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

## Command Execution Rules

- Do not run order-dependent commands in parallel.
- Run sequentially when a later command depends on the previous command result.
- For Git operations, always run `git commit` first, then `git push`.

## Docker Policy

- If code/config used by runtime changed, rebuild and restart containers:
  - `docker compose up --build -d`
- Verify runtime:
  - `docker compose ps`
  - Backend health: `http://localhost:8000/health`
- After frontend code/style changes, rebuild frontend before visual verification:
  - `docker compose up --build -d frontend`

## Validation

After backend changes:

- `python -m pytest backend/tests -q`

After frontend changes (run sequentially, not in parallel):

1. `npm run build` (in `frontend/`)
2. `npm run test:e2e` (in `frontend/`, when UI behavior changed)

Reason:

- Parallel execution can cause process interference and false build failures.

## Generated Files

- `frontend/next-env.d.ts` is auto-generated/updated by Next.js.
- If it changes, review the diff and decide commit inclusion explicitly (do not auto-revert or auto-commit).

## Ops / Troubleshooting

- See `RUNBOOK.md` for operational commands:
  - Next.js cache cleanup (`frontend/.next`)
  - Docker rebuild/restart
  - common local recovery steps
