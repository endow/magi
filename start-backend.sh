#!/usr/bin/env bash
set -euo pipefail

NO_RELOAD=0
if [[ "${1:-}" == "--no-reload" ]]; then
  NO_RELOAD=1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/backend"

if [[ ! -f ".env" ]]; then
  echo "backend/.env not found. Copy backend/.env.example to backend/.env first." >&2
  exit 1
fi

RELOAD_ARG="--reload"
if [[ "${NO_RELOAD}" -eq 1 ]]; then
  RELOAD_ARG=""
fi

echo "Starting backend on http://localhost:8000"
exec python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 ${RELOAD_ARG}
