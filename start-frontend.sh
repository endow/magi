#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/frontend"

if [[ ! -f ".env.local" ]]; then
  echo "frontend/.env.local not found. Copy frontend/.env.example to frontend/.env.local first." >&2
  exit 1
fi

if [[ ! -d "node_modules" ]]; then
  echo "Installing frontend dependencies..."
  npm install
fi

echo "Starting frontend on http://localhost:3000"
exec npm run dev
