# MAGI v0 (Local Use)

One prompt goes to three LLMs in parallel and the outputs are displayed side-by-side.

## Structure

- `backend/`: FastAPI + LiteLLM
- `frontend/`: Next.js (App Router) + Tailwind CSS
- `SPEC.md`: implementation spec
- `AGENTS.md`: coding-agent working rules
- `RUNBOOK.md`: operational commands and troubleshooting

## Prerequisites

- Python 3.10+
- Node.js 20+
- npm 10+

## One-Time Setup

### Backend

```bash
cd backend
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

### Frontend

```bash
cd frontend
npm install
copy .env.example .env.local
```

## Local Run

From repository root:

```bash
.\start-backend.ps1
```

Open another terminal:

```bash
.\start-frontend.ps1
```

URLs:

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`
- Health: `http://localhost:8000/health`

## Docker Run (FE/BE together)

Prerequisites:

- Docker Desktop
- `backend/.env` configured (copy from `backend/.env.example`)

From repository root:

```bash
docker compose up --build -d
```

Check status:

```bash
docker compose ps
```

Stop:

```bash
docker compose down
```

URLs:

- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8000`

## Environment Variables

`backend/.env`

- `OPENAI_API_KEY=`
- `ANTHROPIC_API_KEY=`
- `GOOGLE_API_KEY=`

`frontend/.env.local`

- `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000`

Note: backend maps `GOOGLE_API_KEY` to `GEMINI_API_KEY` for LiteLLM compatibility.

## Model Configuration

Edit `backend/config.json` to define profiles and swap models without code changes.

```json
{
  "default_profile": "cost",
  "profiles": {
    "cost": { "...": "..." },
    "creative": { "...": "..." },
    "logical": { "...": "..." }
  }
}
```

## API Notes

- Endpoint: `POST /api/magi/run`
- Request body: `{ "prompt": "...", "profile": "cost|creative|logical" }`
- Run response includes `consensus` (synthesized final answer from peer-vote deliberation)
- Retry endpoint: `POST /api/magi/retry`
- Retry body: `{ "prompt": "...", "agent": "A|B|C", "profile": "..." }`
- Consensus endpoint: `POST /api/magi/consensus`
- Consensus body: `{ "prompt": "...", "results": AgentResult[], "profile": "..." }`
- Profiles endpoint: `GET /api/magi/profiles`
- Empty prompt or over 4000 chars returns `400`
- Per-model timeout: 20 seconds
- Partial failure is allowed (`status: ERROR`)
- Backend returns `run_id` as UUID

## Current UI Features

- Enter to submit (`Shift+Enter` for newline)
- Prompt character counter (`0/4000`)
- 3-column result cards with:
  - status
  - model id
  - latency
  - response text
- Per-card actions:
  - `Copy` response text
  - `Retry` failed card
- Consensus panel:
  - shown before the 3 result cards
  - shows synthesized conclusion from multi-agent deliberation
  - supports `OK/ERROR` status and latency display
- Profile selector:
  - choose `cost`, `creative`, `logical`
  - selected profile is sent on run/retry/consensus
- Session history panel (memory only, not persisted)
- `run_id` display and copy button

## Troubleshooting (Local)

- `localhost:3000 refused to connect`: start frontend (`.\start-frontend.ps1`).
- `status=ERROR` for one model: verify API key and model name in `backend/config.json`.
- Anthropic/Gemini errors while others work: usually provider-side quota/credits.
