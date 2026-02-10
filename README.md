# MAGI v0.7 (Local Use)

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
- `TAVILY_API_KEY=` (optional, for Fresh mode web retrieval)
- `FRESH_MAX_RESULTS=3` (optional, fresh mode安定化向け推奨)
- `FRESH_CACHE_TTL_SECONDS=1800` (optional)
- `FRESH_SEARCH_DEPTH=basic` (optional: `basic|advanced`)
- `FRESH_PRIMARY_TOPIC=general` (optional: `general|news`)
- `MAGI_DB_PATH=data/magi.db` (optional, Docker/Local 共通推奨)

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
    "balance": { "...": "..." },
    "performance": { "...": "..." }
  }
}
```

## API Notes

- Endpoint: `POST /api/magi/run`
- Request body: `{ "prompt": "...", "profile": "cost|balance|performance", "fresh_mode": false }`
- Run response includes `consensus` (synthesized final answer from peer-vote deliberation)
- Run responses are persisted to local SQLite history
- Retry endpoint: `POST /api/magi/retry`
- Retry body: `{ "prompt": "...", "agent": "A|B|C", "profile": "...", "fresh_mode": false }`
- Consensus endpoint: `POST /api/magi/consensus`
- Consensus body: `{ "prompt": "...", "results": AgentResult[], "profile": "...", "fresh_mode": false }`
- Profiles endpoint: `GET /api/magi/profiles`
- History list endpoint: `GET /api/magi/history?limit=20&offset=0`
- History detail endpoint: `GET /api/magi/history/{run_id}`
- Empty prompt or over 4000 chars returns `400`
- Per-model timeout: profile config (`backend/config.json`) に従う（現行: cost 25s / balance 35s / performance 45s）
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
  - choose `cost`, `balance`, `performance`
  - selected profile is sent on run/retry/consensus
  - default profile is `performance` (from `backend/config.json`)
  - `performance` enables strict debate consensus (requires concrete cross-agent criticisms)
  - UI shows a `strict debate` badge when `performance` is selected
- Fresh mode toggle:
  - default is ON
  - when ON, backend retrieves recent web evidence via Tavily (if `TAVILY_API_KEY` is configured)
  - retrieval uses multi-attempt fallback (`general/news` + query expansion) for non-news topics like game guides
  - if Tavily is unavailable or key is missing, it falls back to normal prompt automatically
- Run history panel (persisted in backend SQLite)
- `run_id` display and copy button

## Troubleshooting (Local)

- `localhost:3000 refused to connect`: start frontend (`.\start-frontend.ps1`).
- `status=ERROR` for one model: verify API key and model name in `backend/config.json`.
- Anthropic/Gemini errors while others work: usually provider-side quota/credits.
- Docker で履歴が消える: `docker-compose.yml` で `./backend/data:/app/data` がマウントされていること、かつ `MAGI_DB_PATH` は未設定または `data/magi.db` を使用する。
