# MAGI v0 (Local Use)

One prompt goes to three LLMs in parallel and the outputs are displayed side-by-side.

## Structure

- `backend/`: FastAPI + LiteLLM
- `frontend/`: Next.js (App Router) + Tailwind CSS
- `SPEC.md`: implementation spec

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

## Environment Variables

`backend/.env`

- `OPENAI_API_KEY=`
- `ANTHROPIC_API_KEY=`
- `GOOGLE_API_KEY=`

`frontend/.env.local`

- `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000`

Note: backend maps `GOOGLE_API_KEY` to `GEMINI_API_KEY` for LiteLLM compatibility.

## Model Configuration

Edit `backend/config.json` to swap models without code changes.

```json
{
  "agents": [
    { "agent": "A", "provider": "openai", "model": "gpt-4o-mini" },
    { "agent": "B", "provider": "anthropic", "model": "claude-haiku-4-5-20251001" },
    { "agent": "C", "provider": "gemini", "model": "gemini-2.5-flash" }
  ],
  "timeout_seconds": 20
}
```

## API Notes

- Endpoint: `POST /api/magi/run`
- Request body: `{ "prompt": "..." }`
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
- Session history panel (memory only, not persisted)
- `run_id` display and copy button

## Troubleshooting (Local)

- `localhost:3000 refused to connect`: start frontend (`.\start-frontend.ps1`).
- `status=ERROR` for one model: verify API key and model name in `backend/config.json`.
- Anthropic/Gemini errors while others work: usually provider-side quota/credits.
