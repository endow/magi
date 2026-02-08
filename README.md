# MAGI v0

One prompt goes to three LLMs in parallel and the outputs are displayed side-by-side.

## Structure

- `backend/`: FastAPI + LiteLLM
- `frontend/`: Next.js (App Router) + Tailwind CSS
- `SPEC.md`: implementation spec used for this project

## Prerequisites

- Python 3.11+
- Node.js 20+
- npm 10+

## Backend Setup

```bash
cd backend
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend URL: `http://localhost:8000`

### Backend Notes

- Endpoint: `POST /api/magi/run`
- Request body: `{ "prompt": "..." }`
- Empty prompt or over 4000 chars returns `400`
- CORS allows `http://localhost:3000`

## Frontend Setup

```bash
cd frontend
npm install
copy .env.example .env.local
npm run dev
```

Frontend URL: `http://localhost:3000`

## Environment Variables

`backend/.env`

- `OPENAI_API_KEY=`
- `ANTHROPIC_API_KEY=`
- `GOOGLE_API_KEY=`

`frontend/.env.local`

- `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000`

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

Note: backend maps `GOOGLE_API_KEY` to `GEMINI_API_KEY` internally for LiteLLM compatibility.

## Behavior Summary

- Three agents run concurrently with `asyncio.gather`
- Each agent call has per-call timeout via `asyncio.wait_for`
- Partial failure is allowed (`status: ERROR` with `error_message`)
- Backend returns a UUID `run_id`
