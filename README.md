# MAGI v0.9 (Local Use)

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
# Docker MCP で使う場合（任意）
copy .env.mcp.example .env.local
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

Note:
- Docker構成には `ollama` が含まれます。
- 初回起動時は `ollama-pull` が `qwen2.5:7b-instruct-q4_K_M` を取得するため、少し時間がかかります。

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
- `HISTORY_CONTEXT_ENABLED=1` (optional: `0/false` で無効化)
- `HISTORY_SIMILARITY_THRESHOLD=0.78` (optional: 0.0-1.0)
- `HISTORY_SIMILAR_CANDIDATES=120` (optional: 類似検索対象の最新履歴件数)
- `HISTORY_MAX_REFERENCES=2` (optional: プロンプトに注入する参照履歴数)
- `HISTORY_FRESHNESS_HALF_LIFE_DAYS=180` (optional: 履歴スコアの鮮度減衰)
- `HISTORY_STALE_WEIGHT=0.55` (optional: stale 履歴の重み)
- `HISTORY_SUPERSEDED_WEIGHT=0.20` (optional: superseded 履歴の重み)
- `history_context` を `backend/config.json` で設定している場合は、`HISTORY_FRESHNESS_HALF_LIFE_DAYS` / `HISTORY_STALE_WEIGHT` / `HISTORY_SUPERSEDED_WEIGHT` より `config.json` 側を優先
- `THREAD_CONTEXT_ENABLED=1` (optional: `0/false` でスレッド文脈注入を無効化)
- `THREAD_CONTEXT_MAX_TURNS=6` (optional: 参照する直近ターン数)
- `OLLAMA_API_BASE=http://ollama:11434` (Docker Compose時の推奨値)

`frontend/.env.local`

- `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000`
- Docker MCP/コンテナ内ブラウザからアクセスする場合は `NEXT_PUBLIC_API_BASE_URL=http://host.docker.internal:8000` を使用
- サンプルとして `frontend/.env.mcp.example` を用意（`copy .env.mcp.example .env.local`）

Note: backend maps `GOOGLE_API_KEY` to `GEMINI_API_KEY` for LiteLLM compatibility.

## Model Configuration

Edit `backend/config.json` to define profiles and swap models without code changes.

```json
{
  "default_profile": "balance",
  "history_context": {
    "strategy": "embedding",
    "provider": "openai",
    "model": "text-embedding-3-small",
    "timeout_seconds": 12,
    "batch_size": 24,
    "freshness_half_life_days": 180,
    "stale_weight": 0.55,
    "superseded_weight": 0.20,
    "deprecations": [
      {
        "id": "authjs-migration",
        "legacy_terms": ["nextauth", "next-auth"],
        "current_terms": ["auth.js", "authjs", "better auth"]
      },
      {
        "id": "next-app-router-migration",
        "legacy_terms": ["pages router", "getserversideprops", "getstaticprops", "_app.tsx", "_document.tsx"],
        "current_terms": ["app router", "route handler", "server component", "layout.tsx"]
      },
      {
        "id": "react-query-rename",
        "legacy_terms": ["react-query", "react query v3"],
        "current_terms": ["tanstack query", "@tanstack/react-query"]
      },
      {
        "id": "next-image-legacy",
        "legacy_terms": ["next/legacy/image", "legacy image component"],
        "current_terms": ["next/image"]
      },
      {
        "id": "prisma-major-upgrade",
        "legacy_terms": ["prisma 4", "prisma4", "previewfeatures"],
        "current_terms": ["prisma 5", "prisma5"]
      },
      {
        "id": "eslint-flat-config",
        "legacy_terms": [".eslintrc", ".eslintrc.js", ".eslintrc.json"],
        "current_terms": ["eslint.config.js", "flat config"]
      },
      {
        "id": "next-font-migration",
        "legacy_terms": ["@next/font", "next font package"],
        "current_terms": ["next/font"]
      },
      {
        "id": "turbo-repo-rename",
        "legacy_terms": ["turborepo", "turbo repo"],
        "current_terms": ["turbo", "turbo.json"]
      }
    ]
  },
  "profiles": {
    "cost": { "...": "..." },
    "balance": { "...": "..." },
    "performance": { "...": "..." },
    "ultra": { "...": "..." }
  }
}
```

`history_context.strategy`:
- `embedding`: 外部埋め込みモデルで類似履歴を検索（失敗時は自動でローカル類似検索へフォールバック）
- `lexical`: ローカル類似検索のみ

`request_router` (optional):
- `enabled=true` のとき、`POST /api/magi/run` で `profile` 未指定の場合のみ入口LLMで自動ルーティング
- 入口LLMは JSON (`intent`, `complexity`, `profile`, `confidence`, `reason`) を返し、`min_confidence` 未満は `default_profile` にフォールバック
- 例: ローカル Ollama で `provider=ollama`, `model=qwen2.5:7b-instruct-q4_K_M`

履歴の扱い:
- データは削除せず保持
- `validity_state` (`active|stale|superseded`) を保持
- 類似検索は `similarity × freshness × validity_weight` でスコアリング
- `deprecations` の `current_terms` を含む新規実行が入ると、`legacy_terms` を含む過去履歴を `superseded` に更新
- 現在のサンプルは `nextauth->auth.js`, `pages router->app router`, `react-query->tanstack query` などを含む

## API Notes

- Endpoint: `POST /api/magi/run`
- Request body: `{ "prompt": "...", "profile": "cost|balance|performance|ultra", "fresh_mode": false, "thread_id": "optional-string" }`
- Run response includes `consensus` (synthesized final answer from peer-vote deliberation)
- Run response includes `thread_id` and `turn_index`
- Run responses are persisted to local SQLite history
- Retry endpoint: `POST /api/magi/retry`
- Retry body: `{ "prompt": "...", "agent": "A|B|C", "profile": "...", "fresh_mode": false, "thread_id": "optional-string" }`
- Consensus endpoint: `POST /api/magi/consensus`
- Consensus body: `{ "prompt": "...", "results": AgentResult[], "profile": "...", "fresh_mode": false, "thread_id": "optional-string" }`
- Profiles endpoint: `GET /api/magi/profiles`
- History list endpoint: `GET /api/magi/history?limit=20&offset=0`
- History detail endpoint: `GET /api/magi/history/{run_id}`
- History thread delete endpoint: `DELETE /api/magi/history/thread/{thread_id}`
- For same `thread_id`, backend injects thread memory into effective prompt; latest turn is injected as a dedicated `[High Priority Latest Turn]` block.
- Empty prompt or over 4000 chars returns `400`
- Per-model timeout: profile config (`backend/config.json`) に従う（現行: cost 25s / balance 35s / performance 45s / ultra 60s）
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
  - choose `cost`, `balance`, `performance`, `ultra`
  - selected profile is sent on run/retry/consensus
  - default profile is `balance` (from `backend/config.json`)
  - `performance` and `ultra` enable strict debate consensus (requires concrete cross-agent criticisms)
  - UI shows a `strict debate` badge when `performance` or `ultra` is selected
  - UI shows a `high cost` badge when `ultra` is selected
- Fresh mode toggle:
  - default is ON
  - when ON, backend retrieves recent web evidence via Tavily (if `TAVILY_API_KEY` is configured)
  - retrieval uses multi-attempt fallback (`general/news` + query expansion) for non-news topics like game guides
  - if Tavily is unavailable or key is missing, it falls back to normal prompt automatically
- Run history panel (persisted in backend SQLite)
- `run_id` display and copy button
- `thread_id` display（同一チャットで継続）
- Thread panel supports grouped turns per thread
- Thread actions are modernized: inline `Rename`, icon-style `Fold/Expand`, and confirmation-based `Delete`

## Troubleshooting (Local)

- `localhost:3000 refused to connect`: start frontend (`.\start-frontend.ps1`).
- `status=ERROR` for one model: verify API key and model name in `backend/config.json`.
- Anthropic/Gemini errors while others work: usually provider-side quota/credits.
- Docker で履歴が消える: `docker-compose.yml` で `./backend/data:/app/data` がマウントされていること、かつ `MAGI_DB_PATH` は未設定または `data/magi.db` を使用する。
