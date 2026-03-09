# MAGI v1.5 実装仕様（現行）

本書は、MVPから拡張された現行版「MAGI」の実装仕様を定義する。  
実装・改修時は、本仕様を正とする。

---

## 目的（最重要）

- ユーザーが1回プロンプトを入力するだけで、同じ質問を3つのLLMに同時に投げ、3つの回答を並べて見られるようにする。
- 3モデル回答に加え、合議結果（consensus）を返し、履歴として保存・再参照できるようにする。
- 軽量・低リスクな文面タスクは `local_only`（ローカル1モデル）で完結させ、必要時のみ3モデル合議へ回す。

---

## スコープ

- backend: FastAPI + LiteLLM  
- frontend: Next.js（App Router）+ Tailwind CSS  
- 1つのリポジトリに `backend/` と `frontend/` を置く（monorepoでOK）  
- ローカル開発最優先。Docker Composeでも同等に起動できる構成にする。

---

## 要件（v1.5）

### 1) バックエンド

- `POST /api/magi/run` を実装する。
- `POST /api/magi/chat` を実装する（v1.5 のUI主要導線）。

#### リクエスト
```json
{ "prompt": "string", "profile": "optional: cost|balance|performance|ultra|performance_preview|ultra_preview|local_only", "fresh_mode": "optional: true|false|null(=auto)", "thread_id": "optional-string", "source_urls": ["optional-https://..."] }
```

#### レスポンス（成功時）
```json
{
  "run_id": "uuid-string",
  "thread_id": "thread-uuid-or-app-id",
  "turn_index": 1,
  "profile": "performance",
  "results": [
    {
      "agent": "A",
      "provider": "openai",
      "model": "gpt-5.4",
      "text": "...",
      "status": "OK",
      "latency_ms": 1234,
      "prompt_tokens": 120,
      "completion_tokens": 240,
      "total_tokens": 360,
      "cost_estimate_usd": 0.0012
    },
    {
      "agent": "B",
      "provider": "anthropic",
      "model": "claude-sonnet-4-20250514",
      "text": "...",
      "status": "ERROR",
      "latency_ms": 20012,
      "error_message": "timeout"
    },
    {
      "agent": "C",
      "provider": "gemini",
      "model": "gemini-3-flash-preview",
      "text": "...",
      "status": "OK",
      "latency_ms": 980
    }
  ],
  "consensus": {
    "provider": "magi",
    "model": "peer_vote_r1",
    "text": "",
    "status": "LOADING",
    "latency_ms": 0,
    "error_code": null
  }
}
```

---

### 並列実行とエラーハンドリング

- 3つのモデルへ **同時並列** で問い合わせる（`asyncio.gather` を使用）。
- **各タスク内で try/except を行い、必ず結果オブジェクトを返す**。
- 失敗したモデルがあっても全体は落とさない。
- タイムアウトは **profileごとに `backend/config.json` の `timeout_seconds`** を適用（`asyncio.wait_for`）。
- 現行設定は `local_only: 30s / cost: 40s / balance: 45s / performance: 35s / ultra: 45s`。
- OpenAI/Gemini は timeout 時に 1 回リトライする（Anthropic/Ollama は単回）。
- `run_id` はUUIDで毎回発行し、レスポンスに含める（履歴はSQLiteに保存する）。
- `thread_id` は会話スレッドを識別し、未指定時は新規生成する。
- `run/retry/consensus` で渡される `thread_id` は UUID 形式のみ有効とし、非UUIDは無視して新規UUIDへフォールバックする。
- 同一 `thread_id` のときは直近ターン文脈をプロンプトに注入し、代名詞参照（「それ」など）に対応する。
- ただし `local_only` では thread/history 文脈注入をスキップし、単発処理を優先する。
- リクエストがURLアンカー（`source_urls` 指定または `prompt` 内に `http/https` URL を含む）の場合、`history_context` はスキップし、古い履歴の混入を抑制する。
- `cost|balance|performance|ultra` は 3モデル結果を入力にして、**3モデル同士の相互レビュー＋投票で合議（consensus）** を実行する。
- `local_only` では合議再生成を行わず、Agent Aの結果を consensus にパススルーする。
- 合議が失敗しても全体レスポンスは返し、`consensus.status="ERROR"` を返す。
- 合議失敗時は `consensus.error_code` を返し、失敗原因を機械可読にする。

---

### LiteLLM 呼び出し仕様（重要）

- LiteLLM の **async API（例: `acompletion`）** を使用。
- messages形式は **OpenAI互換**で統一：

```python
messages = [{"role": "user", "content": prompt}]
```

- system メッセージ・temperature等は指定しない（必要時は仕様変更で明示する）。

#### model文字列の組み立てルール

- LiteLLM に渡す model は `"provider/model"` 形式：

- `openai/gpt-5.4`
- `anthropic/claude-sonnet-4-20250514`
- `gemini/gemini-3-flash-preview`

---

### モデル設定

`backend/config.json`

```json
{
  "default_profile": "balance",
  "request_router": {
    "enabled": true,
    "provider": "ollama",
    "model": "qwen2.5:7b-instruct-q4_K_M",
    "timeout_seconds": 20,
    "min_confidence": 75
  },
  "router_rules": {
    "default_profile": "balance",
    "routes": [
      {
        "when_intents_any": ["translation", "rewrite", "summarize_short"],
        "when_complexity_any": ["low"],
        "when_safety_any": ["low"],
        "when_execution_tiers_any": ["local"],
        "profile": "local_only"
      }
    ]
  },
  "routing_learning": {
    "enabled": true,
    "alpha": 0.05,
    "weight_min": -2.0,
    "weight_max": 2.0,
    "latency_threshold_ms": 8000,
    "cost_threshold": 2.0,
    "decay_lambda_per_day": 0.05,
    "stats_ema_beta": 0.2,
    "epsilon": 0.05
  },
  "temporal_classifier": {
    "enabled": true,
    "provider": "ollama",
    "model": "qwen2.5:7b-instruct-q4_K_M",
    "timeout_seconds": 3,
    "cache_ttl_seconds": 300,
    "min_confidence": 0.5,
    "rules": [
      { "domain": "geopolitics", "force_fresh": true, "min_confidence": 0.45 }
    ]
  },
  "history_context": {
    "strategy": "embedding|lexical",
    "provider": "openai",
    "model": "text-embedding-3-small",
    "timeout_seconds": 12,
    "batch_size": 24,
    "freshness_half_life_days": 180,
    "stale_weight": 0.55,
    "superseded_weight": 0.20,
    "deprecations_source": {
      "enabled": false,
      "url": "https://example.com/magi/deprecations.json",
      "mode": "merge|replace",
      "timeout_seconds": 5,
      "refresh_interval_seconds": 86400
    },
    "deprecations": [
      {
        "id": "example-migration",
        "legacy_terms": ["old term"],
        "current_terms": ["new term"]
      }
    ]
  },
  "semantic_memory": {
    "enabled": false,
    "max_references": 2,
    "min_confidence": 0.75,
    "default_ttl_days": 180,
    "use_llm_extractor": true,
    "extractor_provider": "ollama",
    "extractor_model": "qwen2.5:7b-instruct-q4_K_M",
    "extractor_timeout_seconds": 6,
    "extractor_max_items": 3,
    "merge_similarity_threshold": 0.88
  },
  "repo_knowledge": {
    "enabled": true,
    "root": ".",
    "include_globs": ["README.md", "SPEC.md", "RUNBOOK.md", "backend/**/*.py", "frontend/**/*.tsx"],
    "exclude_globs": ["backend/data/**", "frontend/.next/**", "frontend/node_modules/**"],
    "max_files": 4,
    "max_file_chars": 20000,
    "snippet_chars": 900,
    "min_score": 0.08
  },
  "profiles": {
    "local_only": { "...": "..." },
    "cost": { "...": "..." },
    "balance": { "...": "..." },
    "performance": { "...": "..." },
    "ultra": { "...": "..." }
  }
}
```

- `GET /api/magi/profiles` で利用可能profile一覧を返す。
- `GET /api/magi/health/models` で設定モデルの健全性（`OK/WARN/ERROR`）を返す。
- `GET /api/magi/usage/summary` で provider別/全体の累計使用額を返し、`billing.provider_limits_usd` 設定時は残額も返す。
- `POST /api/magi/run` で `profile` 未指定時、`request_router.enabled=true` なら入口LLMが分類してprofileを自動選択する。
- ルーター出力は `intent/complexity/safety/execution_tier/profile/confidence/reason/needs_web/needs_tools/estimated_steps/ambiguity/escalation_hint` のJSONを想定し、`confidence < min_confidence` は `router_rules.default_profile` にフォールバックする。
- `profiles` は **最低1エージェント** を許容する（`local_only` は1エージェント想定）。
- Router学習（MVP）:
  - `routing_events` に router入出力・実行結果・user feedback を保存する
  - `routing_policy` に key別の profile weight/stats と昇格しきい値を保存する
  - 最終profileは `base_score + policy_weight` で選択する
  - 非 `local_only` の自動ルーティングでは `epsilon-greedy` 探索を行い、低確率で別profileを試す
  - feedback / 実行結果を契機に `weight += alpha * reward` で更新し、`[weight_min, weight_max]` でclampする
  - 重みは `decay_lambda_per_day` で時間減衰し、`stats.avg_reward` は `stats_ema_beta` 指定時にEMAで更新する
- `history_context.strategy=embedding` の場合、履歴類似検索は外部埋め込みモデルを使う（失敗時は lexical にフォールバック）。
- `semantic_memory.enabled=true` の場合、実行結果から抽出した記憶を `semantic_memories` に保存し、同一 `thread_id` の後続実行時に最大 `max_references` 件を補助コンテキストとして注入する。
- `use_llm_extractor=true` の場合、抽出は LLM を優先し、JSON パース失敗/空結果時はルールベース抽出にフォールバックする。
- 保存時は `thread_id + kind` 内で `merge_similarity_threshold` 以上の同義文を同一 memory として統合し、重複蓄積を抑制する。
- semantic memory 抽出/保存に失敗しても本リクエストは失敗させない（best-effort）。
- `repo_knowledge.enabled=true` の場合、ローカルリポジトリ内の許可ファイルを質問ごとに検索し、一致した断片を `[System Knowledge]` としてプロンプトへ注入する。
- repository knowledge は会話履歴ではなく、ローカルファイルを現在のシステム事実ソースとして扱う。
- 履歴は削除せず保持し、`validity_state(active|stale|superseded)` と鮮度減衰を使って参照重みを調整する。
- `deprecations_source.enabled=true` の場合、外部JSONを取得して `deprecations` を `merge|replace` で解決する。取得失敗時はローカル `deprecations` へフォールバックする。
- `deprecations` で技術移行ルール（legacy/current）を定義し、current語を含む新規実行時に過去legacy履歴を `superseded` に更新する。

---

### バリデーション

- `prompt.strip()` が空 → **400**
- 最大文字数（**4000文字**）超過 → **400**

```json
{"detail":"prompt must not be empty"}
```

---

### latency_ms

- **各モデル呼び出し開始 → 成功/エラー確定まで**

---

### 環境変数

```
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GEMINI_API_KEY=
TAVILY_API_KEY=
FRESH_MAX_RESULTS=3
FRESH_CACHE_TTL_SECONDS=1800
FRESH_SEARCH_DEPTH=basic
FRESH_PRIMARY_TOPIC=general
SOURCE_MAX_URLS=5
SOURCE_FETCH_TIMEOUT_SECONDS=10
SOURCE_SNIPPET_CHARS=1200
HISTORY_CONTEXT_ENABLED=1
HISTORY_SIMILARITY_THRESHOLD=0.78
HISTORY_SIMILAR_CANDIDATES=120
HISTORY_MAX_REFERENCES=2
HISTORY_FRESHNESS_HALF_LIFE_DAYS=180
HISTORY_STALE_WEIGHT=0.55
HISTORY_SUPERSEDED_WEIGHT=0.20
THREAD_CONTEXT_ENABLED=1
THREAD_CONTEXT_MAX_TURNS=6
OLLAMA_API_BASE=http://ollama:11434
```

---

### CORS / ポート

- backend: http://localhost:8000  
- frontend: http://localhost:3000  
- CORS許可:
  - http://localhost:3000
  - http://127.0.0.1:3000
  - http://host.docker.internal:3000

---

### ログ

- 各エージェントの開始・成功・エラーを標準出力へ。
- 合議（consensus）の開始・成功・エラーを標準出力へ。

---

### 合議再計算API（現行）

- `POST /api/magi/consensus`
- リクエスト:
  - `prompt`
  - `results`（`AgentResult[]`。通常はA/B/C、`local_only` は1件）
  - `fresh_mode`（true の場合は再検索して最新コンテキストを付与）
- 用途:
  - フロント側の単体Retry後に、最新結果で合議を再計算するため
- 備考:
  - `local_only` では再合議せず、受け取ったAgent結果を consensus として返す

### ルーティング学習API（v1.5）

- `POST /api/magi/routing/feedback`
  - body: `{ thread_id, request_id, rating: -1|0|1, reason? }`
  - 対象 event の `user_rating/user_reason` を更新し、policy update を即時実行
- `GET /api/magi/routing/policy?key=...`
  - key の `weights/stats` を返す
- `GET /api/magi/routing/events?thread_id=...&limit=...`
  - ルーティングイベントのデバッグ参照
- `POST /api/magi/routing/signal`
  - body: `{ thread_id, request_id, signal }`
  - `signal`: `retry | copy_result | consensus_recalc | history_helpful | history_not_helpful | escalated_after_low_confidence | escalated_after_tool_failure | escalated_after_timeout | escalated_after_conflict | escalated_after_user_rephrase | local_completed_without_escalation | local_failed_then_balance_succeeded | balance_failed_then_performance_succeeded`
  - 暗黙シグナルを `implicit_reward` として学習報酬に加算する
- `GET /api/magi/memory?thread_id=...&limit=...`
  - 指定 `thread_id` の semantic memory 一覧を返す
- `PATCH /api/magi/memory/{memory_id}`
  - body: `{ status?, confidence?, expires_at? }`
  - semantic memory の状態/信頼度/期限を更新する
- `DELETE /api/magi/memory/{memory_id}`
  - semantic memory を論理削除（`status=invalid`）する

---

### 最新情報対策（Fresh mode）

- `fresh_mode=true` のとき、backend は Tavily 検索で最新のWebソースを取得し、プロンプトに前段コンテキストとして付与する。
- `source_urls` が未指定でも、`prompt` 内の `http/https` URL を自動抽出して直取得し、`[Direct URL Evidence]` として優先注入する。
- 検索は `general/news` のフォールバックとクエリ拡張を行い、ニュース以外の「攻略・解説」系トピックも拾えるようにする。
- `fresh_mode` 未指定時は temporal classifier（LLM判定）で `needs_fresh/domain/confidence` を推定し、policy条件（`min_confidence` と domain rules）を満たす場合に自動有効化する。
- classifier失敗時はヒューリスティック判定へフォールバックする。
- 取得ソースには `url` と `published_date` を含め、時系列依存の回答で出典明示を促す。
- `TAVILY_API_KEY` 未設定、または検索失敗時は通常プロンプトへ自動フォールバックする（リクエスト自体は失敗させない）。

---

## 2) フロントエンド

- 1画面完結。
- textarea + 送信ボタン。
- chat transcript 表示（ユーザー/アシスタントのターン履歴）。

### 表示項目

- chat reply（統合回答）
- run_id / thread_id / turn_index
- profile / fresh_mode / tokens / cost
- モデル別 status / latency_ms / error_message（実行詳細内）

### UI

- UIは **chat modeのみ**（interaction切替なし）
- 送信中は Chamber ノードと状態バッジ（`Routing / Prep`、`Executing`、`Discussion`、`Debating`、`Conclusion`）で進行状況を表示する
- **run_id表示＋コピー可能**
- 実行メタ情報 / ルーティング情報 / フィードバック / モデル実行状態はアコーディオン内に表示する
- 初期値: profile は **未設定（auto）**、`fresh_mode` は `auto`（未指定）
- profile未設定（auto）の場合、`POST /api/magi/chat` は `profile` を送らずルーター判定に委譲する
- 上段に `Local LLM` ノードを表示し、下段3ノード（合議グループ）との関係を可視化する
- `local_only` 完了時は下段3ノードを `skipped` 表示とし、誤解を避ける

### デザイン

- 黒基調ターミナル風をベースに、MAGIライクな可視化（ノード、接続線、状態表示）を許容
- Conclusion状態を視覚的に優先表示
- lucide-react可

---

## 3) 接続設定（フロント）

- `NEXT_PUBLIC_API_BASE_URL`
- 既定: `http://localhost:8000`

---

## 変更履歴（参考）

### v0.5（履歴永続化）

- SQLiteに `run` 結果を保存して、過去実行を見返せるようにする。
- 保存対象:
  - run_id / thread_id / turn_index / created_at / profile / prompt
  - A/B/C 各結果（status, text, latency, error）
  - consensus結果（status, text, latency, error）
- 主な追加API:
  - `GET /api/magi/history?limit=20&offset=0`
  - `GET /api/magi/history/{run_id}`
  - `DELETE /api/magi/history/thread/{thread_id}`

### v0.7（strict debate consensus + fresh retrieval improvements）

- `performance` / `ultra` profile では、合議を strict debate モードで実行する。
- 各エージェントは投票時に `criticisms`（他案の具体的弱点）を最低2件返す。
- `criticisms` 不足のターンは `ERROR` 扱いにして無効票とする。
- 勝者選定は「票数 + 信頼度 + 批判品質スコア」で重み付けする。

### v0.8（history-aware retrieval with lifecycle control）

- `run` 実行前に履歴DBから類似質問を検索し、プロンプトへ参照コンテキストを付与できる。
- 類似検索は `embedding`（外部埋め込み）または `lexical`（ローカル）を選択できる。
- スコアは `similarity × freshness × validity_weight` で計算する。
- 履歴行に `validity_state` / `superseded_by` / `superseded_at` を保持する。
- `deprecations` ルールで移行イベントを検知し、旧議論を `superseded` に自動更新する。

### v0.9（threaded conversation memory）

- `POST /api/magi/run` は `thread_id` を受け取り、レスポンスに `thread_id` / `turn_index` を返す。
- `thread_id` 未指定なら新規スレッドを作る。
- 同一スレッドの直近ターンを `run/retry/consensus` の有効プロンプトに注入する。
- 直近1ターンは `[High Priority Latest Turn]` ブロックとして別枠注入し、曖昧質問時に最優先参照する。
- 履歴保存に `thread_id` / `turn_index` を保持し、過去会話復元の一貫性を確保する。
- UIではスレッド単位表示・ターン表示に加え、インラインRename、折りたたみトグル、Delete確認付き削除を行える。

### v1.0（router-first execution with local_only path）

- Docker Composeに `ollama` / `ollama-pull` を追加し、ローカル推論モデルを常駐・事前取得できるようにする。
- 入口ルーター（`request_router`）を実装し、`profile` 未指定runを自動分類する。
- ルーター分類ラベルに `execution_tier` / `safety` を追加する。
- `profiles.local_only`（1エージェント: `ollama/qwen2.5:7b-instruct-q4_K_M`）を追加する。
- ルール: `translation|rewrite|summarize_short` かつ `complexity=low` かつ `safety=low` は `local_only` へ、それ以外は `cost` へフォールバックする。
- UIのprofile選択に `auto (unset)` を追加し、初期値を未設定にする。

### v1.1（routing learning + chamber execution UX）

- routing learning を追加:
  - `routing_events` / `routing_policy` テーブルを導入
  - feedback API (`/api/magi/routing/feedback`) で重み更新
  - Router最終スコアを `base + policy` に拡張
- local_only ルールを強化:
  - `execution_tier=local` 条件を追加し、cloud判定の誤配線を抑制
- 実行耐性を改善:
  - Gemini timeout 時に1回リトライ
  - `balance.timeout_seconds` を 45 秒へ調整
- Chamber UI を改善:
  - `Routing / Prep`、`Executing`、`Discussion`、`Conclusion` の状態バッジを表示
  - `Executing` で3ノード点滅を維持
  - `ERROR` カードは `Retry` 優先表示（`Copy` 非表示）

### v1.3（chat-only UI）

- フロントUIを chat mode 一本化（interaction切替を廃止）
- 主導線を `POST /api/magi/chat` に統一
- 実行メタ情報・ルーティング・フィードバック・モデル実行状態をアコーディオンへ集約

### v1.4（staged escalation gate）

- 入口ルーターの返却項目を拡張:
  - `needs_web` / `needs_tools` / `estimated_steps` / `ambiguity` / `escalation_hint`
- `routing_policy` を拡張:
  - profile重みに加えて昇格しきい値を保持
  - 例: `max_local_steps`, `min_local_confidence`, `max_retry_before_escalation`, `escalate_to_profile`
- executor側に `evaluate_escalation(...)` を導入:
  - `local_only` 実行後に `stay/escalate` を判定
  - `WEB_REQUIRED` などで必要時のみ `balance` へ昇格
- `routing/signal` を拡張:
  - 昇格成功/失敗の暗黙シグナルを記録して学習報酬へ反映
- UI/レスポンス:
  - `routing.reason` に `Escalation: <REASON> -> <profile>` を含めて可観測化

### v1.5（semantic memory）

- semantic memory 設定を追加:
  - `semantic_memory.enabled`, `max_references`, `min_confidence`, `default_ttl_days`
  - `use_llm_extractor`, `extractor_provider/model`, `extractor_timeout_seconds`, `extractor_max_items`
  - `merge_similarity_threshold`
- DBに `semantic_memories` テーブルを追加:
  - `memory_id`, `thread_id`, `source_run_id`, `kind`, `content`, `confidence`, `status`, `expires_at` などを保持
- 実行フロー:
  - `run/chat` 完了時に記憶候補を抽出（LLM優先、失敗時はルールベースにフォールバック）
  - 保存時に同義文統合（`thread_id + kind` で類似度閾値以上を同一memoryへマージ）
  - 後続ターンでは有効メモリを `[Semantic Memory]` として注入
- 運用API:
  - `GET /api/magi/memory?thread_id=...&limit=...`
  - `PATCH /api/magi/memory/{memory_id}`
  - `DELETE /api/magi/memory/{memory_id}`（論理削除）
- repository knowledge（ローカルRAG）:
  - `README.md`、`SPEC.md`、`RUNBOOK.md`、主要コードを対象に質問ごとにローカル検索する
  - 一致した断片を `[System Knowledge]` として実行プロンプトへ注入する
  - 除外対象（`node_modules`, `.next`, DB, バイナリ等）は設定で制御する

---

## Definition of Done

- 3モデル並列表示
- 部分失敗OK
- 空入力不可
- 接続失敗判別可
- config差し替えでモデル変更可
- profile未指定時の自動ルーティングが機能する
- 軽量・低リスク文面タスクが `local_only` へ到達する
- `local_only` は合議再生成なしで即時確定し、thread/history文脈注入を行わない
