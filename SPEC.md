# MAGI v0.7 実装仕様（現行）

あなたはソフトウェア開発エージェントです。  
MVPから拡張された現行版「MAGI」を実装・維持してください。

---

## 目的（最重要）

- ユーザーが1回プロンプトを入力するだけで、同じ質問を3つのLLMに同時に投げ、3つの回答を並べて見られるようにする。
- 3モデル回答に加え、合議結果（consensus）を返し、履歴として保存・再参照できるようにする。

---

## スコープ

- backend: FastAPI + LiteLLM  
- frontend: Next.js（App Router）+ Tailwind CSS  
- 1つのリポジトリに `backend/` と `frontend/` を置く（monorepoでOK）  
- ローカル開発最優先。Docker Composeでも同等に起動できる構成にする。

---

## 要件（v0.7）

### 1) バックエンド

- `POST /api/magi/run` を実装する。

#### リクエスト
```json
{ "prompt": "string", "profile": "cost|balance|performance", "fresh_mode": false }
```

#### レスポンス（成功時）
```json
{
  "run_id": "uuid-string",
  "profile": "performance",
  "results": [
    {
      "agent": "A",
      "provider": "openai",
      "model": "gpt-4.1-mini",
      "text": "...",
      "status": "OK",
      "latency_ms": 1234
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
      "model": "gemini-2.5-flash",
      "text": "...",
      "status": "OK",
      "latency_ms": 980
    }
  ],
  "consensus": {
    "provider": "magi",
    "model": "peer_vote_r1",
    "text": "...",
    "status": "OK",
    "latency_ms": 1200
  }
}
```

---

### 並列実行とエラーハンドリング

- 3つのモデルへ **同時並列** で問い合わせる（`asyncio.gather` を使用）。
- **各タスク内で try/except を行い、必ず結果オブジェクトを返す**。
- 失敗したモデルがあっても全体は落とさない。
- タイムアウトは **各モデルごとに20秒** を適用（`asyncio.wait_for`）。
- `run_id` はUUIDで毎回発行し、レスポンスに含める（履歴はSQLiteに保存する）。
- 3モデルの結果を入力にして、**3モデル同士の相互レビュー＋投票で合議（consensus）** を実行する。
- 合議が失敗しても全体レスポンスは返し、`consensus.status="ERROR"` を返す。

---

### LiteLLM 呼び出し仕様（重要）

- LiteLLM の **async API（例: `acompletion`）** を使用。
- messages形式は **OpenAI互換**で統一：

```python
messages = [{"role": "user", "content": prompt}]
```

- system メッセージ・temperature等は **v0では指定しない**。

#### model文字列の組み立てルール

- LiteLLM に渡す model は `"provider/model"` 形式：

- `openai/gpt-4.1-mini`
- `anthropic/claude-sonnet-4-20250514`
- `gemini/gemini-2.5-flash`

---

### モデル設定

`backend/config.json`

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

- `GET /api/magi/profiles` で利用可能profile一覧を返す。

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
GOOGLE_API_KEY=
TAVILY_API_KEY=
FRESH_MAX_RESULTS=3
FRESH_CACHE_TTL_SECONDS=1800
FRESH_SEARCH_DEPTH=basic
FRESH_PRIMARY_TOPIC=general
```

- geminiでも **GOOGLE_API_KEY** を使用。
- 実装では互換性のため、`GOOGLE_API_KEY` を `GEMINI_API_KEY` に内部マッピングして利用する。

---

### CORS / ポート

- backend: http://localhost:8000  
- frontend: http://localhost:3000  
- CORS許可: http://localhost:3000  

---

### ログ

- 各エージェントの開始・成功・エラーを標準出力へ。
- 合議（consensus）の開始・成功・エラーを標準出力へ。

---

### 合議再計算API（v0.7）

- `POST /api/magi/consensus`
- リクエスト:
  - `prompt`
  - `results`（A/B/Cの結果配列）
  - `fresh_mode`（true の場合は再検索して最新コンテキストを付与）
- 用途:
  - フロント側の単体Retry後に、最新結果で合議を再計算するため

---

### 最新情報対策（Fresh mode）

- `fresh_mode=true` のとき、backend は Tavily 検索で最新のWebソースを取得し、プロンプトに前段コンテキストとして付与する。
- 検索は `general/news` のフォールバックとクエリ拡張を行い、ニュース以外の「攻略・解説」系トピックも拾えるようにする。
- 取得ソースには `url` と `published_date` を含め、時系列依存の回答で出典明示を促す。
- `TAVILY_API_KEY` 未設定、または検索失敗時は通常プロンプトへ自動フォールバックする（リクエスト自体は失敗させない）。

---

## 2) フロントエンド

- 1画面完結。
- textarea + 送信ボタン。
- **3カラム表示**。

### 表示項目

- provider/model
- status
- text
- latency_ms
- error_message
- consensus（合議結果）

### UI

- 送信直後に3枠表示
- `Loading...`
- **run_id表示＋コピー可能**
- **合議結果は3カラムより先（上部）に表示する**

### デザイン

- 黒基調ターミナル風をベースに、MAGIライクな可視化（ノード、接続線、状態表示）を許容
- 合議結果（Conclusion）を視覚的に優先表示
- lucide-react可

---

## 3) README

- 前提環境
- 起動手順
- 構成
- APIキー
- 接続先
- `backend/.env.example` を作成し、必要な環境変数キーを記載

---

## 4) 接続

- `NEXT_PUBLIC_API_BASE_URL`
- 既定: http://localhost:8000

---

## 実装済み拡張: v0.5（履歴永続化）

- SQLiteに `run` 結果を保存して、過去実行を見返せるようにする。
- 保存対象:
  - run_id / created_at / profile / prompt
  - A/B/C 各結果（status, text, latency, error）
  - consensus結果（status, text, latency, error）
- 追加API:
  - `GET /api/magi/history?limit=20&offset=0`
  - `GET /api/magi/history/{run_id}`

## 実装済み拡張: v0.7（strict debate consensus + fresh retrieval improvements）

- `performance` profile のみ、合議を strict debate モードで実行する。
- 各エージェントは投票時に `criticisms`（他案の具体的弱点）を最低2件返す。
- `criticisms` 不足のターンは `ERROR` 扱いにして無効票とする。
- 勝者選定は「票数 + 信頼度 + 批判品質スコア」で重み付けする。

---

## Definition of Done

- 3モデル並列表示
- 部分失敗OK
- 空入力不可
- 接続失敗判別可
- config差し替えでモデル変更可
