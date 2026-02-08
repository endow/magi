# MAGI v0.1 実装仕様（完成版プロンプト）

あなたはソフトウェア開発エージェントです。  
最小のMVP「MAGI v0」を実装してください。

---

## 目的（最重要）

- ユーザーが1回プロンプトを入力するだけで、同じ質問を3つのLLMに同時に投げ、3つの回答を並べて見られるようにする。
- まだ「合議」や「人格」や「DB」は不要。まずは **“コピペの面倒” を消す。**

---

## スコープ

- backend: FastAPI + LiteLLM  
- frontend: Next.js（App Router）+ Tailwind CSS  
- 1つのリポジトリに `backend/` と `frontend/` を置く（monorepoでOK）  
- ローカル開発最優先。Dockerは不要（後で追加しやすい構成にする）

---

## 要件（v0）

### 1) バックエンド

- `POST /api/magi/run` を実装する。

#### リクエスト
```json
{ "prompt": "string" }
```

#### レスポンス（成功時）
```json
{
  "run_id": "uuid-string",
  "results": [
    {
      "agent": "A",
      "provider": "openai",
      "model": "gpt-4o-mini",
      "text": "...",
      "status": "OK",
      "latency_ms": 1234
    },
    {
      "agent": "B",
      "provider": "anthropic",
      "model": "claude-haiku-4-5-20251001",
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
  ]
}
```

---

### 並列実行とエラーハンドリング

- 3つのモデルへ **同時並列** で問い合わせる（`asyncio.gather` を使用）。
- **各タスク内で try/except を行い、必ず結果オブジェクトを返す**。
- 失敗したモデルがあっても全体は落とさない。
- タイムアウトは **各モデルごとに20秒** を適用（`asyncio.wait_for`）。
- `run_id` はUUIDで毎回発行し、レスポンスに含める（DB保存はしない）。

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

- `openai/gpt-4o-mini`
- `anthropic/claude-haiku-4-5-20251001`
- `gemini/gemini-2.5-flash`

---

### モデル設定

`backend/config.json`

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

### UI

- 送信直後に3枠表示
- `Loading...`
- **run_id表示＋コピー可能**

### デザイン

- 黒ターミナル風
- 過剰装飾なし
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

## 将来拡張

- run_id中心設計
- **DBなし（v0）**

---

## Definition of Done

- 3モデル並列表示
- 部分失敗OK
- 空入力不可
- 接続失敗判別可
- config差し替えでモデル変更可
