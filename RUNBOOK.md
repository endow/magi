# RUNBOOK.md

ローカル開発およびDocker実行時の運用コマンド集です。

対象リリース基準: `v1.0`

## ローカル起動

Backend:

```powershell
.\start-backend.ps1
```

macOS / Linux:

```bash
./start-backend.sh
```

Frontend:

```powershell
.\start-frontend.ps1
```

macOS / Linux:

```bash
./start-frontend.sh
```

## Docker起動 / 再ビルド

全体を起動または更新:

```bash
docker compose up --build -d
```

状態確認:

```bash
docker compose ps
```

停止:

```bash
docker compose down
```

## Router / Ollama確認（v1.0）

Ollamaおよびルーター関連サービスを確認:

```bash
docker compose ps
```

期待されるサービス:
- `magi-backend`
- `magi-frontend`
- `magi-ollama`

ローカルモデルを明示的に取得/再取得:

```bash
docker compose up ollama-pull
```

ルーターログ確認:

```bash
docker logs --tail 120 magi-backend
```

確認ポイント:
- 成功: `[magi] request_router success ...`
- 信頼度によるフォールバック: `[magi] request_router fallback_by_confidence ...`
- 失敗（network/provider）: `[magi] request_router failed ...`

暗黙シグナル送信（routing learning補助）:

```powershell
Invoke-RestMethod -Method Post http://localhost:8000/api/magi/routing/signal `
  -ContentType "application/json" `
  -Body '{"thread_id":"<thread_uuid>","request_id":"<run_id>","signal":"copy_result"}'
```

macOS / Linux:

```bash
curl -fsS -X POST "http://localhost:8000/api/magi/routing/signal" \
  -H "Content-Type: application/json" \
  -d '{"thread_id":"<thread_uuid>","request_id":"<run_id>","signal":"copy_result"}'
```

### ルーティング判定の一括テスト

リポジトリルートで実行:

```powershell
.\test-routing.ps1
```

macOS / Linux:

```bash
./test-routing.sh
```

境界ケースも含める場合:

```powershell
.\test-routing.ps1 -IncludeBoundary
```

macOS / Linux:

```bash
./test-routing.sh --include-boundary
```

別URLへ向ける場合:

```powershell
.\test-routing.ps1 -ApiBaseUrl http://host.docker.internal:8000
```

macOS / Linux:

```bash
./test-routing.sh --api-base-url=http://host.docker.internal:8000
```

## クイックヘルスチェック

Backend:

```powershell
Invoke-WebRequest -UseBasicParsing http://localhost:8000/health
```

macOS / Linux:

```bash
curl -fsS http://localhost:8000/health
```

Frontend:

```powershell
Invoke-WebRequest -UseBasicParsing http://localhost:3000
```

macOS / Linux:

```bash
curl -fsS http://localhost:3000
```

履歴API:

```powershell
Invoke-RestMethod -Method Get http://localhost:8000/api/magi/history?limit=5&offset=0 | ConvertTo-Json -Depth 6
```

macOS / Linux:

```bash
curl -fsS "http://localhost:8000/api/magi/history?limit=5&offset=0"
```

スレッド削除API:

```powershell
Invoke-RestMethod -Method Delete http://localhost:8000/api/magi/history/thread/<thread_id>
```

macOS / Linux:

```bash
curl -fsS -X DELETE "http://localhost:8000/api/magi/history/thread/<thread_id>"
```

## Next.jsキャッシュ復旧

開発サーバーでモジュール/チャンク不整合（例: `_document` 不足、chunk不足）が出た場合に実行します。

```powershell
if (Test-Path frontend\.next) { cmd /c rmdir /s /q frontend\.next }
cd frontend
npm run dev
```

macOS / Linux:

```bash
rm -rf frontend/.next
cd frontend
npm run dev
```

本番ビルド検証時:

```powershell
if (Test-Path frontend\.next) { cmd /c rmdir /s /q frontend\.next }
cd frontend
npm run build
```

macOS / Linux:

```bash
rm -rf frontend/.next
cd frontend
npm run build
```

## Backend警告

- `trio._core._multierror RuntimeWarning` は環境フックの影響で表示される場合があります。
- アプリ本体ではこの警告をコード側で抑制しています。単発スクリプトで表示されても、通常は致命的ではありません。

## プロバイダのクォータエラー

- Geminiで単一エージェントが `ERROR` になる場合、backendログで `429` / `RESOURCE_EXHAUSTED` を確認してください。
- これはアプリクラッシュではなく、プロバイダ側クォータ/レート制限を示します。
- 時間を置いて再試行するか、プロバイダ側のクォータ/課金上限を引き上げてください。

## Fresh Mode（最新情報）

- `backend/.env` に `TAVILY_API_KEY` を設定すると Fresh mode のWeb取得が有効になります。
- `source_urls` を明示しなくても、`prompt` 内の `http/https` URL は backend が自動抽出して直取得します。
- URLアンカー付きリクエスト（`source_urls` 指定 or `prompt` 内URL含有）では、`history_context` は自動スキップされます。
- `fresh_mode` 自動判定語には `YouTube/動画/攻略動画` も含まれます。
- 任意チューニング:
  - `FRESH_MAX_RESULTS` (default `3`, max `10`)
  - `FRESH_CACHE_TTL_SECONDS` (default `1800`)
  - `FRESH_SEARCH_DEPTH` (`basic` or `advanced`)
  - `FRESH_PRIMARY_TOPIC` (`general` or `news`, default `general`)
- キー未設定、または Tavily リクエスト失敗時は通常プロンプトへフォールバックします（run自体は失敗しません）。

## 履歴DB

- デフォルトDBパス: `backend/data/magi.db`
- `backend/.env` の `MAGI_DB_PATH` で上書き可能
- 推奨上書き（Docker/Local共通）: `MAGI_DB_PATH=data/magi.db`
- Dockerは `./backend/data:/app/data` をバインドマウントします。これがない場合、コンテナ再作成時に履歴が失われる可能性があります。
- 履歴が不要なら backend 停止中にDBファイルを削除できます。次回起動時に再生成されます。

## Consensusモード

- `cost` / `balance`: 通常の peer-vote consensus
- `performance` / `ultra`: strict debate consensus（`min_criticisms=2`）
- strict mode で consensus エラーが出る場合は、backendログの `strict debate requires at least ... criticisms` を確認してください。
