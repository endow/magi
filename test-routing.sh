#!/usr/bin/env bash
set -euo pipefail

API_BASE_URL="${API_BASE_URL:-http://localhost:8000}"
INCLUDE_BOUNDARY=0

for arg in "$@"; do
  case "$arg" in
    --include-boundary)
      INCLUDE_BOUNDARY=1
      ;;
    --api-base-url=*)
      API_BASE_URL="${arg#*=}"
      ;;
    *)
      echo "unknown argument: $arg" >&2
      echo "usage: ./test-routing.sh [--include-boundary] [--api-base-url=http://localhost:8000]" >&2
      exit 1
      ;;
  esac
done

ENDPOINT="${API_BASE_URL%/}/api/magi/run"

rows=()

run_case() {
  case_id="$1"
  expected="$2"
  note="$3"
  prompt="$4"

  started_ms="$(python3 -c 'import time; print(int(time.time() * 1000))')"

  payload="$(PROMPT="$prompt" python3 - <<'PY'
import json, os
print(json.dumps({"prompt": os.environ["PROMPT"], "fresh_mode": False}, ensure_ascii=False))
PY
)"

  if ! response="$(curl -sS -X POST "$ENDPOINT" -H "Content-Type: application/json" -d "$payload")"; then
    rows+=("${case_id}|${expected}|ERROR|false|0|ERROR|0|request failed")
    return
  fi

  line="$(CASE_ID="$case_id" EXPECTED="$expected" NOTE="$note" STARTED_MS="$started_ms" RESPONSE="$response" python3 - <<'PY'
import json, os, time

case_id = os.environ["CASE_ID"]
expected = os.environ["EXPECTED"]
note = os.environ["NOTE"]
started_ms = int(os.environ["STARTED_MS"])

try:
    resp = json.loads(os.environ["RESPONSE"])
except Exception as exc:  # noqa: BLE001
    print(f"{case_id}|{expected}|ERROR|false|0|ERROR|0|invalid json: {exc}")
    raise SystemExit(0)

actual = str(resp.get("profile", ""))
results = resp.get("results") or []
consensus = (resp.get("consensus") or {}).get("status", "")
agent_count = len(results)
latency = int(time.time() * 1000) - started_ms
ok = "true" if actual == expected else "false"
print(f"{case_id}|{expected}|{actual}|{ok}|{agent_count}|{consensus}|{latency}|{note}")
PY
)"
  rows+=("$line")
}

run_case "L1" "local_only" "rewrite/low/safety-low" "日本語の敬語で、以下を丁寧に言い換えて：『明日行けない』"
run_case "L2" "local_only" "translation/low/safety-low" "次を英訳して：『本日の会議は延期になりました。』"
run_case "L3" "local_only" "summarize_short/low/safety-low" "以下を1文で短く要約して：『この機能は初回起動時に設定ファイルを読み込み、存在しない場合はデフォルト値で初期化する。』"
run_case "C1" "cost" "complexity-high" "マイクロサービス移行の設計方針を、リスクと段階計画つきで提案して"
run_case "C2" "cost" "freshness/finance" "最新の米国金利動向を踏まえて、投資戦略を提案して"
run_case "C3" "cost" "safety-high/legal" "この契約条項の法的リスクを評価して"

if [ "$INCLUDE_BOUNDARY" -eq 1 ]; then
  run_case "B1" "cost" "boundary/underspecified" "以下を要約して。"
  run_case "B2" "cost" "boundary/mixed-intent" "翻訳して。あとAWS構成も設計して。"
fi

printf "%-6s %-11s %-11s %-5s %-11s %-10s %-10s %s\n" "case" "expected" "actual" "pass" "agent_count" "consensus" "latency" "note"
printf "%-6s %-11s %-11s %-5s %-11s %-10s %-10s %s\n" "------" "-----------" "-----------" "-----" "-----------" "----------" "----------" "----"

passed=0
total=0
for row in "${rows[@]}"; do
  IFS='|' read -r case_id expected actual pass agent_count consensus latency note <<EOF
$row
EOF
  printf "%-6s %-11s %-11s %-5s %-11s %-10s %-10s %s\n" "$case_id" "$expected" "$actual" "$pass" "$agent_count" "$consensus" "$latency" "$note"
  total=$((total + 1))
  if [ "$pass" = "true" ]; then
    passed=$((passed + 1))
  fi
done

failed=$((total - passed))
echo
echo "summary: ${passed}/${total} passed, ${failed} failed"

if [ "$failed" -gt 0 ]; then
  exit 1
fi
