[CmdletBinding()]
param(
    [string]$ApiBaseUrl = "http://localhost:8000",
    [switch]$IncludeBoundary
)

$ErrorActionPreference = "Stop"

$cases = @(
    @{
        id = "L1"
        expected = "local_only"
        prompt = "日本語の敬語で、以下を丁寧に言い換えて：『明日行けない』"
        note = "rewrite/low/safety-low"
    },
    @{
        id = "L2"
        expected = "local_only"
        prompt = "次を英訳して：『本日の会議は延期になりました。』"
        note = "translation/low/safety-low"
    },
    @{
        id = "L3"
        expected = "local_only"
        prompt = "以下を1文で短く要約して：『この機能は初回起動時に設定ファイルを読み込み、存在しない場合はデフォルト値で初期化する。』"
        note = "summarize_short/low/safety-low"
    },
    @{
        id = "C1"
        expected = "cost"
        prompt = "マイクロサービス移行の設計方針を、リスクと段階計画つきで提案して"
        note = "complexity-high"
    },
    @{
        id = "C2"
        expected = "cost"
        prompt = "最新の米国金利動向を踏まえて、投資戦略を提案して"
        note = "freshness/finance"
    },
    @{
        id = "C3"
        expected = "cost"
        prompt = "この契約条項の法的リスクを評価して"
        note = "safety-high/legal"
    }
)

if ($IncludeBoundary) {
    $cases += @(
        @{
            id = "B1"
            expected = "cost"
            prompt = "以下を要約して。"
            note = "boundary/underspecified"
        },
        @{
            id = "B2"
            expected = "cost"
            prompt = "翻訳して。あとAWS構成も設計して。"
            note = "boundary/mixed-intent"
        }
    )
}

$endpoint = "$($ApiBaseUrl.TrimEnd('/'))/api/magi/run"
$results = @()

foreach ($case in $cases) {
    $body = @{
        prompt = $case.prompt
        fresh_mode = $false
    } | ConvertTo-Json -Compress

    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    try {
        $resp = Invoke-RestMethod -Method Post -Uri $endpoint -ContentType "application/json" -Body $body
        $sw.Stop()
        $actual = [string]$resp.profile
        $agentCount = @($resp.results).Count
        $consensusStatus = [string]$resp.consensus.status
        $ok = $actual -eq $case.expected
        $results += [pscustomobject]@{
            case_id = $case.id
            expected = $case.expected
            actual = $actual
            pass = $ok
            agent_count = $agentCount
            consensus = $consensusStatus
            latency_ms = [int]$sw.ElapsedMilliseconds
            note = $case.note
        }
    } catch {
        $sw.Stop()
        $results += [pscustomobject]@{
            case_id = $case.id
            expected = $case.expected
            actual = "ERROR"
            pass = $false
            agent_count = 0
            consensus = "ERROR"
            latency_ms = [int]$sw.ElapsedMilliseconds
            note = $_.Exception.Message
        }
    }
}

$results | Format-Table -AutoSize

$total = $results.Count
$passed = @($results | Where-Object { $_.pass }).Count
$failed = $total - $passed
Write-Host ""
Write-Host "summary: $passed/$total passed, $failed failed"

if ($failed -gt 0) {
    exit 1
}

