import asyncio
import json
import os
import sqlite3
import uuid
import warnings
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from litellm import acompletion
from pydantic import BaseModel, Field

warnings.filterwarnings(
    "ignore",
    message="You seem to already have a custom sys.excepthook handler installed.*",
    category=RuntimeWarning,
    module="trio._core._multierror",
)

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
# LiteLLM Gemini adapter expects GEMINI_API_KEY in many environments.
if not os.getenv("GEMINI_API_KEY") and os.getenv("GOOGLE_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_API_KEY"]


class AgentConfig(BaseModel):
    agent: str
    provider: str
    model: str


class ConsensusConfig(BaseModel):
    strategy: Literal["peer_vote", "single_model"] = "peer_vote"
    provider: str | None = None
    model: str | None = None
    min_ok_results: int = 2
    rounds: int = 1
    debate_mode: Literal["normal", "strict"] = "normal"
    min_criticisms: int = 0


class ProfileConfig(BaseModel):
    agents: list[AgentConfig]
    consensus: ConsensusConfig
    timeout_seconds: int = 20


class AppConfig(BaseModel):
    default_profile: str | None = None
    profiles: dict[str, ProfileConfig] | None = None
    agents: list[AgentConfig] | None = None
    consensus: ConsensusConfig | None = None
    timeout_seconds: int | None = None


class RunRequest(BaseModel):
    prompt: str
    profile: str | None = None
    fresh_mode: bool = False


class RetryRequest(BaseModel):
    prompt: str
    agent: Literal["A", "B", "C"]
    profile: str | None = None
    fresh_mode: bool = False


class AgentResult(BaseModel):
    agent: str
    provider: str
    model: str
    text: str
    status: Literal["OK", "ERROR"]
    latency_ms: int
    error_message: str | None = None


class ConsensusRequest(BaseModel):
    prompt: str
    results: list[AgentResult]
    profile: str | None = None
    fresh_mode: bool = False


class FreshSource(BaseModel):
    title: str
    url: str
    snippet: str
    published_date: str | None = None


class ConsensusResult(BaseModel):
    provider: str
    model: str
    text: str
    status: Literal["OK", "ERROR"]
    latency_ms: int
    error_message: str | None = None


class DeliberationTurn(BaseModel):
    agent: str
    provider: str
    model: str
    status: Literal["OK", "ERROR"]
    latency_ms: int
    raw_text: str
    revised_answer: str
    preferred_agent: Literal["A", "B", "C"] | None = None
    reason: str | None = None
    confidence: int | None = None
    criticisms: list[str] = Field(default_factory=list)
    error_message: str | None = None


class RunResponse(BaseModel):
    run_id: str
    profile: str
    results: list[AgentResult]
    consensus: ConsensusResult


class RetryResponse(BaseModel):
    run_id: str
    profile: str
    result: AgentResult


class ConsensusResponse(BaseModel):
    run_id: str
    profile: str
    consensus: ConsensusResult


class ProfilesResponse(BaseModel):
    default_profile: str
    profiles: list[str]
    profile_agents: dict[str, list[AgentConfig]]


class HistoryItem(BaseModel):
    run_id: str
    profile: str
    prompt: str
    created_at: str
    results: list[AgentResult]
    consensus: ConsensusResult | None = None


class HistoryListResponse(BaseModel):
    total: int
    items: list[HistoryItem]


@asynccontextmanager
async def app_lifespan(_: FastAPI):
    _init_db()
    yield


app = FastAPI(title="MAGI v0.7 Backend", lifespan=app_lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_FRESH_CACHE: dict[str, tuple[float, list[FreshSource]]] = {}


def load_config() -> AppConfig:
    config_path = Path(__file__).resolve().parents[1] / "config.json"
    with config_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    return AppConfig.model_validate(data)


def _validate_profile(profile: ProfileConfig) -> None:
    if len(profile.agents) != 3:
        raise ValueError("each profile must define exactly 3 agents")


def _resolve_profile(config: AppConfig, requested_profile: str | None) -> tuple[str, ProfileConfig]:
    if config.profiles:
        if not config.default_profile:
            raise ValueError("default_profile is required when profiles are configured")

        profile_key = requested_profile or config.default_profile
        profile = config.profiles.get(profile_key)
        if profile is None:
            raise HTTPException(status_code=400, detail=f"unknown profile: {profile_key}")
        _validate_profile(profile)
        return profile_key, profile

    # Backward compatibility for legacy single-profile config.json
    if not config.agents or not config.consensus:
        raise ValueError("config.json must define either profiles or legacy agents/consensus")
    profile = ProfileConfig(
        agents=config.agents,
        consensus=config.consensus,
        timeout_seconds=config.timeout_seconds or 20,
    )
    _validate_profile(profile)
    return "default", profile


def _extract_text(response: Any) -> str:
    choices = getattr(response, "choices", None)
    if choices is None and isinstance(response, dict):
        choices = response.get("choices")
    if not choices:
        return ""

    choice = choices[0]
    message = getattr(choice, "message", None)
    if message is None and isinstance(choice, dict):
        message = choice.get("message")

    content = getattr(message, "content", None) if message is not None else None
    if content is None and isinstance(message, dict):
        content = message.get("content")

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return "" if content is None else str(content)


def _public_error_message(exc: Exception) -> str:
    message = str(exc).lower()
    if "credit balance is too low" in message:
        return "provider account has insufficient credits"
    if "quota exceeded" in message or "resource_exhausted" in message or "429" in message:
        return "provider quota exceeded (try later or upgrade plan)"
    if "api key" in message or "authentication" in message:
        return "provider authentication failed (check API key)"
    if "not found" in message and "model" in message:
        return "model not found (check config.json model name)"
    if "rate limit" in message or "too many requests" in message:
        return "provider rate limit reached"
    return "provider request failed"


def _validate_prompt(raw_prompt: str) -> str:
    prompt = raw_prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="prompt must not be empty")
    if len(prompt) > 4000:
        raise HTTPException(status_code=400, detail="prompt must be 4000 characters or fewer")
    return prompt


def _fresh_max_results() -> int:
    raw = os.getenv("FRESH_MAX_RESULTS", "3")
    try:
        return max(1, min(10, int(raw)))
    except ValueError:
        return 5


def _fresh_cache_ttl_seconds() -> int:
    raw = os.getenv("FRESH_CACHE_TTL_SECONDS", "1800")
    try:
        return max(0, int(raw))
    except ValueError:
        return 1800


def _fresh_search_depth() -> Literal["basic", "advanced"]:
    raw = os.getenv("FRESH_SEARCH_DEPTH", "basic").strip().lower()
    if raw in {"basic", "advanced"}:
        return raw  # type: ignore[return-value]
    return "basic"


def _fresh_primary_topic() -> Literal["general", "news"]:
    raw = os.getenv("FRESH_PRIMARY_TOPIC", "general").strip().lower()
    if raw in {"general", "news"}:
        return raw  # type: ignore[return-value]
    return "general"


def _cached_fresh_sources(query: str) -> list[FreshSource] | None:
    cached = _FRESH_CACHE.get(query)
    if not cached:
        return None
    created, sources = cached
    ttl_seconds = _fresh_cache_ttl_seconds()
    if ttl_seconds <= 0:
        return None
    age_seconds = perf_counter() - created
    if age_seconds > ttl_seconds:
        _FRESH_CACHE.pop(query, None)
        return None
    return sources


def _cache_fresh_sources(query: str, sources: list[FreshSource]) -> None:
    _FRESH_CACHE[query] = (perf_counter(), sources)


def _fresh_query_attempts(query: str, primary_topic: Literal["general", "news"]) -> list[tuple[str, Literal["general", "news"]]]:
    q = query.strip()
    attempts: list[tuple[str, Literal["general", "news"]]] = []

    if primary_topic == "general":
        attempts.extend(
            [
                (q, "general"),
                (f"{q} guide walkthrough strategy 解説 攻略", "general"),
            ]
        )
        lowered = q.lower()
        if "ff14" in lowered or "ffxiv" in lowered:
            attempts.append(("ffxiv savage floor 4 guide latest", "general"))
        attempts.append((q, "news"))
    else:
        attempts.extend(
            [
                (q, "news"),
                (q, "general"),
                (f"{q} guide walkthrough strategy 解説 攻略", "general"),
            ]
        )

    deduped: list[tuple[str, Literal["general", "news"]]] = []
    seen: set[str] = set()
    for attempt_query, topic in attempts:
        key = f"{topic}::{attempt_query}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append((attempt_query, topic))
    return deduped


async def _tavily_search(
    client: httpx.AsyncClient,
    tavily_api_key: str,
    query: str,
    max_results: int,
    topic: Literal["general", "news"],
) -> list[FreshSource]:
    payload = {
        "api_key": tavily_api_key,
        "query": query,
        "search_depth": _fresh_search_depth(),
        "max_results": max_results,
        "topic": topic,
        "include_answer": False,
        "include_images": False,
        "include_raw_content": False,
    }
    response = await client.post("https://api.tavily.com/search", json=payload)
    response.raise_for_status()
    data = response.json()
    raw_results = data.get("results")
    if not isinstance(raw_results, list):
        return []

    sources: list[FreshSource] = []
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        url = str(item.get("url") or "").strip()
        snippet = str(item.get("content") or "").strip()
        published = str(item.get("published_date") or "").strip() or None
        if not url or not snippet:
            continue
        sources.append(
            FreshSource(
                title=title or "Untitled",
                url=url,
                snippet=snippet,
                published_date=published,
            )
        )
        if len(sources) >= max_results:
            break
    return sources


async def _fetch_fresh_sources(query: str, max_results: int) -> list[FreshSource]:
    cached = _cached_fresh_sources(query)
    if cached is not None:
        return cached

    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        print("[magi] fresh_mode enabled but TAVILY_API_KEY is not set; fallback to normal prompt")
        return []

    primary_topic = _fresh_primary_topic()
    attempts = _fresh_query_attempts(query, primary_topic)
    seen_keys: set[str] = set()
    merged: list[FreshSource] = []

    try:
        async with httpx.AsyncClient(timeout=12.0) as client:
            for attempt_query, topic in attempts:
                found = await _tavily_search(client, tavily_api_key, attempt_query, max_results, topic)
                print(
                    f"[magi] fresh_mode search attempt topic={topic} "
                    f'query="{attempt_query[:60]}" results={len(found)}'
                )
                for item in found:
                    if item.url in seen_keys:
                        continue
                    seen_keys.add(item.url)
                    merged.append(item)
                    if len(merged) >= max_results:
                        _cache_fresh_sources(query, merged)
                        return merged
                if len(merged) >= max_results:
                    break
    except Exception as exc:  # noqa: BLE001
        print(f"[magi] fresh_mode search failed: {type(exc).__name__}: {exc}")
        return []

    _cache_fresh_sources(query, merged)
    return merged


def _inject_fresh_context(prompt: str, sources: list[FreshSource]) -> str:
    if not sources:
        return prompt
    now_utc = datetime.now(timezone.utc).isoformat()
    evidence_lines: list[str] = []
    for idx, source in enumerate(sources, start=1):
        published = source.published_date or "unknown"
        evidence_lines.append(
            f"[S{idx}] title={source.title}\n"
            f"url={source.url}\n"
            f"published={published}\n"
            f"snippet={source.snippet}"
        )
    evidence_text = "\n\n".join(evidence_lines)
    return (
        f"{prompt}\n\n"
        "[Fresh Web Evidence]\n"
        f"retrieved_at_utc={now_utc}\n"
        "Use this evidence as a primary source for time-sensitive facts.\n"
        "If a statement depends on freshness (today/latest/current), cite at least one source URL and published date.\n"
        "If evidence is insufficient, explicitly say what is uncertain.\n\n"
        f"{evidence_text}"
    )


async def _build_effective_prompt(prompt: str, fresh_mode: bool) -> str:
    if not fresh_mode:
        return prompt
    sources = await _fetch_fresh_sources(prompt, _fresh_max_results())
    if not sources:
        return prompt
    print(f"[magi] fresh_mode sources={len(sources)}")
    return _inject_fresh_context(prompt, sources)


def _db_path() -> Path:
    raw_path = os.getenv("MAGI_DB_PATH")
    if raw_path:
        return Path(raw_path)
    return Path(__file__).resolve().parents[1] / "data" / "magi.db"


def _get_db_connection() -> sqlite3.Connection:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _get_db_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                profile TEXT NOT NULL,
                prompt TEXT NOT NULL,
                created_at TEXT NOT NULL,
                consensus_provider TEXT,
                consensus_model TEXT,
                consensus_text TEXT,
                consensus_status TEXT,
                consensus_latency_ms INTEGER,
                consensus_error_message TEXT
            );
            CREATE TABLE IF NOT EXISTS agent_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                agent TEXT NOT NULL,
                provider TEXT NOT NULL,
                model TEXT NOT NULL,
                text TEXT NOT NULL,
                status TEXT NOT NULL,
                latency_ms INTEGER NOT NULL,
                error_message TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_agent_results_run_id ON agent_results(run_id);
            """
        )


def _save_run_history(
    run_id: str,
    profile: str,
    prompt: str,
    results: list[AgentResult],
    consensus: ConsensusResult,
) -> None:
    created_at = datetime.now(timezone.utc).isoformat()
    with _get_db_connection() as conn:
        conn.execute(
            """
            INSERT INTO runs (
                run_id, profile, prompt, created_at,
                consensus_provider, consensus_model, consensus_text, consensus_status,
                consensus_latency_ms, consensus_error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                profile,
                prompt,
                created_at,
                consensus.provider,
                consensus.model,
                consensus.text,
                consensus.status,
                consensus.latency_ms,
                consensus.error_message,
            ),
        )
        conn.executemany(
            """
            INSERT INTO agent_results (
                run_id, agent, provider, model, text, status, latency_ms, error_message
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id,
                    item.agent,
                    item.provider,
                    item.model,
                    item.text,
                    item.status,
                    item.latency_ms,
                    item.error_message,
                )
                for item in results
            ],
        )


def _load_agent_results_map(run_ids: list[str]) -> dict[str, list[AgentResult]]:
    if not run_ids:
        return {}
    placeholders = ",".join("?" for _ in run_ids)
    query = (
        "SELECT run_id, agent, provider, model, text, status, latency_ms, error_message "
        f"FROM agent_results WHERE run_id IN ({placeholders}) ORDER BY id ASC"
    )
    by_run: dict[str, list[AgentResult]] = defaultdict(list)
    with _get_db_connection() as conn:
        rows = conn.execute(query, run_ids).fetchall()
    for row in rows:
        by_run[row["run_id"]].append(
            AgentResult(
                agent=row["agent"],
                provider=row["provider"],
                model=row["model"],
                text=row["text"],
                status=row["status"],
                latency_ms=row["latency_ms"],
                error_message=row["error_message"],
            )
        )
    return by_run


def _history_item_from_row(row: sqlite3.Row, results: list[AgentResult]) -> HistoryItem:
    consensus_status = row["consensus_status"]
    consensus: ConsensusResult | None = None
    if consensus_status:
        consensus = ConsensusResult(
            provider=row["consensus_provider"] or "-",
            model=row["consensus_model"] or "-",
            text=row["consensus_text"] or "",
            status=consensus_status,
            latency_ms=int(row["consensus_latency_ms"] or 0),
            error_message=row["consensus_error_message"],
        )

    return HistoryItem(
        run_id=row["run_id"],
        profile=row["profile"],
        prompt=row["prompt"],
        created_at=row["created_at"],
        results=results,
        consensus=consensus,
    )


def _list_history(limit: int, offset: int) -> HistoryListResponse:
    with _get_db_connection() as conn:
        total = conn.execute("SELECT COUNT(1) AS count FROM runs").fetchone()["count"]
        rows = conn.execute(
            """
            SELECT run_id, profile, prompt, created_at,
                   consensus_provider, consensus_model, consensus_text, consensus_status,
                   consensus_latency_ms, consensus_error_message
            FROM runs
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()

    run_ids = [row["run_id"] for row in rows]
    by_run = _load_agent_results_map(run_ids)
    items = [_history_item_from_row(row, by_run.get(row["run_id"], [])) for row in rows]
    return HistoryListResponse(total=total, items=items)


def _get_history_item(run_id: str) -> HistoryItem | None:
    with _get_db_connection() as conn:
        row = conn.execute(
            """
            SELECT run_id, profile, prompt, created_at,
                   consensus_provider, consensus_model, consensus_text, consensus_status,
                   consensus_latency_ms, consensus_error_message
            FROM runs
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()

    if row is None:
        return None

    by_run = _load_agent_results_map([run_id])
    return _history_item_from_row(row, by_run.get(run_id, []))


def _serialize_result_for_consensus(result: AgentResult) -> str:
    if result.status == "OK":
        body = result.text.strip()
    else:
        body = f"[ERROR] {result.error_message or 'unknown error'}"

    return (
        f"agent={result.agent}\n"
        f"provider={result.provider}\n"
        f"model={result.model}\n"
        f"status={result.status}\n"
        f"text={body}"
    )


def _build_single_model_consensus_prompt(prompt: str, results: list[AgentResult]) -> str:
    joined = "\n\n---\n\n".join(_serialize_result_for_consensus(item) for item in results)
    return (
        "You are MAGI Consensus.\n"
        "Given the user question and multiple model outputs, produce a concise final answer.\n"
        "You must weigh agreement and disagreement between models.\n"
        "Output format:\n"
        "- Conclusion: <one paragraph>\n"
        "- Evidence:\n"
        "  1) ...\n"
        "  2) ...\n"
        "- Confidence: <low|medium|high>\n\n"
        f"User question:\n{prompt}\n\n"
        f"Model outputs:\n{joined}\n"
    )


def _build_peer_vote_prompt(
    prompt: str,
    base_results: list[AgentResult],
    peer_answers: dict[str, str],
    agent_id: str,
    round_index: int,
    consensus_config: ConsensusConfig,
) -> str:
    base_section = "\n\n".join(_serialize_result_for_consensus(item) for item in base_results)
    peer_section = "\n".join(
        f"{key}: {value.strip() if value.strip() else '[NO ANSWER]'}" for key, value in sorted(peer_answers.items())
    )

    strict_rules = ""
    if consensus_config.debate_mode == "strict":
        min_criticisms = max(1, consensus_config.min_criticisms)
        strict_rules = (
            f"- criticisms must include at least {min_criticisms} concrete weaknesses from other agents.\n"
            "- each criticism should reference a specific weakness (logic gap, missing assumption, or evidence issue).\n"
            "- empty criticisms are invalid.\n"
        )

    return (
        f"You are Agent {agent_id} in MAGI deliberation round {round_index}.\n"
        "Task: read all answers, improve your own answer, and vote the best current answer.\n"
        "Return ONLY JSON with this schema:\n"
        '{"revised_answer":"...","preferred_agent":"A|B|C","reason":"...","confidence":0,"criticisms":["..."]}\n'
        "Rules:\n"
        "- revised_answer must be one concise final answer to the user question.\n"
        "- preferred_agent is the agent you think currently has the strongest final answer.\n"
        "- confidence is 0-100 integer.\n\n"
        f"{strict_rules}"
        f"User question:\n{prompt}\n\n"
        f"Initial model outputs:\n{base_section}\n\n"
        f"Current peer answers:\n{peer_section}\n"
    )


def _parse_deliberation_turn(
    text: str,
    agent_config: AgentConfig,
    latency_ms: int,
    consensus_config: ConsensusConfig,
) -> DeliberationTurn:
    stripped = text.strip()
    parsed: dict[str, Any] = {}

    if "{" in stripped and "}" in stripped:
        start = stripped.find("{")
        end = stripped.rfind("}")
        candidate = stripped[start : end + 1]
        try:
            parsed = json.loads(candidate)
        except Exception:  # noqa: BLE001
            parsed = {}

    revised_answer = str(parsed.get("revised_answer", "")).strip() if parsed else ""
    preferred_agent = parsed.get("preferred_agent") if parsed else None
    reason = str(parsed.get("reason", "")).strip() if parsed else ""
    confidence_raw = parsed.get("confidence") if parsed else None
    criticisms_raw = parsed.get("criticisms") if parsed else None

    confidence: int | None = None
    if isinstance(confidence_raw, (int, float)):
        confidence = max(0, min(100, int(confidence_raw)))

    if preferred_agent not in {"A", "B", "C"}:
        preferred_agent = None

    if not revised_answer:
        revised_answer = stripped

    criticisms: list[str] = []
    if isinstance(criticisms_raw, list):
        for item in criticisms_raw:
            if isinstance(item, str):
                value = item.strip()
                if value:
                    criticisms.append(value)

    if consensus_config.debate_mode == "strict":
        min_criticisms = max(1, consensus_config.min_criticisms)
        if len(criticisms) < min_criticisms:
            return DeliberationTurn(
                agent=agent_config.agent,
                provider=agent_config.provider,
                model=agent_config.model,
                status="ERROR",
                latency_ms=latency_ms,
                raw_text=stripped,
                revised_answer=revised_answer,
                error_message=f"strict debate requires at least {min_criticisms} criticisms",
            )

    return DeliberationTurn(
        agent=agent_config.agent,
        provider=agent_config.provider,
        model=agent_config.model,
        status="OK",
        latency_ms=latency_ms,
        raw_text=stripped,
        revised_answer=revised_answer,
        preferred_agent=preferred_agent,
        reason=reason or None,
        confidence=confidence,
        criticisms=criticisms,
    )


def _criticism_quality_score(criticisms: list[str]) -> int:
    # Favor specific critiques while capping impact to avoid overweighting verbosity.
    score = 0
    for item in criticisms:
        words = len(item.split())
        score += min(20, words)
    return min(60, score)


async def _call_model_text(full_model: str, prompt: str, timeout_seconds: int) -> tuple[str, int]:
    start = perf_counter()
    response = await asyncio.wait_for(
        acompletion(
            model=full_model,
            messages=[{"role": "user", "content": prompt}],
        ),
        timeout=timeout_seconds,
    )
    text = _extract_text(response)
    latency_ms = int((perf_counter() - start) * 1000)
    return text, latency_ms


async def _run_single_agent(agent_config: AgentConfig, prompt: str, timeout_seconds: int) -> AgentResult:
    full_model = f"{agent_config.provider}/{agent_config.model}"
    start = perf_counter()
    print(f"[magi] agent={agent_config.agent} start model={full_model}")

    try:
        text, latency_ms = await _call_model_text(full_model, prompt, timeout_seconds)
        print(f"[magi] agent={agent_config.agent} success latency_ms={latency_ms}")
        return AgentResult(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            text=text,
            status="OK",
            latency_ms=latency_ms,
        )
    except asyncio.TimeoutError:
        print(f"[magi] agent={agent_config.agent} error=timeout")
        return AgentResult(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            text="",
            status="ERROR",
            latency_ms=int((perf_counter() - start) * 1000),
            error_message="timeout",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[magi] agent={agent_config.agent} error={type(exc).__name__}: {exc}")
        return AgentResult(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            text="",
            status="ERROR",
            latency_ms=int((perf_counter() - start) * 1000),
            error_message=_public_error_message(exc),
        )


async def _run_deliberation_turn(
    agent_config: AgentConfig,
    prompt: str,
    base_results: list[AgentResult],
    peer_answers: dict[str, str],
    round_index: int,
    consensus_config: ConsensusConfig,
    timeout_seconds: int,
) -> DeliberationTurn:
    full_model = f"{agent_config.provider}/{agent_config.model}"
    start = perf_counter()
    print(f"[magi] deliberation round={round_index} agent={agent_config.agent} start")

    try:
        turn_prompt = _build_peer_vote_prompt(
            prompt,
            base_results,
            peer_answers,
            agent_config.agent,
            round_index,
            consensus_config,
        )
        text, latency_ms = await _call_model_text(full_model, turn_prompt, timeout_seconds)
        turn = _parse_deliberation_turn(text, agent_config, latency_ms, consensus_config)
        print(f"[magi] deliberation round={round_index} agent={agent_config.agent} success latency_ms={latency_ms}")
        return turn
    except asyncio.TimeoutError:
        print(f"[magi] deliberation round={round_index} agent={agent_config.agent} error=timeout")
        return DeliberationTurn(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            status="ERROR",
            latency_ms=int((perf_counter() - start) * 1000),
            raw_text="",
            revised_answer="",
            error_message="timeout",
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[magi] deliberation round={round_index} agent={agent_config.agent} error={type(exc).__name__}: {exc}")
        return DeliberationTurn(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            status="ERROR",
            latency_ms=int((perf_counter() - start) * 1000),
            raw_text="",
            revised_answer="",
            error_message=_public_error_message(exc),
        )


async def _run_single_model_consensus(
    consensus_config: ConsensusConfig,
    prompt: str,
    results: list[AgentResult],
    timeout_seconds: int,
) -> ConsensusResult:
    ok_results = [item for item in results if item.status == "OK" and item.text.strip()]
    if len(ok_results) < consensus_config.min_ok_results:
        return ConsensusResult(
            provider=consensus_config.provider or "-",
            model=consensus_config.model or "-",
            text="",
            status="ERROR",
            latency_ms=0,
            error_message=f"consensus needs at least {consensus_config.min_ok_results} successful results",
        )

    if not consensus_config.provider or not consensus_config.model:
        return ConsensusResult(
            provider="-",
            model="-",
            text="",
            status="ERROR",
            latency_ms=0,
            error_message="single_model consensus requires provider/model",
        )

    full_model = f"{consensus_config.provider}/{consensus_config.model}"
    print(f"[magi] consensus(single_model) start model={full_model}")

    try:
        text, latency_ms = await _call_model_text(
            full_model,
            _build_single_model_consensus_prompt(prompt, results),
            timeout_seconds,
        )
        print(f"[magi] consensus(single_model) success latency_ms={latency_ms}")
        return ConsensusResult(
            provider=consensus_config.provider,
            model=consensus_config.model,
            text=text,
            status="OK",
            latency_ms=latency_ms,
        )
    except asyncio.TimeoutError:
        return ConsensusResult(
            provider=consensus_config.provider,
            model=consensus_config.model,
            text="",
            status="ERROR",
            latency_ms=int(timeout_seconds * 1000),
            error_message="timeout",
        )
    except Exception as exc:  # noqa: BLE001
        return ConsensusResult(
            provider=consensus_config.provider,
            model=consensus_config.model,
            text="",
            status="ERROR",
            latency_ms=0,
            error_message=_public_error_message(exc),
        )


async def _run_peer_vote_consensus(
    consensus_config: ConsensusConfig,
    prompt: str,
    results: list[AgentResult],
    agents: list[AgentConfig],
    timeout_seconds: int,
) -> ConsensusResult:
    start = perf_counter()
    ok_results = [item for item in results if item.status == "OK" and item.text.strip()]
    if len(ok_results) < consensus_config.min_ok_results:
        return ConsensusResult(
            provider="magi",
            model="peer_vote_v1",
            text="",
            status="ERROR",
            latency_ms=0,
            error_message=f"consensus needs at least {consensus_config.min_ok_results} successful results",
        )

    peer_answers = {item.agent: item.text for item in results if item.status == "OK" and item.text.strip()}
    rounds = max(1, consensus_config.rounds)
    last_turns: list[DeliberationTurn] = []

    for round_idx in range(1, rounds + 1):
        tasks = [
            _run_deliberation_turn(agent, prompt, results, peer_answers, round_idx, consensus_config, timeout_seconds)
            for agent in agents
        ]
        turns = await asyncio.gather(*tasks)
        last_turns = turns

        for turn in turns:
            if turn.status == "OK" and turn.revised_answer.strip():
                peer_answers[turn.agent] = turn.revised_answer.strip()

    valid_votes = [
        turn
        for turn in last_turns
        if turn.status == "OK" and turn.preferred_agent in {"A", "B", "C"}
    ]

    if len(valid_votes) < consensus_config.min_ok_results:
        return ConsensusResult(
            provider="magi",
            model="peer_vote_v1",
            text="",
            status="ERROR",
            latency_ms=int((perf_counter() - start) * 1000),
            error_message="insufficient valid deliberation votes",
        )

    vote_count: dict[str, int] = defaultdict(int)
    vote_conf: dict[str, int] = defaultdict(int)
    vote_crit: dict[str, int] = defaultdict(int)
    for turn in valid_votes:
        assert turn.preferred_agent is not None
        vote_count[turn.preferred_agent] += 1
        vote_conf[turn.preferred_agent] += turn.confidence if turn.confidence is not None else 50
        vote_crit[turn.preferred_agent] += _criticism_quality_score(turn.criticisms)

    ranked = sorted(
        vote_count.keys(),
        key=lambda key: (vote_count[key], vote_conf[key], vote_crit[key], key),
        reverse=True,
    )
    winner = ranked[0]

    winner_answer = peer_answers.get(winner, "")
    if not winner_answer:
        winner_turn = next((turn for turn in last_turns if turn.agent == winner and turn.revised_answer.strip()), None)
        winner_answer = winner_turn.revised_answer if winner_turn else ""

    vote_lines = []
    for turn in last_turns:
        if turn.status == "OK":
            vote_lines.append(
                f"- Agent {turn.agent} voted {turn.preferred_agent or '?'}"
                f" (confidence={turn.confidence if turn.confidence is not None else 'n/a'}): "
                f"{turn.reason or 'no reason'}"
            )
            if turn.criticisms:
                vote_lines.append("  criticisms: " + " | ".join(turn.criticisms))
        else:
            vote_lines.append(f"- Agent {turn.agent} failed in deliberation: {turn.error_message or 'unknown error'}")

    text = (
        f"Consensus winner: Agent {winner} "
        f"({vote_count[winner]}/{len(last_turns)} votes, total_confidence={vote_conf[winner]}, "
        f"critique_score={vote_crit[winner]}).\n\n"
        f"Final answer:\n{winner_answer}\n\n"
        "Vote details:\n"
        + "\n".join(vote_lines)
    )

    return ConsensusResult(
        provider="magi",
        model=f"peer_vote_r{rounds}",
        text=text,
        status="OK",
        latency_ms=int((perf_counter() - start) * 1000),
    )


async def _run_consensus(profile: ProfileConfig, prompt: str, results: list[AgentResult]) -> ConsensusResult:
    if profile.consensus.strategy == "single_model":
        return await _run_single_model_consensus(profile.consensus, prompt, results, profile.timeout_seconds)

    return await _run_peer_vote_consensus(
        profile.consensus,
        prompt,
        results,
        profile.agents,
        profile.timeout_seconds,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/magi/profiles", response_model=ProfilesResponse)
async def list_profiles() -> ProfilesResponse:
    try:
        config = load_config()
        profile_key, _ = _resolve_profile(config, None)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"invalid config: {exc}") from exc

    if config.profiles:
        profile_keys = sorted(config.profiles.keys())
        profile_agents = {key: config.profiles[key].agents for key in profile_keys}
        return ProfilesResponse(
            default_profile=config.default_profile or profile_key,
            profiles=profile_keys,
            profile_agents=profile_agents,
        )

    fallback_agents: list[AgentConfig] = []
    if config.agents:
        fallback_agents = config.agents
    return ProfilesResponse(
        default_profile=profile_key,
        profiles=[profile_key],
        profile_agents={profile_key: fallback_agents},
    )


@app.get("/api/magi/history", response_model=HistoryListResponse)
async def list_history(limit: int = 20, offset: int = 0) -> HistoryListResponse:
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100")
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be 0 or greater")
    try:
        return _list_history(limit=limit, offset=offset)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"failed to load history: {exc}") from exc


@app.get("/api/magi/history/{run_id}", response_model=HistoryItem)
async def get_history_item(run_id: str) -> HistoryItem:
    try:
        item = _get_history_item(run_id)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"failed to load history: {exc}") from exc
    if item is None:
        raise HTTPException(status_code=404, detail="run not found")
    return item


@app.post("/api/magi/run", response_model=RunResponse)
async def run_magi(payload: RunRequest) -> RunResponse:
    prompt = _validate_prompt(payload.prompt)

    try:
        config = load_config()
        profile_name, profile = _resolve_profile(config, payload.profile)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"invalid config: {exc}") from exc

    effective_prompt = await _build_effective_prompt(prompt, payload.fresh_mode)
    tasks = [_run_single_agent(agent, effective_prompt, profile.timeout_seconds) for agent in profile.agents]
    results = await asyncio.gather(*tasks)
    consensus = await _run_consensus(profile, effective_prompt, results)
    run_id = str(uuid.uuid4())
    try:
        _save_run_history(run_id, profile_name, prompt, results, consensus)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"failed to persist history: {exc}") from exc

    return RunResponse(run_id=run_id, profile=profile_name, results=results, consensus=consensus)


@app.post("/api/magi/retry", response_model=RetryResponse)
async def retry_agent(payload: RetryRequest) -> RetryResponse:
    prompt = _validate_prompt(payload.prompt)

    try:
        config = load_config()
        profile_name, profile = _resolve_profile(config, payload.profile)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"invalid config: {exc}") from exc

    target_agent = next((agent for agent in profile.agents if agent.agent == payload.agent), None)
    if target_agent is None:
        raise HTTPException(status_code=400, detail=f"agent {payload.agent} is not configured")

    effective_prompt = await _build_effective_prompt(prompt, payload.fresh_mode)
    result = await _run_single_agent(target_agent, effective_prompt, profile.timeout_seconds)
    return RetryResponse(run_id=str(uuid.uuid4()), profile=profile_name, result=result)


@app.post("/api/magi/consensus", response_model=ConsensusResponse)
async def recalc_consensus(payload: ConsensusRequest) -> ConsensusResponse:
    prompt = _validate_prompt(payload.prompt)

    try:
        config = load_config()
        profile_name, profile = _resolve_profile(config, payload.profile)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"invalid config: {exc}") from exc

    effective_prompt = await _build_effective_prompt(prompt, payload.fresh_mode)
    consensus = await _run_consensus(profile, effective_prompt, payload.results)
    return ConsensusResponse(run_id=str(uuid.uuid4()), profile=profile_name, consensus=consensus)
