import asyncio
import json
import os
import uuid
import warnings
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from litellm import acompletion
from pydantic import BaseModel

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


class AppConfig(BaseModel):
    agents: list[AgentConfig]
    consensus: ConsensusConfig
    timeout_seconds: int = 20


class RunRequest(BaseModel):
    prompt: str


class RetryRequest(BaseModel):
    prompt: str
    agent: Literal["A", "B", "C"]


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
    error_message: str | None = None


class RunResponse(BaseModel):
    run_id: str
    results: list[AgentResult]
    consensus: ConsensusResult


class RetryResponse(BaseModel):
    run_id: str
    result: AgentResult


class ConsensusResponse(BaseModel):
    run_id: str
    consensus: ConsensusResult


app = FastAPI(title="MAGI v0 Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_config() -> AppConfig:
    config_path = Path(__file__).resolve().parents[1] / "config.json"
    with config_path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    config = AppConfig.model_validate(data)
    if len(config.agents) != 3:
        raise ValueError("config.json must define exactly 3 agents")
    return config


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
) -> str:
    base_section = "\n\n".join(_serialize_result_for_consensus(item) for item in base_results)
    peer_section = "\n".join(
        f"{key}: {value.strip() if value.strip() else '[NO ANSWER]'}" for key, value in sorted(peer_answers.items())
    )

    return (
        f"You are Agent {agent_id} in MAGI deliberation round {round_index}.\n"
        "Task: read all answers, improve your own answer, and vote the best current answer.\n"
        "Return ONLY JSON with this schema:\n"
        '{"revised_answer":"...","preferred_agent":"A|B|C","reason":"...","confidence":0}\n'
        "Rules:\n"
        "- revised_answer must be one concise final answer to the user question.\n"
        "- preferred_agent is the agent you think currently has the strongest final answer.\n"
        "- confidence is 0-100 integer.\n\n"
        f"User question:\n{prompt}\n\n"
        f"Initial model outputs:\n{base_section}\n\n"
        f"Current peer answers:\n{peer_section}\n"
    )


def _parse_deliberation_turn(text: str, agent_config: AgentConfig, latency_ms: int) -> DeliberationTurn:
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

    confidence: int | None = None
    if isinstance(confidence_raw, (int, float)):
        confidence = max(0, min(100, int(confidence_raw)))

    if preferred_agent not in {"A", "B", "C"}:
        preferred_agent = None

    if not revised_answer:
        revised_answer = stripped

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
    )


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
            latency_ms=int(timeout_seconds * 1000),
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
            latency_ms=0,
            error_message=_public_error_message(exc),
        )


async def _run_deliberation_turn(
    agent_config: AgentConfig,
    prompt: str,
    base_results: list[AgentResult],
    peer_answers: dict[str, str],
    round_index: int,
    timeout_seconds: int,
) -> DeliberationTurn:
    full_model = f"{agent_config.provider}/{agent_config.model}"
    print(f"[magi] deliberation round={round_index} agent={agent_config.agent} start")

    try:
        turn_prompt = _build_peer_vote_prompt(prompt, base_results, peer_answers, agent_config.agent, round_index)
        text, latency_ms = await _call_model_text(full_model, turn_prompt, timeout_seconds)
        turn = _parse_deliberation_turn(text, agent_config, latency_ms)
        print(f"[magi] deliberation round={round_index} agent={agent_config.agent} success latency_ms={latency_ms}")
        return turn
    except asyncio.TimeoutError:
        print(f"[magi] deliberation round={round_index} agent={agent_config.agent} error=timeout")
        return DeliberationTurn(
            agent=agent_config.agent,
            provider=agent_config.provider,
            model=agent_config.model,
            status="ERROR",
            latency_ms=int(timeout_seconds * 1000),
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
            latency_ms=0,
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
            _run_deliberation_turn(agent, prompt, results, peer_answers, round_idx, timeout_seconds)
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
    for turn in valid_votes:
        assert turn.preferred_agent is not None
        vote_count[turn.preferred_agent] += 1
        vote_conf[turn.preferred_agent] += turn.confidence if turn.confidence is not None else 50

    ranked = sorted(vote_count.keys(), key=lambda key: (vote_count[key], vote_conf[key], key), reverse=True)
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
        else:
            vote_lines.append(f"- Agent {turn.agent} failed in deliberation: {turn.error_message or 'unknown error'}")

    text = (
        f"Consensus winner: Agent {winner} "
        f"({vote_count[winner]}/{len(last_turns)} votes, total_confidence={vote_conf[winner]}).\n\n"
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


async def _run_consensus(config: AppConfig, prompt: str, results: list[AgentResult]) -> ConsensusResult:
    if config.consensus.strategy == "single_model":
        return await _run_single_model_consensus(config.consensus, prompt, results, config.timeout_seconds)

    return await _run_peer_vote_consensus(
        config.consensus,
        prompt,
        results,
        config.agents,
        config.timeout_seconds,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/magi/run", response_model=RunResponse)
async def run_magi(payload: RunRequest) -> RunResponse:
    prompt = _validate_prompt(payload.prompt)

    try:
        config = load_config()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"invalid config: {exc}") from exc

    tasks = [_run_single_agent(agent, prompt, config.timeout_seconds) for agent in config.agents]
    results = await asyncio.gather(*tasks)
    consensus = await _run_consensus(config, prompt, results)

    return RunResponse(run_id=str(uuid.uuid4()), results=results, consensus=consensus)


@app.post("/api/magi/retry", response_model=RetryResponse)
async def retry_agent(payload: RetryRequest) -> RetryResponse:
    prompt = _validate_prompt(payload.prompt)

    try:
        config = load_config()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"invalid config: {exc}") from exc

    target_agent = next((agent for agent in config.agents if agent.agent == payload.agent), None)
    if target_agent is None:
        raise HTTPException(status_code=400, detail=f"agent {payload.agent} is not configured")

    result = await _run_single_agent(target_agent, prompt, config.timeout_seconds)
    return RetryResponse(run_id=str(uuid.uuid4()), result=result)


@app.post("/api/magi/consensus", response_model=ConsensusResponse)
async def recalc_consensus(payload: ConsensusRequest) -> ConsensusResponse:
    prompt = _validate_prompt(payload.prompt)

    try:
        config = load_config()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"invalid config: {exc}") from exc

    consensus = await _run_consensus(config, prompt, payload.results)
    return ConsensusResponse(run_id=str(uuid.uuid4()), consensus=consensus)
